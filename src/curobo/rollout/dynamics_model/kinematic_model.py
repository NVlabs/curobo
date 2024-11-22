#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.rollout.dynamics_model.tensor_step import (
    TensorStepAccelerationKernel,
    TensorStepPosition,
    TensorStepPositionClique,
    TensorStepPositionCliqueKernel,
    TensorStepPositionTeleport,
)
from curobo.types.base import TensorDeviceType
from curobo.types.enum import StateType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info
from curobo.util.state_filter import FilterConfig, JointStateFilter


@dataclass
class TimeTrajConfig:
    base_dt: float
    base_ratio: float
    max_dt: float

    def get_dt_array(self, num_points: int):
        dt_array = [self.base_dt] * int(self.base_ratio * num_points)

        smooth_blending = torch.linspace(
            self.base_dt,
            self.max_dt,
            steps=int((1 - self.base_ratio) * num_points),
        ).tolist()
        dt_array += smooth_blending
        if len(dt_array) != num_points:
            dt_array.insert(0, dt_array[0])
        return dt_array

    def update_dt(
        self,
        all_dt: float = None,
        base_dt: float = None,
        max_dt: float = None,
        base_ratio: float = None,
    ):
        if all_dt is not None:
            self.base_dt = all_dt
            self.max_dt = all_dt
            return
        if base_dt is not None:
            self.base_dt = base_dt
        if base_ratio is not None:
            self.base_ratio = base_ratio
        if max_dt is not None:
            self.max_dt = max_dt


@dataclass
class KinematicModelState(Sequence):
    # TODO: subclass this from State
    state_seq: JointState
    ee_pos_seq: Optional[torch.Tensor] = None
    ee_quat_seq: Optional[torch.Tensor] = None
    robot_spheres: Optional[torch.Tensor] = None
    link_pos_seq: Optional[torch.Tensor] = None
    link_quat_seq: Optional[torch.Tensor] = None
    lin_jac_seq: Optional[torch.Tensor] = None
    ang_jac_seq: Optional[torch.Tensor] = None
    link_names: Optional[List[str]] = None

    def __getitem__(self, idx):
        d_list = [
            self.state_seq,
            self.ee_pos_seq,
            self.ee_quat_seq,
            self.robot_spheres,
            self.link_pos_seq,
            self.link_quat_seq,
            self.lin_jac_seq,
            self.ang_jac_seq,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return KinematicModelState(*idx_vals, link_names=self.link_names)

    def __len__(self):
        return len(self.state_seq)

    @property
    def ee_pose(self) -> Pose:
        return Pose(self.ee_pos_seq, self.ee_quat_seq, normalize_rotation=False)

    @property
    def link_pose(self):
        """Deprecated: Use link_poses instead."""
        return self.link_poses

    @property
    def link_poses(self):
        if self.link_names is not None:
            link_pos_seq = self.link_pos_seq.contiguous()
            link_quat_seq = self.link_quat_seq.contiguous()
            link_poses = {}
            for i, v in enumerate(self.link_names):
                link_poses[v] = Pose(
                    link_pos_seq[..., i, :], link_quat_seq[..., i, :], normalize_rotation=False
                )
        else:
            link_poses = None
        return link_poses


@dataclass(frozen=False)
class KinematicModelConfig:
    robot_config: RobotConfig
    dt_traj_params: TimeTrajConfig
    tensor_args: TensorDeviceType
    vel_scale: float = 1.0
    state_estimation_variance: float = 0.0
    batch_size: int = 1
    horizon: int = 5
    control_space: StateType = StateType.ACCELERATION
    state_filter_cfg: Optional[FilterConfig] = None
    teleport_mode: bool = False
    return_full_act_buffer: bool = False
    state_finite_difference_mode: str = "BACKWARD"
    filter_robot_command: bool = False
    # tensor_step_type: TensorStepType = TensorStepType.ACCELERATION

    @staticmethod
    def from_dict(
        data_dict_in, robot_cfg: Union[Dict, RobotConfig], tensor_args=TensorDeviceType()
    ):
        data_dict = deepcopy(data_dict_in)
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        data_dict["robot_config"] = robot_cfg
        data_dict["dt_traj_params"] = TimeTrajConfig(**data_dict["dt_traj_params"])
        data_dict["control_space"] = StateType[data_dict["control_space"]]
        data_dict["state_filter_cfg"] = FilterConfig.from_dict(
            data_dict["state_filter_cfg"]["filter_coeff"],
            enable=data_dict["state_filter_cfg"]["enable"],
            dt=data_dict["dt_traj_params"].base_dt,
            control_space=data_dict["control_space"],
            tensor_args=tensor_args,
            teleport_mode=data_dict["teleport_mode"],
        )

        return KinematicModelConfig(**data_dict, tensor_args=tensor_args)


class KinematicModel(KinematicModelConfig):
    def __init__(
        self,
        kinematic_model_config: KinematicModelConfig,
    ):
        super().__init__(**vars(kinematic_model_config))
        self.dt = self.dt_traj_params.base_dt
        self.robot_model = CudaRobotModel(self.robot_config.kinematics)
        # update cspace to store joint names in the order given by robot model:
        self.n_dofs = self.robot_model.get_dof()
        self._use_clique = True
        self._use_bmm_tensor_step = False
        self._use_clique_kernel = True
        self.d_state = 4 * self.n_dofs  # + 1
        self.d_action = self.n_dofs
        self.d_dof = self.n_dofs

        # Variables for enforcing joint limits
        self.joint_names = self.robot_model.joint_names
        self.joint_limits = self.robot_model.get_joint_limits()

        # #pre-allocating memory for rollouts
        self.state_seq = JointState.from_state_tensor(
            torch.zeros(
                self.batch_size,
                self.horizon,
                self.d_state,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            ),
            dof=int(self.d_dof),
        )
        self.Z = torch.tensor([0.0], device=self.tensor_args.device, dtype=self.tensor_args.dtype)

        dt_array = self.dt_traj_params.get_dt_array(self.horizon)
        self.traj_dt = torch.tensor(
            dt_array, dtype=self.tensor_args.dtype, device=self.tensor_args.device
        )
        # TODO: choose tensor integration type here:
        if self.control_space == StateType.ACCELERATION:
            # self._rollout_step_fn = TensorStepAcceleration(self.tensor_args, self._dt_h)
            # self._cmd_step_fn = TensorStepAcceleration(self.tensor_args, self.traj_dt)

            self._rollout_step_fn = TensorStepAccelerationKernel(
                self.tensor_args,
                self.traj_dt,
                self.n_dofs,
                self.batch_size,
                self.horizon,
            )
            self._cmd_step_fn = TensorStepAccelerationKernel(
                self.tensor_args,
                self.traj_dt,
                self.n_dofs,
                self.batch_size,
                self.horizon,
            )
        elif self.control_space == StateType.VELOCITY:
            raise NotImplementedError()
        elif self.control_space == StateType.JERK:
            raise NotImplementedError()
        elif self.control_space == StateType.POSITION:
            if self.teleport_mode:
                self._rollout_step_fn = TensorStepPositionTeleport(
                    self.tensor_args, self.batch_size, self.horizon
                )
                self._cmd_step_fn = TensorStepPositionTeleport(
                    self.tensor_args, self.batch_size, self.horizon
                )
            else:
                if self._use_clique:
                    if self._use_clique_kernel:
                        if self.state_finite_difference_mode == "BACKWARD":
                            finite_difference = -1
                        elif self.state_finite_difference_mode == "CENTRAL":
                            finite_difference = 0
                        else:
                            log_error(
                                "unknown state finite difference mode: "
                                + self.state_finite_difference_mode
                            )
                        self._rollout_step_fn = TensorStepPositionCliqueKernel(
                            self.tensor_args,
                            self.traj_dt,
                            self.n_dofs,
                            finite_difference_mode=finite_difference,
                            filter_velocity=False,
                            filter_acceleration=False,
                            filter_jerk=False,
                            batch_size=self.batch_size,
                            horizon=self.horizon,
                        )
                        self._cmd_step_fn = TensorStepPositionCliqueKernel(
                            self.tensor_args,
                            self.traj_dt,
                            self.n_dofs,
                            finite_difference_mode=finite_difference,
                            filter_velocity=False,
                            filter_acceleration=self.filter_robot_command,
                            filter_jerk=self.filter_robot_command,
                            batch_size=self.batch_size,
                            horizon=self.horizon,
                        )

                    else:
                        self._rollout_step_fn = TensorStepPositionClique(
                            self.tensor_args,
                            self.traj_dt,
                            batch_size=self.batch_size,
                            horizon=self.horizon,
                        )
                        self._cmd_step_fn = TensorStepPositionClique(
                            self.tensor_args,
                            self.traj_dt,
                            batch_size=self.batch_size,
                            horizon=self.horizon,
                        )
                else:
                    self._rollout_step_fn = TensorStepPosition(
                        self.tensor_args,
                        self.traj_dt,
                        batch_size=self.batch_size,
                        horizon=self.horizon,
                    )
                    self._cmd_step_fn = TensorStepPosition(
                        self.tensor_args,
                        self.traj_dt,
                        batch_size=self.batch_size,
                        horizon=self.horizon,
                    )
        self.update_batch_size(self.batch_size)

        self.state_filter = JointStateFilter(self.state_filter_cfg)
        self._robot_cmd_state_seq = JointState.zeros(
            (1, self.horizon, self.d_action), self.tensor_args
        )
        self._cmd_batch_size = -1
        if not self.teleport_mode:
            self._max_joint_vel = (
                self.get_state_bounds()
                .velocity.view(2, self.d_action)[1, :]
                .reshape(1, 1, self.d_action)
            ) - 0.2
            self._max_joint_acc = self.get_state_bounds().acceleration[1, :] - 0.2
            self._max_joint_jerk = self.get_state_bounds().jerk[1, :] - 0.2

    def update_traj_dt(
        self,
        dt: Union[float, torch.Tensor],
        base_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        base_ratio: Optional[float] = None,
    ):
        self.dt_traj_params.update_dt(dt, base_dt, max_dt, base_ratio)
        dt_array = self.dt_traj_params.get_dt_array(self.horizon)
        self.traj_dt[:] = torch.tensor(
            dt_array, dtype=self.tensor_args.dtype, device=self.tensor_args.device
        )
        self._cmd_step_fn.update_dt(self.traj_dt)
        self._rollout_step_fn.update_dt(self.traj_dt)

    def get_next_state(self, curr_state: torch.Tensor, act: torch.Tensor, dt):
        """Does a single step from the current state
        Args:
        curr_state: current state
        act: action
        dt: time to integrate
        Returns:
        next_state
        TODO: Move this into tensorstep class?
        """

        if self.control_space == StateType.JERK:
            curr_state[2 * self.n_dofs : 3 * self.n_dofs] = (
                curr_state[self.n_dofs : 2 * self.n_dofs] + act * dt
            )
            curr_state[self.n_dofs : 2 * self.n_dofs] = (
                curr_state[self.n_dofs : 2 * self.n_dofs]
                + curr_state[self.n_dofs * 2 : self.n_dofs * 3] * dt
            )

            curr_state[: self.n_dofs] = (
                curr_state[: self.n_dofs] + curr_state[self.n_dofs : 2 * self.n_dofs] * dt
            )
        elif self.control_space == StateType.ACCELERATION:
            curr_state[2 * self.n_dofs : 3 * self.n_dofs] = act
            curr_state[self.n_dofs : 2 * self.n_dofs] = (
                curr_state[self.n_dofs : 2 * self.n_dofs]
                + curr_state[self.n_dofs * 2 : self.n_dofs * 3] * dt
            )

            curr_state[: self.n_dofs] = (
                curr_state[: self.n_dofs]
                + curr_state[self.n_dofs : 2 * self.n_dofs] * dt
                + 0.5 * act * dt * dt
            )
        elif self.control_space == StateType.VELOCITY:
            curr_state[2 * self.n_dofs : 3 * self.n_dofs] = 0.0
            curr_state[self.n_dofs : 2 * self.n_dofs] = act * dt

            curr_state[: self.n_dofs] = (
                curr_state[: self.n_dofs] + curr_state[self.n_dofs : 2 * self.n_dofs] * dt
            )
        elif self.control_space == StateType.POSITION:
            curr_state[2 * self.n_dofs : 3 * self.n_dofs] = 0.0
            curr_state[1 * self.n_dofs : 2 * self.n_dofs] = 0.0
            curr_state[: self.n_dofs] = act
        return curr_state

    def tensor_step(
        self,
        state: JointState,
        act: torch.Tensor,
        state_seq: JointState,
        state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        """
        Args:
        state: [1,N]
        act: [H,N]
        todo:
        Integration  with variable dt along trajectory
        """
        state_seq = self._rollout_step_fn.forward(state, act, state_seq, state_idx)

        return state_seq

    def robot_cmd_tensor_step(
        self,
        state: JointState,
        act: torch.Tensor,
        state_seq: JointState,
        state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        """
        Args:
        state: [1,N]
        act: [H,N]
        todo:
        Integration  with variable dt along trajectory
        """
        state_seq = self._cmd_step_fn.forward(state, act, state_seq, state_idx)
        state_seq.joint_names = self.joint_names
        return state_seq

    def update_cmd_batch_size(self, batch_size):
        if self._cmd_batch_size != batch_size:
            self._robot_cmd_state_seq = JointState.zeros(
                (batch_size, self.horizon, self.d_action), self.tensor_args
            )
            self._cmd_step_fn.update_batch_size(batch_size, self.horizon)
            self._cmd_batch_size = batch_size

    def update_batch_size(self, batch_size, force_update=False):
        if self.batch_size != batch_size:
            # TODO: Remove tensor recreation upon force update?
            self.state_seq = JointState.zeros(
                (batch_size, self.horizon, self.d_action), self.tensor_args
            )

            log_info("Updating state_seq buffer reference (created new tensor)")
            # print("Creating new tensor")
        if force_update:
            self.state_seq = self.state_seq.detach()

        self._rollout_step_fn.update_batch_size(batch_size, self.horizon, force_update)
        self.batch_size = batch_size

    def forward(
        self,
        start_state: JointState,
        act_seq: torch.Tensor,
        start_state_idx: Optional[torch.Tensor] = None,
    ) -> KinematicModelState:
        # filter state if needed:
        start_state_shaped = start_state  # .unsqueeze(1)
        # batch_size, horizon, d_act = act_seq.shape

        batch_size = act_seq.shape[0]
        self.update_batch_size(batch_size, force_update=act_seq.requires_grad)
        state_seq = self.state_seq
        curr_batch_size = self.batch_size
        num_traj_points = self.horizon
        if not state_seq.position.is_contiguous():
            state_seq.position = state_seq.position.contiguous()
        if not state_seq.velocity.is_contiguous():
            state_seq.velocity = state_seq.velocity.contiguous()
        if not state_seq.acceleration.is_contiguous():
            state_seq.acceleration = state_seq.acceleration.contiguous()
        if not state_seq.jerk.is_contiguous():
            state_seq.jerk = state_seq.jerk.contiguous()
        with profiler.record_function("tensor_step"):
            # forward step with step matrix:
            state_seq = self.tensor_step(start_state_shaped, act_seq, state_seq, start_state_idx)

        shape_tup = (curr_batch_size * num_traj_points, self.n_dofs)
        with profiler.record_function("fk + jacobian"):
            (
                ee_pos_seq,
                ee_quat_seq,
                lin_jac_seq,
                ang_jac_seq,
                link_pos_seq,
                link_quat_seq,
                link_spheres,
            ) = self.robot_model.forward(state_seq.position.view(shape_tup))
        link_pos_seq = link_pos_seq.view(
            ((curr_batch_size, num_traj_points, link_pos_seq.shape[1], 3))
        )
        link_quat_seq = link_quat_seq.view(
            ((curr_batch_size, num_traj_points, link_quat_seq.shape[1], 4))
        )
        link_spheres = link_spheres.view(
            (curr_batch_size, num_traj_points, link_spheres.shape[1], link_spheres.shape[-1])
        )
        ee_pos_seq = ee_pos_seq.view((curr_batch_size, num_traj_points, 3))
        ee_quat_seq = ee_quat_seq.view((curr_batch_size, num_traj_points, 4))
        if lin_jac_seq is not None:
            lin_jac_seq = lin_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        if ang_jac_seq is not None:
            ang_jac_seq = ang_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))

        state = KinematicModelState(
            state_seq,
            ee_pos_seq,
            ee_quat_seq,
            link_spheres,
            link_pos_seq,
            link_quat_seq,
            lin_jac_seq,
            ang_jac_seq,
            link_names=self.robot_model.link_names,
        )

        return state

    def integrate_action(self, act_seq):
        if self.action_order == 0:
            return act_seq

        nth_act_seq = self._integrate_matrix_nth @ act_seq
        return nth_act_seq

    def integrate_action_step(self, act, dt):
        for i in range(self.action_order):
            act = act * dt

        return act

    def filter_robot_state(self, current_state: JointState):
        filtered_state = self.state_filter.filter_joint_state(current_state)
        return filtered_state

    @torch.no_grad()
    def get_robot_command(
        self,
        current_state: JointState,
        act_seq: torch.Tensor,
        shift_steps: int = 1,
        state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        if self.return_full_act_buffer:
            if act_seq.shape[0] != self._cmd_batch_size:
                self.update_cmd_batch_size(act_seq.shape[0])
            full_state = self.robot_cmd_tensor_step(
                current_state,
                act_seq,
                self._robot_cmd_state_seq,
                state_idx,
            )
            return full_state
        if shift_steps == 1:
            if self.control_space == StateType.POSITION:
                act_step = act_seq[..., 0, :].clone()
            else:
                act_step = act_seq[..., 0, :].clone()
            cmd = self.state_filter.integrate_action(act_step, current_state)
            return cmd

        # get the first timestep in action buffer
        cmd = current_state.clone()

        for i in range(shift_steps):
            act_step = act_seq[..., i, :]
            # we integrate the action with the current belief:
            cmd = self.state_filter.integrate_action(act_step, cmd)
            if i == 0:
                cmd_buffer = cmd.clone()
            else:
                cmd_buffer = cmd_buffer.stack(cmd)

        return cmd_buffer

    @property
    def action_bound_lows(self):
        if self.control_space == StateType.POSITION:
            # use joint limits:
            return self.joint_limits.position[0]
        if self.control_space == StateType.VELOCITY:
            # use joint limits:
            return self.joint_limits.velocity[0]
        if self.control_space == StateType.ACCELERATION:
            # use joint limits:
            return self.joint_limits.acceleration[0]

    @property
    def action_bound_highs(self):
        if self.control_space == StateType.POSITION:
            # use joint limits:
            return self.joint_limits.position[1]
        if self.control_space == StateType.VELOCITY:
            # use joint limits:
            return self.joint_limits.velocity[1]
        if self.control_space == StateType.ACCELERATION:
            # use joint limits:
            return self.joint_limits.acceleration[1]

    @property
    def init_action_mean(self):
        # output should be d_action * horizon
        if self.control_space == StateType.POSITION:
            # use joint limits:
            return self.retract_config.unsqueeze(0).repeat(self.action_horizon, 1)
        if self.control_space == StateType.VELOCITY or self.control_space == StateType.ACCELERATION:
            # use joint limits:
            return self.retract_config.unsqueeze(0).repeat(self.action_horizon, 1) * 0.0

    @property
    def retract_config(self):
        return self.robot_model.kinematics_config.cspace.retract_config

    @property
    def cspace_distance_weight(self):
        return self.robot_model.kinematics_config.cspace.cspace_distance_weight

    @property
    def null_space_weight(self):
        return self.robot_model.kinematics_config.cspace.null_space_weight

    @property
    def max_acceleration(self):
        return self.get_state_bounds().acceleration[1, 0].item()

    @property
    def max_jerk(self):
        return self.get_state_bounds().jerk[1, 0].item()

    @property
    def action_horizon(self):
        return self._rollout_step_fn.action_horizon

    def get_state_bounds(self):
        joint_limits = self.robot_model.get_joint_limits()
        return joint_limits

    def get_action_from_state(self, state: JointState) -> torch.Tensor:
        if self.control_space == StateType.POSITION:
            return state.position
        if self.control_space == StateType.VELOCITY:
            return state.velocity
        if self.control_space == StateType.ACCELERATION:
            return state.acceleration

    def get_state_from_action(
        self,
        start_state: JointState,
        act_seq: torch.Tensor,
        state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        """Compute State sequence from an action trajectory

        Args:
            start_state (JointState): _description_
            act_seq (torch.Tensor): _description_

        Returns:
            JointState: _description_
        """
        if act_seq.shape[0] != self._cmd_batch_size:
            self.update_cmd_batch_size(act_seq.shape[0])
        full_state = self.robot_cmd_tensor_step(
            start_state,
            act_seq,
            self._robot_cmd_state_seq,
            state_idx,
        )
        return full_state
