# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import augment_joint_state, stack_joint_states
from curobo._src.state.state_robot import RobotState
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration,
    StateFromBSplineKnot,
    StateFromPositionClique,
    StateFromPositionTeleport,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.util.cuda_stream_util import (
    create_cuda_stream_pair,
    cuda_stream_context,
    synchronize_cuda_streams,
)
from curobo._src.curobolib.cuda_ops.tensor_checks import check_float16_tensors, check_float32_tensors
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.state_filter import JointStateFilter

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.transition.robot_state_transition_cfg import (
        RobotStateTransitionCfg,
    )


class RobotStateTransition:
    def __init__(
        self,
        config: RobotStateTransitionCfg,
    ):
        self.config = config
        self.batch_size = self.config.batch_size
        self.interpolation_steps = self.config.interpolation_steps
        self.dt = self.config.dt_traj_params.base_dt
        self.robot_model = Kinematics(self.config.robot_config.kinematics, compute_jacobian=False)
        # update cspace to store joint names in the order given by robot model:
        self.num_dof = self.robot_model.get_dof()
        self.robot_dynamics = None
        self._transition_streams = {}
        self._transition_events = {}
        if self.config.robot_config.dynamics is not None:
            self.robot_dynamics = self._create_dynamics()
            self._transition_streams["robot_dynamics"], self._transition_events["robot_dynamics"] = create_cuda_stream_pair(self.device_cfg.device)
            self._transition_streams["kinematics"], self._transition_events["kinematics"] = create_cuda_stream_pair(self.device_cfg.device)


        self._use_clique = True
        self._use_bmm_tensor_step = False
        self._use_clique_kernel = True
        self.d_state = 4 * self.num_dof  # + 1
        self.action_dim = self.num_dof
        self.d_dof = self.num_dof

        # Variables for enforcing joint limits
        self.joint_names = self.robot_model.joint_names
        self.joint_limits = self.robot_model.get_joint_limits()

        # #pre-allocating memory for rollouts
        self.state_seq = JointState.from_state_tensor(
            torch.zeros(
                self.batch_size,
                self.config.horizon,
                self.d_state,
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            ),
            dof=int(self.d_dof),
        )
        self.state_seq.dt = torch.ones(
            self.state_seq.shape[0], device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        self.Z = torch.tensor([0.0], device=self.device_cfg.device, dtype=self.device_cfg.dtype)

        dt_array = self.config.dt_traj_params.get_dt_array(self.config.horizon)
        self.traj_dt = torch.tensor(
            dt_array, dtype=self.device_cfg.dtype, device=self.device_cfg.device
        )
        if self.control_space == ControlSpace.ACCELERATION:
            # self._rollout_step_fn = TensorStepAcceleration(self.device_cfg, self._dt_h)
            # self._cmd_step_fn = TensorStepAcceleration(self.device_cfg, self.traj_dt)

            self._rollout_step_fn = StateFromAcceleration(
                self.device_cfg,
                self.traj_dt,
                self.num_dof,
                self.batch_size,
                self.config.horizon,
            )
            self._cmd_step_fn = StateFromAcceleration(
                self.device_cfg,
                self.traj_dt,
                self.num_dof,
                self.batch_size,
                self.config.horizon,
            )
        elif self.control_space == ControlSpace.VELOCITY:
            log_and_raise("Velocity control space not implemented for RobotStateTransition")
        elif self.control_space == ControlSpace.POSITION:
            if self.config.teleport_mode:
                self._rollout_step_fn = StateFromPositionTeleport(
                    self.device_cfg, self.batch_size, self.config.horizon
                )
                self._cmd_step_fn = StateFromPositionTeleport(
                    self.device_cfg, self.batch_size, self.config.horizon
                )
            else:
                self._rollout_step_fn = StateFromPositionClique(
                    self.device_cfg,
                    self.traj_dt,
                    self.num_dof,
                    filter_velocity=False,
                    filter_acceleration=False,
                    filter_jerk=False,
                    batch_size=self.batch_size,
                    horizon=self.config.horizon,
                )
                # print(self.filter_robot_command)
                # self.filter_robot_command = False
                self._cmd_step_fn = StateFromPositionClique(
                    self.device_cfg,
                    self.traj_dt,
                    self.num_dof,
                    filter_velocity=False,
                    filter_acceleration=False,
                    filter_jerk=False,
                    batch_size=self.batch_size,
                    horizon=self.config.horizon,
                )
        elif self.control_space in ControlSpace.bspline_types():
            self._rollout_step_fn = StateFromBSplineKnot(
                self.device_cfg,
                self.num_dof,
                batch_size=self.batch_size,
                horizon=self.config.horizon,
                n_knots=self.config.n_knots,
                control_space=self.config.control_space,
                interpolation_steps=self.config.interpolation_steps,
            )

            self._cmd_step_fn = StateFromBSplineKnot(
                self.device_cfg,
                self.num_dof,
                batch_size=self.batch_size,
                horizon=self.config.horizon,
                n_knots=self.config.n_knots,
                control_space=self.config.control_space,
                interpolation_steps=self.config.interpolation_steps,
            )

        self.update_batch_size(self.batch_size)

        self.state_filter = JointStateFilter(self.config.state_filter_cfg)
        self._initialize_robot_cmd_state()

    def _create_dynamics(self):
        """Create robot dynamics model from robot config's dynamics configuration.

        Returns:
            Dynamics: Configured dynamics model with batch size set.
        """
        dynamics_cfg = self.config.robot_config.dynamics

        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg

        if not isinstance(dynamics_cfg, DynamicsCfg):
            log_and_raise(
                f"Unknown dynamics config type: {type(dynamics_cfg).__name__}. "
                "Expected DynamicsCfg."
            )

        dyn_model = Dynamics(dynamics_cfg)

        dyn_model.setup_batch_size(batch_size=self.batch_size, horizon=self.config.horizon)

        log_info(
            f"Created Dynamics with batch_size={self.batch_size}, "
            f"horizon={self.config.horizon}"
        )

        return dyn_model

    def _initialize_robot_cmd_state(self):
        """Initialize robot command state sequence."""
        self._robot_cmd_state_seq = JointState.zeros(
            (1, self.horizon, self.action_dim), self.device_cfg
        )
        self._cmd_batch_size = -1
        if not self.config.teleport_mode:
            self._max_joint_vel = (
                self.get_state_bounds()
                .velocity.view(2, self.action_dim)[1, :]
                .reshape(1, 1, self.action_dim)
            )
            self._max_joint_acc = self.get_state_bounds().acceleration[1, :]  # - 0.2
            self._max_joint_jerk = self.get_state_bounds().jerk[1, :]  # - 0.2
        self._empty_joint_torque = None

    def update_traj_dt(
        self,
        dt: Union[float, torch.Tensor],
        base_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        base_ratio: Optional[float] = None,
    ):
        self.config.dt_traj_params.update_dt(dt, base_dt, max_dt, base_ratio)
        dt_array = self.config.dt_traj_params.get_dt_array(self.horizon)
        self.traj_dt[:] = torch.tensor(
            dt_array, dtype=self.device_cfg.dtype, device=self.device_cfg.device
        )
        self._cmd_step_fn.update_dt(self.traj_dt)
        self._rollout_step_fn.update_dt(self.traj_dt)


    def tensor_step(
        self,
        state: JointState,
        act: torch.Tensor,
        state_seq: JointState,
        state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
    ) -> JointState:
        """Args:
        state: [1,N]
        act: [H,N]

        Todo:
        Integration  with variable dt along trajectory
        """
        state_seq = self._rollout_step_fn.forward(
            state,
            act,
            state_seq,
            state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        return state_seq

    def robot_cmd_tensor_step(
        self,
        state: JointState,
        act: torch.Tensor,
        state_seq: JointState,
        state_idx: Optional[torch.Tensor] = None,
        implicit_goal_state: Optional[JointState] = None,
        implicit_goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
    ) -> JointState:
        """Args:
        state: [1,N]
        act: [H,N]

        Todo:
        Integration  with variable dt along trajectory
        """
        state_seq = self._cmd_step_fn.forward(
            state,
            act,
            state_seq,
            state_idx,
            goal_state=implicit_goal_state,
            goal_state_idx=implicit_goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        state_seq.joint_names = self.joint_names
        return state_seq

    def update_cmd_batch_size(self, batch_size):
        if self._cmd_batch_size != batch_size:
            self._robot_cmd_state_seq = JointState.zeros(
                (batch_size, self.horizon, self.action_dim), self.device_cfg
            )
            self._cmd_step_fn.update_batch_size(batch_size, self.horizon)
            self._cmd_batch_size = batch_size

    def update_batch_size(self, batch_size, force_update=False):
        if self.batch_size != batch_size:
            self.state_seq = JointState.zeros(
                (batch_size, self.horizon, self.action_dim), self.device_cfg
            )

            log_info("Updating state_seq buffer reference (created new tensor)")

            self.robot_model.update_batch_size(batch_size, self.horizon)

            # Update dynamics model batch size if it exists
            if self.robot_dynamics is not None:
                self.robot_dynamics.setup_batch_size(batch_size, self.horizon)
            else:
                self._empty_joint_torque = torch.zeros(
                    (batch_size, self.horizon, self.num_dof),
                    device=self.device_cfg.device,
                    dtype=self.device_cfg.dtype,
                )

        if force_update:
            self.state_seq = self.state_seq.detach()

        self._rollout_step_fn.update_batch_size(batch_size, self.horizon, force_update)
        self.batch_size = batch_size

    def forward(
        self,
        start_state: JointState,
        act_seq: torch.Tensor,
        start_state_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[JointState] = None,
        goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
        idxs_env: Optional[torch.Tensor] = None,
    ) -> RobotState:
        # filter state if needed:
        start_state_shaped = start_state  # .unsqueeze(1)
        # batch_size, horizon, d_act = act_seq.shape
        batch_size = act_seq.shape[0]

        self.update_batch_size(batch_size, force_update=act_seq.requires_grad)
        state_seq = self.state_seq

        check_fn = (
            check_float16_tensors
            if self.device_cfg.dtype == torch.float16
            else check_float32_tensors
        )
        tensors_to_check = dict(
            position=state_seq.position,
            velocity=state_seq.velocity,
            acceleration=state_seq.acceleration,
        )
        if state_seq.jerk is not None:
            tensors_to_check["jerk"] = state_seq.jerk
        check_fn(state_seq.position.device, **tensors_to_check)
        with profiler.record_function("tensor_step"):
            # forward step with step matrix:
            state_seq = self.tensor_step(
                start_state_shaped,
                act_seq,
                state_seq,
                start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )
        state = self.compute_augmented_state(state_seq, idxs_env=idxs_env)
        return state

    def compute_augmented_state(
        self, state_seq: JointState, idxs_env: Optional[torch.Tensor] = None,
    ) -> RobotState:
        joint_torque = None
        if len(state_seq.shape) == 1:
            state_seq = state_seq.unsqueeze(1)
        if len(state_seq.shape) == 2:
            state_seq = state_seq.unsqueeze(1)

        curr_batch_size = state_seq.shape[0]
        num_traj_points = state_seq.shape[1]

        if self.compute_inverse_dynamics:
            with cuda_stream_context("robot_dynamics", self._transition_streams, self._transition_events, self.device_cfg.device):
                joint_torque = self.robot_dynamics.compute_inverse_dynamics(state_seq)

            with cuda_stream_context("kinematics", self._transition_streams, self._transition_events, self.device_cfg.device):
                cuda_robot_model_state = self.robot_model.compute_kinematics(
                    state_seq, idxs_env=idxs_env,
                )

            synchronize_cuda_streams(self._transition_events, self.device_cfg.device)

        else:
            cuda_robot_model_state = self.robot_model.compute_kinematics(
                state_seq, idxs_env=idxs_env,
            )




            if (
                self._empty_joint_torque is None
                or self._empty_joint_torque.shape[0] != curr_batch_size
                or self._empty_joint_torque.shape[1] != num_traj_points
                or self._empty_joint_torque.shape[2] != self.num_dof
            ):
                self._empty_joint_torque = torch.zeros(
                    (curr_batch_size, num_traj_points, self.num_dof),
                    device=self.device_cfg.device,
                    dtype=self.device_cfg.dtype,
                )
            joint_torque = self._empty_joint_torque
        state = RobotState(
            joint_state=state_seq,
            cuda_robot_model_state=cuda_robot_model_state,
            joint_torque=joint_torque,
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
        implicit_goal_state: Optional[JointState] = None,
        implicit_goal_state_idx: Optional[torch.Tensor] = None,
        use_implicit_goal_state: Optional[torch.Tensor] = None,
    ) -> JointState:
        # Remove all this logic and instead return full trajectory
        # do splicing outside of this module.
        # removing state filter also?
        if self.return_full_act_buffer:
            if act_seq.shape[0] != self._cmd_batch_size:
                self.update_cmd_batch_size(act_seq.shape[0])
            full_state = self.robot_cmd_tensor_step(
                current_state,
                act_seq,
                self._robot_cmd_state_seq,
                state_idx,
                implicit_goal_state=implicit_goal_state,
                implicit_goal_state_idx=implicit_goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )
            return full_state
        if shift_steps == 1:
            if self.control_space == ControlSpace.POSITION:
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
                cmd_buffer = stack_joint_states(cmd_buffer, cmd)

        return cmd_buffer

    @property
    def action_bound_lows(self):
        if self.control_space in ControlSpace.position_types():
            # use joint limits:
            return self.joint_limits.position[0]
        if self.control_space == ControlSpace.VELOCITY:
            # use joint limits:
            return self.joint_limits.velocity[0]
        if self.control_space == ControlSpace.ACCELERATION:
            # use joint limits:
            return self.joint_limits.acceleration[0]

    @property
    def action_bound_highs(self):
        if self.control_space in ControlSpace.position_types():
            # use joint limits:
            return self.joint_limits.position[1]
        if self.control_space == ControlSpace.VELOCITY:
            # use joint limits:
            return self.joint_limits.velocity[1]
        if self.control_space == ControlSpace.ACCELERATION:
            # use joint limits:
            return self.joint_limits.acceleration[1]

    @property
    def init_action_mean(self):
        # output should be action_dim * horizon
        if self.control_space in ControlSpace.position_types():
            # use joint limits:
            return self.default_joint_position.unsqueeze(0).repeat(self.action_horizon, 1)
        if (
            self.control_space == ControlSpace.VELOCITY
            or self.control_space == ControlSpace.ACCELERATION
        ):
            # use joint limits:
            return self.default_joint_position.unsqueeze(0).repeat(self.action_horizon, 1) * 0.0

    def get_init_action_mean(self):
        """Get an action mean for rollout. Used mainly to get a seed.

        Returns:
            torch.Tensor: action mean of shape [action_horizon, dof]
        """
        # output should be action_dim * horizon
        if self.control_space in ControlSpace.position_types():
            # use joint limits:

            return self.default_joint_position.unsqueeze(0).repeat(self.action_horizon, 1)

        if (
            self.control_space == ControlSpace.VELOCITY
            or self.control_space == ControlSpace.ACCELERATION
        ):
            # use joint limits:
            return self.default_joint_position.unsqueeze(0).repeat(self.action_horizon, 1) * 0.0

    @property
    def default_joint_position(self):
        return self.robot_model.kinematics_config.cspace.default_joint_position

    @property
    def cspace_distance_weight(self):
        return self.robot_model.kinematics_config.cspace.cspace_distance_weight

    @property
    def null_space_weight(self):
        return self.robot_model.kinematics_config.cspace.null_space_weight

    @property
    def null_space_maximum_distance(self):
        return self.robot_model.kinematics_config.cspace.null_space_maximum_distance

    @property
    def max_acceleration(self):
        return self.get_state_bounds().acceleration[1, :]

    @property
    def max_jerk(self):
        return self.get_state_bounds().jerk[1, :]

    @property
    def max_velocity(self):
        return self.get_state_bounds().velocity[1, :]

    @property
    def action_horizon(self):
        return self._rollout_step_fn.action_horizon

    @property
    def horizon(self):
        return self.config.horizon

    @property
    def n_knots(self):
        return self.config.n_knots

    @property
    def control_space(self):
        return self.config.control_space

    @property
    def device_cfg(self):
        return self.config.device_cfg

    @property
    def teleport_mode(self):
        return self.config.teleport_mode

    @property
    def return_full_act_buffer(self):
        return self.config.return_full_act_buffer

    @property
    def state_finite_difference_mode(self):
        return self.config.state_finite_difference_mode

    @property
    def filter_robot_command(self):
        return self.config.filter_robot_command

    @property
    def compute_inverse_dynamics(self):
        return self.robot_dynamics is not None

    def get_state_bounds(self):
        joint_limits = self.robot_model.get_joint_limits()
        return joint_limits

    def get_action_from_state(self, state: JointState) -> torch.Tensor:
        if self.control_space in ControlSpace.position_types():
            return state.position
        if self.control_space == ControlSpace.VELOCITY:
            return state.velocity
        if self.control_space == ControlSpace.ACCELERATION:
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

    def update_link_mass(self, link_name: str, mass: float):
        """Update mass of a specific link in the dynamics model.

        Args:
            link_name: Name of the link to update.
            mass: New mass value in kg.
        """
        if self.robot_dynamics is not None:
            self.robot_dynamics.update_link_mass(link_name, mass)
        else:
            log_and_raise(
                "Cannot update link mass without inverse dynamics (robot_config.dynamics is None)"
            )

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ):
        """Update inertial properties of a single link.

        Delegates to the robot dynamics model if available. For updating multiple links,
        use :meth:`update_links_inertial` for better performance.

        Args:
            link_name: Name of the link to update.
            mass: New mass value in kg (optional).
            com: Center of mass in link frame, shape [3] (optional).
            inertia: Inertia tensor, shape [6] - [ixx, iyy, izz, ixy, ixz, iyz] (optional).

        Raises:
            RuntimeError: If robot dynamics model is not available.
        """
        if self.robot_dynamics is not None:
            self.robot_dynamics.update_link_inertial(link_name, mass, com, inertia)
        else:
            log_and_raise(
                "Cannot update link inertial properties without inverse dynamics (robot_config.dynamics is None)"
            )

    def update_links_inertial(
        self,
        link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ):
        """Update inertial properties of multiple links efficiently.

        Delegates to the robot dynamics model if available. This method is more efficient
        than calling :meth:`update_link_inertial` multiple times as it recomputes spatial
        inertia only once.

        Args:
            link_properties: Dictionary mapping link names to their properties:
                {
                    "link1": {"mass": 1.0, "com": torch.tensor([0,0,0])},
                    "link2": {"mass": 2.0, "inertia": torch.tensor([...])},
                }

        Raises:
            RuntimeError: If robot dynamics model is not available.
        """
        if self.robot_dynamics is not None:
            self.robot_dynamics.update_links_inertial(link_properties)
        else:
            log_and_raise(
                "Cannot update link inertial properties without inverse dynamics (robot_config.dynamics is None)"
            )

    @profiler.record_function("RobotStateTransition/get_full_dof_from_solution")
    def get_full_dof_from_solution(self, q_js: JointState) -> JointState:
        """This function will all the dof that are locked during optimization.

        Args:
            q_sol: _description_

        Returns:
            _description_
        """
        if self.robot_model.lock_jointstate is None:
            return q_js
        all_joint_names = self.robot_model.all_articulated_joint_names
        lock_joint_state = self.robot_model.lock_jointstate

        new_js = augment_joint_state(q_js, all_joint_names, lock_joint_state)
        return new_js
