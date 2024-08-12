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
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.rollout.cost.bound_cost import BoundCost, BoundCostConfig
from curobo.rollout.cost.dist_cost import DistCost, DistCostConfig
from curobo.rollout.cost.manipulability_cost import ManipulabilityCost, ManipulabilityCostConfig
from curobo.rollout.cost.primitive_collision_cost import (
    PrimitiveCollisionCost,
    PrimitiveCollisionCostConfig,
)
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from curobo.rollout.cost.stop_cost import StopCost, StopCostConfig
from curobo.rollout.dynamics_model.kinematic_model import (
    KinematicModel,
    KinematicModelConfig,
    KinematicModelState,
)
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutConfig, RolloutMetrics, Trajectory
from curobo.types.base import TensorDeviceType
from curobo.types.robot import CSpaceConfig, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import cat_sum, cat_sum_horizon


@dataclass
class ArmCostConfig:
    bound_cfg: Optional[BoundCostConfig] = None
    null_space_cfg: Optional[DistCostConfig] = None
    manipulability_cfg: Optional[ManipulabilityCostConfig] = None
    stop_cfg: Optional[StopCostConfig] = None
    self_collision_cfg: Optional[SelfCollisionCostConfig] = None
    primitive_collision_cfg: Optional[PrimitiveCollisionCostConfig] = None

    @staticmethod
    def _get_base_keys():
        k_list = {
            "null_space_cfg": DistCostConfig,
            "manipulability_cfg": ManipulabilityCostConfig,
            "stop_cfg": StopCostConfig,
            "self_collision_cfg": SelfCollisionCostConfig,
            "bound_cfg": BoundCostConfig,
        }
        return k_list

    @staticmethod
    def from_dict(
        data_dict: Dict,
        robot_config: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        k_list = ArmCostConfig._get_base_keys()
        data = ArmCostConfig._get_formatted_dict(
            data_dict,
            k_list,
            robot_config,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        return ArmCostConfig(**data)

    @staticmethod
    def _get_formatted_dict(
        data_dict: Dict,
        cost_key_list: Dict,
        robot_config: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        data = {}
        for k in cost_key_list:
            if k in data_dict:
                data[k] = cost_key_list[k](**data_dict[k], tensor_args=tensor_args)
        if "primitive_collision_cfg" in data_dict and world_coll_checker is not None:
            data["primitive_collision_cfg"] = PrimitiveCollisionCostConfig(
                **data_dict["primitive_collision_cfg"],
                world_coll_checker=world_coll_checker,
                tensor_args=tensor_args
            )

        return data


@dataclass
class ArmBaseConfig(RolloutConfig):
    model_cfg: Optional[KinematicModelConfig] = None
    cost_cfg: Optional[ArmCostConfig] = None
    constraint_cfg: Optional[ArmCostConfig] = None
    convergence_cfg: Optional[ArmCostConfig] = None
    world_coll_checker: Optional[WorldCollision] = None

    @staticmethod
    def model_from_dict(
        model_data_dict: Dict,
        robot_cfg: RobotConfig,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return KinematicModelConfig.from_dict(model_data_dict, robot_cfg, tensor_args=tensor_args)

    @staticmethod
    def cost_from_dict(
        cost_data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return ArmCostConfig.from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

    @staticmethod
    def world_coll_checker_from_dict(
        world_coll_checker_dict: Optional[Dict] = None,
        world_model_dict: Optional[Union[WorldConfig, Dict]] = None,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        # TODO: Check which type of collision checker and load that.
        if (
            world_coll_checker is None
            and world_model_dict is not None
            and world_coll_checker_dict is not None
        ):
            world_coll_cfg = WorldCollisionConfig.load_from_dict(
                world_coll_checker_dict, world_model_dict, tensor_args
            )

            world_coll_checker = create_collision_checker(world_coll_cfg)
        else:
            log_info("*******USING EXISTING COLLISION CHECKER***********")
        return world_coll_checker

    @classmethod
    @profiler.record_function("arm_base_config/from_dict")
    def from_dict(
        cls,
        robot_cfg: Union[Dict, RobotConfig],
        model_data_dict: Dict,
        cost_data_dict: Dict,
        constraint_data_dict: Dict,
        convergence_data_dict: Dict,
        world_coll_checker_dict: Optional[Dict] = None,
        world_model_dict: Optional[Dict] = None,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        """Create ArmBase class from dictionary

        NOTE: We declare this as a classmethod to allow for derived classes to use it.

        Args:
            robot_cfg (Union[Dict, RobotConfig]): _description_
            model_data_dict (Dict): _description_
            cost_data_dict (Dict): _description_
            constraint_data_dict (Dict): _description_
            convergence_data_dict (Dict): _description_
            world_coll_checker_dict (Optional[Dict], optional): _description_. Defaults to None.
            world_model_dict (Optional[Dict], optional): _description_. Defaults to None.
            world_coll_checker (Optional[WorldCollision], optional): _description_. Defaults to None.
            tensor_args (TensorDeviceType, optional): _description_. Defaults to TensorDeviceType().

        Returns:
            _type_: _description_
        """
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        world_coll_checker = cls.world_coll_checker_from_dict(
            world_coll_checker_dict, world_model_dict, world_coll_checker, tensor_args
        )
        model = cls.model_from_dict(model_data_dict, robot_cfg, tensor_args=tensor_args)
        cost = cls.cost_from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        constraint = cls.cost_from_dict(
            constraint_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        convergence = cls.cost_from_dict(
            convergence_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        return cls(
            model_cfg=model,
            cost_cfg=cost,
            constraint_cfg=constraint,
            convergence_cfg=convergence,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )


class ArmBase(RolloutBase, ArmBaseConfig):
    """
    This rollout function is for reaching a cartesian pose for a robot
    """

    @profiler.record_function("arm_base/init")
    def __init__(self, config: Optional[ArmBaseConfig] = None):
        if config is not None:
            ArmBaseConfig.__init__(self, **vars(config))
        RolloutBase.__init__(self)
        self._init_after_config_load()

    @profiler.record_function("arm_base/init_after_config_load")
    def _init_after_config_load(self):
        # self.current_state = None
        # self.retract_state = None
        self._goal_buffer = Goal()
        self._goal_idx_update = True
        # Create the dynamical system used for rollouts
        self.dynamics_model = KinematicModel(self.model_cfg)

        self.n_dofs = self.dynamics_model.n_dofs
        self.traj_dt = self.dynamics_model.traj_dt
        if self.cost_cfg.bound_cfg is not None:
            self.cost_cfg.bound_cfg.set_bounds(
                self.dynamics_model.get_state_bounds(),
                teleport_mode=self.dynamics_model.teleport_mode,
            )
            self.cost_cfg.bound_cfg.cspace_distance_weight = (
                self.dynamics_model.cspace_distance_weight
            )
            self.cost_cfg.bound_cfg.state_finite_difference_mode = (
                self.dynamics_model.state_finite_difference_mode
            )
            self.cost_cfg.bound_cfg.update_vec_weight(self.dynamics_model.null_space_weight)

            if self.cost_cfg.null_space_cfg is not None:
                self.cost_cfg.bound_cfg.null_space_weight = self.cost_cfg.null_space_cfg.weight
                log_warn(
                    "null space cost is deprecated, use null_space_weight in bound cost instead"
                )
            self.cost_cfg.bound_cfg.dof = self.n_dofs
            self.bound_cost = BoundCost(self.cost_cfg.bound_cfg)

        if self.cost_cfg.manipulability_cfg is not None:
            self.manipulability_cost = ManipulabilityCost(self.cost_cfg.manipulability_cfg)

        if self.cost_cfg.stop_cfg is not None:
            self.cost_cfg.stop_cfg.horizon = self.dynamics_model.horizon
            self.cost_cfg.stop_cfg.dt_traj_params = self.dynamics_model.dt_traj_params
            self.stop_cost = StopCost(self.cost_cfg.stop_cfg)
        self._goal_buffer.retract_state = self.retract_state
        if self.cost_cfg.primitive_collision_cfg is not None:
            self.primitive_collision_cost = PrimitiveCollisionCost(
                self.cost_cfg.primitive_collision_cfg
            )
            if self.dynamics_model.robot_model.total_spheres == 0:
                self.primitive_collision_cost.disable_cost()

        if self.cost_cfg.self_collision_cfg is not None:
            self.cost_cfg.self_collision_cfg.self_collision_kin_config = (
                self.dynamics_model.robot_model.get_self_collision_config()
            )
            self.robot_self_collision_cost = SelfCollisionCost(self.cost_cfg.self_collision_cfg)
            if self.dynamics_model.robot_model.total_spheres == 0:
                self.robot_self_collision_cost.disable_cost()

        # setup constraint terms:
        if self.constraint_cfg.primitive_collision_cfg is not None:
            self.primitive_collision_constraint = PrimitiveCollisionCost(
                self.constraint_cfg.primitive_collision_cfg
            )
            if self.dynamics_model.robot_model.total_spheres == 0:
                self.primitive_collision_constraint.disable_cost()

        if self.constraint_cfg.self_collision_cfg is not None:
            self.constraint_cfg.self_collision_cfg.self_collision_kin_config = (
                self.dynamics_model.robot_model.get_self_collision_config()
            )
            self.robot_self_collision_constraint = SelfCollisionCost(
                self.constraint_cfg.self_collision_cfg
            )

            if self.dynamics_model.robot_model.total_spheres == 0:
                self.robot_self_collision_constraint.disable_cost()

        self.constraint_cfg.bound_cfg.set_bounds(
            self.dynamics_model.get_state_bounds(), teleport_mode=self.dynamics_model.teleport_mode
        )
        self.constraint_cfg.bound_cfg.cspace_distance_weight = (
            self.dynamics_model.cspace_distance_weight
        )
        self.cost_cfg.bound_cfg.state_finite_difference_mode = (
            self.dynamics_model.state_finite_difference_mode
        )
        self.cost_cfg.bound_cfg.dof = self.n_dofs
        self.constraint_cfg.bound_cfg.dof = self.n_dofs
        self.bound_constraint = BoundCost(self.constraint_cfg.bound_cfg)

        if self.convergence_cfg.null_space_cfg is not None:
            self.convergence_cfg.null_space_cfg.dof = self.n_dofs
            self.null_convergence = DistCost(self.convergence_cfg.null_space_cfg)

        # set start state:
        start_state = torch.randn(
            (1, self.dynamics_model.d_state), **(self.tensor_args.as_torch_dict())
        )
        self._start_state = JointState(
            position=start_state[:, : self.dynamics_model.d_dof],
            velocity=start_state[:, : self.dynamics_model.d_dof],
            acceleration=start_state[:, : self.dynamics_model.d_dof],
        )
        self.update_cost_dt(self.dynamics_model.dt_traj_params.base_dt)
        return RolloutBase._init_after_config_load(self)

    def cost_fn(self, state: KinematicModelState, action_batch=None, return_list=False):
        # ee_pos_batch, ee_rot_batch = state_dict["ee_pos_seq"], state_dict["ee_rot_seq"]
        state_batch = state.state_seq
        cost_list = []

        # compute state bound  cost:
        if self.bound_cost.enabled:
            with profiler.record_function("cost/bound"):
                c = self.bound_cost.forward(
                    state_batch,
                    self._goal_buffer.retract_state,
                    self._goal_buffer.batch_retract_state_idx,
                )
                cost_list.append(c)
        if self.cost_cfg.manipulability_cfg is not None and self.manipulability_cost.enabled:
            raise NotImplementedError("Manipulability Cost is not implemented")
        if self.cost_cfg.stop_cfg is not None and self.stop_cost.enabled:
            st_cost = self.stop_cost.forward(state_batch.velocity)
            cost_list.append(st_cost)
        if self.cost_cfg.self_collision_cfg is not None and self.robot_self_collision_cost.enabled:
            with profiler.record_function("cost/self_collision"):
                coll_cost = self.robot_self_collision_cost.forward(state.robot_spheres)
                # cost += coll_cost
                cost_list.append(coll_cost)
        if (
            self.cost_cfg.primitive_collision_cfg is not None
            and self.primitive_collision_cost.enabled
        ):
            with profiler.record_function("cost/collision"):
                coll_cost = self.primitive_collision_cost.forward(
                    state.robot_spheres,
                    env_query_idx=self._goal_buffer.batch_world_idx,
                )
                cost_list.append(coll_cost)
        if return_list:
            return cost_list
        if self.sum_horizon:
            cost = cat_sum_horizon(cost_list)
        else:
            cost = cat_sum(cost_list)
        return cost

    def constraint_fn(
        self,
        state: KinematicModelState,
        out_metrics: Optional[RolloutMetrics] = None,
        use_batch_env: bool = True,
    ) -> RolloutMetrics:
        # setup constraint terms:

        constraint = self.bound_constraint.forward(state.state_seq)

        constraint_list = [constraint]
        if (
            self.constraint_cfg.primitive_collision_cfg is not None
            and self.primitive_collision_constraint.enabled
        ):
            if use_batch_env and self._goal_buffer.batch_world_idx is not None:
                coll_constraint = self.primitive_collision_constraint.forward(
                    state.robot_spheres,
                    env_query_idx=self._goal_buffer.batch_world_idx,
                )
            else:
                coll_constraint = self.primitive_collision_constraint.forward(
                    state.robot_spheres, env_query_idx=None
                )

            constraint_list.append(coll_constraint)
        if (
            self.constraint_cfg.self_collision_cfg is not None
            and self.robot_self_collision_constraint.enabled
        ):
            self_constraint = self.robot_self_collision_constraint.forward(state.robot_spheres)
            constraint_list.append(self_constraint)
        constraint = cat_sum(constraint_list)

        feasible = constraint == 0.0

        if out_metrics is None:
            out_metrics = RolloutMetrics()
        out_metrics.feasible = feasible
        out_metrics.constraint = constraint
        return out_metrics

    def get_metrics(self, state: Union[JointState, KinematicModelState]):
        """Compute metrics given state

        Args:
            state (Union[JointState, URDFModelState]): _description_

        Returns:
            _type_: _description_

        """
        if self.cuda_graph_instance:
            log_error("Cuda graph is using this instance, please break the graph before using this")
        if isinstance(state, JointState):
            state = self._get_augmented_state(state)
        out_metrics = self.constraint_fn(state)
        out_metrics.state = state
        out_metrics = self.convergence_fn(state, out_metrics)
        out_metrics.cost = self.cost_fn(state)
        return out_metrics

    def get_metrics_cuda_graph(self, state: JointState):
        """Use a CUDA Graph to compute metrics

        Args:
            state: _description_

        Raises:
            ValueError: _description_

        Returns:
            _description_
        """
        if not self._metrics_cuda_graph_init:
            # create new cuda graph for metrics:
            self._cu_metrics_state_in = state.detach().clone()
            s = torch.cuda.Stream(device=self.tensor_args.device)
            s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._cu_out_metrics = self.get_metrics(self._cu_metrics_state_in)
            torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)
            self.cu_metrics_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cu_metrics_graph, stream=s):
                self._cu_out_metrics = self.get_metrics(self._cu_metrics_state_in)
            self._metrics_cuda_graph_init = True
            self._cuda_graph_valid = True
        if not self.cuda_graph_instance:
            log_error("cuda graph is invalid")
        if self._cu_metrics_state_in.position.shape != state.position.shape:
            log_error("cuda graph changed")
        self._cu_metrics_state_in.copy_(state)
        self.cu_metrics_graph.replay()
        out_metrics = self._cu_out_metrics
        return out_metrics.clone()

    @abstractmethod
    def convergence_fn(
        self, state: KinematicModelState, out_metrics: Optional[RolloutMetrics] = None
    ):
        if out_metrics is None:
            out_metrics = RolloutMetrics()
        return out_metrics

    def _get_augmented_state(self, state: JointState) -> KinematicModelState:
        aug_state = self.compute_kinematics(state)
        if len(aug_state.state_seq.position.shape) == 2:
            aug_state.state_seq = aug_state.state_seq.unsqueeze(1)
            aug_state.ee_pos_seq = aug_state.ee_pos_seq.unsqueeze(1)
            aug_state.ee_quat_seq = aug_state.ee_quat_seq.unsqueeze(1)
            if aug_state.lin_jac_seq is not None:
                aug_state.lin_jac_seq = aug_state.lin_jac_seq.unsqueeze(1)
            if aug_state.ang_jac_seq is not None:
                aug_state.ang_jac_seq = aug_state.ang_jac_seq.unsqueeze(1)
            aug_state.robot_spheres = aug_state.robot_spheres.unsqueeze(1)
            aug_state.link_pos_seq = aug_state.link_pos_seq.unsqueeze(1)
            aug_state.link_quat_seq = aug_state.link_quat_seq.unsqueeze(1)
        return aug_state

    def compute_kinematics(self, state: JointState) -> KinematicModelState:
        # assume input is joint state?
        h = 0
        current_state = state  # .detach().clone()
        if len(current_state.position.shape) == 1:
            current_state = current_state.unsqueeze(0)

        q = current_state.position
        if len(q.shape) == 3:
            b, h, _ = q.shape
            q = q.view(b * h, -1)

        (
            ee_pos_seq,
            ee_rot_seq,
            lin_jac_seq,
            ang_jac_seq,
            link_pos_seq,
            link_rot_seq,
            link_spheres,
        ) = self.dynamics_model.robot_model.forward(q)

        if h != 0:
            ee_pos_seq = ee_pos_seq.view(b, h, 3)
            ee_rot_seq = ee_rot_seq.view(b, h, 4)
            if lin_jac_seq is not None:
                lin_jac_seq = lin_jac_seq.view(b, h, 3, self.n_dofs)
            if ang_jac_seq is not None:
                ang_jac_seq = ang_jac_seq.view(b, h, 3, self.n_dofs)
            link_spheres = link_spheres.view(b, h, link_spheres.shape[-2], link_spheres.shape[-1])
            link_pos_seq = link_pos_seq.view(b, h, -1, 3)
            link_rot_seq = link_rot_seq.view(b, h, -1, 4)

        state = KinematicModelState(
            current_state,
            ee_pos_seq,
            ee_rot_seq,
            link_spheres,
            link_pos_seq,
            link_rot_seq,
            lin_jac_seq,
            ang_jac_seq,
            link_names=self.kinematics.link_names,
        )
        return state

    def rollout_constraint(
        self, act_seq: torch.Tensor, use_batch_env: bool = True
    ) -> RolloutMetrics:
        if self.cuda_graph_instance:
            log_error("Cuda graph is using this instance, please break the graph before using this")
        state = self.dynamics_model.forward(self.start_state, act_seq)
        metrics = self.constraint_fn(state, use_batch_env=use_batch_env)
        return metrics

    def rollout_constraint_cuda_graph(self, act_seq: torch.Tensor, use_batch_env: bool = True):
        # TODO: move this to RolloutBase
        if not self._rollout_constraint_cuda_graph_init:
            # create new cuda graph for metrics:
            self._cu_rollout_constraint_act_in = act_seq.clone()
            s = torch.cuda.Stream(device=self.tensor_args.device)
            s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    state = self.dynamics_model.forward(self.start_state, act_seq)
                    self._cu_rollout_constraint_out_metrics = self.constraint_fn(
                        state, use_batch_env=use_batch_env
                    )
            torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)
            self.cu_rollout_constraint_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cu_rollout_constraint_graph, stream=s):
                state = self.dynamics_model.forward(self.start_state, act_seq)
                self._cu_rollout_constraint_out_metrics = self.constraint_fn(
                    state, use_batch_env=use_batch_env
                )
            self._rollout_constraint_cuda_graph_init = True
            self._cuda_graph_valid = True
        if not self.cuda_graph_instance:
            log_error("cuda graph is invalid")
        self._cu_rollout_constraint_act_in.copy_(act_seq)
        self.cu_rollout_constraint_graph.replay()
        out_metrics = self._cu_rollout_constraint_out_metrics
        return out_metrics.clone()

    def rollout_fn(self, act_seq) -> Trajectory:
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """

        # print(act_seq.shape, self._goal_buffer.batch_current_state_idx)
        if self.start_state is None:
            raise ValueError("start_state is not set in rollout")
        with profiler.record_function("robot_model/rollout"):
            state = self.dynamics_model.forward(
                self.start_state, act_seq, self._goal_buffer.batch_current_state_idx
            )

        with profiler.record_function("cost/all"):
            cost_seq = self.cost_fn(state, act_seq)

        sim_trajs = Trajectory(actions=act_seq, costs=cost_seq, state=state)

        return sim_trajs

    def update_params(self, goal: Goal):
        """
        Updates the goal targets for the cost functions.

        """
        with profiler.record_function("arm_base/update_params"):
            self._goal_buffer.copy_(goal, update_idx_buffers=self._goal_idx_update)

            if goal.current_state is not None:
                if self.start_state is None:
                    self.start_state = goal.current_state.clone()
                else:
                    self.start_state = self.start_state.copy_(goal.current_state)
            self.batch_size = goal.batch
        return True

    def get_ee_pose(self, current_state):
        current_state = current_state.to(**self.tensor_args)

        (ee_pos_batch, ee_quat_batch) = self.dynamics_model.robot_model.forward(
            current_state[:, : self.dynamics_model.n_dofs]
        )[0:2]

        state = KinematicModelState(current_state, ee_pos_batch, ee_quat_batch)
        return state

    def current_cost(self, current_state: JointState, no_coll=False, return_state=True, **kwargs):
        state = self._get_augmented_state(current_state)

        if "horizon_cost" not in kwargs:
            kwargs["horizon_cost"] = False

        cost = self.cost_fn(state, None, no_coll=no_coll, **kwargs)

        if return_state:
            return cost, state
        else:
            return cost

    def filter_robot_state(self, current_state: JointState) -> JointState:
        return self.dynamics_model.filter_robot_state(current_state)

    def get_robot_command(
        self,
        current_state: JointState,
        act_seq: torch.Tensor,
        shift_steps: int = 1,
        state_idx: Optional[torch.Tensor] = None,
    ) -> JointState:
        return self.dynamics_model.get_robot_command(
            current_state,
            act_seq,
            shift_steps=shift_steps,
            state_idx=state_idx,
        )

    def reset(self):
        self.dynamics_model.state_filter.reset()
        super().reset()

    @property
    def d_action(self):
        return self.dynamics_model.d_action

    @property
    def action_bound_lows(self):
        return self.dynamics_model.action_bound_lows

    @property
    def action_bound_highs(self):
        return self.dynamics_model.action_bound_highs

    @property
    def state_bounds(self) -> Dict[str, List[float]]:
        return self.dynamics_model.get_state_bounds()

    @property
    def dt(self):
        return self.dynamics_model.dt

    @property
    def horizon(self):
        return self.dynamics_model.horizon

    @property
    def action_horizon(self):
        return self.dynamics_model.action_horizon

    def get_init_action_seq(self) -> torch.Tensor:
        act_seq = self.dynamics_model.init_action_mean.unsqueeze(0).repeat(self.batch_size, 1, 1)
        return act_seq

    def reset_shape(self):
        self._goal_idx_update = True
        super().reset_shape()

    def reset_cuda_graph(self):
        super().reset_cuda_graph()

    def get_action_from_state(self, state: JointState):
        return self.dynamics_model.get_action_from_state(state)

    def get_state_from_action(
        self,
        start_state: JointState,
        act_seq: torch.Tensor,
        state_idx: Optional[torch.Tensor] = None,
    ):
        return self.dynamics_model.get_state_from_action(start_state, act_seq, state_idx)

    @property
    def kinematics(self):
        return self.dynamics_model.robot_model

    @property
    def cspace_config(self) -> CSpaceConfig:
        return self.dynamics_model.robot_model.kinematics_config.cspace

    def get_full_dof_from_solution(self, q_js: JointState) -> JointState:
        """This function will all the dof that are locked during optimization.


        Args:
            q_sol: _description_

        Returns:
            _description_
        """
        if self.kinematics.lock_jointstate is None:
            return q_js
        all_joint_names = self.kinematics.all_articulated_joint_names
        lock_joint_state = self.kinematics.lock_jointstate

        new_js = q_js.get_augmented_joint_state(all_joint_names, lock_joint_state)
        return new_js

    @property
    def joint_names(self) -> List[str]:
        return self.kinematics.joint_names

    @property
    def retract_state(self):
        return self.dynamics_model.retract_config

    def update_traj_dt(
        self,
        dt: Union[float, torch.Tensor],
        base_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        base_ratio: Optional[float] = None,
    ):
        self.dynamics_model.update_traj_dt(dt, base_dt, max_dt, base_ratio)
        self.update_cost_dt(dt)

    def update_cost_dt(self, dt: float):
        # scale any temporal costs by dt:
        self.bound_cost.update_dt(dt)
        if self.cost_cfg.primitive_collision_cfg is not None:
            self.primitive_collision_cost.update_dt(dt)
