# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SolverCore -- shared infrastructure component for all solvers.

SolverCore is NOT a base class. It is a component that manages rollouts, optimizers,
collision checker, kinematics, seeds, and goal buffers. Each solver (IK, TrajOpt, MPC)
owns a ``SolverCore`` instance via composition::

    class IKSolver:
        def __init__(self, config, ...):
            self.core = SolverCore(config.core_cfg, ...)
"""
from __future__ import annotations

# Standard Library
from typing import Dict, List, Optional, TypeVar, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

import curobo._src.runtime as curobo_runtime
from curobo._src.cost.cost_pose_metric import PoseCostMetric
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.geom.collision import (
    SceneCollision,
    create_collision_checker,
)
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.optim.optimizer_protocol import Optimizer
from curobo._src.optim.optim_factory import create_optimizer
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsState
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.manager_seed import SeedManager
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core_cfg import SolverCoreCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.torch_util import is_cuda_graph_reset_available

T_BDOF = TypeVar("T_BDOF", bound=torch.Tensor)


class SolverCore:
    """Manages shared solver infrastructure: rollouts, optimizer, collision, kinematics, seeds.

    This is a component, not a base class. Each solver owns an instance via
    ``self.core = SolverCore(...)``.
    """

    def __init__(
        self,
        config: SolverCoreCfg,
        scene_collision_checker: Optional[SceneCollision] = None,
    ):
        self.config = config
        self.device_cfg = config.device_cfg

        self._solve_state: Optional[SolveState] = None
        self._goal_buffer: Optional[GoalRegistry] = None
        self._task_initialized = False

        # Goal buffer manager
        self.goal_registry_manager = GoalManager(config.device_cfg)

        # Rollout and optimizer storage (populated by _initialize_components)
        self.metrics_rollout: RobotRollout
        self.additional_metrics_rollouts: Dict[str, RobotRollout] = {}
        self.optimizer_rollouts: List[RobotRollout]
        self.auxiliary_rollout: RobotRollout
        self.optimizers: List[Optimizer]
        self.optimizer: MultiStageOptimizer

        # Build rollouts, optimizers, collision checker
        self.scene_collision_checker = scene_collision_checker
        self._initialize_components()

        # Attachment manager
        self.attachment_manager = AttachmentManager(
            kinematics=self.kinematics,
            scene_collision=self.scene_collision_checker,
            device_cfg=self.device_cfg,
        )

        # Seed manager
        self.seed_manager = SeedManager(
            self.config.device_cfg,
            self.action_dim,
            self.auxiliary_rollout.action_bound_lows,
            self.auxiliary_rollout.action_bound_highs,
            random_seed=self.config.random_seed,
            action_horizon=self.auxiliary_rollout.action_horizon,
        )

        # Default initial state
        self.init_state = JointState.from_position(
            self.default_joint_position.unsqueeze(0),
            joint_names=self.joint_names,
        )

    @profiler.record_function("solver_core/initialize_components")
    def _initialize_components(self):
        """Build collision checker, rollouts, and optimizers from config."""
        # 1. Collision checker
        if self.scene_collision_checker is None:
            if self.config.scene_collision_cfg is not None:
                self.scene_collision_checker = create_collision_checker(
                    self.config.scene_collision_cfg
                )

        # 2. Rollouts
        self.metrics_rollout = RobotRollout(
            self.config.metrics_rollout_config,
            self.scene_collision_checker,
            use_cuda_graph=self.config.use_cuda_graph,
        )
        self.metrics_rollout.rollout_instance_name = "metrics_rollout"

        self.auxiliary_rollout = RobotRollout(
            self.config.metrics_rollout_config, self.scene_collision_checker
        )
        self.auxiliary_rollout.rollout_instance_name = "auxiliary_rollout"

        # 3. Optimizers
        if self.config.store_debug:
            self.config.use_cuda_graph = False

        for opt_config in self.config.optimizer_configs:
            opt_config.store_debug = self.config.store_debug

        self.optimizers = []
        self.optimizer_rollouts = []
        for i, opt_config in enumerate(self.config.optimizer_configs):
            optimizer_rollout_list = [
                RobotRollout(
                    self.config.optimizer_rollout_configs[i], self.scene_collision_checker
                )
                for _ in range(opt_config.num_rollout_instances)
            ]
            for j, rollout in enumerate(optimizer_rollout_list):
                rollout.rollout_instance_name = f"optimizer_rollout_{i}_{j}"

            optimizer = create_optimizer(
                opt_config, optimizer_rollout_list, self.config.use_cuda_graph
            )
            self.optimizers.append(optimizer)
            self.optimizer_rollouts.extend(optimizer_rollout_list)

        self.optimizer = MultiStageOptimizer(optimizers=self.optimizers)

    # -----------------------------------------------------------------------
    # Properties delegated from rollouts / kinematics
    # -----------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action space (robot degrees of freedom)."""
        return self.metrics_rollout.action_dim

    @property
    def kinematics(self) -> Kinematics:
        """Direct access to the CUDA robot model."""
        return self.auxiliary_rollout.transition_model.robot_model

    @property
    def transition_model(self):
        """Direct access to the transition model."""
        return self.auxiliary_rollout.transition_model

    @property
    def action_horizon(self) -> int:
        """The action horizon of the rollouts."""
        return self.auxiliary_rollout.action_horizon

    @property
    def default_joint_position(self) -> torch.Tensor:
        """The default joint position configuration."""
        return self.auxiliary_rollout.default_joint_position

    @property
    def default_joint_state(self) -> JointState:
        """The default joint state."""
        return JointState.from_position(
            self.auxiliary_rollout.default_joint_state, joint_names=self.joint_names
        )

    @property
    def joint_names(self) -> List[str]:
        """Ordered list of joint names used in optimization."""
        return self.kinematics.joint_names

    @property
    def tool_frames(self) -> List[str]:
        """Ordered list of target link names for which poses are computed/controlled."""
        return self.kinematics.tool_frames.copy()

    @property
    def solve_state(self) -> SolveState:
        return self._solve_state

    # -----------------------------------------------------------------------
    # Rollout management
    # -----------------------------------------------------------------------

    @profiler.record_function("solver_core/get_all_rollout_instances")
    def get_all_rollout_instances(
        self,
        include_optimizer_rollouts: bool = True,
        include_auxiliary_rollout: bool = True,
    ) -> List[RobotRollout]:
        """Returns all rollout instances."""
        rollouts = [self.metrics_rollout]
        if include_auxiliary_rollout:
            rollouts.append(self.auxiliary_rollout)
        if include_optimizer_rollouts:
            rollouts += self.optimizer_rollouts
        return rollouts

    @profiler.record_function("solver_core/update_rollout_params")
    def update_rollout_params(
        self, goal_buffer: GoalRegistry, include_auxiliary_rollout: bool = True
    ):
        """Update goal parameters on all rollouts."""
        for rollout in self.get_all_rollout_instances(
            include_optimizer_rollouts=False,
            include_auxiliary_rollout=include_auxiliary_rollout,
        ):
            rollout.update_params(goal_buffer)
        self.optimizer.update_rollout_params(goal_buffer)

    @profiler.record_function("solver_core/reset_shape")
    def reset_shape(self):
        """Resets the shape of internal components, often needed when batch size changes."""
        self.metrics_rollout.reset_shape()
        self.optimizer.reset_shape()
        if hasattr(self.auxiliary_rollout, "reset_shape"):
            self.auxiliary_rollout.reset_shape()
        for rollout in self.additional_metrics_rollouts.values():
            rollout.reset_shape()
        self.reset_cuda_graph()

    @profiler.record_function("solver_core/reset_seed")
    def reset_seed(self):
        """Resets the seed of the action sample generator."""
        self.seed_manager.reset_seed()
        for rollout in self.get_all_rollout_instances():
            rollout.reset_seed()
        self.optimizer.reset_seed()

    @profiler.record_function("solver_core/reset_cuda_graph")
    def reset_cuda_graph(self):
        """Resets CUDA graphs if they are in use."""
        if not self.config.use_cuda_graph:
            return
        if not self._task_initialized:
            return
        if not is_cuda_graph_reset_available():
            log_and_raise("CUDA graph reset is not available.")
        if hasattr(self.optimizer, "reset_cuda_graph"):
            self.optimizer.reset_cuda_graph()
        if hasattr(self.metrics_rollout, "reset_cuda_graph"):
            self.metrics_rollout.reset_cuda_graph()
        for rollout in self.additional_metrics_rollouts.values():
            if hasattr(rollout, "reset_cuda_graph"):
                rollout.reset_cuda_graph()

    def destroy(self):
        """Release all CUDA graph resources unconditionally."""
        if hasattr(self.optimizer, "reset_cuda_graph"):
            self.optimizer.reset_cuda_graph()
        if hasattr(self.metrics_rollout, "reset_cuda_graph"):
            self.metrics_rollout.reset_cuda_graph()
        for rollout in self.additional_metrics_rollouts.values():
            if hasattr(rollout, "reset_cuda_graph"):
                rollout.reset_cuda_graph()

    # -----------------------------------------------------------------------
    # Goal buffer
    # -----------------------------------------------------------------------

    @profiler.record_function("solver_core/prepare_goal_buffer")
    def prepare_goal_buffer(
        self,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        use_implicit_goal: bool = False,
        seed_goal_state: Optional[JointState] = None,
        goal_state: Optional[JointState] = None,
    ):
        """Update the internal goal buffer.

        Returns:
            Tuple of (goal_buffer, update_reference). update_reference is True when
            the buffer structure changed (new batch size, goal type change).
        """
        goal_buffer, update_reference = self.goal_registry_manager.update_goal_buffer(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            current_js=current_state,
            seed_goal_js=seed_goal_state,
            goal_js=goal_state,
            use_implicit_goal=use_implicit_goal,
        )

        self._solve_state = self.goal_registry_manager.solve_state
        self._goal_buffer = goal_buffer

        if update_reference:
            self.reset_shape()
            self.optimizer.update_num_problems(self._get_problem_batch_size(solve_state))
            self.metrics_rollout.update_batch_size(self._get_problem_batch_size(solve_state))
            for rollout in self.additional_metrics_rollouts.values():
                rollout.update_batch_size(self._get_problem_batch_size(solve_state))
            self._task_initialized = True

        return self._goal_buffer, update_reference

    def _get_problem_batch_size(self, solve_state: SolveState) -> int:
        """Get the problem batch size from solve state.

        The actual batch size depends on the solve type (IK uses seeds, TrajOpt uses seeds).
        """
        # For IK: batch * num_ik_seeds
        # For TrajOpt/MPC: batch * num_trajopt_seeds
        if solve_state.num_ik_seeds is not None and solve_state.num_ik_seeds > 0:
            return solve_state.get_ik_batch_size()
        return solve_state.get_trajopt_batch_size()

    # -----------------------------------------------------------------------
    # Seed preparation
    # -----------------------------------------------------------------------

    @profiler.record_function("solver_core/prepare_action_seeds")
    def prepare_action_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        seed_config: Optional[T_BDOF] = None,
        current_state: Optional[JointState] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepares seed configurations for IK optimization using the SeedManager."""
        return self.seed_manager.prepare_action_seeds(
            batch_size, num_seeds, seed_config, current_state, seed_traj
        )

    @profiler.record_function("solver_core/prepare_trajectory_seeds")
    def prepare_trajectory_seeds(
        self,
        batch_size: int,
        num_seeds: int,
        current_state: JointState,
        seed_config: Optional[T_BDOF] = None,
        seed_traj: Optional[torch.Tensor] = None,
    ):
        """Prepares seed trajectories for optimization."""
        return self.seed_manager.prepare_trajectory_seeds(
            batch_size, num_seeds, current_state, seed_config, seed_traj
        )

    # -----------------------------------------------------------------------
    # Cost toggling (weight factors passed as args, not read from config)
    # -----------------------------------------------------------------------

    @profiler.record_function("solver_core/enable_tool_pose_tracking")
    def enable_tool_pose_tracking(
        self,
        tool_frames: Optional[List[str]] = None,
        non_terminal_weight_factor: float = 0.0,
    ) -> None:
        """Enable goal pose tracking for the given links.

        Args:
            tool_frames: Links to enable tracking for. None means all links.
            non_terminal_weight_factor: Weight factor for non-terminal steps.
        """
        if tool_frames is None:
            tool_frames = self.tool_frames
        tool_pose_criteria = {
            k: ToolPoseCriteria.track_position_and_orientation(
                non_terminal_scale=non_terminal_weight_factor
            )
            for k in tool_frames
        }
        self.update_tool_pose_criteria(tool_pose_criteria)

    @profiler.record_function("solver_core/disable_tool_pose_tracking")
    def disable_tool_pose_tracking(self, tool_frames: Optional[List[str]] = None) -> None:
        """Disable goal pose tracking for the given links.

        Args:
            tool_frames: Links to disable tracking for. None means all links.
        """
        if tool_frames is None:
            tool_frames = self.tool_frames
        tool_pose_criteria = {k: ToolPoseCriteria.disabled() for k in tool_frames}
        self.update_tool_pose_criteria(tool_pose_criteria)

    @profiler.record_function("solver_core/enable_joint_position_tracking")
    def enable_joint_position_tracking(self) -> None:
        """Enable joint position tracking."""
        for rollout in self.get_all_rollout_instances(include_optimizer_rollouts=True):
            cspace_costs = rollout.get_cost_component_by_name("cspace")
            for cspace_cost in cspace_costs:
                if cspace_cost is not None:
                    cspace_cost.enable_cspace_target()
            target_cspace_costs = rollout.get_cost_component_by_name("target_cspace_dist")
            for target_cspace_cost in target_cspace_costs:
                if target_cspace_cost is not None:
                    target_cspace_cost.disable_cost()

    @profiler.record_function("solver_core/disable_joint_position_tracking")
    def disable_joint_position_tracking(self) -> None:
        """Disable joint position tracking."""
        for rollout in self.get_all_rollout_instances(include_optimizer_rollouts=True):
            cspace_costs = rollout.get_cost_component_by_name("cspace")
            for cspace_cost in cspace_costs:
                if cspace_cost is not None:
                    cspace_cost.disable_cspace_target()
            target_cspace_costs = rollout.get_cost_component_by_name("target_cspace_dist")
            for target_cspace_cost in target_cspace_costs:
                if target_cspace_cost is not None:
                    target_cspace_cost.enable_cost()

    # -----------------------------------------------------------------------
    # Pose cost / criteria updates
    # -----------------------------------------------------------------------

    @profiler.record_function("solver_core/update_pose_cost_metric")
    def update_pose_cost_metric(self, pose_cost_metric: Dict[str, PoseCostMetric]):
        """Update pose cost metric weights/parameters for all rollouts."""
        for link_name in pose_cost_metric.keys():
            if link_name not in self.tool_frames:
                log_and_raise(
                    f"Link '{link_name}' not found in target link names: {self.tool_frames}"
                )
        self.metrics_rollout.update_params_cost_managers(pose_cost_metric=pose_cost_metric)
        for rollout in self.additional_metrics_rollouts.values():
            rollout.update_params_cost_managers(pose_cost_metric=pose_cost_metric)
        for rollout in self.optimizer_rollouts:
            rollout.update_params_cost_managers(pose_cost_metric=pose_cost_metric)
        self.auxiliary_rollout.update_params_cost_managers(pose_cost_metric=pose_cost_metric)

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        """Update tool pose criteria for all rollouts."""
        self.metrics_rollout.update_params_cost_managers(tool_pose_criteria=tool_pose_criteria)
        for rollout in self.additional_metrics_rollouts.values():
            rollout.update_params_cost_managers(tool_pose_criteria=tool_pose_criteria)
        for rollout in self.optimizer_rollouts:
            rollout.update_params_cost_managers(tool_pose_criteria=tool_pose_criteria)
        self.auxiliary_rollout.update_params_cost_managers(tool_pose_criteria=tool_pose_criteria)

    # -----------------------------------------------------------------------
    # Sample configs (collision activation distance passed as arg)
    # -----------------------------------------------------------------------

    @profiler.record_function("solver_core/sample_configs")
    def sample_configs(
        self,
        num_samples: int,
        rejection_ratio: int = 10,
        optimizer_collision_activation_distance: float = 0.01,
    ) -> torch.Tensor:
        """Samples feasible joint configurations using rejection sampling.

        Args:
            num_samples: Desired number of feasible configurations.
            rejection_ratio: How many candidates per desired sample.
            optimizer_collision_activation_distance: Collision activation distance.

        Returns:
            Tensor of shape (n, dof) with feasible joint configurations, n <= num_samples.
        """
        if num_samples <= 0:
            return torch.zeros(
                (1, self.action_dim), device=self.device_cfg.device, dtype=self.device_cfg.dtype
            )

        total_to_generate = num_samples * rejection_ratio
        samples = self.auxiliary_rollout.sample_random_actions(total_to_generate)
        samples = samples.view(total_to_generate, 1, self.action_dim)
        self.auxiliary_rollout.update_batch_size(samples.shape[0])

        # Temporarily adjust collision activation distance
        collision_cfg = self.auxiliary_rollout.constraint_manager.config.scene_collision_cfg
        og_collision_activation_distance = None
        if collision_cfg is not None:
            og_collision_activation_distance = collision_cfg.activation_distance.clone()
            collision_cfg.activation_distance[:] = optimizer_collision_activation_distance

        # Disable tracking for sampling
        self.disable_tool_pose_tracking()
        self.disable_joint_position_tracking()

        cost_name = "cspace"
        if self.auxiliary_rollout.constraint_manager.has_cost(cost_name):
            cspace_cost = self.auxiliary_rollout.constraint_manager.get_cost(cost_name)
            cspace_cost.disable_cspace_target()

        metrics = self.auxiliary_rollout.compute_metrics_from_action(samples)

        feasible = metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,
        )
        feasible = feasible.view(total_to_generate)
        samples = samples.view(total_to_generate, self.action_dim)
        feasible_samples = samples[feasible]

        # Restore state
        self.enable_tool_pose_tracking()
        self.enable_joint_position_tracking()
        if collision_cfg is not None and og_collision_activation_distance is not None:
            collision_cfg.activation_distance[:] = og_collision_activation_distance
        if self.auxiliary_rollout.constraint_manager.has_cost(cost_name):
            cspace_cost.enable_cspace_target()

        return feasible_samples[:num_samples]

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        """Computes forward kinematics for a given joint state."""
        return self.kinematics.compute_kinematics(state).clone()

    def get_active_js(self, full_js: JointState) -> JointState:
        return self.kinematics.get_active_js(full_js)

    def get_full_js(self, active_js: JointState) -> JointState:
        return self.kinematics.get_full_js(active_js)

    def update_link_inertial(
        self,
        link_name: str,
        mass: Optional[float] = None,
        com: Optional[torch.Tensor] = None,
        inertia: Optional[torch.Tensor] = None,
    ) -> None:
        """Update inertial properties of a single link across all rollout instances."""
        for rollout in self.get_all_rollout_instances():
            rollout.transition_model.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(
        self,
        link_properties: dict[str, dict[str, Union[float, torch.Tensor]]],
    ) -> None:
        """Update inertial properties of multiple links across all rollout instances."""
        for rollout in self.get_all_rollout_instances():
            rollout.transition_model.update_links_inertial(link_properties)

    @profiler.record_function("solver_core/debug_dump")
    def debug_dump(self, file_path: str):
        if not curobo_runtime.debug_cuda_graphs:
            log_warn("CUDA Graph Debug Mode is not enabled, cannot dump CUDA Graph")
            return
        if self.config.use_cuda_graph:
            self.optimizer.optimizers[-1].debug_dump(file_path)
