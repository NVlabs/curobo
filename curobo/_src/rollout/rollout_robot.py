# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot rollout that forward-simulates joint trajectories and evaluates costs.

Integrates the robot state-transition model, cost manager, and collision
checker to compute trajectory costs for the optimizer.  Optionally wraps
the forward pass in CUDA graphs via the ``use_cuda_graph`` constructor
parameter.
"""

from __future__ import annotations

# Standard Library
from typing import Dict, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
import curobo._src.runtime as curobo_runtime
from curobo._src.cost.cost_base import BaseCost
from curobo._src.geom.collision.collision_scene import SceneCollision, create_scene_collision
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import (
    CostCollection,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer


class RobotRollout:
    """Forward-simulates joint trajectories and evaluates motion-planning costs.

    Combines a state-transition model, cost manager, and collision checker to
    score candidate trajectories.  CUDA-graph wrapping for
    ``compute_metrics_from_state`` and ``compute_metrics_from_action`` is
    controlled by the ``use_cuda_graph`` constructor parameter. Direct time
    scaling uses a graph executor when runtime checks are skipped.
    """

    @profiler.record_function("robot_rollout/init")
    def __init__(
        self,
        config: Optional[RobotRolloutCfg] = None,
        scene_collision_checker: Optional[SceneCollision] = None,
        use_cuda_graph: bool = False,
    ):
        self.config = config
        self._num_particles_goal: Optional[GoalRegistry] = None
        self._metrics_goal: Optional[GoalRegistry] = None
        self.start_state: Optional[JointState] = None
        self._batch_size: Optional[int] = None
        self.rollout_instance_name: Optional[str] = None

        # CUDA graph state
        self._use_cuda_graph = use_cuda_graph
        self._compute_metrics_from_state_executor: Optional[GraphExecutor] = None
        self._compute_metrics_from_action_executor: Optional[GraphExecutor] = None
        self._compute_time_scaled_dt_from_action_executor: Optional[GraphExecutor] = None

        if config is not None:
            config.resolve_requirements()
            self.scene_collision_checker = scene_collision_checker
            self.device_cfg = config.device_cfg
            self.sum_horizon = config.sum_horizon
            self.sampler_seed = config.sampler_seed
            self._initialize_components()

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #

    @profiler.record_function("robot_rollout/init_components")
    def _initialize_components(self):
        """Create transition models, sampler, cost managers, and collision checker."""
        # Transition models: one for optimization, one for metrics
        self.transition_model = self.config.transition_model_cfg.class_type(
            self.config.transition_model_cfg
        )
        self.metrics_transition_model = self.config.transition_model_cfg.class_type(
            self.config.transition_model_cfg
        )

        # Halton sampler
        self.act_sample_gen = SampleBuffer.create_halton_sample_buffer(
            ndims=self.action_dim,
            device_cfg=self.device_cfg,
            up_bounds=self.action_bound_highs,
            low_bounds=self.action_bound_lows,
            seed=self.sampler_seed,
        )

        # Cost managers
        self.cost_manager = None
        self.constraint_manager = None
        self.hybrid_cost_constraint_manager = None
        self.metrics_cost_manager = None
        self.metrics_constraint_manager = None
        self.metrics_hybrid_cost_constraint_manager = None
        self.metrics_convergence_manager = None

        self._cost_manager_list: List = []

        if self.config.cost_cfg is not None:
            self.cost_manager = self.config.cost_cfg.class_type(self.device_cfg)
            self.metrics_cost_manager = self.config.cost_cfg.class_type(self.device_cfg)
            self._cost_manager_list.append(self.cost_manager)
            self._cost_manager_list.append(self.metrics_cost_manager)

        if self.config.constraint_cfg is not None:
            self.constraint_manager = self.config.constraint_cfg.class_type(self.device_cfg)
            self.metrics_constraint_manager = self.config.constraint_cfg.class_type(
                self.device_cfg
            )
            self._cost_manager_list.append(self.constraint_manager)
            self._cost_manager_list.append(self.metrics_constraint_manager)

        if self.config.hybrid_cost_constraint_cfg is not None:
            self.hybrid_cost_constraint_manager = (
                self.config.hybrid_cost_constraint_cfg.class_type(self.device_cfg)
            )
            self.metrics_hybrid_cost_constraint_manager = (
                self.config.hybrid_cost_constraint_cfg.class_type(self.device_cfg)
            )
            self._cost_manager_list.append(self.hybrid_cost_constraint_manager)
            self._cost_manager_list.append(self.metrics_hybrid_cost_constraint_manager)

        if self.config.convergence_cfg is not None:
            self.metrics_convergence_manager = self.config.convergence_cfg.class_type(
                self.device_cfg
            )
            self._cost_manager_list.append(self.metrics_convergence_manager)

        # Scene collision checker
        if self.scene_collision_checker is None:
            if self.config.scene_collision_cfg is not None:
                self.scene_collision_checker = create_scene_collision(
                    self.config.scene_collision_cfg
                )

        # Initialize cost managers with transition model and collision checker
        if self.config.cost_cfg is not None:
            self.cost_manager.initialize_from_config(
                self.config.cost_cfg, self.transition_model, self.scene_collision_checker
            )
            self.metrics_cost_manager.initialize_from_config(
                self.config.cost_cfg, self.metrics_transition_model, self.scene_collision_checker
            )

        if self.config.constraint_cfg is not None:
            self.constraint_manager.initialize_from_config(
                self.config.constraint_cfg, self.transition_model, self.scene_collision_checker
            )
            self.metrics_constraint_manager.initialize_from_config(
                self.config.constraint_cfg,
                self.metrics_transition_model,
                self.scene_collision_checker,
            )

        if self.config.hybrid_cost_constraint_cfg is not None:
            self.hybrid_cost_constraint_manager.initialize_from_config(
                self.config.hybrid_cost_constraint_cfg,
                self.transition_model,
                self.scene_collision_checker,
            )
            self.metrics_hybrid_cost_constraint_manager.initialize_from_config(
                self.config.hybrid_cost_constraint_cfg,
                self.metrics_transition_model,
                self.scene_collision_checker,
            )

        if self.config.convergence_cfg is not None:
            self.metrics_convergence_manager.initialize_from_config(
                self.config.convergence_cfg, self.transition_model, self.scene_collision_checker
            )

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def action_dim(self) -> int:
        return self.transition_model.action_dim

    @property
    def action_horizon(self) -> int:
        return self.transition_model.action_horizon

    @property
    def horizon(self) -> int:
        return self.transition_model.horizon

    def get_hessian_block_half_bandwidth(self) -> Optional[int]:
        """Derive conservative action-knot Hessian bandwidth from all costs."""
        dense_temporal_support_span = max(
            (
                manager.dense_temporal_support_span
                for manager in (
                    self.cost_manager,
                    self.constraint_manager,
                    self.hybrid_cost_constraint_manager,
                )
                if manager is not None
            ),
            default=0,
        )
        return self.transition_model.get_hessian_block_half_bandwidth(dense_temporal_support_span)

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self.transition_model.action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self.transition_model.action_bound_highs

    @property
    def action_bounds(self) -> torch.Tensor:
        return self.device_cfg.to_device(
            torch.stack([self.action_bound_lows, self.action_bound_highs])
        )

    @property
    def state_bounds(self):
        return self.action_bounds

    @property
    def dt(self) -> float:
        return self.transition_model.dt

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    @property
    def default_joint_state(self) -> torch.Tensor:
        return self.transition_model.default_joint_position

    @property
    def default_joint_position(self) -> torch.Tensor:
        return self.transition_model.default_joint_position

    @property
    def valid_compute_metrics_from_state_cuda_graph(self) -> bool:
        return (
            self._compute_metrics_from_state_executor is not None
            and self._compute_metrics_from_state_executor.is_initialized
        )

    @property
    def valid_compute_metrics_from_action_cuda_graph(self) -> bool:
        return (
            self._compute_metrics_from_action_executor is not None
            and self._compute_metrics_from_action_executor.is_initialized
        )

    # ------------------------------------------------------------------ #
    #  Core: evaluate_action (optimization path)                           #
    # ------------------------------------------------------------------ #

    def evaluate_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutResult:
        """Compute costs and constraints from action sequence."""
        batch_size = act_seq.shape[0]
        if batch_size != self._batch_size:
            self.update_batch_size(batch_size)
        state = self._compute_state_from_action_impl(act_seq)
        costs_and_constraints = self._compute_costs_and_constraints_impl(state, **kwargs)
        return RolloutResult(
            actions=act_seq,
            state=state,
            costs_and_constraints=costs_and_constraints,
        )

    # ------------------------------------------------------------------ #
    #  Metrics (with optional CUDA graph)                                  #
    # ------------------------------------------------------------------ #

    @profiler.record_function("robot_rollout/compute_metrics_from_state")
    def compute_metrics_from_state(self, state: JointState, **kwargs) -> RolloutMetrics:
        """Evaluate costs, constraints, and convergence for a given state.

        Uses the *metrics* cost managers (not the optimization-path ones) so
        convergence criteria are also computed.  When ``use_cuda_graph`` is
        enabled the first call records a CUDA graph and subsequent calls
        replay it.

        Args:
            state: Joint state trajectory.  Shape
                ``(batch, horizon, n_dof)`` for position/velocity fields.

        Returns:
            :class:`RolloutMetrics` containing per-problem feasibility,
            cost/constraint values, and convergence tolerances.
        """
        if self._use_cuda_graph and not kwargs:
            if self._compute_metrics_from_state_executor is None:
                self._compute_metrics_from_state_executor = create_graph_executor(
                    capture_fn=self._compute_metrics_from_state_impl,
                    device=self.device_cfg.device,
                )
            return self._compute_metrics_from_state_executor(state)
        return self._compute_metrics_from_state_impl(state, **kwargs)

    @profiler.record_function("robot_rollout/compute_metrics_from_action")
    def compute_metrics_from_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutMetrics:
        """Forward-simulate actions and evaluate metrics including convergence.

        Integrates the action sequence through the metrics transition model
        to produce a state trajectory, then delegates to
        :meth:`compute_metrics_from_state`.  Batch size is updated
        automatically.  CUDA-graph replay is used when enabled.

        Args:
            act_seq: Action sequence tensor of shape
                ``(batch, action_horizon, action_dim)``.

        Returns:
            :class:`RolloutMetrics` with the ``actions`` field populated.
        """
        batch_size = act_seq.shape[0]
        self.update_batch_size(batch_size)
        if self._use_cuda_graph and not kwargs:
            if self._compute_metrics_from_action_executor is None:
                self._compute_metrics_from_action_executor = create_graph_executor(
                    capture_fn=self._compute_metrics_from_action_impl,
                    device=self.device_cfg.device,
                )
            return self._compute_metrics_from_action_executor(act_seq)
        return self._compute_metrics_from_action_impl(act_seq, **kwargs)

    # ------------------------------------------------------------------ #
    #  State computation (public, used by solvers)                         #
    # ------------------------------------------------------------------ #

    @profiler.record_function("robot_rollout/compute_time_scaled_dt_from_action")
    def compute_time_scaled_dt_from_action(
        self,
        act_seq: torch.Tensor,
        joint_limits: JointLimits,
        epsilon: float = 1e-3,
        minimum_dt: float = -1.0,
        maximum_dt: float = -1.0,
    ) -> torch.Tensor:
        """Compute one feasible absolute dt per action sequence.

        When runtime checks are skipped, repeated calls replay through a graph executor. Negative
        minimum or maximum bounds disable the corresponding clamp.
        """
        if curobo_runtime.skip_checks:
            if self._compute_time_scaled_dt_from_action_executor is None:
                self._compute_time_scaled_dt_from_action_executor = create_graph_executor(
                    capture_fn=self._compute_time_scaled_dt_from_action_impl,
                    device=self.device_cfg.device,
                    clone_outputs=False,
                    joint_limits=joint_limits,
                    epsilon=epsilon,
                    minimum_dt=minimum_dt,
                    maximum_dt=maximum_dt,
                )
            return self._compute_time_scaled_dt_from_action_executor(act_seq)
        return self._compute_time_scaled_dt_from_action_impl(
            act_seq,
            joint_limits,
            epsilon=epsilon,
            minimum_dt=minimum_dt,
            maximum_dt=maximum_dt,
        )

    def _compute_time_scaled_dt_from_action_impl(
        self,
        act_seq: torch.Tensor,
        joint_limits: JointLimits,
        epsilon: float = 1e-3,
        minimum_dt: float = -1.0,
        maximum_dt: float = -1.0,
    ) -> torch.Tensor:
        """Compute time scaling eagerly for direct execution or graph capture."""
        return self.transition_model.compute_time_scaled_dt_from_action(
            self.start_state,
            act_seq,
            joint_limits,
            epsilon=epsilon,
            minimum_dt=minimum_dt,
            maximum_dt=maximum_dt,
            start_state_idx=(
                self._num_particles_goal.idxs_current_js
                if self._num_particles_goal is not None
                else None
            ),
            goal_state=(
                self._num_particles_goal.seed_goal_js
                if self._num_particles_goal is not None
                else None
            ),
            goal_state_idx=(
                self._num_particles_goal.idxs_seed_goal_js
                if self._num_particles_goal is not None
                else None
            ),
            use_implicit_goal_state=(
                self._num_particles_goal.seed_enable_implicit_goal_js
                if self._num_particles_goal is not None
                else None
            ),
        )

    @profiler.record_function("robot_rollout/compute_state_from_action")
    def compute_state_from_action(self, act_seq: torch.Tensor, **kwargs) -> RobotState:
        return self._compute_state_from_action_impl(act_seq)

    def compute_state_from_action_metrics(self, act_seq: torch.Tensor, **kwargs) -> RobotState:
        return self._compute_state_from_action_metrics_impl(act_seq, **kwargs)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    @profiler.record_function("robot_rollout/update_params")
    def update_params(self, goal: GoalRegistry, num_particles: int = None) -> bool:
        """Update goal targets and rebatch for the current optimization round.

        Copies ``goal.current_js`` into the cached start state, expands the
        goal by ``num_particles`` seeds on the first call (creating
        ``_num_particles_goal``), and keeps a separate ``_metrics_goal`` clone
        with full index buffers.  On subsequent calls only tensor *values* are
        overwritten so the pre-allocated buffers are reused.

        Args:
            goal: Goal registry holding target poses, joint states, and
                index buffers.  Shape conventions follow
                :class:`GoalRegistry` (batch-first).
            num_particles: Number of particles (seeds) per problem used by
                the optimizer.  When not ``None`` the batch size is set to
                ``num_seeds * batch_size`` after expansion.

        Returns:
            ``True`` unconditionally (signature required by the
            :class:`Rollout` protocol).
        """
        batch_goal = goal

        if goal.current_js is not None:
            if self.start_state is None:
                self.start_state = goal.current_js.clone()
            else:
                self.start_state = self.start_state.copy_(goal.current_js, allow_clone=False)

        if self._num_particles_goal is None:
            if num_particles is not None:
                batch_goal = goal.repeat_seeds(num_particles, repeat_seed_idx_buffers=True)
            self._num_particles_goal = batch_goal
        else:
            self._num_particles_goal.copy_(goal, update_idx_buffers=False, allow_clone=False)

        if self._metrics_goal is None:
            self._metrics_goal = goal.clone()
        else:
            self._metrics_goal.copy_(goal, update_idx_buffers=True, allow_clone=False)

        if goal.seed_goal_js is not None and goal.seed_goal_js.dt is not None:
            self._alias_seed_goal_dt(goal.seed_goal_js.dt)

        if num_particles is not None:
            batch_size = self._num_particles_goal.num_seeds * self._num_particles_goal.batch_size
            self.update_batch_size(batch_size)
        return True

    def _alias_seed_goal_dt(self, goal_dt: torch.Tensor) -> None:
        """Point both cached goal registries at the central per-seed dt tensor."""
        for cached_goal in (self._metrics_goal, self._num_particles_goal):
            if cached_goal is None:
                log_and_raise("Rollout not initialized. Call update_params first.")
            if cached_goal.seed_goal_js is None:
                log_and_raise("seed_goal_js is None")
            cached_dt = cached_goal.seed_goal_js.dt
            if cached_dt is not None and cached_dt.shape != goal_dt.shape:
                log_and_raise("dt shape mismatch")
            cached_goal.seed_goal_js.dt = goal_dt

    def alias_goal_tool_poses(self, goal_tool_poses: GoalToolPose) -> None:
        """Point cached goal registries at one fixed-address tool-pose target."""
        for cached_goal in (self._metrics_goal, self._num_particles_goal):
            if cached_goal is None or cached_goal.link_goal_poses is None:
                log_and_raise("Tool-pose rollout goal has not been initialized")
            if cached_goal.link_goal_poses.shape != goal_tool_poses.shape:
                log_and_raise(
                    "Tool-pose goal shape mismatch: "
                    f"{cached_goal.link_goal_poses.shape} != {goal_tool_poses.shape}"
                )
            if cached_goal.link_goal_poses.tool_frames != goal_tool_poses.tool_frames:
                log_and_raise("Tool-pose goal frames cannot change after initialization")
            cached_goal.link_goal_poses = goal_tool_poses

    @property
    def has_goal_pose_routes(self) -> bool:
        """Return whether fixed tool-pose route buffers have been installed."""
        cached_goals = (self._metrics_goal, self._num_particles_goal)
        return all(goal is not None and goal.idxs_link_pose is not None for goal in cached_goals)

    def route_goal_poses(self, goal_query_idx: torch.Tensor) -> None:
        """Copy per-state tool-pose goal routes into fixed rollout buffers.

        This operation is allocation-free and safe to capture inside RRT CUDA
        graphs after :attr:`has_goal_pose_routes` becomes true.

        Args:
            goal_query_idx: Goal row for every rollout state with shape
                ``(batch, 1)`` and dtype ``torch.int32``.
        """
        for cached_goal in (self._metrics_goal, self._num_particles_goal):
            if cached_goal is None or cached_goal.idxs_link_pose is None:
                log_and_raise("Tool-pose rollout goal routes have not been initialized")
            if (
                cached_goal.idxs_link_pose.shape != goal_query_idx.shape
                or cached_goal.idxs_link_pose.dtype != goal_query_idx.dtype
                or cached_goal.idxs_link_pose.device != goal_query_idx.device
            ):
                log_and_raise(
                    "Tool-pose goal route shape, dtype, and device must match rollout buffers"
                )
            cached_goal.idxs_link_pose.copy_(goal_query_idx)

    def update_goal_dt(self, goal: GoalRegistry) -> bool:
        """Hot-patch the per-seed trajectory dt without a full goal update.

        Makes both the particle-expanded and metrics goal registries reference
        ``goal.seed_goal_js.dt``. Shape checks protect existing captured tensor
        contracts.

        Args:
            goal: Goal registry whose ``seed_goal_js.dt`` will be referenced.
                Must not be ``None`` and must have matching dt shape.

        Returns:
            ``True`` on success.

        Raises:
            RuntimeError: If the rollout has not been initialized via
                :meth:`update_params` or if dt shapes are mismatched.
        """
        if goal.seed_goal_js is None:
            log_and_raise("seed_goal_js is None")
        if goal.seed_goal_js.dt is None:
            log_and_raise("dt is None in seed_goal_js")
        self._alias_seed_goal_dt(goal.seed_goal_js.dt)
        return True

    def update_batch_size(self, batch_size: int) -> None:
        if self._batch_size is None or self._batch_size != batch_size:
            self._batch_size = batch_size
            self.transition_model.update_batch_size(batch_size)
            for manager in self._cost_manager_list:
                manager.setup_batch_tensors(batch_size, self.horizon)

    def update_dt(self, dt: float) -> None:
        self.transition_model.update_dt(dt)
        for manager in self._cost_manager_list:
            manager.update_dt(dt)

    def reset(
        self,
        reset_problem_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        if reset_problem_ids is not None and self._num_particles_goal is not None:
            return
        for manager in self._cost_manager_list:
            manager.reset(reset_problem_ids=reset_problem_ids, **kwargs)

    def reset_shape(self):
        self._num_particles_goal = None
        self._metrics_goal = None

    def reset_cuda_graph(self) -> bool:
        if self._compute_metrics_from_state_executor is not None:
            self._compute_metrics_from_state_executor.reset()
        if self._compute_metrics_from_action_executor is not None:
            self._compute_metrics_from_action_executor.reset()
        if self._compute_time_scaled_dt_from_action_executor is not None:
            self._compute_time_scaled_dt_from_action_executor.reset()
            self._compute_time_scaled_dt_from_action_executor = None
        return self.reset_shape()

    def reset_seed(self) -> None:
        self.act_sample_gen.reset()

    def filter_robot_state(self, current_state: JointState) -> JointState:
        return current_state

    def get_robot_command(
        self,
        current_state,
        act_seq,
        shift_steps: int = 1,
        state_idx: Optional[torch.Tensor] = None,
    ):
        log_and_raise("get_robot_command is not implemented")

    # ------------------------------------------------------------------ #
    #  Sampling                                                            #
    # ------------------------------------------------------------------ #

    def sample_random_actions(self, n: int = 0, bounded: bool = True) -> torch.Tensor:
        return self.act_sample_gen.get_samples(n, bounded=bounded)

    def get_initial_action(
        self, use_random: bool = True, use_zero: bool = False, **kwargs
    ) -> torch.Tensor:
        num_samples = self.batch_size
        if num_samples is None:
            num_samples = 1
        init_action = None
        if use_random:
            n_samples = num_samples * self.action_horizon
            init_action = self.sample_random_actions(n=n_samples, bounded=True)
        elif use_zero:
            init_action = torch.zeros(
                (num_samples, self.action_horizon, self.action_dim),
                **self.device_cfg.as_torch_dict(),
            )
        if init_action is None:
            log_and_raise("get_init_action_seq is not implemented")
        init_action = init_action.view(num_samples, self.action_horizon, self.action_dim)
        return init_action

    # ------------------------------------------------------------------ #
    #  Cost management                                                     #
    # ------------------------------------------------------------------ #

    def update_params_cost_managers(self, **kwargs) -> None:
        for manager in self._cost_manager_list:
            manager.update_params(**kwargs)

    def enable_cost_component(self, name: str) -> None:
        for manager in self._cost_manager_list:
            if name in manager.get_cost_component_names():
                manager.enable_cost_component(name)

    def disable_cost_component(self, name: str) -> None:
        for manager in self._cost_manager_list:
            if name in manager.get_cost_component_names():
                manager.disable_cost_component(name)

    def get_cost_component_names(self) -> List[str]:
        names = []
        for manager in self._cost_manager_list:
            names.extend(manager.get_cost_component_names())
        return names

    def get_all_cost_components(self) -> Dict[str, BaseCost]:
        cost_components = {}
        for manager in self._cost_manager_list:
            cost_components.update(manager.get_cost_components())
        return cost_components

    def get_cost_component_by_name(self, name: str) -> List[BaseCost]:
        cost_terms = []
        for manager in self._cost_manager_list:
            if manager.has_cost(name):
                cost_terms.append(manager.get_cost(name))
        return cost_terms

    # ------------------------------------------------------------------ #
    #  Internal: optimization path                                         #
    # ------------------------------------------------------------------ #

    def _compute_state_from_action_impl(self, act_seq: torch.Tensor) -> RobotState:
        return self.transition_model.forward(
            self.start_state,
            act_seq,
            (
                self._num_particles_goal.idxs_current_js
                if self._num_particles_goal is not None
                else None
            ),
            (
                self._num_particles_goal.seed_goal_js
                if self._num_particles_goal is not None
                else None
            ),
            (
                self._num_particles_goal.idxs_seed_goal_js
                if self._num_particles_goal is not None
                else None
            ),
            (
                self._num_particles_goal.seed_enable_implicit_goal_js
                if self._num_particles_goal is not None
                else None
            ),
            idxs_env=(
                self._num_particles_goal.idxs_env if self._num_particles_goal is not None else None
            ),
        )

    def _compute_costs_and_constraints_impl(
        self, state: RobotState, **kwargs
    ) -> CostsAndConstraints:
        costs_and_constraints = CostsAndConstraints()
        goal = self._num_particles_goal
        if self.config.cost_cfg is not None:
            cost_collection = self.cost_manager.compute_costs(state, goal=goal, **kwargs)
            costs_and_constraints.costs.merge(cost_collection)
        if self.config.constraint_cfg is not None:
            constraint_collection = self.constraint_manager.compute_costs(
                state, goal=goal, **kwargs
            )
            costs_and_constraints.constraints.merge(constraint_collection)
        if self.config.hybrid_cost_constraint_cfg is not None:
            hybrid_cost_collection = self.hybrid_cost_constraint_manager.compute_costs(
                state, goal=goal, **kwargs
            )
            costs_and_constraints.hybrid_costs_constraints.merge(hybrid_cost_collection)
        return costs_and_constraints

    # ------------------------------------------------------------------ #
    #  Internal: metrics path                                              #
    # ------------------------------------------------------------------ #

    def _compute_state_from_action_metrics_impl(
        self, act_seq: torch.Tensor, **kwargs
    ) -> RobotState:
        idxs_env = kwargs.get("idxs_env_query")
        if idxs_env is None:
            idxs_env = kwargs.get("env_query_idx")
        if idxs_env is None and self._metrics_goal is not None:
            idxs_env = self._metrics_goal.idxs_env
        state = self.metrics_transition_model.forward(
            self.start_state,
            act_seq,
            self._metrics_goal.idxs_current_js if self._metrics_goal is not None else None,
            self._metrics_goal.seed_goal_js if self._metrics_goal is not None else None,
            (self._metrics_goal.idxs_seed_goal_js if self._metrics_goal is not None else None),
            (
                self._metrics_goal.seed_enable_implicit_goal_js
                if self._metrics_goal is not None
                else None
            ),
            idxs_env=idxs_env,
        )
        return state

    def _compute_costs_and_constraints_metrics_impl(
        self, state: RobotState, **kwargs
    ) -> CostsAndConstraints:
        costs_and_constraints = CostsAndConstraints()
        goal = self._metrics_goal
        if self.config.cost_cfg is not None:
            cost_collection = self.metrics_cost_manager.compute_costs(state, goal=goal, **kwargs)
            costs_and_constraints.costs.merge(cost_collection)
        if self.config.constraint_cfg is not None:
            constraint_collection = self.metrics_constraint_manager.compute_costs(
                state, goal=goal, **kwargs
            )
            costs_and_constraints.constraints.merge(constraint_collection)
        if self.config.hybrid_cost_constraint_cfg is not None:
            hybrid_cost_collection = self.metrics_hybrid_cost_constraint_manager.compute_costs(
                state, goal=goal, **kwargs
            )
            costs_and_constraints.hybrid_costs_constraints.merge(hybrid_cost_collection)
        return costs_and_constraints

    def _compute_convergence_metrics_impl(self, state: RobotState, **kwargs) -> CostCollection:
        goal = self._metrics_goal
        convergence_collection = CostCollection()
        if self.config.convergence_cfg is not None:
            convergence_collection = self.metrics_convergence_manager.compute_convergence(
                state, goal=goal
            )
        return convergence_collection

    def _compute_metrics_from_state_impl(self, state: JointState, **kwargs) -> RolloutMetrics:
        costs_and_constraints = self._compute_costs_and_constraints_metrics_impl(state, **kwargs)
        convergence = self._compute_convergence_metrics_impl(state, **kwargs)
        return RolloutMetrics(
            costs_and_constraints=costs_and_constraints,
            feasible=costs_and_constraints.get_feasible(),
            state=state,
            convergence=convergence,
        )

    def _compute_metrics_from_action_impl(self, act_seq: torch.Tensor, **kwargs) -> RolloutMetrics:
        state = self._compute_state_from_action_metrics_impl(act_seq, **kwargs)
        metrics = self._compute_metrics_from_state_impl(state, **kwargs)
        metrics.actions = act_seq
        return metrics
