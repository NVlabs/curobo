# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Optional collision-inequality repair for graph-planner samples.

Projection is deliberately independent of ordinary RRT growth. A planner may
call the projector for difficult constrained problems without adding autograd
or CUDA Graph work to the normal sampling path. This module repairs samples
toward the constraints in the graph rollout. Those constraints can include a
task manifold when the rollout is given a corresponding runtime goal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import torch

from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.gradient.line_search_strategy import LineSearchType
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise


class _RRTProjectionRollout(Protocol):
    """Minimal generic rollout surface required by the projector."""

    @property
    def batch_size(self) -> Optional[int]: ...

    def compute_metrics_from_action(self, actions: torch.Tensor, **kwargs) -> Any: ...

    def update_batch_size(self, batch_size: int) -> None: ...

    def route_goal_poses(self, goal_query_idx: torch.Tensor) -> None: ...


class _RRTSampleProjectionResources(Protocol):
    """Narrow planner-resource surface required by :class:`RRTSampleProjector`."""

    action_dim: int
    action_bound_lows: torch.Tensor
    action_bound_highs: torch.Tensor
    auxiliary_rollout: _RRTProjectionRollout

    def check_samples_feasibility(
        self,
        samples: torch.Tensor,
        clone_output: bool = True,
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return exact Boolean feasibility for each environment- and goal-routed sample."""
        ...


@dataclass(frozen=True)
class RRTSampleProjectionConfig:
    """Immutable profile for a fresh fixed-shape cuRobo L-BFGS repair.

    The leading zero in ``line_search_scales`` represents retaining the input
    sample. Positive entries are the probes passed to L-BFGS.
    """

    iterations: int = 5
    projection_slots: int = 4
    history: int = 3
    line_search_scales: Tuple[float, ...] = (0.0, 0.3, 1.0)
    step_scale: float = 0.05
    initial_step_scale: float = 0.1
    lbfgs_epsilon: float = 0.01
    use_cuda_graph: bool = False

    def __post_init__(self) -> None:
        """Validate the fixed optimizer and capture profile."""

        def require_positive_integer(name: str, value: int) -> None:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                log_and_raise(f"{name} must be a positive integer")

        def require_positive_finite(name: str, value: float) -> None:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                log_and_raise(f"{name} must be finite and positive")

        require_positive_integer("iterations", self.iterations)
        require_positive_integer("projection_slots", self.projection_slots)
        require_positive_integer("history", self.history)
        for name, value in (
            ("step_scale", self.step_scale),
            ("initial_step_scale", self.initial_step_scale),
            ("lbfgs_epsilon", self.lbfgs_epsilon),
        ):
            require_positive_finite(name, value)
        if not isinstance(self.line_search_scales, tuple) or not self.line_search_scales:
            log_and_raise("line_search_scales must be a non-empty tuple")
        previous_scale = -1.0
        for scale in self.line_search_scales:
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not math.isfinite(scale)
                or scale < 0.0
            ):
                log_and_raise("line_search_scales must contain finite nonnegative values")
            if scale <= previous_scale:
                log_and_raise("line_search_scales must be strictly increasing")
            previous_scale = scale
        if self.line_search_scales[0] != 0.0:
            log_and_raise("line_search_scales must start at 0.0")
        if len(self.line_search_scales) < 2:
            log_and_raise("line_search_scales must include at least one positive probe")
        if not isinstance(self.use_cuda_graph, bool):
            log_and_raise("use_cuda_graph must be a boolean")


@dataclass
class RRTSampleProjectionResult:
    """Repaired samples and before/after feasibility diagnostics.

    ``initial_violation``, ``final_violation``, and ``displacement`` are
    meaningful for selected slots where ``violation_measured`` is true.
    Unselected samples retain zero diagnostic placeholders.
    """

    samples: torch.Tensor
    feasible: torch.Tensor
    initially_feasible: torch.Tensor
    initial_violation: torch.Tensor
    final_violation: torch.Tensor
    violation_measured: torch.Tensor
    displacement: torch.Tensor

    @property
    def repaired_count(self) -> int:
        """Count initially infeasible samples repaired to feasibility."""
        return int(torch.count_nonzero(~self.initially_feasible & self.feasible).item())


def _strict_constraint_residuals(metrics) -> torch.Tensor:
    """Flatten only strict inequality constraints into positive residuals."""
    values = metrics.costs_and_constraints.constraints.values.copy()
    if not values:
        # Preserve an autograd path for an unconstrained rollout.
        return (
            metrics.actions.reshape(metrics.actions.shape[0], -1).sum(dim=-1, keepdim=True) * 0.0
        )
    return torch.cat(
        [torch.clamp(value, min=0.0).reshape(value.shape[0], -1) for value in values],
        dim=-1,
    )


class _ConstraintOnlyRRTProjectionRollout:
    """Adapt graph metrics to the scalar Rollout objective expected by L-BFGS."""

    def __init__(
        self,
        resources: _RRTSampleProjectionResources,
        num_problems: int,
        num_particles: int,
        device_cfg: DeviceCfg,
        route_environments: bool,
        route_goals: bool,
    ) -> None:
        self.resources = resources
        self.base_rollout = resources.auxiliary_rollout
        self.num_problems = num_problems
        self.num_particles = num_particles
        self.device_cfg = device_cfg
        self.route_environments = route_environments
        self.route_goals = route_goals
        self.sum_horizon = True
        self._batch_size: Optional[int] = None
        self._env_query_idx = (
            torch.zeros(num_problems, device=device_cfg.device, dtype=torch.int32)
            if route_environments
            else None
        )
        self._goal_query_idx = (
            torch.zeros(num_problems, device=device_cfg.device, dtype=torch.int32)
            if route_goals
            else None
        )

    @property
    def action_dim(self) -> int:
        return self.resources.action_dim

    @property
    def action_horizon(self) -> int:
        return 1

    @property
    def horizon(self) -> int:
        return 1

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self.resources.action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self.resources.action_bound_highs

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def expected_batch_size(self) -> int:
        return self.num_problems * self.num_particles

    def set_routes(
        self,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> None:
        """Copy per-problem environment and goal IDs into stable storage."""
        if self.route_environments:
            if self._env_query_idx is None:
                log_and_raise("captured projection requires env_query_idx")
            self._env_query_idx.copy_(env_query_idx)
        if self.route_goals:
            if self._goal_query_idx is None:
                log_and_raise("captured projection requires goal_query_idx")
            self._goal_query_idx.copy_(goal_query_idx)

    def clamp_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply the graph planner's joint-position action bounds."""
        return torch.clamp(actions, self.action_bound_lows, self.action_bound_highs)

    def _expanded_env_query_idx(self) -> Optional[torch.Tensor]:
        if self._env_query_idx is None:
            return None
        return self._env_query_idx.repeat_interleave(self.num_particles)

    def _expanded_goal_query_idx(self) -> Optional[torch.Tensor]:
        if self._goal_query_idx is None:
            return None
        return self._goal_query_idx.repeat_interleave(self.num_particles)

    def _rollout_metrics(self, actions: torch.Tensor, clamp_actions: bool = True):
        rollout_actions = self.clamp_actions(actions) if clamp_actions else actions
        kwargs = {"use_cuda_graph": False}
        expanded_env = self._expanded_env_query_idx()
        if expanded_env is not None:
            kwargs["idxs_env_query"] = expanded_env
        expanded_goal = self._expanded_goal_query_idx()
        if expanded_goal is not None:
            self.base_rollout.route_goal_poses(expanded_goal.view(-1, 1))
        return self.base_rollout.compute_metrics_from_action(rollout_actions, **kwargs)

    def evaluate_action(self, act_seq: torch.Tensor, **kwargs: Any) -> RolloutResult:
        """Return strict positive constraint merit as the sole scalar cost."""
        if act_seq.shape != (self.expected_batch_size, 1, self.action_dim):
            log_and_raise(
                "projection rollout actions must have shape "
                f"({self.expected_batch_size}, 1, {self.action_dim})"
            )
        metrics = self._rollout_metrics(act_seq)
        residuals = _strict_constraint_residuals(metrics)
        merit = residuals.sum(dim=-1).view(self.expected_batch_size, 1, 1)
        costs = CostCollection(values=[merit], names=["projection_constraint_merit"])
        return RolloutResult(
            actions=act_seq,
            state=getattr(metrics, "state", None),
            costs_and_constraints=CostsAndConstraints(costs=costs),
        )

    def evaluate_problem_residuals(
        self,
        actions: torch.Tensor,
        clamp_actions: bool = True,
    ) -> torch.Tensor:
        """Evaluate B problems while preserving the optimizer's B*C rollout shape."""
        expanded_actions = (
            actions.unsqueeze(1)
            .expand(self.num_problems, self.num_particles, self.action_dim)
            .reshape(self.expected_batch_size, 1, self.action_dim)
        )
        metrics = self._rollout_metrics(expanded_actions, clamp_actions=clamp_actions)
        residuals = _strict_constraint_residuals(metrics)
        return residuals.view(self.num_problems, self.num_particles, -1)[:, 0]

    def compute_metrics_from_action(self, act_seq: torch.Tensor, **kwargs):
        return self._rollout_metrics(act_seq)

    def update_batch_size(self, batch_size: int) -> None:
        if batch_size != self.expected_batch_size:
            log_and_raise(
                "projection rollout batch must remain "
                f"{self.expected_batch_size}, got {batch_size}"
            )
        self._batch_size = batch_size
        self.base_rollout.update_batch_size(batch_size)

    def update_params(self, **kwargs) -> bool:
        return True

    def update_dt(self, dt, **kwargs) -> bool:
        return True

    def reset(self, reset_problem_ids=None, **kwargs) -> bool:
        return True

    def reset_shape(self) -> bool:
        return True

    def reset_seed(self) -> None:
        return None


class _ConstraintOnlyRRTProjectionBaseRollout:
    """Evaluate one baseline action through particle-sized rollout storage."""

    def __init__(self, particle_rollout: _ConstraintOnlyRRTProjectionRollout) -> None:
        self.particle_rollout = particle_rollout
        self.num_problems = particle_rollout.num_problems
        self.sum_horizon = True
        self._batch_size: Optional[int] = None

    @property
    def action_dim(self) -> int:
        return self.particle_rollout.action_dim

    @property
    def action_horizon(self) -> int:
        return 1

    @property
    def horizon(self) -> int:
        return 1

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self.particle_rollout.action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self.particle_rollout.action_bound_highs

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    def evaluate_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutResult:
        """Return one merit per problem without resizing the shared rollout."""
        expected_shape = (self.num_problems, 1, self.action_dim)
        if act_seq.shape != expected_shape:
            log_and_raise(f"projection base actions must have shape {expected_shape}")
        expanded_actions = (
            act_seq.unsqueeze(1)
            .expand(
                self.num_problems,
                self.particle_rollout.num_particles,
                1,
                self.action_dim,
            )
            .reshape(-1, 1, self.action_dim)
        )
        metrics = self.particle_rollout._rollout_metrics(expanded_actions)
        residuals = _strict_constraint_residuals(metrics).view(
            self.num_problems,
            self.particle_rollout.num_particles,
            -1,
        )[:, 0]
        merit = residuals.sum(dim=-1).view(self.num_problems, 1, 1)
        costs = CostCollection(values=[merit], names=["projection_constraint_merit"])
        return RolloutResult(
            actions=act_seq,
            state=None,
            costs_and_constraints=CostsAndConstraints(costs=costs),
        )

    def compute_metrics_from_action(
        self,
        act_seq: torch.Tensor,
        **kwargs: Any,
    ) -> RolloutResult:
        return self.evaluate_action(act_seq, **kwargs)

    def update_batch_size(self, batch_size: int) -> None:
        if batch_size != self.num_problems:
            log_and_raise(
                f"projection base rollout batch must remain {self.num_problems}, got {batch_size}"
            )
        self._batch_size = batch_size

    def update_params(self, **kwargs: Any) -> bool:
        return True

    def update_dt(self, dt: Any, **kwargs: Any) -> bool:
        return True


class RRTSampleProjector:
    """Repair colliding samples with a fresh bounded cuRobo L-BFGS solve.

    The scalar objective contains only the rollout's strict ``constraints``
    collection, matching graph-planner feasibility. Regular and hybrid costs
    are intentionally excluded. L-BFGS provides a matrix-free inverse-Hessian
    approximation from scalar reverse-mode gradients, while joint bounds are
    applied by the adapter. There is no projection-specific trust region.
    """

    def __init__(
        self,
        resources: _RRTSampleProjectionResources,
        config: Optional[RRTSampleProjectionConfig] = None,
    ) -> None:
        """Initialize a lazy fixed-shape optimizer from planner resources."""
        self.resources = resources
        self.config = config or RRTSampleProjectionConfig()
        self._optimizer: Optional[LBFGSOpt] = None
        self._optimizer_rollout: Optional[_ConstraintOnlyRRTProjectionRollout] = None
        self._optimizer_base_rollout: Optional[_ConstraintOnlyRRTProjectionBaseRollout] = None
        self._optimizer_signature: Optional[
            Tuple[
                torch.Size,
                torch.dtype,
                torch.device,
                bool,
                bool,
            ]
        ] = None
        self._cuda_graph_executor: Optional[GraphExecutor] = None
        self._cuda_graph_signature: Optional[
            Tuple[
                torch.Size,
                torch.dtype,
                torch.device,
                Optional[Tuple[torch.Size, torch.dtype, torch.device]],
                Optional[Tuple[torch.Size, torch.dtype, torch.device]],
            ]
        ] = None

    def _build_optimizer(
        self,
        samples: torch.Tensor,
        route_environments: bool,
        route_goals: bool,
    ) -> None:
        device_cfg = DeviceCfg(device=samples.device, dtype=samples.dtype)
        num_problems = samples.shape[0]
        optimizer_line_search_scales = self.config.line_search_scales[1:]
        num_particles = len(optimizer_line_search_scales)
        adapter = _ConstraintOnlyRRTProjectionRollout(
            self.resources,
            num_problems=num_problems,
            num_particles=num_particles,
            device_cfg=device_cfg,
            route_environments=route_environments,
            route_goals=route_goals,
        )
        base_adapter = _ConstraintOnlyRRTProjectionBaseRollout(adapter)
        use_cuda_kernels = samples.is_cuda and samples.dtype == torch.float32
        optimizer_config = LBFGSOptCfg(
            num_iters=self.config.iterations,
            inner_iters=self.config.iterations,
            num_problems=num_problems,
            num_particles=num_particles,
            history=self.config.history,
            line_search_scale=list(optimizer_line_search_scales),
            line_search_type=LineSearchType.APPROX_WOLFE,
            step_scale=self.config.step_scale,
            initial_step_scale=self.config.initial_step_scale,
            epsilon=self.config.lbfgs_epsilon,
            fixed_iters=True,
            return_best_action=True,
            device_cfg=device_cfg,
            use_cuda_kernel_line_search=use_cuda_kernels,
        )
        self._optimizer_rollout = adapter
        self._optimizer_base_rollout = base_adapter
        self._optimizer = LBFGSOpt(
            optimizer_config,
            [adapter, base_adapter],
            use_cuda_graph=False,
        )
        self._optimizer_signature = (
            samples.shape,
            samples.dtype,
            samples.device,
            route_environments,
            route_goals,
        )

    def _ensure_optimizer(
        self,
        samples: torch.Tensor,
        env_query_idx: Optional[torch.Tensor],
        goal_query_idx: Optional[torch.Tensor],
    ) -> None:
        route_environments = env_query_idx is not None
        route_goals = goal_query_idx is not None
        signature = (
            samples.shape,
            samples.dtype,
            samples.device,
            route_environments,
            route_goals,
        )
        if self._optimizer is None or self._optimizer_signature != signature:
            self._reset_cuda_graph_executor()
            self._optimizer = None
            self._optimizer_rollout = None
            self._optimizer_base_rollout = None
            self._build_optimizer(samples, route_environments, route_goals)

    def _project_optimization_tensors(
        self,
        samples: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Run one fresh fixed-iteration solve without nested graph replay."""
        if (
            self._optimizer is None
            or self._optimizer_rollout is None
            or self._optimizer_base_rollout is None
        ):
            log_and_raise("projection optimizer must be initialized before use")
        self._optimizer_rollout.set_routes(env_query_idx, goal_query_idx)
        initial_residuals = self._optimizer_rollout.evaluate_problem_residuals(
            samples,
            clamp_actions=False,
        )
        bounded_seed = self._optimizer_rollout.clamp_actions(samples).unsqueeze(1)
        self._optimizer.reinitialize(bounded_seed, clear_optimizer_state=True)
        optimized = self._optimizer.optimize_without_timing(bounded_seed)
        projected = self._optimizer_rollout.clamp_actions(optimized[:, 0]).detach()
        final_residuals = self._optimizer_rollout.evaluate_problem_residuals(projected)
        return (
            projected,
            initial_residuals.sum(dim=-1),
            final_residuals.sum(dim=-1),
        )

    def _validate_samples(self, samples: torch.Tensor) -> None:
        if (
            samples.ndim != 2
            or samples.shape[0] == 0
            or samples.shape[1] != self.resources.action_dim
        ):
            log_and_raise("samples must have shape (batch, action_dim)")
        if not torch.is_floating_point(samples):
            log_and_raise("samples must use a floating-point dtype")
        if samples.device != self.resources.action_bound_lows.device:
            log_and_raise("samples and projection resources must share a device")
        if samples.dtype != self.resources.action_bound_lows.dtype:
            log_and_raise("samples and projection resources must share a dtype")

    def _validate_env_query_idx(
        self,
        samples: torch.Tensor,
        env_query_idx: Optional[torch.Tensor],
    ) -> None:
        if env_query_idx is None:
            return
        if (
            env_query_idx.shape != (samples.shape[0],)
            or env_query_idx.dtype != torch.int32
            or env_query_idx.device != samples.device
        ):
            log_and_raise("env_query_idx must be int32 on samples.device with shape (batch,)")
        explicit_capacity = getattr(self.resources, "environment_capacity", None)
        if callable(explicit_capacity):
            explicit_capacity = explicit_capacity()
        capacities = [] if explicit_capacity is None else [int(explicit_capacity)]
        kinematics = getattr(self.resources, "kinematics", None)
        kinematics_config = getattr(getattr(kinematics, "config", None), "kinematics_config", None)
        robot_capacity = getattr(kinematics_config, "num_envs", None)
        if robot_capacity is not None:
            capacities.append(int(robot_capacity))
        scene_capacity = getattr(
            getattr(self.resources, "scene_collision_checker", None),
            "num_envs",
            None,
        )
        if scene_capacity is not None:
            capacities.append(int(scene_capacity))
        if not capacities or any(capacity <= 0 for capacity in capacities):
            log_and_raise(
                "projection resources must expose a positive environment capacity "
                "when env_query_idx is provided"
            )
        environment_capacity = min(capacities)
        env_bounds = torch.aminmax(env_query_idx)
        if int(env_bounds[0].item()) < 0 or int(env_bounds[1].item()) >= environment_capacity:
            log_and_raise(f"env_query_idx values must be in [0, {environment_capacity})")

    def _validate_goal_query_idx(
        self,
        samples: torch.Tensor,
        goal_query_idx: Optional[torch.Tensor],
    ) -> None:
        if goal_query_idx is None:
            return
        if (
            goal_query_idx.shape != (samples.shape[0],)
            or goal_query_idx.dtype != torch.int32
            or goal_query_idx.device != samples.device
        ):
            log_and_raise("goal_query_idx must be int32 on samples.device with shape (batch,)")

    def _check_feasibility(
        self,
        samples: torch.Tensor,
        env_query_idx: Optional[torch.Tensor],
        goal_query_idx: Optional[torch.Tensor],
        *,
        clone_output: Optional[bool] = None,
    ) -> torch.Tensor:
        """Call legacy or goal-aware resources without passing empty routes."""
        route_kwargs = {}
        if clone_output is not None:
            route_kwargs["clone_output"] = clone_output
        if env_query_idx is not None:
            route_kwargs["env_query_idx"] = env_query_idx
        if goal_query_idx is not None:
            route_kwargs["goal_query_idx"] = goal_query_idx
        return self.resources.check_samples_feasibility(samples, **route_kwargs)

    def _select_projection_slots(
        self,
        samples: torch.Tensor,
        initially_feasible: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a fixed number of invalid samples and mask padding slots."""
        slot_count = min(self.config.projection_slots, samples.shape[0])
        invalid = ~initially_feasible
        if eligible_mask is not None:
            invalid = invalid & eligible_mask
        invalid_scores = invalid.to(dtype=torch.int64)
        slot_scores, slot_indices = torch.topk(
            invalid_scores,
            k=slot_count,
            largest=True,
            sorted=True,
        )
        return samples[slot_indices], slot_indices, slot_scores.to(dtype=torch.bool)

    def _merge_slot_values(
        self,
        base: torch.Tensor,
        slot_indices: torch.Tensor,
        slot_values: torch.Tensor,
        slot_active: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter active slot values while preserving unselected samples."""
        active_shape = (slot_active.shape[0],) + (1,) * (slot_values.ndim - 1)
        merged_slot_values = torch.where(
            slot_active.view(active_shape),
            slot_values,
            base[slot_indices],
        )
        return base.index_copy(0, slot_indices, merged_slot_values)

    def _reset_cuda_graph_executor(self) -> None:
        if self._cuda_graph_executor is not None:
            self._cuda_graph_executor.reset()
        self._cuda_graph_executor = None
        self._cuda_graph_signature = None

    def _get_cuda_graph_executor(
        self,
        samples: torch.Tensor,
        env_query_idx: Optional[torch.Tensor],
        goal_query_idx: Optional[torch.Tensor],
    ) -> GraphExecutor:
        self._ensure_optimizer(samples, env_query_idx, goal_query_idx)
        env_signature = (
            None
            if env_query_idx is None
            else (env_query_idx.shape, env_query_idx.dtype, env_query_idx.device)
        )
        goal_signature = (
            None
            if goal_query_idx is None
            else (goal_query_idx.shape, goal_query_idx.dtype, goal_query_idx.device)
        )
        signature = (
            samples.shape,
            samples.dtype,
            samples.device,
            env_signature,
            goal_signature,
        )
        expected_rollout_batch = samples.shape[0] * (len(self.config.line_search_scales) - 1)
        rollout_batch_size = self.resources.auxiliary_rollout.batch_size
        rollout_was_rebatched = (
            self._cuda_graph_executor is not None
            and rollout_batch_size is not None
            and rollout_batch_size != expected_rollout_batch
        )
        if (
            self._cuda_graph_executor is None
            or self._cuda_graph_signature != signature
            or rollout_was_rebatched
        ):
            self._reset_cuda_graph_executor()
            self._cuda_graph_executor = create_graph_executor(
                capture_fn=self._project_optimization_tensors,
                device=samples.device,
                use_cuda_graph=True,
                clone_outputs=True,
            )
            self._cuda_graph_signature = signature
        return self._cuda_graph_executor

    def warmup_cuda_graph(
        self,
        samples: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Capture and prime the fixed-shape L-BFGS solve before latency-sensitive use."""
        self._validate_samples(samples)
        self._validate_env_query_idx(samples, env_query_idx)
        self._validate_goal_query_idx(samples, goal_query_idx)
        if not samples.is_cuda:
            log_and_raise("CUDA graph projection requires CUDA samples")
        slot_count = min(self.config.projection_slots, samples.shape[0])
        projection_samples = samples[:slot_count]
        projection_env = None if env_query_idx is None else env_query_idx[:slot_count]
        projection_goal = None if goal_query_idx is None else goal_query_idx[:slot_count]
        # Prime the exact-feasibility executor used before and after every
        # repair. Environment-routed feasibility owns a separate CUDA graph.
        self._check_feasibility(
            projection_samples,
            projection_env,
            projection_goal,
            clone_output=False,
        )
        executor = self._get_cuda_graph_executor(
            projection_samples,
            projection_env,
            projection_goal,
        )
        resolved_env = (
            torch.zeros(slot_count, device=samples.device, dtype=torch.int32)
            if projection_env is None
            else projection_env
        )
        resolved_goal = (
            torch.zeros(slot_count, device=samples.device, dtype=torch.int32)
            if projection_goal is None
            else projection_goal
        )
        executor.warmup(projection_samples, resolved_env, resolved_goal)
        # CUDA can defer one-time graph-executable upload/initialization until
        # the first replay. Prime that replay here instead of charging it to
        # the first planning query that encounters an invalid target.
        executor.replay(clone_outputs=False)
        torch.cuda.synchronize(samples.device)

    def reset_cuda_graph(self) -> None:
        """Release capture and fixed-shape optimizer state."""
        self._reset_cuda_graph_executor()
        self._optimizer = None
        self._optimizer_rollout = None
        self._optimizer_base_rollout = None
        self._optimizer_signature = None

    def project(
        self,
        samples: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
        eligible_mask: Optional[torch.Tensor] = None,
    ) -> RRTSampleProjectionResult:
        """Repair eligible invalid samples toward their collision-free sets."""
        self._validate_samples(samples)
        self._validate_env_query_idx(samples, env_query_idx)
        self._validate_goal_query_idx(samples, goal_query_idx)
        if eligible_mask is not None and (
            eligible_mask.shape != (samples.shape[0],)
            or eligible_mask.dtype != torch.bool
            or eligible_mask.device != samples.device
        ):
            log_and_raise("eligible_mask must be bool on samples.device with shape (batch,)")
        if self.config.use_cuda_graph and not samples.is_cuda:
            log_and_raise("CUDA graph projection requires CUDA samples")
        initially_feasible = self._check_feasibility(
            samples,
            env_query_idx,
            goal_query_idx,
        ).reshape(-1)
        projection_samples, slot_indices, slot_active = self._select_projection_slots(
            samples,
            initially_feasible,
            eligible_mask,
        )
        projection_env = None if env_query_idx is None else env_query_idx[slot_indices]
        projection_goal = None if goal_query_idx is None else goal_query_idx[slot_indices]

        initial_violation = torch.zeros(
            samples.shape[0], device=samples.device, dtype=samples.dtype
        )
        final_violation = torch.zeros_like(initial_violation)
        displacement = torch.zeros_like(initial_violation)
        violation_measured = torch.zeros_like(initially_feasible)

        if not bool(slot_active.any().item()):
            return RRTSampleProjectionResult(
                samples.clone(),
                initially_feasible.clone(),
                initially_feasible,
                initial_violation,
                final_violation,
                violation_measured,
                displacement,
            )

        if self.config.use_cuda_graph:
            executor = self._get_cuda_graph_executor(
                projection_samples,
                projection_env,
                projection_goal,
            )
            resolved_env = (
                torch.zeros(
                    projection_samples.shape[0],
                    device=samples.device,
                    dtype=torch.int32,
                )
                if projection_env is None
                else projection_env
            )
            resolved_goal = (
                torch.zeros_like(resolved_env) if projection_goal is None else projection_goal
            )
            optimization_output = executor(
                projection_samples,
                resolved_env,
                resolved_goal,
            )
        else:
            self._ensure_optimizer(projection_samples, projection_env, projection_goal)
            resolved_env = (
                torch.zeros(
                    projection_samples.shape[0],
                    device=samples.device,
                    dtype=torch.int32,
                )
                if projection_env is None
                else projection_env
            )
            resolved_goal = (
                torch.zeros_like(resolved_env) if projection_goal is None else projection_goal
            )
            optimization_output = self._project_optimization_tensors(
                projection_samples,
                resolved_env,
                resolved_goal,
            )

        projected = self._merge_slot_values(
            samples.clone(),
            slot_indices,
            optimization_output[0],
            slot_active,
        )
        initial_violation = self._merge_slot_values(
            initial_violation,
            slot_indices,
            optimization_output[1],
            slot_active,
        )
        final_violation = self._merge_slot_values(
            final_violation,
            slot_indices,
            optimization_output[2],
            slot_active,
        )
        violation_measured = self._merge_slot_values(
            violation_measured,
            slot_indices,
            torch.ones_like(slot_active),
            slot_active,
        )
        slot_displacement = torch.linalg.vector_norm(
            optimization_output[0] - projection_samples,
            dim=-1,
        )
        displacement = self._merge_slot_values(
            displacement,
            slot_indices,
            slot_displacement,
            slot_active,
        )
        active_slot_indices = slot_indices[slot_active]
        active_projection_env = None if projection_env is None else projection_env[slot_active]
        active_projection_goal = None if projection_goal is None else projection_goal[slot_active]
        active_slot_feasible = self._check_feasibility(
            optimization_output[0][slot_active],
            active_projection_env,
            active_projection_goal,
        ).reshape(-1)
        feasible = initially_feasible.index_copy(
            0,
            active_slot_indices,
            active_slot_feasible,
        )
        return RRTSampleProjectionResult(
            projected,
            feasible,
            initially_feasible,
            initial_violation,
            final_violation,
            violation_measured,
            displacement,
        )
