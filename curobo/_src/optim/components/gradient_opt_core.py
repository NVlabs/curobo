# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared infrastructure for gradient-based line search optimizers.

Manages: rollouts, line search, best tracking, debug recording, iteration state,
cost/gradient evaluation, CUDA graph executors, and all lifecycle methods.

The optimizer provides a step_direction_fn and optional hooks for buffer management.
Composed by LBFGSOpt, LSR1Opt, and ConjugateGradientOpt.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.autograd.profiler as profiler

import curobo._src.runtime as curobo_runtime
from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.best_tracker import BestTracker
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.gradient.line_search_strategy import LineSearchStrategyFactory
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.tensor_util import check_nan_last_dimension


class GradientOptCore:
    """Shared gradient optimizer infrastructure.

    Encapsulates everything except the step direction computation.
    The owning optimizer provides:
      - step_direction_fn(iteration_state) -> Tensor
      - on_reinitialize(mask): clear optimizer-specific buffers
      - on_initial_state(iteration_state, mask): set reference points
      - on_resize(num_problems): resize optimizer-specific buffers
      - on_shift(shift_steps): shift optimizer-specific buffers
    """

    _graphable_methods: set = {"_opt_iters", "_prepare_initial_iteration_state"}

    def __init__(
        self,
        config,
        rollout_list: List[Rollout],
        step_direction_fn: Callable,
        *,
        on_reinitialize: Optional[Callable] = None,
        on_initial_state: Optional[Callable] = None,
        on_resize: Optional[Callable] = None,
        on_shift: Optional[Callable] = None,
        use_cuda_graph: bool = False,
    ):
        """Initialize gradient optimizer core with rollouts, callbacks, and config.

        Sets up rollout instances, action bounds, best-solution tracker, debug
        recorder, and backward-pass vector. Line-search strategies and CUDA graph
        executors are created lazily: the owning optimizer must call
        :meth:`update_num_problems` then :meth:`finish_init` after assigning
        ``self._core``.

        Args:
            config: Gradient optimizer configuration (device, iterations,
                line-search parameters, convergence thresholds, etc.).
            rollout_list: Rollout instances; ``[0]`` is the main rollout,
                ``[1]`` (if present) is used for the initial evaluation.
                Length must equal ``config.num_rollout_instances``.
            step_direction_fn: ``fn(iteration_state) -> Tensor`` that computes
                the quasi-Newton or conjugate-gradient step direction.
            on_reinitialize: Optional callback ``fn(mask)`` to clear
                optimizer-specific buffers (e.g. L-BFGS history).
            on_initial_state: Optional callback
                ``fn(iteration_state, mask)`` to set reference points
                (e.g. initial Hessian approximation).
            on_resize: Optional callback ``fn(num_problems)`` to resize
                optimizer-specific buffers when the batch size changes.
            on_shift: Optional callback ``fn(shift_steps)`` to shift
                optimizer-specific buffers for MPC warm-starting.
            use_cuda_graph: If True, CUDA graph executors are created for
                ``_opt_iters`` and ``_prepare_initial_iteration_state``
                after :meth:`finish_init` is called.
        """
        if len(rollout_list) != config.num_rollout_instances:
            log_and_raise(
                f"num_rollout_instances {config.num_rollout_instances} "
                f"!= len(rollout_list) {len(rollout_list)}"
            )

        self.config = config
        self.device_cfg = config.device_cfg
        self.opt_dt = 0.0
        self.use_cuda_graph = use_cuda_graph
        self._enabled = True
        self._iteration_state: Optional[OptimizationIterationState] = None
        self._og_num_iters = config.num_iters

        # Callbacks
        self._step_direction_fn = step_direction_fn
        self._on_reinitialize = on_reinitialize
        self._on_initial_state = on_initial_state
        self._on_resize = on_resize
        self._on_shift = on_shift

        # Rollout instances: [0] = main, [1] = initial evaluation
        self.rollout_fn = rollout_list[0]
        self._rollout_list = rollout_list
        self.rollout_fn.sum_horizon = True

        # Components
        self._bounds = ActionBounds(
            self.rollout_fn.action_bound_lows,
            self.rollout_fn.action_bound_highs,
            self.action_horizon,
            config.step_scale,
        )
        self._best = BestTracker(config.device_cfg)
        self._debug = DebugRecorder() if config.store_debug else None

        # Check CUDA kernel feasibility for line search
        if self.opt_dim >= 1024:
            self.config.use_cuda_kernel_line_search = False
        if len(self.config.line_search_scale) > self.opt_dim:
            self.config.use_cuda_kernel_line_search = False

        # Line search: created lazily in update_num_problems
        self._line_search_strategy = None
        self._line_search_context = None
        self._initial_line_search_strategy = None
        self._initial_line_search_context = None

        # Gradient backward vector
        self._l_vec = None

        # CUDA graph executors
        self._executors: Dict[str, Optional[GraphExecutor]] = {}

        # NOTE: update_num_problems is NOT called here. The owning optimizer
        # must call it after self._core is assigned so hooks can access the core.

        # Create CUDA graph executors for standalone use (after update_num_problems)
        self._deferred_cuda_graph = use_cuda_graph

    def finish_init(self):
        """Call after the owning optimizer has assigned self._core and called update_num_problems."""
        if self._deferred_cuda_graph:
            self._executors["_opt_iters"] = create_graph_executor(
                capture_fn=self._opt_iters,
                device=self.device_cfg.device,
                use_cuda_graph=True,
            )
            self._executors["_prepare_initial_iteration_state"] = create_graph_executor(
                capture_fn=self._prepare_initial_iteration_state,
                device=self.device_cfg.device,
                use_cuda_graph=True,
            )

    # -- Properties --

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def action_horizon(self) -> int:
        return self.rollout_fn.action_horizon

    @property
    def action_dim(self) -> int:
        return self.rollout_fn.action_dim

    @property
    def opt_dim(self) -> int:
        """Return the total optimization dimensionality (action_horizon * action_dim)."""
        return self.action_horizon * self.action_dim

    @property
    def outer_iters(self) -> int:
        """Return the number of outer iterations (num_iters / inner_iters)."""
        return self.config.outer_iters

    @property
    def horizon(self):
        return self.rollout_fn.horizon

    @property
    def action_bound_lows(self):
        return self.rollout_fn.action_bound_lows

    @property
    def action_bound_highs(self):
        return self.rollout_fn.action_bound_highs

    @property
    def action_step_max(self):
        return self._bounds.step_max

    @property
    def action_horizon_step_max(self):
        """Return per-timestep maximum step size, refreshing bounds first.

        Returns:
            Tensor: Shape ``(action_horizon, action_dim)`` with the maximum
            allowed step magnitude at each timestep.
        """
        self._bounds.refresh(
            self.action_bound_lows, self.action_bound_highs, self.action_horizon
        )
        return self._bounds.horizon_step_max

    @property
    def action_horizon_bounds_lows(self):
        self._bounds.refresh(
            self.action_bound_lows, self.action_bound_highs, self.action_horizon
        )
        return self._bounds.horizon_lows

    @property
    def action_horizon_bounds_highs(self):
        self._bounds.refresh(
            self.action_bound_lows, self.action_bound_highs, self.action_horizon
        )
        return self._bounds.horizon_highs

    @property
    def solve_time(self) -> float:
        """Return wall-clock time in seconds for the most recent optimize call."""
        return self.opt_dt

    @property
    def solver_names(self):
        return [self.config.solver_name]

    # -- Dispatch --

    def _dispatch(self, method_name: str, *args):
        executor = self._executors.get(method_name)
        if executor is not None and self.use_cuda_graph:
            return executor(*args)
        return getattr(self, method_name)(*args)

    # -- Core algorithm --

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Run the full optimization loop and return the best action sequence.

        Delegates to :meth:`_optimize_impl` and records wall-clock solve time
        via a CUDA event timer.

        Args:
            seed_action: Initial action tensor of shape
                ``(num_problems, action_horizon, action_dim)`` or a flat
                view ``(num_problems, action_horizon * action_dim)``.

        Returns:
            Tensor of shape ``(num_problems, action_horizon, action_dim)``
            containing the best (or last non-NaN) action found.
        """
        timer = CudaEventTimer().start()
        out = self._optimize_impl(seed_action)
        self.opt_dt = timer.stop()
        return out

    def _optimize_impl(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Execute the outer optimization loop over line-search iterations.

        Prepares the initial iteration state (from a cached reinitialize state
        or by evaluating the seed), runs ``outer_iters`` rounds of
        ``_opt_iters``, and selects the result. When
        ``config.return_best_action`` is False the last iterate is returned
        unless it contains NaN, in which case the best-tracked action is used.

        Args:
            seed_action: Seed action tensor of shape
                ``(num_problems, action_horizon * action_dim)`` or
                ``(num_problems, action_horizon, action_dim)``.

        Returns:
            Tensor of shape ``(num_problems, action_horizon, action_dim)``.
        """
        q = seed_action.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )

        if self._iteration_state is not None:
            iteration_state = self._iteration_state
            self._iteration_state = None
        else:
            iteration_state = self._dispatch(
                "_prepare_initial_iteration_state", q
            )

        self._record_iteration_state(iteration_state)

        for _ in range(self.outer_iters):
            iteration_state = self._dispatch("_opt_iters", iteration_state)

            if not self.config.fixed_iters:
                if BestTracker.check_convergence(
                    iteration_state.converged,
                    self.config.converged_ratio,
                ):
                    break

        q = iteration_state.action.view(
            self.config.num_problems, self.action_horizon * self.action_dim
        )
        best_q = iteration_state.best_action.view(
            self.config.num_problems, self.action_horizon * self.action_dim
        )

        if not self.config.return_best_action:
            nan_labels = check_nan_last_dimension(q).view(
                self.config.num_problems, 1
            )
            q_best = torch.where(nan_labels, best_q, q)
        else:
            q_best = best_q.clone()
        return q_best.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )

    def _opt_iters(
        self, iteration_state: OptimizationIterationState
    ) -> OptimizationIterationState:
        """Run ``inner_iters`` consecutive optimization steps.

        This method is the CUDA-graphable inner loop. Each iteration calls
        :meth:`_opt_step` and records the state for debug tracing.

        Args:
            iteration_state: Current optimization state carrying action,
                gradient, cost, best-tracking, and step direction tensors.

        Returns:
            Updated :class:`OptimizationIterationState` after all inner
            iterations.
        """
        for iteration in range(self.config.inner_iters):
            if curobo_runtime.debug_nan:
                print("iteration: ", iteration)
            iteration_state = self._opt_step(iteration_state)
            self._record_iteration_state(iteration_state)
        return iteration_state

    @profiler.record_function("gradient_opt_core/opt_step")
    def _opt_step(
        self, iteration_state: OptimizationIterationState
    ) -> OptimizationIterationState:
        """Execute a single line-search step followed by a step-direction update.

        Performs the line search to find the next action and its gradient, then
        calls the owning optimizer's ``step_direction_fn`` to compute the next
        descent direction (e.g. L-BFGS two-loop recursion).

        Args:
            iteration_state: Current state with action, cost, gradient, and
                step direction tensors.

        Returns:
            Updated :class:`OptimizationIterationState` with the new action,
            cost, gradient, and step direction.
        """
        if curobo_runtime.debug_nan:
            if torch.isnan(iteration_state.action).any():
                log_and_raise("iteration_state.action is nan")
            if torch.isnan(iteration_state.step_direction).any():
                log_and_raise("iteration_state.step_direction is nan")

        next_state = self._line_search_strategy.search(
            iteration_state, context=self._line_search_context
        )

        if curobo_runtime.debug_nan:
            if torch.isnan(next_state.action).any():
                log_and_raise("next_state.action is nan")
            if torch.isnan(next_state.exploration_gradient).any():
                log_and_raise("next_state.gradient is nan")
            if torch.isnan(next_state.exploration_action).any():
                log_and_raise("next_state.exploration_action is nan")

        step_direction = self._step_direction_fn(next_state)

        if curobo_runtime.debug_nan:
            if torch.isnan(step_direction).any():
                log_and_raise("step_direction is nan")

        next_state.step_direction = step_direction
        return next_state

    def _prepare_initial_iteration_state(
        self, action: torch.Tensor
    ) -> OptimizationIterationState:
        iteration_state = self._create_initial_iteration_state(action)
        iteration_state = self._run_initial_evaluation(iteration_state)
        if self._on_initial_state:
            self._on_initial_state(iteration_state, None)
        return iteration_state

    def _create_initial_iteration_state(
        self, action: torch.Tensor
    ) -> OptimizationIterationState:
        grad_q = action.detach().clone() * 0.0
        return OptimizationIterationState(
            action=action.clone(),
            gradient=grad_q.clone(),
            cost=self._best.cost.clone(),
            best_cost=self._best.cost.clone(),
            best_action=self._best.action.clone(),
            best_iteration=self._best.iteration,
            current_iteration=self._best.current_iteration,
            step_direction=grad_q.clone(),
            exploration_action=action.clone(),
            exploration_gradient=grad_q.clone(),
            exploration_cost=self._best.cost.clone(),
            converged=self._best.converged,
        )

    def _run_initial_evaluation(
        self, iteration_state: OptimizationIterationState
    ) -> OptimizationIterationState:
        iteration_state = self._initial_line_search_strategy.search(
            iteration_state, context=self._initial_line_search_context
        )
        initial_step_direction = (
            -self.config.initial_step_scale * iteration_state.exploration_gradient
        )
        iteration_state.step_direction.copy_(initial_step_direction)
        return iteration_state

    # -- Cost/gradient evaluation --

    @profiler.record_function("gradient_opt_core/cost_and_gradient")
    def _compute_cost_constraint_and_gradient(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate rollout cost and compute its gradient via back-propagation.

        Runs the main rollout (``rollout_list[0]``), sums cost and constraint
        terms over the horizon, and back-propagates through the rollout graph
        to obtain the action gradient.

        Args:
            x: Action tensor of shape
                ``(num_problems, num_particles, action_horizon * action_dim)``.

        Returns:
            Tuple of:
                - cost: Shape ``(num_problems, num_particles, 1)``.
                - gradient: Same shape as ``x``.
        """
        x_n = x.detach().requires_grad_(True)
        x_in = x_n.view(
            self.config.num_problems * self.config.num_particles,
            self.action_horizon,
            self.action_dim,
        )
        trajectories = self.rollout_fn.evaluate_action(x_in, use_cuda_graph=False)
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(
            sum_horizon=True
        )
        if costs.shape != (self.config.num_problems * self.config.num_particles,):
            log_and_raise(
                f"costs.shape != (num_problems*num_particles,): {costs.shape}"
            )
        cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        cost.backward(gradient=self._l_vec, retain_graph=False)
        g_x = x_n.grad.detach()
        return cost, g_x

    @profiler.record_function("gradient_opt_core/cost_and_gradient_initial")
    def _compute_cost_constraint_and_gradient_initial(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_n = x.detach().requires_grad_(True)
        x_in = x_n.view(
            self.config.num_problems * self.config.num_particles,
            self.action_horizon,
            self.action_dim,
        )
        initial_rollout = self._rollout_list[1] if len(self._rollout_list) > 1 else self._rollout_list[0]
        trajectories = initial_rollout.evaluate_action(
            x_in, use_cuda_graph=False
        )
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(
            sum_horizon=True
        )
        if costs.shape != (self.config.num_problems * self.config.num_particles,):
            log_and_raise(
                f"costs.shape != (num_problems*num_particles,): {costs.shape}"
            )
        cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        cost.backward(gradient=self._l_vec, retain_graph=False)
        g_x = x_n.grad.detach()
        return cost, g_x

    # -- Lifecycle --

    def reinitialize(
        self,
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        clear_optimizer_state: bool = True,
        reset_num_iters: bool = False,
    ) -> None:
        """Re-seed the optimizer with a new action and optionally reset state.

        Clears the best-solution tracker, invokes the ``on_reinitialize``
        callback, evaluates the seed action through the initial rollout, and
        caches the resulting iteration state for the next :meth:`optimize` call.

        Args:
            action: Seed action tensor of shape
                ``(num_problems, action_horizon, action_dim)`` or flat.
            mask: Optional boolean tensor of shape ``(num_problems,)``.
                When provided, only masked problems should be re-seeded.
            clear_optimizer_state: If True, clear quasi-Newton / AL state.
                If False, preserve history for trajectory refinement.
            reset_num_iters: If True, restore ``num_iters`` to its
                original configured value.
        """
        action = action.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )

        if mask is None and self._debug:
            self._debug.clear()

        if reset_num_iters:
            self.config.num_iters = self._og_num_iters

        self._best.clear(mask)
        if self._on_reinitialize:
            self._on_reinitialize(mask)

        iteration_state = self._create_initial_iteration_state(action)
        iteration_state = self._run_initial_evaluation(iteration_state)

        if self._on_initial_state:
            self._on_initial_state(iteration_state, mask)

        if self._iteration_state is None:
            self._iteration_state = iteration_state
        else:
            self._iteration_state.copy_(iteration_state)

    def shift(self, shift_steps: int = 0) -> bool:
        return self._shift(shift_steps)

    def _shift(self, shift_steps: int = 0) -> bool:
        if shift_steps == 0:
            return True
        if self._on_shift:
            self._on_shift(shift_steps)
        self._best.reset()
        return True

    def update_num_problems(self, num_problems: int):
        assert num_problems > 0
        self.config.num_problems = num_problems

        self._best.resize(num_problems, self.action_horizon, self.action_dim)

        if self._on_resize:
            self._on_resize(num_problems)

        self._l_vec = torch.ones(
            (num_problems, self.config.num_particles, 1),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

        if self._line_search_context is None:
            self._line_search_context = self._create_line_search_context()
            self._line_search_strategy = LineSearchStrategyFactory.get_strategy(
                self.config.line_search_type
            )
            self._initial_line_search_context = self._create_line_search_context(
                compute_costs_and_gradients=(
                    self._compute_cost_constraint_and_gradient_initial
                ),
            )
            self._initial_line_search_strategy = LineSearchStrategyFactory.get_strategy(
                self.config.line_search_type
            )

        self._line_search_context.update_num_problems(num_problems)
        self._line_search_strategy.update_num_problems(
            num_problems, self._line_search_context
        )
        self._initial_line_search_context.update_num_problems(num_problems)
        self._initial_line_search_strategy.update_num_problems(
            num_problems, self._initial_line_search_context
        )

        for rollout in self._rollout_list:
            rollout.update_batch_size(
                batch_size=num_problems * self.config.num_particles
            )

    def _create_line_search_context(
        self, compute_costs_and_gradients=None
    ) -> LineSearchContext:
        if compute_costs_and_gradients is None:
            compute_costs_and_gradients = self._compute_cost_constraint_and_gradient
        return LineSearchContext(
            device_cfg=self.device_cfg,
            line_search_c_1=self.config.line_search_wolfe_c_1,
            line_search_c_2=self.config.line_search_wolfe_c_2,
            opt_dim=self.opt_dim,
            use_cuda_kernel_line_search=self.config.use_cuda_kernel_line_search,
            compute_costs_and_gradients=compute_costs_and_gradients,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            step_scale=self.config.step_scale,
            fix_terminal_action=self.config.fix_terminal_action,
            action_horizon_step_max=self._bounds.step_max,
            line_search_scale=self.config.line_search_scale,
            num_problems=self.config.num_problems,
            convergence_iteration=self.config.convergence_iteration,
            cost_delta_threshold=self.config.cost_delta_threshold,
            cost_relative_threshold=self.config.cost_relative_threshold,
        )

    def update_rollout_params(self, goal):
        for rollout in self._rollout_list:
            rollout.update_params(goal, num_particles=self.config.num_particles)

    def update_goal_dt(self, goal):
        for rollout in self._rollout_list:
            rollout.update_goal_dt(goal)

    def get_all_rollout_instances(self) -> List[Rollout]:
        return self._rollout_list

    def compute_metrics(self, action: torch.Tensor):
        return self.rollout_fn.compute_metrics_from_action(action)

    def reset_shape(self):
        for rollout in self._rollout_list:
            rollout.reset_shape()

    def reset_seed(self) -> bool:
        return True

    def reset_cuda_graph(self):
        for executor in self._executors.values():
            if executor is not None:
                executor.reset()
        if hasattr(self.rollout_fn, "reset_cuda_graph"):
            self.rollout_fn.reset_cuda_graph()

    def get_recorded_trace(self) -> Dict[str, Any]:
        if self._debug:
            return self._debug.get_trace()
        return {"debug": [], "debug_cost": []}

    def update_solver_params(self, solver_params: Dict[str, Dict[str, Any]]) -> bool:
        if self.config.solver_name not in solver_params:
            log_and_raise(
                f"Optimizer {self.config.solver_name} not found in {solver_params}"
            )
        for param_name, param_value in solver_params[self.config.solver_name].items():
            setattr(self.config, param_name, param_value)
        if self.config.inner_iters > self.config.num_iters:
            log_and_raise(
                f"inner_iters {self.config.inner_iters} > num_iters "
                f"{self.config.num_iters}"
            )
        if self.config.num_iters % self.config.inner_iters != 0:
            log_and_raise(
                f"num_iters {self.config.num_iters} is not a multiple of "
                f"inner_iters {self.config.inner_iters}"
            )
        if self.outer_iters <= 0:
            log_and_raise(
                f"outer_iters {self.outer_iters} <= 0"
            )
        return True

    def update_niters(self, niters: int):
        self.config.update_niters(niters)

    def _record_iteration_state(self, iteration_state: OptimizationIterationState):
        if self._debug:
            self._debug.record(iteration_state, self.action_horizon, self.action_dim)

    def debug_dump(self, file_path: str = ""):
        if self.use_cuda_graph:
            from curobo._src.util.logging import log_warn

            for method_name, executor in self._executors.items():
                if executor is not None:
                    file_name = (
                        f"{file_path}_{self.config.solver_name}_{method_name}.dot"
                    )
                    log_warn(
                        f"Dumping CUDA Graph for {method_name} to {file_name}"
                    )
                    executor.debug_dump(file_name)
