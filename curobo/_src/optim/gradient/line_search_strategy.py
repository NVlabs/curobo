# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This module contains the line search strategies for use with step direction methods.
A line search strategy is used to find the optimal step size along the search direction to
converge quickly. This is important for solving non-convex problems as taking the full step at
each iteration can lead to significant sub-optimal performance. The line search strategies are
used by :class:`~curobo.optim.gradient.line_search_opt.LineSearchOpt` to optimize alongside methods
to calculate the step direction.

cuRobo implements several line search strategies, including:

1. Greedy line search
   (:class:`~curobo.optim.gradient.line_search_strategy.GreedyLineSearchStrategy`)
2. Armijo line search
   (:class:`~curobo.optim.gradient.line_search_strategy.ArmijoLineSearchStrategy`)
3. Wolfe line search (:class:`~curobo.optim.gradient.line_search_strategy.WolfeLineSearchStrategy`)
4. Strong Wolfe line search
   (:class:`~curobo.optim.gradient.line_search_strategy.StrongWolfeLineSearchStrategy`)
5. Approximate Wolfe line search
   (:class:`~curobo.optim.gradient.line_search_strategy.ApproxWolfeLineSearchStrategy`)

The line search strategies implemented try different step sizes in parallel for all problems in the
batch and then select the optimal step size per problem. We found Approximate Wolfe line search to
work well in practice for trajectory optimization and inverse kinematics problems. For applications
where intermediate solutions to optimization should also be feasible, we recommend using the
Strong Wolfe line search (e.g., model predictive control).

When enabled, ``line_search_backtracking`` retries a failed probe fan at one tenth
the step scale. Each problem owns one GPU ``beta`` value, reset to one after an
accepted step. Retries preserve the search direction and optimizer history and
consume the existing iteration budget. MPC profiles enable it; other solvers use
the fallback policies listed below by default. Cost acceptance does not replace
trajectory validation.

Given the cost :math:`c(x)` evaluated at a point :math:`x`, gradient :math:`g(x)` and a step
direction :math:`p`, the line search methods try different step sizes :math:`l` and check if the
following conditions are satisfied:

- Armijo condition: :math:`f(x + l p) <= f(x) + c_1 l \nabla f(x)^T p`
- Weak Wolfe condition: :math:`\nabla f(x + l p)^T p >= c_2 \nabla f(x)^T p`
- Strong Wolfe condition: :math:`|\nabla f(x + l p)^T p| <= c_2 |\nabla f(x)^T p|`

where :math:`c_1` and :math:`c_2` are parameters used to tune the line search,
with :math:`c_1 = 1e-3` and :math:`c_2 = 0.9` being common choices. Some line search strategies use
fallback mechanisms to be robust to optimization difficulties.

.. list-table::
   :header-rows: 1
   :align: left

   * - Method
     - Conditions
     - Fallbacks

   * - Greedy
     - Largest reduction in cost
     - None

   * - Armijo
     - Armijo condition
     - None

   * - Wolfe
     - Armijo condition & Weak Wolfe condition
     - Armijo condition

   * - Strong Wolfe
     - Armijo condition & Strong Wolfe condition
     - None

   * - Approximate Wolfe
     - Armijo condition & Weak Wolfe condition
     - Armijo condition, :math:`l = l_1`

"""

# Standard Library
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import Callable, Tuple

import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo._src.curobolib.cuda_ops.optimization import wolfe_line_search

# Third Party
from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.gradient.line_search_result import LineSearchResult
from curobo._src.optim.gradient.line_search_state import LineSearchState
from curobo._src.optim.gradient.update_best_solution import update_best_solution
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import get_torch_jit_decorator

__all__ = [
    "LineSearchType",
    "LineSearchStrategy",
    "GreedyLineSearchStrategy",
    "ArmijoLineSearchStrategy",
    "WolfeLineSearchStrategy",
    "StrongWolfeLineSearchStrategy",
    "ApproxWolfeLineSearchStrategy",
    "LineSearchStatus",
    "LineSearchStrategyFactory",
]


class LineSearchType(Enum):
    GREEDY = "greedy"
    ARMIJO = "armijo"
    WOLFE = "wolfe"
    STRONG_WOLFE = "strong_wolfe"
    APPROX_WOLFE = "approx_wolfe"
    APPROX_STRONG_WOLFE = "approx_strong_wolfe"


class LineSearchStatus(IntEnum):
    """Outcome used to advance the exploration state."""

    BASELINE = 0
    WOLFE = 1
    ARMIJO = 2
    GREEDY = 3
    FALLBACK = 4
    BACKTRACK = 5


class LineSearchStrategy(ABC):
    """Base class for line search strategies with common functionality."""

    @abstractmethod
    def search(
        self,
        iteration_state: OptimizationIterationState,
        context: LineSearchContext,
    ) -> LineSearchResult:
        """Perform line search to find optimal step size.

        Args:
            iteration_state: Current iteration state
            context: Context containing parameters needed for line search

        Returns:
            LineSearchResult containing the selected action, cost, and gradient
        """
        pass

    def update_num_problems(self, num_problems: int, context: LineSearchContext):
        pass

    @staticmethod
    def _get_search_scales(context: LineSearchContext) -> torch.Tensor:
        """Return fixed scales with shape ``(n_linesearch,)``."""
        return context.get_line_search_scale()

    @profiler.record_function("LineSearchStrategy/prepare_search_points")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _prepare_search_points(
        self,
        x: torch.Tensor,
        step_direction: torch.Tensor,
        context: LineSearchContext,
        line_search_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare points to evaluate during line search.

        Args:
            x: The current point. Shape: (num_problems, action_horizon, action_dim)
            step_direction: The direction to search in.
                Shape: (num_problems, action_horizon, action_dim)
            context: The context containing parameters needed for line search.
            line_search_scale: Positive scales with shape ``(n_linesearch,)``.

        Returns:
            x_set: The set of points to evaluate.
                   Shape: (num_problems, n_linesearch, action_horizon * action_dim)
            step_direction: The direction to search in.
                   Shape: (num_problems, 1, action_horizon * action_dim)
        """
        if x.ndim != 3:
            log_and_raise(
                f"x must have shape (num_problems, action_horizon, action_dim). Got {x.shape}"
            )
        if step_direction.ndim != 3:
            log_and_raise(
                "step_direction must have shape (num_problems, action_horizon, action_dim). "
                + f"Got {step_direction.shape}"
            )

        num_problems = x.shape[0]
        if x.shape != (num_problems, context.action_horizon, context.action_dim):
            log_and_raise(
                f"x must have shape (num_problems, action_horizon, action_dim). Got {x.shape}"
            )
        if step_direction.shape != (num_problems, context.action_horizon, context.action_dim):
            log_and_raise(
                "step_direction must have shape (num_problems, action_horizon, action_dim). "
                + f"Got {step_direction.shape}"
            )

        step_direction = step_direction.detach()

        x = x.detach()
        probe_scales = line_search_scale.view(1, -1, 1, 1)
        if context.line_search_backtracking:
            probe_scales = context.beta.view(-1, 1, 1, 1) * probe_scales
        x_set = self.jit_get_x_set(step_direction, x, probe_scales)

        x_set = x_set.view(
            x_set.shape[0],
            x_set.shape[1],
            context.action_horizon * context.action_dim,
        )
        step_direction = step_direction.view(
            step_direction.shape[0], 1, context.action_horizon * context.action_dim
        )
        x_set = x_set.detach().requires_grad_(True)

        return x_set, step_direction

    @profiler.record_function("LineSearchStrategy/_post_search_processing")
    def _post_search_processing(
        self,
        line_search_result: LineSearchResult,
        context: LineSearchContext,
        iteration_state: OptimizationIterationState,
    ) -> OptimizationIterationState:
        """Post-search processing."""
        if context.line_search_backtracking and not (
            context.use_cuda_kernel_line_search and isinstance(self, BaseWolfeLineSearchStrategy)
        ):
            retry = ~line_search_result.curvature_update_valid
            context.beta.copy_(torch.where(retry, context.beta * 0.1, 1.0))
            line_search_result.line_search_status = torch.where(
                retry, LineSearchStatus.BACKTRACK, line_search_result.line_search_status
            )
        exploration_state = line_search_result.exploration_state

        next_iteration_state = OptimizationIterationState(
            action=line_search_result.selected_state.action.reshape(
                line_search_result.selected_state.action.shape[0],
                context.action_horizon,
                context.action_dim,
            ).contiguous(),
            exploration_action=exploration_state.action.reshape(
                exploration_state.action.shape[0], context.action_horizon, context.action_dim
            ).contiguous(),
            exploration_gradient=exploration_state.gradient.reshape(
                exploration_state.gradient.shape[0], context.action_horizon, context.action_dim
            ).contiguous(),
            exploration_cost=exploration_state.cost,
            gradient=line_search_result.selected_state.gradient.reshape(
                line_search_result.selected_state.gradient.shape[0],
                context.action_horizon,
                context.action_dim,
            ).contiguous(),
            cost=line_search_result.selected_state.cost,
            step_direction=iteration_state.step_direction,
            best_action=iteration_state.best_action,
            best_cost=iteration_state.best_cost,
            current_iteration=iteration_state.current_iteration,
            best_iteration=iteration_state.best_iteration,
            state=iteration_state.state,
            converged=iteration_state.converged,
            line_search_index=line_search_result.line_search_index,
            line_search_status=line_search_result.line_search_status,
            curvature_update_valid=line_search_result.curvature_update_valid,
        )

        return next_iteration_state

    @staticmethod
    def _baseline_state(
        iteration_state: OptimizationIterationState,
        context: LineSearchContext,
    ) -> LineSearchState:
        """Create a line-search state from the cached exploration baseline."""
        batch_size = context.num_problems
        indices = torch.full(
            (batch_size, context.n_linesearch),
            -1,
            device=iteration_state.exploration_action.device,
            dtype=torch.int32,
        )
        return LineSearchState(
            action=iteration_state.exploration_action,
            cost=iteration_state.exploration_cost,
            gradient=iteration_state.exploration_gradient,
            idxs=indices,
        )

    @staticmethod
    def _gather_trial_state(
        x_set: torch.Tensor,
        cost: torch.Tensor,
        gradient: torch.Tensor,
        local_index: torch.Tensor,
        context: LineSearchContext,
    ) -> LineSearchState:
        """Gather one evaluated trial per problem."""
        batch_size, n_linesearch, _ = x_set.shape
        batch_index = torch.arange(batch_size, device=x_set.device)
        action = x_set[batch_index, local_index]
        selected_cost = cost.view(batch_size, n_linesearch)[batch_index, local_index]
        selected_gradient = gradient[batch_index, local_index]
        repeated_index = local_index.to(dtype=torch.int32).view(-1, 1).expand(-1, n_linesearch)
        return LineSearchState(
            action=action.view(batch_size, context.action_horizon, context.action_dim),
            cost=selected_cost,
            gradient=selected_gradient.view(
                batch_size, context.action_horizon, context.action_dim
            ),
            idxs=repeated_index,
        )

    @staticmethod
    def _choose_state(
        baseline: LineSearchState,
        trial: LineSearchState,
        choose_trial: torch.Tensor,
    ) -> LineSearchState:
        """Choose complete action, cost, and gradient tuples without mixing sources."""
        action_mask = choose_trial.view(-1, 1, 1)
        index_mask = choose_trial.view(-1, 1)
        return LineSearchState(
            action=torch.where(action_mask, trial.action, baseline.action),
            cost=torch.where(choose_trial, trial.cost, baseline.cost),
            gradient=torch.where(action_mask, trial.gradient, baseline.gradient),
            idxs=torch.where(index_mask, trial.idxs, baseline.idxs),
        )

    @staticmethod
    def _make_result(
        selected_state: LineSearchState,
        exploration_state: LineSearchState,
        exploration_index: torch.Tensor,
        status: torch.Tensor,
        curvature_update_valid: torch.Tensor,
    ) -> LineSearchResult:
        """Construct a complete line-search result."""
        return LineSearchResult(
            selected_state=selected_state,
            exploration_state=exploration_state,
            line_search_index=exploration_index.to(dtype=torch.int32),
            line_search_status=status.to(dtype=torch.int32),
            curvature_update_valid=curvature_update_valid,
        )

    def _compute_costs_and_gradients(
        self,
        x_set: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute costs and gradients at all search points."""
        return context.compute_costs_and_gradients(x_set)

    @staticmethod
    @get_torch_jit_decorator()
    def jit_get_x_set(
        step_vec: torch.Tensor, x: torch.Tensor, line_search_scales: torch.Tensor
    ) -> torch.Tensor:
        """Get the set of points to evaluate during line search.

        Args:
            step_vec: The step vector to evaluate. Shape: (num_problems, action_horizon, action_dim)
            x: The current point. Shape: (num_problems, action_horizon, action_dim)
            line_search_scales: Per-problem line search scales to evaluate.
                Shape: (num_problems, num_particles, 1, 1)

        Returns:
            The set of points to evaluate. Shape: (B, H, N, D)
        """
        x = x.unsqueeze(1)  # batch, 1, action_horizon, action_dim
        step_vec = step_vec.unsqueeze(1)  # batch, 1, action_horizon, action_dim
        x_set = (
            x + line_search_scales * step_vec
        )  # batch, num_particles, action_horizon, action_dim
        return x_set

    def _update_best_solution(
        self, iteration_state: OptimizationIterationState, context: LineSearchContext
    ) -> OptimizationIterationState:
        """Update the best solution."""
        return update_best_solution(
            iteration_state,
            context.action_horizon,
            context.action_dim,
            context.cost_delta_threshold,
            context.cost_relative_threshold,
            context.convergence_iteration,
        )


class GreedyLineSearchStrategy(LineSearchStrategy):
    """Greedy line search strategy.

    Select the minimum-cost positive probe when it improves the cached baseline.
    """

    @profiler.record_function("GreedyLineSearchStrategy/search")
    def search(
        self,
        iteration_state: OptimizationIterationState,
        context: LineSearchContext,
    ) -> OptimizationIterationState:
        """Perform greedy line search to find optimal step size."""
        x = iteration_state.exploration_action

        step_direction = iteration_state.step_direction
        line_search_scale = self._get_search_scales(context)
        x_set, step_direction = self._prepare_search_points(
            x, step_direction, context, line_search_scale
        )
        b, n_line_search, _ = x_set.shape

        c, g_x = self._compute_costs_and_gradients(x_set, context)

        c = c.detach()
        g_x = g_x.detach()
        best_trial_cost, local_index = torch.min(c.squeeze(-1), dim=1)
        trial_state = self._gather_trial_state(x_set, c, g_x, local_index, context)
        baseline_state = self._baseline_state(iteration_state, context)
        choose_trial = best_trial_cost < baseline_state.cost
        if context.line_search_backtracking:
            choose_trial = choose_trial & torch.isfinite(best_trial_cost)
        exploration_state = self._choose_state(baseline_state, trial_state, choose_trial)
        status = torch.where(
            choose_trial,
            torch.full_like(local_index, LineSearchStatus.GREEDY, dtype=torch.int32),
            torch.full_like(local_index, LineSearchStatus.BASELINE, dtype=torch.int32),
        )
        exploration_index = torch.where(
            choose_trial,
            local_index.to(dtype=torch.int32),
            torch.full_like(local_index, -1, dtype=torch.int32),
        )
        result = self._make_result(
            selected_state=exploration_state,
            exploration_state=exploration_state,
            exploration_index=exploration_index,
            status=status,
            curvature_update_valid=choose_trial,
        )

        result = self._post_search_processing(result, context, iteration_state)
        result = self._update_best_solution(result, context)
        return result


class ArmijoLineSearchStrategy(LineSearchStrategy):
    """Armijo line search strategy.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    .. math::
        c(x + l p) <= c(x) + c_1 l g(x)^T p
    """

    @profiler.record_function("ArmijoLineSearchStrategy/search")
    def search(
        self,
        iteration_state: OptimizationIterationState,
        context: LineSearchContext,
    ) -> OptimizationIterationState:
        """Perform armijo line search to find optimal step size."""
        x = iteration_state.exploration_action
        step_direction = iteration_state.step_direction
        line_search_scale = self._get_search_scales(context)
        x_set, step_direction = self._prepare_search_points(
            x, step_direction, context, line_search_scale
        )
        b, n_line_search, _ = x_set.shape

        c, g_x = self._compute_costs_and_gradients(x_set, context)

        c = c.detach()
        g_x = g_x.detach()

        step_vec_T = step_direction.transpose(-1, -2)
        g_step = iteration_state.exploration_gradient.view(b, 1, -1) @ step_vec_T
        armijo_scales = line_search_scale
        if context.line_search_backtracking:
            armijo_scales = context.beta.view(-1, 1) * armijo_scales
        armijo = c.squeeze(-1) <= iteration_state.exploration_cost.view(-1, 1) + (
            context.line_search_c_1 * armijo_scales * g_step.squeeze(-1)
        )
        if context.line_search_backtracking:
            armijo = (
                armijo
                & torch.isfinite(c.squeeze(-1))
                & torch.isfinite(g_x).all(-1)
                & torch.isfinite(g_step.squeeze(-1))
                & (g_step.squeeze(-1) < 0.0)
                & (c.squeeze(-1) < iteration_state.exploration_cost.view(-1, 1))
            )
        success_scale = torch.where(armijo, line_search_scale, torch.zeros_like(line_search_scale))
        _, local_index = torch.max(success_scale, dim=1)
        choose_trial = torch.any(armijo, dim=1)
        trial_state = self._gather_trial_state(x_set, c, g_x, local_index, context)
        baseline_state = self._baseline_state(iteration_state, context)
        exploration_state = self._choose_state(baseline_state, trial_state, choose_trial)
        status = torch.where(
            choose_trial,
            torch.full_like(local_index, LineSearchStatus.ARMIJO, dtype=torch.int32),
            torch.full_like(local_index, LineSearchStatus.BASELINE, dtype=torch.int32),
        )
        exploration_index = torch.where(
            choose_trial,
            local_index.to(dtype=torch.int32),
            torch.full_like(local_index, -1, dtype=torch.int32),
        )
        result = self._make_result(
            selected_state=exploration_state,
            exploration_state=exploration_state,
            exploration_index=exploration_index,
            status=status,
            curvature_update_valid=choose_trial,
        )
        result = self._post_search_processing(result, context, iteration_state)

        result = self._update_best_solution(result, context)
        return result


class BaseWolfeLineSearchStrategy(LineSearchStrategy):
    """Base class for all Wolfe line search variants."""

    def __init__(self):
        self._output_buffers = None
        super().__init__()

    @profiler.record_function("BaseWolfeLineSearchStrategy/search")
    def search(
        self,
        iteration_state: OptimizationIterationState,
        context: LineSearchContext,
    ) -> OptimizationIterationState:
        """Perform Wolfe line search to find optimal step size.

        Args:
            x: The current point. Shape: (num_problems, action_horizon, action_dim)
            step_direction: The direction to search in.
                Shape: (num_problems, action_horizon, action_dim)
            context: The context containing parameters needed for line search

        Returns:
            LineSearchResult containing the selected action, cost, and gradient
        """
        x = iteration_state.exploration_action
        line_search_scale = self._get_search_scales(context)
        step_direction = iteration_state.step_direction
        # Prepare search points
        x_set, step_vec = self._prepare_search_points(
            x, step_direction, context, line_search_scale
        )

        # Compute costs and gradients
        c, g_x = self._compute_costs_and_gradients(x_set, context)

        if context.use_cuda_kernel_line_search:
            line_search_result = self._cuda_kernel_wolfe_search(
                x_set,
                c,
                g_x,
                step_vec,
                iteration_state,
                context,
                line_search_scale,
            )

            new_iteration_state = self._post_search_processing(
                line_search_result, context, iteration_state
            )

        else:
            line_search_result = self._torch_wolfe_search(
                x_set,
                step_vec,
                c,
                g_x,
                iteration_state,
                line_search_scale,
                context,
                self._compute_curvature_condition,
                self._handle_no_valid_step,
            )

            new_iteration_state = self._post_search_processing(
                line_search_result, context, iteration_state
            )
            new_iteration_state = self._update_best_solution(new_iteration_state, context)

        return new_iteration_state

    def update_num_problems(self, num_problems: int, context: LineSearchContext):
        if context.use_cuda_kernel_line_search:
            num_problems = context.num_problems
            n_linesearch = context.n_linesearch
            curvature_update_default = isinstance(self, ApproxWolfeLineSearchStrategy)
            if self._output_buffers is None:
                self._output_buffers = (
                    torch.zeros(
                        (num_problems, n_linesearch),
                        dtype=torch.int32,
                        device=context.device_cfg.device,
                    ),
                    torch.zeros(
                        (num_problems, n_linesearch),
                        dtype=torch.int32,
                        device=context.device_cfg.device,
                    ),
                    torch.full(
                        (num_problems,),
                        -1,
                        dtype=torch.int32,
                        device=context.device_cfg.device,
                    ),
                    torch.zeros(
                        (num_problems,),
                        dtype=torch.int32,
                        device=context.device_cfg.device,
                    ),
                    torch.full(
                        (num_problems,),
                        curvature_update_default,
                        dtype=torch.bool,
                        device=context.device_cfg.device,
                    ),
                )

            (
                exploration_idx,
                selected_idx,
                line_search_index,
                line_search_status,
                curvature_update_valid,
            ) = self._output_buffers

            if exploration_idx.shape != (num_problems, n_linesearch):
                log_info(
                    f"exploration_idx.shape != (num_problems, n_linesearch): {exploration_idx.shape} != ({num_problems}, {n_linesearch})"
                )
                exploration_idx = torch.zeros(
                    (num_problems, n_linesearch),
                    dtype=torch.int32,
                    device=context.device_cfg.device,
                )
            if selected_idx.shape != (num_problems, n_linesearch):
                log_info(
                    f"selected_idx.shape != (num_problems, n_linesearch): {selected_idx.shape} != ({num_problems}, {n_linesearch})"
                )
                selected_idx = torch.zeros(
                    (num_problems, n_linesearch),
                    dtype=torch.int32,
                    device=context.device_cfg.device,
                )

            if line_search_index.shape != (num_problems,):
                line_search_index = torch.full(
                    (num_problems,),
                    -1,
                    dtype=torch.int32,
                    device=context.device_cfg.device,
                )
                line_search_status = torch.zeros(
                    (num_problems,),
                    dtype=torch.int32,
                    device=context.device_cfg.device,
                )
                curvature_update_valid = torch.full(
                    (num_problems,),
                    curvature_update_default,
                    dtype=torch.bool,
                    device=context.device_cfg.device,
                )

            self._output_buffers = (
                exploration_idx,
                selected_idx,
                line_search_index,
                line_search_status,
                curvature_update_valid,
            )

        return super().update_num_problems(num_problems, context)

    @profiler.record_function("BaseWolfeLineSearchStrategy/_torch_wolfe_search")
    # @staticmethod
    def _torch_wolfe_search(
        self,
        x_set: torch.Tensor,
        step_vec: torch.Tensor,
        c: torch.Tensor,
        g_x: torch.Tensor,
        iteration_state: OptimizationIterationState,
        line_search_scale: torch.Tensor,
        context: LineSearchContext,
        compute_curvature_condition: Callable,
        handle_no_valid_step: Callable,
    ) -> LineSearchResult:
        """Implement Wolfe search using PyTorch operations."""
        # with torch.no_grad():
        if True:
            b, n_line_search, _ = x_set.shape
            step_vec_T = step_vec.transpose(-1, -2)
            g_full_step = g_x @ step_vec_T
            g_step = iteration_state.exploration_gradient.view(b, 1, context.opt_dim) @ step_vec_T

            # Wolfe condition 1 (sufficient decrease)
            armijo_scales = line_search_scale
            if context.line_search_backtracking:
                armijo_scales = context.beta.view(-1, 1) * armijo_scales
            armijo_1 = c.squeeze(-1) <= iteration_state.exploration_cost.view(-1, 1) + (
                context.line_search_c_1 * armijo_scales * g_step.squeeze(-1)
            )
            if context.line_search_backtracking:
                armijo_1 = (
                    armijo_1
                    & torch.isfinite(c.squeeze(-1))
                    & torch.isfinite(g_full_step.squeeze(-1))
                    & torch.isfinite(g_step.squeeze(-1))
                    & (g_step.squeeze(-1) < 0.0)
                    & (c.squeeze(-1) < iteration_state.exploration_cost.view(-1, 1))
                )

            # Wolfe condition 2 (curvature) - implemented by subclasses
            wolfe_2 = compute_curvature_condition(g_full_step, g_step, context).squeeze(-1)

            # Combined Wolfe conditions
            wolfe = torch.logical_and(armijo_1, wolfe_2)

            # Find largest step size that satisfies conditions
            wolfe_scale = torch.where(
                wolfe, line_search_scale, torch.zeros_like(line_search_scale)
            )
            _, wolfe_index = torch.max(wolfe_scale, dim=1)
            has_wolfe = torch.any(wolfe, dim=1)
            armijo_scale = torch.where(
                armijo_1, line_search_scale, torch.zeros_like(line_search_scale)
            )
            _, armijo_index = torch.max(armijo_scale, dim=1)
            has_armijo = torch.any(armijo_1, dim=1)

            baseline_state = self._baseline_state(iteration_state, context)
            wolfe_state = self._gather_trial_state(x_set, c, g_x, wolfe_index, context)
            selected_state = self._choose_state(baseline_state, wolfe_state, has_wolfe)

            if isinstance(self, StrongWolfeLineSearchStrategy):
                exploration_index = wolfe_index
                choose_exploration = has_wolfe
                status = torch.where(
                    has_wolfe,
                    torch.full_like(wolfe_index, LineSearchStatus.WOLFE, dtype=torch.int32),
                    torch.full_like(wolfe_index, LineSearchStatus.BASELINE, dtype=torch.int32),
                )
            elif context.line_search_backtracking:
                exploration_index = torch.where(has_wolfe, wolfe_index, armijo_index)
                choose_exploration = has_wolfe | has_armijo
                status = torch.where(
                    has_wolfe,
                    LineSearchStatus.WOLFE,
                    torch.where(has_armijo, LineSearchStatus.ARMIJO, LineSearchStatus.BASELINE),
                ).to(torch.int32)
            else:
                exploration_index, choose_exploration, status = handle_no_valid_step(
                    wolfe_index, has_wolfe, armijo_index, has_armijo, context
                )

            trial_state = self._gather_trial_state(x_set, c, g_x, exploration_index, context)
            exploration_state = self._choose_state(baseline_state, trial_state, choose_exploration)
            if context.line_search_backtracking:
                selected_state = exploration_state
            local_index = torch.where(
                choose_exploration,
                exploration_index.to(dtype=torch.int32),
                torch.full_like(exploration_index, -1, dtype=torch.int32),
            )

            return self._make_result(
                selected_state=selected_state,
                exploration_state=exploration_state,
                exploration_index=local_index,
                status=status,
                curvature_update_valid=choose_exploration,
            )

    @profiler.record_function("BaseWolfeLineSearchStrategy/_cuda_kernel_wolfe_search")
    def _cuda_kernel_wolfe_search(
        self,
        search_action: torch.Tensor,
        search_cost: torch.Tensor,
        search_gradient: torch.Tensor,
        step_direction: torch.Tensor,
        iteration_state: OptimizationIterationState,
        context: LineSearchContext,
        line_search_scale: torch.Tensor,
    ) -> LineSearchResult:
        """Implement Wolfe search using CUDA kernels."""
        (
            exploration_idx,
            selected_idx,
            line_search_index,
            line_search_status,
            curvature_update_valid,
        ) = self._get_cuda_kernel_output_buffers(context)
        # Call CUDA kernel
        is_strong_wolfe = isinstance(self, StrongWolfeLineSearchStrategy)
        is_approx_wolfe = isinstance(self, ApproxWolfeLineSearchStrategy)
        is_approx_strong_wolfe = isinstance(self, ApproxStrongWolfeLineSearchStrategy)
        if is_approx_strong_wolfe:
            log_and_raise("Approx strong Wolfe not implemented with cuda kernel")

        for tensor_name in (
            "best_action",
            "exploration_action",
            "exploration_gradient",
            "action",
            "gradient",
        ):
            tensor = getattr(iteration_state, tensor_name)
            if not tensor.is_contiguous():
                setattr(iteration_state, tensor_name, tensor.contiguous())
        search_cost = search_cost.contiguous()
        search_action = search_action.contiguous()
        search_gradient = search_gradient.contiguous()
        step_direction = step_direction.contiguous()

        (new_iteration_state, exploration_idx, selected_idx) = wolfe_line_search(
            iteration_state,
            context,
            exploration_idx,
            selected_idx,
            search_cost,
            search_action,
            search_gradient,
            step_direction,
            is_strong_wolfe,
            is_approx_wolfe,
            line_search_scale,
            line_search_index,
            line_search_status,
            curvature_update_valid,
        )
        b, h, _ = search_action.shape

        # We don't have direct access to the step size from the CUDA kernel
        # Could be added as an output if needed
        selected_state = LineSearchState(
            action=new_iteration_state.action.detach().view(
                b, context.action_horizon, context.action_dim
            ),
            cost=new_iteration_state.cost.detach(),
            gradient=new_iteration_state.gradient.detach().view(
                b, context.action_horizon, context.action_dim
            ),
            idxs=selected_idx,
        )
        exploration_state = LineSearchState(
            action=new_iteration_state.exploration_action.detach().view(
                b, context.action_horizon, context.action_dim
            ),
            cost=new_iteration_state.exploration_cost.detach(),
            gradient=new_iteration_state.exploration_gradient.detach().view(
                b, context.action_horizon, context.action_dim
            ),
            idxs=exploration_idx,
        )

        return LineSearchResult(
            selected_state=selected_state,
            exploration_state=exploration_state,
            line_search_index=new_iteration_state.line_search_index,
            line_search_status=new_iteration_state.line_search_status,
            curvature_update_valid=new_iteration_state.curvature_update_valid,
        )

    @profiler.record_function("BaseWolfeLineSearchStrategy/_get_cuda_kernel_output_buffers")
    def _get_cuda_kernel_output_buffers(
        self, context: LineSearchContext
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare output buffers for CUDA line search."""
        if self._output_buffers is None:
            log_and_raise("Output buffers are not initialized")

        return self._output_buffers

    @abstractmethod
    def _compute_curvature_condition(
        self, g_full_step: torch.Tensor, g_step: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Compute the curvature condition (Wolfe condition 2)."""
        pass

    @abstractmethod
    def _handle_no_valid_step(
        self,
        wolfe_index: torch.Tensor,
        has_wolfe: torch.Tensor,
        armijo_index: torch.Tensor,
        has_armijo: torch.Tensor,
        valid: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle the case where no step satisfies the Wolfe conditions."""
        pass


class WolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Standard Wolfe line search.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * g(x)^T p

    Wolfe condition 2:

    .. math::
        g(x + l * p)^T p >= c_2 * g(x)^T p
    """

    @staticmethod
    def _compute_curvature_condition(
        g_full_step: torch.Tensor, g_step: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Compute the standard Wolfe curvature condition."""
        wolfe_condition = g_full_step >= context.line_search_c_2 * g_step
        return wolfe_condition

    def _handle_no_valid_step(
        self,
        wolfe_index: torch.Tensor,
        has_wolfe: torch.Tensor,
        armijo_index: torch.Tensor,
        has_armijo: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fall back to the largest Armijo step, then the baseline."""
        del context
        index = torch.where(has_wolfe, wolfe_index, armijo_index)
        choose = has_wolfe | has_armijo
        status = torch.where(
            has_wolfe,
            torch.full_like(index, LineSearchStatus.WOLFE, dtype=torch.int32),
            torch.where(
                has_armijo,
                torch.full_like(index, LineSearchStatus.ARMIJO, dtype=torch.int32),
                torch.full_like(index, LineSearchStatus.BASELINE, dtype=torch.int32),
            ),
        )
        return index, choose, status


class StrongWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Strong Wolfe line search with stricter curvature condition.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * g(x)^T p

    Strong Wolfe condition 2:

    .. math::
        |g(x + l * p)^T p| <= c_2 * |g(x)^T p|
    """

    @staticmethod
    def _compute_curvature_condition(
        g_full_step: torch.Tensor, g_step: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Compute the strong Wolfe curvature condition."""
        strong_wolfe_condition = torch.abs(g_full_step) <= context.line_search_c_2 * torch.abs(
            g_step
        )
        return strong_wolfe_condition

    @staticmethod
    def _handle_no_valid_step(
        wolfe_index: torch.Tensor,
        has_wolfe: torch.Tensor,
        armijo_index: torch.Tensor,
        has_armijo: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Don't fall back to Armijo condition."""
        del armijo_index, has_armijo, context
        status = torch.where(
            has_wolfe,
            torch.full_like(wolfe_index, LineSearchStatus.WOLFE, dtype=torch.int32),
            torch.full_like(wolfe_index, LineSearchStatus.BASELINE, dtype=torch.int32),
        )
        return wolfe_index, has_wolfe, status


class ApproxWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Approximate Wolfe line search with relaxed conditions.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * g(x)^T p

    Weak Wolfe condition 2:

    .. math::
        g(x + l * p)^T p >= c_2 * g(x)^T p

    When the weak Wolfe condition fails, the line search falls back to condition 1. If
    condition 1 also fails, the line search returns step size line_search_scale[1] .
    """

    @staticmethod
    def _compute_curvature_condition(
        g_full_step: torch.Tensor, g_step: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Compute the approximate Wolfe curvature condition."""
        wolfe_condition = g_full_step >= context.line_search_c_2 * g_step

        return wolfe_condition

    @staticmethod
    def _handle_no_valid_step(
        wolfe_index: torch.Tensor,
        has_wolfe: torch.Tensor,
        armijo_index: torch.Tensor,
        has_armijo: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For approximate Wolfe, fallback to armijo condition, return step size at index 1."""
        fallback_index = min(1, context.n_linesearch - 1)
        fixed_fallback = torch.full_like(wolfe_index, fallback_index)
        index = torch.where(
            has_wolfe,
            wolfe_index,
            torch.where(has_armijo, armijo_index, fixed_fallback),
        )
        status = torch.where(
            has_wolfe,
            torch.full_like(index, LineSearchStatus.WOLFE, dtype=torch.int32),
            torch.where(
                has_armijo,
                torch.full_like(index, LineSearchStatus.ARMIJO, dtype=torch.int32),
                torch.full_like(index, LineSearchStatus.FALLBACK, dtype=torch.int32),
            ),
        )
        return index, torch.ones_like(has_wolfe), status


class ApproxStrongWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Approximate strong Wolfe line search with relaxed conditions.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * g(x)^T p

    Strong Wolfe condition 2:

    .. math::
        |g(x + l * p)^T p| <= c_2 * |g(x)^T p|

    When strong wolfe condition fails, the line search falls back to condition 1. If
    condition 1 also fails, the line search returns step size line_search_scale[1] .
    """

    @staticmethod
    def _compute_curvature_condition(
        g_full_step: torch.Tensor, g_step: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Compute the approximate Wolfe curvature condition."""
        wolfe_condition = torch.abs(g_full_step) <= context.line_search_c_2 * torch.abs(g_step)

        return wolfe_condition

    @staticmethod
    def _handle_no_valid_step(
        wolfe_index: torch.Tensor,
        has_wolfe: torch.Tensor,
        armijo_index: torch.Tensor,
        has_armijo: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For approximate Wolfe, fallback to armijo condition, return step size at index 1."""
        return ApproxWolfeLineSearchStrategy._handle_no_valid_step(
            wolfe_index,
            has_wolfe,
            armijo_index,
            has_armijo,
            context,
        )


class LineSearchStrategyFactory:
    """Factory for creating line search strategies."""

    _strategies = {
        LineSearchType.GREEDY: GreedyLineSearchStrategy,
        LineSearchType.ARMIJO: ArmijoLineSearchStrategy,
        LineSearchType.WOLFE: WolfeLineSearchStrategy,
        LineSearchType.STRONG_WOLFE: StrongWolfeLineSearchStrategy,
        LineSearchType.APPROX_WOLFE: ApproxWolfeLineSearchStrategy,
        LineSearchType.APPROX_STRONG_WOLFE: ApproxStrongWolfeLineSearchStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy_type: LineSearchType) -> LineSearchStrategy:
        """Get a line search strategy by type."""
        if strategy_type not in cls._strategies:
            log_and_raise(f"Unknown line search type: {strategy_type}")
        return cls._strategies[strategy_type]()

    @classmethod
    def register_strategy(cls, strategy_type: LineSearchType, strategy: LineSearchStrategy):
        """Register a new line search strategy. Useful for adding custom strategies.

        Args:
            strategy_type (LineSearchType): The type of line search strategy to register
            strategy (LineSearchStrategy): The strategy to register
        """
        if strategy_type in cls._strategies:
            log_and_raise(f"Line search strategy {strategy_type} already registered")
        cls._strategies[strategy_type] = strategy
