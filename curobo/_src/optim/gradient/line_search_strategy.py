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

Given the cost :math:`c(x)` evaluated at a point :math:`x`, gradient :math:`g(x)` and a step
direction :math:`p`, the line search methods try different step sizes :math:`l` and check if the
following conditions are satisfied:

- Armijo condition: :math:`f(x + l * p) <= f(x) + c_1 * l * (g(x) * p^T)`
- Weak Wolfe condition: :math:`g(x + l * p) * p^T >= c_2 * g(x) * p^T`
- Strong Wolfe condition: :math:`|g(x + l * p) * p^T| <= c_2 * |g(x) * p^T|`

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
     - Armijo condition & Strong Wolfe condition
     - Armijo condition, :math:`l = l_1`

"""

# Standard Library
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Tuple

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
    "LineSearchStrategyFactory",
]


class LineSearchType(Enum):
    GREEDY = "greedy"
    ARMIJO = "armijo"
    WOLFE = "wolfe"
    STRONG_WOLFE = "strong_wolfe"
    APPROX_WOLFE = "approx_wolfe"
    APPROX_STRONG_WOLFE = "approx_strong_wolfe"


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

    @profiler.record_function("LineSearchStrategy/prepare_search_points")
    @get_torch_jit_decorator(only_valid_for_compile=True)
    def _prepare_search_points(
        self,
        x: torch.Tensor,
        step_direction: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare points to evaluate during line search.

        Args:
            x: The current point. Shape: (num_problems, action_horizon, action_dim)
            step_direction: The direction to search in.
                Shape: (num_problems, action_horizon, action_dim)
            context: The context containing parameters needed for line search

        Returns:
            x_set: The set of points to evaluate.
                   Shape: (num_problems, line_search_scale.shape[1], action_horizon * action_dim)
            step_direction: The direction to search in.
                   Shape: (num_problems, 1, action_horizon * action_dim)
        """
        if x.ndim != 3:
            log_and_raise(f"x must have shape (num_problems, action_horizon, action_dim). Got {x.shape}")
        if step_direction.ndim != 3:
            log_and_raise(
                "step_direction must have shape (num_problems, action_horizon, action_dim). "
                + f"Got {step_direction.shape}"
            )

        num_problems = x.shape[0]
        if x.shape != (num_problems, context.action_horizon, context.action_dim):
            log_and_raise(f"x must have shape (num_problems, action_horizon, action_dim). Got {x.shape}")
        if step_direction.shape != (num_problems, context.action_horizon, context.action_dim):
            log_and_raise(
                "step_direction must have shape (num_problems, action_horizon, action_dim). "
                + f"Got {step_direction.shape}"
            )

        step_direction = step_direction.detach()

        if (context.step_scale != 0.0 and context.step_scale != 1.0) or (
            context.fix_terminal_action and context.action_horizon > 1
        ):
            step_direction = self.scale_action(
                step_direction,
                context.action_horizon_step_max,
                context.step_scale,
                context.fix_terminal_action,
                context.action_horizon,
            )

        x = x.detach()
        x_set = self.jit_get_x_set(
            step_direction,
            x,
            context.line_search_scale,
        )

        x_set = x_set.view(
            x_set.shape[0],
            context.line_search_scale.shape[1],
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
        exploration_state = line_search_result.exploration_state

        idxs_selected = exploration_state.idxs

        next_iteration_state = OptimizationIterationState(
            action=line_search_result.selected_state.action.view(
                line_search_result.selected_state.action.shape[0],
                context.action_horizon,
                context.action_dim,
            ),
            exploration_action=exploration_state.action.view(
                exploration_state.action.shape[0], context.action_horizon, context.action_dim
            ),
            exploration_gradient=exploration_state.gradient.view(
                exploration_state.gradient.shape[0], context.action_horizon, context.action_dim
            ),
            exploration_cost=exploration_state.cost,
            gradient=line_search_result.selected_state.gradient.view(
                line_search_result.selected_state.gradient.shape[0],
                context.action_horizon,
                context.action_dim,
            ),
            cost=line_search_result.selected_state.cost,
            step_direction=iteration_state.step_direction,
            best_action=iteration_state.best_action,
            best_cost=iteration_state.best_cost,
            current_iteration=iteration_state.current_iteration,
            best_iteration=iteration_state.best_iteration,
            state=iteration_state.state,
            converged=iteration_state.converged,
        )

        # update best action:
        # next_iteration_state = self._update_best(next_iteration_state)

        return next_iteration_state
        iteration_state.action = line_search_result.selected_state.action.view(
            line_search_result.selected_state.action.shape[0],
            context.action_horizon,
            context.action_dim,
        )
        iteration_state.exploration_action = exploration_state.action.view(
            exploration_state.action.shape[0], context.action_horizon, context.action_dim
        )
        iteration_state.exploration_gradient = exploration_state.gradient.view(
            exploration_state.gradient.shape[0], context.action_horizon, context.action_dim
        )
        iteration_state.exploration_cost = exploration_state.cost

        iteration_state.gradient = line_search_result.selected_state.gradient.view(
            line_search_result.selected_state.gradient.shape[0],
            context.action_horizon,
            context.action_dim,
        )
        iteration_state.cost = line_search_result.selected_state.cost

        return iteration_state

    def _compute_costs_and_gradients(
        self,
        x_set: torch.Tensor,
        context: LineSearchContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
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
            line_search_scales: The list of line search scales to evaluate.
                Shape: (1, num_particles, 1, 1)

        Returns:
            The set of points to evaluate. Shape: (B, H, N, D)
        """
        x = x.unsqueeze(1)  # batch, 1, action_horizon, action_dim
        step_vec = step_vec.unsqueeze(1)  # batch, 1, action_horizon, action_dim
        x_set = x + line_search_scales * step_vec  # batch, num_particles, action_horizon, action_dim
        return x_set

    @staticmethod
    @get_torch_jit_decorator()
    def scale_action(
        dx: torch.Tensor,  # batch, horizon, action_dim
        action_step_max: torch.Tensor,
        step_scale: float,
        fix_terminal_action: bool,
        action_horizon: int,
    ):
        # dx: (num_problems, action_horizon, action_dim)
        if step_scale != 0.0 and step_scale != 1.0:
            # dx_flat = dx.view(dx.shape[0], -1)
            action_step_max_flat = action_step_max.view(1, 1, -1)

            diff = torch.abs(dx) / action_step_max_flat
            scale_value = torch.max(diff.view(dx.shape[0], -1), dim=-1, keepdim=False)[0]

            new_scale = torch.clamp(scale_value, min=1.0)

            new_scale = new_scale.view(dx.shape[0], 1, 1)
            # only perfom for dx that are greater than 1:
            # new_scale = torch.nan_to_num(new_scale, 1.0)
            dx_scaled = dx / new_scale
        else:
            dx_scaled = dx
        if fix_terminal_action and action_horizon > 1:
            dx_scaled[:, action_horizon - 1 :, :] = 0.0

        return dx_scaled

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

    The greedy line search strategy selects the step size that minimizes the cost by
    evaluating the cost at all points in the line search scale and selecting the minimum.
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

        x_set, step_direction = self._prepare_search_points(x, step_direction, context)
        b, n_line_search, _ = x_set.shape

        c, g_x = self._compute_costs_and_gradients(x_set, context)

        c = c.detach()
        g_x = g_x.detach()
        best_c, m_idx = torch.min(c, dim=1)
        best_c = best_c.view(context.num_problems)
        m = m_idx.squeeze() + context.c_idx
        g_x = g_x.view(b * n_line_search, context.opt_dim)
        xs = x_set.view(b * n_line_search, context.opt_dim)
        best_x = xs[m].clone()
        best_grad = g_x[m].view(b, context.opt_dim).clone()

        selected_state = LineSearchState(
            action=best_x.detach().view(b, context.action_horizon, context.action_dim).clone(),
            cost=best_c.detach().clone(),
            gradient=best_grad.detach().view(b, context.action_horizon, context.action_dim).clone(),
            idxs=m.clone(),
        )

        exploration_state = LineSearchState(
            action=best_x.detach().view(b, context.action_horizon, context.action_dim).clone(),
            cost=best_c.detach().clone(),
            gradient=best_grad.detach().view(b, context.action_horizon, context.action_dim).clone(),
            idxs=m.clone(),
        )

        result = LineSearchResult(
            selected_state=selected_state,
            exploration_state=exploration_state,
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
        c(x + l * p) <= c(x) + c_1 * l * (g(x) * p)
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
        x_set, step_direction = self._prepare_search_points(x, step_direction, context)
        b, n_line_search, _ = x_set.shape

        c, g_x = self._compute_costs_and_gradients(x_set, context)

        c = c.detach()
        g_x = g_x.detach()

        c_0 = c[:, 0:1, :]
        g_0 = g_x[:, 0:1, :]

        step_vec_T = step_direction.transpose(-1, -2)
        g_step = g_0 @ step_vec_T
        # g_step = g_step.squeeze(-1)
        # condition 1:
        delta_c = c - c_0
        condition = context.line_search_c_1 * context.line_search_scale.squeeze(-1) * g_step
        armijo_1 = delta_c <= condition

        # get the last occurence of true (this will be the largest admissable alpha value):
        # wolfe will have 1 for indices that satisfy.
        # find the
        step_success = armijo_1 * (context.line_search_scale.squeeze(-1) + 0.1)

        _, m_idx = torch.max(step_success, dim=-2)

        m = m_idx.squeeze() + context.c_idx
        g_x = g_x.view(b * n_line_search, -1)
        xs = x_set.view(b * n_line_search, -1)
        cs = c.view(b * n_line_search)
        best_c = cs[m]

        best_x = xs[m]
        best_grad = g_x[m].view(b, context.opt_dim)

        selected_state = LineSearchState(
            action=best_x.detach().view(b, context.action_horizon, context.action_dim),
            cost=best_c.detach(),
            gradient=best_grad.detach().view(b, context.action_horizon, context.action_dim),
            idxs=m,
        )

        exploration_state = LineSearchState(
            action=best_x.detach().view(b, context.action_horizon, context.action_dim),
            cost=best_c.detach(),
            gradient=best_grad.detach().view(b, context.action_horizon, context.action_dim),
            idxs=m,
        )

        result = LineSearchResult(
            selected_state=selected_state,
            exploration_state=exploration_state,
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
        step_direction = iteration_state.step_direction
        # Prepare search points
        x_set, step_vec = self._prepare_search_points(x, step_direction, context)

        # Compute costs and gradients
        c, g_x = self._compute_costs_and_gradients(x_set, context)

        if context.use_cuda_kernel_line_search:
            line_search_result = self._cuda_kernel_wolfe_search(
                x_set, c, g_x, step_vec, iteration_state, context
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
                )

            exploration_idx, selected_idx = self._output_buffers

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

            self._output_buffers = (
                exploration_idx,
                selected_idx,
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
        context: LineSearchContext,
        compute_curvature_condition: Callable,
        handle_no_valid_step: Callable,
    ) -> LineSearchResult:
        """Implement Wolfe search using PyTorch operations."""
        # with torch.no_grad():
        if True:
            b, n_line_search, _ = x_set.shape
            c_0 = c[:, 0:1]
            step_vec_T = step_vec.transpose(-1, -2)
            g_full_step = g_x @ step_vec_T
            g_step = g_full_step[:, 0:1]

            # Wolfe condition 1 (sufficient decrease)
            # armijo:
            armijo_1 = (
                c <= c_0 + context.line_search_c_1 * context.line_search_scale.squeeze(-1) * g_step
            )

            # Wolfe condition 2 (curvature) - implemented by subclasses
            wolfe_2 = compute_curvature_condition(g_full_step, g_step, context)

            # Combined Wolfe conditions
            wolfe = torch.logical_and(armijo_1, wolfe_2)

            # Find largest step size that satisfies conditions
            step_success = wolfe * (context.line_search_scale.squeeze(-1) + 0.1)

            g_x_1 = g_x.view(b * n_line_search, -1)
            xs_1 = x_set.view(b * n_line_search, -1)
            cs_1 = c.view(b * n_line_search)

            _, m_idx = torch.max(step_success, dim=-2)

            selected_m = m_idx.squeeze() + context.c_idx

            selected_x = xs_1[selected_m]
            selected_c = cs_1[selected_m]
            selected_grad = g_x_1[selected_m].view(b, context.opt_dim)

            idxs_selected = (
                selected_m.view(-1, 1)
                .repeat(1, context.line_search_scale.shape[1])
                .to(dtype=torch.int32)
            )

            selected_state = LineSearchState(
                action=selected_x.detach().view(b, context.action_horizon, context.action_dim),
                cost=selected_c.detach(),
                gradient=selected_grad.detach().view(b, context.action_horizon, context.action_dim),
                idxs=idxs_selected,
            )
            # Handle case where no step satisfies conditions

            fallback_m_idx = handle_no_valid_step(m_idx, armijo_1, context)

            # Extract result
            fallback_m = fallback_m_idx.squeeze() + context.c_idx

            fallback_x = xs_1[fallback_m]
            fallback_c = cs_1[fallback_m]
            fallback_grad = g_x_1[fallback_m].view(b, context.opt_dim)
            idxs_fallback = (
                fallback_m.view(-1, 1)
                .repeat(1, context.line_search_scale.shape[1])
                .to(dtype=torch.int32)
            )

            exploration_state = LineSearchState(
                action=fallback_x.detach().view(b, context.action_horizon, context.action_dim),
                cost=fallback_c.detach(),
                gradient=fallback_grad.detach().view(b, context.action_horizon, context.action_dim),
                idxs=idxs_fallback,
            )

            return LineSearchResult(
                selected_state=selected_state,
                exploration_state=exploration_state,
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
    ) -> LineSearchResult:
        """Implement Wolfe search using CUDA kernels."""
        exploration_idx, selected_idx = self._get_cuda_kernel_output_buffers(context)

        # Call CUDA kernel
        is_strong_wolfe = isinstance(self, StrongWolfeLineSearchStrategy)
        is_approx_wolfe = isinstance(self, ApproxWolfeLineSearchStrategy)
        is_approx_strong_wolfe = isinstance(self, ApproxStrongWolfeLineSearchStrategy)
        if is_approx_strong_wolfe:
            log_and_raise("Approx strong Wolfe not implemented with cuda kernel")

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
        )

    @profiler.record_function("BaseWolfeLineSearchStrategy/_get_cuda_kernel_output_buffers")
    def _get_cuda_kernel_output_buffers(
        self, context: LineSearchContext
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare output buffers for CUDA line search."""
        if self._output_buffers is None:
            log_and_raise("Output buffers are not initialized")

        exploration_idx, selected_idx = self._output_buffers

        return exploration_idx, selected_idx

    @abstractmethod
    def _compute_curvature_condition(
        self, g_full_step: torch.Tensor, g_step: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Compute the curvature condition (Wolfe condition 2)."""
        pass

    @abstractmethod
    def _handle_no_valid_step(
        self, m_idx: torch.Tensor, wolfe_1: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Handle the case where no step satisfies the Wolfe conditions."""
        pass


class WolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Standard Wolfe line search.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * (g(x) * p)

    Wolfe condition 2:

    .. math::
        g(x + l * p) >= c_2 * g(x) * p
    """

    @staticmethod
    def _compute_curvature_condition(
        g_full_step: torch.Tensor, g_step: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Compute the standard Wolfe curvature condition."""
        wolfe_condition = g_full_step >= context.line_search_c_2 * g_step
        return wolfe_condition

    def _handle_no_valid_step(
        self, m_idx: torch.Tensor, armijo_1: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Fall back to armijo condition"""
        step_success = armijo_1 * (context.line_search_scale.squeeze(-1) + 0.1)
        _, armijo_idx = torch.max(step_success, dim=-2)
        m_idx = torch.where(m_idx == 0, armijo_idx, m_idx)
        return m_idx


class StrongWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Strong Wolfe line search with stricter curvature condition.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * (g(x) * p)

    Strong Wolfe condition 2:

    .. math::
        |g(x + l * p)| <= c_2 * |g(x) * p|
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
        m_idx: torch.Tensor, armijo_1: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """Don't fall back to Armijo condition."""
        return m_idx


class ApproxWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Approximate Wolfe line search with relaxed conditions.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * (g(x) * p)

    Strong Wolfe condition 2:

    .. math::
        |g(x + l * p)| <= c_2 * |g(x) * p|

    When strong wolfe condition fails, the line search falls back to condition 1. If
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
        m_idx: torch.Tensor, armijo_1: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """For approximate Wolfe, fallback to armijo condition, return step size at index 1."""
        step_success = armijo_1 * (context.line_search_scale.squeeze(-1) + 0.1)
        _, armijo_idx = torch.max(step_success, dim=-2)

        m_idx = torch.where(m_idx == 0, armijo_idx, m_idx)

        # Default to step size 1 if no conditions satisfied
        m_idx = torch.where(m_idx == 0, 1, m_idx)
        return m_idx


class ApproxStrongWolfeLineSearchStrategy(BaseWolfeLineSearchStrategy):
    """Approximate strong Wolfe line search with relaxed conditions.

    Given a step direction :math:`p`, a current point :math:`x`,
    the cost function :math:`c(x)`, and the gradient function :math:`g(x)`,
    we want to find the largest step size :math:`l` such that:

    Wolfe condition 1:

    .. math::
        c(x + l * p) <= c(x) + c_1 * l * (g(x) * p)

    Strong Wolfe condition 2:

    .. math::
        |g(x + l * p)| <= c_2 * |g(x) * p|

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
        m_idx: torch.Tensor, armijo_1: torch.Tensor, context: LineSearchContext
    ) -> torch.Tensor:
        """For approximate Wolfe, fallback to armijo condition, return step size at index 1."""
        step_success = armijo_1 * (context.line_search_scale.squeeze(-1) + 0.1)
        _, armijo_idx = torch.max(step_success, dim=-2)

        m_idx = torch.where(m_idx == 0, armijo_idx, m_idx)

        # Default to step size 1 if no conditions satisfied
        m_idx = torch.where(m_idx == 0, 1, m_idx)
        return m_idx


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
