# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L-SR1 optimizer using symmetric rank-1 updates for quasi-Newton step directions.

Composes GradientOptCore for gradient evaluation and lifecycle management with
QuasiNewtonBuffers for limited-memory (s, y) history. Uses rank-1 Hessian
approximation instead of the two-loop recursion used by L-BFGS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.util.logging import log_info
from curobo._src.util.torch_util import get_torch_jit_decorator


@get_torch_jit_decorator()
def jit_lsr1_compute_step_direction(
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    grad: torch.Tensor,
    m: int,
    epsilon: float,
    stable_mode: bool,
    hessian_0: torch.Tensor,
):
    """Compute the step direction using a limited-memory symmetric rank-1 (SR1) update.

    Args:
        y_buffer: (m, b, n, 1) past gradient differences.
        s_buffer: (m, b, n, 1) past step differences.
        grad: (b, 1, n) current gradient.
        m: number of stored pairs.
        epsilon: small positive value to guard against tiny denominators.
        stable_mode: if True, perform safeguards on the curvature denominator.
        hessian_0: (b, 1, 1) scaling applied as H0 = hessian_0 * I.

    Returns:
        p: (b, 1, n) search direction.
    """
    batch = grad.shape[0]
    n = grad.shape[-1]
    grad = grad.view(batch, 1, n)

    numerator = s_buffer[-1].transpose(-1, -2) @ y_buffer[-1]
    if stable_mode:
        numerator = torch.where(numerator <= 0.0, epsilon, numerator)
    denominator = y_buffer[-1].transpose(-1, -2) @ y_buffer[-1]
    var1 = numerator / torch.clamp(denominator, min=epsilon)
    if stable_mode:
        var1 = torch.nan_to_num(var1, epsilon, epsilon, epsilon)
    gamma = torch.nn.functional.relu(var1)

    Hg = hessian_0 * gamma * grad

    for i in range(m):
        h_y = hessian_0 * gamma * y_buffer[i].transpose(-1, -2)
        u = s_buffer[i].transpose(-1, -2) - h_y

        denom = u @ y_buffer[i]
        denom = torch.where(torch.abs(denom) < epsilon, epsilon, denom)

        numerator_term = u @ grad.transpose(-1, -2)
        Hg = Hg + (numerator_term / denom) * u

    r = -1.0 * Hg
    return r


class LSR1Opt:
    """L-SR1 optimizer with symmetric rank-1 Hessian approximation.

    Uses GradientOptCore for gradient evaluation and lifecycle management, and
    QuasiNewtonBuffers for (s, y) history. Computes step directions via a rank-1
    update of the inverse Hessian approximation.
    """

    @profiler.record_function("lsr1_opt/init")
    def __init__(
        self,
        config: LBFGSOptCfg,
        rollout_list: List[Rollout],
        use_cuda_graph: bool = False,
    ):
        self._qn = QuasiNewtonBuffers(config.device_cfg, config.history)
        self._hessian_0: Optional[torch.Tensor] = None

        # SR1 doesn't use a CUDA kernel for step direction
        config.use_cuda_kernel_step_direction = False

        opt_dim = rollout_list[0].action_horizon * rollout_list[0].action_dim
        if config.history > opt_dim:
            log_info("LSR1: history >= opt_dim, reducing history to opt_dim-1")
            config.history = opt_dim

        self._core = GradientOptCore(
            config,
            rollout_list,
            step_direction_fn=self._get_step_direction_impl,
            on_reinitialize=self._on_reinitialize,
            on_initial_state=self._on_initial_state,
            on_resize=self._on_resize,
            on_shift=self._on_shift,
            use_cuda_graph=use_cuda_graph,
        )
        self._core.update_num_problems(config.num_problems)
        self._core.finish_init()

    # -- SR1-specific step direction --

    @torch.no_grad()
    def _get_step_direction_impl(
        self, iteration_state: OptimizationIterationState
    ) -> torch.Tensor:
        q = iteration_state.exploration_action.view(-1, self._core.opt_dim)
        grad_q = iteration_state.exploration_gradient.view(-1, 1, self._core.opt_dim)

        self._qn.update(q, grad_q)

        dq = jit_lsr1_compute_step_direction(
            self._qn.y,
            self._qn.s,
            grad_q,
            self._core.config.history,
            self._core.config.epsilon,
            self._core.config.stable_mode,
            self._hessian_0,
        )
        return dq.view(-1, self._core.action_horizon, self._core.action_dim)

    # -- Hooks called by GradientOptCore --

    def _on_reinitialize(self, mask):
        self._qn.clear(mask)

    def _on_initial_state(self, iteration_state, mask):
        self._qn.set_reference(
            iteration_state.exploration_action.view(-1, self._core.opt_dim, 1),
            iteration_state.exploration_gradient.view(-1, self._core.opt_dim, 1),
            mask=mask,
        )

    def _on_resize(self, num_problems):
        self._qn.resize(num_problems, self._core.opt_dim)
        self._hessian_0 = torch.ones(
            (num_problems, 1, 1),
            device=self._core.device_cfg.device,
            dtype=self._core.device_cfg.dtype,
        )

    def _on_shift(self, shift_steps):
        self._qn.shift(shift_steps, self._core.action_dim)

    # -- Protocol: delegate to core --

    @property
    def config(self):
        return self._core.config

    @property
    def device_cfg(self):
        return self._core.device_cfg

    @property
    def opt_dt(self):
        return self._core.opt_dt

    @opt_dt.setter
    def opt_dt(self, value):
        self._core.opt_dt = value

    @property
    def use_cuda_graph(self):
        return self._core.use_cuda_graph

    @property
    def enabled(self):
        return self._core.enabled

    def enable(self):
        self._core.enable()

    def disable(self):
        self._core.disable()

    @property
    def action_horizon(self):
        return self._core.action_horizon

    @property
    def action_dim(self):
        return self._core.action_dim

    @property
    def opt_dim(self):
        return self._core.opt_dim

    @property
    def outer_iters(self):
        return self._core.outer_iters

    @property
    def horizon(self):
        return self._core.horizon

    @property
    def action_bound_lows(self):
        return self._core.action_bound_lows

    @property
    def action_bound_highs(self):
        return self._core.action_bound_highs

    @property
    def action_step_max(self):
        return self._core.action_step_max

    @property
    def action_horizon_step_max(self):
        return self._core.action_horizon_step_max

    @property
    def action_horizon_bounds_lows(self):
        return self._core.action_horizon_bounds_lows

    @property
    def action_horizon_bounds_highs(self):
        return self._core.action_horizon_bounds_highs

    @property
    def solve_time(self):
        return self._core.solve_time

    @property
    def solver_names(self):
        return self._core.solver_names

    @property
    def rollout_fn(self):
        return self._core.rollout_fn

    @property
    def _rollout_list(self):
        return self._core._rollout_list

    @property
    def _graphable_methods(self):
        return self._core._graphable_methods

    @property
    def _executors(self):
        return self._core._executors

    @_executors.setter
    def _executors(self, value):
        self._core._executors = value

    def optimize(self, seed_action):
        """Run the full optimization loop and return the best action sequence."""
        return self._core.optimize(seed_action)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        """Reset optimizer state and seed with new actions before a fresh solve."""
        return self._core.reinitialize(action, mask, clear_optimizer_state, reset_num_iters)

    def shift(self, shift_steps=0):
        return self._core.shift(shift_steps)

    def _shift(self, shift_steps=0):
        return self._core._shift(shift_steps)

    def update_num_problems(self, num_problems):
        """Resize internal buffers to accommodate a new number of parallel problems."""
        return self._core.update_num_problems(num_problems)

    def update_rollout_params(self, goal):
        return self._core.update_rollout_params(goal)

    def update_goal_dt(self, goal):
        return self._core.update_goal_dt(goal)

    def get_all_rollout_instances(self):
        return self._core.get_all_rollout_instances()

    def compute_metrics(self, action):
        """Evaluate cost and constraint metrics for the given action sequence."""
        return self._core.compute_metrics(action)

    def reset_shape(self):
        return self._core.reset_shape()

    def reset_seed(self):
        return self._core.reset_seed()

    def reset_cuda_graph(self):
        return self._core.reset_cuda_graph()

    def get_recorded_trace(self):
        return self._core.get_recorded_trace()

    def update_solver_params(self, solver_params):
        return self._core.update_solver_params(solver_params)

    def update_niters(self, niters):
        return self._core.update_niters(niters)

    def debug_dump(self, file_path=""):
        return self._core.debug_dump(file_path)
