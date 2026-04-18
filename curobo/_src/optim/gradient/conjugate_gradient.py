# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nonlinear Conjugate Gradient optimizer with Fletcher-Reeves, Polak-Ribiere, and Dai-Yuan.

Composes GradientOptCore for gradient evaluation and lifecycle management with
conjugate-direction step computation. Supports line search, CUDA graph acceleration,
and configurable beta-update method (FR, PR, or DY).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.line_search_strategy import LineSearchType
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import shift_buffer
from curobo._src.util.torch_util import get_torch_jit_decorator


@get_torch_jit_decorator()
def jit_cg_compute_step_direction(
    grad: torch.Tensor,
    prev_grad: torch.Tensor,
    prev_step: torch.Tensor,
    max_beta: float,
    method: str,
):
    """Compute the conjugate gradient step direction using the specified beta method.

    Calculates the conjugate direction by combining the current negative gradient
    with the previous step direction scaled by a beta coefficient. Supports
    Fletcher-Reeves (FR), Polak-Ribiere (PR), and Dai-Yuan (DY) update rules.

    Args:
        grad: (b, 1, n) current gradient.
        prev_grad: (b, 1, n) gradient from the previous iteration (updated in-place).
        prev_step: (b, 1, n) step direction from the previous iteration
            (updated in-place).
        max_beta: maximum allowed value for the beta coefficient.
        method: one of ``"FR"``, ``"PR"``, or ``"DY"``.

    Returns:
        Tuple of (step, prev_grad, prev_step) where step is (b, 1, n) search
        direction and the buffers are updated in-place for the next iteration.
    """
    batch = grad.shape[0]
    n = grad.shape[-1]

    # ``method`` is validated upstream in ``ConjugateGradientOptCfg.__post_init__``
    # via ``log_and_raise``; the ``else`` branch handles "DY" and keeps ``beta``
    # defined on every control-flow path (required by TorchScript).
    if method == "FR":  # Fletcher-Reeves
        beta = (grad @ grad.transpose(-1, -2)) / (
            prev_grad @ prev_grad.transpose(-1, -2)
        )
    elif method == "PR":  # Polak-Ribiere
        beta = (grad @ (grad - prev_grad).transpose(-1, -2)) / (
            prev_grad @ prev_grad.transpose(-1, -2)
        )
    else:  # method == "DY" — Dai-Yuan
        beta = (grad @ grad.transpose(-1, -2)) / (
            -prev_step @ (grad - prev_grad).transpose(-1, -2)
        )

    beta = torch.nan_to_num(beta, 0.0)
    beta = torch.clamp(beta, 0, max_beta)
    step = -1.0 * grad + beta * prev_step

    step = step.view(batch, 1, n)
    prev_step.copy_(step)
    prev_grad.copy_(grad)
    return step, prev_grad, prev_step


@get_torch_jit_decorator()
def jit_cg_shift_buffers(
    prev_grad,
    prev_step,
    shift_steps: int,
    action_dim: int,
):
    """Shift CG history buffers for MPC warm-start by rolling action dimensions.

    Rolls the gradient and step buffers left by ``shift_steps * action_dim``
    elements and zero-fills the trailing entries, keeping the conjugate
    direction consistent after the planning horizon advances.

    Args:
        prev_grad: (b, 1, n) previous gradient buffer (updated in-place).
        prev_step: (b, 1, n) previous step direction buffer (updated in-place).
        shift_steps: number of time steps to shift forward.
        action_dim: number of action dimensions per time step.

    Returns:
        Tuple of (prev_grad, prev_step) with shifted contents.
    """
    shift_d = shift_steps * action_dim
    prev_grad.copy_(shift_buffer(prev_grad, shift_d, action_dim, shift_steps))
    prev_step.copy_(shift_buffer(prev_step, shift_d, action_dim, shift_steps))
    return prev_grad, prev_step


@dataclass
class ConjugateGradientOptCfg:
    """Flat configuration for Conjugate Gradient optimizer."""

    # General
    num_iters: int = 100
    solver_type: str = "conjugate_gradient"
    solver_name: str = "conjugate_gradient"
    device_cfg: DeviceCfg = DeviceCfg()
    store_debug: bool = False
    debug_info: Any = None
    num_problems: int = 1
    num_particles: Optional[int] = None
    sync_cuda_time: bool = True
    use_coo_sparse: bool = True
    step_scale: float = 1.0
    inner_iters: int = 25
    _num_rollout_instances: int = 1

    # Convergence
    cost_convergence: float = 1.0e-11
    cost_delta_threshold: float = 0.0
    cost_relative_threshold: float = 0.0
    converged_ratio: float = 0.8
    fixed_iters: bool = True
    convergence_iteration: int = 0
    minimum_iters: Optional[int] = None
    return_best_action: bool = True

    # Line search
    line_search_scale: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.7, 1.0])
    line_search_type: LineSearchType = LineSearchType.APPROX_WOLFE
    use_cuda_kernel_line_search: bool = True
    fix_terminal_action: bool = False
    line_search_wolfe_c_1: float = 1e-5
    line_search_wolfe_c_2: float = 0.9
    initial_step_scale: float = 0.1

    # CG-specific
    #: Conjugate direction update rule. Valid values are ``"FR"``
    #: (Fletcher-Reeves), ``"PR"`` (Polak-Ribiere), and ``"DY"``
    #: (Dai-Yuan).
    cg_method: str = "FR"
    #: Upper clamp for the conjugate direction coefficient beta. Prevents
    #: the step direction from becoming excessively large when gradient
    #: norms change rapidly between iterations.
    max_beta: float = 10.0

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = len(self.line_search_scale)
        self.line_search_type = LineSearchType(self.line_search_type)
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0
        if self.cost_relative_threshold >= 1.0:
            log_and_raise("cost_relative_threshold must be less than 1.0")
        if self.cg_method not in ["FR", "PR", "DY"]:
            log_and_raise(
                f"Invalid cg_method: {self.cg_method}, choose from ['FR', 'PR', 'DY']"
            )

    @property
    def num_rollout_instances(self):
        return self._num_rollout_instances

    @property
    def outer_iters(self):
        return math.ceil(self.num_iters / self.inner_iters)

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict["device_cfg"] = device_cfg
        if "num_particles" not in child_dict:
            child_dict["num_particles"] = None
        dataclass_field_names = {f.name for f in fields(cls)}
        for k in [k for k in child_dict if k not in dataclass_field_names]:
            child_dict.pop(k)
        return child_dict

    def update_niters(self, niters: int):
        self.num_iters = niters


class ConjugateGradientOpt:
    """Nonlinear Conjugate Gradient optimizer for cuRobo trajectory optimization.

    Computes conjugate step directions using Fletcher-Reeves, Polak-Ribiere, or
    Dai-Yuan beta updates. Uses GradientOptCore for gradient evaluation, line
    search, best-solution tracking, and CUDA graph lifecycle.
    """

    @profiler.record_function("conjugate_gradient/init")
    def __init__(
        self,
        config: ConjugateGradientOptCfg,
        rollout_list: List[Rollout],
        use_cuda_graph: bool = False,
    ):
        self._prev_step: Optional[torch.Tensor] = None
        self._prev_grad_q: Optional[torch.Tensor] = None

        self._core = GradientOptCore(
            config,
            rollout_list,
            step_direction_fn=self._get_step_direction_impl,
            on_reinitialize=self._on_reinitialize,
            on_shift=self._on_shift,
            use_cuda_graph=use_cuda_graph,
        )
        self._core.update_num_problems(config.num_problems)
        self._core.finish_init()

    # -- CG-specific step direction --

    def _get_step_direction_impl(self, iteration_state: OptimizationIterationState):
        grad_q = iteration_state.exploration_gradient.view(-1, 1, self._core.opt_dim)
        if self._prev_grad_q is None:
            self._prev_grad_q = grad_q.clone()
        if self._prev_step is None:
            self._prev_step = -1.0 * grad_q.clone()

        dq, self._prev_grad_q, self._prev_step = jit_cg_compute_step_direction(
            grad_q,
            self._prev_grad_q,
            self._prev_step,
            self._core.config.max_beta,
            self._core.config.cg_method,
        )
        return dq.view(-1, self._core.action_horizon, self._core.action_dim)

    # -- Hooks --

    def _on_reinitialize(self, mask):
        self._prev_step = None
        self._prev_grad_q = None

    def _on_shift(self, shift_steps):
        if self._prev_grad_q is not None and self._prev_step is not None:
            self._prev_grad_q, self._prev_step = jit_cg_shift_buffers(
                self._prev_grad_q,
                self._prev_step,
                shift_steps,
                self._core.action_dim,
            )

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
