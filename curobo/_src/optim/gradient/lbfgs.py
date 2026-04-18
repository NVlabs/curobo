# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""L-BFGS optimizer using two-loop recursion for quasi-Newton step directions.

Composes GradientOptCore for gradient evaluation and lifecycle management with
QuasiNewtonBuffers for limited-memory (s, y) history. Supports line search,
CUDA graph acceleration, and optional cuRobo CUDA kernels for the two-loop step.
"""

from __future__ import annotations

# Standard Library
import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

from curobo._src.curobolib.cuda_ops.optimization import LBFGScu
from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.optim.gradient.lbfgs_jit_helpers import (
    jit_lbfgs_compute_step_direction,
)
from curobo._src.optim.gradient.line_search_strategy import LineSearchType
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_info

__all__ = ["LBFGSOptCfg", "LBFGSOpt"]


@dataclass
class LBFGSOptCfg:
    """Flat configuration for L-BFGS optimizer. All fields in one place."""

    # General
    num_iters: int = 100
    solver_type: str = "lbfgs"
    solver_name: str = "lbfgs"
    device_cfg: DeviceCfg = DeviceCfg()
    store_debug: bool = False
    debug_info: Any = None
    num_problems: int = 1
    num_particles: Optional[int] = None
    sync_cuda_time: bool = True
    use_coo_sparse: bool = True
    step_scale: float = 1.0
    inner_iters: int = 25
    _num_rollout_instances: int = 2

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
    #: Sufficient-decrease constant for the Wolfe line search (Armijo
    #: condition). The step is accepted when cost decrease exceeds
    #: ``c_1 * step_size * directional_derivative``. Very small values
    #: (default 1e-5) make the condition easy to satisfy.
    line_search_wolfe_c_1: float = 1e-5
    #: Curvature condition constant for the Wolfe line search. The step is
    #: accepted when the new directional derivative is at least
    #: ``c_2 * old_directional_derivative``. Values close to 1.0 are
    #: typical for quasi-Newton methods like L-BFGS.
    line_search_wolfe_c_2: float = 0.9

    # L-BFGS specific
    history: int = 7
    #: Damping constant added to the denominator of the two-loop recursion
    #: to prevent division by near-zero values in the Hessian approximation.
    #: Larger values bias the step direction toward steepest descent.
    epsilon: float = 0.01
    use_cuda_kernel_step_direction: bool = True
    #: When True, uses a numerically stable variant of the L-BFGS two-loop
    #: recursion that guards against NaN/Inf in the (s, y) history buffers.
    #: Must be True; the unstable path is disabled.
    stable_mode: bool = True
    #: When True, the CUDA kernel for the two-loop recursion uses shared
    #: memory for intermediate buffers, giving a significant speed-up on
    #: small-to-medium opt_dim. Automatically disabled when the shared
    #: memory requirement exceeds the hardware limit.
    use_cuda_kernel_shared_buffers: bool = True
    initial_step_scale: float = 0.1

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = len(self.line_search_scale)
        self.line_search_type = LineSearchType(self.line_search_type)
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0
        if self.cost_relative_threshold >= 1.0:
            log_and_raise("cost_relative_threshold must be less than 1.0")
        if not self.stable_mode:
            log_and_raise("LBFGS: stable_mode must be true")
        if self._num_rollout_instances != 2:
            log_and_raise("LBFGS: _num_rollout_instances must be 2")

    @property
    def num_rollout_instances(self):
        return self._num_rollout_instances

    @property
    def outer_iters(self):
        outer_iters = math.ceil(self.num_iters / self.inner_iters)
        if outer_iters <= 0:
            log_and_raise(
                f"outer_iters {outer_iters} <= 0, consider setting num_iters to a multiple"
                f" of inner_iters: {self.inner_iters} and greater than inner_iters"
            )
        return outer_iters

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        """Create config dict, filtering to only valid fields."""
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
        if self.num_iters % self.inner_iters != 0:
            log_and_raise(
                f"num_iters {self.num_iters} is not a multiple of inner_iters {self.inner_iters}, "
                "consider setting num_iters to a multiple of inner_iters"
            )
        if self.num_iters < self.inner_iters:
            log_and_raise(
                f"num_iters {self.num_iters} is less than inner_iters {self.inner_iters}, "
                "consider setting num_iters to a multiple of inner_iters"
            )


class LBFGSOpt:
    """L-BFGS optimizer with limited-memory two-loop recursion step directions.

    Uses GradientOptCore for gradient evaluation and lifecycle management, and
    QuasiNewtonBuffers for (s, y) history. Supports Wolfe and strong-Wolfe line
    search, CUDA graph wrapping, and an optional CUDA kernel fast path.
    """

    @profiler.record_function("lbfgs_opt/init")
    def __init__(
        self,
        config: LBFGSOptCfg,
        rollout_list: List[Rollout],
        use_cuda_graph: bool = False,
    ):
        self._qn = QuasiNewtonBuffers(config.device_cfg, config.history)

        # Check CUDA kernel feasibility for step direction
        opt_dim = rollout_list[0].action_horizon * rollout_list[0].action_dim
        shared_memory_needed = (((2 * opt_dim) + 2) * config.history + 32 + 1) * 4
        max_shared_memory = 65536
        if opt_dim >= 1024 or config.history > 31 or shared_memory_needed > max_shared_memory:
            if shared_memory_needed > max_shared_memory:
                log_info(
                    f"LBFGS: Not using CUDA kernel - shared memory requirement "
                    f"({shared_memory_needed} bytes) exceeds hardware limit "
                    f"({max_shared_memory} bytes) for opt_dim={opt_dim}, "
                    f"history={config.history}"
                )
            else:
                log_info("LBFGS: Not using LBFGS Cuda Kernel as opt_dim>=1024 or history>31")
            config.use_cuda_kernel_step_direction = False

        if config.history > opt_dim:
            log_info("LBFGS: history >= opt_dim, reducing history to opt_dim-1")
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

    # -- LBFGS-specific step direction --

    @torch.no_grad()
    def _get_step_direction_impl(
        self, iteration_state: OptimizationIterationState
    ) -> torch.Tensor:
        q = iteration_state.exploration_action.view(-1, self._core.opt_dim)
        grad_q = iteration_state.exploration_gradient.view(-1, 1, self._core.opt_dim)

        if self._core.config.use_cuda_kernel_step_direction:
            with profiler.record_function("lbfgs/fused"):
                dq = LBFGScu.apply(
                    self._qn.step_q_buffer,
                    self._qn.rho,
                    self._qn.y,
                    self._qn.s,
                    q,
                    grad_q,
                    self._qn.x_0,
                    self._qn.grad_0,
                    self._core.config.epsilon,
                    self._core.config.stable_mode,
                    self._core.config.use_cuda_kernel_shared_buffers,
                )
        else:
            self._update_buffers(q, grad_q)
            dq = jit_lbfgs_compute_step_direction(
                self._qn.alpha,
                self._qn.rho,
                self._qn.y,
                self._qn.s,
                grad_q,
                self._core.config.history,
                self._core.config.epsilon,
                self._core.config.stable_mode,
            )

        return dq.view(-1, self._core.action_horizon, self._core.action_dim)

    def _update_buffers(self, q: torch.Tensor, grad_q: torch.Tensor):
        """Update quasi-Newton history buffers."""
        self._qn.update(q, grad_q)

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
