# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scaled gradient descent optimizer for cuRobo trajectory optimization.

Applies step = -scale * gradient each iteration with no line search. Tracks
the best solution per problem via BestTracker and supports CUDA graph acceleration.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.autograd.profiler as profiler

import curobo._src.runtime as curobo_runtime
from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.best_tracker import BestTracker
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.gradient.update_best_solution import update_best_solution
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import check_nan_last_dimension


@dataclass
class GradientDescentOptCfg:
    """Flat configuration for Gradient Descent optimizer."""

    # General
    num_iters: int = 100
    solver_type: str = "gradient_descent"
    solver_name: str = "gradient_descent"
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

    # GD-specific
    #: Fixed multiplier applied to the negative gradient each iteration
    #: (``step = -gradient_descent_step_scale * gradient``). Acts as the
    #: learning rate since this optimizer does not use line search.
    gradient_descent_step_scale: float = 0.001

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0
        if self.cost_relative_threshold >= 1.0:
            log_and_raise("cost_relative_threshold must be less than 1.0")

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


class GradientDescentOpt:
    """Scaled gradient descent optimizer with per-problem best-solution tracking.

    Applies step = -gradient_descent_step_scale * gradient each iteration. Uses
    BestTracker for convergence detection and supports CUDA graph acceleration.
    """

    _graphable_methods: set = {"_opt_iters"}

    @profiler.record_function("gradient_descent/init")
    def __init__(
        self,
        config: GradientDescentOptCfg,
        rollout_list: List[Rollout],
        use_cuda_graph: bool = False,
    ):
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

        self.rollout_fn = rollout_list[0]
        self._rollout_list = rollout_list
        self.rollout_fn.sum_horizon = True

        self._bounds = ActionBounds(
            self.rollout_fn.action_bound_lows,
            self.rollout_fn.action_bound_highs,
            self.action_horizon,
            config.step_scale,
        )
        self._best = BestTracker(config.device_cfg)
        self._debug = DebugRecorder() if config.store_debug else None

        self._l_vec = None
        self._executors: Dict[str, Optional[GraphExecutor]] = {}

        self.update_num_problems(config.num_problems)

        if use_cuda_graph:
            self._executors["_opt_iters"] = create_graph_executor(
                capture_fn=self._opt_iters,
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
    def action_horizon(self):
        return self.rollout_fn.action_horizon

    @property
    def action_dim(self):
        return self.rollout_fn.action_dim

    @property
    def opt_dim(self):
        return self.action_horizon * self.action_dim

    @property
    def outer_iters(self):
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
    def solve_time(self):
        return self.opt_dt

    @property
    def solver_names(self):
        return [self.config.solver_name]

    # -- Core algorithm --

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        timer = CudaEventTimer().start()
        out = self._optimize_impl(seed_action)
        self.opt_dt = timer.stop()
        return out

    def _optimize_impl(self, seed_action: torch.Tensor) -> torch.Tensor:
        q = seed_action.view(self.config.num_problems, self.action_horizon, self.action_dim)

        if self._iteration_state is not None:
            iteration_state = self._iteration_state
            self._iteration_state = None
        else:
            iteration_state = self._prepare_initial_iteration_state(q)

        self._record_iteration_state(iteration_state)

        for _ in range(self.outer_iters):
            iteration_state = self._dispatch("_opt_iters", iteration_state)
            if not self.config.fixed_iters:
                if BestTracker.check_convergence(
                    iteration_state.converged, self.config.converged_ratio
                ):
                    break

        q = iteration_state.action.view(self.config.num_problems, self.opt_dim)
        best_q = iteration_state.best_action.view(self.config.num_problems, self.opt_dim)

        if not self.config.return_best_action:
            nan_labels = check_nan_last_dimension(q).view(self.config.num_problems, 1)
            q_best = torch.where(nan_labels, best_q, q)
        else:
            q_best = best_q.clone()
        return q_best.view(self.config.num_problems, self.action_horizon, self.action_dim)

    def _dispatch(self, method_name, *args):
        executor = self._executors.get(method_name)
        if executor is not None and self.use_cuda_graph:
            return executor(*args)
        return getattr(self, method_name)(*args)

    def _opt_iters(self, iteration_state):
        for _ in range(self.config.inner_iters):
            iteration_state = self._opt_step(iteration_state)
            self._record_iteration_state(iteration_state)
        return iteration_state

    @profiler.record_function("gradient_descent/opt_step")
    def _opt_step(self, iteration_state):
        action_next = iteration_state.action + iteration_state.step_direction
        cost_next, gradient_next = self._compute_cost_and_gradient(action_next)

        next_state = OptimizationIterationState(
            action=action_next.view(self.config.num_problems, self.action_horizon, self.action_dim),
            gradient=gradient_next.view(self.config.num_problems, self.action_horizon, self.action_dim),
            exploration_gradient=gradient_next.view(self.config.num_problems, self.action_horizon, self.action_dim),
            cost=cost_next.view(self.config.num_problems),
            best_cost=iteration_state.best_cost,
            best_action=iteration_state.best_action,
            best_iteration=iteration_state.best_iteration,
            current_iteration=iteration_state.current_iteration,
            converged=iteration_state.converged,
        )

        next_state = update_best_solution(
            next_state, self.action_horizon, self.action_dim,
            self.config.cost_delta_threshold, self.config.cost_relative_threshold,
            self.config.convergence_iteration,
        )

        step_direction = (
            -self.config.gradient_descent_step_scale
            * next_state.exploration_gradient
        )
        next_state.step_direction = step_direction
        return next_state

    def _prepare_initial_iteration_state(self, action):
        grad_q = action.detach().clone() * 0.0
        step_direction = -self.config.gradient_descent_step_scale * grad_q
        return OptimizationIterationState(
            action=action.clone(),
            gradient=grad_q.clone(),
            cost=self._best.cost.clone(),
            best_cost=self._best.cost.clone(),
            best_action=self._best.action.clone(),
            best_iteration=self._best.iteration,
            current_iteration=self._best.current_iteration,
            step_direction=step_direction,
            exploration_action=action.clone(),
            exploration_gradient=grad_q.clone(),
            exploration_cost=self._best.cost.clone(),
            converged=self._best.converged,
        )

    @profiler.record_function("gradient_descent/cost_and_gradient")
    def _compute_cost_and_gradient(self, x):
        x_n = x.detach().requires_grad_(True)
        x_in = x_n.view(
            self.config.num_problems * self.config.num_particles,
            self.action_horizon, self.action_dim,
        )
        trajectories = self.rollout_fn.evaluate_action(x_in, use_cuda_graph=False)
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
        if costs.shape != (self.config.num_problems * self.config.num_particles,):
            log_and_raise(f"costs.shape mismatch: {costs.shape}")
        cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        cost.backward(gradient=self._l_vec, retain_graph=False)
        return cost, x_n.grad.detach()

    # -- Lifecycle --

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        action = action.view(self.config.num_problems, self.action_horizon, self.action_dim)
        if mask is None and self._debug:
            self._debug.clear()
        if reset_num_iters:
            self.config.num_iters = self._og_num_iters
        self._best.clear(mask)
        iteration_state = self._prepare_initial_iteration_state(action)
        if self._iteration_state is None:
            self._iteration_state = iteration_state
        else:
            self._iteration_state.copy_(iteration_state)

    def shift(self, shift_steps=0):
        return self._shift(shift_steps)

    def _shift(self, shift_steps=0):
        if shift_steps == 0:
            return True
        self._best.reset()
        return True

    def update_num_problems(self, num_problems):
        assert num_problems > 0
        self.config.num_problems = num_problems
        self._best.resize(num_problems, self.action_horizon, self.action_dim)
        self._l_vec = torch.ones(
            (num_problems, self.config.num_particles, 1),
            device=self.device_cfg.device, dtype=self.device_cfg.dtype,
        )
        for rollout in self._rollout_list:
            rollout.update_batch_size(batch_size=num_problems * self.config.num_particles)

    def update_rollout_params(self, goal):
        for rollout in self._rollout_list:
            rollout.update_params(goal, num_particles=self.config.num_particles)

    def update_goal_dt(self, goal):
        for rollout in self._rollout_list:
            rollout.update_goal_dt(goal)

    def get_all_rollout_instances(self):
        return self._rollout_list

    def compute_metrics(self, action):
        return self.rollout_fn.compute_metrics_from_action(action)

    def reset_shape(self):
        for rollout in self._rollout_list:
            rollout.reset_shape()

    def reset_seed(self):
        return True

    def reset_cuda_graph(self):
        for executor in self._executors.values():
            if executor is not None:
                executor.reset()
        if hasattr(self.rollout_fn, "reset_cuda_graph"):
            self.rollout_fn.reset_cuda_graph()

    def get_recorded_trace(self):
        if self._debug:
            return self._debug.get_trace()
        return {"debug": [], "debug_cost": []}

    def update_solver_params(self, solver_params):
        if self.config.solver_name not in solver_params:
            log_and_raise(f"Optimizer {self.config.solver_name} not found in {solver_params}")
        for k, v in solver_params[self.config.solver_name].items():
            setattr(self.config, k, v)
        return True

    def update_niters(self, niters):
        self.config.update_niters(niters)

    def _record_iteration_state(self, iteration_state):
        if self._debug:
            self._debug.record(iteration_state, self.action_horizon, self.action_dim)

    def debug_dump(self, file_path=""):
        pass


# Keep for backward compat: unused, broken, will be removed in step 1.10
class LineSearchGradientDescentOpt(GradientDescentOpt):
    """NOTE: This class is not used. Does not work as expected."""

    pass
