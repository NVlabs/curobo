# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SciPy-backed optimizer that evaluates costs on GPU and minimizes on CPU."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import OptimizeResult as ScipyOptimizeResult
from scipy.optimize import minimize

from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise

__all__ = ["ScipyOpt", "ScipyOptCfg", "CudaGraphScipyOpt"]


@dataclass
class ScipyOptCfg:
    """Flat configuration for scipy optimizer."""

    # General
    num_iters: int = 100
    solver_type: str = "scipy"
    solver_name: str = "scipy"
    device_cfg: DeviceCfg = DeviceCfg()
    store_debug: bool = False
    debug_info: Any = None
    num_problems: int = 1
    num_particles: Optional[int] = None
    sync_cuda_time: bool = True
    use_coo_sparse: bool = True
    step_scale: float = 1.0
    inner_iters: int = 1
    _num_rollout_instances: int = 1

    # Scipy-specific
    #: Method name passed to ``scipy.optimize.minimize`` (e.g. ``"SLSQP"``,
    #: ``"L-BFGS-B"``, ``"COBYLA"``). Constraint-aware methods like SLSQP
    #: receive cuRobo constraints as inequality constraints.
    scipy_minimize_method: str = "SLSQP"
    scipy_minimize_kwargs: dict = field(default_factory=dict)
    #: When True, action vectors are converted to float64 before sending to
    #: SciPy on CPU. Required for SLSQP (automatically enabled). Reduces
    #: numerical issues in SciPy's Fortran routines at the cost of an extra
    #: dtype conversion round-trip.
    use_float64_on_cpu: bool = False

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.scipy_minimize_method == "SLSQP" and not self.use_float64_on_cpu:
            self.use_float64_on_cpu = True

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


class ScipyOpt:
    """SciPy optimizer that computes costs and gradients on GPU via cuRobo rollouts.

    Evaluates the objective, constraints, and gradients on GPU, copies them to CPU
    for ``scipy.optimize.minimize``. When ``use_cuda_graph=True``, GPU evaluations
    are wrapped in CUDA graph executors for faster repeated evaluation.
    """

    _graphable_methods: set = set()

    def __init__(
        self,
        config: ScipyOptCfg,
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
        self._debug = DebugRecorder() if config.store_debug else None

        self._bounds = ActionBounds(
            self.rollout_fn.action_bound_lows,
            self.rollout_fn.action_bound_highs,
            self.action_horizon,
            config.step_scale,
        )

        bounds_array = torch.cat(
            (
                self._bounds.horizon_lows.view(1, -1),
                self._bounds.horizon_highs.view(1, -1),
            ),
            dim=0,
        ).cpu().numpy()
        self._scipy_action_bounds = [
            (float(bounds_array[0, i]), float(bounds_array[1, i]))
            for i in range(bounds_array.shape[1])
        ]

        self._opt_init = None
        self.l_vec = None
        self._executors: Dict[str, Optional[GraphExecutor]] = {}

        # CUDA graph executors for GPU evaluation functions (created lazily)
        self._cost_constraint_grad_executor: Optional[GraphExecutor] = None
        self._constraint_executor: Optional[GraphExecutor] = None
        self._constraint_grad_executor: Optional[GraphExecutor] = None

        self.update_num_problems(config.num_problems)

    # -- Properties --

    @property
    def enabled(self):
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
    def action_horizon_bounds_lows(self):
        return self._bounds.horizon_lows

    @property
    def action_horizon_bounds_highs(self):
        return self._bounds.horizon_highs

    @property
    def solve_time(self):
        return self.opt_dt

    @property
    def solver_names(self):
        return [self.config.solver_name]

    # -- Core --

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        timer = CudaEventTimer().start()
        out = self._optimize_impl(seed_action)
        self.opt_dt = timer.stop()
        return out

    def _optimize_impl(self, seed_action: torch.Tensor) -> torch.Tensor:
        seed_action = seed_action.view(self.config.num_problems, self.opt_dim)
        self._opt_init.requires_grad = False
        self._opt_init.copy_(seed_action)
        self._opt_init.requires_grad = True
        q = self._opt_init

        if self.config.use_float64_on_cpu:
            q_np = q.detach().cpu().to(torch.float64).view(-1).numpy()
        else:
            q_np = q.detach().cpu().view(-1).numpy()

        options = {}
        if self.config.scipy_minimize_method in ["L-BFGS-B"]:
            options = {"maxiter": self.config.num_iters, "ftol": 1e-8, "gtol": 1e-5}
            if self.config.store_debug:
                options["disp"] = True
        elif self.config.scipy_minimize_method in ["SLSQP"]:
            options = {"maxiter": self.config.num_iters}
            if self.config.store_debug:
                options["disp"] = True
        else:
            options = {"maxiter": self.config.num_iters}

        if self.config.scipy_minimize_method in [
            "SLSQP", "COBYLA", "COBYQA", "trust-constr"
        ]:
            constraint = None
            if self._has_constraints():
                constraint = {
                    "type": "ineq",
                    "fun": self._constraint_fn,
                    "jac": self._constraint_gradient_fn,
                }
            opt_result = minimize(
                fun=self._cost_and_gradient_fn,
                x0=q_np,
                callback=self._iteration_callback if self.config.store_debug else None,
                method=self.config.scipy_minimize_method,
                jac=True,
                bounds=self._scipy_action_bounds,
                options=options,
                constraints=constraint,
                **self.config.scipy_minimize_kwargs,
            )
        else:
            opt_result = minimize(
                fun=self._cost_constraint_and_gradient_fn,
                x0=q_np,
                callback=self._iteration_callback if self.config.store_debug else None,
                method=self.config.scipy_minimize_method,
                jac=True,
                bounds=self._scipy_action_bounds,
                options=options,
                **self.config.scipy_minimize_kwargs,
            )

        q_best = torch.as_tensor(
            opt_result.x, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        return q_best.view(self.config.num_problems, self.action_horizon, self.action_dim)

    # -- GPU evaluation --

    def _gpu_to_cpu_wrapper(self, gpu_fn, x_np: np.ndarray):
        input_dtype = self.device_cfg.dtype
        if self.config.use_float64_on_cpu and x_np.dtype == np.float64:
            input_dtype = torch.float64
        x = torch.as_tensor(x_np, device=self.device_cfg.device, dtype=input_dtype)
        result = gpu_fn(x)
        if result is None:
            return None
        elif isinstance(result, tuple):
            if self.config.use_float64_on_cpu:
                return tuple(r.cpu().to(torch.float64).numpy() for r in result)
            return tuple(r.cpu().numpy() for r in result)
        else:
            if self.config.use_float64_on_cpu:
                return result.cpu().to(torch.float64).numpy()
            return result.cpu().numpy()

    def _prepare_action_for_rollout(self, x_gpu):
        x_n = x_gpu.detach().requires_grad_(True)
        x_in = x_n.view(
            self.config.num_problems * self.config.num_particles,
            self.action_horizon, self.rollout_fn.action_dim,
        )
        return x_n, x_in

    def _evaluate_trajectory(self, x_in, get_constraint=False, get_cost=True):
        trajectories = self.rollout_fn.evaluate_action(x_in)
        cost = None
        if get_cost:
            costs = trajectories.costs_and_constraints.get_sum_cost(
                sum_horizon=True, include_all_hybrid=True,
            )
            if len(costs.shape) == 2:
                cost = torch.sum(
                    costs.view(self.config.num_problems, self.config.num_particles, self.horizon),
                    dim=-1, keepdim=True,
                )
            else:
                cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        constraint = None
        if get_constraint:
            if trajectories.costs_and_constraints.constraints is not None:
                constraints = trajectories.costs_and_constraints.get_sum_constraint(
                    sum_horizon=True, include_all_hybrid=False,
                )
            else:
                constraints = None
            if constraints is not None:
                if len(constraints.shape) == 2:
                    constraint = torch.sum(
                        constraints.view(self.config.num_problems, self.config.num_particles, self.horizon),
                        dim=-1, keepdim=True,
                    )
                else:
                    constraint = constraints.view(self.config.num_problems, self.config.num_particles, 1)
                constraint = -1.0 * constraint
        return trajectories, cost, constraint

    def _compute_gradient(self, value, x_n, negate=False):
        value.backward(gradient=self.l_vec, retain_graph=False)
        grad = x_n.grad.detach()
        if negate:
            grad = -1.0 * grad
        return grad

    def _cost_constraint_and_gradient_fn_gpu(self, x_gpu):
        if self.use_cuda_graph:
            if self._cost_constraint_grad_executor is None:
                self._cost_constraint_grad_executor = create_graph_executor(
                    capture_fn=self._cost_constraint_and_gradient_fn_gpu_impl,
                    device=self.device_cfg.device,
                )
            return self._cost_constraint_grad_executor(x_gpu, clone_outputs=True)
        return self._cost_constraint_and_gradient_fn_gpu_impl(x_gpu)

    def _cost_constraint_and_gradient_fn_gpu_impl(self, x_gpu):
        x_n, x_in = self._prepare_action_for_rollout(x_gpu)
        _, cost, _ = self._evaluate_trajectory(x_in, get_constraint=False, get_cost=True)
        gradient = self._compute_gradient(cost, x_n)
        return (cost.view(-1).detach(), gradient.view(-1).detach())

    def _cost_constraint_and_gradient_fn(self, x_np):
        return self._gpu_to_cpu_wrapper(self._cost_constraint_and_gradient_fn_gpu, x_np)

    def _cost_and_gradient_fn_gpu(self, x_gpu):
        return self._cost_constraint_and_gradient_fn_gpu(x_gpu)

    def _cost_and_gradient_fn(self, x_np):
        return self._gpu_to_cpu_wrapper(self._cost_and_gradient_fn_gpu, x_np)

    def _constraint_fn_gpu(self, x_gpu):
        if self.use_cuda_graph:
            if self._constraint_executor is None:
                self._constraint_executor = create_graph_executor(
                    capture_fn=self._constraint_fn_gpu_impl,
                    device=self.device_cfg.device,
                )
            return self._constraint_executor(x_gpu, clone_outputs=True)
        return self._constraint_fn_gpu_impl(x_gpu)

    def _constraint_fn_gpu_impl(self, x_gpu):
        x_n, x_in = self._prepare_action_for_rollout(x_gpu)
        _, _, constraint = self._evaluate_trajectory(x_in, get_constraint=True, get_cost=False)
        if constraint is None:
            return None
        return constraint.view(-1).detach()

    def _constraint_fn(self, x_np):
        return self._gpu_to_cpu_wrapper(self._constraint_fn_gpu, x_np)

    def _constraint_gradient_fn_gpu(self, x_gpu):
        if self.use_cuda_graph:
            if self._constraint_grad_executor is None:
                self._constraint_grad_executor = create_graph_executor(
                    capture_fn=self._constraint_gradient_fn_gpu_impl,
                    device=self.device_cfg.device,
                )
            return self._constraint_grad_executor(x_gpu, clone_outputs=True)
        return self._constraint_gradient_fn_gpu_impl(x_gpu)

    def _constraint_gradient_fn_gpu_impl(self, x_gpu):
        x_n, x_in = self._prepare_action_for_rollout(x_gpu)
        _, _, constraint = self._evaluate_trajectory(x_in, get_constraint=True, get_cost=False)
        if constraint is None:
            return None
        gradient = self._compute_gradient(constraint, x_n, negate=True)
        return gradient.view(-1)

    def _constraint_gradient_fn(self, x_np):
        return self._gpu_to_cpu_wrapper(self._constraint_gradient_fn_gpu, x_np)

    def _has_constraints(self):
        q = self._opt_init.detach().clone()
        return self._constraint_fn_gpu_impl(q) is not None

    def _iteration_callback(self, intermediate_result):
        if self._debug:
            if isinstance(intermediate_result, ScipyOptimizeResult):
                q = torch.as_tensor(
                    intermediate_result.x, device=self.device_cfg.device, dtype=self.device_cfg.dtype,
                )
                cost = torch.as_tensor(
                    intermediate_result.fun, dtype=self.device_cfg.dtype, device=self.device_cfg.device,
                )
                action = q.view(-1, self.action_horizon, self.action_dim).clone()
                cost = cost.view(-1)
                state = OptimizationIterationState(action=action, cost=cost)
            elif isinstance(intermediate_result, np.ndarray):
                q = torch.as_tensor(
                    intermediate_result, device=self.device_cfg.device, dtype=self.device_cfg.dtype,
                )
                cost = torch.as_tensor(
                    1.0, dtype=self.device_cfg.dtype, device=self.device_cfg.device,
                )
                action = q.view(-1, self.action_horizon, self.action_dim).clone()
                state = OptimizationIterationState(action=action, cost=cost)
            self._debug.record(state, self.action_horizon, self.action_dim)

    # -- Lifecycle --

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        action = action.view(self.config.num_problems, self.action_horizon, self.action_dim)
        if mask is None and self._debug:
            self._debug.clear()
        if reset_num_iters:
            self.config.num_iters = self._og_num_iters

    def shift(self, shift_steps=0):
        return self._shift(shift_steps)

    def _shift(self, shift_steps=0):
        return True

    def update_num_problems(self, num_problems):
        assert num_problems > 0
        if num_problems > 1:
            log_and_raise("ScipyOpt only supports solving 1 optimization problem.")
        self.config.num_problems = num_problems
        self._opt_init = torch.zeros(
            (num_problems, self.opt_dim), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        self.l_vec = torch.ones(
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
        if self._cost_constraint_grad_executor is not None:
            self._cost_constraint_grad_executor.reset()
        if self._constraint_executor is not None:
            self._constraint_executor.reset()
        if self._constraint_grad_executor is not None:
            self._constraint_grad_executor.reset()

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

    def debug_dump(self, file_path=""):
        pass


# Backward compat alias: factory references this for use_cuda_graph=True with scipy
CudaGraphScipyOpt = ScipyOpt
