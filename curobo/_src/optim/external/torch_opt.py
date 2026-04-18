# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper that adapts any ``torch.optim`` optimizer for cuRobo trajectory optimization."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import torch

from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise


@dataclass
class TorchOptCfg:
    """Flat configuration for torch optimizer wrapper."""

    # General
    num_iters: int = 100
    solver_type: str = "torch"
    solver_name: str = "torch"
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

    # Torch-specific
    #: Name of a ``torch.optim`` optimizer class (e.g. ``"Adam"``,
    #: ``"SGD"``, ``"LBFGS"``). Used to look up the class via
    #: ``getattr(torch.optim, torch_optim_name)`` when
    #: :attr:`torch_optim_class` is None.
    torch_optim_name: str = "Adam"
    torch_optim_kwargs: dict = field(default_factory=dict)
    #: Explicit optimizer class to use instead of looking up by name. When
    #: set, :attr:`torch_optim_name` is ignored for class resolution. This
    #: allows using custom optimizer classes not in ``torch.optim``.
    torch_optim_class: Optional[Any] = None

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.torch_optim_class is None:
            self.torch_optim_class = getattr(torch.optim, self.torch_optim_name)

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


class TorchOpt:
    """Adapts any ``torch.optim`` optimizer (Adam, SGD, LBFGS, etc.) for cuRobo rollouts.

    Evaluates the cost function via cuRobo rollouts, backpropagates through the
    computation graph, and delegates the parameter update to the wrapped PyTorch
    optimizer.
    """

    _graphable_methods: set = set()

    def __init__(
        self,
        config: TorchOptCfg,
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

        self._torch_optimizer = None
        self._optimization_variable = None
        self.best_cost = None
        self.best_q = None
        self.l_vec = None
        self._executors: Dict[str, Any] = {}

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

    def _optimize_impl(self, seed_action: torch.Tensor):
        if self._torch_optimizer is None:
            self._init_torch_optimizer()

        seed_action = seed_action.view(self.config.num_problems, self.opt_dim)

        with torch.no_grad():
            self._optimization_variable.copy_(
                seed_action.view(self.config.num_problems, self.action_horizon, self.action_dim)
            )

        iteration_state = OptimizationIterationState(
            action=seed_action.view(self.config.num_problems, self.action_horizon, self.action_dim),
            best_action=self.best_q.view(self.config.num_problems, self.action_horizon, self.action_dim),
            best_cost=self.best_cost,
        )
        self._record_iteration_state(iteration_state)

        for _ in range(self.config.num_iters):
            self._torch_optimizer.zero_grad()
            iteration_state = self._opt_step(iteration_state)
            self._record_iteration_state(iteration_state)

        return iteration_state.best_action.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )

    def _opt_step(self, iteration_state):
        loss = self._loss_fn(self._optimization_variable)
        cost = loss.view(self.config.num_problems)
        mask = cost < iteration_state.best_cost
        iteration_state.best_cost.copy_(torch.where(mask, cost, iteration_state.best_cost))
        mask_expanded = mask.view(-1, 1, 1).expand(-1, self.action_horizon, self.action_dim)
        iteration_state.best_action.copy_(
            torch.where(mask_expanded, self._optimization_variable.detach(), iteration_state.best_action)
        )

        loss.backward(gradient=self.l_vec)

        if self.config.torch_optim_class.__name__ == "LBFGS":
            self._current_q = self._optimization_variable
            self._torch_optimizer.step(self._closure)
        else:
            self._torch_optimizer.step()

        next_action = self._optimization_variable.detach().clone()
        return OptimizationIterationState(
            action=next_action,
            cost=cost,
            best_cost=iteration_state.best_cost,
            best_action=iteration_state.best_action,
        )

    def _loss_fn(self, x):
        x_in = x.view(
            self.config.num_problems * self.config.num_particles,
            self.action_horizon, self.rollout_fn.action_dim,
        )
        trajectories = self.rollout_fn.evaluate_action(x_in)
        costs = trajectories.costs_and_constraints.get_sum_cost(
            sum_horizon=True, include_all_hybrid=False,
        )
        if len(costs.shape) == 2:
            cost = torch.sum(
                costs.view(self.config.num_problems, self.config.num_particles, self.action_horizon),
                dim=-1, keepdim=True,
            )
        else:
            cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        return cost

    def _closure(self):
        self._torch_optimizer.zero_grad()
        loss = self._loss_fn(self._current_q)
        loss.backward(gradient=self.l_vec)
        return torch.mean(loss)

    def _init_torch_optimizer(self):
        if self._torch_optimizer is None and self._optimization_variable is not None:
            self._torch_optimizer = self.config.torch_optim_class(
                [self._optimization_variable], **self.config.torch_optim_kwargs
            )
            if self.config.torch_optim_class.__name__ == "Adam":
                try:
                    self._torch_optimizer.step = torch.compile(self._torch_optimizer.step)
                except Exception:
                    pass

    # -- Lifecycle --

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        action = action.view(self.config.num_problems, self.action_horizon, self.action_dim)
        if mask is None and self._debug:
            self._debug.clear()
        if reset_num_iters:
            self.config.num_iters = self._og_num_iters
        self.best_cost[:] = 50000000000.0
        self._torch_optimizer = self.config.torch_optim_class(
            [self._optimization_variable], **self.config.torch_optim_kwargs
        )
        if self.config.torch_optim_class.__name__ == "Adam":
            try:
                self._torch_optimizer.step = torch.compile(self._torch_optimizer.step)
            except Exception:
                pass

    def shift(self, shift_steps=0):
        return self._shift(shift_steps)

    def _shift(self, shift_steps=0):
        return True

    def update_num_problems(self, num_problems):
        assert num_problems > 0
        self.config.num_problems = num_problems

        self._optimization_variable = torch.zeros(
            (num_problems, self.action_horizon, self.action_dim),
            device=self.device_cfg.device, dtype=self.device_cfg.dtype,
        )
        self._optimization_variable.requires_grad = True
        self.best_cost = (
            torch.ones((num_problems,), device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            * 5000000.0
        )
        self.best_q = torch.zeros(
            (num_problems, self.opt_dim), device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        self.l_vec = torch.ones(
            (num_problems, self.config.num_particles, 1),
            device=self.device_cfg.device, dtype=self.device_cfg.dtype,
        )

        self._torch_optimizer = None
        self._init_torch_optimizer()

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
        pass

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
