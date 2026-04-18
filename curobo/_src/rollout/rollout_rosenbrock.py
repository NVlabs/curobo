# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout that evaluates the Rosenbrock cost function for optimizer testing.

Provides action-space bounds and a forward pass that computes the Rosenbrock
objective, useful for verifying optimizer convergence independently of a full
robot model.  Supports optional CUDA-graph wrapping via ``use_cuda_graph``.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.rollout.metrics import (
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
    CostCollection,
)
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_graph_util import create_graph_executor, GraphExecutor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer


@dataclass
class RosenbrockCfg:
    """Configuration for the Rosenbrock rollout class."""

    #: Device and dtype for tensors.
    device_cfg: DeviceCfg

    #: The 'a' parameter in the Rosenbrock function (a-x)^2 + b(y-x^2)^2
    a: float = 1.0
    #: The 'b' parameter in the Rosenbrock function (a-x)^2 + b(y-x^2)^2
    b: float = 100.0
    #: Number of dimensions for the Rosenbrock function
    dimensions: int = 2
    #: Time horizon for the rollout
    time_horizon: int = 1
    #: Action horizon (usually same as horizon)
    time_action_horizon: int = 1
    #: Whether to sum costs across the horizon.
    sum_horizon: bool = False
    #: Seed for the random number generator.
    sampler_seed: int = 1312

    @classmethod
    def create(cls, config_dict: Dict, device_cfg: DeviceCfg = DeviceCfg()):
        """Create RosenbrockCfg from a dictionary."""
        return cls(
            a=config_dict.get("a", 1.0),
            b=config_dict.get("b", 100.0),
            dimensions=config_dict.get("dimensions", 2),
            time_horizon=config_dict.get("time_horizon", 1),
            time_action_horizon=config_dict.get("time_action_horizon", 1),
            device_cfg=device_cfg,
            sum_horizon=config_dict.get("sum_horizon", False),
            sampler_seed=config_dict.get("sampler_seed", 1312),
        )


class RosenbrockRollout:
    """Rollout that evaluates the Rosenbrock cost for optimizer testing.

    f(x, y) = (a - x)^2 + b(y - x^2)^2

    For higher dimensions, uses the generalized form:
    f(x) = sum_{i=1}^{n-1} [ (a - x_i)^2 + b(x_{i+1} - x_i^2)^2 ]

    Useful for verifying optimizer convergence without a full robot model.
    Supports optional CUDA-graph wrapping via ``use_cuda_graph``.
    """

    def __init__(self, config: Optional[RosenbrockCfg] = None, use_cuda_graph: bool = False):
        if config is not None:
            self.a = config.a
            self.b = config.b
            self.dimensions = config.dimensions
            self.time_horizon = config.time_horizon
            self.time_action_horizon = config.time_action_horizon
            self.device_cfg = config.device_cfg
            self.sum_horizon = config.sum_horizon
            self.sampler_seed = config.sampler_seed

        # CUDA graph state
        self._use_cuda_graph = use_cuda_graph
        self._compute_metrics_from_state_executor: Optional[GraphExecutor] = None
        self._compute_metrics_from_action_executor: Optional[GraphExecutor] = None

        self.start_state = None
        self.rollout_instance_name = None

        if config is not None:
            self._initialize_components()

    def _initialize_components(self):
        """Set up action bounds and sampler."""
        self._action_bound_lows = (
            torch.ones((self.action_horizon,), **self.device_cfg.as_torch_dict()) * -1.5
        )
        self._action_bound_highs = (
            torch.ones((self.action_horizon,), **self.device_cfg.as_torch_dict()) * 2.0
        )
        self._batch_size = 1

        self.act_sample_gen = SampleBuffer.create_halton_sample_buffer(
            ndims=self.action_dim,
            device_cfg=self.device_cfg,
            up_bounds=self.action_bound_highs,
            low_bounds=self.action_bound_lows,
            seed=self.sampler_seed,
        )

    # -- Properties --

    @property
    def action_dim(self) -> int:
        return self.dimensions

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self._action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self._action_bound_highs

    @property
    def action_bounds(self) -> torch.Tensor:
        return self.device_cfg.to_device(
            torch.stack([self.action_bound_lows, self.action_bound_highs])
        )

    @property
    def horizon(self) -> int:
        return self.time_horizon

    @property
    def action_horizon(self) -> int:
        return self.time_action_horizon

    @property
    def state_bounds(self) -> Dict[str, List[float]]:
        return {"position": [-2.048, 2.048]}

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    @property
    def dt(self) -> float:
        return 1.0

    # -- Core --

    def evaluate_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutResult:
        batch_size = act_seq.shape[0]
        if batch_size != self._batch_size:
            self.update_batch_size(batch_size)
        state = self._compute_state_from_action_impl(act_seq)
        costs_and_constraints = self._compute_costs_and_constraints_impl(state, **kwargs)
        return RolloutResult(
            actions=act_seq,
            state=state,
            costs_and_constraints=costs_and_constraints,
        )

    def compute_metrics_from_state(self, state: JointState, **kwargs) -> RolloutMetrics:
        if self._use_cuda_graph:
            if self._compute_metrics_from_state_executor is None:
                self._compute_metrics_from_state_executor = create_graph_executor(
                    capture_fn=self._compute_metrics_from_state_impl,
                    device=self.device_cfg.device,
                )
            return self._compute_metrics_from_state_executor(state)
        return self._compute_metrics_from_state_impl(state, **kwargs)

    def compute_metrics_from_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutMetrics:
        batch_size = act_seq.shape[0]
        self.update_batch_size(batch_size)
        if self._use_cuda_graph:
            if self._compute_metrics_from_action_executor is None:
                self._compute_metrics_from_action_executor = create_graph_executor(
                    capture_fn=self._compute_metrics_from_action_impl,
                    device=self.device_cfg.device,
                )
            return self._compute_metrics_from_action_executor(act_seq)
        return self._compute_metrics_from_action_impl(act_seq, **kwargs)

    # -- Lifecycle --

    def update_params(self, a: float = None, b: float = None, **kwargs) -> bool:
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        return True

    def update_batch_size(self, batch_size: int) -> None:
        if self._batch_size is None or self._batch_size != batch_size:
            self._batch_size = batch_size

    def update_dt(self, dt, **kwargs) -> bool:
        return True

    def reset(self, reset_problem_ids=None, **kwargs) -> bool:
        return True

    def reset_shape(self) -> bool:
        return True

    def reset_seed(self) -> None:
        self.act_sample_gen.reset()

    def reset_cuda_graph(self) -> bool:
        if self._compute_metrics_from_state_executor is not None:
            self._compute_metrics_from_state_executor.reset()
        if self._compute_metrics_from_action_executor is not None:
            self._compute_metrics_from_action_executor.reset()
        return self.reset_shape()

    # -- Sampling --

    def sample_random_actions(self, n: int = 0, bounded: bool = True) -> torch.Tensor:
        return self.act_sample_gen.get_samples(n, bounded=bounded)

    def get_initial_action(
        self, use_random: bool = True, use_zero: bool = False, **kwargs
    ) -> torch.Tensor:
        num_samples = self.batch_size
        if num_samples is None:
            num_samples = 1
        init_action = None
        if use_random:
            n_samples = num_samples * self.action_horizon
            init_action = self.sample_random_actions(n=n_samples, bounded=True)
        elif use_zero:
            init_action = torch.zeros(
                (num_samples, self.action_horizon, self.action_dim),
                **self.device_cfg.as_torch_dict(),
            )
        if init_action is None:
            log_and_raise("get_init_action_seq is not implemented")
        init_action = init_action.view(num_samples, self.action_horizon, self.action_dim)
        return init_action

    def get_all_cost_components(self):
        return {}

    # -- Internal --

    def _compute_state_from_action_impl(self, act_seq: torch.Tensor) -> JointState:
        return JointState.from_position(act_seq)

    def _compute_state_from_action_metrics_impl(self, act_seq: torch.Tensor) -> JointState:
        return self._compute_state_from_action_impl(act_seq)

    def _compute_costs_and_constraints_metrics_impl(
        self, state: JointState, **kwargs
    ) -> CostsAndConstraints:
        return self._compute_costs_and_constraints_impl(state, **kwargs)

    def _compute_costs_and_constraints_impl(
        self, state: JointState, **kwargs
    ) -> CostsAndConstraints:
        x = state.position
        costs_and_constraints = CostsAndConstraints()
        term1 = (self.a - x[:, :, 0]) ** 2
        term2 = self.b * (x[:, :, 1] - x[:, :, 0] ** 2) ** 2
        rosenbrock_cost = term1 + term2
        for i in range(1, self.action_dim - 1):
            term1 = (self.a - x[:, :, i]) ** 2
            term2 = self.b * (x[:, :, i + 1] - x[:, :, i] ** 2) ** 2
            rosenbrock_cost += term1 + term2
        rosenbrock_cost = rosenbrock_cost.unsqueeze(-1)
        costs_and_constraints.costs.add(rosenbrock_cost, "rosenbrock")
        return costs_and_constraints

    def _compute_metrics_from_state_impl(self, state: JointState, **kwargs) -> RolloutMetrics:
        costs_and_constraints = self._compute_costs_and_constraints_impl(state)
        convergence = costs_and_constraints.get_sum_cost(sum_horizon=False)
        convergence_collection = CostCollection()
        convergence_collection.add(convergence, "convergence")
        return RolloutMetrics(
            costs_and_constraints=costs_and_constraints,
            feasible=costs_and_constraints.get_feasible(),
            state=state,
            convergence=convergence,
        )

    def _compute_metrics_from_action_impl(
        self, act_seq: torch.Tensor, **kwargs
    ) -> RolloutMetrics:
        state = self._compute_state_from_action_impl(act_seq)
        metrics = self._compute_metrics_from_state_impl(state, **kwargs)
        metrics.actions = act_seq
        return metrics
