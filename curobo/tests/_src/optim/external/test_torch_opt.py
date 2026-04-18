#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the Torch optimizer wrapper."""

from __future__ import annotations

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def quadratic_cost_fn(state):
    """Simple quadratic cost function for testing."""
    costs = torch.sum((state - 3.0) ** 2, dim=-1).unsqueeze(-1)
    return costs


class MockRolloutForTorch:
    """Standalone mock rollout for Torch optimizer testing (no base class)."""

    def __init__(
        self,
        num_dof: int = 3,
        action_horizon: int = 5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device_cfg = DeviceCfg(device=device, dtype=dtype)
        self.sum_horizon = True
        self.sampler_seed = 1
        self._action_dim = num_dof
        self._action_horizon = action_horizon
        self._tensor_args = self.device_cfg
        self._action_bound_lows = torch.ones(num_dof, device=device, dtype=dtype) * -5.0
        self._action_bound_highs = torch.ones(num_dof, device=device, dtype=dtype) * 5.0
        self._batch_size = None
        self.start_state = None
        self.act_sample_gen = SampleBuffer.create_halton_sample_buffer(
            ndims=num_dof, device_cfg=self.device_cfg,
            up_bounds=self._action_bound_highs, low_bounds=self._action_bound_lows,
            seed=self.sampler_seed,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self._action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self._action_bound_highs

    @property
    def action_bounds(self) -> torch.Tensor:
        return torch.stack([self._action_bound_lows, self._action_bound_highs])

    @property
    def state_bounds(self) -> torch.Tensor:
        return self.action_bounds

    @property
    def horizon(self) -> int:
        return self._action_horizon

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def dt(self) -> float:
        return 1.0

    def evaluate_action(self, act_seq, **kwargs):
        batch_size = act_seq.shape[0]
        if self._batch_size != batch_size:
            self.update_batch_size(batch_size)
        state = self._compute_state_from_action_impl(act_seq)
        cc = self._compute_costs_and_constraints_impl(state, **kwargs)
        return RolloutResult(actions=act_seq, state=state, costs_and_constraints=cc)

    def compute_metrics_from_state(self, state, **kwargs):
        cc = self._compute_costs_and_constraints_impl(state, **kwargs)
        convergence = cc.get_sum_cost(sum_horizon=False)
        return RolloutMetrics(costs_and_constraints=cc, feasible=cc.get_feasible(), state=state, convergence=convergence)

    def compute_metrics_from_action(self, act_seq, **kwargs):
        batch_size = act_seq.shape[0]
        self.update_batch_size(batch_size)
        state = self._compute_state_from_action_impl(act_seq)
        metrics = self.compute_metrics_from_state(state, **kwargs)
        metrics.actions = act_seq
        return metrics

    def update_batch_size(self, batch_size):
        if self._batch_size is None or self._batch_size != batch_size:
            self._batch_size = batch_size

    def update_params(self, **kwargs):
        return True

    def update_dt(self, dt, **kwargs):
        return True

    def reset(self, reset_problem_ids=None, **kwargs):
        return True

    def reset_shape(self):
        return True

    def reset_seed(self):
        self.act_sample_gen.reset()

    def reset_cuda_graph(self):
        return self.reset_shape()

    def sample_random_actions(self, n=0, bounded=True):
        return self.act_sample_gen.get_samples(n, bounded=bounded)

    def get_initial_action(self, use_random=True, use_zero=False, **kwargs):
        num_samples = self._batch_size or 1
        if use_random:
            n_samples = num_samples * self._action_horizon
            init_action = self.sample_random_actions(n=n_samples, bounded=True)
        elif use_zero:
            init_action = torch.zeros(
                (num_samples, self._action_horizon, self._action_dim), **self.device_cfg.as_torch_dict())
        else:
            init_action = torch.zeros(
                (num_samples, self._action_horizon, self._action_dim), **self.device_cfg.as_torch_dict())
        return init_action.view(num_samples, self._action_horizon, self._action_dim)

    def get_all_cost_components(self):
        return {}

    def filter_robot_state(self, state):
        return state

    def _compute_state_from_action_impl(self, act_seq):
        return act_seq

    def _compute_costs_and_constraints_impl(self, state, **kwargs):
        costs = quadratic_cost_fn(state)
        costs_collection = CostCollection(
            values=[costs], names=["cost"], weights=[1.0], sq_weights=[1.0]
        )

        return CostsAndConstraints(costs=costs_collection, constraints=None)


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    """Get dtype for testing."""
    return torch.float32


class TestTorchOptCfg:
    """Test cases for TorchOptCfg dataclass."""

    def test_init_default(self, device, dtype):
        """Test default initialization."""
        cfg = TorchOptCfg(
            num_iters=50,
            solver_type="torch",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        assert cfg.torch_optim_name == "Adam"
        assert cfg.torch_optim_kwargs == {}
        assert cfg.num_iters == 50

    def test_init_custom_optimizer(self, device, dtype):
        """Test initialization with custom optimizer."""
        cfg = TorchOptCfg(
            num_iters=100,
            solver_type="torch",
            torch_optim_name="SGD",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        assert cfg.torch_optim_name == "SGD"
        assert cfg.torch_optim_class == torch.optim.SGD

    def test_init_with_kwargs(self, device, dtype):
        """Test initialization with custom kwargs."""
        custom_kwargs = {"lr": 0.01, "momentum": 0.9}
        cfg = TorchOptCfg(
            num_iters=50,
            solver_type="torch",
            torch_optim_name="SGD",
            torch_optim_kwargs=custom_kwargs,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        assert cfg.torch_optim_kwargs == custom_kwargs

    def test_post_init_sets_optimizer_class(self, device, dtype):
        """Test that __post_init__ sets optimizer class."""
        cfg = TorchOptCfg(
            num_iters=50,
            solver_type="torch",
            torch_optim_name="Adam",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        assert cfg.torch_optim_class == torch.optim.Adam


class TestTorchOpt:
    """Test cases for TorchOpt optimizer."""

    @pytest.fixture
    def rollout_fn(self, device, dtype):
        """Create a mock rollout function."""
        return MockRolloutForTorch(
            num_dof=3, action_horizon=5, device=device, dtype=dtype
        )

    def test_init_basic(self, device, dtype, rollout_fn):
        """Test basic initialization."""
        cfg = TorchOptCfg(
            num_iters=10,
            solver_type="torch",
            torch_optim_name="Adam",
            torch_optim_kwargs={"lr": 0.01},
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])

        assert optimizer.config.num_iters == 10
        assert optimizer._torch_optimizer is not None

    @pytest.mark.parametrize(
        "optimizer_name,kwargs",
        [
            ("Adam", {"lr": 0.01}),
            ("SGD", {"lr": 0.1}),
            ("RMSprop", {"lr": 0.01}),
        ],
    )
    def test_different_torch_optimizers(self, device, dtype, rollout_fn, optimizer_name, kwargs):
        """Test with different torch optimizers."""
        cfg = TorchOptCfg(
            num_iters=10,
            solver_type="torch",
            torch_optim_name=optimizer_name,
            torch_optim_kwargs=kwargs,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Should not raise errors
        result = optimizer.optimize(initial_action)
        assert result.shape == initial_action.shape

    def test_optimize_converges(self, device, dtype, rollout_fn):
        """Test that optimization improves cost."""
        cfg = TorchOptCfg(
            num_iters=50,
            solver_type="torch",
            torch_optim_name="Adam",
            torch_optim_kwargs={"lr": 0.1},
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Start far from optimum (optimum is at 3.0)
        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer.optimize(initial_action)

        # Check that result is closer to optimum (3.0) than initial (0.0)
        initial_cost = quadratic_cost_fn(initial_action).mean()
        final_cost = quadratic_cost_fn(result).mean()

        assert final_cost < initial_cost
        # Result should be closer to 3.0
        assert torch.abs(result.mean() - 3.0) < torch.abs(initial_action.mean() - 3.0)

    def test_update_num_problems(self, device, dtype, rollout_fn):
        """Test updating number of problems."""
        cfg = TorchOptCfg(
            num_iters=10,
            solver_type="torch",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])

        # Update to 2 problems
        optimizer.update_num_problems(2)

        assert optimizer.config.num_problems == 2
        assert optimizer._optimization_variable.shape[0] == 2
        assert optimizer.best_cost.shape == (2,)

    def test_reinitialize(self, device, dtype, rollout_fn):
        """Test optimizer reinitialize."""
        cfg = TorchOptCfg(
            num_iters=10,
            solver_type="torch",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Create initial action for reinitialize
        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Reinitialize should not raise errors
        optimizer.reinitialize(initial_action)
        optimizer.reinitialize(initial_action, clear_optimizer_state=False)

        # After reinitialize, best cost should be reset to large value
        assert optimizer.best_cost[0] > 1e6

    def test_store_debug(self, device, dtype, rollout_fn):
        """Test optimization with debug storage."""
        cfg = TorchOptCfg(
            num_iters=5,
            solver_type="torch",
            store_debug=True,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        initial_action = torch.ones(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer.optimize(initial_action)

        # Debug info should be recorded
        assert len(optimizer.get_recorded_trace()["debug"]) > 0
        assert result.shape == initial_action.shape

    def test_loss_function(self, device, dtype, rollout_fn):
        """Test internal loss function."""
        cfg = TorchOptCfg(
            num_iters=10,
            solver_type="torch",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Create test action
        test_action = torch.ones(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )
        test_action.requires_grad = True

        # Compute loss
        loss = optimizer._loss_fn(test_action)

        assert loss.shape == (1, 1, 1)
        assert not torch.isnan(loss).any()
        assert loss.requires_grad  # Should be differentiable

    def test_opt_step(self, device, dtype, rollout_fn):
        """Test single optimization step."""
        cfg = TorchOptCfg(
            num_iters=10,
            solver_type="torch",
            torch_optim_kwargs={"lr": 0.01},
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        from curobo._src.optim.optimization_iteration_state import OptimizationIterationState

        # Create initial state
        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )
        initial_cost = torch.tensor([1000.0], device=device, dtype=dtype)

        iteration_state = OptimizationIterationState(
            action=initial_action,
            best_action=initial_action.clone(),
            best_cost=initial_cost,
        )

        # Perform one step
        next_state = optimizer._opt_step(iteration_state)

        assert next_state.action.shape == initial_action.shape
        assert next_state.best_cost is not None


class TestTorchOptEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def rollout_fn(self, device, dtype):
        """Create a mock rollout function."""
        return MockRolloutForTorch(
            num_dof=3, action_horizon=5, device=device, dtype=dtype
        )

    def test_multiple_problems(self, device, dtype, rollout_fn):
        """Test with multiple problems."""
        cfg = TorchOptCfg(
            num_iters=10,
            solver_type="torch",
            torch_optim_kwargs={"lr": 0.1},
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = TorchOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(3)

        initial_action = torch.zeros(
            3, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer.optimize(initial_action)
        assert result.shape == (3, rollout_fn.action_horizon, rollout_fn.action_dim)

    def test_different_learning_rates(self, device, dtype, rollout_fn):
        """Test optimization with different learning rates."""
        for lr in [0.001, 0.01, 0.1]:
            cfg = TorchOptCfg(
                num_iters=20,
                solver_type="torch",
                torch_optim_kwargs={"lr": lr},
                device_cfg=DeviceCfg(device=device, dtype=dtype),
            )

            optimizer = TorchOpt(cfg, [rollout_fn])
            optimizer.update_num_problems(1)

            initial_action = torch.zeros(
                1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
            )

            result = optimizer.optimize(initial_action)

            # Should converge (though speed may vary)
            initial_cost = quadratic_cost_fn(initial_action).mean()
            final_cost = quadratic_cost_fn(result).mean()
            assert final_cost < initial_cost

