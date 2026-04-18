#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the Scipy optimizer wrapper."""

from __future__ import annotations

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def quadratic_cost_fn(state):
    """Simple quadratic cost function for testing."""
    costs = torch.sum((state - 5.0) ** 2, dim=-1).unsqueeze(-1)
    return costs


class MockRolloutWithConstraints:
    """Standalone mock rollout with constraints (no base class)."""

    def __init__(
        self,
        num_dof: int = 3,
        action_horizon: int = 5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        has_constraints: bool = False,
    ):
        self.device_cfg = DeviceCfg(device=device, dtype=dtype)
        self.sum_horizon = True
        self.sampler_seed = 1
        self._action_dim = num_dof
        self._action_horizon = action_horizon
        self._tensor_args = self.device_cfg
        self._has_constraints = has_constraints
        self._action_bound_lows = torch.ones(num_dof, device=device, dtype=dtype) * -10.0
        self._action_bound_highs = torch.ones(num_dof, device=device, dtype=dtype) * 10.0
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

        constraints = None
        if self._has_constraints:
            constraints = state.sum(dim=-1, keepdim=True) - 5.0
            constraints = CostCollection(
                values=[constraints],
                names=["constraint"],
                weights=[1.0],
                sq_weights=[1.0],
            )

        return CostsAndConstraints(costs=costs_collection, constraints=constraints)


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    """Get dtype for testing."""
    return torch.float32


class TestScipyOptCfg:
    """Test cases for ScipyOptCfg dataclass."""

    def test_init_default(self, device, dtype):
        """Test default initialization."""
        cfg = ScipyOptCfg(
            num_iters=50,
            solver_type="scipy",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        assert cfg.scipy_minimize_method == "SLSQP"
        assert cfg.scipy_minimize_kwargs == {}
        assert cfg.num_iters == 50

    def test_init_custom_method(self, device, dtype):
        """Test initialization with custom scipy method."""
        cfg = ScipyOptCfg(
            num_iters=100,
            solver_type="scipy",
            scipy_minimize_method="L-BFGS-B",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        assert cfg.scipy_minimize_method == "L-BFGS-B"

    def test_init_with_kwargs(self, device, dtype):
        """Test initialization with custom kwargs."""
        custom_kwargs = {"ftol": 1e-8, "gtol": 1e-6}
        cfg = ScipyOptCfg(
            num_iters=50,
            solver_type="scipy",
            scipy_minimize_kwargs=custom_kwargs,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        assert cfg.scipy_minimize_kwargs == custom_kwargs


class TestScipyOpt:
    """Test cases for ScipyOpt optimizer."""

    @pytest.fixture
    def rollout_fn(self, device, dtype):
        """Create a mock rollout function."""
        return MockRolloutWithConstraints(
            num_dof=3,
            action_horizon=5,
            device=device,
            dtype=dtype,
            has_constraints=False,
        )

    @pytest.fixture
    def rollout_fn_with_constraints(self, device, dtype):
        """Create a mock rollout function with constraints."""
        return MockRolloutWithConstraints(
            num_dof=3,
            action_horizon=5,
            device=device,
            dtype=dtype,
            has_constraints=True,
        )

    def test_init_basic(self, device, dtype, rollout_fn):
        """Test basic initialization."""
        cfg = ScipyOptCfg(
            num_iters=10,
            solver_type="scipy",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])

        assert optimizer.config.num_iters == 10
        assert len(optimizer._scipy_action_bounds) == rollout_fn.action_horizon * rollout_fn.action_dim

    def test_bounds_initialization(self, device, dtype, rollout_fn):
        """Test that action bounds are correctly initialized."""
        cfg = ScipyOptCfg(
            num_iters=10,
            solver_type="scipy",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])

        # Check bounds
        for low, high in optimizer._scipy_action_bounds:
            assert low == -10.0
            assert high == 10.0

    @pytest.mark.parametrize("method", ["L-BFGS-B"])
    def test_optimize_different_methods(self, device, dtype, rollout_fn, method):
        """Test optimization with different scipy methods."""
        cfg = ScipyOptCfg(
            num_iters=20,
            solver_type="scipy",
            scipy_minimize_method=method,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Create initial action (far from optimum at 5.0)
        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Optimize
        result = optimizer.optimize(initial_action)

        # Check that result is closer to optimum (5.0) than initial (0.0)
        initial_cost = quadratic_cost_fn(initial_action).mean()
        final_cost = quadratic_cost_fn(result).mean()

        assert final_cost < initial_cost
        assert result.shape == initial_action.shape

    def test_optimize_with_slsqp(self, device, dtype, rollout_fn):
        """Test optimization with SLSQP method (automatically converts to float64)."""
        cfg = ScipyOptCfg(
            num_iters=30,
            solver_type="scipy",
            scipy_minimize_method="SLSQP",  # Float64 conversion auto-enabled for SLSQP
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        initial_action = torch.zeros(
            1,
            rollout_fn.action_horizon,
            rollout_fn.action_dim,
            device=device,
            dtype=dtype,
        )

        result = optimizer.optimize(initial_action)

        # Check optimization ran and improved
        initial_cost = quadratic_cost_fn(initial_action).mean()
        final_cost = quadratic_cost_fn(result).mean()

        assert result.shape == initial_action.shape
        assert final_cost < initial_cost

    def test_update_num_problems_single_only(self, device, dtype, rollout_fn):
        """Test that ScipyOpt only supports single problem."""
        cfg = ScipyOptCfg(
            num_iters=10,
            solver_type="scipy",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])

        # Should work with num_problems=1
        optimizer.update_num_problems(1)

        # Should raise error with num_problems>1
        with pytest.raises(ValueError):
            optimizer.update_num_problems(2)

    def test_reinitialize(self, device, dtype, rollout_fn):
        """Test optimizer reinitialize."""
        cfg = ScipyOptCfg(
            num_iters=10,
            solver_type="scipy",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Create initial action for reinitialize
        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Reinitialize should not raise errors
        optimizer.reinitialize(initial_action)

    def test_cost_and_gradient_computation(self, device, dtype, rollout_fn):
        """Test internal cost and gradient computation."""
        cfg = ScipyOptCfg(
            num_iters=10,
            solver_type="scipy",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Create test action
        test_action = torch.ones(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Test GPU wrapper functions
        x_gpu = test_action.view(-1)
        cost, gradient = optimizer._cost_constraint_and_gradient_fn_gpu(x_gpu)

        assert cost.shape == (1,)
        assert gradient.shape == x_gpu.shape
        assert not torch.isnan(cost).any()
        assert not torch.isnan(gradient).any()

    def test_store_debug(self, device, dtype, rollout_fn):
        """Test optimization with debug storage."""
        cfg = ScipyOptCfg(
            num_iters=5,
            solver_type="scipy",
            scipy_minimize_method="L-BFGS-B",  # Use L-BFGS-B which works with float32
            store_debug=True,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        initial_action = torch.ones(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer.optimize(initial_action)

        # Debug info should be recorded
        assert len(optimizer.get_recorded_trace()["debug"]) > 0
        assert result.shape == initial_action.shape


class TestScipyOptCudaGraph:
    """Test cases for ScipyOpt with use_cuda_graph=True."""

    @pytest.fixture
    def rollout_fn(self, device, dtype):
        """Create a mock rollout function."""
        return MockRolloutWithConstraints(
            num_dof=3,
            action_horizon=5,
            device=device,
            dtype=dtype,
            has_constraints=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_cuda_graph(self, rollout_fn):
        """Test CudaGraphScipyOpt initialization."""
        cfg = ScipyOptCfg(
            num_iters=10,
            solver_type="scipy",
            device_cfg=DeviceCfg(device="cuda", dtype=torch.float32),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn], use_cuda_graph=True)

        assert optimizer._cost_constraint_grad_executor is None
        assert optimizer._constraint_executor is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimize_with_cuda_graph(self, rollout_fn):
        """Test optimization with CUDA graphs."""
        cfg = ScipyOptCfg(
            num_iters=10,
            solver_type="scipy",
            scipy_minimize_method="L-BFGS-B",
            device_cfg=DeviceCfg(device="cuda", dtype=torch.float32),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn], use_cuda_graph=True)
        optimizer.update_num_problems(1)

        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device="cuda", dtype=torch.float32
        )

        # First optimization (records CUDA graph)
        result = optimizer.optimize(initial_action)

        # Check that executors are created
        assert optimizer._cost_constraint_grad_executor is not None

        # Second optimization (uses cached graph)
        result2 = optimizer.optimize(initial_action)

        assert result.shape == result2.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reset_cuda_graph(self, rollout_fn):
        """Test CUDA graph reset."""
        cfg = ScipyOptCfg(
            num_iters=5,
            solver_type="scipy",
            device_cfg=DeviceCfg(device="cuda", dtype=torch.float32),
        )

        optimizer = ScipyOpt(cfg, [rollout_fn], use_cuda_graph=True)
        optimizer.update_num_problems(1)

        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device="cuda", dtype=torch.float32
        )

        # Run optimization to create graphs
        optimizer.optimize(initial_action)

        # Reset should not raise errors
        optimizer.reset_cuda_graph()

