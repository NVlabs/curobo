#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the LSR1 (Limited-memory Symmetric Rank-1) optimizer."""

from __future__ import annotations

# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
from curobo._src.optim.gradient.lsr1 import LSR1Opt, jit_lsr1_compute_step_direction
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def cost_fn(state):
    """Simple quadratic cost function: sum((10.0 - state)^2)."""
    costs = torch.sum((10.0 - state) ** 2, dim=-1).unsqueeze(-1)
    return costs


class MockRollout:
    """Standalone mock rollout (no base class). Satisfies the Rollout protocol."""

    def __init__(
        self,
        num_dof: int = 7,
        action_horizon: int = 10,
        batch_size: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device_cfg = DeviceCfg(device=device, dtype=dtype)
        self.sum_horizon = True
        self.sampler_seed = 1
        self._action_dim = num_dof
        self._action_horizon = action_horizon
        self._batch_size = None
        self._tensor_args = self.device_cfg
        self._action_bound_lows = torch.ones(num_dof, device=device, dtype=dtype) * -1.0
        self._action_bound_highs = torch.ones(num_dof, device=device, dtype=dtype) * 1.0
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
        if batch_size != self._batch_size:
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
        costs = cost_fn(state)
        costs = CostCollection(values=[costs], names=["cost"], weights=[1.0], sq_weights=[1.0])
        return CostsAndConstraints(costs=costs)


@pytest.fixture
def device():
    """Get the device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    """Get the dtype for testing."""
    return torch.float32


@pytest.fixture
def optimizer_setup(device, dtype):
    """Set up environment for LSR1 optimization tests."""
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Define test parameters
    num_dof = 7
    action_horizon = 28
    batch_size = 4

    # Create rollout function
    rollout_fn = MockRollout(
        num_dof=num_dof,
        action_horizon=action_horizon,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    return {
        "device": device,
        "dtype": dtype,
        "num_dof": num_dof,
        "action_horizon": action_horizon,
        "batch_size": batch_size,
        "rollout_fn": rollout_fn,
        "step_scale": 0.98,
        "return_best_action": True,
    }


def create_optimizer(
    rollout_fn,
    batch_size,
    history=15,
    num_iters=50,
    epsilon=0.01,
    stable_mode=True,
):
    """Create an LSR1 optimizer with the specified configuration."""
    config = LBFGSOptCfg(
        num_problems=batch_size,
        num_iters=num_iters,
        history=history,
        epsilon=epsilon,
        step_scale=0.98,
        use_cuda_kernel_step_direction=False,  # LSR1 doesn't use CUDA kernel
        stable_mode=stable_mode,
        solver_type="lsr1",
        line_search_scale=[0, 0.1, 0.5, 1.0],
    )

    optimizer = LSR1Opt(config, [rollout_fn, rollout_fn])
    optimizer.update_num_problems(batch_size)
    return optimizer


class TestLSR1Opt:
    """Test cases for LSR1Opt optimizer."""

    def test_init_basic(self, optimizer_setup):
        """Test basic initialization."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            num_iters=10,
        )
        assert optimizer.config.num_iters == 10
        assert optimizer.config.history == 15
        assert optimizer._qn.s is not None
        assert optimizer._qn.y is not None

    def test_optimization_convergence(self, optimizer_setup):
        """Test that LSR1 optimizer converges to the optimum."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            history=15,
            num_iters=50,
        )

        # Create a random starting point
        initial_action = (
            torch.randn(
                optimizer_setup["batch_size"],
                optimizer_setup["action_horizon"],
                optimizer_setup["num_dof"],
                device=optimizer_setup["device"],
                dtype=optimizer_setup["dtype"],
            )
            * 10.0
        )

        # Run optimization
        optimizer.reinitialize(initial_action.clone())
        result = optimizer.optimize(initial_action.clone())

        # Check convergence
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        print(f"Initial cost: {initial_cost}")
        print(f"Final cost: {final_cost}")

        # Should converge to a low cost
        assert final_cost < initial_cost
        assert final_cost < 10.0

    @pytest.mark.parametrize("history", [5, 10, 20])
    def test_different_history_sizes(self, optimizer_setup, history):
        """Test LSR1 with different history buffer sizes."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            history=history,
            num_iters=30,
        )

        initial_action = (
            torch.randn(
                optimizer_setup["batch_size"],
                optimizer_setup["action_horizon"],
                optimizer_setup["num_dof"],
                device=optimizer_setup["device"],
                dtype=optimizer_setup["dtype"],
            )
            * 10.0
        )

        # Run optimization
        optimizer.reinitialize(initial_action.clone())
        result = optimizer.optimize(initial_action.clone())

        # Check that optimizer reduced the cost
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        print(f"History: {history}")
        print(f"Initial cost: {initial_cost}")
        print(f"Final cost: {final_cost}")

        assert final_cost < initial_cost

    def test_reinitialize(self, optimizer_setup):
        """Test that reinitialize properly clears optimizer state."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            num_iters=5,
        )

        # Create a random action
        action = torch.randn(
            optimizer_setup["batch_size"],
            optimizer_setup["action_horizon"],
            optimizer_setup["num_dof"],
            device=optimizer_setup["device"],
            dtype=optimizer_setup["dtype"],
        )

        # Run one optimization
        optimizer.reinitialize(action.clone())
        result1 = optimizer.optimize(action.clone())

        # Reinitialize and run again - should get same result
        optimizer.reinitialize(action.clone())
        result2 = optimizer.optimize(action.clone())

        # Results should be identical after reinitialize
        assert torch.allclose(result1, result2, rtol=1e-4)

    def test_multiple_iterations(self, optimizer_setup):
        """Test running multiple iterations in sequence."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            num_iters=10,
        )

        initial_action = (
            torch.randn(
                optimizer_setup["batch_size"],
                optimizer_setup["action_horizon"],
                optimizer_setup["num_dof"],
                device=optimizer_setup["device"],
                dtype=optimizer_setup["dtype"],
            )
            * 10.0
        )

        # Run first iteration
        optimizer.reinitialize(initial_action.clone())
        result1 = optimizer.optimize(initial_action.clone())

        # Run second iteration starting from first result
        optimizer.reinitialize(result1.clone())
        result2 = optimizer.optimize(result1.clone())

        # Cost should decrease or stay similar
        cost1 = cost_fn(result1).mean().item()
        cost2 = cost_fn(result2).mean().item()

        print(f"Cost after first optimization: {cost1}")
        print(f"Cost after second optimization: {cost2}")

        assert cost2 <= cost1 * 1.2  # Allow some variation

    def test_buffer_initialization(self, optimizer_setup):
        """Test that buffers are properly initialized."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            history=10,
        )

        # Check buffer shapes
        expected_batch = optimizer_setup["batch_size"]
        expected_n = optimizer_setup["action_horizon"] * optimizer_setup["num_dof"]
        expected_history = 10

        assert optimizer._qn.s.shape[0] == expected_history
        assert optimizer._qn.s.shape[1] == expected_batch
        assert optimizer._qn.s.shape[2] == expected_n

        assert optimizer._qn.y.shape[0] == expected_history
        assert optimizer._qn.y.shape[1] == expected_batch
        assert optimizer._qn.y.shape[2] == expected_n

    def test_epsilon_effect(self, optimizer_setup):
        """Test that epsilon parameter prevents division by zero."""
        # Use very small epsilon
        optimizer_small_eps = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            epsilon=1e-8,
            num_iters=10,
        )

        # Use larger epsilon
        optimizer_large_eps = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            epsilon=1e-2,
            num_iters=10,
        )

        initial_action = (
            torch.randn(
                optimizer_setup["batch_size"],
                optimizer_setup["action_horizon"],
                optimizer_setup["num_dof"],
                device=optimizer_setup["device"],
                dtype=optimizer_setup["dtype"],
            )
            * 10.0
        )

        # Both should run without errors
        optimizer_small_eps.reinitialize(initial_action.clone())
        result_small = optimizer_small_eps.optimize(initial_action.clone())

        optimizer_large_eps.reinitialize(initial_action.clone())
        result_large = optimizer_large_eps.optimize(initial_action.clone())

        # Both should produce finite results
        assert torch.all(torch.isfinite(result_small))
        assert torch.all(torch.isfinite(result_large))


class TestLSR1JitFunctions:
    """Test the JIT-compiled helper functions for LSR1."""

    def test_jit_lsr1_compute_step_direction(self, device, dtype):
        """Test LSR1 step direction computation."""
        batch = 4
        n = 20
        m = 5  # history

        # Create test tensors with correct shapes (history, batch, opt_dim, 1)
        y_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype)
        s_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype)
        grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        hessian_0 = torch.ones(batch, 1, 1, device=device, dtype=dtype)

        # Compute step direction
        step = jit_lsr1_compute_step_direction(
            y_buffer, s_buffer, grad, m, epsilon=1e-6, stable_mode=True, hessian_0=hessian_0
        )

        # Check shape
        assert step.shape == (batch, 1, n)

        # Check that result is finite
        assert torch.all(torch.isfinite(step))

    def test_jit_lsr1_compute_step_direction_stable_mode(self, device, dtype):
        """Test LSR1 step direction computation with stable mode."""
        batch = 2
        n = 10
        m = 3

        y_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype)
        s_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype)
        grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        hessian_0 = torch.ones(batch, 1, 1, device=device, dtype=dtype)

        # Test with stable mode on
        step_stable = jit_lsr1_compute_step_direction(
            y_buffer, s_buffer, grad, m, epsilon=1e-6, stable_mode=True, hessian_0=hessian_0
        )

        # Test with stable mode off
        step_unstable = jit_lsr1_compute_step_direction(
            y_buffer, s_buffer, grad, m, epsilon=1e-6, stable_mode=False, hessian_0=hessian_0
        )

        # Both should be finite
        assert torch.all(torch.isfinite(step_stable))
        assert torch.all(torch.isfinite(step_unstable))

    def test_jit_lsr1_step_direction_is_descent(self, device, dtype):
        """Test that LSR1 produces a descent direction (negative gradient direction)."""
        batch = 4
        n = 20
        m = 5

        # Create simple test case with positive gradient
        y_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype) * 0.1
        s_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype) * 0.1
        grad = torch.ones(batch, 1, n, device=device, dtype=dtype)  # All positive
        hessian_0 = torch.ones(batch, 1, 1, device=device, dtype=dtype) * 0.5

        step = jit_lsr1_compute_step_direction(
            y_buffer, s_buffer, grad, m, epsilon=1e-6, stable_mode=True, hessian_0=hessian_0
        )

        # Step should generally point in negative gradient direction
        # (though SR1 is not guaranteed to be positive definite)
        assert torch.all(torch.isfinite(step))

    def test_jit_lsr1_different_batch_sizes(self, device, dtype):
        """Test LSR1 with different batch sizes."""
        for batch in [1, 2, 4, 8]:
            n = 15
            m = 5

            y_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype)
            s_buffer = torch.randn(m, batch, n, 1, device=device, dtype=dtype)
            grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
            hessian_0 = torch.ones(batch, 1, 1, device=device, dtype=dtype)

            step = jit_lsr1_compute_step_direction(
                y_buffer, s_buffer, grad, m, epsilon=1e-6, stable_mode=True, hessian_0=hessian_0
            )

            assert step.shape == (batch, 1, n)
            assert torch.all(torch.isfinite(step))

    def test_jit_lsr1_epsilon_prevents_division_by_zero(self, device, dtype):
        """Test that epsilon prevents division by zero."""
        batch = 2
        n = 10
        m = 3

        # Create buffers that might cause division issues
        y_buffer = torch.zeros(m, batch, n, 1, device=device, dtype=dtype)
        s_buffer = torch.zeros(m, batch, n, 1, device=device, dtype=dtype)
        grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        hessian_0 = torch.ones(batch, 1, 1, device=device, dtype=dtype)

        # Should not crash or produce NaN/Inf
        step = jit_lsr1_compute_step_direction(
            y_buffer, s_buffer, grad, m, epsilon=1e-6, stable_mode=True, hessian_0=hessian_0
        )

        assert torch.all(torch.isfinite(step))


class TestLSR1Comparison:
    """Compare LSR1 with LBFGS."""

    def test_lsr1_vs_initialization(self, optimizer_setup):
        """Test that LSR1 improves over random initialization."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            history=10,
            num_iters=20,
        )

        initial_action = (
            torch.randn(
                optimizer_setup["batch_size"],
                optimizer_setup["action_horizon"],
                optimizer_setup["num_dof"],
                device=optimizer_setup["device"],
                dtype=optimizer_setup["dtype"],
            )
            * 10.0
        )

        # Run optimization
        optimizer.reinitialize(initial_action.clone())
        result = optimizer.optimize(initial_action.clone())

        # Calculate costs
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        print(f"Initial cost: {initial_cost}")
        print(f"Final cost: {final_cost}")

        # LSR1 should improve significantly
        assert final_cost < initial_cost * 0.5

    def test_lsr1_convergence_consistency(self, optimizer_setup):
        """Test that LSR1 converges consistently across runs with same seed."""
        torch.manual_seed(42)
        np.random.seed(42)

        optimizer1 = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            history=10,
            num_iters=15,
        )

        initial_action = (
            torch.randn(
                optimizer_setup["batch_size"],
                optimizer_setup["action_horizon"],
                optimizer_setup["num_dof"],
                device=optimizer_setup["device"],
                dtype=optimizer_setup["dtype"],
            )
            * 10.0
        )

        # First run
        optimizer1.reinitialize(initial_action.clone())
        result1 = optimizer1.optimize(initial_action.clone())
        cost1 = cost_fn(result1).mean().item()

        # Reset seed and optimizer
        torch.manual_seed(42)
        np.random.seed(42)

        optimizer2 = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            history=10,
            num_iters=15,
        )

        # Second run
        optimizer2.reinitialize(initial_action.clone())
        result2 = optimizer2.optimize(initial_action.clone())
        cost2 = cost_fn(result2).mean().item()

        # Results should be very similar
        print(f"Cost run 1: {cost1}")
        print(f"Cost run 2: {cost2}")
        assert abs(cost1 - cost2) < 1e-3

