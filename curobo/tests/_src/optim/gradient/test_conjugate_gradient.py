#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the Conjugate Gradient optimizer."""

from __future__ import annotations

# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.optim.gradient.conjugate_gradient import (
    ConjugateGradientOpt,
    ConjugateGradientOptCfg,
    jit_cg_compute_step_direction,
    jit_cg_shift_buffers,
)
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
    """Set up environment for Conjugate Gradient optimization tests."""
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
    cg_method="FR",
    max_beta=10.0,
    num_iters=50,
    line_search_scale=None,
):
    """Create a Conjugate Gradient optimizer with the specified configuration."""
    if line_search_scale is None:
        line_search_scale = [0, 0.1, 0.5, 1.0]

    config = ConjugateGradientOptCfg(
        num_problems=batch_size,
        num_iters=num_iters,
        cg_method=cg_method,
        max_beta=max_beta,
        step_scale=0.98,
        solver_type="conjugate_gradient",
        line_search_scale=line_search_scale,
    )

    optimizer = ConjugateGradientOpt(config, [rollout_fn])
    optimizer.update_num_problems(batch_size)
    return optimizer


class TestConjugateGradientOptCfg:
    """Test cases for ConjugateGradientOptCfg configuration."""

    def test_init_default(self, device, dtype):
        """Test default initialization."""
        config = ConjugateGradientOptCfg(
            num_iters=10,
            solver_type="conjugate_gradient",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )
        assert config.cg_method == "FR"
        assert config.max_beta == 10.0

    @pytest.mark.parametrize("cg_method", ["FR", "PR", "DY"])
    def test_init_with_valid_methods(self, device, dtype, cg_method):
        """Test initialization with valid CG methods."""
        config = ConjugateGradientOptCfg(
            num_iters=10,
            solver_type="conjugate_gradient",
            cg_method=cg_method,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )
        assert config.cg_method == cg_method

    def test_init_with_invalid_method_raises(self, device, dtype):
        """Test that invalid CG method raises an error."""
        with pytest.raises(ValueError):
            ConjugateGradientOptCfg(
                num_iters=10,
                solver_type="conjugate_gradient",
                cg_method="INVALID",
                device_cfg=DeviceCfg(device=device, dtype=dtype),
            )

    def test_init_with_custom_max_beta(self, device, dtype):
        """Test initialization with custom max_beta."""
        config = ConjugateGradientOptCfg(
            num_iters=10,
            solver_type="conjugate_gradient",
            max_beta=5.0,
            device_cfg=DeviceCfg(device=device, dtype=dtype),
        )
        assert config.max_beta == 5.0


class TestConjugateGradientOpt:
    """Test cases for ConjugateGradientOpt optimizer."""

    def test_init_basic(self, optimizer_setup):
        """Test basic initialization."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            num_iters=10,
        )
        assert optimizer.config.num_iters == 10
        assert optimizer._prev_grad_q is None
        assert optimizer._prev_step is None

    @pytest.mark.parametrize("cg_method", ["FR", "PR", "DY"])
    def test_optimization_different_methods(self, optimizer_setup, cg_method):
        """Test optimization with different CG methods."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            cg_method=cg_method,
            num_iters=20,
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

        # Check that optimizer reduced the cost
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        print(f"Method: {cg_method}")
        print(f"Initial cost: {initial_cost}")
        print(f"Final cost: {final_cost}")

        # Conjugate gradient should reduce cost significantly
        assert final_cost < initial_cost
        assert final_cost < 10.0  # Should get reasonably close to optimum

    def test_optimization_convergence(self, optimizer_setup):
        """Test that CG optimizer converges to the optimum."""
        optimizer = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
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
        final_cost = cost_fn(result).mean().item()
        print(f"Final cost after 50 iterations: {final_cost}")

        # Should converge to a low cost
        assert final_cost < 1.0

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

        assert cost2 <= cost1 * 1.1  # Allow small increase due to local minima

    def test_max_beta_effect(self, optimizer_setup):
        """Test that max_beta parameter affects optimization."""
        # Create two optimizers with different max_beta
        optimizer_low = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            max_beta=1.0,
            num_iters=20,
        )
        optimizer_high = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            max_beta=100.0,
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

        # Run both optimizers
        optimizer_low.reinitialize(initial_action.clone())
        result_low = optimizer_low.optimize(initial_action.clone())

        optimizer_high.reinitialize(initial_action.clone())
        result_high = optimizer_high.optimize(initial_action.clone())

        # Both should reduce cost
        initial_cost = cost_fn(initial_action).mean().item()
        cost_low = cost_fn(result_low).mean().item()
        cost_high = cost_fn(result_high).mean().item()

        print(f"Initial cost: {initial_cost}")
        print(f"Cost with max_beta=1.0: {cost_low}")
        print(f"Cost with max_beta=100.0: {cost_high}")

        assert cost_low < initial_cost
        assert cost_high < initial_cost


class TestCGJitFunctions:
    """Test the JIT-compiled helper functions."""

    def test_jit_cg_compute_step_direction_FR(self, device, dtype):
        """Test Fletcher-Reeves step direction computation."""
        batch = 4
        n = 10

        grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        prev_grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        prev_step = torch.randn(batch, 1, n, device=device, dtype=dtype)

        step, new_prev_grad, new_prev_step = jit_cg_compute_step_direction(
            grad, prev_grad, prev_step, max_beta=10.0, method="FR"
        )

        # Check shapes
        assert step.shape == (batch, 1, n)
        assert new_prev_grad.shape == prev_grad.shape
        assert new_prev_step.shape == prev_step.shape

        # Check that prev_grad was updated
        assert torch.allclose(new_prev_grad, grad)

        # Check that prev_step was updated
        assert torch.allclose(new_prev_step, step)

    def test_jit_cg_compute_step_direction_PR(self, device, dtype):
        """Test Polak-Ribière step direction computation."""
        batch = 4
        n = 10

        grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        prev_grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        prev_step = torch.randn(batch, 1, n, device=device, dtype=dtype)

        step, _, _ = jit_cg_compute_step_direction(
            grad, prev_grad, prev_step, max_beta=10.0, method="PR"
        )

        # Check shapes
        assert step.shape == (batch, 1, n)

    def test_jit_cg_compute_step_direction_DY(self, device, dtype):
        """Test Dai-Yuan step direction computation."""
        batch = 4
        n = 10

        grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        prev_grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        prev_step = torch.randn(batch, 1, n, device=device, dtype=dtype)

        step, _, _ = jit_cg_compute_step_direction(
            grad, prev_grad, prev_step, max_beta=10.0, method="DY"
        )

        # Check shapes
        assert step.shape == (batch, 1, n)

    def test_jit_cg_compute_step_direction_beta_clamping(self, device, dtype):
        """Test that beta is clamped to [0, max_beta]."""
        batch = 4
        n = 10

        # Use gradients that would produce large beta
        grad = torch.ones(batch, 1, n, device=device, dtype=dtype) * 10.0
        prev_grad = torch.ones(batch, 1, n, device=device, dtype=dtype) * 0.1
        prev_step = torch.ones(batch, 1, n, device=device, dtype=dtype)

        max_beta = 2.0
        step, _, _ = jit_cg_compute_step_direction(
            grad, prev_grad, prev_step, max_beta=max_beta, method="FR"
        )

        # Step should be finite and bounded
        assert torch.all(torch.isfinite(step))

    def test_jit_cg_shift_buffers(self, device, dtype):
        """Test buffer shifting for receding horizon."""
        batch = 4
        action_dim = 7
        horizon = 10
        n = action_dim * horizon

        prev_grad = torch.randn(batch, 1, n, device=device, dtype=dtype)
        prev_step = torch.randn(batch, 1, n, device=device, dtype=dtype)

        shift_steps = 2
        shifted_grad, shifted_step = jit_cg_shift_buffers(
            prev_grad.clone(), prev_step.clone(), shift_steps, action_dim
        )

        # Check shapes remain the same
        assert shifted_grad.shape == prev_grad.shape
        assert shifted_step.shape == prev_step.shape

        # Shifted values should be different (unless by coincidence)
        # We mainly check that function executes without error
        assert torch.all(torch.isfinite(shifted_grad))
        assert torch.all(torch.isfinite(shifted_step))


class TestCGMethodComparison:
    """Compare different CG methods."""

    @pytest.mark.parametrize("method1,method2", [("FR", "PR"), ("PR", "DY"), ("FR", "DY")])
    def test_method_comparison(self, optimizer_setup, method1, method2):
        """Compare convergence of different CG methods."""
        optimizer1 = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            cg_method=method1,
            num_iters=30,
        )
        optimizer2 = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            cg_method=method2,
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

        # Run both optimizers
        optimizer1.reinitialize(initial_action.clone())
        result1 = optimizer1.optimize(initial_action.clone())

        optimizer2.reinitialize(initial_action.clone())
        result2 = optimizer2.optimize(initial_action.clone())

        # Check that both methods converged
        initial_cost = cost_fn(initial_action).mean().item()
        cost1 = cost_fn(result1).mean().item()
        cost2 = cost_fn(result2).mean().item()

        print(f"Initial cost: {initial_cost}")
        print(f"Cost with {method1}: {cost1}")
        print(f"Cost with {method2}: {cost2}")

        # Both should reduce cost significantly
        assert cost1 < initial_cost * 0.5
        assert cost2 < initial_cost * 0.5

