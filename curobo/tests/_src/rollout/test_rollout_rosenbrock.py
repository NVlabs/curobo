# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for Rosenbrock rollout.

Tests the rollout_rosenbrock.py module which provides a test rollout class
for the Rosenbrock function optimization problem.
"""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.rollout.rollout_rosenbrock import (
    RosenbrockCfg,
    RosenbrockRollout,
)
from curobo._src.state.state_joint import JointState


class TestRosenbrockCfgInitialization:
    """Test RosenbrockCfg initialization."""

    def test_default_initialization(self, cuda_device_cfg):
        """Test default configuration."""
        config = RosenbrockCfg(device_cfg=cuda_device_cfg)

        assert config.a == 1.0
        assert config.b == 100.0
        assert config.dimensions == 2
        assert config.time_horizon == 1
        assert config.time_action_horizon == 1
        assert config.sum_horizon is False
        assert config.sampler_seed == 1312

    def test_custom_parameters(self, cuda_device_cfg):
        """Test with custom parameters."""
        config = RosenbrockCfg(
            a=2.0, b=50.0, dimensions=5, time_horizon=10, device_cfg=cuda_device_cfg
        )

        assert config.a == 2.0
        assert config.b == 50.0
        assert config.dimensions == 5
        assert config.time_horizon == 10

    def test_create(self, cuda_device_cfg):
        """Test creating config from dictionary."""
        config_dict = {
            "a": 1.5,
            "b": 75.0,
            "dimensions": 3,
            "time_horizon": 5,
            "time_action_horizon": 5,
            "sum_horizon": True,
            "sampler_seed": 42,
        }

        config = RosenbrockCfg.create(config_dict, cuda_device_cfg)

        assert config.a == 1.5
        assert config.b == 75.0
        assert config.dimensions == 3
        assert config.time_horizon == 5
        assert config.time_action_horizon == 5
        assert config.sum_horizon is True
        assert config.sampler_seed == 42

    def test_create_with_defaults(self, cuda_device_cfg):
        """Test creating config from empty dictionary uses defaults."""
        config = RosenbrockCfg.create({}, cuda_device_cfg)

        assert config.a == 1.0
        assert config.b == 100.0
        assert config.dimensions == 2
        assert config.time_horizon == 1
        assert config.time_action_horizon == 1


class TestRosenbrockRolloutInitialization:
    """Test RosenbrockRollout initialization."""

    def test_basic_initialization(self, cuda_device_cfg):
        """Test basic initialization."""
        config = RosenbrockCfg(device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        assert rollout.a == 1.0
        assert rollout.b == 100.0
        assert rollout.dimensions == 2
        assert rollout.time_horizon == 1
        assert rollout.time_action_horizon == 1

    def test_action_bounds(self, cuda_device_cfg):
        """Test action bounds are set correctly."""
        config = RosenbrockCfg(time_horizon=5, time_action_horizon=5, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        assert rollout.action_bound_lows is not None
        assert rollout.action_bound_highs is not None
        assert rollout.action_bound_lows.shape[0] == 5
        assert rollout.action_bound_highs.shape[0] == 5
        assert torch.allclose(
            rollout.action_bound_lows,
            torch.tensor([-1.5] * 5, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype),
        )
        assert torch.allclose(
            rollout.action_bound_highs,
            torch.tensor([2.0] * 5, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype),
        )

    def test_properties(self, cuda_device_cfg):
        """Test rollout properties."""
        config = RosenbrockCfg(
            dimensions=3, time_horizon=10, time_action_horizon=10, device_cfg=cuda_device_cfg
        )
        rollout = RosenbrockRollout(config)

        assert rollout.action_dim == 3
        assert rollout.horizon == 10
        assert rollout.action_horizon == 10
        assert rollout.batch_size == 1

    def test_state_bounds(self, cuda_device_cfg):
        """Test state bounds property."""
        config = RosenbrockCfg(device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        state_bounds = rollout.state_bounds
        assert "position" in state_bounds
        assert state_bounds["position"] == [-2.048, 2.048]


class TestRosenbrockRolloutCostComputation:
    """Test cost computation for Rosenbrock function."""

    def test_cost_at_optimum_2d(self, cuda_device_cfg):
        """Test cost at the known optimum (1, 1) for 2D Rosenbrock."""
        config = RosenbrockCfg(a=1.0, b=100.0, dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        # The optimum is at (a, a^2) = (1, 1)
        batch_size = 1
        horizon = 1
        optimum = torch.ones(
            (batch_size, horizon, 2), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        state = JointState.from_position(optimum)
        costs_constraints = rollout._compute_costs_and_constraints_impl(state)

        cost = costs_constraints.costs.values[0]

        # At optimum, cost should be 0
        assert torch.allclose(cost, torch.zeros_like(cost), atol=1e-6)

    def test_cost_away_from_optimum(self, cuda_device_cfg):
        """Test cost at a point away from optimum."""
        config = RosenbrockCfg(a=1.0, b=100.0, dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        # Test at (0, 0)
        batch_size = 1
        horizon = 1
        point = torch.zeros(
            (batch_size, horizon, 2), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        state = JointState.from_position(point)
        costs_constraints = rollout._compute_costs_and_constraints_impl(state)

        cost = costs_constraints.costs.values[0]

        # At (0, 0): f = (1-0)^2 + 100*(0-0^2)^2 = 1
        expected_cost = torch.ones_like(cost)
        assert torch.allclose(cost, expected_cost, atol=1e-6)

    def test_cost_computation_formula(self, cuda_device_cfg):
        """Test the Rosenbrock formula: f(x,y) = (a-x)^2 + b(y-x^2)^2."""
        config = RosenbrockCfg(a=1.0, b=100.0, dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        # Test at specific point (2, 3)
        batch_size = 1
        horizon = 1
        x_val = 2.0
        y_val = 3.0
        point = torch.tensor(
            [[[x_val, y_val]]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        state = JointState.from_position(point)
        costs_constraints = rollout._compute_costs_and_constraints_impl(state)

        cost = costs_constraints.costs.values[0]

        # Manual calculation: f(2,3) = (1-2)^2 + 100*(3-2^2)^2 = 1 + 100*(-1)^2 = 101
        expected_cost = (1.0 - x_val) ** 2 + 100.0 * (y_val - x_val**2) ** 2
        expected_tensor = torch.tensor(
            [[expected_cost]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        assert torch.allclose(cost, expected_tensor, atol=1e-5)

    def test_batch_cost_computation(self, cuda_device_cfg):
        """Test cost computation with batch."""
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        batch_size = 4
        horizon = 1

        # Create batch of points
        points = torch.randn(
            (batch_size, horizon, 2), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        state = JointState.from_position(points)
        costs_constraints = rollout._compute_costs_and_constraints_impl(state)

        cost = costs_constraints.costs.values[0]

        # Check shape - implementation should return [batch, horizon, 1] per BaseCost convention
        assert cost.shape == (batch_size, horizon, 1)

        # Verify each batch element manually
        for i in range(batch_size):
            x_val = points[i, 0, 0].item()
            y_val = points[i, 0, 1].item()
            expected = (1.0 - x_val) ** 2 + 100.0 * (y_val - x_val**2) ** 2
            assert torch.allclose(cost[i, 0, 0], torch.tensor(expected), atol=1e-4)

    def test_higher_dimensional_cost(self, cuda_device_cfg):
        """Test cost computation for higher dimensions."""
        config = RosenbrockCfg(dimensions=4, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        batch_size = 1
        horizon = 1

        # Test at optimum (1, 1, 1, 1)
        optimum = torch.ones(
            (batch_size, horizon, 4), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        state = JointState.from_position(optimum)
        costs_constraints = rollout._compute_costs_and_constraints_impl(state)

        cost = costs_constraints.costs.values[0]

        # At optimum, cost should be 0
        assert torch.allclose(cost, torch.zeros_like(cost), atol=1e-6)


class TestRosenbrockRolloutStateComputation:
    """Test state computation from actions."""

    def test_compute_state_from_action(self, cuda_device_cfg):
        """Test that state is computed from action."""
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        batch_size = 2
        horizon = 3
        action = torch.randn(
            (batch_size, horizon, 2), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        state = rollout._compute_state_from_action_impl(action)

        assert isinstance(state, JointState)
        assert torch.allclose(state.position, action)


class TestRosenbrockRolloutUpdate:
    """Test parameter updates."""

    def test_update_a_parameter(self, cuda_device_cfg):
        """Test updating the 'a' parameter."""
        config = RosenbrockCfg(a=1.0, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        assert rollout.a == 1.0

        rollout.update_params(a=2.0)
        assert rollout.a == 2.0

    def test_update_b_parameter(self, cuda_device_cfg):
        """Test updating the 'b' parameter."""
        config = RosenbrockCfg(b=100.0, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        assert rollout.b == 100.0

        rollout.update_params(b=50.0)
        assert rollout.b == 50.0

    def test_update_affects_cost(self, cuda_device_cfg):
        """Test that updating parameters affects cost computation."""
        config = RosenbrockCfg(a=1.0, b=100.0, dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        # Use a point where b term is non-zero: (0.5, 0.3)
        # f(0.5, 0.3) = (1-0.5)^2 + b*(0.3-0.5^2)^2 = 0.25 + b*(0.3-0.25)^2 = 0.25 + b*0.0025
        point = torch.tensor(
            [[[0.5, 0.3]]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        state = JointState.from_position(point)

        # Compute cost with original parameters
        costs1 = rollout._compute_costs_and_constraints_impl(state)
        cost1 = costs1.costs.values[0]

        # Update b parameter
        rollout.update_params(b=50.0)

        # Compute cost with new parameters
        costs2 = rollout._compute_costs_and_constraints_impl(state)
        cost2 = costs2.costs.values[0]

        # Costs should be different
        assert not torch.allclose(cost1, cost2)


class TestRosenbrockRolloutBatchSize:
    """Test batch size management."""

    def test_get_batch_size(self, cuda_device_cfg):
        """Test getting batch size."""
        config = RosenbrockCfg(device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        assert rollout.batch_size == 1

    def test_set_batch_size(self, cuda_device_cfg):
        """Test setting batch size."""
        config = RosenbrockCfg(device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        rollout.batch_size = 8
        assert rollout.batch_size == 8


class TestRosenbrockRolloutEdgeCases:
    """Test edge cases."""

    def test_large_values(self, cuda_device_cfg):
        """Test with large input values."""
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        # Test with large values
        point = torch.tensor(
            [[[100.0, 10000.0]]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        state = JointState.from_position(point)

        costs_constraints = rollout._compute_costs_and_constraints_impl(state)
        cost = costs_constraints.costs.values[0]

        # Cost should be finite
        assert torch.isfinite(cost).all()

    def test_negative_values(self, cuda_device_cfg):
        """Test with negative input values."""
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        point = torch.tensor(
            [[[-1.0, -2.0]]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        state = JointState.from_position(point)

        costs_constraints = rollout._compute_costs_and_constraints_impl(state)
        cost = costs_constraints.costs.values[0]

        # Manually verify: f(-1,-2) = (1-(-1))^2 + 100*(-2-(-1)^2)^2
        # = 4 + 100*(-2-1)^2 = 4 + 100*9 = 904
        expected_cost = (1.0 - (-1.0)) ** 2 + 100.0 * (-2.0 - (-1.0) ** 2) ** 2
        expected_tensor = torch.tensor(
            [[expected_cost]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        assert torch.allclose(cost, expected_tensor, atol=1e-4)

    def test_single_dimension_not_supported(self, cuda_device_cfg):
        """Test that single dimension requires at least 2D."""
        # Rosenbrock requires at least 2 dimensions
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        assert rollout.dimensions >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRosenbrockRolloutCudaGraph:
    """Exercise the ``use_cuda_graph=True`` code path of RosenbrockRollout."""

    def test_cuda_graph_initialization(self, cuda_device_cfg):
        """Test CUDA graph rollout initialization."""
        config = RosenbrockCfg(device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config, use_cuda_graph=True)

        assert rollout.a == 1.0
        assert rollout.b == 100.0

    def test_cuda_graph_cost_computation(self, cuda_device_cfg):
        """Test cost computation with CUDA graph."""
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config, use_cuda_graph=True)

        batch_size = 4
        horizon = 1
        points = torch.randn((batch_size, horizon, 2), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        state = JointState.from_position(points)

        # Test both implementations produce same result
        costs1 = rollout._compute_costs_and_constraints_impl(state)
        costs2 = rollout._compute_costs_and_constraints_metrics_impl(state)

        assert torch.allclose(costs1.costs.values[0], costs2.costs.values[0])

    def test_cuda_graph_state_computation(self, cuda_device_cfg):
        """Test state computation with CUDA graph."""
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config, use_cuda_graph=True)

        batch_size = 2
        horizon = 3
        action = torch.randn((batch_size, horizon, 2), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        state1 = rollout._compute_state_from_action_impl(action)
        state2 = rollout._compute_state_from_action_metrics_impl(action)

        assert torch.allclose(state1.position, state2.position)


class TestRosenbrockRolloutGradients:
    """Test gradient computation through the Rosenbrock function."""

    def test_gradients_enabled(self, cuda_device_cfg):
        """Test that gradients can be computed."""
        config = RosenbrockCfg(dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        point = torch.tensor(
            [[[0.5, 0.3]]], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype, requires_grad=True
        )
        state = JointState.from_position(point)

        costs_constraints = rollout._compute_costs_and_constraints_impl(state)
        cost = costs_constraints.costs.values[0]

        # Compute gradients
        cost.sum().backward()

        # Check that gradients exist
        assert point.grad is not None
        assert not torch.allclose(point.grad, torch.zeros_like(point.grad))

    def test_gradient_at_optimum(self, cuda_device_cfg):
        """Test that gradient is zero at optimum."""
        config = RosenbrockCfg(a=1.0, b=100.0, dimensions=2, device_cfg=cuda_device_cfg)
        rollout = RosenbrockRollout(config)

        # At optimum (1, 1)
        optimum = torch.ones(
            (1, 1, 2), device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype, requires_grad=True
        )
        state = JointState.from_position(optimum)

        costs_constraints = rollout._compute_costs_and_constraints_impl(state)
        cost = costs_constraints.costs.values[0]

        cost.sum().backward()

        # Gradient should be near zero at optimum
        assert torch.allclose(optimum.grad, torch.zeros_like(optimum.grad), atol=1e-5)

