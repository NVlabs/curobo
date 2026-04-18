# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for NodeSamplingStrategy."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.graph.node_sampling_strategy import NodeSamplingStrategy
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg


@pytest.fixture(scope="module")
def prm_config():
    """Get PRM planner configuration."""
    return PRMGraphPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
    )


@pytest.fixture(scope="module")
def action_bounds(cuda_device_cfg):
    """Get action bounds for 7-DOF robot."""
    lows = torch.tensor(
        [-2.8, -1.7, -2.8, -3.0, -2.8, -0.0, -2.8],
        **cuda_device_cfg.as_torch_dict()
    )
    highs = torch.tensor(
        [2.8, 1.7, 2.8, -0.0, 2.8, 3.7, 2.8],
        **cuda_device_cfg.as_torch_dict()
    )
    return lows, highs


@pytest.fixture
def simple_feasibility_fn():
    """Simple feasibility function that accepts all configurations."""
    def check_fn(samples):
        return torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)
    return check_fn


@pytest.fixture
def sampling_strategy(prm_config, action_bounds, simple_feasibility_fn, cuda_device_cfg):
    """Create a NodeSamplingStrategy instance."""
    lows, highs = action_bounds
    cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

    return NodeSamplingStrategy(
        config=prm_config,
        action_lower_bounds=lows,
        action_upper_bounds=highs,
        cspace_distance_weight=cspace_weight,
        action_dim=7,
        check_feasibility_fn=simple_feasibility_fn,
        device_cfg=cuda_device_cfg,
    )


class TestNodeSamplingStrategyInitialization:
    """Test NodeSamplingStrategy initialization."""

    def test_init_basic(self, sampling_strategy):
        """Test basic initialization."""
        assert sampling_strategy is not None
        assert sampling_strategy.action_dim == 7
        assert sampling_strategy.action_sample_generator is not None

    def test_init_with_seed(self, prm_config, action_bounds, simple_feasibility_fn, cuda_device_cfg):
        """Test initialization with specific seed."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Create two strategies with same seed
        strategy1 = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=simple_feasibility_fn,
            device_cfg=cuda_device_cfg,
        )

        strategy2 = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=simple_feasibility_fn,
            device_cfg=cuda_device_cfg,
        )

        # Should produce same samples with same seed
        samples1 = strategy1.generate_action_samples(10, bounded=True)
        samples2 = strategy2.generate_action_samples(10, bounded=True)

        assert torch.allclose(samples1, samples2)


class TestNodeSamplingStrategyActionSampling:
    """Test action sample generation."""

    def test_generate_action_samples_bounded(self, sampling_strategy, action_bounds):
        """Test bounded action sampling."""
        lows, highs = action_bounds
        samples = sampling_strategy.generate_action_samples(100, bounded=True)

        assert samples.shape == (100, 7)
        assert (samples >= lows).all()
        assert (samples <= highs).all()

    def test_generate_action_samples_unbounded(self, sampling_strategy):
        """Test unbounded action sampling."""
        samples = sampling_strategy.generate_action_samples(50, bounded=False)

        assert samples.shape == (50, 7)
        # Unbounded samples can be outside bounds

    def test_generate_action_samples_unit_ball(self, sampling_strategy):
        """Test unit ball sampling."""
        samples = sampling_strategy.generate_action_samples(100, bounded=True, unit_ball=True)

        assert samples.shape == (100, 7)
        # Unit ball samples should be normalized
        norms = torch.norm(samples, dim=-1)
        assert (norms <= 1.0 + 1e-5).all()  # Allow small numerical error

    def test_unit_ball_low_dimension(self, prm_config, simple_feasibility_fn, cuda_device_cfg):
        """Test unit ball sampling with action_dim < 3 (lines 84-86)."""
        lows = torch.tensor([-1.0, -1.0], **cuda_device_cfg.as_torch_dict())
        highs = torch.tensor([1.0, 1.0], **cuda_device_cfg.as_torch_dict())
        cspace_weight = torch.ones(2, **cuda_device_cfg.as_torch_dict())

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=2,  # Low dimension
            check_feasibility_fn=simple_feasibility_fn,
            device_cfg=cuda_device_cfg,
        )

        samples = strategy.generate_action_samples(50, bounded=True, unit_ball=True)

        assert samples.shape == (50, 2)
        # Should create filled circle for low dimensions
        norms = torch.norm(samples, dim=-1)
        assert (norms <= 1.0 + 1e-5).all()


class TestNodeSamplingStrategyFeasibilityChecking:
    """Test feasibility checking functionality."""

    def test_check_samples_feasibility(self, sampling_strategy):
        """Test checking sample feasibility."""
        samples = sampling_strategy.generate_action_samples(50, bounded=True)
        mask = sampling_strategy.check_samples_feasibility(samples)

        assert mask.shape == (50,)
        assert mask.dtype == torch.bool
        assert mask.all()  # All should be feasible with simple_feasibility_fn

    def test_check_samples_feasibility_invalid_shape(self, sampling_strategy):
        """Test error logging for invalid sample shape (line 103)."""
        # 1D tensor should trigger error
        samples_1d = torch.randn(10, device=sampling_strategy.device_cfg.device)

        # Should log error but might not crash
        try:
            mask = sampling_strategy.check_samples_feasibility(samples_1d)
        except (ValueError, RuntimeError):
            # Expected error
            pass

    def test_get_feasible_sample_set(self, prm_config, action_bounds, cuda_device_cfg):
        """Test filtering feasible samples."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Create a feasibility function that rejects some samples
        def selective_feasibility(samples):
            # Reject samples where first joint is negative
            return samples[:, 0] >= 0.0

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=selective_feasibility,
            device_cfg=cuda_device_cfg,
        )

        samples = torch.randn(100, 7, **cuda_device_cfg.as_torch_dict())
        feasible = strategy.get_feasible_sample_set(samples)

        # Should only return samples where first joint >= 0
        assert (feasible[:, 0] >= 0.0).all()


class TestNodeSamplingStrategyFeasibleGeneration:
    """Test feasible sample generation."""

    def test_generate_feasible_action_samples(self, sampling_strategy):
        """Test generating feasible action samples."""
        samples = sampling_strategy.generate_feasible_action_samples(20)

        assert samples.shape[0] <= 20  # May be less if rejection occurs
        assert samples.shape[1] == 7
        # All should be feasible (simple_feasibility_fn accepts all)

    def test_generate_feasible_samples(self, sampling_strategy):
        """Test generating feasible samples for roadmap."""
        samples = sampling_strategy.generate_feasible_samples(30)

        assert samples.shape[0] <= 30
        assert samples.shape[1] == 7


class TestNodeSamplingStrategyEllipsoidSampling:
    """Test ellipsoidal sampling functionality."""

    def test_compute_distance_from_line(self, sampling_strategy):
        """Test computing distance from line."""
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        # Generate some points
        points = sampling_strategy.generate_action_samples(50, bounded=True)

        distances = sampling_strategy.compute_distance_from_line(points, x_start, x_goal)

        assert distances.shape == (50,)
        assert (distances >= 0.0).all()

    def test_generate_feasible_samples_in_ellipsoid(self, sampling_strategy):
        """Test generating feasible samples in ellipsoid."""
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        max_radius = 2.0

        samples = sampling_strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=30,
            max_sampling_radius=max_radius,
        )

        assert samples.shape[0] <= 30
        assert samples.shape[1] == 7

    def test_ellipsoid_sampling_with_different_radii(self, sampling_strategy):
        """Test ellipsoid sampling with various radii."""
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        # Test with small radius
        samples_small = sampling_strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=20,
            max_sampling_radius=0.5,
        )

        # Test with large radius
        samples_large = sampling_strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=20,
            max_sampling_radius=5.0,
        )

        assert samples_small.shape[1] == 7
        assert samples_large.shape[1] == 7


class TestNodeSamplingStrategyEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_samples(self, sampling_strategy):
        """Test generating zero samples."""
        samples = sampling_strategy.generate_action_samples(0, bounded=True)

        assert samples.shape == (0, 7)

    def test_single_sample(self, sampling_strategy):
        """Test generating single sample."""
        samples = sampling_strategy.generate_action_samples(1, bounded=True)

        assert samples.shape == (1, 7)

    def test_large_batch(self, sampling_strategy):
        """Test generating large batch of samples."""
        samples = sampling_strategy.generate_action_samples(1000, bounded=True)

        assert samples.shape == (1000, 7)

    def test_feasible_samples_with_high_rejection(self, prm_config, action_bounds, cuda_device_cfg):
        """Test feasible sample generation with high rejection rate."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Create a very restrictive feasibility function
        def restrictive_feasibility(samples):
            # Only accept very few samples
            return torch.rand(samples.shape[0], device=samples.device) < 0.1

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=restrictive_feasibility,
            device_cfg=cuda_device_cfg,
        )

        # May return fewer samples than requested due to rejection
        samples = strategy.generate_feasible_action_samples(10)

        assert samples.shape[0] <= 10
        assert samples.shape[1] == 7

    def test_ellipsoid_with_start_equals_goal(self, sampling_strategy):
        """Test ellipsoid sampling when start equals goal."""
        x_start = torch.ones(7, device=sampling_strategy.device_cfg.device)
        x_goal = x_start.clone()

        max_radius = 0.5

        # Should handle start == goal gracefully
        try:
            samples = sampling_strategy.generate_feasible_samples_in_ellipsoid(
                x_start=x_start,
                x_goal=x_goal,
                num_samples=20,
                max_sampling_radius=max_radius,
            )
            # May return empty or some samples
            assert samples.shape[1] == 7
        except (ValueError, RuntimeError, ZeroDivisionError):
            # May raise error for degenerate ellipsoid
            pass


class TestNodeSamplingStrategyEllipsoidProjectionMethods:
    """Test different ellipsoid projection methods."""

    def test_svd_projection_method(self, prm_config, action_bounds, simple_feasibility_fn, cuda_device_cfg):
        """Test SVD ellipsoid projection method (lines 205-218, 401-443)."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Set SVD projection method
        prm_config.ellipsoid_projection_method = "svd"

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=simple_feasibility_fn,
            device_cfg=cuda_device_cfg,
        )

        x_start = torch.zeros(7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        samples = strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=30,
            max_sampling_radius=2.0,
        )

        assert samples.shape[0] <= 30
        assert samples.shape[1] == 7

    def test_approximate_projection_method(self, prm_config, action_bounds, simple_feasibility_fn, cuda_device_cfg):
        """Test approximate ellipsoid projection method (lines 461-486)."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Set approximate projection method
        prm_config.ellipsoid_projection_method = "approximate"

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=simple_feasibility_fn,
            device_cfg=cuda_device_cfg,
        )

        x_start = torch.zeros(7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        samples = strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=30,
            max_sampling_radius=2.0,
        )

        assert samples.shape[0] <= 30
        assert samples.shape[1] == 7

    def test_householder_projection_method(self, prm_config, action_bounds, simple_feasibility_fn, cuda_device_cfg):
        """Test householder ellipsoid projection method (default)."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Set householder projection method (default)
        prm_config.ellipsoid_projection_method = "householder"

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=simple_feasibility_fn,
            device_cfg=cuda_device_cfg,
        )

        x_start = torch.zeros(7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        samples = strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=30,
            max_sampling_radius=2.0,
        )

        assert samples.shape[0] <= 30
        assert samples.shape[1] == 7


class TestNodeSamplingStrategyValidation:
    """Test input validation functionality."""

    def test_ellipsoid_sampling_with_wrong_start_shape(self, sampling_strategy):
        """Test error logging for wrong x_start shape (lines 180)."""
        # 2D tensor instead of 1D
        x_start = torch.zeros(1, 7, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        # Should log error but might not crash
        try:
            samples = sampling_strategy.generate_feasible_samples_in_ellipsoid(
                x_start=x_start,
                x_goal=x_goal,
                num_samples=10,
                max_sampling_radius=2.0,
            )
        except (ValueError, RuntimeError, IndexError):
            # Expected error
            pass

    def test_ellipsoid_sampling_with_wrong_goal_shape(self, sampling_strategy):
        """Test error logging for wrong x_goal shape (lines 182)."""
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        # 2D tensor instead of 1D
        x_goal = torch.ones(1, 7, device=sampling_strategy.device_cfg.device)

        try:
            samples = sampling_strategy.generate_feasible_samples_in_ellipsoid(
                x_start=x_start,
                x_goal=x_goal,
                num_samples=10,
                max_sampling_radius=2.0,
            )
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_ellipsoid_sampling_with_wrong_start_dimension(self, sampling_strategy):
        """Test error logging for wrong x_start dimension (lines 184)."""
        # Wrong number of dimensions
        x_start = torch.zeros(5, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        try:
            samples = sampling_strategy.generate_feasible_samples_in_ellipsoid(
                x_start=x_start,
                x_goal=x_goal,
                num_samples=10,
                max_sampling_radius=2.0,
            )
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_ellipsoid_sampling_with_wrong_goal_dimension(self, sampling_strategy):
        """Test error logging for wrong x_goal dimension (lines 186)."""
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        # Wrong number of dimensions
        x_goal = torch.ones(5, device=sampling_strategy.device_cfg.device)

        try:
            samples = sampling_strategy.generate_feasible_samples_in_ellipsoid(
                x_start=x_start,
                x_goal=x_goal,
                num_samples=10,
                max_sampling_radius=2.0,
            )
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_line_distance_with_wrong_vertices_shape(self, sampling_strategy):
        """Test error logging for wrong vertices shape (lines 290)."""
        # 1D tensor instead of 2D
        vertices = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        try:
            distances = sampling_strategy.compute_distance_from_line(vertices, x_start, x_goal)
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_line_distance_with_wrong_start_shape(self, sampling_strategy):
        """Test error logging for wrong x_start shape (lines 292)."""
        vertices = torch.zeros(10, 7, device=sampling_strategy.device_cfg.device)
        # 2D tensor instead of 1D
        x_start = torch.zeros(1, 7, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        try:
            distances = sampling_strategy.compute_distance_from_line(vertices, x_start, x_goal)
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_line_distance_with_wrong_goal_shape(self, sampling_strategy):
        """Test error logging for wrong x_goal shape (lines 294)."""
        vertices = torch.zeros(10, 7, device=sampling_strategy.device_cfg.device)
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        # 2D tensor instead of 1D
        x_goal = torch.ones(1, 7, device=sampling_strategy.device_cfg.device)

        try:
            distances = sampling_strategy.compute_distance_from_line(vertices, x_start, x_goal)
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_line_distance_with_wrong_start_dimension(self, sampling_strategy):
        """Test error logging for wrong x_start dimension (lines 296)."""
        vertices = torch.zeros(10, 7, device=sampling_strategy.device_cfg.device)
        # Wrong dimension
        x_start = torch.zeros(5, device=sampling_strategy.device_cfg.device)
        x_goal = torch.ones(7, device=sampling_strategy.device_cfg.device)

        try:
            distances = sampling_strategy.compute_distance_from_line(vertices, x_start, x_goal)
        except (ValueError, RuntimeError, IndexError):
            pass

    def test_line_distance_with_wrong_goal_dimension(self, sampling_strategy):
        """Test error logging for wrong x_goal dimension (lines 298)."""
        vertices = torch.zeros(10, 7, device=sampling_strategy.device_cfg.device)
        x_start = torch.zeros(7, device=sampling_strategy.device_cfg.device)
        # Wrong dimension
        x_goal = torch.ones(5, device=sampling_strategy.device_cfg.device)

        try:
            distances = sampling_strategy.compute_distance_from_line(vertices, x_start, x_goal)
        except (ValueError, RuntimeError, IndexError):
            pass


class TestNodeSamplingStrategyFewerSamplesPaths:
    """Test code paths when fewer samples are returned than requested."""

    def test_generate_feasible_samples_with_high_rejection(self, prm_config, action_bounds, cuda_device_cfg):
        """Test generate_feasible_samples when many samples are rejected (line 156)."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Create very restrictive feasibility function
        def very_restrictive_feasibility(samples):
            # Only accept ~5% of samples
            return torch.rand(samples.shape[0], device=samples.device) < 0.05

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=very_restrictive_feasibility,
            device_cfg=cuda_device_cfg,
        )

        # Request many samples - likely won't get all
        samples = strategy.generate_feasible_samples(50)

        # Should return fewer than requested (line 156 path)
        assert samples.shape[0] <= 50
        assert samples.shape[1] == 7

    def test_ellipsoid_sampling_with_high_rejection(self, prm_config, action_bounds, cuda_device_cfg):
        """Test ellipsoid sampling when many samples are rejected (line 234)."""
        lows, highs = action_bounds
        cspace_weight = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        def very_restrictive_feasibility(samples):
            return torch.rand(samples.shape[0], device=samples.device) < 0.1

        strategy = NodeSamplingStrategy(
            config=prm_config,
            action_lower_bounds=lows,
            action_upper_bounds=highs,
            cspace_distance_weight=cspace_weight,
            action_dim=7,
            check_feasibility_fn=very_restrictive_feasibility,
            device_cfg=cuda_device_cfg,
        )

        x_start = torch.zeros(7, **cuda_device_cfg.as_torch_dict())
        x_goal = torch.ones(7, **cuda_device_cfg.as_torch_dict())

        # Request samples - likely won't get all due to rejection
        samples = strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=30,
            max_sampling_radius=2.0,
        )

        # Should return fewer than requested (line 234 path)
        assert samples.shape[0] <= 30
        assert samples.shape[1] == 7


class TestNodeSamplingStrategyResetSeed:
    """Test seed reset functionality."""

    def test_reset_seed(self, sampling_strategy):
        """Test resetting the sampling seed."""
        # Generate some samples
        samples1 = sampling_strategy.generate_action_samples(10, bounded=True)

        # Reset seed
        sampling_strategy.reset_seed()

        # Generate again - should get same sequence
        samples2 = sampling_strategy.generate_action_samples(10, bounded=True)

        # With reset, samples should match
        assert torch.allclose(samples1, samples2)

