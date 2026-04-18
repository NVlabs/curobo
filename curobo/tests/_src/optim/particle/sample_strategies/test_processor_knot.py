# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for KnotParticleProcessor."""

import pytest
import torch

# CuRobo
from curobo._src.optim.particle.sample_strategies.processor_knot import KnotParticleProcessor
from curobo._src.types.device_cfg import DeviceCfg


class TestKnotParticleProcessor:
    """Test cases for KnotParticleProcessor implementation."""

    @pytest.fixture
    def device_cfg(self):
        """Fixture for tensor arguments."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def basic_processor(self, device_cfg):
        """Fixture for basic knot processor."""
        return KnotParticleProcessor(
            horizon=10,
            action_dim=3,
            n_knots=4,
            degree=3,
            device_cfg=device_cfg,
        )

    @pytest.fixture
    def linear_processor(self, device_cfg):
        """Fixture for linear knot processor."""
        return KnotParticleProcessor(
            horizon=8,
            action_dim=2,
            n_knots=3,
            degree=1,
            device_cfg=device_cfg,
        )

    def test_init_basic(self, device_cfg):
        """Test basic initialization."""
        processor = KnotParticleProcessor(
            horizon=12,
            action_dim=5,
            n_knots=6,
            degree=2,
            device_cfg=device_cfg,
        )

        assert processor.horizon == 12
        assert processor.action_dim == 5
        assert processor.n_knots == 6
        assert processor.degree == 2
        assert processor.device_cfg == device_cfg

    def test_init_default_tensor_args(self):
        """Test initialization with default tensor arguments."""
        processor = KnotParticleProcessor(
            horizon=8,
            action_dim=3,
            n_knots=4,
        )

        assert processor.device_cfg is not None
        assert isinstance(processor.device_cfg, DeviceCfg)

    def test_process_samples_shape(self, basic_processor):
        """Test that process_samples returns correct output shape."""
        batch_size = 5
        input_shape = (batch_size, basic_processor.n_knots * basic_processor.action_dim)
        samples = torch.randn(input_shape)

        processed = basic_processor.process_samples(samples)

        expected_shape = (batch_size, basic_processor.horizon, basic_processor.action_dim)
        assert processed.shape == expected_shape

    def test_process_samples_device_dtype(self, basic_processor):
        """Test that processed samples maintain correct device and dtype."""
        batch_size = 3
        input_shape = (batch_size, basic_processor.n_knots * basic_processor.action_dim)
        samples = torch.randn(input_shape, dtype=basic_processor.device_cfg.dtype)

        processed = basic_processor.process_samples(samples)

        assert processed.device == basic_processor.device_cfg.device
        assert processed.dtype == basic_processor.device_cfg.dtype

    def test_process_samples_reshaping(self, basic_processor):
        """Test correct reshaping of input samples to knot format."""
        batch_size = 2
        n_knots = basic_processor.n_knots
        action_dim = basic_processor.action_dim

        # Create samples with known pattern
        samples = torch.arange(batch_size * n_knots * action_dim, dtype=torch.float32)
        samples = samples.view(batch_size, n_knots * action_dim)

        processed = basic_processor.process_samples(samples)

        # Should not raise an error and should produce valid output
        assert processed.shape == (batch_size, basic_processor.horizon, action_dim)
        assert torch.all(torch.isfinite(processed))

    def test_process_samples_filter_smooth_ignored(self, basic_processor):
        """Test that filter_smooth parameter is ignored (B-splines are inherently smooth)."""
        batch_size = 2
        input_shape = (batch_size, basic_processor.n_knots * basic_processor.action_dim)
        samples = torch.randn(input_shape)

        processed_false = basic_processor.process_samples(samples, filter_smooth=False)
        processed_true = basic_processor.process_samples(samples, filter_smooth=True)

        # Should be identical regardless of filter_smooth value
        torch.testing.assert_close(processed_false, processed_true)

    def test_bspline_static_method_basic(self):
        """Test the static bspline method with basic input."""
        # Simple control points
        control_points = torch.tensor([0.0, 1.0, 2.0, 1.0])

        result = KnotParticleProcessor.bspline(control_points, n=10, degree=3)

        assert result.shape == (10,)
        assert result.dtype == control_points.dtype
        assert result.device == control_points.device

    def test_bspline_static_method_custom_time(self):
        """Test bspline method with custom time array."""
        control_points = torch.tensor([1.0, 2.0, 3.0])
        time_array = torch.tensor([0.0, 1.0, 2.0])

        result = KnotParticleProcessor.bspline(control_points, t_arr=time_array, n=5, degree=2)

        assert result.shape == (5,)

    def test_bspline_boundary_behavior(self):
        """Test that B-spline interpolation respects boundary conditions."""
        # Linear case: should interpolate between first and last control points
        control_points = torch.tensor([0.0, 5.0, 10.0])

        result = KnotParticleProcessor.bspline(control_points, n=100, degree=1)

        # First and last values should be close to control point range
        assert result[0] >= control_points.min() - 0.1
        assert result[0] <= control_points.max() + 0.1
        assert result[-1] >= control_points.min() - 0.1
        assert result[-1] <= control_points.max() + 0.1

    def test_bspline_smoothness(self):
        """Test that B-spline produces smooth curves."""
        # Create control points with known discontinuity
        control_points = torch.tensor([0.0, 0.0, 10.0, 10.0])

        result = KnotParticleProcessor.bspline(control_points, n=20, degree=3)

        # Result should be smoother than control points (no sudden jumps)
        differences = torch.diff(result)
        max_diff = torch.max(torch.abs(differences))

        # Should be much smoother than direct linear interpolation
        assert max_diff < 5.0  # Much less than the 10.0 jump in control points

    def test_different_degrees(self, device_cfg):
        """Test processor with different B-spline degrees."""
        degrees = [1, 2, 3, 4]

        for degree in degrees:
            processor = KnotParticleProcessor(
                horizon=8,
                action_dim=2,
                n_knots=5,
                degree=degree,
                device_cfg=device_cfg,
            )

            batch_size = 2
            samples = torch.randn(batch_size, 5 * 2)
            processed = processor.process_samples(samples)

            assert processed.shape == (batch_size, 8, 2)
            assert torch.all(torch.isfinite(processed))

    def test_different_knot_counts(self, device_cfg):
        """Test processor with different numbers of knots."""
        knot_counts = [2, 3, 5, 8, 10]

        for n_knots in knot_counts:
            processor = KnotParticleProcessor(
                horizon=12,
                action_dim=3,
                n_knots=n_knots,
                degree=min(3, n_knots - 1),  # Degree must be less than n_knots
                device_cfg=device_cfg,
            )

            batch_size = 2
            samples = torch.randn(batch_size, n_knots * 3)
            processed = processor.process_samples(samples)

            assert processed.shape == (batch_size, 12, 3)

    def test_different_horizons(self, device_cfg):
        """Test processor with different horizon lengths."""
        horizons = [5, 10, 20, 50]

        for horizon in horizons:
            processor = KnotParticleProcessor(
                horizon=horizon,
                action_dim=2,
                n_knots=4,
                degree=3,
                device_cfg=device_cfg,
            )

            batch_size = 2
            samples = torch.randn(batch_size, 4 * 2)
            processed = processor.process_samples(samples)

            assert processed.shape == (batch_size, horizon, 2)

    def test_different_action_dimensions(self, device_cfg):
        """Test processor with different action dimensions."""
        action_dims = [1, 2, 5, 7, 10]

        for action_dim in action_dims:
            processor = KnotParticleProcessor(
                horizon=8,
                action_dim=action_dim,
                n_knots=4,
                degree=3,
                device_cfg=device_cfg,
            )

            batch_size = 2
            samples = torch.randn(batch_size, 4 * action_dim)
            processed = processor.process_samples(samples)

            assert processed.shape == (batch_size, 8, action_dim)

    def test_single_batch_processing(self, basic_processor):
        """Test processing with single batch."""
        samples = torch.randn(1, basic_processor.n_knots * basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        assert processed.shape == (1, basic_processor.horizon, basic_processor.action_dim)

    def test_large_batch_processing(self, basic_processor):
        """Test processing with large batch size."""
        batch_size = 100
        samples = torch.randn(batch_size, basic_processor.n_knots * basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        assert processed.shape == (batch_size, basic_processor.horizon, basic_processor.action_dim)

    def test_zero_batch_processing(self, basic_processor):
        """Test processing with zero batch size."""
        samples = torch.zeros(0, basic_processor.n_knots * basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        assert processed.shape == (0, basic_processor.horizon, basic_processor.action_dim)

    def test_interpolation_consistency(self, linear_processor):
        """Test that interpolation is consistent across dimensions."""
        batch_size = 1
        n_knots = linear_processor.n_knots
        action_dim = linear_processor.action_dim

        # Create samples where each dimension has the same knot pattern
        samples = torch.zeros(batch_size, n_knots * action_dim)
        for dim in range(action_dim):
            for knot in range(n_knots):
                samples[0, dim * n_knots + knot] = knot  # 0, 1, 2 for each dimension

        processed = linear_processor.process_samples(samples)

        # Each dimension should have similar interpolation pattern
        for dim in range(action_dim):
            dim_trajectory = processed[0, :, dim]
            # Should be monotonically increasing (roughly)
            differences = torch.diff(dim_trajectory)
            assert torch.all(differences >= -0.1)  # Allow small numerical errors

    def test_bspline_device_handling(self):
        """Test that bspline method handles device transfers correctly."""
        # Test with CPU tensor
        control_points_cpu = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cpu"))
        result_cpu = KnotParticleProcessor.bspline(control_points_cpu, n=5)
        assert result_cpu.device == torch.device("cpu")

        # Test with CUDA tensor if available
        if torch.cuda.is_available():
            control_points_cuda = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cuda:0"))
            result_cuda = KnotParticleProcessor.bspline(control_points_cuda, n=5)
            assert result_cuda.device == torch.device("cuda:0")

    def test_bspline_dtype_preservation(self):
        """Test that bspline method preserves dtype."""
        for dtype in [torch.float32, torch.float64]:
            control_points = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
            result = KnotParticleProcessor.bspline(control_points, n=5)
            assert result.dtype == dtype

    def test_knot_to_trajectory_mapping(self, basic_processor):
        """Test that knot samples are correctly mapped to trajectories."""
        batch_size = 1

        # Create simple knot pattern: all zeros except one knot
        samples = torch.zeros(batch_size, basic_processor.n_knots * basic_processor.action_dim)

        # Set middle knot of first dimension to 1.0
        mid_knot = basic_processor.n_knots // 2
        samples[0, mid_knot] = 1.0

        processed = basic_processor.process_samples(samples)

        # First dimension should have non-zero values influenced by the knot
        first_dim_traj = processed[0, :, 0]
        assert torch.any(first_dim_traj != 0.0)

        # Other dimensions should remain close to zero
        other_dims_traj = processed[0, :, 1:]
        assert torch.all(torch.abs(other_dims_traj) < 0.1)

    def test_edge_case_single_knot(self, device_cfg):
        """Test processor with single knot (if degree allows)."""
        processor = KnotParticleProcessor(
            horizon=5,
            action_dim=2,
            n_knots=2,
            degree=1,  # Constant interpolation
            device_cfg=device_cfg,
        )

        batch_size = 2
        samples = torch.randn(batch_size, 2 * 2)
        processed = processor.process_samples(samples)

        assert processed.shape == (batch_size, 5, 2)

    def test_numerical_stability(self, basic_processor):
        """Test numerical stability with extreme values."""
        batch_size = 2

        # Test with very large values
        large_samples = torch.full(
            (batch_size, basic_processor.n_knots * basic_processor.action_dim), 1000.0
        )
        large_processed = basic_processor.process_samples(large_samples)
        assert torch.all(torch.isfinite(large_processed))

        # Test with very small values
        small_samples = torch.full(
            (batch_size, basic_processor.n_knots * basic_processor.action_dim), 1e-6
        )
        small_processed = basic_processor.process_samples(small_samples)
        assert torch.all(torch.isfinite(small_processed))

    def test_spline_smoothness_property(self, basic_processor):
        """Test that B-spline interpolation produces smooth trajectories."""
        batch_size = 1

        # Create random knot points
        samples = torch.randn(batch_size, basic_processor.n_knots * basic_processor.action_dim)
        processed = basic_processor.process_samples(samples)

        # Calculate second derivatives (measure of curvature/smoothness)
        for dim in range(basic_processor.action_dim):
            trajectory = processed[0, :, dim]
            if len(trajectory) > 2:
                # Finite difference approximation of second derivative
                second_deriv = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]

                # For cubic B-splines, second derivatives should be relatively smooth
                # (no sudden jumps in curvature)
                second_deriv_changes = torch.diff(second_deriv)
                max_change = torch.max(torch.abs(second_deriv_changes))

                # This is a heuristic test - actual bounds depend on input
                assert max_change < 100.0  # Should not have extreme curvature changes
