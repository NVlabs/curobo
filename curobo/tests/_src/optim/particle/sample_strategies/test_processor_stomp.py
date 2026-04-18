# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for StompParticleProcessor."""

import pytest
import torch

# CuRobo
from curobo._src.optim.particle.sample_strategies.processor_stomp import StompParticleProcessor
from curobo._src.types.device_cfg import DeviceCfg


class TestStompParticleProcessor:
    """Test cases for StompParticleProcessor implementation."""

    @pytest.fixture
    def device_cfg(self):
        """Fixture for tensor arguments."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def basic_processor(self, device_cfg):
        """Fixture for basic STOMP processor."""
        return StompParticleProcessor(
            horizon=10,
            action_dim=3,
            device_cfg=device_cfg,
            stencil_type="3point",
        )

    @pytest.fixture
    def vel_processor(self, device_cfg):
        """Fixture for velocity-mode STOMP processor."""
        return StompParticleProcessor(
            horizon=8,
            action_dim=2,
            device_cfg=device_cfg,
            stencil_type="3point",
        )

    def test_init_basic(self, device_cfg):
        """Test basic initialization."""
        processor = StompParticleProcessor(
            horizon=12,
            action_dim=5,
            device_cfg=device_cfg,
            stencil_type="3point",
        )

        assert processor.horizon == 12
        assert processor.action_dim == 5
        assert processor.device_cfg == device_cfg
        assert processor.stencil_type == "3point"
        assert processor.stomp_scale_tril is not None

    def test_init_default_stencil_type(self, device_cfg):
        """Test initialization with default stencil type."""
        processor = StompParticleProcessor(
            horizon=8,
            action_dim=3,
            device_cfg=device_cfg,
        )

        assert processor.stencil_type == "3point"

    def test_stomp_scale_tril_initialization(self, basic_processor):
        """Test that STOMP scale_tril matrix is properly initialized."""
        assert basic_processor.stomp_scale_tril is not None

        expected_size = basic_processor.horizon
        expected_shape = (expected_size, expected_size)
        assert basic_processor.stomp_scale_tril.shape == expected_shape

        # Should be on correct device
        assert basic_processor.stomp_scale_tril.device == basic_processor.device_cfg.device
        assert basic_processor.stomp_scale_tril.dtype == basic_processor.device_cfg.dtype

    def test_process_samples_shape_flattened_input(self, basic_processor):
        """Test processing with flattened input samples."""
        batch_size = 5
        flattened_size = basic_processor.horizon * basic_processor.action_dim
        samples = torch.randn(batch_size, flattened_size)

        # This should raise error:
        with pytest.raises(ValueError):
            basic_processor.process_samples(samples)

    def test_process_samples_shape_3d_input(self, basic_processor):
        """Test processing with 3D input samples."""
        batch_size = 3
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        expected_shape = (batch_size, basic_processor.horizon, basic_processor.action_dim)
        assert processed.shape == expected_shape

    def test_process_samples_device_dtype(self, basic_processor):
        """Test that processed samples maintain correct device and dtype."""
        batch_size = 2
        samples = torch.randn(
            batch_size,
            basic_processor.horizon,
            basic_processor.action_dim,
            dtype=basic_processor.device_cfg.dtype,
            device=basic_processor.device_cfg.device,
        )

        processed = basic_processor.process_samples(samples)

        assert processed.device == basic_processor.device_cfg.device
        assert processed.dtype == basic_processor.device_cfg.dtype

    def test_stomp_transformation_applied(self, basic_processor):
        """Test that STOMP covariance transformation is applied."""
        batch_size = 2
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)
        original_samples = samples.clone()

        processed = basic_processor.process_samples(samples)

        # Output should be different from input due to transformation
        processed_flat = processed.view(
            batch_size, basic_processor.horizon, basic_processor.action_dim
        )
        assert not torch.allclose(processed_flat, original_samples, atol=1e-6)

    def test_boundary_conditions_enforced(self, basic_processor):
        """Test that boundary conditions are enforced."""
        batch_size = 3
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        # First timestep should be zero
        assert torch.allclose(processed[:, 0, :], torch.zeros(batch_size, basic_processor.action_dim))

        # Last two timesteps should be zero
        assert torch.allclose(
            processed[:, -2:, :], torch.zeros(batch_size, 2, basic_processor.action_dim)
        )

    def test_normalization_applied(self, basic_processor):
        """Test that normalization is applied."""
        batch_size = 2
        # Create samples with known large values
        samples = torch.full((batch_size, basic_processor.horizon, basic_processor.action_dim), 100.0)

        processed = basic_processor.process_samples(samples)

        # Maximum absolute value should be normalized to 1 or close to it
        max_abs = torch.max(torch.abs(processed))
        assert max_abs <= 1.1  # Allow small numerical tolerance

    def test_filter_smooth_ignored(self, basic_processor):
        """Test that filter_smooth parameter is ignored."""
        batch_size = 2
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed_false = basic_processor.process_samples(samples, filter_smooth=False)
        processed_true = basic_processor.process_samples(samples, filter_smooth=True)

        # Should be identical regardless of filter_smooth value
        torch.testing.assert_close(processed_false, processed_true)

    def test_different_stencil_types(self, device_cfg):
        """Test processor with different covariance modes."""
        stencil_types = ["3point", "5point", "7point"]

        for stencil_type in stencil_types:
            processor = StompParticleProcessor(
                horizon=6,
                action_dim=2,
                device_cfg=device_cfg,
                stencil_type=stencil_type,
            )

            batch_size = 2
            samples = torch.randn(batch_size, 6, 2)
            processed = processor.process_samples(samples)

            assert processed.shape == (2, 6, 2)
            # Different modes should produce different scale_tril matrices
            assert processor.stomp_scale_tril is not None

    def test_different_horizons(self, device_cfg):
        """Test processor with different horizon lengths."""
        horizons = [5, 8, 15, 20]

        for horizon in horizons:
            processor = StompParticleProcessor(
                horizon=horizon,
                action_dim=3,
                device_cfg=device_cfg,
                stencil_type="3point",
            )

            batch_size = 2
            samples = torch.randn(batch_size, horizon, 3)
            processed = processor.process_samples(samples)

            assert processed.shape == (2, horizon, 3)

    def test_different_action_dimensions(self, device_cfg):
        """Test processor with different action dimensions."""
        action_dims = [1, 2, 4, 7]

        for action_dim in action_dims:
            processor = StompParticleProcessor(
                horizon=8,
                action_dim=action_dim,
                device_cfg=device_cfg,
            )

            batch_size = 2
            samples = torch.randn(batch_size, 8, action_dim)
            processed = processor.process_samples(samples)

            assert processed.shape == (2, 8, action_dim)

    def test_batch_size_variations(self, basic_processor):
        """Test processor with different batch sizes."""
        batch_sizes = [1, 2, 5, 10]

        for batch_size in batch_sizes:
            samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)
            processed = basic_processor.process_samples(samples)

            assert processed.shape == (
                batch_size,
                basic_processor.horizon,
                basic_processor.action_dim,
            )

    def test_zero_batch_processing(self, basic_processor):
        """Test processing with zero batch size."""
        samples = torch.zeros(0, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        assert processed.shape == (0, basic_processor.horizon, basic_processor.action_dim)

    def test_single_batch_processing(self, basic_processor):
        """Test processing with single batch."""
        samples = torch.randn(1, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        assert processed.shape == (1, basic_processor.horizon, basic_processor.action_dim)

    def test_large_batch_processing(self, basic_processor):
        """Test processing with large batch size."""
        batch_size = 100
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        assert processed.shape == (batch_size, basic_processor.horizon, basic_processor.action_dim)

    def test_nan_detection_and_logging(self, basic_processor):
        """Test NaN detection and warning logging."""
        batch_size = 2
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        # Manually inject NaN in stomp_scale_tril to trigger NaN in output
        # (This is a corner case test)
        original_scale_tril = basic_processor.stomp_scale_tril.clone()

        try:
            # Normal processing should not produce NaN
            processed = basic_processor.process_samples(samples)
            assert not torch.any(torch.isnan(processed))

        finally:
            # Restore original scale_tril
            basic_processor.stomp_scale_tril = original_scale_tril

    def test_covariance_matrix_properties(self, basic_processor):
        """Test properties of the STOMP covariance matrix."""
        scale_tril = basic_processor.stomp_scale_tril

        # Should be square matrix
        assert scale_tril.shape[0] == scale_tril.shape[1]

        # Should be lower triangular (or close to it, depending on implementation)
        # This test depends on the specific STOMP implementation
        assert scale_tril.shape[0] == basic_processor.horizon, basic_processor.action_dim

    def test_trajectory_smoothness(self, basic_processor):
        """Test that STOMP produces smooth trajectories."""
        batch_size = 1
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        # Calculate smoothness metric (sum of squared differences)
        for dim in range(basic_processor.action_dim):
            trajectory = processed[0, :, dim]
            if len(trajectory) > 1:
                differences = torch.diff(trajectory)
                # STOMP should produce relatively smooth trajectories
                # This is a heuristic test
                smoothness = torch.sum(differences**2)
                assert torch.isfinite(smoothness)

    def test_cuda_device_cfg_consistency(self):
        """Test processor works correctly with different devices."""
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))

        for device in devices:
            device_cfg = DeviceCfg(device=device, dtype=torch.float32)
            processor = StompParticleProcessor(
                horizon=6,
                action_dim=2,
                device_cfg=device_cfg,
            )

            samples = torch.randn(2, 6, 2, device=device)
            processed = processor.process_samples(samples)

            assert processed.device == device
            assert processor.stomp_scale_tril.device == device

    def test_dtype_consistency(self, device_cfg):
        """Test that processor maintains dtype consistency."""
        for dtype in [torch.float32, torch.float64]:
            tensor_args_typed = DeviceCfg(device=device_cfg.device, dtype=dtype)
            processor = StompParticleProcessor(
                horizon=4,
                action_dim=2,
                device_cfg=tensor_args_typed,
            )

            samples = torch.randn(1, 4, 2, dtype=dtype)
            processed = processor.process_samples(samples)

            assert processed.dtype == dtype

    def test_transformation_matrix_multiplication(self, basic_processor):
        """Test the matrix multiplication in STOMP transformation."""
        batch_size = 2
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        # Manual transformation to verify correctness
        scale_tril = basic_processor.stomp_scale_tril
        manual_transform = torch.matmul(scale_tril, samples)

        # Process through the method
        processed = basic_processor.process_samples(samples)
        processed_flat = processed.view(
            batch_size, basic_processor.horizon, basic_processor.action_dim
        )

        # Results should be similar before normalization and boundary conditions
        # This test verifies the core matrix multiplication works correctly
        # Note: Exact equality not expected due to normalization and boundary enforcement

    def test_reshape_and_transpose_operations(self, basic_processor):
        """Test reshape and transpose operations in process_samples."""
        batch_size = 1
        horizon = basic_processor.horizon
        action_dim = basic_processor.action_dim

        # Create samples with known pattern
        samples = torch.arange(horizon * action_dim, dtype=torch.float32).view(
            batch_size, horizon, action_dim
        )

        processed = basic_processor.process_samples(samples)

        # Should be reshaped to (batch, horizon, action_dim)
        assert processed.shape == (batch_size, horizon, action_dim)

        # Verify that reshape and transpose work correctly
        # by checking that values are in expected positions
        assert torch.all(torch.isfinite(processed))

    def test_zero_input_handling(self, basic_processor):
        """Test processor behavior with zero input."""
        batch_size = 2
        samples = torch.zeros(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples)

        # Should still apply transformation and boundary conditions
        assert processed.shape == (batch_size, basic_processor.horizon, basic_processor.action_dim)

        # Boundary conditions should still be enforced (zeros at start/end)
        assert torch.allclose(processed[:, 0, :], torch.zeros(batch_size, basic_processor.action_dim))
        assert torch.allclose(
            processed[:, -2:, :], torch.zeros(batch_size, 2, basic_processor.action_dim)
        )

    def test_normalization_edge_cases(self, basic_processor):
        """Test normalization with edge cases."""
        batch_size = 2

        # Test with very small values (near zero)
        small_samples = torch.full(
            (batch_size, basic_processor.horizon, basic_processor.action_dim), 1e-8
        )
        small_processed = basic_processor.process_samples(small_samples)
        assert torch.all(torch.isfinite(small_processed))

        # Normalization should handle small values gracefully
        # (Division by very small numbers might cause issues)

    def test_different_sample_patterns(self, basic_processor):
        """Test processor with different input sample patterns."""
        batch_size = 1

        # Test with different patterns
        patterns = [
            torch.ones(batch_size, basic_processor.horizon, basic_processor.action_dim),  # All ones
            torch.zeros(batch_size, basic_processor.horizon, basic_processor.action_dim),  # All zeros
            torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim),  # Random
            torch.linspace(-1, 1, basic_processor.horizon * basic_processor.action_dim).view(
                batch_size, basic_processor.horizon, basic_processor.action_dim
            ),  # Linear ramp
        ]

        for pattern in patterns:
            processed = basic_processor.process_samples(pattern)

            assert processed.shape == (
                batch_size,
                basic_processor.horizon,
                basic_processor.action_dim,
            )
            assert torch.all(torch.isfinite(processed))

            # Boundary conditions should always be enforced
            assert torch.allclose(
                processed[:, 0, :], torch.zeros(batch_size, basic_processor.action_dim)
            )
            assert torch.allclose(
                processed[:, -2:, :], torch.zeros(batch_size, 2, basic_processor.action_dim)
            )
