# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for StandardParticleProcessor."""

import pytest
import torch

# CuRobo
from curobo._src.optim.particle.sample_strategies.processor_standard import (
    StandardParticleProcessor,
)
from curobo._src.types.device_cfg import DeviceCfg


class TestStandardParticleProcessor:
    """Test cases for StandardParticleProcessor implementation."""

    @pytest.fixture
    def device_cfg(self):
        """Fixture for tensor arguments."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def basic_processor(self, device_cfg):
        """Fixture for basic processor."""
        return StandardParticleProcessor(
            horizon=10,
            action_dim=3,
            device_cfg=device_cfg,
        )

    @pytest.fixture
    def filtered_processor(self, device_cfg):
        """Fixture for processor with filter coefficients."""
        return StandardParticleProcessor(
            horizon=15,
            action_dim=5,
            device_cfg=device_cfg,
            filter_coeffs=(0.3, 0.3, 0.4),
        )

    def test_init_basic(self, device_cfg):
        """Test basic initialization."""
        processor = StandardParticleProcessor(
            horizon=8,
            action_dim=4,
            device_cfg=device_cfg,
        )

        assert processor.horizon == 8
        assert processor.action_dim == 4
        assert processor.device_cfg == device_cfg
        assert processor.filter_coeffs is None
        assert processor.stomp_matrix is None
        assert processor.stomp_scale_tril is None

    def test_init_with_filter_coeffs(self, device_cfg):
        """Test initialization with filter coefficients."""
        filter_coeffs = (0.2, 0.4, 0.4)
        processor = StandardParticleProcessor(
            horizon=12,
            action_dim=6,
            device_cfg=device_cfg,
            filter_coeffs=filter_coeffs,
        )

        assert processor.filter_coeffs == filter_coeffs

    def test_process_samples_no_filtering(self, basic_processor):
        """Test processing samples without any filtering."""
        batch_size = 5
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples, filter_smooth=False)

        # Without filtering, output should be identical to input
        torch.testing.assert_close(processed, samples)
        assert processed.shape == samples.shape

    def test_process_samples_standard_filtering(self, filtered_processor):
        """Test processing samples with standard filter coefficients."""
        batch_size = 3
        samples = torch.randn(batch_size, filtered_processor.horizon, filtered_processor.action_dim)
        original_samples = samples.clone()

        processed = filtered_processor.process_samples(samples, filter_smooth=False)

        assert processed.shape == samples.shape

        # First two timesteps should be unchanged
        torch.testing.assert_close(processed[:, :2, :], original_samples[:, :2, :])

        # Later timesteps should be modified by filtering
        # (Exact values depend on filter implementation)
        assert not torch.allclose(processed[:, 2:, :], original_samples[:, 2:, :])

    def test_process_samples_smooth_filtering(self, basic_processor):
        """Test processing samples with smooth STOMP filtering."""
        batch_size = 4
        samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)

        processed = basic_processor.process_samples(samples, filter_smooth=True)

        assert processed.shape == samples.shape
        # Should initialize STOMP matrices
        assert basic_processor.stomp_matrix is not None
        assert basic_processor.stomp_scale_tril is not None

    def test_filter_samples_no_coeffs(self, basic_processor):
        """Test _filter_samples method with no filter coefficients."""
        samples = torch.randn(2, basic_processor.horizon, basic_processor.action_dim)
        original_samples = samples.clone()

        filtered = basic_processor._filter_samples(samples)

        # Should be unchanged
        torch.testing.assert_close(filtered, original_samples)

    def test_filter_samples_with_coeffs(self, filtered_processor):
        """Test _filter_samples method with filter coefficients."""
        batch_size = 3
        horizon = filtered_processor.horizon
        action_dim = filtered_processor.action_dim

        # Create samples with known pattern for predictable filtering
        samples = torch.ones(batch_size, horizon, action_dim)
        samples[:, 0, :] = 1.0
        samples[:, 1, :] = 2.0

        filtered = filtered_processor._filter_samples(samples)

        # First two timesteps should be unchanged
        assert torch.allclose(filtered[:, 0, :], torch.ones(batch_size, action_dim))
        assert torch.allclose(filtered[:, 1, :], torch.full((batch_size, action_dim), 2.0))

        # Third timestep should be filtered combination
        beta_0, beta_1, beta_2 = filtered_processor.filter_coeffs
        expected_third = beta_0 * 1.0 + beta_1 * 2.0 + beta_2 * 1.0
        assert torch.allclose(filtered[:, 2, :], torch.full((batch_size, action_dim), expected_third))

    def test_filter_smooth_empty_samples(self, basic_processor):
        """Test _filter_smooth method with empty samples."""
        empty_samples = torch.zeros(0, basic_processor.horizon, basic_processor.action_dim)

        result = basic_processor._filter_smooth(empty_samples)

        assert result.shape == empty_samples.shape
        torch.testing.assert_close(result, empty_samples)

    def test_filter_smooth_initialization(self, basic_processor):
        """Test that _filter_smooth properly initializes STOMP matrices."""
        samples = torch.randn(2, basic_processor.horizon, basic_processor.action_dim)

        # Initially should be None
        assert basic_processor.stomp_matrix is None
        assert basic_processor.stomp_scale_tril is None

        basic_processor._filter_smooth(samples)

        # Should be initialized after first call
        assert basic_processor.stomp_matrix is not None
        assert basic_processor.stomp_scale_tril is not None

        # Check shapes
        expected_shape = (basic_processor.horizon, basic_processor.horizon)
        assert basic_processor.stomp_matrix.shape == expected_shape

    def test_filter_smooth_lazy_initialization(self, basic_processor):
        """Test that STOMP matrices are only initialized once."""
        samples1 = torch.randn(2, basic_processor.horizon, basic_processor.action_dim)
        samples2 = torch.randn(3, basic_processor.horizon, basic_processor.action_dim)

        # First call should initialize
        basic_processor._filter_smooth(samples1)
        matrix_first = basic_processor.stomp_matrix
        scale_tril_first = basic_processor.stomp_scale_tril

        # Second call should reuse
        basic_processor._filter_smooth(samples2)
        matrix_second = basic_processor.stomp_matrix
        scale_tril_second = basic_processor.stomp_scale_tril

        # Should be the same objects (not just equal values)
        assert matrix_first is matrix_second
        assert scale_tril_first is scale_tril_second

    def test_filter_smooth_normalization(self, basic_processor):
        """Test that smooth filtering normalizes by maximum absolute value."""
        # Create samples with known large values
        samples = torch.zeros(1, basic_processor.horizon, basic_processor.action_dim)
        samples[0, 5, 1] = 100.0  # Large value at specific position

        filtered = basic_processor._filter_smooth(samples)

        # Result should be normalized
        max_abs = torch.max(torch.abs(filtered))
        assert torch.allclose(max_abs, torch.tensor(1.0), atol=1e-6)

    def test_different_horizon_sizes(self, device_cfg):
        """Test processor with different horizon sizes."""
        horizons = [1, 5, 10, 20, 50]

        for horizon in horizons:
            processor = StandardParticleProcessor(
                horizon=horizon,
                action_dim=3,
                device_cfg=device_cfg,
            )

            samples = torch.randn(2, horizon, 3)
            processed = processor.process_samples(samples, filter_smooth=False)

            assert processed.shape == (2, horizon, 3)

    def test_different_action_dimensions(self, device_cfg):
        """Test processor with different action dimensions."""
        action_dims = [1, 2, 5, 7, 10]

        for action_dim in action_dims:
            processor = StandardParticleProcessor(
                horizon=8,
                action_dim=action_dim,
                device_cfg=device_cfg,
            )

            samples = torch.randn(3, 8, action_dim)
            processed = processor.process_samples(samples, filter_smooth=False)

            assert processed.shape == (3, 8, action_dim)

    def test_batch_size_variations(self, basic_processor):
        """Test processor with different batch sizes."""
        batch_sizes = [1, 2, 5, 10, 20]

        for batch_size in batch_sizes:
            samples = torch.randn(batch_size, basic_processor.horizon, basic_processor.action_dim)
            processed = basic_processor.process_samples(samples, filter_smooth=False)

            assert processed.shape == (
                batch_size,
                basic_processor.horizon,
                basic_processor.action_dim,
            )

    def test_filter_coeffs_edge_cases(self, device_cfg):
        """Test filter coefficients with edge case values."""
        # All zeros (should zero out later timesteps)
        processor_zeros = StandardParticleProcessor(
            horizon=5,
            action_dim=2,
            device_cfg=device_cfg,
            filter_coeffs=(0.0, 0.0, 0.0),
        )

        samples = torch.ones(1, 5, 2)
        filtered = processor_zeros._filter_samples(samples)

        # Timesteps 2+ should be zero
        assert torch.allclose(filtered[:, 2:, :], torch.zeros(1, 3, 2))

        # Identity filter (should preserve current timestep only)
        processor_identity = StandardParticleProcessor(
            horizon=5,
            action_dim=2,
            device_cfg=device_cfg,
            filter_coeffs=(1.0, 0.0, 0.0),
        )

        samples2 = torch.randn(1, 5, 2)
        filtered2 = processor_identity._filter_samples(samples2)

        # First two timesteps unchanged, later ones should equal current timestep
        torch.testing.assert_close(filtered2[:, :2, :], samples2[:, :2, :])

    def test_cuda_device_cfg_consistency(self):
        """Test that processor works correctly with different devices."""
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))

        for device in devices:
            device_cfg = DeviceCfg(device=device, dtype=torch.float32)
            processor = StandardParticleProcessor(
                horizon=6,
                action_dim=4,
                device_cfg=device_cfg,
            )

            samples = torch.randn(2, 6, 4, device=device)
            processed = processor.process_samples(samples, filter_smooth=True)

            assert processed.device == device
            if processor.stomp_matrix is not None:
                assert processor.stomp_matrix.device == device

    def test_dtype_consistency(self, device_cfg):
        """Test that processor maintains dtype consistency."""
        for dtype in [torch.float32, torch.float64]:
            tensor_args_typed = DeviceCfg(device=device_cfg.device, dtype=dtype)
            processor = StandardParticleProcessor(
                horizon=4,
                action_dim=2,
                device_cfg=tensor_args_typed,
            )

            samples = torch.randn(1, 4, 2, dtype=dtype)
            processed = processor.process_samples(samples, filter_smooth=False)

            assert processed.dtype == dtype

    def test_in_place_modification_behavior(self, filtered_processor):
        """Test whether processing modifies input in-place or creates new tensor."""
        samples = torch.randn(2, filtered_processor.horizon, filtered_processor.action_dim)
        original_samples = samples.clone()

        # Test standard filtering
        processed = filtered_processor._filter_samples(samples)

        # The input tensor should be modified in-place for the filtering operation
        # (This is the current behavior based on the implementation)
        # First two timesteps should still match original
        torch.testing.assert_close(samples[:, :2, :], original_samples[:, :2, :])

    def test_filter_temporal_smoothing(self, device_cfg):
        """Test that temporal filtering provides smoothing effect."""
        # Create processor with smoothing coefficients
        processor = StandardParticleProcessor(
            horizon=10,
            action_dim=1,
            device_cfg=device_cfg,
            filter_coeffs=(0.5, 0.3, 0.2),
        )

        # Create noisy step function
        samples = torch.zeros(1, 10, 1)
        samples[0, :5, 0] = 1.0
        samples[0, 5:, 0] = -1.0

        filtered = processor._filter_samples(samples)

        # Filtered signal should be smoother (less abrupt transitions)
        # Check that transition region is smoothed
        transition_region = filtered[0, 4:8, 0]
        # Should have intermediate values, not just -1 and 1
        assert torch.any((transition_region > -0.9) & (transition_region < 0.9))
