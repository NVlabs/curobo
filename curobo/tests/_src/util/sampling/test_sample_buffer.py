# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for SampleBuffer."""

import numpy as np
import pytest
import torch

from curobo._src.types.device_cfg import DeviceCfg

# CuRobo
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.sampling.sequencer_halton import HaltonSequencer
from curobo._src.util.sampling.sequencer_random import RandomSequencer
from curobo._src.util.sampling.sequencer_roberts import RobertsSequencer


@pytest.mark.parametrize("ndims", [1, 3, 7])
class TestSampleBuffer:
    """Test cases for SampleBuffer implementation."""

    @pytest.fixture
    def device_cfg(self):
        """Fixture for tensor arguments."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    @pytest.fixture
    def halton_sequencer(self, ndims):
        """Fixture for Halton sequencer."""
        return HaltonSequencer(ndims=ndims, seed=123)

    @pytest.fixture
    def random_sequencer(self, ndims):
        """Fixture for Random sequencer."""
        return RandomSequencer(ndims=ndims, seed=456)

    @pytest.fixture
    def roberts_sequencer(self, ndims):
        """Fixture for Roberts sequencer."""
        return RobertsSequencer(ndims=ndims, seed=789)

    def test_init_basic(self, halton_sequencer, device_cfg, ndims):
        """Test basic initialization."""
        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            up_bounds=[2.0],
            low_bounds=[-1.0],
            store_buffer=1000,
        )

        assert buffer.sequencer == halton_sequencer
        assert buffer.ndims == ndims
        assert buffer.device_cfg == device_cfg
        assert buffer.fixed_samples is True
        assert buffer._store_buffer == 1000

    def test_init_no_buffer(self, random_sequencer, device_cfg, ndims):
        """Test initialization without buffering."""
        buffer = SampleBuffer(
            sequencer=random_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        assert buffer.fixed_samples is False
        assert buffer._sample_buffer is None
        assert buffer._store_buffer is None

    def test_bounds_setup(self, halton_sequencer, device_cfg, ndims):
        """Test that bounds are properly set up."""
        # Create bounds that work with variable ndims
        up_bounds = [2.0 + i for i in range(ndims)]
        low_bounds = [-1.0 - i for i in range(ndims)]

        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            up_bounds=up_bounds,
            low_bounds=low_bounds,
        )

        expected_range = torch.tensor(
            [up_bounds[i] - low_bounds[i] for i in range(ndims)], dtype=device_cfg.dtype
        )
        expected_low = torch.tensor(low_bounds, dtype=device_cfg.dtype)

        torch.testing.assert_close(buffer.range_b, expected_range)
        torch.testing.assert_close(buffer.low_bounds, expected_low)

    def test_get_samples_unbounded(self, halton_sequencer, device_cfg, ndims):
        """Test getting unbounded samples."""
        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        samples = buffer.get_samples(100, bounded=False)

        assert samples.shape == (100, ndims)
        assert samples.dtype == device_cfg.dtype
        assert samples.device == device_cfg.device
        assert torch.all(samples >= 0.0)
        assert torch.all(samples <= 1.0)

    def test_get_samples_bounded(self, random_sequencer, device_cfg, ndims):
        """Test getting bounded samples."""
        up_bounds = [2.0]
        low_bounds = [-1.0]

        buffer = SampleBuffer(
            sequencer=random_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            up_bounds=up_bounds,
            low_bounds=low_bounds,
            store_buffer=None,
        )

        samples = buffer.get_samples(50, bounded=True)

        assert samples.shape == (50, ndims)
        assert torch.all(samples >= -1.0)
        assert torch.all(samples <= 2.0)

    def test_get_gaussian_samples(self, roberts_sequencer, device_cfg, ndims):
        """Test getting Gaussian-distributed samples."""
        buffer = SampleBuffer(
            sequencer=roberts_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        samples = buffer.get_gaussian_samples(1000)

        assert samples.shape == (1000, ndims)
        assert samples.dtype == device_cfg.dtype
        assert samples.device == device_cfg.device

        # Check basic Gaussian properties (mean ≈ 0, reasonable range)
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)

        # Mean should be close to 0
        assert torch.all(torch.abs(mean) < 0.2)
        # Standard deviation should be reasonable for unit variance
        assert torch.all(std > 0.5) and torch.all(std < 2.0)

    def test_get_gaussian_samples_variance(self, halton_sequencer, device_cfg, ndims):
        """Test Gaussian samples with different variance."""
        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        variance = 4.0
        samples = buffer.get_gaussian_samples(1000, variance=variance)

        assert samples.shape == (1000, ndims)

        # Standard deviation should be approximately sqrt(variance) = 2.0
        std = torch.std(samples, dim=0)
        expected_std = np.sqrt(variance)
        assert torch.all(torch.abs(std - expected_std) < 0.5)

    def test_buffered_sampling(self, halton_sequencer, device_cfg, ndims):
        """Test sampling with pre-generated buffer."""
        buffer_size = 500
        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=buffer_size,
        )

        # Buffer should be pre-generated
        assert buffer._sample_buffer is not None
        assert buffer._sample_buffer.shape == (buffer_size, ndims)

        # Get samples from buffer
        samples = buffer.get_samples(100, bounded=False)
        assert samples.shape == (100, ndims)

    def test_buffered_sampling_consistency(self, random_sequencer, device_cfg, ndims):
        """Test that buffered sampling is consistent across calls."""
        buffer = SampleBuffer(
            sequencer=random_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=1000,
        )

        # Reset to ensure consistent state
        buffer.reset()

        # Get same number of samples twice
        samples1 = buffer.get_samples(50, bounded=False)
        samples2 = buffer.get_samples(50, bounded=False)

        # Samples should be different (random sampling from buffer)
        assert not torch.allclose(samples1, samples2)
        assert samples1.shape == samples2.shape == (50, ndims)

    def test_reset_functionality(self, halton_sequencer, device_cfg, ndims):
        """Test reset functionality."""
        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        # Generate some samples
        initial_samples = buffer.get_samples(10, bounded=False)

        # Reset and generate again
        buffer.reset()
        reset_samples = buffer.get_samples(10, bounded=False)

        # Should be identical for deterministic sequencer
        torch.testing.assert_close(initial_samples, reset_samples)

    def test_fast_forward_functionality(self, roberts_sequencer, device_cfg, ndims):
        """Test fast forward functionality."""
        buffer = SampleBuffer(
            sequencer=roberts_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        # Generate samples normally
        buffer.reset()
        normal_samples = buffer.get_samples(20, bounded=False)

        # Reset, fast forward, then generate
        buffer.reset()
        buffer.fast_forward(10)
        ff_samples = buffer.get_samples(10, bounded=False)

        # Fast forwarded samples should match last 10 normal samples
        torch.testing.assert_close(normal_samples[10:], ff_samples)

    def test_fast_forward_with_buffered_samples_warning(self, halton_sequencer, device_cfg, ndims):
        """Test that fast forward with buffered samples logs warning."""
        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=100,
        )

        # Fast forward should raise an error with buffered samples
        with pytest.raises(ValueError, match="fast forward will not work with fixed samples"):
            buffer.fast_forward(5)

    def test_bound_samples_static_method(self, device_cfg, ndims):
        """Test the static bound_samples method."""
        samples = torch.tensor([[i / ndims for i in range(ndims)]], dtype=device_cfg.dtype)
        range_b = torch.tensor([2.0 * (i + 1) for i in range(ndims)], dtype=device_cfg.dtype)
        low_bounds = torch.tensor([-1.0 * (i + 1) for i in range(ndims)], dtype=device_cfg.dtype)

        bounded = SampleBuffer.bound_samples(samples, range_b, low_bounds)

        expected = torch.tensor(
            [[samples[0, i] * range_b[i] + low_bounds[i] for i in range(ndims)]],
            dtype=device_cfg.dtype,
        )
        torch.testing.assert_close(bounded, expected)

    def test_gaussian_transform_static_method(self, device_cfg, ndims):
        """Test the static gaussian_transform method."""
        # Use samples that will give predictable results
        uniform_samples = torch.tensor([[0.5] * ndims], dtype=device_cfg.dtype)
        proj_mat = torch.sqrt(torch.tensor([2.0], dtype=device_cfg.dtype))
        i_mat = torch.eye(ndims, dtype=device_cfg.dtype)
        std_dev = 1.0

        gaussian = SampleBuffer.gaussian_transform(uniform_samples, proj_mat, i_mat, std_dev)

        assert gaussian.shape == (1, ndims)
        # At 0.5, erfinv(0) = 0, so result should be close to 0
        assert torch.all(torch.abs(gaussian) < 0.1)

    def test_sample_by_random_index_static_method(self, device_cfg, ndims):
        """Test the static sample_by_random_index method."""
        # Create a sample buffer with variable ndims
        sample_data = [[0.1 + j * 0.1 for j in range(ndims)] for i in range(3)]
        sample_buffer = torch.tensor(sample_data, dtype=device_cfg.dtype)
        num_samples = 2
        generator = torch.Generator()
        generator.manual_seed(123)

        samples, indices = SampleBuffer.sample_by_random_index(
            sample_buffer, num_samples, generator, device_cfg.device, None
        )

        assert samples.shape == (2, ndims)
        assert indices.shape == (2,)
        assert torch.all(indices >= 0)
        assert torch.all(indices < 3)

        # Verify that samples correspond to selected indices
        for i in range(num_samples):
            torch.testing.assert_close(samples[i], sample_buffer[indices[i]])

    def test_create_halton_sample_buffer_classmethod(self, device_cfg, ndims):
        """Test the create_halton_sample_buffer class method."""
        buffer = SampleBuffer.create_halton_sample_buffer(
            ndims=ndims,
            up_bounds=[1.0],
            low_bounds=[0.0],
            store_buffer=200,
            seed=111,
            device_cfg=device_cfg,
        )

        assert isinstance(buffer, SampleBuffer)
        assert buffer.ndims == ndims
        assert isinstance(buffer.sequencer, HaltonSequencer)
        assert buffer.sequencer.seed == 111
        assert buffer._store_buffer == 200

    def test_create_random_sample_buffer_classmethod(self, device_cfg, ndims):
        """Test the create_random_sample_buffer class method."""
        buffer = SampleBuffer.create_random_sample_buffer(
            ndims=ndims,
            up_bounds=[2.0],
            low_bounds=[-1.0],
            store_buffer=300,
            seed=222,
            device_cfg=device_cfg,
        )

        assert isinstance(buffer, SampleBuffer)
        assert buffer.ndims == ndims
        assert isinstance(buffer.sequencer, RandomSequencer)
        assert buffer.sequencer.seed == 222

    def test_create_roberts_sample_buffer_classmethod(self, device_cfg, ndims):
        """Test the create_roberts_sample_buffer class method."""
        buffer = SampleBuffer.create_roberts_sample_buffer(
            ndims=ndims,
            up_bounds=[3.0],
            low_bounds=[-2.0],
            store_buffer=400,
            seed=333,
            device_cfg=device_cfg,
        )

        assert isinstance(buffer, SampleBuffer)
        assert buffer.ndims == ndims
        assert isinstance(buffer.sequencer, RobertsSequencer)
        assert buffer.sequencer.seed == 333

    def test_different_sequencers_different_results(self, device_cfg, ndims):
        """Test that different sequencers produce different results."""
        halton_buffer = SampleBuffer(
            HaltonSequencer(ndims=ndims, seed=123),
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        random_buffer = SampleBuffer(
            RandomSequencer(ndims=ndims, seed=123),
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        halton_samples = halton_buffer.get_samples(100, bounded=False)
        random_samples = random_buffer.get_samples(100, bounded=False)

        # Different sequencers should produce different samples
        assert not torch.allclose(halton_samples, random_samples)

    def test_buffer_index_reuse(self, random_sequencer, device_cfg, ndims):
        """Test that index buffer is reused for efficiency."""
        buffer = SampleBuffer(
            sequencer=random_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=100,
        )

        # Generate samples twice with same size
        buffer.get_samples(10, bounded=False)
        buffer.get_samples(10, bounded=False)

        # Index buffer should be created and reused
        assert buffer._index_buffer is not None
        assert buffer._index_buffer.shape == (10,)

    def test_zero_samples(self, halton_sequencer, device_cfg, ndims):
        """Test requesting zero samples."""
        buffer = SampleBuffer(
            sequencer=halton_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
        )

        samples = buffer.get_samples(0, bounded=False)
        assert samples.shape == (0, ndims)

        gaussian_samples = buffer.get_gaussian_samples(0)
        assert gaussian_samples.shape == (0, ndims)

    def test_large_sample_requests(self, roberts_sequencer, device_cfg, ndims):
        """Test handling of large sample requests."""
        buffer = SampleBuffer(
            sequencer=roberts_sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            store_buffer=None,
        )

        large_samples = buffer.get_samples(5000, bounded=False)
        assert large_samples.shape == (5000, ndims)
        assert torch.all(large_samples >= 0.0)
        assert torch.all(large_samples <= 1.0)

    def test_cuda_device_consistency(self, ndims):
        """Test that all tensors are on the correct device."""
        device = torch.device("cuda:0")

        device_cfg = DeviceCfg(device=device, dtype=torch.float32)
        sequencer = HaltonSequencer(ndims=ndims, seed=123)

        buffer = SampleBuffer(
            sequencer=sequencer,
            ndims=ndims,
            device_cfg=device_cfg,
            up_bounds=[1.0],
            low_bounds=[0.0],
        )

        samples = buffer.get_samples(10, bounded=True)
        assert samples.device == device

        gaussian_samples = buffer.get_gaussian_samples(10)
        assert gaussian_samples.device == device

    def test_dtype_consistency(self, halton_sequencer, ndims):
        """Test that samples maintain consistent dtype."""
        for dtype in [torch.float32, torch.float64]:
            device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=dtype)

            buffer = SampleBuffer(
                sequencer=halton_sequencer,
                ndims=ndims,
                device_cfg=device_cfg,
            )

            samples = buffer.get_samples(5, bounded=False)
            assert samples.dtype == dtype

            gaussian_samples = buffer.get_gaussian_samples(5)
            assert gaussian_samples.dtype == dtype
