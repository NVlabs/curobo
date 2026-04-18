# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for depth filtering class.

Tests the FilterDepth class in curobo._src.perception.filter_depth:
- Shape contract: FilterDepth requires batched (B, H, W) inputs
- Range filtering: Filter depths outside valid range
- Flying pixel detection: Remove artifacts at depth discontinuities
- Bilateral filtering: Smooth depth while preserving edges
"""

import pytest
import torch

from curobo._src.perception.filter_depth import (
    FilterDepth,
    FilterDepthConfig,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Get test device (CUDA if available, else CPU)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for FilterDepth tests")
    return torch.device("cuda")


@pytest.fixture
def simple_depth_image(device):
    """Create a simple flat depth image (batched, B=1)."""
    return torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)


@pytest.fixture
def depth_with_discontinuity(device):
    """Create depth image with a sharp discontinuity (flying pixel source)."""
    depth = torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)
    # Add a close object (creates depth discontinuity at edges)
    depth[0, 200:280, 280:360] = 0.5
    return depth


@pytest.fixture
def depth_with_invalid_values(device):
    """Create depth image with various invalid values."""
    depth = torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)
    depth[0, 0:50, :] = 0.0  # Zero (invalid)
    depth[0, 50:100, :] = 0.05  # Too close
    depth[0, 400:450, :] = 15.0  # Too far
    depth[0, 450:480, :] = -1.0  # Negative (invalid)
    return depth


@pytest.fixture
def filter_default(device):
    """Create FilterDepth with default settings."""
    return FilterDepth(
        image_shape=(480, 640),
        depth_minimum_distance=0.1,
        depth_maximum_distance=10.0,
        flying_pixel_threshold=0.5,
        bilateral_kernel_size=None,  # Disable bilateral for faster tests
        device=str(device),
    )


@pytest.fixture
def filter_no_flying(device):
    """Create FilterDepth without flying pixel filter."""
    return FilterDepth(
        image_shape=(480, 640),
        depth_minimum_distance=0.1,
        depth_maximum_distance=10.0,
        flying_pixel_threshold=None,
        bilateral_kernel_size=None,
        device=str(device),
    )


# ============================================================================
# Test FilterDepth Class
# ============================================================================


class TestFilterDepthInit:
    """Test FilterDepth initialization."""

    def test_init_default(self, device):
        depth_filter = FilterDepth(
            image_shape=(480, 640),
            device=str(device),
        )
        assert depth_filter.image_shape == (480, 640)
        assert depth_filter.num_batch == 1
        assert depth_filter.config.depth_minimum_distance == 0.1
        assert depth_filter.config.depth_maximum_distance == 10.0

    def test_init_custom_config(self, device):
        depth_filter = FilterDepth(
            image_shape=(240, 320),
            depth_minimum_distance=0.2,
            depth_maximum_distance=5.0,
            flying_pixel_threshold=0.7,
            bilateral_kernel_size=7,
            device=str(device),
        )
        assert depth_filter.config.depth_minimum_distance == 0.2
        assert depth_filter.config.depth_maximum_distance == 5.0
        assert depth_filter.config.flying_pixel_threshold == 0.7
        assert depth_filter.config.bilateral_kernel_size == 7

    def test_init_batched(self, device):
        depth_filter = FilterDepth(
            image_shape=(480, 640),
            num_batch=4,
            device=str(device),
        )
        assert depth_filter.num_batch == 4
        assert depth_filter._depth_out.shape == (4, 480, 640)
        assert depth_filter._valid_mask_out.shape == (4, 480, 640)

    def test_init_even_kernel_raises(self, device):
        with pytest.raises(ValueError):
            FilterDepth(
                image_shape=(480, 640),
                bilateral_kernel_size=4,
                device=str(device),
            )

    def test_from_config(self, device):
        config = FilterDepthConfig(
            depth_minimum_distance=0.3,
            depth_maximum_distance=8.0,
            flying_pixel_threshold=0.6,
        )
        depth_filter = FilterDepth.from_config(
            config, image_shape=(480, 640), device=str(device)
        )
        assert depth_filter.config.depth_minimum_distance == 0.3
        assert depth_filter.config.depth_maximum_distance == 8.0


# ============================================================================
# Test Shape Contract
# ============================================================================


class TestShapeContract:
    """FilterDepth requires batched (B, H, W) inputs."""

    def test_rejects_2d_input(self, filter_default, device):
        depth_2d = torch.full((480, 640), 2.0, dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match=r"\(B, H, W\)"):
            filter_default(depth_2d)

    def test_rejects_4d_input(self, filter_default, device):
        depth_4d = torch.full((1, 1, 480, 640), 2.0, dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match=r"\(B, H, W\)"):
            filter_default(depth_4d)

    def test_accepts_batched_input(self, filter_default, device):
        depth = torch.full((3, 480, 640), 2.0, dtype=torch.float32, device=device)
        filtered, mask = filter_default(depth)
        assert filtered.shape == (3, 480, 640)
        assert mask.shape == (3, 480, 640)

    def test_rejects_mismatched_output_buffer(self, filter_default, device):
        depth = torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)
        bad_out = torch.zeros((480, 640), dtype=torch.float32, device=device)
        with pytest.raises(ValueError, match=r"depth_out"):
            filter_default(depth, depth_out=bad_out)


# ============================================================================
# Test Basic Filtering
# ============================================================================


class TestFilterDepthBasic:
    """Test basic depth filtering functionality."""

    def test_returns_tuple(self, filter_default, simple_depth_image):
        result = filter_default(simple_depth_image)

        assert isinstance(result, tuple)
        assert len(result) == 2

        filtered_depth, mask = result
        assert filtered_depth.shape == simple_depth_image.shape
        assert mask.shape == simple_depth_image.shape
        assert mask.dtype == torch.bool

    def test_flat_depth_valid(self, filter_default, simple_depth_image):
        filtered_depth, mask = filter_default(simple_depth_image)

        valid_ratio = mask.sum().item() / mask.numel()
        assert valid_ratio > 0.95, f"Expected >95% valid, got {valid_ratio:.1%}"

    def test_output_device(self, filter_default, simple_depth_image):
        filtered_depth, mask = filter_default(simple_depth_image)

        assert filtered_depth.device == simple_depth_image.device
        assert mask.device == simple_depth_image.device


# ============================================================================
# Test Range Filtering
# ============================================================================


class TestRangeFiltering:
    """Test depth range filtering functionality."""

    def test_filter_too_close(self, filter_no_flying, device):
        depth = torch.full((1, 480, 640), 0.05, dtype=torch.float32, device=device)
        _, mask = filter_no_flying(depth)

        assert not mask.any(), "Depths below min should be filtered"

    def test_filter_too_far(self, filter_no_flying, device):
        depth = torch.full((1, 480, 640), 15.0, dtype=torch.float32, device=device)
        _, mask = filter_no_flying(depth)

        assert not mask.any(), "Depths above max should be filtered"

    def test_valid_range_passes(self, filter_no_flying, device):
        depth = torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)
        _, mask = filter_no_flying(depth)

        assert mask.all(), "Valid depths should pass"

    def test_mixed_valid_invalid(self, filter_no_flying, depth_with_invalid_values):
        _, mask = filter_no_flying(depth_with_invalid_values)

        assert not mask[0, 25, 320].item(), "Zero depth should be filtered"
        assert not mask[0, 75, 320].item(), "Too close depth should be filtered"
        assert not mask[0, 425, 320].item(), "Too far depth should be filtered"
        assert not mask[0, 465, 320].item(), "Negative depth should be filtered"
        assert mask[0, 250, 320].item(), "Valid depth (2.0m) should pass"


# ============================================================================
# Test Flying Pixel Filter
# ============================================================================


class TestFlyingPixelFilter:
    """Test flying pixel filter functionality."""

    def test_flat_depth_minimal_filtering(self, filter_default, simple_depth_image):
        _, mask = filter_default(simple_depth_image)

        valid_ratio = mask.sum().item() / mask.numel()
        assert valid_ratio > 0.95

    def test_discontinuity_filtered(self, filter_default, depth_with_discontinuity):
        _, mask = filter_default(depth_with_discontinuity)

        num_filtered = (~mask).sum().item()
        assert num_filtered > 0, "Should filter pixels at depth discontinuities"

    def test_disable_flying_filter(self, filter_no_flying, depth_with_discontinuity):
        _, mask = filter_no_flying(depth_with_discontinuity)

        assert mask.all(), "All valid-range depths should pass when flying filter disabled"


# ============================================================================
# Test Config Updates
# ============================================================================


class TestConfigUpdate:
    """Test dynamic configuration updates."""

    def test_update_depth_range(self, filter_default, device):
        depth = torch.full((1, 480, 640), 8.0, dtype=torch.float32, device=device)

        _, mask1 = filter_default(depth)
        assert mask1.any(), "8m should pass with max=10"

        filter_default.update_config(depth_maximum_distance=5.0)

        _, mask2 = filter_default(depth)
        assert not mask2.any(), "8m should fail with max=5"

    def test_update_flying_threshold(self, filter_default, depth_with_discontinuity):
        _, mask1 = filter_default(depth_with_discontinuity)
        initial_filtered = (~mask1).sum().item()

        filter_default.update_config(flying_pixel_threshold=0.1)
        _, mask2 = filter_default(depth_with_discontinuity)
        less_aggressive_filtered = (~mask2).sum().item()

        assert less_aggressive_filtered <= initial_filtered


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_different_image_size(self, filter_default, device):
        """Filter initialised at (480, 640) still works on a different shape
        by allocating fresh buffers for that call."""
        small_depth = torch.full((1, 240, 320), 2.0, dtype=torch.float32, device=device)

        filtered_depth, mask = filter_default(small_depth)
        assert filtered_depth.shape == (1, 240, 320)
        assert mask.shape == (1, 240, 320)

    def test_different_batch_size(self, filter_default, device):
        """Filter initialised at num_batch=1 still works on a larger batch."""
        depth = torch.full((4, 480, 640), 2.0, dtype=torch.float32, device=device)

        filtered_depth, mask = filter_default(depth)
        assert filtered_depth.shape == (4, 480, 640)
        assert mask.shape == (4, 480, 640)

    def test_zero_depth_filtered(self, filter_default, device):
        depth = torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)
        depth[0, 200:280, 280:360] = 0.0

        _, mask = filter_default(depth)

        assert not mask[0, 240, 320].item(), "Zero depth should be filtered"

    def test_negative_depth_filtered(self, filter_no_flying, device):
        depth = torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)
        depth[0, 200:280, :] = -1.0

        _, mask = filter_no_flying(depth)

        assert not mask[0, 240, 320].item(), "Negative depth should be filtered"


# ============================================================================
# Test Performance Characteristics
# ============================================================================


class TestPerformance:
    """Test performance-related aspects."""

    def test_large_image(self, device):
        depth_filter = FilterDepth(
            image_shape=(1080, 1920),
            bilateral_kernel_size=None,
            device=str(device),
        )
        depth = torch.full((1, 1080, 1920), 2.0, dtype=torch.float32, device=device)

        filtered_depth, mask = depth_filter(depth)

        assert filtered_depth.shape == (1, 1080, 1920)
        assert mask.shape == (1, 1080, 1920)

    def test_buffer_reuse(self, filter_default, device):
        """Test that pre-allocated buffers are reused across calls."""
        depth1 = torch.full((1, 480, 640), 2.0, dtype=torch.float32, device=device)
        depth2 = torch.full((1, 480, 640), 3.0, dtype=torch.float32, device=device)

        out1, _ = filter_default(depth1)
        out2, _ = filter_default(depth2)

        assert out1.data_ptr() == out2.data_ptr()

    def test_batched_buffer_reuse(self, device):
        """Pre-allocated batched buffer is reused when shape matches."""
        depth_filter = FilterDepth(
            image_shape=(480, 640),
            num_batch=2,
            bilateral_kernel_size=None,
            device=str(device),
        )
        d1 = torch.full((2, 480, 640), 2.0, dtype=torch.float32, device=device)
        d2 = torch.full((2, 480, 640), 3.0, dtype=torch.float32, device=device)

        out1, _ = depth_filter(d1)
        out2, _ = depth_filter(d2)

        assert out1.data_ptr() == out2.data_ptr()


# ============================================================================
# Test FilterDepthConfig
# ============================================================================


class TestFilterDepthConfig:
    """Test FilterDepthConfig dataclass."""

    def test_default_values(self):
        config = FilterDepthConfig()

        assert config.depth_minimum_distance == 0.1
        assert config.depth_maximum_distance == 10.0
        assert config.flying_pixel_threshold == 0.5
        assert config.bilateral_kernel_size == 5
        assert config.bilateral_sigma_spatial == 2.0
        assert config.bilateral_sigma_depth == 0.05

    def test_custom_values(self):
        config = FilterDepthConfig(
            depth_minimum_distance=0.2,
            depth_maximum_distance=5.0,
            flying_pixel_threshold=0.7,
            bilateral_kernel_size=7,
        )

        assert config.depth_minimum_distance == 0.2
        assert config.depth_maximum_distance == 5.0
        assert config.flying_pixel_threshold == 0.7
        assert config.bilateral_kernel_size == 7
