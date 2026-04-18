# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for computer vision functions."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.cv import (
    extract_depth_from_structured_pointcloud,
    get_projection_rays,
    project_depth_to_pointcloud,
    project_depth_using_rays,
)


@pytest.fixture
def intrinsics_matrix():
    """Create a sample camera intrinsics matrix."""
    # Typical camera intrinsics: fx, fy, cx, cy
    fx, fy = 525.0, 525.0
    cx, cy = 320.0, 240.0

    K = torch.tensor([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    return K


@pytest.fixture
def batch_intrinsics():
    """Create batch of intrinsics matrices."""
    K1 = torch.tensor([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
    K2 = torch.tensor([[500.0, 0.0, 300.0], [0.0, 500.0, 230.0], [0.0, 0.0, 1.0]])

    return torch.stack([K1, K2])


class TestProjectDepthToPointcloud:
    """Test project_depth_to_pointcloud function."""

    def test_basic_projection(self, intrinsics_matrix):
        """Test basic depth to pointcloud projection."""
        # Create simple depth image
        depth_image = torch.ones(480, 640) * 1.0  # 1 meter depth

        pointcloud = project_depth_to_pointcloud(depth_image, intrinsics_matrix)

        assert pointcloud.shape == (480, 640, 3)
        # Center point should be approximately at (0, 0, 1)
        center_point = pointcloud[240, 320]
        assert torch.allclose(center_point[2], torch.tensor(1.0), atol=0.01)

    def test_projection_with_varying_depth(self, intrinsics_matrix):
        """Test projection with varying depth values."""
        depth_image = torch.rand(100, 100) * 5.0  # Random depths 0-5m

        pointcloud = project_depth_to_pointcloud(depth_image, intrinsics_matrix)

        assert pointcloud.shape == (100, 100, 3)
        # Z values should match depth values
        assert torch.allclose(pointcloud[..., 2], depth_image, atol=1e-5)

    def test_projection_zero_depth(self, intrinsics_matrix):
        """Test projection with zero depth."""
        depth_image = torch.zeros(50, 50)

        pointcloud = project_depth_to_pointcloud(depth_image, intrinsics_matrix)

        assert pointcloud.shape == (50, 50, 3)
        # All points should be at origin
        assert torch.allclose(pointcloud, torch.zeros_like(pointcloud))


class TestGetProjectionRays:
    """Test get_projection_rays function."""

    def test_basic_ray_generation(self, batch_intrinsics):
        """Test basic projection ray generation."""
        height, width = 480, 640

        rays = get_projection_rays(height, width, batch_intrinsics)

        assert rays.shape == (2, height * width, 3)

    def test_ray_generation_with_depth_scaling(self, batch_intrinsics):
        """Test ray generation with depth scaling factor."""
        height, width = 240, 320
        depth_to_meter = 0.001  # mm to meters

        rays = get_projection_rays(height, width, batch_intrinsics, depth_to_meter)

        assert rays.shape == (2, height * width, 3)
        # Rays should be scaled
        assert rays.abs().max() < 10.0  # Reasonable range

    def test_ray_generation_small_image(self, intrinsics_matrix):
        """Test ray generation with small image."""
        height, width = 10, 10
        intrinsics_batch = intrinsics_matrix.unsqueeze(0)

        rays = get_projection_rays(height, width, intrinsics_batch)

        assert rays.shape == (1, height * width, 3)

    def test_ray_generation_single_pixel(self, intrinsics_matrix):
        """Test ray generation with 1x1 image."""
        height, width = 1, 1
        intrinsics_batch = intrinsics_matrix.unsqueeze(0)

        rays = get_projection_rays(height, width, intrinsics_batch)

        assert rays.shape == (1, 1, 3)


class TestExtractDepthFromStructuredPointcloud:
    """Test extract_depth_from_structured_pointcloud function."""

    def test_basic_depth_extraction(self):
        """Test basic depth extraction from structured pointcloud."""
        batch, height, width = 1, 10, 10
        # Create structured pointcloud [b, h, w, 3]
        pointcloud = torch.rand(batch, height, width, 3)
        pointcloud[:, :, :, 2] = torch.arange(height * width, dtype=torch.float32).reshape(
            height, width
        ) * 0.1

        output_image = torch.zeros(batch, height, width)
        depth_image = extract_depth_from_structured_pointcloud(pointcloud, output_image)

        assert depth_image.shape == (batch, height, width)
        # Z-coordinates should match depth
        assert torch.allclose(depth_image[0], pointcloud[0, :, :, 2])

    def test_depth_extraction_different_sizes(self):
        """Test depth extraction with different image sizes."""
        batch, height, width = 2, 20, 30
        pointcloud = torch.rand(batch, height, width, 3)
        pointcloud[:, :, :, 2] = 1.5  # Constant depth

        output_image = torch.zeros(batch, height, width)
        depth_image = extract_depth_from_structured_pointcloud(pointcloud, output_image)

        assert depth_image.shape == (batch, height, width)
        assert torch.allclose(depth_image, torch.ones(batch, height, width) * 1.5)

    def test_depth_extraction_preserves_structure(self):
        """Test that spatial structure is preserved."""
        batch, height, width = 1, 5, 5
        pointcloud = torch.zeros(batch, height, width, 3)
        # Set unique depth for each pixel
        for h in range(height):
            for w in range(width):
                pointcloud[0, h, w, 2] = h * width + w

        output_image = torch.zeros(batch, height, width)
        depth_image = extract_depth_from_structured_pointcloud(pointcloud, output_image)

        # Each pixel should have its unique depth value
        for h in range(height):
            for w in range(width):
                assert depth_image[0, h, w] == h * width + w


class TestProjectDepthUsingRays:
    """Test project_depth_using_rays function."""

    def test_basic_depth_projection_with_rays(self, batch_intrinsics):
        """Test basic depth projection using rays."""
        height, width = 100, 100
        depth_image = torch.ones(2, height, width) * 2.0  # 2 meters

        rays = get_projection_rays(height, width, batch_intrinsics)

        points = project_depth_using_rays(depth_image, rays)

        assert points.shape == (2, height * width, 3)

    def test_depth_projection_with_filtering(self, intrinsics_matrix):
        """Test depth projection with origin filtering."""
        height, width = 50, 50
        intrinsics_batch = intrinsics_matrix.unsqueeze(0)

        # Create depth with some near-zero values
        depth_image = torch.rand(1, height, width) * 3.0
        depth_image[0, :10, :10] = 0.005  # Near-zero region

        rays = get_projection_rays(height, width, intrinsics_batch)

        # With filtering
        points_filtered = project_depth_using_rays(
            depth_image, rays, filter_origin=True, depth_threshold=0.01
        )

        # Without filtering
        points_unfiltered = project_depth_using_rays(
            depth_image, rays, filter_origin=False
        )

        assert points_filtered.shape == points_unfiltered.shape
        # Filtered points should have zeros where depth was below threshold
        assert torch.sum(points_filtered.abs()) < torch.sum(points_unfiltered.abs())

    def test_depth_projection_all_valid(self, intrinsics_matrix):
        """Test depth projection without filtering."""
        height, width = 30, 40
        intrinsics_batch = intrinsics_matrix.unsqueeze(0)

        depth_image = torch.rand(1, height, width) * 5.0 + 0.5  # 0.5-5.5m
        rays = get_projection_rays(height, width, intrinsics_batch)

        points = project_depth_using_rays(depth_image, rays, filter_origin=False)

        assert points.shape == (1, height * width, 3)
        # All points should be non-zero
        assert points.abs().sum() > 0


class TestCVIntegration:
    """Test integration of CV functions."""

    def test_depth_to_pc_to_depth_roundtrip(self, intrinsics_matrix):
        """Test converting depth to PC and back."""
        height, width = 50, 50
        original_depth = torch.rand(height, width) * 3.0

        # Depth to pointcloud
        pc = project_depth_to_pointcloud(original_depth, intrinsics_matrix)

        # Extract Z values (depth)
        reconstructed_depth = pc[..., 2]

        # Should match original depth
        assert torch.allclose(reconstructed_depth, original_depth, atol=1e-5)

    def test_rays_projection_integration(self, intrinsics_matrix):
        """Test ray-based projection."""
        height, width = 100, 100
        intrinsics_batch = intrinsics_matrix.unsqueeze(0)

        # Create depth image
        depth_image = torch.rand(1, height, width) * 2.0 + 1.0

        # Get rays
        rays = get_projection_rays(height, width, intrinsics_batch)

        # Project using rays
        points = project_depth_using_rays(depth_image, rays)

        assert points.shape == (1, height * width, 3)
        # Points should have reasonable coordinates
        assert points.abs().max() < 100.0  # Sanity check


class TestCVEdgeCases:
    """Test edge cases in CV functions."""

    def test_single_pixel_depth(self, intrinsics_matrix):
        """Test with 1x1 depth image."""
        depth_image = torch.tensor([[2.0]])

        pc = project_depth_to_pointcloud(depth_image, intrinsics_matrix)

        assert pc.shape == (1, 1, 3)

    def test_batch_intrinsics_different_sizes(self):
        """Test with different camera parameters."""
        K_wide = torch.tensor([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
        K_narrow = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])

        intrinsics_batch = torch.stack([K_wide, K_narrow])

        height, width = 100, 100
        rays = get_projection_rays(height, width, intrinsics_batch)

        # Different intrinsics should produce different rays
        assert rays.shape == (2, height * width, 3)
        assert not torch.allclose(rays[0], rays[1])

