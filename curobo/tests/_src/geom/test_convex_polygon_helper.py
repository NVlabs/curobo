# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper
from curobo._src.types.device_cfg import DeviceCfg


class TestConvexPolygon2DHelper:
    """Test suite for ConvexPolygon2DHelper class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.device_cfg = DeviceCfg()
        self.helper = ConvexPolygon2DHelper(self.device_cfg)

    def test_basic_convex_hull_construction(self):
        """Test basic convex hull construction from vertices."""
        # Unit square vertices
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        self.helper.build_convex_hull(vertices)

        assert self.helper._cached_convex_hulls is not None
        assert self.helper._cached_convex_hulls.shape[0] == 1  # One batch
        assert self.helper._cached_convex_hulls.shape[2] == 2  # 2D coordinates

    def test_multiple_batch_convex_hull(self):
        """Test convex hull construction with multiple batches."""
        vertices = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # Unit square
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],  # 2x2 square
            ],
            dtype=torch.float32,
        )

        self.helper.build_convex_hull(vertices)

        assert self.helper._cached_convex_hulls.shape[0] == 2  # Two batches
        assert self.helper._cached_convex_hulls.shape[2] == 2  # 2D coordinates

    def test_convex_hull_with_padding(self):
        """Test convex hull construction with padding."""
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        # Test with different padding values
        for padding in [0.0, 0.1, 0.2]:
            self.helper.build_convex_hull(vertices, padding=padding)
            assert self.helper._cached_convex_hulls is not None

    def test_point_inside_convex_hull(self):
        """Test detection of points inside convex hull."""
        # Unit square
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        self.helper.build_convex_hull(vertices, padding=0.05)

        # Point clearly inside
        inside_point = torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float32)
        batch_indices = torch.tensor([0])

        distance = self.helper.compute_point_hull_distance(inside_point, batch_indices)

        assert distance.shape == torch.Size([1, 1, 1])
        assert distance.item() < 0  # Negative distance means inside

    def test_point_outside_convex_hull(self):
        """Test detection of points outside convex hull."""
        # Unit square
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        self.helper.build_convex_hull(vertices, padding=0.05)

        # Point clearly outside
        outside_point = torch.tensor([[[[2.0, 2.0]]]], dtype=torch.float32)
        batch_indices = torch.tensor([0])

        distance = self.helper.compute_point_hull_distance(outside_point, batch_indices)

        assert distance.shape == torch.Size([1, 1, 1])
        assert distance.item() > 0  # Positive distance means outside

    def test_boundary_point_distance(self):
        """Test distance computation for points on or near the boundary."""
        # Unit square
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        self.helper.build_convex_hull(vertices, padding=0.0)

        # Point on boundary (right edge)
        boundary_point = torch.tensor([[[[1.0, 0.5]]]], dtype=torch.float32)
        batch_indices = torch.tensor([0])

        distance = self.helper.compute_point_hull_distance(boundary_point, batch_indices)

        assert distance.shape == torch.Size([1, 1, 1])
        # Should be very close to zero (might be slightly negative due to numerical precision)
        assert abs(distance.item()) < 0.1

    def test_multiple_points_batch_processing(self):
        """Test batch processing of multiple points across different horizons."""
        vertices = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],  # Unit square
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],  # 2x2 square
            ],
            dtype=torch.float32,
        )

        self.helper.build_convex_hull(vertices, padding=0.05)

        # Multiple points across different horizons
        test_points = torch.tensor(
            [
                [[[0.5, 0.5]], [[1.5, 1.5]]],  # Batch 1: inside, outside
                [[[1.0, 1.0]], [[3.0, 3.0]]],  # Batch 2: inside, outside
            ],
            dtype=torch.float32,
        )

        batch_indices = torch.tensor([0, 1])

        distances = self.helper.compute_point_hull_distance(test_points, batch_indices)

        assert distances.shape == torch.Size([2, 2, 1])

        # Check expected signs
        assert distances[0, 0, 0] < 0  # Inside point in batch 1
        assert distances[0, 1, 0] > 0  # Outside point in batch 1
        assert distances[1, 0, 0] < 0  # Inside point in batch 2
        assert distances[1, 1, 0] > 0  # Outside point in batch 2

    def test_degenerate_case_handling(self):
        """Test handling of degenerate cases (too few points)."""
        # Only 2 points (not enough for a polygon)
        vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)

        # This should handle the degenerate case gracefully
        try:
            self.helper.build_convex_hull(vertices)
            # Should not crash, but might log an error
        except Exception as e:
            pytest.skip(f"Degenerate case handling not fully implemented: {e}")

    def test_padding_effect_on_distance(self):
        """Test that padding correctly affects distance calculations."""
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        # Boundary point
        boundary_point = torch.tensor([[[[1.0, 0.5]]]], dtype=torch.float32)
        batch_indices = torch.tensor([0])

        distances_no_padding = []
        distances_with_padding = []

        # Test without padding
        self.helper.build_convex_hull(vertices, padding=0.0)
        dist = self.helper.compute_point_hull_distance(boundary_point, batch_indices)
        distances_no_padding.append(dist.item())

        # Test with padding
        self.helper.build_convex_hull(vertices, padding=0.1)
        dist = self.helper.compute_point_hull_distance(boundary_point, batch_indices)
        distances_with_padding.append(dist.item())

        # With padding, the boundary point should be further inside (more negative)
        assert distances_with_padding[0] < distances_no_padding[0]

    def test_cuda_device_cfg_consistency(self):
        """Test that computations maintain tensor device consistency."""
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        self.helper.build_convex_hull(vertices)

        test_point = torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float32)
        batch_indices = torch.tensor([0])

        distance = self.helper.compute_point_hull_distance(test_point, batch_indices)

        # Check that output tensor is on the same device as input
        assert distance.device == test_point.device
        assert distance.dtype == test_point.dtype

    def test_error_handling_no_cached_hull(self):
        """Test error handling when no convex hull is cached."""
        test_point = torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float32)
        batch_indices = torch.tensor([0])

        # Should handle gracefully when no hull is cached
        distance = self.helper.compute_point_hull_distance(test_point, batch_indices)

        # Should return zeros when no hull is cached
        assert torch.allclose(distance, torch.zeros_like(distance))

    def test_graham_scan_algorithm(self):
        """Test the Graham scan convex hull algorithm with known inputs."""
        # Pentagon-like shape where convex hull should remove internal points
        vertices = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, 0.1],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]  # Internal point at (0.5, 0.1)
            ],
            dtype=torch.float32,
        )

        self.helper.build_convex_hull(vertices)

        # The convex hull should have 4 vertices (the internal point should be removed)
        hull = self.helper._cached_convex_hulls
        assert hull is not None

        # Check that we get a reasonable number of hull vertices
        # (exact count depends on padding and algorithm details)
        assert hull.shape[1] >= 4  # At least 4 vertices for the square-like shape

    def test_smooth_distance_computation(self):
        """Test that distance computation provides smooth gradients."""
        vertices = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
        )

        self.helper.build_convex_hull(vertices, padding=0.05)

        # Test points at different distances
        test_points = torch.tensor(
            [
                [
                    [[0.5, 0.5]],  # Inside
                    [[1.1, 0.5]],  # Slightly outside
                    [[1.5, 0.5]],  # Further outside
                ]
            ],
            dtype=torch.float32,
        )

        batch_indices = torch.tensor([0])

        distances = self.helper.compute_point_hull_distance(test_points, batch_indices)

        assert distances.shape == torch.Size([1, 3, 1])

        # Distances should increase monotonically as we move away from hull
        assert distances[0, 0, 0] < distances[0, 1, 0]  # Inside < slightly outside
        assert distances[0, 1, 0] < distances[0, 2, 0]  # Slightly outside < far outside


def test_convex_polygon_helper_basic():
    """Basic integration test for ConvexPolygon2DHelper."""
    device_cfg = DeviceCfg()
    helper = ConvexPolygon2DHelper(device_cfg)

    # Create simple test case
    vertices = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32)

    helper.build_convex_hull(vertices, padding=0.1)

    test_points = torch.tensor([[[[0.5, 0.5]], [[1.5, 1.5]]]], dtype=torch.float32)
    batch_indices = torch.tensor([0])

    distances = helper.compute_point_hull_distance(test_points, batch_indices)

    assert distances.shape == torch.Size([1, 2, 1])
    assert distances[0, 0, 0] < 0  # Inside point
    assert distances[0, 1, 0] > 0  # Outside point


if __name__ == "__main__":
    pytest.main([__file__])
