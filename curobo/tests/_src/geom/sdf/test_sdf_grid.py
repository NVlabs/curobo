# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for SDF grid module (deprecated but still needs coverage)."""

# Third Party
import torch

# CuRobo
from curobo._src.geom.sdf_grid import (
    SDFGrid,
    compute_sdf_gradient,
    lookup_distance,
)


class TestLookupDistance:
    """Test lookup_distance function."""

    def test_basic_lookup(self):
        """Test basic distance lookup."""
        # Create a simple 3x3x3 distance matrix
        num_voxels = torch.tensor([3, 3, 3])
        dist_matrix_flat = torch.arange(27, dtype=torch.float32)  # 3*3*3 = 27 voxels

        # Lookup at position (1, 1, 1) - should be index 13 (1*9 + 1*3 + 1)
        pt = torch.tensor([[1, 1, 1]], dtype=torch.long)
        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

        expected_index = 1 * (3 * 3) + 1 * 3 + 1  # = 13
        assert dist[0] == dist_matrix_flat[expected_index]

    def test_lookup_at_origin(self):
        """Test distance lookup at origin."""
        num_voxels = torch.tensor([5, 5, 5])
        dist_matrix_flat = torch.ones(125) * 10.0
        dist_matrix_flat[0] = 0.0  # Distance at origin is 0

        pt = torch.tensor([[0, 0, 0]], dtype=torch.long)
        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

        assert dist[0] == 0.0

    def test_lookup_batch(self):
        """Test batch distance lookup."""
        num_voxels = torch.tensor([4, 4, 4])
        dist_matrix_flat = torch.arange(64, dtype=torch.float32)

        # Multiple points
        pt = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.long)

        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

        assert dist.shape[0] == 4
        # Check first point (0,0,0) -> index 0
        assert dist[0] == 0.0
        # Check second point (1,0,0) -> index 16 (1*16 + 0*4 + 0)
        assert dist[1] == 16.0

    def test_lookup_different_grid_sizes(self):
        """Test lookup with non-cubic grids."""
        num_voxels = torch.tensor([2, 3, 4])
        dist_matrix_flat = torch.arange(24, dtype=torch.float32)  # 2*3*4 = 24

        pt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

        expected_index = 1 * (3 * 4) + 2 * 4 + 3  # = 12 + 8 + 3 = 23
        assert dist[0] == dist_matrix_flat[expected_index]


class TestComputeSdfGradient:
    """Test compute_sdf_gradient function."""

    def test_basic_gradient(self):
        """Test basic SDF gradient computation."""
        num_voxels = torch.tensor([5, 5, 5])
        # Create a simple distance field (sphere-like)
        dist_matrix_flat = torch.rand(125)

        pt = torch.tensor([[2, 2, 2]], dtype=torch.long)
        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

        grad = compute_sdf_gradient(pt, dist_matrix_flat, num_voxels, dist)

        # Gradient should have shape (1, 3) for x, y, z
        assert grad.shape == (1, 3)

    def test_gradient_at_boundary(self):
        """Test gradient computation at grid boundary."""
        num_voxels = torch.tensor([3, 3, 3])
        dist_matrix_flat = torch.ones(27)

        # Point at boundary
        pt = torch.tensor([[0, 0, 0]], dtype=torch.long)
        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

        grad = compute_sdf_gradient(pt, dist_matrix_flat, num_voxels, dist)

        assert grad.shape == (1, 3)

    def test_gradient_batch(self):
        """Test batch gradient computation."""
        num_voxels = torch.tensor([4, 4, 4])
        dist_matrix_flat = torch.rand(64)

        pt = torch.tensor([
            [1, 1, 1],
            [2, 2, 2]
        ], dtype=torch.long)
        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

        grad = compute_sdf_gradient(pt, dist_matrix_flat, num_voxels, dist)

        # Should compute gradients for both points
        assert grad.shape == (2, 3)


class TestSDFGrid:
    """Test SDFGrid autograd function."""

    def test_sdf_grid_forward(self):
        """Test SDFGrid forward pass."""
        num_voxels = torch.tensor([5, 5, 5])
        dist_matrix_flat = torch.arange(125, dtype=torch.float32)

        # Query point
        pt = torch.tensor([[2.5, 2.5, 2.5]], dtype=torch.float32)

        # Forward pass
        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)

        # Should return distance with extra dimension
        assert dist.shape == (1, 1)
        assert dist.item() >= 0

    def test_sdf_grid_backward(self):
        """Test SDFGrid backward pass (gradient computation)."""
        num_voxels = torch.tensor([5, 5, 5])
        dist_matrix_flat = torch.rand(125, dtype=torch.float32)

        # Query point with requires_grad
        pt = torch.tensor([[2.5, 2.5, 2.5]], dtype=torch.float32, requires_grad=True)

        # Forward and backward pass
        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)
        dist.sum().backward()

        # Gradient should be computed
        assert pt.grad is not None
        assert pt.grad.shape == pt.shape

    def test_sdf_grid_multiple_points(self):
        """Test SDFGrid with multiple query points."""
        num_voxels = torch.tensor([4, 4, 4])
        dist_matrix_flat = torch.rand(64, dtype=torch.float32)

        # Multiple query points
        pt = torch.tensor([
            [1.5, 1.5, 1.5],
            [2.5, 2.5, 2.5]
        ], dtype=torch.float32)

        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)

        assert dist.shape == (2, 1)

    def test_sdf_grid_gradient_flow(self):
        """Test that gradients flow through SDFGrid correctly."""
        num_voxels = torch.tensor([5, 5, 5])
        dist_matrix_flat = torch.rand(125, dtype=torch.float32)

        pt = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32, requires_grad=True)

        # Forward pass
        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)

        # Create a loss and backpropagate
        loss = dist.sum()
        loss.backward()

        # Verify gradient exists and has correct shape
        assert pt.grad is not None
        assert pt.grad.shape == (1, 3)


class TestSDFGridEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_voxel_grid(self):
        """Test with minimal grid size."""
        num_voxels = torch.tensor([2, 2, 2])
        dist_matrix_flat = torch.ones(8)

        pt = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)

        assert dist.shape == (1, 1)

    def test_large_grid(self):
        """Test with larger grid."""
        num_voxels = torch.tensor([10, 10, 10])
        dist_matrix_flat = torch.rand(1000)

        pt = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32)
        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)

        assert dist.shape == (1, 1)

    def test_non_cubic_grid(self):
        """Test with non-cubic grid dimensions."""
        num_voxels = torch.tensor([3, 5, 7])
        dist_matrix_flat = torch.rand(105)  # 3*5*7

        pt = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)

        assert dist.shape == (1, 1)

    def test_lookup_consistency(self):
        """Test that lookup is consistent with indexing formula."""
        num_voxels = torch.tensor([4, 5, 6])
        dist_matrix_flat = torch.arange(120, dtype=torch.float32)

        # Test several positions
        positions = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [3, 4, 5]
        ]

        for pos in positions:
            pt = torch.tensor([pos], dtype=torch.long)
            dist = lookup_distance(pt, dist_matrix_flat, num_voxels)

            # Calculate expected index
            expected_idx = pos[0] * (num_voxels[1] * num_voxels[2]) + pos[1] * num_voxels[2] + pos[2]
            expected_dist = dist_matrix_flat[int(expected_idx)]

            assert torch.allclose(dist[0], expected_dist)


class TestSDFGridIntegration:
    """Integration tests for SDF grid operations."""

    def test_forward_backward_integration(self):
        """Test complete forward-backward pass."""
        num_voxels = torch.tensor([5, 5, 5])
        dist_matrix_flat = torch.rand(125, dtype=torch.float32)

        # Create query points that require gradients
        pts = torch.tensor([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ], dtype=torch.float32, requires_grad=True)

        # Forward pass
        distances = SDFGrid.apply(pts, dist_matrix_flat, num_voxels)

        # Backward pass
        loss = distances.sum()
        loss.backward()

        # Verify everything worked
        assert distances.shape == (3, 1)
        assert pts.grad is not None
        assert pts.grad.shape == pts.shape

    def test_gradient_numerical_stability(self):
        """Test that gradients are numerically stable."""
        num_voxels = torch.tensor([5, 5, 5])
        dist_matrix_flat = torch.rand(125, dtype=torch.float32)

        pt = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32, requires_grad=True)

        dist = SDFGrid.apply(pt, dist_matrix_flat, num_voxels)
        dist.sum().backward()

        # Gradients should not be NaN or Inf
        assert not torch.isnan(pt.grad).any()
        assert not torch.isinf(pt.grad).any()

