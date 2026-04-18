# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for mesh SDF alignment Warp kernel.

Tests the wp_mesh_sdf_alignment kernel for computing Jacobian and residuals
for pose estimation via Levenberg-Marquardt optimization.
"""

# Third Party
import numpy as np
import pytest
import torch
import trimesh
import warp as wp

# CuRobo
from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.wp_mesh_sdf_alignment import (
    jacobian_reduce_kernel,
    mesh_surface_distance_query_kernel,
)
from curobo._src.util.warp import init_warp

# Initialize Warp at module load
init_warp()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def simple_box_mesh():
    """Create a simple box mesh for testing."""
    return trimesh.creation.box(extents=[0.2, 0.2, 0.2])


@pytest.fixture(scope="module")
def sphere_mesh():
    """Create a sphere mesh for testing."""
    return trimesh.creation.icosphere(subdivisions=3, radius=0.1)


# ============================================================================
# Helper Functions
# ============================================================================


def perturb_surface_points(
    points: torch.Tensor,
    normals: torch.Tensor,
    offset: float = 1e-4,
) -> torch.Tensor:
    """Move surface points slightly along their normals.

    The kernel rejects points with distance <= 1e-8 (numerical stability).
    This helper pushes surface-sampled points slightly off the surface
    to simulate realistic depth sensor observations.

    Args:
        points: [N, 3] surface points.
        normals: [N, 3] surface normals.
        offset: Distance to move points along normals.

    Returns:
        Perturbed points [N, 3].
    """
    return points + normals * offset


def run_alignment_kernel(
    robot_mesh: RobotMesh,
    observed_points: torch.Tensor,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    max_distance: float = 1.0,
    distance_threshold: float = 0.5,
    use_huber: bool = False,
    huber_delta: float = 0.02,
):
    """Run two-pass alignment kernels and return results.

    Args:
        robot_mesh: RobotMesh instance with mesh.
        observed_points: [N, 3] observed points in world frame.
        position: [3] object position.
        quaternion: [4] object quaternion (wxyz).
        max_distance: Maximum SDF query distance.
        distance_threshold: Distance threshold for valid correspondences.
        use_huber: Use Huber loss.
        huber_delta: Huber threshold.

    Returns:
        Tuple of (JtJ, Jtr, sum_sq_residuals, valid_count) as torch tensors.
    """
    device = "cuda:0"
    n_points = len(observed_points)

    # Ensure tensors are on device
    points = observed_points.to(device=device, dtype=torch.float32).contiguous()
    pos = position.to(device=device, dtype=torch.float32).contiguous()
    quat = quaternion.to(device=device, dtype=torch.float32).contiguous()

    # Intermediate buffers for Pass 1
    distance_values = torch.zeros(n_points, dtype=torch.float32, device=device)
    gradients_world = torch.zeros(n_points, 3, dtype=torch.float32, device=device)
    valid_mask = torch.zeros(n_points, dtype=torch.int32, device=device)

    # Output buffers for Pass 2
    JtJ_out = torch.zeros(36, dtype=torch.float32, device=device)
    Jtr_out = torch.zeros(6, dtype=torch.float32, device=device)
    sum_sq_residuals = torch.zeros(1, dtype=torch.float32, device=device)
    valid_count = torch.zeros(1, dtype=torch.int32, device=device)

    # Pass 1: Query mesh surface distances
    wp.launch(
        kernel=mesh_surface_distance_query_kernel,
        dim=n_points,
        inputs=[
            wp.from_torch(points, dtype=wp.vec3),
            n_points,
            wp.from_torch(pos, dtype=wp.float32),
            wp.from_torch(quat, dtype=wp.float32),
            robot_mesh.mesh_id,
            max_distance,
            distance_threshold,
            wp.from_torch(distance_values, dtype=wp.float32),
            wp.from_torch(gradients_world, dtype=wp.vec3),
            wp.from_torch(valid_mask, dtype=wp.int32),
        ],
        device=device,
    )

    # Pass 2: Compute Jacobian and block-reduce
    wp.launch(
        kernel=jacobian_reduce_kernel,
        dim=n_points,
        inputs=[
            wp.from_torch(points, dtype=wp.vec3),
            wp.from_torch(distance_values, dtype=wp.float32),
            wp.from_torch(gradients_world, dtype=wp.vec3),
            wp.from_torch(valid_mask, dtype=wp.int32),
            n_points,
            1 if use_huber else 0,
            huber_delta,
            wp.from_torch(JtJ_out, dtype=wp.float32),
            wp.from_torch(Jtr_out, dtype=wp.float32),
            wp.from_torch(sum_sq_residuals, dtype=wp.float32),
            wp.from_torch(valid_count, dtype=wp.int32),
        ],
        block_dim=256,
        device=device,
    )

    wp.synchronize()

    return JtJ_out, Jtr_out, sum_sq_residuals, valid_count


# ============================================================================
# Test with Identity Pose (Points on Surface)
# ============================================================================


class TestIdentityPose:
    """Test kernel with identity pose where points are slightly off surface."""

    def test_residuals_near_zero(self, simple_box_mesh):
        """Test that residuals are near zero when points are near surface."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Sample points on surface and perturb slightly off
        # Kernel rejects points with dist <= 1e-8 for numerical stability
        points, normals = robot_mesh.sample_surface_points(500)
        points = perturb_surface_points(points, normals, offset=1e-4)

        # Identity pose
        position = torch.zeros(3)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])  # wxyz identity

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        # All points should be valid (they're slightly off surface)
        assert valid_count.item() == 500

        # Sum of squared residuals should be very small (points are 1e-4 away)
        mean_sq_residual = sum_sq.item() / valid_count.item()
        assert mean_sq_residual < 1e-6  # < 1mm^2

    def test_jtr_near_zero(self, simple_box_mesh):
        """Test that J^T @ r is small at near-optimal pose."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Sample points on surface and perturb slightly
        # With 1e-4 offset, Jtr will be small but not exactly zero
        points, normals = robot_mesh.sample_surface_points(500)
        points = perturb_surface_points(points, normals, offset=1e-4)

        # Identity pose
        position = torch.zeros(3)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        # J^T @ r should be small (gradient near optimum with small perturbation)
        assert torch.abs(Jtr).max() < 1e-2


# ============================================================================
# Test with Perturbed Pose
# ============================================================================


class TestPerturbedPose:
    """Test kernel with perturbed pose (points not on surface)."""

    def test_translation_perturbation(self, simple_box_mesh):
        """Test residuals with translation perturbation."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Sample points on surface
        points, _ = robot_mesh.sample_surface_points(500)

        # Perturbed pose (translate by 1cm)
        position = torch.tensor([0.01, 0.0, 0.0])
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        # Should have valid correspondences
        assert valid_count.item() > 0

        # Residuals should be non-zero (pose is wrong)
        mean_sq_residual = sum_sq.item() / max(valid_count.item(), 1)
        assert mean_sq_residual > 1e-6

        # Jtr should be non-zero (gradient away from optimum)
        assert torch.abs(Jtr).max() > 1e-6

    def test_rotation_perturbation(self, sphere_mesh):
        """Test residuals with rotation perturbation."""
        robot_mesh = RobotMesh.from_trimesh(sphere_mesh, device="cuda:0")

        # Sample points on surface
        points, _ = robot_mesh.sample_surface_points(500)

        # Small rotation (5 degrees around z-axis)
        angle = 0.087  # ~5 degrees in radians
        quaternion = torch.tensor([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        position = torch.zeros(3)

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        # Should have valid correspondences
        assert valid_count.item() > 0


# ============================================================================
# Test JtJ Properties
# ============================================================================


class TestJtJProperties:
    """Test properties of J^T @ J matrix."""

    def test_jtj_is_symmetric(self, simple_box_mesh):
        """Test that J^T @ J is symmetric."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Sample points with perturbation to get non-trivial Jacobian
        points, _ = robot_mesh.sample_surface_points(500)

        position = torch.tensor([0.01, 0.005, -0.01])
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        # Reshape to 6x6
        JtJ_matrix = JtJ.view(6, 6)

        # Check symmetry
        assert torch.allclose(JtJ_matrix, JtJ_matrix.T, atol=1e-5)

    def test_jtj_is_positive_semidefinite(self, simple_box_mesh):
        """Test that J^T @ J is positive semi-definite."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, _ = robot_mesh.sample_surface_points(500)

        position = torch.tensor([0.01, 0.005, -0.01])
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        JtJ_matrix = JtJ.view(6, 6)

        # Eigenvalues should be non-negative
        eigenvalues = torch.linalg.eigvalsh(JtJ_matrix)
        assert (eigenvalues >= -1e-6).all()

    def test_jtj_diagonal_positive(self, simple_box_mesh):
        """Test that J^T @ J diagonal is positive."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, _ = robot_mesh.sample_surface_points(500)

        position = torch.tensor([0.01, 0.005, -0.01])
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        JtJ_matrix = JtJ.view(6, 6)

        # Diagonal should be positive (sum of squares)
        assert (JtJ_matrix.diag() >= 0).all()


# ============================================================================
# Test Huber Loss
# ============================================================================


class TestHuberLoss:
    """Test Huber loss functionality."""

    def test_huber_reduces_outlier_influence(self, simple_box_mesh):
        """Test that Huber loss reduces influence of outliers."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Sample points and add an outlier
        points, _ = robot_mesh.sample_surface_points(100)

        # Large pose error to create large residuals
        position = torch.tensor([0.05, 0.0, 0.0])  # 5cm translation
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Without Huber
        JtJ_no_huber, Jtr_no_huber, sum_sq_no_huber, _ = run_alignment_kernel(
            robot_mesh,
            points,
            position,
            quaternion,
            use_huber=False,
        )

        # With Huber
        JtJ_huber, Jtr_huber, sum_sq_huber, _ = run_alignment_kernel(
            robot_mesh,
            points,
            position,
            quaternion,
            use_huber=True,
            huber_delta=0.01,  # 1cm threshold
        )

        # Huber should reduce sum of squared residuals for large errors
        # (because it uses linear loss beyond threshold)
        assert sum_sq_huber.item() < sum_sq_no_huber.item()


# ============================================================================
# Test with Different Mesh Shapes
# ============================================================================


class TestDifferentMeshes:
    """Test with different mesh geometries."""

    def test_sphere_mesh(self, sphere_mesh):
        """Test with sphere mesh."""
        robot_mesh = RobotMesh.from_trimesh(sphere_mesh, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(500)
        points = perturb_surface_points(points, normals, offset=1e-4)
        position = torch.zeros(3)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        assert valid_count.item() == 500
        mean_sq_residual = sum_sq.item() / valid_count.item()
        assert mean_sq_residual < 1e-6

    def test_cylinder_mesh(self):
        """Test with cylinder mesh."""
        cylinder = trimesh.creation.cylinder(radius=0.05, height=0.2)
        robot_mesh = RobotMesh.from_trimesh(cylinder, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(500)
        points = perturb_surface_points(points, normals, offset=1e-4)
        position = torch.zeros(3)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        assert valid_count.item() == 500
        mean_sq_residual = sum_sq.item() / valid_count.item()
        assert mean_sq_residual < 1e-6


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_single_point(self, simple_box_mesh):
        """Test with single point."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(1)
        points = perturb_surface_points(points, normals, offset=1e-4)
        position = torch.zeros(3)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        assert valid_count.item() == 1

    def test_many_points(self, simple_box_mesh):
        """Test with many points."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(10000)
        points = perturb_surface_points(points, normals, offset=1e-4)
        position = torch.zeros(3)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh, points, position, quaternion
        )

        assert valid_count.item() == 10000

    def test_points_outside_threshold(self, simple_box_mesh):
        """Test that points far from surface are rejected."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Create points far from surface
        points = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device="cuda:0",
        )
        position = torch.zeros(3)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        JtJ, Jtr, sum_sq, valid_count = run_alignment_kernel(
            robot_mesh,
            points,
            position,
            quaternion,
            distance_threshold=0.1,  # 10cm threshold, points are 90cm away
        )

        # No valid correspondences (points too far)
        assert valid_count.item() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

