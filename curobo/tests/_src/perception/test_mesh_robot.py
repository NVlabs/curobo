# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RobotMesh class.

Tests both rigid object and articulated robot mesh functionality,
including vertex updates, BVH refit, and surface point sampling.
"""

# Third Party
import numpy as np
import pytest
import torch
import trimesh
import warp as wp

# CuRobo
from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

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
    return trimesh.creation.icosphere(subdivisions=2, radius=0.1)


@pytest.fixture(scope="module")
def franka_kinematics(cuda_device_cfg):
    """Load Franka robot kinematics for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(
        robot_data,
        tool_frames=robot_data["robot_cfg"]["kinematics"]["mesh_link_names"],
    )
    return Kinematics(cfg)


# ============================================================================
# Test RobotMesh.from_trimesh (Rigid Objects)
# ============================================================================


class TestRobotMeshFromTrimesh:
    """Test RobotMesh creation from trimesh (rigid objects)."""

    def test_basic_creation(self, simple_box_mesh):
        """Test basic creation from trimesh."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        assert robot_mesh is not None
        assert robot_mesh.n_vertices == len(simple_box_mesh.vertices)
        assert robot_mesh.n_faces == len(simple_box_mesh.faces)
        assert not robot_mesh.is_articulated
        assert robot_mesh.current_joint_angles is None

    def test_mesh_property(self, simple_box_mesh):
        """Test that mesh property returns valid wp.Mesh."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        assert isinstance(robot_mesh.mesh, wp.Mesh)
        assert robot_mesh.mesh_id is not None

    def test_vertices_on_device(self, simple_box_mesh):
        """Test that vertices are on correct device."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        assert robot_mesh.vertices.device.type == "cuda"
        assert robot_mesh.vertices.dtype == torch.float32

    def test_update_is_noop_for_rigid(self, simple_box_mesh):
        """Test that update() is no-op for rigid objects."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Store original vertices
        original_verts = robot_mesh.vertices.clone()

        # Call update with dummy joint angles
        robot_mesh.update(torch.zeros(7, device="cuda:0"))

        # Vertices should be unchanged
        assert torch.allclose(robot_mesh.vertices, original_verts)

    def test_different_mesh_shapes(self, simple_box_mesh, sphere_mesh):
        """Test with different mesh shapes."""
        box_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        sphere = RobotMesh.from_trimesh(sphere_mesh, device="cuda:0")

        assert box_mesh.n_vertices != sphere.n_vertices
        assert box_mesh.n_faces != sphere.n_faces


# ============================================================================
# Test RobotMesh.from_kinematics (Articulated Robots)
# ============================================================================


class TestRobotMeshFromKinematics:
    """Test RobotMesh creation from Kinematics (articulated robots)."""

    def test_basic_creation(self, franka_kinematics):
        """Test basic creation from kinematics."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        assert robot_mesh is not None
        assert robot_mesh.n_vertices > 0
        assert robot_mesh.n_faces > 0
        assert robot_mesh.is_articulated
        assert robot_mesh.current_joint_angles is not None

    def test_initial_joint_angles(self, franka_kinematics):
        """Test initialization with specific joint angles."""
        initial_joints = torch.tensor([0.5, -0.3, 0.2, -1.0, 0.1, 0.5, 0.3], device="cuda:0")
        robot_mesh = RobotMesh.from_kinematics(
            franka_kinematics, device="cuda:0", initial_joint_angles=initial_joints
        )

        assert torch.allclose(robot_mesh.current_joint_angles, initial_joints)

    def test_zero_init_by_default(self, franka_kinematics):
        """Test that default initialization is zero config."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        assert torch.allclose(
            robot_mesh.current_joint_angles,
            torch.zeros(7, device="cuda:0"),
        )

    def test_mesh_property(self, franka_kinematics):
        """Test that mesh property returns valid wp.Mesh."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        assert isinstance(robot_mesh.mesh, wp.Mesh)
        assert robot_mesh.mesh_id is not None


# ============================================================================
# Test update() for Articulated Robots
# ============================================================================


class TestRobotMeshUpdate:
    """Test vertex updates for articulated robots."""

    def test_update_changes_vertices(self, franka_kinematics):
        """Test that update() changes vertex positions."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        # Store vertices at zero config
        verts_zero = robot_mesh.vertices.clone()

        # Update to different config
        new_joints = torch.tensor([0.5, -0.5, 0.3, -1.5, 0.2, 1.0, 0.5], device="cuda:0")
        robot_mesh.update(new_joints)

        # Vertices should be different
        assert not torch.allclose(robot_mesh.vertices, verts_zero, atol=0.01)

    def test_update_stores_joint_angles(self, franka_kinematics):
        """Test that update() stores current joint angles."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        new_joints = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], device="cuda:0")
        robot_mesh.update(new_joints)

        assert torch.allclose(robot_mesh.current_joint_angles, new_joints)

    def test_update_with_batched_input(self, franka_kinematics):
        """Test update with [1, N] shaped input."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        new_joints = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], device="cuda:0")
        robot_mesh.update(new_joints)

        # Should work without error
        assert robot_mesh.current_joint_angles.shape == (7,)

    def test_multiple_updates(self, franka_kinematics):
        """Test multiple consecutive updates."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        configs = [
            torch.zeros(7, device="cuda:0"),
            torch.ones(7, device="cuda:0") * 0.5,
            torch.randn(7, device="cuda:0") * 0.3,
        ]

        prev_verts = None
        for config in configs:
            robot_mesh.update(config)

            if prev_verts is not None:
                # Each update should change vertices
                assert not torch.allclose(robot_mesh.vertices, prev_verts, atol=0.01)

            prev_verts = robot_mesh.vertices.clone()


# ============================================================================
# Test sample_surface_points()
# ============================================================================


class TestSampleSurfacePoints:
    """Test surface point sampling."""

    def test_basic_sampling_rigid(self, simple_box_mesh):
        """Test basic sampling for rigid mesh."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(100)

        assert points.shape == (100, 3)
        assert normals.shape == (100, 3)

    def test_basic_sampling_articulated(self, franka_kinematics):
        """Test basic sampling for articulated mesh."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(500)

        assert points.shape == (500, 3)
        assert normals.shape == (500, 3)

    def test_normals_are_normalized(self, simple_box_mesh):
        """Test that normals are unit vectors."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(100)

        norms = torch.norm(normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_points_on_device(self, simple_box_mesh):
        """Test that sampled points are on correct device."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(50)

        assert points.device.type == "cuda"
        assert normals.device.type == "cuda"

    def test_different_sample_counts(self, simple_box_mesh):
        """Test sampling different numbers of points."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        for n in [10, 50, 100, 500, 1000]:
            points, normals = robot_mesh.sample_surface_points(n, resample=True)
            assert points.shape == (n, 3)
            assert normals.shape == (n, 3)

    def test_cached_sampling(self, simple_box_mesh):
        """Test that sampling uses cached indices by default."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # First call generates cache
        points1, _ = robot_mesh.sample_surface_points(100)

        # Second call should use same cache (but different vertex positions for articulated)
        # For rigid, cache indices are same but we can't guarantee exact same points
        # due to floating point
        points2, _ = robot_mesh.sample_surface_points(100)

        # Cache should exist
        assert robot_mesh._sample_cache is not None
        assert len(robot_mesh._sample_cache.face_indices) == 100

    def test_resample_flag(self, simple_box_mesh):
        """Test that resample=True generates new samples."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Generate initial samples
        robot_mesh.sample_surface_points(100)
        cache1_indices = robot_mesh._sample_cache.face_indices.clone()

        # Force resample
        robot_mesh.sample_surface_points(100, resample=True)
        cache2_indices = robot_mesh._sample_cache.face_indices.clone()

        # Indices should be different (with high probability)
        # Note: there's a tiny chance they're the same by random chance
        assert not torch.equal(cache1_indices, cache2_indices)

    def test_sampling_after_update(self, franka_kinematics):
        """Test that sampling reflects updated vertex positions."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        # Sample at zero config
        points_zero, _ = robot_mesh.sample_surface_points(500)

        # Update to new config
        robot_mesh.update(torch.tensor([0.5, -0.5, 0.3, -1.5, 0.2, 1.0, 0.5], device="cuda:0"))

        # Sample again (same cache indices, different vertex positions)
        points_new, _ = robot_mesh.sample_surface_points(500)

        # Points should be different
        assert not torch.allclose(points_zero, points_new, atol=0.01)


# ============================================================================
# Test SDF Query Compatibility
# ============================================================================


class TestSDFQueryCompatibility:
    """Test that mesh works correctly with Warp SDF queries."""

    def test_mesh_id_is_valid(self, simple_box_mesh):
        """Test that mesh_id is a valid Warp mesh ID."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # mesh_id should be a valid uint64
        assert robot_mesh.mesh_id is not None
        assert isinstance(robot_mesh.mesh, wp.Mesh)

    def test_mesh_has_correct_topology(self, simple_box_mesh):
        """Test that Warp mesh has correct number of vertices/faces."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Warp mesh should have same topology as input
        assert robot_mesh.mesh is not None
        # The mesh object exists and is valid
        assert robot_mesh.n_vertices == len(simple_box_mesh.vertices)
        assert robot_mesh.n_faces == len(simple_box_mesh.faces)

    def test_mesh_id_constant_after_update(self, franka_kinematics):
        """Test that mesh_id remains constant after update (CUDA graph safe)."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        # Store mesh_id before update
        mesh_id_before = robot_mesh.mesh_id

        # Update mesh
        robot_mesh.update(torch.tensor([0.5, -0.5, 0.3, -1.5, 0.2, 1.0, 0.5], device="cuda:0"))

        # mesh_id should be the same (same mesh object, updated vertices)
        mesh_id_after = robot_mesh.mesh_id

        assert mesh_id_before == mesh_id_after

    def test_mesh_object_constant_after_update(self, franka_kinematics):
        """Test that mesh object reference is constant after update."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        # Store mesh reference before update
        mesh_before = robot_mesh.mesh

        # Update mesh
        robot_mesh.update(torch.tensor([0.5, -0.5, 0.3, -1.5, 0.2, 1.0, 0.5], device="cuda:0"))

        # mesh should be the same object
        mesh_after = robot_mesh.mesh

        assert mesh_before is mesh_after


# ============================================================================
# Test get_trimesh()
# ============================================================================


class TestGetTrimesh:
    """Test conversion back to trimesh."""

    def test_get_trimesh_rigid(self, simple_box_mesh):
        """Test get_trimesh for rigid object."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        tm = robot_mesh.get_trimesh()

        assert isinstance(tm, trimesh.Trimesh)
        assert len(tm.vertices) == robot_mesh.n_vertices
        assert len(tm.faces) == robot_mesh.n_faces

    def test_get_trimesh_articulated(self, franka_kinematics):
        """Test get_trimesh for articulated robot."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        tm = robot_mesh.get_trimesh()

        assert isinstance(tm, trimesh.Trimesh)
        # Note: trimesh may merge duplicate vertices, so we just check it's valid
        assert len(tm.vertices) > 0
        assert len(tm.faces) > 0

    def test_get_trimesh_after_update(self, franka_kinematics):
        """Test that get_trimesh returns updated vertices."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        # Get trimesh at zero config
        tm_zero = robot_mesh.get_trimesh()
        verts_zero = tm_zero.vertices.copy()

        # Update to new config
        robot_mesh.update(torch.tensor([0.5, -0.5, 0.3, -1.5, 0.2, 1.0, 0.5], device="cuda:0"))

        # Get trimesh at new config
        tm_new = robot_mesh.get_trimesh()

        # Vertices should be different
        assert not np.allclose(verts_zero, tm_new.vertices, atol=0.01)


# ============================================================================
# Test Surface Point Validity (Warp Kernel)
# ============================================================================


@wp.kernel
def query_mesh_sdf_kernel(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_dist: float,
    distances: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
):
    """Query SDF at each point and store signed distance."""
    tid = wp.tid()
    p = points[tid]

    result = wp.mesh_query_point(mesh_id, p, max_dist)

    if result.result:
        # Compute signed distance
        closest = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
        delta = p - closest
        dist = wp.length(delta) * result.sign
        distances[tid] = dist
        valid[tid] = 1
    else:
        distances[tid] = max_dist
        valid[tid] = 0


class TestSurfacePointValidity:
    """Test that sampled points actually lie on the mesh surface using Warp kernel."""

    def test_sampled_points_on_surface_rigid(self, simple_box_mesh):
        """Test that sampled points have near-zero SDF for rigid mesh."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        n_samples = 100
        points, _ = robot_mesh.sample_surface_points(n_samples)

        # Query SDF using Warp kernel
        points_wp = wp.from_torch(points, dtype=wp.vec3)
        distances = wp.zeros(n_samples, dtype=wp.float32, device="cuda:0")
        valid = wp.zeros(n_samples, dtype=wp.int32, device="cuda:0")

        wp.launch(
            kernel=query_mesh_sdf_kernel,
            dim=n_samples,
            inputs=[points_wp, robot_mesh.mesh_id, 1.0, distances, valid],
            device="cuda:0",
        )

        distances_torch = wp.to_torch(distances)
        valid_torch = wp.to_torch(valid)

        # All queries should be valid
        assert valid_torch.sum() == n_samples

        # All distances should be very small (points on surface)
        assert torch.abs(distances_torch).max() < 0.001  # Less than 1mm

    def test_sampled_points_on_surface_articulated(self, franka_kinematics):
        """Test that sampled points have near-zero SDF for articulated mesh."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        n_samples = 500
        points, _ = robot_mesh.sample_surface_points(n_samples)

        # Query SDF using Warp kernel
        points_wp = wp.from_torch(points, dtype=wp.vec3)
        distances = wp.zeros(n_samples, dtype=wp.float32, device="cuda:0")
        valid = wp.zeros(n_samples, dtype=wp.int32, device="cuda:0")

        wp.launch(
            kernel=query_mesh_sdf_kernel,
            dim=n_samples,
            inputs=[points_wp, robot_mesh.mesh_id, 1.0, distances, valid],
            device="cuda:0",
        )

        distances_torch = wp.to_torch(distances)
        valid_torch = wp.to_torch(valid)

        # All queries should be valid
        assert valid_torch.sum() == n_samples

        # All distances should be very small (points on surface)
        assert torch.abs(distances_torch).max() < 0.001  # Less than 1mm

    def test_sampled_points_on_surface_after_update(self, franka_kinematics):
        """Test that sampled points remain on surface after update."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        # Update to new configuration
        robot_mesh.update(torch.tensor([0.5, -0.5, 0.3, -1.5, 0.2, 1.0, 0.5], device="cuda:0"))

        n_samples = 500
        points, _ = robot_mesh.sample_surface_points(n_samples)

        # Query SDF using Warp kernel
        points_wp = wp.from_torch(points, dtype=wp.vec3)
        distances = wp.zeros(n_samples, dtype=wp.float32, device="cuda:0")
        valid = wp.zeros(n_samples, dtype=wp.int32, device="cuda:0")

        wp.launch(
            kernel=query_mesh_sdf_kernel,
            dim=n_samples,
            inputs=[points_wp, robot_mesh.mesh_id, 1.0, distances, valid],
            device="cuda:0",
        )

        distances_torch = wp.to_torch(distances)
        valid_torch = wp.to_torch(valid)

        # All queries should be valid
        assert valid_torch.sum() == n_samples

        # All distances should be very small (points on surface)
        assert torch.abs(distances_torch).max() < 0.001  # Less than 1mm


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_samples(self, simple_box_mesh):
        """Test sampling minimal number of points."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        points, normals = robot_mesh.sample_surface_points(1)

        assert points.shape == (1, 3)
        assert normals.shape == (1, 3)

    def test_large_sample_count(self, simple_box_mesh):
        """Test sampling many points."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")

        # Sample more points than vertices
        points, normals = robot_mesh.sample_surface_points(10000)

        assert points.shape == (10000, 3)
        assert normals.shape == (10000, 3)

    def test_extreme_joint_angles(self, franka_kinematics):
        """Test update with extreme joint angles."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")

        # Use approximate Franka joint limits
        extreme_joints = torch.tensor([2.5, 1.5, 2.5, -0.5, 2.5, 3.5, 2.5], device="cuda:0")
        robot_mesh.update(extreme_joints)

        # Should not crash, and mesh should still be valid
        points, normals = robot_mesh.sample_surface_points(100)
        assert points.shape == (100, 3)
        assert not torch.isnan(points).any()
        assert not torch.isnan(normals).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

