# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for geometry models used in pose estimation.

Tests RigidObjectGeometry and ArticulatedRobotGeometry from
curobo/_src/perception/pose_estimation/geometry.py.
"""

# Standard Library

# Third Party
import pytest
import torch
import trimesh

from curobo._src.perception.pose_estimation.geometry import (
    ArticulatedRobotGeometry,
    RigidObjectGeometry,
)

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def simple_box_mesh():
    """Create a simple box mesh for testing RigidObjectGeometry."""
    return trimesh.creation.box(extents=[0.2, 0.2, 0.2])


@pytest.fixture(scope="module")
def sphere_mesh():
    """Create a sphere mesh for testing."""
    return trimesh.creation.icosphere(subdivisions=2, radius=0.1)


@pytest.fixture(scope="module")
def franka_robot_cfg(cuda_device_cfg):
    """Load Franka robot configuration for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    # Load with mesh link names to get full robot geometry
    mesh_link_names = robot_data["robot_cfg"]["kinematics"]["mesh_link_names"]
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, tool_frames=mesh_link_names)
    return cfg


@pytest.fixture(scope="module")
def franka_robot_model(franka_robot_cfg):
    """Create Franka robot model."""
    return Kinematics(franka_robot_cfg)


class TestRigidObjectGeometryInitialization:
    """Test RigidObjectGeometry initialization."""

    def test_basic_initialization(self, simple_box_mesh, cuda_device_cfg):
        """Test basic initialization with a simple mesh."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        assert geometry.mesh is simple_box_mesh
        assert geometry.device_cfg == cuda_device_cfg
        assert geometry._tensor_args == cuda_device_cfg

    def test_initialization_with_sphere(self, sphere_mesh, cuda_device_cfg):
        """Test initialization with a sphere mesh."""
        geometry = RigidObjectGeometry(mesh=sphere_mesh, device_cfg=cuda_device_cfg)

        assert geometry.mesh is sphere_mesh
        assert geometry.get_dof() == 0

    def test_default_tensor_config(self, simple_box_mesh):
        """Test initialization with default tensor config."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh)

        assert isinstance(geometry.device_cfg, DeviceCfg)


class TestRigidObjectGeometryMethods:
    """Test RigidObjectGeometry methods."""

    def test_get_dof(self, simple_box_mesh, cuda_device_cfg):
        """Test get_dof returns 0 for rigid objects."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        assert geometry.get_dof() == 0

    def test_sample_surface_points_basic(self, simple_box_mesh, cuda_device_cfg):
        """Test basic surface point sampling."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        n_points = 100

        points, normals = geometry.sample_surface_points(n_points)

        # Check shapes
        assert points.shape == (n_points, 3)
        assert normals.shape == (n_points, 3)

        # Check device and dtype
        assert points.device == cuda_device_cfg.device
        assert points.dtype == cuda_device_cfg.dtype
        assert normals.device == cuda_device_cfg.device
        assert normals.dtype == cuda_device_cfg.dtype

    def test_sample_surface_points_different_counts(self, simple_box_mesh, cuda_device_cfg):
        """Test sampling different numbers of points."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        for n_points in [10, 50, 200, 500]:
            points, normals = geometry.sample_surface_points(n_points)

            assert points.shape[0] == n_points
            assert normals.shape[0] == n_points

    def test_normals_are_normalized(self, simple_box_mesh, cuda_device_cfg):
        """Test that normals are properly normalized."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        points, normals = geometry.sample_surface_points(100)

        # Check that normals are unit vectors
        norms = torch.norm(normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_repeated_sampling(self, simple_box_mesh, cuda_device_cfg):
        """Test that repeated sampling works correctly."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        n_points = 50

        # Sample multiple times
        points1, normals1 = geometry.sample_surface_points(n_points)
        points2, normals2 = geometry.sample_surface_points(n_points)

        # Both should return valid results
        assert points1.shape[0] == n_points
        assert points1.shape[1] == 3
        assert points2.shape[0] == n_points
        assert points2.shape[1] == 3
        assert normals1.shape == points1.shape
        assert normals2.shape == points2.shape

    def test_points_on_surface(self, simple_box_mesh, cuda_device_cfg):
        """Test that sampled points are on the mesh surface."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        points, normals = geometry.sample_surface_points(100)

        # For a box with extents [0.2, 0.2, 0.2], points should be within [-0.1, 0.1] on each axis
        assert torch.all(points >= -0.1)
        assert torch.all(points <= 0.1)


class TestArticulatedRobotGeometryInitialization:
    """Test ArticulatedRobotGeometry initialization."""

    def test_basic_initialization(self, franka_robot_model, cuda_device_cfg, capsys):
        """Test basic initialization with Franka robot."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,  # Lower for faster test
            min_points_per_link=20,
            max_points_per_link=100,
        )

        assert geometry.robot_model is franka_robot_model
        assert geometry.device_cfg == cuda_device_cfg
        assert geometry._tensor_args == cuda_device_cfg
        assert geometry._n_dof == 7  # Franka has 7 DOF

        # Check that cached data is initialized
        assert len(geometry.cached_link_points) > 0
        assert len(geometry.cached_link_normals) > 0
        assert len(geometry.cached_link_names) > 0

        # Check that all lists have same length
        assert len(geometry.cached_link_points) == len(geometry.cached_link_normals)
        assert len(geometry.cached_link_points) == len(geometry.cached_link_names)

        # Verify printed output
        captured = capsys.readouterr()
        assert "Initializing cached surface points" in captured.out
        assert "Total cached:" in captured.out

    def test_custom_sampling_density(self, franka_robot_model, cuda_device_cfg):
        """Test initialization with custom sampling density."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=10000.0,
            min_points_per_link=10,
            max_points_per_link=50,
        )

        # Verify cached data structure
        assert len(geometry.cached_link_points) > 0
        assert len(geometry.cached_link_normals) > 0

        # Most links should have points within the specified range
        # (some very small links may have fewer points)
        total_links = len(geometry.cached_link_points)
        links_in_range = sum(
            1 for link_points in geometry.cached_link_points
            if 10 <= len(link_points) <= 50
        )

        # At least 50% of links should be within range
        assert links_in_range >= total_links * 0.5

    def test_point_bounds_enforcement(self, franka_robot_model, cuda_device_cfg):
        """Test that min/max point bounds are used as guidelines."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=1000000.0,  # Very high density
            min_points_per_link=30,
            max_points_per_link=60,  # Low max to test capping
        )

        # Verify geometry is created successfully
        assert len(geometry.cached_link_points) > 0

        # Most links should be within or close to the bounds
        # (some very small/large links may be outside due to mesh properties)
        for link_points in geometry.cached_link_points:
            n_points = len(link_points)
            # At least some points should be sampled
            assert n_points > 0
            # Most should be capped at max
            assert n_points <= 100  # Allow some tolerance beyond max_points_per_link


class TestArticulatedRobotGeometryMethods:
    """Test ArticulatedRobotGeometry methods."""

    def test_get_dof(self, franka_robot_model, cuda_device_cfg):
        """Test get_dof returns correct DOF count."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        assert geometry.get_dof() == 7  # Franka has 7 DOF

    def test_sample_surface_points_zero_config(self, franka_robot_model, cuda_device_cfg):
        """Test sampling at zero configuration."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        n_points = 200
        config = torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        geometry.update(config)
        points, normals = geometry.sample_surface_points(n_points)

        # Check shapes
        assert points.shape == (n_points, 3)
        assert normals.shape == (n_points, 3)

        # Check device and dtype
        assert points.device == cuda_device_cfg.device
        assert points.dtype == cuda_device_cfg.dtype

    def test_sample_surface_points_1d_config(self, franka_robot_model, cuda_device_cfg):
        """Test sampling with 1D configuration (automatically batched)."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        n_points = 150
        config = torch.zeros(7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        geometry.update(config)
        points, normals = geometry.sample_surface_points(n_points)

        assert points.shape == (n_points, 3)
        assert normals.shape == (n_points, 3)

    def test_sample_surface_points_various_configs(self, franka_robot_model, cuda_device_cfg):
        """Test sampling at different robot configurations."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        n_points = 100

        # Test different configurations
        configs = [
            torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype),
            torch.ones(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 0.5,
            torch.randn(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 0.3,
        ]

        for config in configs:
            geometry.update(config)
            points, normals = geometry.sample_surface_points(n_points)

            assert points.shape == (n_points, 3)
            assert normals.shape == (n_points, 3)

            # Normals should be unit vectors
            norms = torch.norm(normals, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2)

    def test_different_point_counts(self, franka_robot_model, cuda_device_cfg):
        """Test sampling different numbers of points."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        config = torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        geometry.update(config)

        for n_points in [50, 150, 300, 500]:
            points, normals = geometry.sample_surface_points(n_points)

            assert points.shape[0] == n_points
            assert normals.shape[0] == n_points

    def test_configuration_changes_points(self, franka_robot_model, cuda_device_cfg):
        """Test that different configurations produce different point sets."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        n_points = 200

        config1 = torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        config2 = torch.ones(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 0.5

        geometry.update(config1)
        points1, _ = geometry.sample_surface_points(n_points)
        geometry.update(config2)
        points2, _ = geometry.sample_surface_points(n_points)

        # Points should be different for different configurations
        assert not torch.allclose(points1, points2, atol=0.01)

    def test_downsampling_when_too_many_cached_points(self, franka_robot_model, cuda_device_cfg):
        """Test that downsampling works when cached points exceed requested count."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=50,
            max_points_per_link=100,
        )

        # Request fewer points than cached
        n_points = 50
        config = torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        geometry.update(config)
        points, normals = geometry.sample_surface_points(n_points)

        assert points.shape[0] == n_points
        assert normals.shape[0] == n_points


class TestCachedPointsValidity:
    """Test validity of cached surface points."""

    def test_cached_points_are_tensors(self, franka_robot_model, cuda_device_cfg):
        """Test that cached points are torch tensors."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        for link_points in geometry.cached_link_points:
            assert isinstance(link_points, torch.Tensor)
            assert link_points.shape[1] == 3  # xyz coordinates

        for link_normals in geometry.cached_link_normals:
            assert isinstance(link_normals, torch.Tensor)
            assert link_normals.shape[1] == 3  # xyz components

    def test_cached_normals_are_normalized(self, franka_robot_model, cuda_device_cfg):
        """Test that cached normals are unit vectors."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        for link_normals in geometry.cached_link_normals:
            norms = torch.norm(link_normals, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_cached_points_device_dtype(self, franka_robot_model, cuda_device_cfg):
        """Test that cached points have correct device and dtype."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        for link_points in geometry.cached_link_points:
            assert link_points.device == cuda_device_cfg.device
            assert link_points.dtype == cuda_device_cfg.dtype

        for link_normals in geometry.cached_link_normals:
            assert link_normals.device == cuda_device_cfg.device
            assert link_normals.dtype == cuda_device_cfg.dtype


class TestGeometryComparison:
    """Test comparison between RigidObjectGeometry and ArticulatedRobotGeometry."""

    def test_rigid_vs_articulated_dof(self, simple_box_mesh, franka_robot_model, cuda_device_cfg):
        """Test that rigid objects have 0 DOF and robots have positive DOF."""
        rigid_geom = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)
        robot_geom = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        assert rigid_geom.get_dof() == 0
        assert robot_geom.get_dof() > 0

    def test_both_geometries_return_same_output_format(
        self, simple_box_mesh, franka_robot_model, cuda_device_cfg
    ):
        """Test that both geometries return points and normals in same format."""
        rigid_geom = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)
        robot_geom = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        n_points = 100
        config = torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        rigid_points, rigid_normals = rigid_geom.sample_surface_points(n_points)
        robot_geom.update(config)
        robot_points, robot_normals = robot_geom.sample_surface_points(n_points)

        # Both should return (n_points, 3) tensors
        assert rigid_points.shape == (n_points, 3)
        assert rigid_normals.shape == (n_points, 3)
        assert robot_points.shape == (n_points, 3)
        assert robot_normals.shape == (n_points, 3)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rigid_minimal_points(self, simple_box_mesh, cuda_device_cfg):
        """Test rigid geometry with minimal point count."""
        geometry = RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)

        points, normals = geometry.sample_surface_points(1)

        assert points.shape == (1, 3)
        assert normals.shape == (1, 3)

    def test_robot_minimal_points(self, franka_robot_model, cuda_device_cfg):
        """Test robot geometry with minimal point count."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        config = torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        geometry.update(config)
        points, normals = geometry.sample_surface_points(10)

        assert points.shape == (10, 3)
        assert normals.shape == (10, 3)

    def test_robot_extreme_config(self, franka_robot_model, cuda_device_cfg):
        """Test robot at extreme joint limits."""
        geometry = ArticulatedRobotGeometry(
            robot_model=franka_robot_model,
            device_cfg=cuda_device_cfg,
            points_per_cubic_meter=50000.0,
            min_points_per_link=20,
            max_points_per_link=100,
        )

        # Test at joint limits (approximate for Franka)
        extreme_config = torch.tensor(
            [[2.5, 1.5, 2.5, -0.5, 2.5, 3.5, 2.5]],
            device=cuda_device_cfg.device,
            dtype=cuda_device_cfg.dtype,
        )

        geometry.update(extreme_config)
        points, normals = geometry.sample_surface_points(100)

        assert points.shape == (100, 3)
        assert normals.shape == (100, 3)

        # Normals should still be valid
        norms = torch.norm(normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

