# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for geometry types module."""

# Standard Library
import os
import tempfile

# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Material,
    Mesh,
    Obstacle,
    PointCloud,
    SceneCfg,
    Sphere,
    VoxelGrid,
    batch_tensor_cube,
    tensor_capsule,
    tensor_cube,
    tensor_sphere,
)
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg():
    """Create tensor configuration."""
    return DeviceCfg()


class TestMaterial:
    """Test Material class."""

    def test_material_creation(self):
        """Test creating material with default values."""
        mat = Material()
        assert mat.metallic == 0.0
        assert mat.roughness == 0.4

    def test_material_custom_values(self):
        """Test creating material with custom values."""
        mat = Material(metallic=0.5, roughness=0.8)
        assert mat.metallic == 0.5
        assert mat.roughness == 0.8


class TestObstacle:
    """Test Obstacle base class."""

    def test_obstacle_get_trimesh_raises(self):
        """Test that base Obstacle.get_trimesh_mesh raises NotImplementedError."""
        obs = Obstacle(name="test", pose=[0, 0, 0, 1, 0, 0, 0])
        with pytest.raises(NotImplementedError):
            obs.get_trimesh_mesh()

    def test_obstacle_save_as_mesh(self):
        """Test saving obstacle as mesh file."""
        cuboid = Cuboid(name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.5, 0.5, 0.5])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            temp_file = f.name

        try:
            cuboid.save_as_mesh(temp_file)
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_obstacle_save_as_mesh_with_transform(self):
        """Test saving obstacle as mesh file with pose transformation."""
        cuboid = Cuboid(name="test", pose=[1, 2, 3, 1, 0, 0, 0], dims=[0.5, 0.5, 0.5])
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            temp_file = f.name

        try:
            cuboid.save_as_mesh(temp_file, transform_with_pose=True)
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_obstacle_get_sphere(self):
        """Test getting sphere approximation of obstacle."""
        cuboid = Cuboid(name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.6, 0.4, 0.8])
        sphere = cuboid.get_sphere()
        assert sphere is not None
        assert sphere.radius == 0.4  # Minimum dimension


class TestCuboid:
    """Test Cuboid class."""

    def test_cuboid_creation(self):
        """Test creating cuboid with pose."""
        cuboid = Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 2, 3])
        assert cuboid.name == "box"
        assert cuboid.dims == [1, 2, 3]

    def test_cuboid_without_pose_raises(self):
        """Test that creating cuboid without pose raises error."""
        with pytest.raises(ValueError):
            cuboid = Cuboid(name="box", dims=[1, 2, 3])
            # Trigger post_init check
            cuboid.__post_init__()

    def test_cuboid_get_trimesh_with_color(self):
        """Test getting trimesh with color."""
        cuboid = Cuboid(
            name="box",
            pose=[0, 0, 0, 1, 0, 0, 0],
            dims=[1, 1, 1],
            color=[1.0, 0.0, 0.0, 1.0],
        )
        mesh = cuboid.get_trimesh_mesh()
        assert mesh is not None

    def test_cuboid_get_trimesh_with_transform(self):
        """Test getting trimesh with pose transformation."""
        cuboid = Cuboid(name="box", pose=[1, 2, 3, 1, 0, 0, 0], dims=[0.5, 0.5, 0.5])
        mesh = cuboid.get_trimesh_mesh(transform_with_pose=True)
        assert mesh is not None


class TestCapsule:
    """Test Capsule class."""

    def test_capsule_creation(self):
        """Test creating capsule."""
        capsule = Capsule(
            name="cap",
            pose=[0, 0, 0, 1, 0, 0, 0],
            radius=0.1,
            base=[0, 0, 0],
            tip=[0, 0, 1],
        )
        assert capsule.name == "cap"
        assert capsule.radius == 0.1

    def test_capsule_get_trimesh(self):
        """Test getting trimesh from capsule."""
        capsule = Capsule(
            name="cap",
            pose=[0, 0, 0, 1, 0, 0, 0],
            radius=0.1,
            base=[0, 0, 0],
            tip=[0, 0, 1],
            color=[0, 1, 0, 1],
        )
        mesh = capsule.get_trimesh_mesh()
        assert mesh is not None

    def test_capsule_invalid_orientation_raises(self):
        """Test that invalid capsule orientation raises error."""
        capsule = Capsule(
            name="cap",
            pose=[0, 0, 0, 1, 0, 0, 0],
            radius=0.1,
            base=[1, 0, 0],  # Invalid: not at xy origin
            tip=[0, 0, 1],
        )
        with pytest.raises(ValueError):
            capsule.get_trimesh_mesh()


class TestCylinder:
    """Test Cylinder class."""

    def test_cylinder_creation(self):
        """Test creating cylinder."""
        cylinder = Cylinder(name="cyl", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.5, height=1.0)
        assert cylinder.name == "cyl"
        assert cylinder.radius == 0.5
        assert cylinder.height == 1.0

    def test_cylinder_get_trimesh(self):
        """Test getting trimesh from cylinder."""
        cylinder = Cylinder(
            name="cyl",
            pose=[0, 0, 0, 1, 0, 0, 0],
            radius=0.3,
            height=2.0,
            color=[0, 0, 1, 1],
        )
        mesh = cylinder.get_trimesh_mesh()
        assert mesh is not None

    def test_cylinder_get_trimesh_with_transform(self):
        """Test getting trimesh with transformation."""
        cylinder = Cylinder(name="cyl", pose=[1, 0, 0, 1, 0, 0, 0], radius=0.2, height=1.0)
        mesh = cylinder.get_trimesh_mesh(transform_with_pose=True)
        assert mesh is not None


class TestSphere:
    """Test Sphere class."""

    def test_sphere_with_position_deprecated(self):
        """Test sphere with deprecated position parameter."""
        sphere = Sphere(name="sph", position=[1, 2, 3], radius=0.5)
        assert sphere.pose == [1, 2, 3, 1, 0, 0, 0]

    def test_sphere_get_cuboid_override(self):
        """Test that sphere's get_cuboid sets identity orientation."""
        sphere = Sphere(name="sph", pose=[0, 0, 0, 0.707, 0.707, 0, 0], radius=0.5)
        cuboid = sphere.get_cuboid()
        # Orientation should be reset to identity
        assert cuboid.pose[3:] == [1, 0, 0, 0]

    def test_sphere_get_trimesh(self):
        """Test getting trimesh from sphere."""
        sphere = Sphere(name="sph", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.3, color=[1, 1, 0, 1])
        mesh = sphere.get_trimesh_mesh()
        assert mesh is not None


class TestMesh:
    """Test Mesh class."""

    def test_mesh_from_vertices_and_faces(self):
        """Test creating mesh from vertices and faces."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        mesh = Mesh(name="tri", pose=[0, 0, 0, 1, 0, 0, 0], vertices=vertices, faces=faces)
        assert mesh.name == "tri"

    def test_mesh_scale_in_post_init(self):
        """Test that mesh vertices are scaled in post_init."""
        vertices = [[1, 1, 1], [2, 2, 2]]
        faces = [[0, 1, 0]]
        mesh = Mesh(
            name="scaled",
            pose=[0, 0, 0, 1, 0, 0, 0],
            vertices=vertices,
            faces=faces,
            scale=[2.0, 2.0, 2.0],
        )
        # Scale should be applied to vertices
        assert mesh.vertices[0][0] == 2.0

    def test_mesh_get_trimesh_from_vertices(self):
        """Test get_trimesh_mesh from vertices."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        faces = [[0, 1, 2], [0, 1, 3]]
        mesh = Mesh(name="tet", pose=[0, 0, 0, 1, 0, 0, 0], vertices=vertices, faces=faces)
        trimesh_obj = mesh.get_trimesh_mesh()
        assert trimesh_obj is not None

    def test_mesh_get_trimesh_with_transform(self):
        """Test get_trimesh_mesh with pose transformation."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        mesh = Mesh(name="tri", pose=[5, 5, 5, 1, 0, 0, 0], vertices=vertices, faces=faces)
        trimesh_obj = mesh.get_trimesh_mesh(transform_with_pose=True)
        assert trimesh_obj is not None

    def test_mesh_get_mesh_data(self):
        """Test getting mesh data."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        mesh = Mesh(name="tri", pose=[0, 0, 0, 1, 0, 0, 0], vertices=vertices, faces=faces)
        verts, faces_out = mesh.get_mesh_data()
        assert verts is not None
        assert faces_out is not None

    def test_mesh_from_pointcloud(self):
        """Test creating mesh from pointcloud."""
        pointcloud = np.random.rand(100, 3)
        mesh = Mesh.from_pointcloud(pointcloud, pitch=0.05, name="pc_mesh")
        assert mesh.name == "pc_mesh"

    def test_mesh_from_pointcloud_with_filter(self):
        """Test creating mesh from pointcloud with filtering."""
        pointcloud = np.array([[0.01, 0, 0], [0.02, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        mesh = Mesh.from_pointcloud(pointcloud, pitch=0.1, filter_close_points=0.5)
        assert mesh is not None


class TestPointCloud:
    """Test PointCloud class."""

    def test_pointcloud_creation_from_array(self):
        """Test creating pointcloud from numpy array."""
        points = np.random.rand(50, 3)
        pc = PointCloud(name="pc", pose=[0, 0, 0, 1, 0, 0, 0], points=points)
        assert pc.name == "pc"

    def test_pointcloud_creation_from_tensor(self):
        """Test creating pointcloud from torch tensor."""
        points = torch.rand(50, 3)
        pc = PointCloud(name="pc", pose=[0, 0, 0, 1, 0, 0, 0], points=points)
        assert pc.name == "pc"

    def test_pointcloud_with_scale(self):
        """Test pointcloud with scale applied."""
        points = np.ones((10, 3))
        pc = PointCloud(name="pc", pose=[0, 0, 0, 1, 0, 0, 0], points=points, scale=[2, 2, 2])
        # Scale should be applied
        assert pc.points[0, 0] == 2.0

    def test_pointcloud_get_trimesh(self):
        """Test getting trimesh from pointcloud."""
        points = np.random.rand(100, 3) * 2 - 1  # Points in [-1, 1]
        pc = PointCloud(name="pc", pose=[0, 0, 0, 1, 0, 0, 0], points=points)
        mesh = pc.get_trimesh_mesh()
        assert mesh is not None

    def test_pointcloud_get_mesh_data(self):
        """Test getting mesh data from pointcloud."""
        points = np.random.rand(100, 3)
        pc = PointCloud(name="pc", pose=[0, 0, 0, 1, 0, 0, 0], points=points)
        verts, faces = pc.get_mesh_data()
        assert verts is not None
        assert faces is not None


class TestVoxelGrid:
    """Test VoxelGrid class."""

    def test_voxelgrid_creation(self):
        """Test creating voxel grid."""
        voxel = VoxelGrid(
            name="grid", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1.0, 1.0, 1.0], voxel_size=0.05
        )
        assert voxel.name == "grid"
        assert voxel.voxel_size == 0.05

    def test_voxelgrid_get_grid_shape(self):
        """Test getting grid shape."""
        voxel = VoxelGrid(
            name="grid", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1.0, 1.0, 1.0], voxel_size=0.1
        )
        grid_shape, low, high = voxel.get_grid_shape()
        assert len(grid_shape) == 3
        assert len(low) == 3
        assert len(high) == 3

    def test_voxelgrid_create_xyzr_tensor(self, device_cfg):
        """Test creating XYZR tensor."""
        voxel = VoxelGrid(
            name="grid",
            pose=[0, 0, 0, 1, 0, 0, 0],
            dims=[0.5, 0.5, 0.5],
            voxel_size=0.1,
            device_cfg=device_cfg,
        )
        xyzr = voxel.create_xyzr_tensor(device_cfg=device_cfg)
        assert xyzr.shape[-1] == 4

    def test_voxelgrid_create_xyzr_with_transform(self, device_cfg):
        """Test creating XYZR tensor with transformation to origin."""
        voxel = VoxelGrid(
            name="grid",
            pose=[1, 2, 3, 1, 0, 0, 0],
            dims=[0.5, 0.5, 0.5],
            voxel_size=0.1,
            device_cfg=device_cfg,
        )
        xyzr = voxel.create_xyzr_tensor(transform_to_origin=True, device_cfg=device_cfg)
        assert xyzr.shape[-1] == 4

    def test_voxelgrid_clone(self, device_cfg):
        """Test cloning voxel grid."""
        feature_tensor = torch.rand(100, 1, device=device_cfg.device)
        voxel = VoxelGrid(
            name="grid",
            pose=[0, 0, 0, 1, 0, 0, 0],
            dims=[1.0, 1.0, 1.0],
            voxel_size=0.1,
            feature_tensor=feature_tensor,
            device_cfg=device_cfg,
        )
        voxel_clone = voxel.clone()
        assert voxel_clone.name == voxel.name
        assert torch.allclose(voxel_clone.feature_tensor, voxel.feature_tensor)


class TestSceneCfg:
    """Test SceneCfg class."""

    def test_scenecfg_creation_empty(self):
        """Test creating empty scene."""
        scene = SceneCfg()
        assert len(scene) == 0

    def test_scenecfg_create(self):
        """Test creating scene from dictionary."""
        scene_dict = {
            "cuboid": {"box1": {"dims": [1, 1, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}},
            "sphere": {"sph1": {"radius": 0.5, "pose": [1, 0, 0, 1, 0, 0, 0]}},
        }
        scene = SceneCfg.create(scene_dict)
        assert len(scene.cuboid) == 1
        assert len(scene.sphere) == 1

    def test_scenecfg_add_obstacle(self):
        """Test adding obstacle to scene."""
        scene = SceneCfg()
        cuboid = Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
        scene.add_obstacle(cuboid)
        assert len(scene.cuboid) == 1
        assert len(scene) == 1

    def test_scenecfg_get_obstacle(self):
        """Test getting obstacle by name."""
        scene = SceneCfg()
        cuboid = Cuboid(name="box1", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
        scene.add_obstacle(cuboid)
        found = scene.get_obstacle("box1")
        assert found is not None
        assert found.name == "box1"

    def test_scenecfg_get_obstacle_not_found(self):
        """Test getting non-existent obstacle returns None."""
        scene = SceneCfg()
        found = scene.get_obstacle("nonexistent")
        assert found is None

    def test_scenecfg_remove_obstacle(self):
        """Test removing obstacle by name."""
        scene = SceneCfg()
        cuboid = Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
        scene.add_obstacle(cuboid)
        scene.remove_obstacle("box")
        assert len(scene) == 0

    def test_scenecfg_randomize_color(self):
        """Test randomizing colors of obstacles."""
        scene = SceneCfg(
            cuboid=[
                Cuboid(name="box1", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]),
                Cuboid(name="box2", pose=[1, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]),
            ]
        )
        scene.randomize_color(r=[0, 1], g=[0, 1], b=[0, 1])
        assert scene.objects[0].color is not None

    def test_scenecfg_add_color(self):
        """Test adding color to all obstacles."""
        scene = SceneCfg(
            cuboid=[Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])]
        )
        scene.add_color([1.0, 0.0, 0.0, 1.0])
        assert scene.objects[0].color == [1.0, 0.0, 0.0, 1.0]

    def test_scenecfg_add_material(self):
        """Test adding material to all obstacles."""
        scene = SceneCfg(
            cuboid=[Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])]
        )
        mat = Material(metallic=0.8, roughness=0.2)
        scene.add_material(mat)
        assert scene.objects[0].material.metallic == 0.8

    def test_scenecfg_get_obb_world(self):
        """Test converting scene to OBB world."""
        scene = SceneCfg(
            sphere=[Sphere(name="sph", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.5)],
            cuboid=[Cuboid(name="box", pose=[1, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])],
        )
        obb_world = scene.get_obb_world()
        # Sphere should be converted to cuboid
        assert len(obb_world.cuboid) == 2

    def test_scenecfg_get_mesh_world(self):
        """Test converting scene to mesh world."""
        scene = SceneCfg(
            cuboid=[Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])],
            sphere=[Sphere(name="sph", pose=[1, 0, 0, 1, 0, 0, 0], radius=0.5)],
        )
        mesh_world = scene.get_mesh_world()
        assert len(mesh_world.mesh) == 2

    def test_scenecfg_get_collision_check_world(self):
        """Test getting collision check world."""
        scene = SceneCfg(
            cuboid=[Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])],
            cylinder=[Cylinder(name="cyl", pose=[1, 0, 0, 1, 0, 0, 0], radius=0.2, height=1.0)],
        )
        coll_world = scene.get_collision_check_world()
        # Cylinder should be converted to mesh
        assert len(coll_world.mesh) >= 1
        assert len(coll_world.cuboid) == 1

    def test_scenecfg_get_cache_dict(self):
        """Test getting cache dictionary."""
        scene = SceneCfg(
            cuboid=[Cuboid(name="box1", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])],
            mesh=[
                Mesh(
                    name="mesh1",
                    pose=[0, 0, 0, 1, 0, 0, 0],
                    vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    faces=[[0, 1, 2]],
                )
            ],
        )
        cache = scene.get_cache_dict()
        assert cache["obb"] == 1
        assert cache["mesh"] == 1

    def test_scenecfg_clone(self):
        """Test cloning scene configuration."""
        scene = SceneCfg(
            cuboid=[Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])]
        )
        scene_clone = scene.clone()
        assert len(scene_clone.cuboid) == len(scene.cuboid)

    def test_scenecfg_save_scene_as_mesh(self):
        """Test saving scene as mesh file."""
        scene = SceneCfg(
            cuboid=[Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.5, 0.5, 0.5])]
        )
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            temp_file = f.name

        try:
            scene.save_scene_as_mesh(temp_file)
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestHelperFunctions:
    """Test helper tensor functions."""

    def test_tensor_sphere(self, device_cfg):
        """Test tensor_sphere function."""
        pt = [1.0, 2.0, 3.0]
        radius = 0.5
        tensor = tensor_sphere(pt, radius, device_cfg=device_cfg)
        assert tensor.shape == (4,)
        assert tensor[3] == radius

    def test_tensor_sphere_with_existing_tensor(self, device_cfg):
        """Test tensor_sphere with existing tensor."""
        existing = torch.zeros(4, device=device_cfg.device)
        pt = [1.0, 2.0, 3.0]
        tensor = tensor_sphere(pt, 0.5, tensor=existing, device_cfg=device_cfg)
        assert torch.allclose(tensor[:3], torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device))

    def test_tensor_capsule(self, device_cfg):
        """Test tensor_capsule function."""
        base = [0.0, 0.0, 0.0]
        tip = [0.0, 0.0, 1.0]
        radius = 0.1
        tensor = tensor_capsule(base, tip, radius, device_cfg=device_cfg)
        assert tensor.shape == (7,)
        assert tensor[6] == radius

    def test_tensor_cube(self, device_cfg):
        """Test tensor_cube function."""
        pose = [1, 2, 3, 1, 0, 0, 0]
        dims = [0.5, 0.6, 0.7]
        cube_tensors = tensor_cube(pose, dims, device_cfg=device_cfg)
        assert len(cube_tensors) == 2
        assert cube_tensors[0].shape == (3,)  # dims
        assert cube_tensors[1].shape == (1, 7)  # inverse pose with batch dim

    def test_batch_tensor_cube(self, device_cfg):
        """Test batch_tensor_cube function."""
        poses = [[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0]]
        dims_list = [[1, 1, 1], [0.5, 0.5, 0.5]]
        cube_tensors = batch_tensor_cube(poses, dims_list, device_cfg=device_cfg)
        assert len(cube_tensors) == 2
        assert cube_tensors[0].shape[0] == 2  # batch of dims
        assert cube_tensors[1].shape[0] == 2  # batch of poses
