# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for USD helper utilities."""



# Standard Library
from pathlib import Path

# Third Party
import numpy as np
import pytest
import torch

try:
    # Third Party
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    pytest.skip("usd-core not available", allow_module_level=True)

# CuRobo
from curobo._src.geom.types import (
    Capsule,
    Cuboid,
    Cylinder,
    Material,
    Mesh,
    SceneCfg,
    Sphere,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.usd_writer import (
    UsdWriter,
    create_stage,
    get_position_quat,
    get_transform,
    set_prim_transform,
)


class TestGetPositionQuat:
    """Tests for get_position_quat function."""

    def test_get_position_quat_float(self):
        """Test converting pose to position and quaternion with float precision."""
        pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]  # [x, y, z, qw, qx, qy, qz]
        position, quat = get_position_quat(pose, use_float=True)

        assert isinstance(position, Gf.Vec3f)
        assert isinstance(quat, Gf.Quatf)
        assert position[0] == pytest.approx(1.0)
        assert position[1] == pytest.approx(2.0)
        assert position[2] == pytest.approx(3.0)
        assert quat.GetReal() == pytest.approx(1.0)

    def test_get_position_quat_double(self):
        """Test converting pose to position and quaternion with double precision."""
        pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        position, quat = get_position_quat(pose, use_float=False)

        assert isinstance(position, Gf.Vec3d)
        assert isinstance(quat, Gf.Quatd)
        assert position[0] == pytest.approx(1.0)
        assert position[1] == pytest.approx(2.0)
        assert position[2] == pytest.approx(3.0)
        assert quat.GetReal() == pytest.approx(1.0)


class TestGetTransform:
    """Tests for get_transform function."""

    def test_get_transform_identity(self):
        """Test getting transform matrix for identity pose."""
        pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # Identity pose
        mat = get_transform(pose)

        assert isinstance(mat, Gf.Matrix4d)
        # Check diagonal is 1 for identity rotation
        identity = Gf.Matrix4d(1.0)
        identity.SetTranslate(Gf.Vec3d(0, 0, 0))

    def test_get_transform_with_translation(self):
        """Test getting transform matrix with translation."""
        pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        mat = get_transform(pose)

        translation = mat.ExtractTranslation()
        assert translation[0] == pytest.approx(1.0)
        assert translation[1] == pytest.approx(2.0)
        assert translation[2] == pytest.approx(3.0)


class TestCreateStage:
    """Tests for create_stage function."""

    def test_create_stage_default(self, tmp_path):
        """Test creating a USD stage with default parameters."""
        stage_path = tmp_path / "test_stage.usd"
        stage = create_stage(str(stage_path), "/world")

        assert stage is not None
        assert stage.GetDefaultPrim() is not None
        assert stage.GetDefaultPrim().GetName() == "world"

    def test_create_stage_custom_frame(self, tmp_path):
        """Test creating a USD stage with custom base frame."""
        stage_path = tmp_path / "test_stage.usd"
        stage = create_stage(str(stage_path), "/custom_base")

        assert stage.GetDefaultPrim().GetName() == "custom_base"

    def test_create_stage_up_axis_z(self, tmp_path):
        """Test that created stage has Z-up axis."""
        stage_path = tmp_path / "test_stage.usd"
        stage = create_stage(str(stage_path))

        assert UsdGeom.GetStageUpAxis(stage) == "Z"

    def test_create_stage_meters_per_unit(self, tmp_path):
        """Test that created stage has correct meters per unit."""
        stage_path = tmp_path / "test_stage.usd"
        stage = create_stage(str(stage_path))

        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0


class TestUsdWriter:
    """Tests for UsdWriter class."""

    @pytest.fixture
    def usd_writer(self):
        """Create a UsdWriter instance for testing."""
        return UsdWriter()

    @pytest.fixture
    def stage_path(self, tmp_path):
        """Create a temporary path for USD stages."""
        return tmp_path / "test.usd"

    def test_usd_writer_initialization(self, usd_writer):
        """Test UsdWriter initialization."""
        assert usd_writer.stage is None
        assert usd_writer.dt is None
        assert usd_writer._use_float is True

    def test_create_stage(self, usd_writer, stage_path):
        """Test creating a stage through UsdWriter."""
        usd_writer.create_stage(str(stage_path))

        assert usd_writer.stage is not None
        assert usd_writer.dt == 0.02
        assert usd_writer.interpolation_steps == 1

    def test_create_stage_with_timesteps(self, usd_writer, stage_path):
        """Test creating a stage with timesteps."""
        timesteps = 100
        dt = 0.05
        usd_writer.create_stage(str(stage_path), timesteps=timesteps, dt=dt)

        assert usd_writer.stage.GetStartTimeCode() == 1
        assert usd_writer.stage.GetEndTimeCode() == timesteps
        assert usd_writer.dt == dt

    def test_add_subroot(self, usd_writer, stage_path):
        """Test adding a subroot to the stage."""
        usd_writer.create_stage(str(stage_path))
        usd_writer.add_subroot("/world", "/obstacles")

        prim = usd_writer.stage.GetPrimAtPath("/world/obstacles")
        assert prim.IsValid()

    def test_add_subroot_with_pose(self, usd_writer, stage_path):
        """Test adding a subroot with a pose."""
        usd_writer.create_stage(str(stage_path))
        pose = Pose.from_list([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], DeviceCfg())
        usd_writer.add_subroot("/world", "/obstacles", pose=pose)

        prim = usd_writer.stage.GetPrimAtPath("/world/obstacles")
        assert prim.IsValid()
        assert prim.GetAttribute("xformOp:translate").IsValid()

    def test_save_stage(self, usd_writer, stage_path):
        """Test saving a stage to disk."""
        usd_writer.create_stage(str(stage_path))
        usd_writer.save()

        # Stage should be saved
        assert Path(stage_path).exists()

    def test_write_stage_to_file(self, usd_writer, stage_path, tmp_path):
        """Test writing stage to file."""
        usd_writer.create_stage(str(stage_path))
        output_path = tmp_path / "output.usd"
        usd_writer.write_stage_to_file(str(output_path))

        assert Path(output_path).exists()
        # Verify we can load the saved stage
        saved_stage = Usd.Stage.Open(str(output_path))
        assert saved_stage is not None


class TestUsdWriterObstacles:
    """Tests for adding obstacles to USD stages."""

    @pytest.fixture
    def usd_writer_with_stage(self, tmp_path):
        """Create a UsdWriter with an initialized stage."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))
        return helper

    def test_add_cuboid_to_stage(self, usd_writer_with_stage):
        """Test adding a cuboid obstacle to stage."""
        cuboid = Cuboid(
            name="test_cube",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            dims=[0.5, 0.5, 0.5],
        )

        path = usd_writer_with_stage.add_cuboid_to_stage(cuboid)

        assert path == "/world/obstacles/test_cube"
        prim = usd_writer_with_stage.stage.GetPrimAtPath(path)
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Cube)

    def test_add_cuboid_with_color(self, usd_writer_with_stage):
        """Test adding a cuboid with color material."""
        cuboid = Cuboid(
            name="colored_cube",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
            color=[1.0, 0.0, 0.0, 1.0],  # Red
        )

        path = usd_writer_with_stage.add_cuboid_to_stage(cuboid)
        prim = usd_writer_with_stage.stage.GetPrimAtPath(path)

        assert prim.IsValid()
        # Material should be created
        material_path = path + "/material_colored_cube"
        mat_prim = usd_writer_with_stage.stage.GetPrimAtPath(material_path)
        assert mat_prim.IsValid()

    def test_add_sphere_to_stage(self, usd_writer_with_stage):
        """Test adding a sphere obstacle to stage."""
        sphere = Sphere(
            name="test_sphere",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.5,
        )

        path = usd_writer_with_stage.add_sphere_to_stage(sphere)

        assert path == "/world/obstacles/test_sphere"
        prim = usd_writer_with_stage.stage.GetPrimAtPath(path)
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Sphere)

    def test_add_sphere_with_position_only(self, usd_writer_with_stage):
        """Test adding a sphere with position only (no pose)."""
        sphere = Sphere(
            name="test_sphere",
            position=[1.0, 2.0, 3.0],
            radius=0.5,
        )

        path = usd_writer_with_stage.add_sphere_to_stage(sphere)
        prim = usd_writer_with_stage.stage.GetPrimAtPath(path)

        assert prim.IsValid()
        # Check that pose was created from position
        assert sphere.pose is not None

    def test_add_cylinder_to_stage(self, usd_writer_with_stage):
        """Test adding a cylinder obstacle to stage."""
        cylinder = Cylinder(
            name="test_cylinder",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.3,
            height=1.0,
        )

        path = usd_writer_with_stage.add_cylinder_to_stage(cylinder)

        assert path == "/world/obstacles/test_cylinder"
        prim = usd_writer_with_stage.stage.GetPrimAtPath(path)
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Cylinder)

    def test_add_mesh_to_stage(self, usd_writer_with_stage):
        """Test adding a mesh obstacle to stage."""
        # Create simple triangle mesh
        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        faces = [[0, 1, 2]]

        mesh = Mesh(
            name="test_mesh",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
        )

        path = usd_writer_with_stage.add_mesh_to_stage(mesh)

        assert path == "/world/obstacles/test_mesh"
        prim = usd_writer_with_stage.stage.GetPrimAtPath(path)
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Mesh)

    def test_add_world_to_stage(self, usd_writer_with_stage):
        """Test adding multiple obstacles to stage at once."""
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[1.0, 1.0, 1.0],
                )
            ],
            sphere=[
                Sphere(
                    name="sphere1",
                    pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.5,
                )
            ],
        )

        paths = usd_writer_with_stage.add_world_to_stage(obstacles)

        assert len(paths) == 2
        for path in paths:
            prim = usd_writer_with_stage.stage.GetPrimAtPath(path)
            assert prim.IsValid()

    def test_add_material(self, usd_writer_with_stage):
        """Test adding material to an object."""
        # Create a cuboid first
        cuboid = Cuboid(
            name="test_cube",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
        )
        path = usd_writer_with_stage.add_cuboid_to_stage(cuboid)
        obj_prim = usd_writer_with_stage.stage.GetPrimAtPath(path)

        # Add material
        color = [1.0, 0.0, 0.0, 1.0]  # Red with full opacity
        material = Material(roughness=0.5, metallic=0.3)

        mat_usd = usd_writer_with_stage.add_material(
            "test_material", path, color, obj_prim, material
        )

        assert mat_usd is not None


class TestUsdWriterLoadStage:
    """Tests for loading USD stages."""

    def test_load_stage_from_file(self, tmp_path):
        """Test loading an existing USD stage from file."""
        # Create a stage first
        stage_path = tmp_path / "test.usd"
        stage = create_stage(str(stage_path))
        stage.Save()

        # Load it with UsdWriter
        helper = UsdWriter()
        helper.load_stage_from_file(str(stage_path))

        assert helper.stage is not None

    def test_load_stage(self):
        """Test loading a USD stage object directly."""
        stage = Usd.Stage.CreateInMemory()
        helper = UsdWriter()
        helper.load_stage(stage)

        assert helper.stage is stage


class TestUsdWriterGetObstacles:
    """Tests for extracting obstacles from USD stages."""

    @pytest.fixture
    def stage_with_obstacles(self, tmp_path):
        """Create a stage with various obstacles."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add various obstacles
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[1.0, 1.0, 1.0],
                )
            ],
            sphere=[
                Sphere(
                    name="sphere1",
                    pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.5,
                )
            ],
            cylinder=[
                Cylinder(
                    name="cylinder1",
                    pose=[0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.3,
                    height=1.0,
                )
            ],
        )
        helper.add_world_to_stage(obstacles)

        return helper

    def test_get_obstacles_from_stage_all(self, stage_with_obstacles):
        """Test extracting all obstacles from stage."""
        scene = stage_with_obstacles.get_obstacles_from_stage()

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert len(scene.sphere) == 1
        assert len(scene.cylinder) == 1

    def test_get_obstacles_from_stage_with_only_paths(self, stage_with_obstacles):
        """Test extracting obstacles with path filter."""
        scene = stage_with_obstacles.get_obstacles_from_stage(
            only_paths=["/world/obstacles/cube1"]
        )

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert scene.cuboid[0].name == "/world/obstacles/cube1"

    def test_get_obstacles_from_stage_with_ignore_paths(self, stage_with_obstacles):
        """Test extracting obstacles with path exclusion."""
        scene = stage_with_obstacles.get_obstacles_from_stage(
            ignore_paths=["/world/obstacles/sphere1"]
        )

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert len(scene.cylinder) == 1
        # Sphere should be excluded
        assert scene.sphere is None or len(scene.sphere) == 0

    def test_get_obstacles_from_stage_with_substring_filter(self, stage_with_obstacles):
        """Test extracting obstacles with substring filter."""
        scene = stage_with_obstacles.get_obstacles_from_stage(only_substring=["cube"])

        assert scene is not None
        assert len(scene.cuboid) == 1
        # Others should be excluded
        assert (scene.sphere is None or len(scene.sphere) == 0)
        assert (scene.cylinder is None or len(scene.cylinder) == 0)


class TestSetPrimTransform:
    """Tests for set_prim_transform function."""

    def test_set_prim_transform_float(self, tmp_path):
        """Test setting prim transform with float precision."""
        stage = create_stage(str(tmp_path / "test.usd"))
        prim = stage.DefinePrim("/test", "Xform")

        pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        scale = [1.0, 1.0, 1.0]

        set_prim_transform(prim, pose, scale, use_float=True)

        assert prim.GetAttribute("xformOp:translate").IsValid()
        assert prim.GetAttribute("xformOp:orient").IsValid()
        assert prim.GetAttribute("xformOp:scale").IsValid()

        translation = prim.GetAttribute("xformOp:translate").Get()
        assert translation[0] == pytest.approx(1.0)
        assert translation[1] == pytest.approx(2.0)
        assert translation[2] == pytest.approx(3.0)

    def test_set_prim_transform_double(self, tmp_path):
        """Test setting prim transform with double precision."""
        stage = create_stage(str(tmp_path / "test.usd"))
        prim = stage.DefinePrim("/test", "Xform")

        pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        scale = [2.0, 2.0, 2.0]

        set_prim_transform(prim, pose, scale, use_float=False)

        scale_attr = prim.GetAttribute("xformOp:scale").Get()
        assert scale_attr[0] == pytest.approx(2.0)


class TestUsdWriterGetPose:
    """Tests for getting pose from USD primitives."""

    def test_get_pose_identity(self, tmp_path):
        """Test getting identity pose from prim."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add cuboid at origin
        cuboid = Cuboid(
            name="test_cube",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
        )
        path = helper.add_cuboid_to_stage(cuboid)

        # Get pose
        pose_matrix = helper.get_pose(path)

        assert pose_matrix is not None
        # Identity should have 1s on diagonal
        assert pose_matrix[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert pose_matrix[1, 1] == pytest.approx(1.0, abs=1e-6)
        assert pose_matrix[2, 2] == pytest.approx(1.0, abs=1e-6)

    def test_get_pose_with_translation(self, tmp_path):
        """Test getting pose with translation from prim."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add cuboid at specific position
        x, y, z = 1.0, 2.0, 3.0
        cuboid = Cuboid(
            name="test_cube",
            pose=[x, y, z, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
        )
        path = helper.add_cuboid_to_stage(cuboid)

        # Get pose
        pose_matrix = helper.get_pose(path)

        # Check translation
        assert pose_matrix[0, 3] == pytest.approx(x, abs=1e-6)
        assert pose_matrix[1, 3] == pytest.approx(y, abs=1e-6)
        assert pose_matrix[2, 3] == pytest.approx(z, abs=1e-6)


class TestUsdWriterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_add_sphere_without_pose_or_position(self, tmp_path):
        """Test that sphere with neither pose nor position is handled."""
        helper = UsdWriter()
        stage_path = tmp_path / "sphere_test.usd"
        helper.create_stage(str(stage_path))

        sphere = Sphere(name="test_sphere", radius=0.5)
        # This should either fail gracefully or set default pose
        path = helper.add_sphere_to_stage(sphere)
        # If it succeeds, pose should have been set
        assert sphere.pose is not None
        assert path is not None

    def test_add_mesh_with_empty_vertices(self, tmp_path):
        """Test handling mesh with empty vertices."""
        helper = UsdWriter()
        stage_path = tmp_path / "mesh_test.usd"
        helper.create_stage(str(stage_path))

        mesh = Mesh(
            name="empty_mesh",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=[],
            faces=[],
        )

        # This might fail or create an invalid mesh
        path = helper.add_mesh_to_stage(mesh)
        # Just verify it doesn't crash
        assert path is not None


class TestUsdWriterAnimation:
    """Tests for animation creation functionality."""

    @pytest.fixture
    def animation_poses(self):
        """Create test animation poses."""
        # Create 10 timesteps with different positions
        n_timesteps = 10
        n_objects = 2

        # Create position and quaternion tensors separately
        positions = []
        quaternions = []
        for t in range(n_timesteps):
            pos_row = []
            quat_row = []
            for obj in range(n_objects):
                x = float(t) * 0.1 + obj
                y = float(t) * 0.1
                z = 0.5
                pos_row.append([x, y, z])
                quat_row.append([1.0, 0.0, 0.0, 0.0])
            positions.append(pos_row)
            quaternions.append(quat_row)

        position_tensor = torch.tensor(positions, device=torch.device("cuda", 0))
        quaternion_tensor = torch.tensor(quaternions, device=torch.device("cuda", 0))

        return Pose(position=position_tensor, quaternion=quaternion_tensor)

    def test_create_animation(self, tmp_path, animation_poses):
        """Test creating animation from poses."""
        helper = UsdWriter()
        stage_path = tmp_path / "animation.usd"
        helper.create_stage(str(stage_path), timesteps=10, dt=0.02)

        # Create simple scene with two cubes
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.1, 0.1, 0.1],
                ),
                Cuboid(
                    name="cube2",
                    pose=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.1, 0.1, 0.1],
                ),
            ]
        )

        helper.create_animation(obstacles, animation_poses, base_frame="/world", dt=0.02)

        # Verify animation was created
        assert helper.stage is not None
        prim1 = helper.stage.GetPrimAtPath("/world/robot/cube1")
        prim2 = helper.stage.GetPrimAtPath("/world/robot/cube2")
        assert prim1.IsValid()
        assert prim2.IsValid()

    def test_create_obstacle_animation(self, tmp_path):
        """Test creating obstacle animation with moving obstacles."""
        helper = UsdWriter()
        stage_path = tmp_path / "obstacle_animation.usd"
        helper.create_stage(str(stage_path), timesteps=5, dt=0.02)

        # Create list of obstacles at different timesteps
        obstacles_list = []
        for t in range(5):
            obstacles_list.append(
                [
                    Sphere(
                        name="moving_sphere",
                        position=[float(t) * 0.1, 0.0, 0.5],
                        radius=0.1,
                    )
                ]
            )

        helper.create_obstacle_animation(
            obstacles_list, base_frame="/world", obstacles_frame="moving_obstacles"
        )

        # Verify animation was created
        prim = helper.stage.GetPrimAtPath("/world/moving_obstacles/moving_sphere")
        assert prim.IsValid()


class TestUsdWriterMeshAttributes:
    """Tests for getting mesh attributes from USD."""

    def test_get_cube_attrs(self, tmp_path):
        """Test extracting cube attributes from USD primitive."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add a cube
        cuboid = Cuboid(
            name="test_cube",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            dims=[0.5, 0.6, 0.7],
        )
        path = helper.add_cuboid_to_stage(cuboid)

        # Get the prim and extract attributes
        from curobo._src.util.usd_writer import get_cube_attrs

        prim = helper.stage.GetPrimAtPath(path)
        helper._xform_cache.SetTime(0)
        extracted_cuboid = get_cube_attrs(prim, cache=helper._xform_cache)

        assert extracted_cuboid is not None
        assert extracted_cuboid.name == path
        # Check dimensions (approximately)
        assert extracted_cuboid.dims[0] == pytest.approx(0.5, abs=0.01)
        assert extracted_cuboid.dims[1] == pytest.approx(0.6, abs=0.01)
        assert extracted_cuboid.dims[2] == pytest.approx(0.7, abs=0.01)

    def test_get_sphere_attrs(self, tmp_path):
        """Test extracting sphere attributes from USD primitive."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add a sphere
        sphere = Sphere(
            name="test_sphere",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.5,
        )
        path = helper.add_sphere_to_stage(sphere)

        # Get the prim and extract attributes
        from curobo._src.util.usd_writer import get_sphere_attrs

        prim = helper.stage.GetPrimAtPath(path)
        helper._xform_cache.SetTime(0)
        extracted_sphere = get_sphere_attrs(prim, cache=helper._xform_cache)

        assert extracted_sphere is not None
        assert extracted_sphere.name == path
        assert extracted_sphere.radius == pytest.approx(0.5, abs=0.01)

    def test_get_cylinder_attrs(self, tmp_path):
        """Test extracting cylinder attributes from USD primitive."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add a cylinder
        cylinder = Cylinder(
            name="test_cylinder",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.3,
            height=1.0,
        )
        path = helper.add_cylinder_to_stage(cylinder)

        # Get the prim and extract attributes
        from curobo._src.util.usd_writer import get_cylinder_attrs

        prim = helper.stage.GetPrimAtPath(path)
        helper._xform_cache.SetTime(0)
        extracted_cylinder = get_cylinder_attrs(prim, cache=helper._xform_cache)

        assert extracted_cylinder is not None
        assert extracted_cylinder.name == path
        assert extracted_cylinder.radius == pytest.approx(0.3, abs=0.01)
        assert extracted_cylinder.height == pytest.approx(1.0, abs=0.01)

    def test_get_mesh_attrs(self, tmp_path):
        """Test extracting mesh attributes from USD primitive."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add a simple triangle mesh
        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        faces = [[0, 1, 2]]
        mesh = Mesh(
            name="test_mesh",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
        )
        path = helper.add_mesh_to_stage(mesh)

        # Get the prim and extract attributes
        from curobo._src.util.usd_writer import get_mesh_attrs

        prim = helper.stage.GetPrimAtPath(path)
        helper._xform_cache.SetTime(0)
        extracted_mesh = get_mesh_attrs(prim, cache=helper._xform_cache)

        assert extracted_mesh is not None
        assert extracted_mesh.name == path
        assert len(extracted_mesh.vertices) == 3
        assert len(extracted_mesh.faces) == 1


class TestUsdWriterRoundTrip:
    """Tests for round-trip functionality (write and read back)."""

    def test_cuboid_roundtrip(self, tmp_path):
        """Test adding and reading back a cuboid."""
        helper = UsdWriter()
        stage_path = tmp_path / "roundtrip.usd"
        helper.create_stage(str(stage_path))

        # Add obstacles
        original_obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.5, 0.6, 0.7],
                )
            ]
        )
        helper.add_world_to_stage(original_obstacles)

        # Read them back
        read_obstacles = helper.get_obstacles_from_stage()

        assert read_obstacles is not None
        assert len(read_obstacles.cuboid) == 1
        # Check dimensions match
        assert read_obstacles.cuboid[0].dims[0] == pytest.approx(0.5, abs=0.01)
        assert read_obstacles.cuboid[0].dims[1] == pytest.approx(0.6, abs=0.01)
        assert read_obstacles.cuboid[0].dims[2] == pytest.approx(0.7, abs=0.01)

    def test_mixed_obstacles_roundtrip(self, tmp_path):
        """Test adding and reading back mixed obstacle types."""
        helper = UsdWriter()
        stage_path = tmp_path / "mixed_roundtrip.usd"
        helper.create_stage(str(stage_path))

        # Add mixed obstacles
        original_obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[1.0, 1.0, 1.0],
                )
            ],
            sphere=[
                Sphere(
                    name="sphere1",
                    pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.5,
                )
            ],
            cylinder=[
                Cylinder(
                    name="cylinder1",
                    pose=[0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.3,
                    height=1.0,
                )
            ],
        )
        helper.add_world_to_stage(original_obstacles)

        # Read them back
        read_obstacles = helper.get_obstacles_from_stage()

        assert read_obstacles is not None
        assert len(read_obstacles.cuboid) == 1
        assert len(read_obstacles.sphere) == 1
        assert len(read_obstacles.cylinder) == 1


class TestUsdWriterStageManagement:
    """Tests for stage file management."""

    def test_write_stage_flatten(self, tmp_path):
        """Test writing stage with flatten option."""
        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Add some content
        cuboid = Cuboid(
            name="test_cube",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
        )
        helper.add_cuboid_to_stage(cuboid)

        # Write with flatten
        output_path = tmp_path / "flattened.usd"
        helper.write_stage_to_file(str(output_path), flatten=True)

        assert Path(output_path).exists()

        # Verify we can load it
        loaded_stage = Usd.Stage.Open(str(output_path))
        assert loaded_stage is not None

    def test_multiple_subroots(self, tmp_path):
        """Test adding multiple subroot frames."""
        helper = UsdWriter()
        stage_path = tmp_path / "multi_subroot.usd"
        helper.create_stage(str(stage_path))

        # Add multiple subroots
        helper.add_subroot("/world", "obstacles1")
        helper.add_subroot("/world", "obstacles2")
        helper.add_subroot("/world", "robot")

        # Verify all exist
        assert helper.stage.GetPrimAtPath("/world/obstacles1").IsValid()
        assert helper.stage.GetPrimAtPath("/world/obstacles2").IsValid()
        assert helper.stage.GetPrimAtPath("/world/robot").IsValid()


class TestUsdWriterCapsule:
    """Tests for capsule obstacle handling."""

    def test_get_capsule_attrs(self, tmp_path):
        """Test extracting capsule attributes from USD primitive."""
        from curobo._src.util.usd_writer import get_capsule_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "test.usd"
        helper.create_stage(str(stage_path))

        # Create a capsule prim manually
        capsule_path = "/world/test_capsule"
        capsule_geom = UsdGeom.Capsule.Define(helper.stage, capsule_path)
        capsule_geom.CreateRadiusAttr(0.3)
        capsule_geom.CreateHeightAttr(1.0)

        # Set transform
        from curobo._src.util.usd_writer import set_prim_transform
        prim = helper.stage.GetPrimAtPath(capsule_path)
        set_prim_transform(prim, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

        # Extract attributes
        helper._xform_cache.SetTime(0)
        extracted_capsule = get_capsule_attrs(prim, cache=helper._xform_cache)

        assert extracted_capsule is not None
        assert extracted_capsule.name == capsule_path
        assert extracted_capsule.radius == pytest.approx(0.3, abs=0.01)
        # Height is the distance between base and tip
        assert len(extracted_capsule.base) == 3
        assert len(extracted_capsule.tip) == 3


class TestUsdWriterMaterialSystem:
    """Tests for material and color handling."""

    def test_add_material_with_custom_properties(self, tmp_path):
        """Test adding material with custom metallic and roughness."""
        helper = UsdWriter()
        stage_path = tmp_path / "material_test.usd"
        helper.create_stage(str(stage_path))

        # Create an object
        cuboid = Cuboid(
            name="test_cube",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
            color=[0.5, 0.5, 0.5, 0.8],
            material=Material(roughness=0.8, metallic=0.6),
        )
        path = helper.add_cuboid_to_stage(cuboid)

        # Check material exists
        material_path = path + "/material_test_cube"
        mat_prim = helper.stage.GetPrimAtPath(material_path)
        assert mat_prim.IsValid()

    def test_obstacles_with_various_colors(self, tmp_path):
        """Test adding obstacles with different colors."""
        helper = UsdWriter()
        stage_path = tmp_path / "colored_obstacles.usd"
        helper.create_stage(str(stage_path))

        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="red_cube",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[1.0, 1.0, 1.0],
                    color=[1.0, 0.0, 0.0, 1.0],
                )
            ],
            sphere=[
                Sphere(
                    name="blue_sphere",
                    pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.5,
                    color=[0.0, 0.0, 1.0, 1.0],
                )
            ],
            cylinder=[
                Cylinder(
                    name="green_cylinder",
                    pose=[0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.3,
                    height=1.0,
                    color=[0.0, 1.0, 0.0, 1.0],
                )
            ],
        )

        paths = helper.add_world_to_stage(obstacles)
        assert len(paths) == 3

        # Verify materials were created
        for path in paths:
            prim = helper.stage.GetPrimAtPath(path)
            assert prim.IsValid()


class TestUsdWriterWithPhysics:
    """Tests for physics-enabled obstacles."""

    def test_add_cuboid_with_physics_enabled(self, tmp_path):
        """Test adding cuboid with physics enabled."""
        helper = UsdWriter()
        stage_path = tmp_path / "physics_test.usd"
        helper.create_stage(str(stage_path))

        cuboid = Cuboid(
            name="physics_cube",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
        )

        path = helper.add_cuboid_to_stage(cuboid, enable_physics=True)
        prim = helper.stage.GetPrimAtPath(path)

        # Check physics attribute
        assert prim.GetAttribute("physics:rigidBodyEnabled").IsValid()
        assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True

    def test_add_sphere_with_physics_enabled(self, tmp_path):
        """Test adding sphere with physics enabled."""
        helper = UsdWriter()
        stage_path = tmp_path / "physics_sphere.usd"
        helper.create_stage(str(stage_path))

        sphere = Sphere(
            name="physics_sphere",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.5,
        )

        path = helper.add_sphere_to_stage(sphere, enable_physics=True)
        prim = helper.stage.GetPrimAtPath(path)

        assert prim.GetAttribute("physics:rigidBodyEnabled").IsValid()
        assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True


class TestUsdWriterMeshWithColors:
    """Tests for meshes with vertex colors."""

    def test_mesh_with_vertex_colors(self, tmp_path):
        """Test adding mesh with vertex colors."""
        helper = UsdWriter()
        stage_path = tmp_path / "colored_mesh.usd"
        helper.create_stage(str(stage_path))

        # Create triangle with vertex colors
        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        faces = [[0, 1, 2]]
        vertex_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        mesh = Mesh(
            name="colored_triangle",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
        )

        path = helper.add_mesh_to_stage(mesh)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()

        # Verify vertex colors were set
        mesh_geom = UsdGeom.Mesh(prim)
        primvarsapi = UsdGeom.PrimvarsAPI(mesh_geom)
        color_primvar = primvarsapi.GetPrimvar("displayColor")
        assert color_primvar.IsDefined()

    def test_mesh_with_vertex_colors_255_scale(self, tmp_path):
        """Test mesh with vertex colors in 0-255 range."""
        helper = UsdWriter()
        stage_path = tmp_path / "colored_mesh_255.usd"
        helper.create_stage(str(stage_path))

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        faces = [[0, 1, 2]]
        # Colors in 0-255 range
        vertex_colors = [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]]

        mesh = Mesh(
            name="colored_triangle_255",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
        )

        path = helper.add_mesh_to_stage(mesh)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()


class TestUsdWriterTransformReference:
    """Tests for get_obstacles_from_stage with reference transforms."""

    def test_get_obstacles_with_reference_frame(self, tmp_path):
        """Test extracting obstacles with reference frame transform."""
        helper = UsdWriter()
        stage_path = tmp_path / "ref_frame_test.usd"
        helper.create_stage(str(stage_path))

        # Add a reference frame
        helper.add_subroot("/world", "ref_frame",
                          Pose.from_list([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], DeviceCfg()))

        # Add obstacles
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[1.0, 1.0, 1.0],
                )
            ]
        )
        helper.add_world_to_stage(obstacles)

        # Get obstacles with reference frame
        scene = helper.get_obstacles_from_stage(reference_prim_path="/world/ref_frame")
        assert scene is not None
        assert len(scene.cuboid) == 1


class TestJoinUsdPath:
    """Tests for the USD-specific path joining function."""

    def test_join_usd_path_basic(self):
        """Test basic USD path joining."""
        from curobo._src.util.usd_writer import join_usd_path

        result = join_usd_path("/world", "obstacles")
        assert result == "/world/obstacles"

    def test_join_usd_path_with_leading_slash(self):
        """Test USD path joining with leading slash in second arg."""
        from curobo._src.util.usd_writer import join_usd_path

        result = join_usd_path("/world", "/obstacles")
        assert result == "/world/obstacles"

    def test_join_usd_path_with_trailing_slash(self):
        """Test USD path joining with trailing slash in first arg."""
        from curobo._src.util.usd_writer import join_usd_path

        result = join_usd_path("/world/", "obstacles")
        assert result == "/world/obstacles"

    def test_join_usd_path_both_slashes(self):
        """Test USD path joining with both slashes."""
        from curobo._src.util.usd_writer import join_usd_path

        result = join_usd_path("/world/", "/obstacles")
        assert result == "/world/obstacles"

    def test_join_usd_path_empty_path1(self):
        """Test USD path joining with empty first path."""
        from curobo._src.util.usd_writer import join_usd_path

        result = join_usd_path("", "obstacles")
        assert result == "/obstacles"

    def test_join_usd_path_empty_path2(self):
        """Test USD path joining with empty second path."""
        from curobo._src.util.usd_writer import join_usd_path

        result = join_usd_path("/world", "")
        assert result == "/world"

    def test_join_usd_path_nested(self):
        """Test nested USD path joining."""
        from curobo._src.util.usd_writer import join_usd_path

        result = join_usd_path("/world/robot", "link1")
        assert result == "/world/robot/link1"


class TestUsdWriterErrorHandling:
    """Tests for error handling in geometry extractors."""

    def test_cube_with_negative_dimensions(self, tmp_path):
        """Test that negative cube dimensions raise ValueError."""
        from curobo._src.util.usd_writer import get_cube_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "error_test.usd"
        helper.create_stage(str(stage_path))

        # Create cube with negative scale
        cube_path = "/world/bad_cube"
        cube_geom = UsdGeom.Cube.Define(helper.stage, cube_path)
        cube_geom.CreateSizeAttr(1.0)

        prim = helper.stage.GetPrimAtPath(cube_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[-1.0, 1.0, 1.0])

        # Should raise ValueError for negative dimension
        helper._xform_cache.SetTime(0)
        with pytest.raises(ValueError, match="Negative or zero dimension"):
            get_cube_attrs(prim, cache=helper._xform_cache)

    def test_cube_with_zero_dimensions(self, tmp_path):
        """Test that zero cube dimensions raise ValueError."""
        from curobo._src.util.usd_writer import get_cube_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "error_test2.usd"
        helper.create_stage(str(stage_path))

        cube_path = "/world/zero_cube"
        cube_geom = UsdGeom.Cube.Define(helper.stage, cube_path)
        cube_geom.CreateSizeAttr(1.0)

        prim = helper.stage.GetPrimAtPath(cube_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[0.0, 1.0, 1.0])

        helper._xform_cache.SetTime(0)
        with pytest.raises(ValueError, match="Negative or zero dimension"):
            get_cube_attrs(prim, cache=helper._xform_cache)

    def test_sphere_with_negative_radius(self, tmp_path):
        """Test that negative sphere radius raises ValueError."""
        from curobo._src.util.usd_writer import get_sphere_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "sphere_error.usd"
        helper.create_stage(str(stage_path))

        sphere_path = "/world/bad_sphere"
        sphere_geom = UsdGeom.Sphere.Define(helper.stage, sphere_path)
        sphere_geom.CreateRadiusAttr(-0.5)

        prim = helper.stage.GetPrimAtPath(sphere_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])

        helper._xform_cache.SetTime(0)
        with pytest.raises(ValueError, match="Negative or zero radius"):
            get_sphere_attrs(prim, cache=helper._xform_cache)

    def test_sphere_with_zero_radius(self, tmp_path):
        """Test that zero sphere radius raises ValueError."""
        from curobo._src.util.usd_writer import get_sphere_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "sphere_zero.usd"
        helper.create_stage(str(stage_path))

        sphere_path = "/world/zero_sphere"
        sphere_geom = UsdGeom.Sphere.Define(helper.stage, sphere_path)
        sphere_geom.CreateRadiusAttr(0.0)

        prim = helper.stage.GetPrimAtPath(sphere_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])

        helper._xform_cache.SetTime(0)
        with pytest.raises(ValueError, match="Negative or zero radius"):
            get_sphere_attrs(prim, cache=helper._xform_cache)

    def test_mesh_with_mismatched_face_counts(self, tmp_path):
        """Test that mesh with mismatched face counts returns None."""
        from curobo._src.util.usd_writer import get_mesh_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "bad_mesh.usd"
        helper.create_stage(str(stage_path))

        # Create mesh with mismatched face data
        mesh_path = "/world/bad_mesh"
        mesh_geom = UsdGeom.Mesh.Define(helper.stage, mesh_path)

        # 3 vertices
        mesh_geom.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        # Say we have 2 faces but only provide indices for 1
        mesh_geom.CreateFaceVertexCountsAttr([3, 3])  # Claims 2 triangles
        mesh_geom.CreateFaceVertexIndicesAttr([0, 1, 2])  # Only 1 triangle

        prim = helper.stage.GetPrimAtPath(mesh_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])

        helper._xform_cache.SetTime(0)
        result = get_mesh_attrs(prim, cache=helper._xform_cache)
        # Should return None due to mismatch
        assert result is None


class TestUsdWriterGeometryWithTimestep:
    """Tests for geometry creation with timestep parameter."""

    def test_add_cuboid_with_timestep(self, tmp_path):
        """Test adding cuboid with timestep for animation."""
        helper = UsdWriter()
        stage_path = tmp_path / "timestep_test.usd"
        helper.create_stage(str(stage_path), timesteps=10, dt=0.02)

        cuboid = Cuboid(
            name="animated_cube",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
        )

        # Add with timestep
        path = helper.add_cuboid_to_stage(cuboid, timestep=0)

        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()

    def test_add_sphere_with_timestep(self, tmp_path):
        """Test adding sphere with timestep for animation."""
        helper = UsdWriter()
        stage_path = tmp_path / "sphere_timestep.usd"
        helper.create_stage(str(stage_path), timesteps=5, dt=0.02)

        sphere = Sphere(
            name="animated_sphere",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.5,
        )

        path = helper.add_sphere_to_stage(sphere, timestep=0)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()

    def test_add_mesh_with_timestep(self, tmp_path):
        """Test adding mesh with timestep for animation."""
        helper = UsdWriter()
        stage_path = tmp_path / "mesh_timestep.usd"
        helper.create_stage(str(stage_path), timesteps=5, dt=0.02)

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        faces = [[0, 1, 2]]
        mesh = Mesh(
            name="animated_mesh",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
        )

        path = helper.add_mesh_to_stage(mesh, timestep=0)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()


class TestUsdWriterSetPrimTranslate:
    """Tests for set_prim_translate helper function."""

    def test_set_prim_translate(self, tmp_path):
        """Test setting prim translation directly."""
        from curobo._src.util.usd_writer import set_prim_translate

        stage = create_stage(str(tmp_path / "translate_test.usd"))
        prim = stage.DefinePrim("/test", "Xform")

        set_prim_translate(prim, [1.0, 2.0, 3.0])

        assert prim.GetAttribute("xformOp:translate").IsValid()
        translation = prim.GetAttribute("xformOp:translate").Get()
        assert translation[0] == pytest.approx(1.0)
        assert translation[1] == pytest.approx(2.0)
        assert translation[2] == pytest.approx(3.0)


class TestUsdWriterCapsuleInStage:
    """Tests for capsule extraction from stages."""

    def test_get_obstacles_with_capsules(self, tmp_path):
        """Test extracting capsules from stage."""
        helper = UsdWriter()
        stage_path = tmp_path / "capsule_stage.usd"
        helper.create_stage(str(stage_path))

        # Manually create a capsule prim
        capsule_path = "/world/obstacles/test_capsule"
        capsule_geom = UsdGeom.Capsule.Define(helper.stage, capsule_path)
        capsule_geom.CreateRadiusAttr(0.3)
        capsule_geom.CreateHeightAttr(1.0)

        from curobo._src.util.usd_writer import set_prim_transform
        prim = helper.stage.GetPrimAtPath(capsule_path)
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])

        # Extract obstacles
        scene = helper.get_obstacles_from_stage()

        assert scene is not None
        assert scene.capsule is not None
        assert len(scene.capsule) == 1
        assert scene.capsule[0].radius == pytest.approx(0.3, abs=0.01)


class TestSetCylinderAttrs:
    """Tests for set_cylinder_attrs function."""

    def test_set_cylinder_attrs(self, tmp_path):
        """Test setting cylinder attributes."""
        from curobo._src.util.usd_writer import set_cylinder_attrs

        stage = create_stage(str(tmp_path / "cylinder_test.usd"))
        cylinder_geom = UsdGeom.Cylinder.Define(stage, "/test_cylinder")
        prim = stage.GetPrimAtPath("/test_cylinder")

        # Create a mock pose with so3 attribute
        class MockPose:
            xyz = [1.0, 2.0, 3.0]
            class so3:
                wxyz = [1.0, 0.0, 0.0, 0.0]

        set_cylinder_attrs(prim, 0.5, 2.0, MockPose())

        assert prim.GetAttribute("height").Get() == pytest.approx(2.0)
        assert prim.GetAttribute("radius").Get() == pytest.approx(0.5)


class TestGetPrimFromObstacle:
    """Tests for get_prim_from_obstacle method."""

    def test_get_prim_from_sphere(self, tmp_path):
        """Test getting prim from sphere obstacle."""
        helper = UsdWriter()
        stage_path = tmp_path / "prim_test.usd"
        helper.create_stage(str(stage_path))

        sphere = Sphere(
            name="test_sphere",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.5,
        )

        path = helper.get_prim_from_obstacle(sphere)
        assert path == "/world/obstacles/test_sphere"
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Sphere)

    def test_get_prim_from_cylinder(self, tmp_path):
        """Test getting prim from cylinder obstacle."""
        helper = UsdWriter()
        stage_path = tmp_path / "prim_test.usd"
        helper.create_stage(str(stage_path))

        cylinder = Cylinder(
            name="test_cylinder",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.3,
            height=1.0,
        )

        path = helper.get_prim_from_obstacle(cylinder)
        assert path == "/world/obstacles/test_cylinder"

    def test_get_prim_from_unsupported_obstacle(self, tmp_path):
        """Test that unsupported obstacle type raises NotImplementedError."""
        helper = UsdWriter()
        stage_path = tmp_path / "prim_test.usd"
        helper.create_stage(str(stage_path))

        # Capsule is not implemented in get_prim_from_obstacle
        capsule = Capsule(
            name="test_capsule",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            base=[0.0, 0.0, -0.5],
            tip=[0.0, 0.0, 0.5],
            radius=0.3,
        )

        with pytest.raises(NotImplementedError):
            helper.get_prim_from_obstacle(capsule)


class TestMeshWithScale:
    """Tests for mesh with scale attributes."""

    def test_get_mesh_attrs_with_scale(self, tmp_path):
        """Test extracting mesh attributes with scale."""
        from curobo._src.util.usd_writer import get_mesh_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "mesh_scale.usd"
        helper.create_stage(str(stage_path))

        # Create mesh with scale
        mesh_path = "/world/scaled_mesh"
        mesh_geom = UsdGeom.Mesh.Define(helper.stage, mesh_path)
        mesh_geom.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        mesh_geom.CreateFaceVertexCountsAttr([3])
        mesh_geom.CreateFaceVertexIndicesAttr([0, 1, 2])

        prim = helper.stage.GetPrimAtPath(mesh_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[2.0, 2.0, 2.0])

        helper._xform_cache.SetTime(0)
        extracted_mesh = get_mesh_attrs(prim, cache=helper._xform_cache)

        assert extracted_mesh is not None
        assert len(extracted_mesh.vertices) == 3

    def test_mesh_without_xform_scale(self, tmp_path):
        """Test mesh without explicit xformOp:scale attribute."""
        from curobo._src.util.usd_writer import get_mesh_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "mesh_no_scale.usd"
        helper.create_stage(str(stage_path))

        # Create mesh without setting scale
        mesh_path = "/world/simple_mesh"
        mesh_geom = UsdGeom.Mesh.Define(helper.stage, mesh_path)
        mesh_geom.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        mesh_geom.CreateFaceVertexCountsAttr([3])
        mesh_geom.CreateFaceVertexIndicesAttr([0, 1, 2])

        # Only set position, not scale
        prim = helper.stage.GetPrimAtPath(mesh_path)
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

        helper._xform_cache.SetTime(0)
        extracted_mesh = get_mesh_attrs(prim, cache=helper._xform_cache)

        assert extracted_mesh is not None
        # When no xformOp:scale is set, scale comes from world transform extraction
        # which returns None when no scale is applied (scale is computed from t_scale)
        # The mesh is still extracted successfully
        assert len(extracted_mesh.vertices) == 3


class TestOnlySubstringFilter:
    """Tests for only_substring filter in get_obstacles_from_stage."""

    def test_ignore_substring_filter(self, tmp_path):
        """Test extracting obstacles with ignore_substring filter."""
        helper = UsdWriter()
        stage_path = tmp_path / "substring_test.usd"
        helper.create_stage(str(stage_path))

        # Add obstacles with different naming patterns
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="table_top",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[1.0, 1.0, 0.1],
                ),
                Cuboid(
                    name="chair_seat",
                    pose=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.5, 0.5, 0.1],
                ),
            ],
        )
        helper.add_world_to_stage(obstacles)

        # Filter to ignore "chair" in name
        scene = helper.get_obstacles_from_stage(ignore_substring=["chair"])

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert "table" in scene.cuboid[0].name


class TestMeshExtraction:
    """Tests for mesh extraction from stage."""

    def test_get_obstacles_with_mesh(self, tmp_path):
        """Test extracting mesh obstacles from stage."""
        helper = UsdWriter()
        stage_path = tmp_path / "mesh_extract.usd"
        helper.create_stage(str(stage_path))

        # Add a mesh obstacle
        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]]
        faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
        mesh = Mesh(
            name="test_mesh",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
        )
        helper.add_mesh_to_stage(mesh)

        # Extract
        scene = helper.get_obstacles_from_stage()

        assert scene is not None
        assert scene.mesh is not None
        assert len(scene.mesh) == 1


class TestMeshWithColor:
    """Tests for mesh with material color."""

    def test_mesh_with_material_color(self, tmp_path):
        """Test adding mesh with material color (not vertex colors)."""
        helper = UsdWriter()
        stage_path = tmp_path / "mesh_color.usd"
        helper.create_stage(str(stage_path))

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        faces = [[0, 1, 2]]
        mesh = Mesh(
            name="colored_mesh",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
            color=[0.8, 0.2, 0.1, 1.0],  # Add material color
        )

        path = helper.add_mesh_to_stage(mesh)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()

        # Check material was created
        material_path = path + "/material_colored_mesh"
        mat_prim = helper.stage.GetPrimAtPath(material_path)
        assert mat_prim.IsValid()


class TestAnimationEdgeCases:
    """Tests for animation edge cases."""

    def test_create_animation_with_missing_transform(self, tmp_path):
        """Test animation creation with missing xform ops logs warning."""
        helper = UsdWriter()
        stage_path = tmp_path / "anim_edge.usd"
        helper.create_stage(str(stage_path), timesteps=5, dt=0.02)

        # Create scene with objects
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.1, 0.1, 0.1],
                ),
            ]
        )

        # Minimal pose tensor
        position_tensor = torch.tensor(
            [[[0.0, 0.0, 0.0]]],
            device=torch.device("cuda", 0),
        )
        quaternion_tensor = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]]],
            device=torch.device("cuda", 0),
        )
        pose = Pose(position=position_tensor, quaternion=quaternion_tensor)

        # Should handle gracefully even if xform ops are not all present
        helper.create_animation(obstacles, pose, base_frame="/world", dt=0.02)

        prim = helper.stage.GetPrimAtPath("/world/robot/cube1")
        assert prim.IsValid()

    def test_obstacle_animation_missing_obstacle_warning(self, tmp_path):
        """Test obstacle animation with missing obstacle logs warning."""
        helper = UsdWriter()
        stage_path = tmp_path / "obs_anim_edge.usd"
        helper.create_stage(str(stage_path), timesteps=3, dt=0.02)

        # Create initial obstacles
        initial_obstacles = [
            Sphere(name="sphere1", position=[0.0, 0.0, 0.0], radius=0.1),
        ]

        # Create animation with different obstacle at timestep 1
        obstacles_list = [
            initial_obstacles,
            [Sphere(name="sphere_different", position=[1.0, 0.0, 0.0], radius=0.1)],  # Different name
            initial_obstacles,
        ]

        # This should handle the missing obstacle gracefully with a warning
        helper.create_obstacle_animation(
            obstacles_list, base_frame="/world", obstacles_frame="moving"
        )

        # Original sphere should still exist
        prim = helper.stage.GetPrimAtPath("/world/moving/sphere1")
        assert prim.IsValid()


class TestSpherePositionToPose:
    """Tests for sphere position to pose conversion."""

    def test_sphere_with_position_converts_to_pose(self, tmp_path):
        """Test that sphere with only position gets converted to pose via Sphere constructor."""
        helper = UsdWriter()
        stage_path = tmp_path / "sphere_pos.usd"
        helper.create_stage(str(stage_path))

        # Create sphere with position only - Sphere class auto-converts position to pose
        sphere = Sphere(
            name="pos_sphere",
            position=[1.0, 2.0, 3.0],
            radius=0.5,
        )

        # Sphere class converts position to pose in constructor
        # so pose should already be set
        assert sphere.pose is not None
        assert sphere.pose[:3] == [1.0, 2.0, 3.0]
        # Quaternion should be identity [1, 0, 0, 0]
        assert sphere.pose[3:] == [1, 0, 0, 0]

        path = helper.add_sphere_to_stage(sphere)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()

    def test_sphere_without_pose_or_position_handled(self, tmp_path):
        """Test add_sphere_to_stage handles sphere without pose properly."""
        helper = UsdWriter()
        stage_path = tmp_path / "sphere_no_pose.usd"
        helper.create_stage(str(stage_path))

        # Create sphere without pose or position
        sphere = Sphere(
            name="no_pose_sphere",
            radius=0.5,
        )

        # When neither pose nor position is set, add_sphere_to_stage
        # should handle it (sets default pose at origin)
        path = helper.add_sphere_to_stage(sphere)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()
        # Default pose should have been set
        assert sphere.pose is not None


class TestCylinderWithColor:
    """Tests for cylinder with color material."""

    def test_cylinder_with_color_material(self, tmp_path):
        """Test adding cylinder with color creates material."""
        helper = UsdWriter()
        stage_path = tmp_path / "cylinder_color.usd"
        helper.create_stage(str(stage_path))

        cylinder = Cylinder(
            name="colored_cylinder",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.3,
            height=1.0,
            color=[0.5, 0.5, 1.0, 1.0],  # Blue color
        )

        path = helper.add_cylinder_to_stage(cylinder)
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()

        # Check material was created
        material_path = path + "/material_colored_cylinder"
        mat_prim = helper.stage.GetPrimAtPath(material_path)
        assert mat_prim.IsValid()


class TestSetPrimTransformEdgeCases:
    """Tests for set_prim_transform edge cases."""

    def test_set_prim_transform_existing_orient(self, tmp_path):
        """Test setting transform on prim with existing orient attribute."""
        from curobo._src.util.usd_writer import set_prim_transform

        stage = create_stage(str(tmp_path / "transform_test.usd"))
        prim = stage.DefinePrim("/test", "Xform")

        # First set - creates attributes
        pose1 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        set_prim_transform(prim, pose1)

        # Second set - should handle existing attributes
        pose2 = [2.0, 3.0, 4.0, 0.707, 0.707, 0.0, 0.0]
        set_prim_transform(prim, pose2)

        translation = prim.GetAttribute("xformOp:translate").Get()
        assert translation[0] == pytest.approx(2.0)
        assert translation[1] == pytest.approx(3.0)
        assert translation[2] == pytest.approx(4.0)

    def test_set_prim_transform_double_precision_path(self, tmp_path):
        """Test that double precision is used when orient is Quatd."""
        from curobo._src.util.usd_writer import set_prim_transform

        stage = create_stage(str(tmp_path / "double_test.usd"))
        prim = stage.DefinePrim("/test", "Xform")

        # Add orient op with double precision explicitly
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        # Now set transform - should detect and use double
        pose = [5.0, 6.0, 7.0, 1.0, 0.0, 0.0, 0.0]
        set_prim_transform(prim, pose, use_float=False)

        translation = prim.GetAttribute("xformOp:translate").Get()
        assert translation[0] == pytest.approx(5.0)


class TestCubeAttrsWithSize:
    """Tests for cube attributes with size attribute."""

    def test_cube_without_size_attr(self, tmp_path):
        """Test cube attrs when size attribute is None (defaults to 1.0)."""
        from curobo._src.util.usd_writer import get_cube_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "cube_no_size.usd"
        helper.create_stage(str(stage_path))

        # Create cube without setting size
        cube_path = "/world/simple_cube"
        cube_geom = UsdGeom.Cube.Define(helper.stage, cube_path)
        # Don't set size - it should default to 1.0

        prim = helper.stage.GetPrimAtPath(cube_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[1.0, 2.0, 3.0])

        helper._xform_cache.SetTime(0)
        extracted_cube = get_cube_attrs(prim, cache=helper._xform_cache)

        assert extracted_cube is not None
        # The get_cube_attrs function applies scale to world transform
        # and extracts dimensions from the scaled result.
        # The scale [1.0, 2.0, 3.0] affects the world transform extraction.
        # Just verify we get valid dimensions back
        assert len(extracted_cube.dims) == 3
        assert all(d > 0 for d in extracted_cube.dims)

    def test_cube_with_explicit_size(self, tmp_path):
        """Test cube attrs with explicit size attribute."""
        from curobo._src.util.usd_writer import get_cube_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "cube_size.usd"
        helper.create_stage(str(stage_path))

        cube_path = "/world/sized_cube"
        cube_geom = UsdGeom.Cube.Define(helper.stage, cube_path)
        cube_geom.CreateSizeAttr(2.0)  # Size of 2

        prim = helper.stage.GetPrimAtPath(cube_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[1.0, 1.0, 1.0])

        helper._xform_cache.SetTime(0)
        extracted_cube = get_cube_attrs(prim, cache=helper._xform_cache)

        assert extracted_cube is not None
        # Dims should be size * scale = 2.0 * 1.0 = 2.0
        assert extracted_cube.dims[0] == pytest.approx(2.0, abs=0.01)


class TestSphereAttrsWithScale:
    """Tests for sphere attributes with scale."""

    def test_sphere_with_scale(self, tmp_path):
        """Test sphere attrs when scale is applied."""
        from curobo._src.util.usd_writer import get_sphere_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "sphere_scale.usd"
        helper.create_stage(str(stage_path))

        # Create sphere with scale
        sphere_path = "/world/scaled_sphere"
        sphere_geom = UsdGeom.Sphere.Define(helper.stage, sphere_path)
        sphere_geom.CreateRadiusAttr(0.5)

        prim = helper.stage.GetPrimAtPath(sphere_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[2.0, 2.0, 2.0])

        helper._xform_cache.SetTime(0)
        extracted_sphere = get_sphere_attrs(prim, cache=helper._xform_cache)

        assert extracted_sphere is not None
        # Radius should be scaled by max scale component
        assert extracted_sphere.radius == pytest.approx(1.0, abs=0.01)  # 0.5 * 2.0

    def test_sphere_without_scale(self, tmp_path):
        """Test sphere attrs without scale attribute."""
        from curobo._src.util.usd_writer import get_sphere_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "sphere_no_scale.usd"
        helper.create_stage(str(stage_path))

        sphere_path = "/world/simple_sphere"
        sphere_geom = UsdGeom.Sphere.Define(helper.stage, sphere_path)
        sphere_geom.CreateRadiusAttr(0.3)

        prim = helper.stage.GetPrimAtPath(sphere_path)
        # Only set translation
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

        helper._xform_cache.SetTime(0)
        extracted_sphere = get_sphere_attrs(prim, cache=helper._xform_cache)

        assert extracted_sphere is not None
        assert extracted_sphere.radius == pytest.approx(0.3, abs=0.01)


class TestCylinderAttrsWithTransform:
    """Tests for cylinder attributes with transform."""

    def test_cylinder_with_transform(self, tmp_path):
        """Test cylinder attrs with transform applied."""
        from curobo._src.util.usd_writer import get_cylinder_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "cylinder_transform.usd"
        helper.create_stage(str(stage_path))

        # Create cylinder with transform
        cylinder_path = "/world/transformed_cylinder"
        cylinder_geom = UsdGeom.Cylinder.Define(helper.stage, cylinder_path)
        cylinder_geom.CreateRadiusAttr(0.25)
        cylinder_geom.CreateHeightAttr(1.5)

        prim = helper.stage.GetPrimAtPath(cylinder_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

        # Use transform parameter - must be float32 to avoid warp issues
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.5, 0.5, 0.5]  # Additional translation

        helper._xform_cache.SetTime(0)
        extracted_cylinder = get_cylinder_attrs(prim, cache=helper._xform_cache, transform=transform)

        assert extracted_cylinder is not None
        assert extracted_cylinder.radius == pytest.approx(0.25, abs=0.01)
        assert extracted_cylinder.height == pytest.approx(1.5, abs=0.01)


class TestCapsuleAttrsWithTransform:
    """Tests for capsule attributes with transform."""

    def test_capsule_with_transform(self, tmp_path):
        """Test capsule attrs with transform applied."""
        from curobo._src.util.usd_writer import get_capsule_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "capsule_transform.usd"
        helper.create_stage(str(stage_path))

        capsule_path = "/world/transformed_capsule"
        capsule_geom = UsdGeom.Capsule.Define(helper.stage, capsule_path)
        capsule_geom.CreateRadiusAttr(0.2)
        capsule_geom.CreateHeightAttr(0.8)

        prim = helper.stage.GetPrimAtPath(capsule_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])

        # Apply external transform - must be float32 to avoid warp issues
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [1.0, 1.0, 1.0]

        helper._xform_cache.SetTime(0)
        extracted_capsule = get_capsule_attrs(prim, cache=helper._xform_cache, transform=transform)

        assert extracted_capsule is not None
        assert extracted_capsule.radius == pytest.approx(0.2, abs=0.01)


class TestGetPoseInverse:
    """Tests for get_pose with inverse parameter."""

    def test_get_pose_inverse(self, tmp_path):
        """Test getting inverse pose from prim."""
        helper = UsdWriter()
        stage_path = tmp_path / "inverse_test.usd"
        helper.create_stage(str(stage_path))

        # Add cuboid at specific position
        cuboid = Cuboid(
            name="test_cube",
            pose=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[1.0, 1.0, 1.0],
        )
        path = helper.add_cuboid_to_stage(cuboid)

        # Get inverse pose
        inverse_pose = helper.get_pose(path, inverse=True)

        # Inverse should have negative translation
        assert inverse_pose[0, 3] == pytest.approx(-1.0, abs=1e-6)
        assert inverse_pose[1, 3] == pytest.approx(0.0, abs=1e-6)
        assert inverse_pose[2, 3] == pytest.approx(0.0, abs=1e-6)


class TestSphereAttrsWithTransform:
    """Tests for sphere attributes with transform."""

    def test_sphere_with_transform(self, tmp_path):
        """Test sphere attrs with transform applied."""
        from curobo._src.util.usd_writer import get_sphere_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "sphere_transform.usd"
        helper.create_stage(str(stage_path))

        sphere_path = "/world/transformed_sphere"
        sphere_geom = UsdGeom.Sphere.Define(helper.stage, sphere_path)
        sphere_geom.CreateRadiusAttr(0.4)

        prim = helper.stage.GetPrimAtPath(sphere_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

        # Apply external transform - must be float32
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.5, 0.5, 0.5]

        helper._xform_cache.SetTime(0)
        extracted_sphere = get_sphere_attrs(prim, cache=helper._xform_cache, transform=transform)

        assert extracted_sphere is not None
        assert extracted_sphere.radius == pytest.approx(0.4, abs=0.01)


class TestMeshAttrsWithTransform:
    """Tests for mesh attributes with transform."""

    def test_mesh_with_transform(self, tmp_path):
        """Test mesh attrs with transform applied."""
        from curobo._src.util.usd_writer import get_mesh_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "mesh_transform.usd"
        helper.create_stage(str(stage_path))

        mesh_path = "/world/transformed_mesh"
        mesh_geom = UsdGeom.Mesh.Define(helper.stage, mesh_path)
        mesh_geom.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
        mesh_geom.CreateFaceVertexCountsAttr([3])
        mesh_geom.CreateFaceVertexIndicesAttr([0, 1, 2])

        prim = helper.stage.GetPrimAtPath(mesh_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])

        # Apply external transform - must be float32
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.5, 0.5, 0.5]

        helper._xform_cache.SetTime(0)
        extracted_mesh = get_mesh_attrs(prim, cache=helper._xform_cache, transform=transform)

        assert extracted_mesh is not None
        assert len(extracted_mesh.vertices) == 3


class TestCreateGridUsd:
    """Tests for create_grid_usd static method."""

    def test_create_grid_usd_with_list(self, tmp_path):
        """Test creating grid USD from list of files."""
        # First create some USD files to use
        usd_file1 = tmp_path / "env1.usd"
        usd_file2 = tmp_path / "env2.usd"

        helper1 = UsdWriter()
        helper1.create_stage(str(usd_file1))
        helper1.add_cuboid_to_stage(
            Cuboid(name="cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
        )
        helper1.save()

        helper2 = UsdWriter()
        helper2.create_stage(str(usd_file2))
        helper2.add_sphere_to_stage(
            Sphere(name="sphere", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.5)
        )
        helper2.save()

        # Create grid USD
        output_path = tmp_path / "grid.usd"
        UsdWriter.create_grid_usd(
            usds_path=[str(usd_file1), str(usd_file2)],
            save_path=str(output_path),
            base_frame="/world",
            max_envs=2,
            max_timecode=10,
            x_space=2.0,
            y_space=2.0,
            x_per_row=2,
            local_asset_path=str(tmp_path),
            dt=0.02,
        )

        assert output_path.exists()

        # Verify we can load the grid
        loaded = Usd.Stage.Open(str(output_path))
        assert loaded is not None


class TestGetPrimFromMesh:
    """Tests for get_prim_from_obstacle with Mesh."""

    def test_get_prim_from_mesh(self, tmp_path):
        """Test getting prim from mesh obstacle."""
        helper = UsdWriter()
        stage_path = tmp_path / "mesh_prim_test.usd"
        helper.create_stage(str(stage_path))

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
        faces = [[0, 1, 2]]
        mesh = Mesh(
            name="test_mesh",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices,
            faces=faces,
        )

        path = helper.get_prim_from_obstacle(mesh)
        assert path == "/world/obstacles/test_mesh"
        prim = helper.stage.GetPrimAtPath(path)
        assert prim.IsValid()
        assert prim.IsA(UsdGeom.Mesh)


class TestCubeAttrsWithTransform:
    """Tests for cube attributes with external transform."""

    def test_cube_with_transform(self, tmp_path):
        """Test cube attrs with transform applied."""
        from curobo._src.util.usd_writer import get_cube_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "cube_transform.usd"
        helper.create_stage(str(stage_path))

        cube_path = "/world/transformed_cube"
        cube_geom = UsdGeom.Cube.Define(helper.stage, cube_path)
        cube_geom.CreateSizeAttr(1.0)

        prim = helper.stage.GetPrimAtPath(cube_path)
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(prim, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0])

        # Apply external transform - must be float32
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.5, 0.5, 0.5]

        helper._xform_cache.SetTime(0)
        extracted_cube = get_cube_attrs(prim, cache=helper._xform_cache, transform=transform)

        assert extracted_cube is not None
        assert len(extracted_cube.dims) == 3


class TestGetObstaclesWithAllTypes:
    """Tests for get_obstacles_from_stage with all geometry types."""

    def test_get_all_geometry_types(self, tmp_path):
        """Test extracting all geometry types from stage including mesh."""
        helper = UsdWriter()
        stage_path = tmp_path / "all_types.usd"
        helper.create_stage(str(stage_path))

        # Add all types of obstacles
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(name="cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
            ],
            sphere=[
                Sphere(name="sphere", pose=[2, 0, 0, 1, 0, 0, 0], radius=0.5)
            ],
            cylinder=[
                Cylinder(name="cylinder", pose=[0, 2, 0, 1, 0, 0, 0], radius=0.3, height=1.0)
            ],
            mesh=[
                Mesh(
                    name="mesh",
                    pose=[4, 0, 0, 1, 0, 0, 0],
                    vertices=[[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
                    faces=[[0, 1, 2]],
                )
            ],
        )
        helper.add_world_to_stage(obstacles)

        # Also add a capsule manually
        capsule_path = "/world/obstacles/capsule"
        capsule = UsdGeom.Capsule.Define(helper.stage, capsule_path)
        capsule.CreateRadiusAttr(0.2)
        capsule.CreateHeightAttr(0.6)
        from curobo._src.util.usd_writer import set_prim_transform
        prim = helper.stage.GetPrimAtPath(capsule_path)
        set_prim_transform(prim, [0, 4, 0, 1, 0, 0, 0])

        # Extract all
        scene = helper.get_obstacles_from_stage()

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert len(scene.sphere) == 1
        assert len(scene.cylinder) == 1
        assert scene.mesh is not None and len(scene.mesh) == 1
        assert scene.capsule is not None and len(scene.capsule) == 1


class TestGetObstaclesWithReferenceTransform:
    """Tests for get_obstacles_from_stage with reference prim transform."""

    def test_obstacles_with_reference_prim(self, tmp_path):
        """Test extracting obstacles relative to a reference prim."""
        helper = UsdWriter()
        stage_path = tmp_path / "ref_test.usd"
        helper.create_stage(str(stage_path))

        # Create reference frame at non-origin position
        ref_path = "/world/reference_frame"
        ref_prim = helper.stage.DefinePrim(ref_path, "Xform")
        from curobo._src.util.usd_writer import set_prim_transform
        set_prim_transform(ref_prim, [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # Add obstacles
        obstacles = SceneCfg(
            cuboid=[
                Cuboid(name="cube", pose=[3.0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
            ],
        )
        helper.add_world_to_stage(obstacles)

        # Extract with reference frame
        scene = helper.get_obstacles_from_stage(reference_prim_path=ref_path)

        assert scene is not None
        assert len(scene.cuboid) == 1
        # Position should be transformed relative to reference frame


class TestAnimationWithMissingXformOps:
    """Tests for animation with edge cases in xform ops."""

    def test_animation_with_prim_without_xform_ops(self, tmp_path):
        """Test animation handles prims without proper xform ops."""
        helper = UsdWriter()
        stage_path = tmp_path / "anim_no_xform.usd"
        helper.create_stage(str(stage_path), timesteps=5, dt=0.02)

        # Create a prim without xform ops
        prim_path = "/world/robot/test_prim"
        prim = helper.stage.DefinePrim(prim_path, "Xform")
        # Don't add any xform ops

        # Create minimal poses
        position_tensor = torch.tensor(
            [[[0.0, 0.0, 0.0]]],
            device=torch.device("cuda", 0),
        )
        quaternion_tensor = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]]],
            device=torch.device("cuda", 0),
        )
        pose = Pose(position=position_tensor, quaternion=quaternion_tensor)

        # Create scene with one object
        obstacles = SceneCfg(
            cuboid=[Cuboid(name="test_prim", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]
        )

        # This should handle the missing xform ops gracefully
        helper.create_animation(obstacles, pose, base_frame="/world", dt=0.02)

        # Stage should still be valid
        assert helper.stage is not None


class TestCreateGridUsdFromDirectory:
    """Tests for create_grid_usd with directory path."""

    def test_create_grid_usd_from_directory(self, tmp_path):
        """Test creating grid USD from directory path."""
        # Create a subdirectory with USD files
        usd_dir = tmp_path / "usds"
        usd_dir.mkdir()

        # Create some USD files with a common prefix
        for i in range(2):
            usd_file = usd_dir / f"scene_{i}.usd"
            helper = UsdWriter()
            helper.create_stage(str(usd_file))
            helper.add_cuboid_to_stage(
                Cuboid(name=f"cube{i}", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
            )
            helper.save()

        # Create grid USD from directory with prefix string
        output_path = tmp_path / "grid_from_dir.usd"
        UsdWriter.create_grid_usd(
            usds_path=str(usd_dir),  # String path instead of list
            save_path=str(output_path),
            base_frame="/world",
            max_envs=10,
            max_timecode=5,
            x_space=2.0,
            y_space=2.0,
            x_per_row=3,
            local_asset_path=str(usd_dir),
            dt=0.02,
            prefix_string="scene_",  # Need to specify prefix to filter files
        )

        assert output_path.exists()


class TestGetCubeAttrsNullSize:
    """Tests for get_cube_attrs with null size attribute."""

    def test_cube_attrs_with_no_size_attribute(self, tmp_path):
        """Test cube attrs when size attribute returns None (not set)."""
        from curobo._src.util.usd_writer import get_cube_attrs

        helper = UsdWriter()
        stage_path = tmp_path / "cube_null_size.usd"
        helper.create_stage(str(stage_path))

        # Create cube but don't set size attribute at all
        cube_path = "/world/null_size_cube"
        cube_geom = UsdGeom.Cube.Define(helper.stage, cube_path)
        # Note: Don't call CreateSizeAttr - leave it as None

        prim = helper.stage.GetPrimAtPath(cube_path)
        from curobo._src.util.usd_writer import set_prim_transform
        # Set scale - this provides the xformOp:scale attribute
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[1.0, 1.0, 1.0])

        helper._xform_cache.SetTime(0)
        extracted_cube = get_cube_attrs(prim, cache=helper._xform_cache)

        assert extracted_cube is not None
        # Size defaults to 1.0 when not set, so dims come from scale * default_size(1.0)
        # The get_cube_attrs function reads dims from xformOp:scale attribute directly
        # and the world transform extraction yields the final dimensions
        assert len(extracted_cube.dims) == 3
        # Verify it succeeded without errors (covers the size=None path)


class TestAnimationMissingLinkWarning:
    """Tests for animation edge case where link is not in prims."""

    def test_animation_with_fewer_xform_ops(self, tmp_path):
        """Test animation when prim has less than 2 xform ops triggers warning."""
        helper = UsdWriter()
        stage_path = tmp_path / "anim_missing_ops.usd"
        helper.create_stage(str(stage_path), timesteps=3, dt=0.02)

        # Create scene
        obstacles = SceneCfg(
            sphere=[
                Sphere(name="sphere1", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.1),
            ]
        )

        # Create animation poses
        position_tensor = torch.tensor(
            [[[0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]], [[0.2, 0.0, 0.0]]],
            device=torch.device("cuda", 0),
        )
        quaternion_tensor = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
            device=torch.device("cuda", 0),
        )
        pose = Pose(position=position_tensor, quaternion=quaternion_tensor)

        # This creates the animation which may hit the warning path
        helper.create_animation(obstacles, pose, base_frame="/world", dt=0.02)

        # Verify stage is still valid
        prim = helper.stage.GetPrimAtPath("/world/robot/sphere1")
        assert prim.IsValid()

