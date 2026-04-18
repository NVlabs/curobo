# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for UsdSceneParser class."""

# Standard Library

# Third Party
import pytest

# CuRobo
from curobo._src.geom.types import Cuboid, Cylinder, SceneCfg, Sphere

try:
    # Third Party
    from pxr import Usd, UsdGeom

    from curobo._src.util.usd_scene_parser import (
        UsdSceneParser,
        get_capsule_attrs,
        get_cube_attrs,
        get_cylinder_attrs,
        get_mesh_attrs,
        get_sphere_attrs,
    )
    from curobo._src.util.usd_util import create_stage, set_prim_transform
    from curobo._src.util.usd_writer import UsdWriter
except ImportError:
    pytest.skip("usd-core not available", allow_module_level=True)


class TestUsdSceneParser:
    """Tests for UsdSceneParser class."""

    @pytest.fixture
    def parser(self):
        """Create a UsdSceneParser instance for testing."""
        return UsdSceneParser()

    @pytest.fixture
    def stage_with_obstacles(self, tmp_path):
        """Create a USD stage with obstacles using UsdWriter."""
        writer = UsdWriter()
        stage_path = tmp_path / "test_scene.usd"
        writer.create_stage(str(stage_path))

        obstacles = SceneCfg(
            cuboid=[
                Cuboid(
                    name="cube1",
                    pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.5, 0.6, 0.7],
                )
            ],
            sphere=[
                Sphere(
                    name="sphere1",
                    pose=[4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.5,
                )
            ],
            cylinder=[
                Cylinder(
                    name="cylinder1",
                    pose=[0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.3,
                    height=1.0,
                )
            ],
        )
        writer.add_world_to_stage(obstacles)
        writer.save()

        return str(stage_path)

    def test_parser_initialization(self, parser):
        """Test UsdSceneParser initialization."""
        assert parser.stage is None

    def test_load_stage_from_file(self, parser, stage_with_obstacles):
        """Test loading stage from file."""
        parser.load_stage_from_file(stage_with_obstacles)
        assert parser.stage is not None

    def test_load_stage_directly(self, parser):
        """Test loading stage object directly."""
        stage = Usd.Stage.CreateInMemory()
        parser.load_stage(stage)
        assert parser.stage is stage

    def test_get_obstacles_from_stage(self, parser, stage_with_obstacles):
        """Test extracting obstacles from stage."""
        parser.load_stage_from_file(stage_with_obstacles)
        scene = parser.get_obstacles_from_stage()

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert len(scene.sphere) == 1
        assert len(scene.cylinder) == 1

    def test_get_obstacles_with_only_paths(self, parser, stage_with_obstacles):
        """Test filtering obstacles by path prefix."""
        parser.load_stage_from_file(stage_with_obstacles)
        scene = parser.get_obstacles_from_stage(
            only_paths=["/world/obstacles/cube1"]
        )

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert scene.sphere is None or len(scene.sphere) == 0

    def test_get_obstacles_with_ignore_paths(self, parser, stage_with_obstacles):
        """Test excluding obstacles by path prefix."""
        parser.load_stage_from_file(stage_with_obstacles)
        scene = parser.get_obstacles_from_stage(
            ignore_paths=["/world/obstacles/sphere1"]
        )

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert len(scene.cylinder) == 1
        assert scene.sphere is None or len(scene.sphere) == 0

    def test_get_obstacles_with_substring_filter(self, parser, stage_with_obstacles):
        """Test filtering obstacles by substring."""
        parser.load_stage_from_file(stage_with_obstacles)
        scene = parser.get_obstacles_from_stage(only_substring=["cube"])

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert scene.sphere is None or len(scene.sphere) == 0
        assert scene.cylinder is None or len(scene.cylinder) == 0

    def test_get_obstacles_with_ignore_substring(self, parser, stage_with_obstacles):
        """Test excluding obstacles by substring."""
        parser.load_stage_from_file(stage_with_obstacles)
        scene = parser.get_obstacles_from_stage(ignore_substring=["cylinder"])

        assert scene is not None
        assert len(scene.cuboid) == 1
        assert len(scene.sphere) == 1
        assert scene.cylinder is None or len(scene.cylinder) == 0

    def test_get_pose(self, parser, stage_with_obstacles):
        """Test getting pose of a prim."""
        parser.load_stage_from_file(stage_with_obstacles)
        pose_matrix = parser.get_pose("/world/obstacles/cube1")

        assert pose_matrix is not None
        assert pose_matrix.shape == (4, 4)
        # Check translation
        assert pose_matrix[0, 3] == pytest.approx(1.0, abs=0.1)
        assert pose_matrix[1, 3] == pytest.approx(2.0, abs=0.1)
        assert pose_matrix[2, 3] == pytest.approx(3.0, abs=0.1)

    def test_get_pose_inverse(self, parser, stage_with_obstacles):
        """Test getting inverse pose of a prim."""
        parser.load_stage_from_file(stage_with_obstacles)
        pose_matrix = parser.get_pose("/world/obstacles/cube1", inverse=True)

        assert pose_matrix is not None
        # Inverse translation should be negated
        assert pose_matrix[0, 3] == pytest.approx(-1.0, abs=0.1)


class TestGeometryExtractors:
    """Tests for geometry extraction functions."""

    @pytest.fixture
    def helper_with_stage(self, tmp_path):
        """Create a UsdWriter with an initialized stage."""
        helper = UsdWriter()
        stage_path = tmp_path / "geom_test.usd"
        helper.create_stage(str(stage_path))
        return helper

    def test_get_cube_attrs(self, helper_with_stage):
        """Test extracting cube attributes."""
        cuboid = Cuboid(
            name="test_cube",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            dims=[0.5, 0.6, 0.7],
        )
        path = helper_with_stage.add_cuboid_to_stage(cuboid)
        prim = helper_with_stage.stage.GetPrimAtPath(path)

        helper_with_stage._xform_cache.SetTime(0)
        extracted = get_cube_attrs(prim, cache=helper_with_stage._xform_cache)

        assert extracted is not None
        assert extracted.dims[0] == pytest.approx(0.5, abs=0.01)
        assert extracted.dims[1] == pytest.approx(0.6, abs=0.01)
        assert extracted.dims[2] == pytest.approx(0.7, abs=0.01)

    def test_get_sphere_attrs(self, helper_with_stage):
        """Test extracting sphere attributes."""
        sphere = Sphere(
            name="test_sphere",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.5,
        )
        path = helper_with_stage.add_sphere_to_stage(sphere)
        prim = helper_with_stage.stage.GetPrimAtPath(path)

        helper_with_stage._xform_cache.SetTime(0)
        extracted = get_sphere_attrs(prim, cache=helper_with_stage._xform_cache)

        assert extracted is not None
        assert extracted.radius == pytest.approx(0.5, abs=0.01)

    def test_get_cylinder_attrs(self, helper_with_stage):
        """Test extracting cylinder attributes."""
        cylinder = Cylinder(
            name="test_cylinder",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.3,
            height=1.0,
        )
        path = helper_with_stage.add_cylinder_to_stage(cylinder)
        prim = helper_with_stage.stage.GetPrimAtPath(path)

        helper_with_stage._xform_cache.SetTime(0)
        extracted = get_cylinder_attrs(prim, cache=helper_with_stage._xform_cache)

        assert extracted is not None
        assert extracted.radius == pytest.approx(0.3, abs=0.01)
        assert extracted.height == pytest.approx(1.0, abs=0.01)

    def test_get_capsule_attrs(self, tmp_path):
        """Test extracting capsule attributes."""
        helper = UsdWriter()
        stage_path = tmp_path / "capsule_test.usd"
        helper.create_stage(str(stage_path))

        # Create capsule manually
        capsule_path = "/world/test_capsule"
        capsule_geom = UsdGeom.Capsule.Define(helper.stage, capsule_path)
        capsule_geom.CreateRadiusAttr(0.25)
        capsule_geom.CreateHeightAttr(0.8)

        prim = helper.stage.GetPrimAtPath(capsule_path)
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])

        helper._xform_cache.SetTime(0)
        extracted = get_capsule_attrs(prim, cache=helper._xform_cache)

        assert extracted is not None
        assert extracted.radius == pytest.approx(0.25, abs=0.01)

    def test_get_cube_attrs_with_zero_dimension_raises(self, tmp_path):
        """Test that zero cube dimension raises ValueError."""
        stage = create_stage(str(tmp_path / "zero_cube.usd"))
        cube_path = "/world/zero_cube"
        cube_geom = UsdGeom.Cube.Define(stage, cube_path)
        cube_geom.CreateSizeAttr(1.0)

        prim = stage.GetPrimAtPath(cube_path)
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0], scale=[0.0, 1.0, 1.0])

        cache = UsdGeom.XformCache()
        cache.SetTime(0)

        with pytest.raises(ValueError, match="Negative or zero dimension"):
            get_cube_attrs(prim, cache=cache)

    def test_get_sphere_attrs_with_zero_radius_raises(self, tmp_path):
        """Test that zero sphere radius raises ValueError."""
        stage = create_stage(str(tmp_path / "zero_sphere.usd"))
        sphere_path = "/world/zero_sphere"
        sphere_geom = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere_geom.CreateRadiusAttr(0.0)

        prim = stage.GetPrimAtPath(sphere_path)
        set_prim_transform(prim, [0, 0, 0, 1, 0, 0, 0])

        cache = UsdGeom.XformCache()
        cache.SetTime(0)

        with pytest.raises(ValueError, match="Negative or zero radius"):
            get_sphere_attrs(prim, cache=cache)


class TestUsdSceneParserRoundTrip:
    """Tests for round-trip (write then read) functionality."""

    def test_cuboid_roundtrip(self, tmp_path):
        """Test writing and reading cuboid."""
        # Write
        writer = UsdWriter()
        stage_path = tmp_path / "roundtrip.usd"
        writer.create_stage(str(stage_path))

        original = Cuboid(
            name="test_cube",
            pose=[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            dims=[0.5, 0.6, 0.7],
        )
        writer.add_cuboid_to_stage(original)
        writer.save()

        # Read with parser
        parser = UsdSceneParser()
        parser.load_stage_from_file(str(stage_path))
        scene = parser.get_obstacles_from_stage()

        assert len(scene.cuboid) == 1
        assert scene.cuboid[0].dims[0] == pytest.approx(0.5, abs=0.01)
        assert scene.cuboid[0].dims[1] == pytest.approx(0.6, abs=0.01)
        assert scene.cuboid[0].dims[2] == pytest.approx(0.7, abs=0.01)

    def test_mixed_obstacles_roundtrip(self, tmp_path):
        """Test writing and reading mixed obstacles."""
        # Write
        writer = UsdWriter()
        stage_path = tmp_path / "mixed_roundtrip.usd"
        writer.create_stage(str(stage_path))

        obstacles = SceneCfg(
            cuboid=[
                Cuboid(name="cube", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
            ],
            sphere=[
                Sphere(name="sphere", pose=[2, 0, 0, 1, 0, 0, 0], radius=0.5)
            ],
            cylinder=[
                Cylinder(name="cyl", pose=[0, 2, 0, 1, 0, 0, 0], radius=0.3, height=1.0)
            ],
        )
        writer.add_world_to_stage(obstacles)
        writer.save()

        # Read with parser
        parser = UsdSceneParser()
        parser.load_stage_from_file(str(stage_path))
        scene = parser.get_obstacles_from_stage()

        assert len(scene.cuboid) == 1
        assert len(scene.sphere) == 1
        assert len(scene.cylinder) == 1

