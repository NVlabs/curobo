# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ContentPath."""

# Third Party
import pytest

# CuRobo
from curobo._src.types.content_path import ContentPath
from curobo._src.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_scene_configs_path,
    join_path,
)


class TestContentPath:
    """Test ContentPath class."""

    def test_default_initialization(self):
        """Test default ContentPath initialization."""
        cp = ContentPath()
        assert cp.robot_config_root_path == get_robot_configs_path()
        assert cp.robot_xrdf_root_path == get_robot_configs_path()
        assert cp.robot_urdf_root_path == get_assets_path()
        assert cp.robot_asset_root_path == get_assets_path()
        assert cp.scene_config_root_path == get_scene_configs_path()
        assert cp.world_asset_root_path == get_assets_path()

    def test_initialization_with_custom_paths(self):
        """Test ContentPath with custom paths."""
        cp = ContentPath(
            robot_config_root_path="/custom/robot/config",
            robot_urdf_root_path="/custom/urdf",
            scene_config_root_path="/custom/scene",
        )
        assert cp.robot_config_root_path == "/custom/robot/config"
        assert cp.robot_urdf_root_path == "/custom/urdf"
        assert cp.scene_config_root_path == "/custom/scene"

    def test_robot_config_file_sets_absolute_path(self):
        """Test that robot_config_file sets robot_config_absolute_path."""
        cp = ContentPath(robot_config_file="franka.yml")
        expected_path = join_path(get_robot_configs_path(), "franka.yml")
        assert cp.robot_config_absolute_path == expected_path

    def test_robot_config_file_with_custom_root(self):
        """Test robot_config_file with custom root path."""
        custom_root = "/custom/robot/configs"
        cp = ContentPath(robot_config_root_path=custom_root, robot_config_file="ur10.yml")
        expected_path = join_path(custom_root, "ur10.yml")
        assert cp.robot_config_absolute_path == expected_path

    def test_robot_config_file_and_absolute_path_conflict(self):
        """Test that providing both robot_config_file and absolute_path raises error."""
        with pytest.raises(Exception):  # log_and_raise throws exception
            ContentPath(
                robot_config_file="franka.yml", robot_config_absolute_path="/absolute/path.yml"
            )

    def test_robot_xrdf_file_sets_absolute_path(self):
        """Test that robot_xrdf_file sets robot_xrdf_absolute_path."""
        cp = ContentPath(robot_xrdf_file="robot.xrdf")
        expected_path = join_path(get_robot_configs_path(), "robot.xrdf")
        assert cp.robot_xrdf_absolute_path == expected_path

    def test_robot_xrdf_file_with_custom_root(self):
        """Test robot_xrdf_file with custom root path."""
        custom_root = "/custom/xrdf"
        cp = ContentPath(robot_xrdf_root_path=custom_root, robot_xrdf_file="robot.xrdf")
        expected_path = join_path(custom_root, "robot.xrdf")
        assert cp.robot_xrdf_absolute_path == expected_path

    def test_robot_xrdf_file_and_absolute_path_conflict(self):
        """Test that providing both robot_xrdf_file and absolute_path raises error."""
        with pytest.raises(Exception):
            ContentPath(
                robot_xrdf_file="robot.xrdf", robot_xrdf_absolute_path="/absolute/robot.xrdf"
            )

    def test_robot_urdf_file_sets_absolute_path(self):
        """Test that robot_urdf_file sets robot_urdf_absolute_path."""
        cp = ContentPath(robot_urdf_file="robot.urdf")
        expected_path = join_path(get_assets_path(), "robot.urdf")
        assert cp.robot_urdf_absolute_path == expected_path

    def test_robot_urdf_file_with_custom_root(self):
        """Test robot_urdf_file with custom root path."""
        custom_root = "/custom/urdf"
        cp = ContentPath(robot_urdf_root_path=custom_root, robot_urdf_file="robot.urdf")
        expected_path = join_path(custom_root, "robot.urdf")
        assert cp.robot_urdf_absolute_path == expected_path

    def test_robot_urdf_file_and_absolute_path_conflict(self):
        """Test that providing both robot_urdf_file and absolute_path raises error."""
        with pytest.raises(Exception):
            ContentPath(
                robot_urdf_file="robot.urdf", robot_urdf_absolute_path="/absolute/robot.urdf"
            )

    def test_robot_asset_subroot_path_sets_absolute_path(self):
        """Test that robot_asset_subroot_path sets robot_asset_absolute_path."""
        cp = ContentPath(robot_asset_subroot_path="franka")
        expected_path = join_path(get_assets_path(), "franka")
        assert cp.robot_asset_absolute_path == expected_path

    def test_robot_asset_subroot_with_custom_root(self):
        """Test robot_asset_subroot_path with custom root path."""
        custom_root = "/custom/assets"
        cp = ContentPath(robot_asset_root_path=custom_root, robot_asset_subroot_path="franka")
        expected_path = join_path(custom_root, "franka")
        assert cp.robot_asset_absolute_path == expected_path

    def test_robot_asset_subroot_and_absolute_path_conflict(self):
        """Test that providing both robot_asset_subroot and absolute_path raises error."""
        with pytest.raises(Exception):
            ContentPath(
                robot_asset_subroot_path="franka", robot_asset_absolute_path="/absolute/franka"
            )

    def test_scene_config_file_sets_absolute_path(self):
        """Test that scene_config_file sets scene_config_absolute_path."""
        cp = ContentPath(scene_config_file="scene.yml")
        expected_path = join_path(get_scene_configs_path(), "scene.yml")
        assert cp.scene_config_absolute_path == expected_path

    def test_scene_config_file_with_custom_root(self):
        """Test scene_config_file with custom root path."""
        custom_root = "/custom/scenes"
        cp = ContentPath(scene_config_root_path=custom_root, scene_config_file="scene.yml")
        expected_path = join_path(custom_root, "scene.yml")
        assert cp.scene_config_absolute_path == expected_path

    def test_scene_config_file_and_absolute_path_conflict(self):
        """Test that providing both scene_config_file and absolute_path raises error."""
        with pytest.raises(Exception):
            ContentPath(
                scene_config_file="scene.yml", scene_config_absolute_path="/absolute/scene.yml"
            )

    def test_get_robot_configuration_path_with_config(self):
        """Test get_robot_configuration_path when robot_config_absolute_path is set."""
        cp = ContentPath(robot_config_absolute_path="/path/to/config.yml")
        assert cp.get_robot_configuration_path() == "/path/to/config.yml"

    def test_get_robot_configuration_path_with_xrdf(self):
        """Test get_robot_configuration_path falls back to XRDF."""
        cp = ContentPath(robot_xrdf_absolute_path="/path/to/robot.xrdf")
        assert cp.get_robot_configuration_path() == "/path/to/robot.xrdf"

    def test_get_robot_configuration_path_with_neither(self):
        """Test get_robot_configuration_path raises error when no config found."""
        cp = ContentPath()
        with pytest.raises(Exception):  # log_and_raise
            cp.get_robot_configuration_path()

    def test_get_robot_configuration_path_prefers_config_over_xrdf(self):
        """Test that robot_config_absolute_path is preferred over xrdf."""
        cp = ContentPath(
            robot_config_absolute_path="/path/to/config.yml",
            robot_xrdf_absolute_path="/path/to/robot.xrdf",
        )
        assert cp.get_robot_configuration_path() == "/path/to/config.yml"

    def test_absolute_paths_take_priority(self):
        """Test that absolute paths override file paths."""
        cp = ContentPath(robot_config_absolute_path="/absolute/config.yml")
        assert cp.robot_config_absolute_path == "/absolute/config.yml"

    def test_frozen_dataclass(self):
        """Test that ContentPath is frozen."""
        cp = ContentPath()
        with pytest.raises(Exception):  # FrozenInstanceError
            cp.robot_config_root_path = "/new/path"

    def test_multiple_file_paths_simultaneously(self):
        """Test setting multiple file paths at once."""
        cp = ContentPath(
            robot_config_file="config.yml",
            robot_urdf_file="robot.urdf",
            scene_config_file="scene.yml",
        )
        assert cp.robot_config_absolute_path is not None
        assert cp.robot_urdf_absolute_path is not None
        assert cp.scene_config_absolute_path is not None

    def test_world_asset_root_path(self):
        """Test world_asset_root_path default."""
        cp = ContentPath()
        assert cp.world_asset_root_path == get_assets_path()

    def test_custom_world_asset_root_path(self):
        """Test custom world_asset_root_path."""
        custom_path = "/custom/world/assets"
        cp = ContentPath(world_asset_root_path=custom_path)
        assert cp.world_asset_root_path == custom_path

