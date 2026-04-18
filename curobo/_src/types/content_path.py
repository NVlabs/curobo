# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contains a class for storing file paths."""

# Standard Library
from dataclasses import dataclass
from typing import Optional

# CuRobo
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_scene_configs_path,
    join_path,
)


@dataclass(frozen=True)
class ContentPath:
    """Dataclass to store root path of configuration and assets."""

    #: Root path for robot configuration file, either xrdf or yml.
    robot_config_root_path: str = get_robot_configs_path()

    #: Root path for robot XRDF.
    robot_xrdf_root_path: str = get_robot_configs_path()

    #: Root path for robot URDF.
    robot_urdf_root_path: str = get_assets_path()

    #: Root path for robot meshes and textures.
    robot_asset_root_path: str = get_assets_path()

    #: Root path for world description files (yml).
    scene_config_root_path: str = get_scene_configs_path()

    #: Root path for world assets (meshes, nvblox maps).
    world_asset_root_path: str = get_assets_path()

    #: Absolute path to the robot configuration file. If this is provided, the
    #: :var:`robot_config_root_path`` will be ignored.
    robot_config_absolute_path: Optional[str] = None

    #: Absolute path to the robot XRDF file. If this is provided, the :var:`robot_xrdf_root_path`
    #: will be ignored.
    robot_xrdf_absolute_path: Optional[str] = None

    #: Absolute path to the robot URDF file. If this is provided, the :var:`robot_urdf_root_path`
    #: will be ignored.
    robot_urdf_absolute_path: Optional[str] = None

    #: Absolute path to the robot assets. If this is provided, the :var:`robot_asset_root_path`
    #: will be ignored.
    robot_asset_absolute_path: Optional[str] = None

    #: Absolute path to the world description file. If this is provided, the
    #: :var:`scene_config_root_path` will be ignored.
    scene_config_absolute_path: Optional[str] = None

    #: Relative path to the robot configuration file. If this is provided, the
    #: robot_config_absolute_path is initialized with the concatenation of
    #: robot_config_root_path and robot_config_file.
    robot_config_file: Optional[str] = None

    #: Relative path to the robot XRDF file. If this is provided, the
    #: robot_xrdf_absolute_path is initialized with the concatenation of
    #: robot_xrdf_root_path and robot_xrdf_file.
    robot_xrdf_file: Optional[str] = None

    #: Relative path to the robot URDF file. If this is provided, the
    #: robot_urdf_absolute_path is initialized with the concatenation of
    #: robot_urdf_root_path and robot_urdf_file.
    robot_urdf_file: Optional[str] = None

    #: Relative path to the robot assets.
    robot_asset_subroot_path: Optional[str] = None

    #: Relative path to the world description file. If this is provided, the
    #: scene_config_absolute_path is initialized with the concatenation of
    #: scene_config_root_path and scene_config_file.
    scene_config_file: Optional[str] = None

    def __post_init__(self):
        if self.robot_config_file is not None:
            if self.robot_config_absolute_path is not None:
                log_and_raise(
                    "robot_config_file and robot_config_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_config_absolute_path",
                join_path(self.robot_config_root_path, self.robot_config_file),
            )
        if self.robot_xrdf_file is not None:
            if self.robot_xrdf_absolute_path is not None:
                log_and_raise(
                    "robot_xrdf_file and robot_xrdf_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_xrdf_absolute_path",
                join_path(self.robot_xrdf_root_path, self.robot_xrdf_file),
            )
        if self.robot_urdf_file is not None:
            if self.robot_urdf_absolute_path is not None:
                log_and_raise(
                    "robot_urdf_file and robot_urdf_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_urdf_absolute_path",
                join_path(self.robot_urdf_root_path, self.robot_urdf_file),
            )
        if self.robot_asset_subroot_path is not None:
            if self.robot_asset_absolute_path is not None:
                log_and_raise(
                    "robot_asset_subroot_path and robot_asset_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_asset_absolute_path",
                join_path(self.robot_asset_root_path, self.robot_asset_subroot_path),
            )

        if self.scene_config_file is not None:
            if self.scene_config_absolute_path is not None:
                log_and_raise(
                    "scene_config_file and scene_config_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "scene_config_absolute_path",
                join_path(self.scene_config_root_path, self.scene_config_file),
            )

    def get_robot_configuration_path(self):
        """Get the robot configuration path."""
        if self.robot_config_absolute_path is None:
            log_info("cuRobo configuration file not found, trying XRDF")
            if self.robot_xrdf_absolute_path is None:
                log_and_raise("No Robot configuration file found")
            else:
                return self.robot_xrdf_absolute_path
        return self.robot_config_absolute_path
