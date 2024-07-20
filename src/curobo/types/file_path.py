#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""Contains a class for storing file paths."""
# Standard Library
from dataclasses import dataclass
from typing import Optional

# CuRobo
from curobo.util.logger import log_error, log_info
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_world_configs_path,
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

    #: Root path for robot USD.
    robot_usd_root_path: str = get_assets_path()

    #: Root path for robot meshes and textures.
    robot_asset_root_path: str = get_assets_path()

    #: Root path for world description files (yml).
    world_config_root_path: str = get_world_configs_path()

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

    #: Absolute path to the robot USD file. If this is provided, the :var:`robot_usd_root_path`
    #: will be ignored.
    robot_usd_absolute_path: Optional[str] = None

    #: Absolute path to the robot assets. If this is provided, the :var:`robot_asset_root_path`
    #: will be ignored.
    robot_asset_absolute_path: Optional[str] = None

    #: Absolute path to the world description file. If this is provided, the
    #: :var:`world_config_root_path` will be ignored.
    world_config_absolute_path: Optional[str] = None

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

    #: Relative path to the robot USD file. If this is provided, the
    #: robot_usd_absolute_path is initialized with the concatenation of
    #: robot_usd_root_path and robot_usd_file.
    robot_usd_file: Optional[str] = None

    #: Relative path to the robot assets.
    robot_asset_subroot_path: Optional[str] = None

    #: Relative path to the world description file. If this is provided, the
    #: world_config_absolute_path is initialized with the concatenation of
    #: world_config_root_path and world_config_file.
    world_config_file: Optional[str] = None

    def __post_init__(self):
        if self.robot_config_file is not None:
            if self.robot_config_absolute_path is not None:
                log_error(
                    "robot_config_file and robot_config_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_config_absolute_path",
                join_path(self.robot_config_root_path, self.robot_config_file),
            )
        if self.robot_xrdf_file is not None:
            if self.robot_xrdf_absolute_path is not None:
                log_error(
                    "robot_xrdf_file and robot_xrdf_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_xrdf_absolute_path",
                join_path(self.robot_xrdf_root_path, self.robot_xrdf_file),
            )
        if self.robot_urdf_file is not None:
            if self.robot_urdf_absolute_path is not None:
                log_error(
                    "robot_urdf_file and robot_urdf_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_urdf_absolute_path",
                join_path(self.robot_urdf_root_path, self.robot_urdf_file),
            )
        if self.robot_usd_file is not None:
            if self.robot_usd_absolute_path is not None:
                log_error("robot_usd_file and robot_usd_absolute_path cannot be provided together.")
            object.__setattr__(
                self,
                "robot_usd_absolute_path",
                join_path(self.robot_usd_root_path, self.robot_usd_file),
            )
        if self.robot_asset_subroot_path is not None:
            if self.robot_asset_absolute_path is not None:
                log_error(
                    "robot_asset_subroot_path and robot_asset_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "robot_asset_absolute_path",
                join_path(self.robot_asset_root_path, self.robot_asset_subroot_path),
            )

        if self.world_config_file is not None:
            if self.world_config_absolute_path is not None:
                log_error(
                    "world_config_file and world_config_absolute_path cannot be provided together."
                )
            object.__setattr__(
                self,
                "world_config_absolute_path",
                join_path(self.world_config_root_path, self.world_config_file),
            )

    def get_robot_configuration_path(self):
        """Get the robot configuration path."""
        if self.robot_config_absolute_path is None:
            log_info("cuRobo configuration file not found, trying XRDF")
            if self.robot_xrdf_absolute_path is None:
                log_error("No Robot configuration file found")
            else:
                return self.robot_xrdf_absolute_path
        return self.robot_config_absolute_path
