# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""This module contains a function to load robot representation from a yaml or xrdf file."""

# Standard Library

# CuRobo
from curobo._src.types.content_path import ContentPath
from curobo._src.util.logging import log_and_raise
from curobo._src.util.xrdf_util import convert_xrdf_to_curobo
from curobo._src.util_file import join_path, load_yaml


def load_robot_yaml(content_path: ContentPath = ContentPath()) -> dict:
    """Load robot representation from a yaml or xrdf file.

    Args:
        content_path: Path to the robot configuration files.

    Returns:
        dict: Robot representation as a dictionary.
    """
    if isinstance(content_path, str):
        log_and_raise("content_path should be of type ContentPath")

    robot_data = load_yaml(content_path.get_robot_configuration_path())

    if "format" in robot_data and robot_data["format"] == "xrdf":
        robot_data = convert_xrdf_to_curobo(
            content_path=content_path,
        )
        robot_data["robot_cfg"]["kinematics"]["asset_root_path"] = (
            content_path.robot_asset_absolute_path
        )

    if "robot_cfg" not in robot_data:
        robot_data["robot_cfg"] = robot_data
    if "kinematics" not in robot_data["robot_cfg"]:
        robot_data["robot_cfg"]["kinematics"] = robot_data
    if content_path.robot_urdf_absolute_path is not None:
        robot_data["robot_cfg"]["kinematics"]["urdf_path"] = content_path.robot_urdf_absolute_path
    if content_path.robot_asset_absolute_path is not None:
        robot_data["robot_cfg"]["kinematics"]["asset_root_path"] = (
            content_path.robot_asset_absolute_path
        )
    if isinstance(robot_data["robot_cfg"]["kinematics"]["collision_spheres"], str):
        robot_data["robot_cfg"]["kinematics"]["collision_spheres"] = join_path(
            content_path.robot_config_root_path,
            robot_data["robot_cfg"]["kinematics"]["collision_spheres"],
        )
    return robot_data
