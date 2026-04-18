# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""CuRobo Content Module

Provides robot models, configurations, world assets, and weights bundled with CuRobo.
This module contains all content files needed for examples and typical use cases.
"""

from pathlib import Path
from typing import List

__all__ = [
    "get_content_root",
    "get_assets_path",
    "get_configs_path",
    "get_robot_configs_path",
    "get_task_configs_path",
    "get_scene_configs_path",
    "list_available_robots",
    "get_robot_path",
]


def get_content_root() -> Path:
    """Get the root directory of the content module.

    Returns:
        Path: Absolute path to the content directory.
    """
    return Path(__file__).parent


def get_assets_path() -> Path:
    """Get path to robot and scene assets (URDFs, meshes, etc.).

    Returns:
        Path: Path to assets/ directory.
    """
    return get_content_root() / "assets"


def get_configs_path() -> Path:
    """Get path to configuration files.

    Returns:
        Path: Path to configs/ directory.
    """
    return get_content_root() / "configs"


def get_robot_configs_path() -> Path:
    """Get path to robot configuration files.

    Returns:
        Path: Path to configs/robot/ directory.
    """
    return get_configs_path() / "robot"


def get_task_configs_path() -> Path:
    """Get path to task configuration files.

    Returns:
        Path: Path to configs/task/ directory.
    """
    return get_configs_path() / "task"


def get_scene_configs_path() -> Path:
    """Get path to scene configuration files.

    Returns:
        Path: Path to configs/scene/ directory.
    """
    return get_configs_path() / "scene"


def list_available_robots() -> List[str]:
    """List all available robot configuration files.

    Returns:
        List[str]: Names of available robots (without file extension).

    Example:
        >>> from curobo.content import list_available_robots
        >>> robots = list_available_robots()
        >>> print(robots)
        ['franka', 'ur5e', 'ur10e', ...]
    """
    robot_configs = get_robot_configs_path()
    if not robot_configs.exists():
        return []

    # Return .yml/.yaml files without extension
    configs = []
    for f in robot_configs.glob("*.y*ml"):
        configs.append(f.stem)
    return sorted(configs)


def get_robot_path(robot_name: str) -> Path:
    """Get path to a specific robot's assets.

    Args:
        robot_name: Name of the robot (e.g., 'franka', 'ur5').

    Returns:
        Path: Path to the robot's directory in assets/robot/.

    Raises:
        FileNotFoundError: If robot directory not found.

    Example:
        >>> from curobo.content import get_robot_path
        >>> franka_path = get_robot_path('franka')
        >>> print(franka_path)
        /path/to/curobo/content/assets/robot/franka
    """
    robot_path = get_robot_configs_path() / (robot_name + ".yml")
    if not robot_path.exists():
        available = list_available_robots()
        raise FileNotFoundError(f"Robot '{robot_name}' not found. Available robots: {available}")
    return robot_path


