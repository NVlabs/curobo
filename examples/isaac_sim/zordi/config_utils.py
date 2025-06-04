"""
Copyright 2024 Zordi, Inc. All rights reserved.

Simple configuration utilities for managing paths in Zordi simulation examples.
Includes YAML loading with path resolution.
"""

import os
from typing import Any, Dict

import yaml


def get_zordi_assets_path() -> str:
    """Get the path to zordi_sim_assets from environment variable.

    Returns:
        str: Path to zordi_sim_assets directory

    Raises:
        RuntimeError: If ZORDI_SIM_ASSETS_PATH environment variable is not set
    """
    path = os.environ.get("ZORDI_SIM_ASSETS_PATH")
    if path is None:
        raise RuntimeError(
            "ZORDI_SIM_ASSETS_PATH environment variable is not set. "
            "Please set it to point to your zordi_sim_assets directory. "
            "Example: export ZORDI_SIM_ASSETS_PATH=/path/to/zordi_sim_assets"
        )
    # Remove trailing slash to avoid double slashes in path concatenation
    return path.rstrip("/")


def get_urdf_path() -> str:
    """Get path to XArm7 URDF file."""
    return os.path.join(
        get_zordi_assets_path(), "robot_resources", "urdf", "xarm7.urdf"
    )


def get_plant_usd_path() -> str:
    """Get path to plant USD file."""
    return os.path.join(
        get_zordi_assets_path(), "lightwheel", "Scene001_kinematics.usd"
    )


def get_plant_root() -> str:
    """Get path to plant assets directory."""
    return os.path.join(get_zordi_assets_path(), "lightwheel")


def get_robot_usd_path() -> str:
    """Get path to XArm7 USD file."""
    return os.path.join(
        get_zordi_assets_path(), "robot_resources", "usd", "xarm7", "xarm7.usd"
    )


def get_robot_assets_path() -> str:
    """Get path to robot assets directory."""
    return os.path.join(get_zordi_assets_path(), "robot_resources")


def resolve_zordi_paths_in_data(data: Any) -> Any:
    """Recursively replace ZORDI_PLACEHOLDER with actual path in data structure.

    Args:
        data: Dictionary, list, or other data structure that may contain placeholder strings

    Returns:
        Data structure with resolved paths
    """
    if isinstance(data, dict):
        return {key: resolve_zordi_paths_in_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [resolve_zordi_paths_in_data(item) for item in data]
    elif isinstance(data, str) and "ZORDI_PLACEHOLDER" in data:
        return data.replace("ZORDI_PLACEHOLDER", get_zordi_assets_path())
    else:
        return data


def load_yaml_with_zordi_paths(file_path: str) -> Dict[str, Any]:
    """Load YAML file and resolve ZORDI_PLACEHOLDER with actual paths.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with resolved paths
    """
    # Load the original YAML
    data = load_yaml_file(file_path)

    # Resolve any path placeholders
    resolved_data = resolve_zordi_paths_in_data(data)

    return resolved_data


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Load YAML file using standard Python yaml library."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_robot_config_with_zordi_paths(
    robot_config_file: str = "xarm7.yml",
) -> Dict[str, Any]:
    """Load robot configuration YAML with resolved Zordi paths.

    Args:
        robot_config_file: Name of the robot config file in the robot configs directory

    Returns:
        Dictionary with resolved robot configuration

    Example:
        robot_cfg = load_robot_config_with_zordi_paths("xarm7.yml")["robot_cfg"]
    """
    # Try to import curobo utilities, fall back to manual path construction
    try:
        from curobo.util_file import get_robot_configs_path, join_path

        robot_cfg_path = get_robot_configs_path()
        config_file_path = join_path(robot_cfg_path, robot_config_file)
    except ImportError:
        # Fallback for when curobo is not available (e.g., in Isaac Lab environment)
        # Construct the path manually relative to this file
        import pathlib

        this_file_dir = pathlib.Path(__file__).parent.absolute()
        config_file_path = (
            this_file_dir.parent.parent.parent
            / "src"
            / "curobo"
            / "content"
            / "configs"
            / "robot"
            / robot_config_file
        )
        config_file_path = str(config_file_path)

    # Load YAML file
    data = load_yaml_file(config_file_path)

    # Resolve paths
    return resolve_zordi_paths_in_data(data)
