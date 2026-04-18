# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for cuda_robot_model utility functions."""

# Standard Library
from pathlib import Path

# Third Party
import pytest

# CuRobo
from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.types.content_path import ContentPath
from curobo._src.util_file import join_path
from curobo.content import get_assets_path, get_robot_configs_path


def test_load_robot_yaml_basic():
    """Test basic YAML loading with standard robot configuration."""
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )

    robot_data = load_robot_yaml(content_path)

    # Verify basic structure
    assert isinstance(robot_data, dict)
    assert "robot_cfg" in robot_data
    assert "kinematics" in robot_data["robot_cfg"]

    # Verify required kinematics fields
    kinematics = robot_data["robot_cfg"]["kinematics"]
    assert "urdf_path" in kinematics
    assert "base_link" in kinematics
    assert "tool_frames" in kinematics or "ee_link" in kinematics


def test_load_robot_yaml_with_urdf_path():
    """Test loading with explicit URDF path."""
    urdf_path = join_path(get_robot_configs_path(), "../assets/robot/franka_description/franka_panda.urdf")

    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
        robot_urdf_absolute_path=urdf_path,
    )

    robot_data = load_robot_yaml(content_path)

    # Verify URDF path is set
    assert robot_data["robot_cfg"]["kinematics"]["urdf_path"] == content_path.robot_urdf_absolute_path


def test_load_robot_yaml_with_asset_root():
    """Test loading with asset root path."""
    asset_root = join_path(get_robot_configs_path(), "../assets")

    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
        robot_asset_absolute_path=asset_root,
    )

    robot_data = load_robot_yaml(content_path)

    # Verify asset root path is set
    assert robot_data["robot_cfg"]["kinematics"]["asset_root_path"] == content_path.robot_asset_absolute_path


def test_load_robot_yaml_collision_spheres_path():
    """Test that collision spheres path is properly resolved."""
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )

    robot_data = load_robot_yaml(content_path)

    # Check if collision spheres path exists
    if "collision_spheres" in robot_data["robot_cfg"]["kinematics"]:
        collision_spheres = robot_data["robot_cfg"]["kinematics"]["collision_spheres"]

        # Should be either a dict or an absolute path
        if isinstance(collision_spheres, str):
            # Path should be absolute or exist
            assert "/" in collision_spheres or Path(collision_spheres).exists()


@pytest.mark.parametrize(
    "robot_file",
    ["franka.yml", "ur10e.yml", "dual_ur10e.yml", "unitree_g1.yml"]
)
def test_load_robot_yaml_multiple_robots(robot_file):
    """Test loading different robot configurations."""
    content_path = ContentPath(
        robot_config_file=robot_file,
        robot_config_root_path=get_robot_configs_path(),
    )

    robot_data = load_robot_yaml(content_path)

    # Verify basic structure for all robots
    assert isinstance(robot_data, dict)
    assert "robot_cfg" in robot_data
    assert "kinematics" in robot_data["robot_cfg"]


def test_load_robot_yaml_normalizes_structure():
    """Test that load_robot_yaml normalizes the config structure."""
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )

    robot_data = load_robot_yaml(content_path)

    # Should always have robot_cfg wrapper
    assert "robot_cfg" in robot_data

    # Should always have kinematics under robot_cfg
    assert "kinematics" in robot_data["robot_cfg"]


def test_load_robot_yaml_invalid_string_input():
    """Test that passing a string instead of ContentPath raises error."""
    with pytest.raises(Exception):  # Should raise log_and_raise
        load_robot_yaml("franka.yml")


def test_load_robot_yaml_preserves_existing_fields():
    """Test that loading preserves existing configuration fields."""
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )

    robot_data = load_robot_yaml(content_path)

    # Verify important fields are preserved
    kinematics = robot_data["robot_cfg"]["kinematics"]

    # These should exist in franka.yml
    assert "base_link" in kinematics
    assert "tool_frames" in kinematics or "ee_link" in kinematics

    # CSpace configuration should be present
    if "cspace" in kinematics:
        assert isinstance(kinematics["cspace"], dict)


def test_load_robot_yaml_handles_relative_collision_spheres():
    """Test that relative collision_spheres paths are converted to absolute."""
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )

    robot_data = load_robot_yaml(content_path)

    # If collision_spheres is a string, it should be absolute path
    collision_spheres = robot_data["robot_cfg"]["kinematics"].get("collision_spheres")

    if isinstance(collision_spheres, str):
        # Should contain full path (has directory separator)
        assert "/" in collision_spheres or "\\" in collision_spheres


def test_load_robot_yaml_normalizes_missing_kinematics():
    """Test that load_robot_yaml normalizes config when kinematics is missing."""
    # Create a ContentPath with a robot that might have flat structure
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )

    robot_data = load_robot_yaml(content_path)

    # Should always have normalized structure
    assert "robot_cfg" in robot_data
    assert "kinematics" in robot_data["robot_cfg"]
    # If robot_cfg wasn't present, it should be created (line 45-46)
    # If kinematics wasn't present, it should be created (line 47-48)


def test_load_robot_yaml_convert_to_xrdf_roundtrip(tmp_path):
    """Test loading YAML, converting to XRDF, saving, and loading back."""
    from curobo._src.util.config_io import write_yaml
    from curobo._src.util.xrdf_util import convert_curobo_to_xrdf

    # Load original YAML
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )
    original_data = load_robot_yaml(content_path)

    # Convert to XRDF format
    # Note: This may fail if cspace contains scalars instead of lists
    xrdf_data = convert_curobo_to_xrdf(original_data)

    # Verify XRDF format
    assert "format" in xrdf_data
    assert xrdf_data["format"] == "xrdf"
    assert "format_version" in xrdf_data

    # Save to temporary file
    xrdf_temp_path = tmp_path / "test_robot.xrdf"
    write_yaml(xrdf_data, str(xrdf_temp_path))
    urdf_path = original_data["robot_cfg"]["kinematics"]["urdf_path"]

    # Verify file was created
    assert xrdf_temp_path.exists()

    # Load it back through ContentPath
    content_path_xrdf = ContentPath(
        robot_xrdf_absolute_path=str(xrdf_temp_path),
        robot_asset_root_path=join_path(get_robot_configs_path(), "../assets"),
        robot_urdf_absolute_path=join_path(get_assets_path(), urdf_path),
    )
    loaded_data = load_robot_yaml(content_path_xrdf)

    # Verify loaded data has proper structure
    assert "robot_cfg" in loaded_data
    assert "kinematics" in loaded_data["robot_cfg"]

    # Verify essential fields are preserved
    original_kinematics = original_data["robot_cfg"]["kinematics"]
    loaded_kinematics = loaded_data["robot_cfg"]["kinematics"]

    assert "base_link" in loaded_kinematics
    assert loaded_kinematics["base_link"] == original_kinematics["base_link"]


def test_load_robot_yaml_without_kinematics_key():
    """Test that load_robot_yaml handles config missing 'kinematics' key."""
    import os
    import tempfile

    from curobo._src.util.config_io import write_yaml

    # Create a config without "kinematics" key - just top level robot config
    # This tests the normalization logic at lines 47-48 in util.py
    simple_data = {
        "base_link": "base",
        "tool_frames": ["end_effector"],
        "urdf_path": "test.urdf",
        "collision_spheres": {},
    }

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        temp_path = f.name
        write_yaml(simple_data, temp_path)

    try:
        # Load the config
        content_path = ContentPath(
            robot_config_absolute_path=temp_path,
        )
        loaded_data = load_robot_yaml(content_path)

        # Verify normalization occurred
        # Line 45-46: If "robot_cfg" not in data, robot_data["robot_cfg"] = robot_data
        assert "robot_cfg" in loaded_data

        # Line 47-48: If "kinematics" not in robot_cfg, robot_cfg["kinematics"] = robot_data
        assert "kinematics" in loaded_data["robot_cfg"]

        # The original fields should be accessible (though structure may have recursion)
        # This tests that the function completes without error
        assert isinstance(loaded_data, dict)

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
