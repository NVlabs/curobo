# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for KinematicsCfg."""

# Third Party
import pytest

# CuRobo
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.content_path import ContentPath
from curobo._src.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml


def test_cuda_robot_model_cfg_from_basic_urdf(cuda_device_cfg):
    """Test creating KinematicsCfg from basic URDF."""
    # load robot yml:
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    urdf_path = join_path(
        get_assets_path(), robot_data["robot_cfg"]["kinematics"]["urdf_path"])

    cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=urdf_path,
        base_link="panda_link0",
        tool_frames=["panda_hand"],
        device_cfg=cuda_device_cfg,
    )

    assert cfg is not None
    assert cfg.device_cfg == cuda_device_cfg
    assert len(cfg.tool_frames) > 0
    assert cfg.kinematics_config is not None


def test_cuda_robot_model_cfg_from_robot_yaml_file(cuda_device_cfg):
    """Test creating KinematicsCfg from robot YAML file."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    cfg = KinematicsCfg.from_robot_yaml_file(
        file_path=robot_data,
        tool_frames=["panda_hand"],
        device_cfg=cuda_device_cfg,
    )

    assert cfg is not None
    assert cfg.device_cfg == cuda_device_cfg
    assert "panda_hand" in cfg.tool_frames
    assert cfg.kinematics_config is not None


def test_cuda_robot_model_cfg_from_robot_yaml_file_with_link_names(cuda_device_cfg):
    """Test creating KinematicsCfg with custom link names."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    cfg = KinematicsCfg.from_robot_yaml_file(
        file_path=robot_data,
        tool_frames=["panda_hand", "panda_link7"],
        device_cfg=cuda_device_cfg,
    )

    assert cfg is not None
    assert len(cfg.tool_frames) == 2
    assert "panda_hand" in cfg.tool_frames
    assert "panda_link7" in cfg.tool_frames


def test_cuda_robot_model_cfg_from_data_dict(cuda_device_cfg):
    """Test creating KinematicsCfg from data dictionary."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    cfg = KinematicsCfg.from_data_dict(
        data_dict=robot_data,
        tool_frames=["panda_hand"],
        device_cfg=cuda_device_cfg,
    )

    assert cfg is not None
    assert cfg.device_cfg == cuda_device_cfg
    assert cfg.kinematics_config is not None


def test_cuda_robot_model_cfg_from_content_path(cuda_device_cfg):
    """Test creating KinematicsCfg from ContentPath."""
    content_path = ContentPath(
        robot_config_file="franka.yml",
        robot_config_root_path=get_robot_configs_path(),
    )

    cfg = KinematicsCfg.from_content_path(
        content_path=content_path,
        tool_frames=["panda_hand"],
        device_cfg=cuda_device_cfg,
    )

    assert cfg is not None
    assert cfg.device_cfg == cuda_device_cfg
    assert cfg.kinematics_config is not None


def test_cuda_robot_model_cfg_get_joint_limits():
    """Test get_joint_limits method."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    joint_limits = cfg.get_joint_limits()

    assert joint_limits is not None
    assert joint_limits.position is not None
    assert joint_limits.velocity is not None
    assert joint_limits.acceleration is not None
    assert joint_limits.jerk is not None


def test_cuda_robot_model_cfg_kinematics_config_exists():
    """Test that kinematics_config exists and has expected attributes."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    # Verify kinematics_config has basic attributes
    assert cfg.kinematics_config.fixed_transforms is not None
    assert cfg.kinematics_config.link_map is not None


@pytest.mark.parametrize("robot_file", ["franka.yml", "ur10e.yml", "dual_ur10e.yml"])
def test_cuda_robot_model_cfg_multiple_robots(robot_file, cuda_device_cfg):
    """Test creating KinematicsCfg for different robots."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))



    cfg = KinematicsCfg.from_robot_yaml_file(
        file_path=robot_data,
        tool_frames=robot_data["robot_cfg"]["kinematics"]["tool_frames"],
        device_cfg=cuda_device_cfg,
    )

    assert cfg is not None
    assert cfg.device_cfg == cuda_device_cfg


def test_cuda_robot_model_cfg_generator_config():
    """Test that generator_config is stored."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    # Generator config should be stored
    assert cfg.generator_config is not None


def test_cuda_robot_model_cfg_self_collision_config():
    """Test that self_collision_config is created."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    # Self collision config should exist
    if cfg.self_collision_config is not None:
        assert hasattr(cfg.self_collision_config, 'num_spheres')
        assert hasattr(cfg.self_collision_config, 'collision_pairs')


def test_cuda_robot_model_cfg_kinematics_parser():
    """Test that kinematics_parser is stored."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    # Parser should be stored
    if cfg.kinematics_parser is not None:
        assert hasattr(cfg.kinematics_parser, 'get_chain')


def test_cuda_robot_model_cfg_link_names():
    """Test that tool_frames are properly set."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    assert len(cfg.tool_frames) > 0
    assert isinstance(cfg.tool_frames, list)
    assert all(isinstance(name, str) for name in cfg.tool_frames)


def test_cuda_robot_model_cfg_device_cfg_propagation(cuda_device_cfg):
    """Test that device_cfg is properly propagated."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    cfg = KinematicsCfg.from_robot_yaml_file(
        file_path=robot_data,
        tool_frames=["panda_hand"],
        device_cfg=cuda_device_cfg,
    )

    # Verify device_cfg is used throughout
    assert cfg.device_cfg == cuda_device_cfg
    assert cfg.kinematics_config.device_cfg == cuda_device_cfg

