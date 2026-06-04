# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import KinematicsCfg
from curobo._src.robot.loader.kinematics_loader import (
    KinematicsLoader,
    KinematicsLoaderCfg,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def _load_simple_mimic_params():
    return load_yaml(join_path(get_robot_configs_path(), "simple_mimic_robot.yml"))[
        "robot_cfg"
    ]["kinematics"]


def _add_cspace_joint(robot_params, joint_name, default_position=0.3):
    cspace = robot_params["cspace"]
    cspace["joint_names"].append(joint_name)
    cspace["default_joint_position"].append(default_position)
    cspace["null_space_weight"].append(1)
    cspace["cspace_distance_weight"].append(1)


def _remove_cspace_joint(robot_params, joint_name):
    cspace = robot_params["cspace"]
    joint_idx = cspace["joint_names"].index(joint_name)
    cspace["joint_names"].pop(joint_idx)
    for key in ["default_joint_position", "null_space_weight", "cspace_distance_weight"]:
        if cspace.get(key) is not None:
            cspace[key].pop(joint_idx)


def _cpu_loader_cfg(robot_params):
    return KinematicsLoaderCfg(
        **robot_params, device_cfg=DeviceCfg(device=torch.device("cpu"))
    )


def _identity_link_poses(loader, q, query_link_names, kinematics_config):
    position = torch.zeros(
        (1, len(query_link_names), 3),
        device=loader.device_cfg.device,
        dtype=loader.device_cfg.dtype,
    )
    quaternion = torch.zeros(
        (1, len(query_link_names), 4),
        device=loader.device_cfg.device,
        dtype=loader.device_cfg.dtype,
    )
    quaternion[..., 0] = 1.0
    return Pose(position=position, quaternion=quaternion)


def test_cuda_robot_generator_config():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = KinematicsLoaderCfg(**robot_params)
    # Test that tool_frames is set
    assert config.tool_frames is not None
    assert len(config.tool_frames) > 0


def test_cuda_robot_generator():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = KinematicsLoaderCfg(**robot_params)
    robot_generator = KinematicsLoader(config)
    assert robot_generator.kinematics_config.num_dof == 7


def test_cuda_robot_config():
    robot_file = "franka.yml"
    config = KinematicsCfg.from_robot_yaml_file(robot_file)
    assert config.kinematics_config.num_dof == 7


def test_cuda_robot_generator_config_cspace():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = KinematicsLoaderCfg(**robot_params)
    assert len(config.cspace.max_jerk) == len(config.cspace.joint_names)
    robot_generator = KinematicsLoader(config)

    assert len(robot_generator.cspace.max_jerk) == 7


def test_cspace_only_configured_lock_joint_uses_configured_value():
    robot_params = _load_simple_mimic_params()
    robot_params["tool_frames"] = ["chain_1_link_2"]
    robot_params["lock_joints"] = {"active_joint_2": 0.7}
    robot_params["cspace"]["default_joint_position"] = None

    robot_generator = KinematicsLoader(_cpu_loader_cfg(robot_params))

    assert robot_generator.lock_jointstate.joint_names == ["active_joint_2"]
    assert torch.allclose(
        robot_generator.lock_jointstate.position,
        torch.tensor([0.7], device=robot_generator.device_cfg.device),
    )
    assert robot_generator.non_fixed_joint_names == [
        "chain_1_active_joint_1",
        "active_joint_2",
    ]
    assert "active_joint_2" not in robot_generator.joint_names
    assert "active_joint_2" not in robot_generator.cspace.joint_names


def test_configured_lock_joint_must_resolve_to_tree_or_cspace():
    robot_params = _load_simple_mimic_params()
    robot_params["lock_joints"] = {"missing_joint": 0.1}

    with pytest.raises(ValueError, match="not found in the kinematic tree or cspace"):
        KinematicsLoader(_cpu_loader_cfg(robot_params))


def test_cspace_only_name_must_be_configured_lock():
    robot_params = _load_simple_mimic_params()
    robot_params["lock_joints"] = None
    _add_cspace_joint(robot_params, "joint_typo")

    with pytest.raises(ValueError, match="not active in the configured tree"):
        KinematicsLoader(_cpu_loader_cfg(robot_params))


def test_non_parser_cspace_lock_joint_is_rejected():
    robot_params = _load_simple_mimic_params()
    robot_params["lock_joints"] = {"joint_typo": 0.7}
    _add_cspace_joint(robot_params, "joint_typo")

    with pytest.raises(ValueError, match="not parser actuated joints"):
        KinematicsLoader(_cpu_loader_cfg(robot_params))


def test_active_tree_joint_missing_from_cspace_raises():
    robot_params = _load_simple_mimic_params()
    robot_params["lock_joints"] = None
    _remove_cspace_joint(robot_params, "active_joint_2")

    with pytest.raises(ValueError, match="cspace is missing active tree joints"):
        KinematicsLoader(_cpu_loader_cfg(robot_params))


def test_duplicate_cspace_joint_names_raise():
    robot_params = _load_simple_mimic_params()
    robot_params["lock_joints"] = None
    _add_cspace_joint(robot_params, "active_joint_2")

    with pytest.raises(ValueError, match="duplicate joint names"):
        KinematicsLoader(_cpu_loader_cfg(robot_params))


def test_mimic_lock_joint_name_points_to_active_joint():
    robot_params = _load_simple_mimic_params()
    robot_params["lock_joints"] = {"chain_1_mimic_joint_2": 0.1}

    with pytest.raises(ValueError, match="lock the active joint instead"):
        KinematicsLoader(_cpu_loader_cfg(robot_params))


def test_cspace_lock_updates_non_fixed_names_when_tree_lock_exists(monkeypatch):
    robot_params = _load_simple_mimic_params()
    robot_params["tool_frames"] = ["chain_1_link_2"]
    robot_params["lock_joints"] = {
        "chain_1_active_joint_1": 0.2,
        "active_joint_2": 0.7,
    }
    monkeypatch.setattr(KinematicsLoader, "_get_link_poses", _identity_link_poses)

    robot_generator = KinematicsLoader(_cpu_loader_cfg(robot_params))

    assert robot_generator.lock_jointstate.joint_names == [
        "chain_1_active_joint_1",
        "active_joint_2",
    ]
    assert robot_generator.non_fixed_joint_names == [
        "chain_1_active_joint_1",
        "active_joint_2",
    ]
    assert "active_joint_2" not in robot_generator.joint_names
