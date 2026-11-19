# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import copy

import numpy as np
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


@pytest.mark.parametrize(
    "scales",
    [
        (0.5, 1.0, 1.0),
        (1.0, 0.5, 1.0),
        (1.0, 1.0, 0.5),
        (0.5, 0.4, 0.3),
        (
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6],
        ),
    ],
    ids=["velocity", "acceleration", "jerk", "combined", "per_joint"],
)
def test_cspace_limit_scales_apply_once(scales: tuple[float | list[float], ...]) -> None:
    """Apply scalar and per-joint derivative scales once, independently."""
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    base_limits = KinematicsLoader(
        KinematicsLoaderCfg(**copy.deepcopy(robot_params))
    ).kinematics_config.joint_limits

    scaled_params = copy.deepcopy(robot_params)
    for field, scale in zip(("velocity", "acceleration", "jerk"), scales):
        scaled_params["cspace"][field + "_scale"] = scale
    scaled_limits = KinematicsLoader(
        KinematicsLoaderCfg(**scaled_params)
    ).kinematics_config.joint_limits

    for field, scale in zip(("velocity", "acceleration", "jerk"), scales):
        base = getattr(base_limits, field)
        torch.testing.assert_close(getattr(scaled_limits, field), base * base.new_tensor(scale))
    torch.testing.assert_close(scaled_limits.position, base_limits.position)
    torch.testing.assert_close(scaled_limits.effort, base_limits.effort)


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


def test_collision_pruned_cspace_joints_use_default_lock_positions():
    robot_params = load_yaml(join_path(get_robot_configs_path(), "unitree_g1.yml"))[
        "kinematics"
    ]
    robot_params["load_collision_spheres"] = False
    default_positions = dict(
        zip(
            robot_params["cspace"]["joint_names"],
            robot_params["cspace"]["default_joint_position"],
            strict=True,
        )
    )
    expected_lock_names = [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
    ]

    robot_generator = KinematicsLoader(_cpu_loader_cfg(robot_params))

    assert robot_generator.total_spheres == 0
    assert robot_generator.lock_jointstate.joint_names == expected_lock_names
    assert torch.allclose(
        robot_generator.lock_jointstate.position,
        torch.tensor(
            [default_positions[name] for name in expected_lock_names],
            device=robot_generator.device_cfg.device,
        ),
    )


def test_collision_pruned_cspace_joint_without_default_requires_lock():
    robot_params = _load_simple_mimic_params()
    robot_params["tool_frames"] = ["chain_1_link_2"]
    robot_params["lock_joints"] = None
    robot_params["load_collision_spheres"] = False
    robot_params["cspace"]["default_joint_position"] = None

    with pytest.raises(ValueError, match="not explicitly locked"):
        KinematicsLoader(_cpu_loader_cfg(robot_params))


def test_collision_pruning_does_not_hide_unknown_cspace_joint():
    robot_params = _load_simple_mimic_params()
    robot_params["lock_joints"] = None
    robot_params["load_collision_spheres"] = False
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


def test_link_inertials_outside_the_tree_are_lumped_into_their_tree_link(lumped_inertial_robot):
    """Links the tree leaves out keep their mass: it is folded into the nearest tree link."""
    robot = lumped_inertial_robot
    kp = KinematicsLoader(
        _cpu_loader_cfg(
            {
                "urdf_path": str(robot.urdf_path),
                "base_link": "base_link",
                "tool_frames": ["arm_link"],
            }
        )
    ).kinematics_config
    names = kp.link_name_to_idx_map
    assert not {"camera_link", "mount_link", "bar_link"} & set(names)  # outside the tree
    mc = kp.link_masses_com[names["arm_link"]].double().numpy()
    six = kp.link_inertias[names["arm_link"], :6].double().numpy()
    assert mc[3] == pytest.approx(robot.mass, rel=1e-6)
    np.testing.assert_allclose(mc[:3], robot.com, atol=1e-6, err_msg="center of mass (link frame)")
    np.testing.assert_allclose(
        six,
        robot.inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
        atol=1e-6,
        err_msg="inertia about the COM, link axes [ixx, iyy, izz, ixy, ixz, iyz]",
    )
    # a tree link with nothing fixed to it keeps its own URDF values
    base = kp.link_masses_com[names["base_link"]].double().numpy()
    np.testing.assert_allclose(base, [0.0, 0.0, 0.0, 5.0], atol=1e-6)
