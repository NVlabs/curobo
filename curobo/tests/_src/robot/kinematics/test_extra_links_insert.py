# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for inserting extra_links between existing links using child_link_name."""

# Standard Library
import os
import xml.etree.ElementTree as ET

# Third Party
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def _build_robot_with_virtual_base():
    """Load simple_mimic_robot with a virtual X-prismatic base joint inserted."""
    config = load_yaml(join_path(get_robot_configs_path(), "simple_mimic_robot.yml"))
    robot_data = config["robot_cfg"]
    robot_data.pop("version", None)

    robot_data["kinematics"]["extra_links"] = {
        "virtual_base": {
            "parent_link_name": "base_link",
            "child_link_name": "chain_1_link_1",
            "link_name": "virtual_base",
            "joint_name": "virtual_base_x",
            "joint_type": "X_PRISM",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": [-5.0, 5.0],
        },
    }
    robot_data["kinematics"]["cspace"]["joint_names"] = [
        "virtual_base_x",
        "chain_1_active_joint_1",
        "active_joint_2",
    ]
    robot_data["kinematics"]["cspace"]["default_joint_position"] = [0.0, 0.3, 0.0]
    robot_data["kinematics"]["cspace"]["null_space_weight"] = [1, 1, 1]
    robot_data["kinematics"]["cspace"]["cspace_distance_weight"] = [1, 1, 1]
    robot_data["kinematics"]["lock_joints"] = None

    device_cfg = DeviceCfg()
    robot_cfg = RobotCfg.create(robot_data, device_cfg)
    kin = Kinematics(robot_cfg.kinematics)
    return kin, device_cfg


def test_child_link_insert_fk():
    """Verify inserted prismatic joint translates the end-effector."""
    kin, device_cfg = _build_robot_with_virtual_base()

    assert "virtual_base_x" in kin.joint_names

    q_zero = torch.zeros(1, len(kin.joint_names), **device_cfg.as_torch_dict())
    state_zero = kin.compute_kinematics(JointState.from_position(q_zero, joint_names=kin.joint_names))
    ee_pos_zero = state_zero.tool_poses["ee_link"].position.squeeze().clone()

    q_shifted = q_zero.clone()
    q_shifted[0, 0] = 1.0
    state_shifted = kin.compute_kinematics(JointState.from_position(q_shifted, joint_names=kin.joint_names))
    ee_pos_shifted = state_shifted.tool_poses["ee_link"].position.squeeze().clone()

    delta = (ee_pos_shifted - ee_pos_zero).cpu()
    assert abs(delta[0].item() - 1.0) < 0.01, f"Expected ~1m X shift, got {delta[0].item()}"
    assert abs(delta[1].item()) < 0.01, f"Unexpected Y shift: {delta[1].item()}"
    assert abs(delta[2].item()) < 0.01, f"Unexpected Z shift: {delta[2].item()}"


def test_child_link_insert_exported_urdf(tmp_path):
    """Export URDF and verify the parent-child chain includes the inserted link."""
    kin, _ = _build_robot_with_virtual_base()

    urdf_path = str(tmp_path / "test_insert.urdf")
    kin_config = kin.kinematics_config
    kin_config.export_to_urdf(
        robot_name="test_insert",
        output_path=urdf_path,
        include_spheres=False,
    )

    assert os.path.exists(urdf_path)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    all_joints = root.findall("joint")
    links = {l.get("name") for l in root.findall("link")}

    assert "virtual_base" in links, f"virtual_base link missing. Links: {links}"

    joint_by_child = {j.find("child").get("link"): j for j in all_joints}

    vb_joint = joint_by_child["virtual_base"]
    assert vb_joint.get("name") == "virtual_base_x"
    assert vb_joint.find("parent").get("link") == "base_link"

    chain_joint = joint_by_child["chain_1_link_1"]
    assert chain_joint.find("parent").get("link") == "virtual_base", (
        f"chain_1_link_1's parent joint should come from virtual_base, "
        f"got {chain_joint.find('parent').get('link')}"
    )
