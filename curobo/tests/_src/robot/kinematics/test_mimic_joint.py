# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState


def test_mimic_config():
    cfg = KinematicsCfg.from_robot_yaml_file("simple_mimic_robot.yml", ["ee_link"])

    robot_model = Kinematics(cfg)
    q = JointState.from_position(
        robot_model.default_joint_position.view(-1).clone(), joint_names=robot_model.joint_names
    )
    q = robot_model.get_full_js(q)

    q_mimic = robot_model.get_mimic_js(q)
    assert len(q_mimic.joint_names) == 3


def test_negative_axis_mimic_joint_offset_fk(tmp_path, cuda_device_cfg):
    """FK of a negative-axis mimic joint with a non-zero offset follows URDF mimic semantics."""
    multiplier, offset = -0.676, 0.149
    urdf_path = tmp_path / "neg_axis_mimic.urdf"
    urdf_path.write_text(
        f"""<?xml version="1.0"?>
<robot name="neg_axis_mimic">
  <link name="base_link"/>
  <link name="driver_link"/>
  <link name="tip_link"/>
  <link name="tip_tcp"/>
  <joint name="driver_joint" type="revolute">
    <parent link="base_link"/>
    <child link="driver_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="1"/>
  </joint>
  <joint name="tip_joint" type="revolute">
    <parent link="driver_link"/>
    <child link="tip_link"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-3.0" upper="3.0" effort="10" velocity="10"/>
    <mimic joint="driver_joint" multiplier="{multiplier}" offset="{offset}"/>
  </joint>
  <joint name="tcp_joint" type="fixed">
    <parent link="tip_link"/>
    <child link="tip_tcp"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""
    )
    kin_dict = {
        "urdf_path": str(urdf_path),
        "base_link": "base_link",
        "tool_frames": ["tip_tcp"],
        "cspace": {
            "joint_names": ["driver_joint"],
            "default_joint_position": [0.0],
            "null_space_weight": [1.0],
            "cspace_distance_weight": [1.0],
            "max_acceleration": 1.0,
            "max_jerk": 10.0,
        },
    }
    robot_model = Kinematics(KinematicsCfg.from_data_dict(kin_dict, device_cfg=cuda_device_cfg))

    q = cuda_device_cfg.to_device([[-0.8], [0.0], [0.5]])
    state = robot_model.compute_kinematics(
        JointState.from_position(q, joint_names=robot_model.joint_names)
    )
    position = state.tool_poses["tip_tcp"].position.view(-1, 3)

    # The tip joint value is multiplier * q + offset about -z, i.e. its negation about +z.
    q = q.view(-1)
    tip_angle = q - (multiplier * q + offset)
    expected = torch.stack(
        [
            0.05 * torch.cos(q) + 0.1 * torch.cos(tip_angle),
            0.05 * torch.sin(q) + 0.1 * torch.sin(tip_angle),
            torch.full_like(q, 0.1),
        ],
        dim=-1,
    )
    assert torch.allclose(position, expected, atol=1e-5)
