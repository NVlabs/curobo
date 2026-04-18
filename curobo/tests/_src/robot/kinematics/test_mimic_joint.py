# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

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
