# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.mark.parametrize("robot_file", ["franka.yml", "dual_ur10e.yml", "unitree_g1.yml"])
def test_kinematics_com(robot_file: str):
    device_cfg = DeviceCfg()
    # load a urdf:

    config_file = load_yaml(join_path(get_robot_configs_path(), robot_file))
    # base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    # ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    # urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    # robot_cfg = RobotCfg.from_basic(urdf_file, base_link, ee_link, device_cfg)

    robot_cfg = RobotCfg.create(
        config_file["robot_cfg"] if "robot_cfg" in config_file else config_file, device_cfg
    )

    kin_model = Kinematics(robot_cfg.kinematics, compute_com=True)

    q_test = torch.as_tensor(kin_model.default_joint_position, **(device_cfg.as_torch_dict())).view(1, -1)
    state = kin_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=kin_model.joint_names)
    )

    assert state.robot_com.shape == (1, 1, 4)


@pytest.mark.parametrize("robot_file", ["franka.yml", "dual_ur10e.yml", "unitree_g1.yml"])
def test_kinematics_com_gradient(robot_file: str):
    device_cfg = DeviceCfg(dtype=torch.float32)
    # load a urdf:

    config_file = load_yaml(join_path(get_robot_configs_path(), robot_file))

    robot_cfg = RobotCfg.create(
        config_file["robot_cfg"] if "robot_cfg" in config_file else config_file, device_cfg
    )

    kin_model = Kinematics(robot_cfg.kinematics, compute_com=True)

    q_test = torch.as_tensor(kin_model.default_joint_position, **(device_cfg.as_torch_dict())).view(1, -1)
    state = kin_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=kin_model.joint_names)
    )

    def com_loss(q):
        state = kin_model.compute_kinematics(
            JointState.from_position(q, joint_names=kin_model.joint_names)
        )
        com_loss = torch.sum(state.robot_com[..., :3] ** 2)
        return com_loss

    # use torch autograd check:
    q_test.requires_grad = True
    torch.autograd.gradcheck(com_loss, (q_test,), eps=1e-1, atol=1e-1)
