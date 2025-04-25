#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Example computing fk using curobo"""
# Third Party
import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_path, join_path, load_yaml


def demo_basic_robot():
    tensor_args = TensorDeviceType()

    # load a urdf:
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))

    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]  # Send global path starting with "/"
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    kin_model = CudaRobotModel(robot_cfg.kinematics)

    # compute forward kinematics:

    q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
    out = kin_model.get_state(q)
    # here is the kinematics state:
    # print(out)


def demo_full_config_robot():
    setup_curobo_logger("info")
    tensor_args = TensorDeviceType()
    # load a urdf:
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(config_file, tensor_args)

    kin_model = CudaRobotModel(robot_cfg.kinematics)

    # compute forward kinematics:
    q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
    out = kin_model.get_state(q)
    # here is the kinematics state:
    # print(out)


if __name__ == "__main__":
    demo_basic_robot()
    demo_full_config_robot()
