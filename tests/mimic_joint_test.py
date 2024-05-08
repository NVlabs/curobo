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

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.state import JointState


def test_mimic_config():
    cfg = CudaRobotModelConfig.from_robot_yaml_file("simple_mimic_robot.yml", "ee_link")

    # print(cfg.kinematics_config.fixed_transforms)
    robot_model = CudaRobotModel(cfg)
    q = JointState.from_position(robot_model.retract_config, joint_names=robot_model.joint_names)
    q = robot_model.get_full_js(q)

    # print(q)
    q_mimic = robot_model.get_mimic_js(q)
    assert len(q_mimic) == 3


def test_robotiq_mimic_config():
    cfg = CudaRobotModelConfig.from_robot_yaml_file("ur5e_robotiq_2f_140.yml", "grasp_frame")

    robot_model = CudaRobotModel(cfg)
    q = JointState.from_position(robot_model.retract_config, joint_names=robot_model.joint_names)
    q = robot_model.get_full_js(q)

    q_mimic = robot_model.get_mimic_js(q)
    assert len(q_mimic.joint_names) == 12
