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
from curobo.cuda_robot_model.util import load_robot_yaml
from curobo.types.file_path import ContentPath


def test_xrdf_kinematics():
    robot_file = "ur10e.xrdf"
    urdf_file = "robot/ur_description/ur10e.urdf"
    content_path = ContentPath(robot_xrdf_file=robot_file, robot_urdf_file=urdf_file)
    robot_data = load_robot_yaml(content_path)
    robot_data["robot_cfg"]["kinematics"]["ee_link"] = "wrist_3_link"

    cfg = CudaRobotModelConfig.from_data_dict(robot_data["robot_cfg"]["kinematics"])

    robot = CudaRobotModel(cfg)
    q_test = robot.tensor_args.to_device([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, -1)

    kin_pose = robot.get_state(q_test)
    expected_position = robot.tensor_args.to_device([1.1842, 0.2907, 0.0609]).view(1, -1)

    error = kin_pose.ee_pose.position - expected_position
    assert error.norm() < 0.01
