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
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig


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

    assert "link_names" not in robot_data["robot_cfg"]["kinematics"]


def test_xrdf_motion_gen():
    robot_file = "ur10e.xrdf"
    urdf_file = "robot/ur_description/ur10e.urdf"
    content_path = ContentPath(robot_xrdf_file=robot_file, robot_urdf_file=urdf_file)
    robot_data = load_robot_yaml(content_path)
    robot_data["robot_cfg"]["kinematics"]["ee_link"] = "wrist_3_link"

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_data,
        "collision_table.yml",
        use_cuda_graph=True,
        ee_link_name="tool0",
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(warmup_js_trajopt=False)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)
    result = motion_gen.plan_single(start_state, retract_pose)

    assert result.success.item()
