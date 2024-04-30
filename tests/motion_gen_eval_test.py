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

# Third Party
import pytest
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.evaluator import TrajEvaluatorConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def run_motion_gen(robot_file, evaluate_interpolated_trajectory, max_acc, max_jerk):
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))
    dof = len(robot_data["robot_cfg"]["kinematics"]["cspace"]["joint_names"])

    robot_data["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"] = [
        max_acc for i in range(9)
    ]
    robot_data["robot_cfg"]["kinematics"]["cspace"]["max_jerk"] = [max_jerk for i in range(9)]

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_data,
        world_file,
        tensor_args,
        use_cuda_graph=False,
        maximum_trajectory_dt=1.5,
        evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
    )
    motion_gen = MotionGen(motion_gen_config)

    retract_cfg = motion_gen.get_retract_config()

    start_state = JointState.from_position(retract_cfg.view(1, -1).clone())
    goal_state = JointState.from_position(retract_cfg.view(1, -1).clone() + 0.2)

    result = motion_gen.plan_single_js(
        start_state, goal_state, MotionGenPlanConfig(max_attempts=5, enable_graph_attempt=10)
    )
    return result


@pytest.mark.parametrize(
    "robot_file, evaluate_interpolated_traj, max_acc, max_jerk",
    [
        ("franka.yml", False, 1.0, 500.0),
        ("franka.yml", True, 0.1, 500.0),
        ("franka.yml", True, 1.0, 500.0),
        ("ur5e.yml", False, 1.0, 500.0),
        ("ur5e.yml", True, 0.1, 500.0),
        ("ur5e.yml", True, 1.0, 500.0),
    ],
)
def test_motion_gen_trajectory(robot_file, evaluate_interpolated_traj, max_acc, max_jerk):
    result = run_motion_gen(robot_file, evaluate_interpolated_traj, max_acc, max_jerk)

    assert result.success.item()
    assert torch.max(torch.abs(result.optimized_plan.acceleration)) <= max_acc
    assert torch.max(torch.abs(result.optimized_plan.jerk)) <= max_jerk
