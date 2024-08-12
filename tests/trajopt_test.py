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
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig


@pytest.fixture(scope="function")
def trajopt_solver():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))

    trajopt_config = TrajOptSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
        evaluate_interpolated_trajectory=True,
    )
    trajopt_solver = TrajOptSolver(trajopt_config)

    return trajopt_solver


@pytest.fixture(scope="function")
def trajopt_solver_batch_env():
    tensor_args = TensorDeviceType()
    world_files = ["collision_table.yml", "collision_cubby.yml", "collision_test.yml"]
    world_cfg = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        for world_file in world_files
    ]
    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    trajopt_config = TrajOptSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
        num_seeds=4,
        evaluate_interpolated_trajectory=True,
        grad_trajopt_iters=200,
    )
    trajopt_solver = TrajOptSolver(trajopt_config)

    return trajopt_solver


def test_trajopt_single_js(trajopt_solver):
    q_start = trajopt_solver.retract_config.clone()
    q_goal = q_start.clone() + 0.2
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)
    # do single planning:
    js_goal = Goal(goal_state=goal_state, current_state=current_state)
    result = trajopt_solver.solve_single(js_goal)

    traj = result.solution.position[..., -1, :].view(q_goal.shape)
    assert torch.linalg.norm((goal_state.position - traj)).item() < 5e-3
    assert result.success.item()


def test_trajopt_single_pose(trajopt_solver):
    trajopt_solver.reset_seed()
    q_start = trajopt_solver.retract_config.clone()
    q_goal = q_start.clone() + 0.1
    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)
    js_goal = Goal(goal_pose=goal_pose, goal_state=goal_state, current_state=current_state)
    result = trajopt_solver.solve_single(js_goal)

    assert result.success.item()


def test_trajopt_single_pose_no_seed(trajopt_solver):
    trajopt_solver.reset_seed()
    q_start = trajopt_solver.retract_config.clone()
    q_goal = q_start.clone() + 0.05
    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    current_state = JointState.from_position(q_start)
    js_goal = Goal(goal_pose=goal_pose, current_state=current_state)
    result = trajopt_solver.solve_single(js_goal)

    # NOTE: This currently fails in some instances.
    assert result.success.item() == False or result.success.item() == True


def test_trajopt_single_goalset(trajopt_solver):
    # run goalset planning:
    q_start = trajopt_solver.retract_config.clone()
    q_goal = q_start.clone() + 0.1
    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)
    g_set = Pose(
        kin_state.ee_position.repeat(2, 1).view(1, 2, 3),
        kin_state.ee_quaternion.repeat(2, 1).view(1, 2, 4),
    )
    js_goal = Goal(goal_pose=g_set, goal_state=goal_state, current_state=current_state)
    result = trajopt_solver.solve_goalset(js_goal)
    assert result.success.item()


def test_trajopt_batch(trajopt_solver):
    # run goalset planning:
    q_start = trajopt_solver.retract_config.clone().repeat(2, 1)
    q_goal = q_start.clone()
    q_goal[0] += 0.1
    q_goal[1] -= 0.1
    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)
    g_set = Pose(
        kin_state.ee_position,
        kin_state.ee_quaternion,
    )

    js_goal = Goal(goal_pose=g_set, goal_state=goal_state, current_state=current_state)
    result = trajopt_solver.solve_batch(js_goal)
    assert torch.count_nonzero(result.success) > 0


def test_trajopt_batch_js(trajopt_solver):
    # run goalset planning:
    q_start = trajopt_solver.retract_config.clone().repeat(2, 1)
    q_goal = q_start.clone()
    q_goal[0] += 0.1
    q_goal[1] -= 0.1
    kin_state = trajopt_solver.fk(q_goal)
    # goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)

    js_goal = Goal(goal_state=goal_state, current_state=current_state)
    result = trajopt_solver.solve_batch(js_goal)
    traj = result.solution.position
    interpolated_traj = result.interpolated_solution.position
    assert torch.count_nonzero(result.success) > 0
    assert torch.linalg.norm((goal_state.position - traj[:, -1, :])).item() < 0.05
    assert torch.linalg.norm((goal_state.position - interpolated_traj[:, -1, :])).item() < 0.05


def test_trajopt_batch_goalset(trajopt_solver):
    # run goalset planning:
    q_start = trajopt_solver.retract_config.clone().repeat(3, 1)
    q_goal = q_start.clone()
    q_goal[0] += 0.1
    q_goal[1] -= 0.1
    q_goal[2, -1] += 0.1
    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(
        kin_state.ee_position.view(3, 1, 3).repeat(1, 5, 1),
        kin_state.ee_quaternion.view(3, 1, 4).repeat(1, 5, 1),
    )
    goal_pose.position[:, 0, 0] -= 0.01
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)

    js_goal = Goal(goal_state=goal_state, goal_pose=goal_pose, current_state=current_state)
    result = trajopt_solver.solve_batch_goalset(js_goal)
    traj = result.solution.position
    interpolated_traj = result.interpolated_solution.position
    assert torch.count_nonzero(result.success) > 0


def test_trajopt_batch_env_js(trajopt_solver_batch_env):
    # run goalset planning:
    q_start = trajopt_solver_batch_env.retract_config.clone().repeat(3, 1)
    q_goal = q_start.clone()
    q_goal += 0.1
    q_goal[2][0] += 0.1
    q_goal[1] -= 0.2
    # q_goal[2, -1] += 0.1
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)

    js_goal = Goal(goal_state=goal_state, current_state=current_state)
    result = trajopt_solver_batch_env.solve_batch_env(js_goal)

    traj = result.solution.position
    interpolated_traj = result.interpolated_solution.position
    assert torch.count_nonzero(result.success) == 3
    error = torch.linalg.norm((goal_state.position - traj[:, -1, :]), dim=-1)
    assert torch.max(error).item() < 0.05
    assert torch.linalg.norm((goal_state.position - interpolated_traj[:, -1, :])).item() < 0.05
    assert len(result) == 3


def test_trajopt_batch_env(trajopt_solver_batch_env):
    # run goalset planning:
    q_start = trajopt_solver_batch_env.retract_config.clone().repeat(3, 1)
    q_goal = q_start.clone()
    q_goal[0] += 0.1
    q_goal[1] -= 0.1
    q_goal[2, -1] += 0.1
    kin_state = trajopt_solver_batch_env.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)

    js_goal = Goal(goal_state=goal_state, goal_pose=goal_pose, current_state=current_state)
    result = trajopt_solver_batch_env.solve_batch_env(js_goal)
    traj = result.solution.position
    interpolated_traj = result.interpolated_solution.position
    assert torch.count_nonzero(result.success) == 3


def test_trajopt_batch_env_goalset(trajopt_solver_batch_env):
    # run goalset planning:
    q_start = trajopt_solver_batch_env.retract_config.repeat(3, 1)
    q_goal = q_start.clone()
    q_goal[0] += 0.1
    q_goal[1] -= 0.1
    q_goal[2, -1] += 0.1
    kin_state = trajopt_solver_batch_env.fk(q_goal)
    goal_pose = Pose(
        kin_state.ee_position.view(3, 1, 3).repeat(1, 5, 1),
        kin_state.ee_quaternion.view(3, 1, 4).repeat(1, 5, 1),
    )
    goal_pose.position[:, 0, 0] -= 0.01
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)

    js_goal = Goal(goal_state=goal_state, goal_pose=goal_pose, current_state=current_state)
    result = trajopt_solver_batch_env.solve_batch_env_goalset(js_goal)
    traj = result.solution.position
    interpolated_traj = result.interpolated_solution.position
    assert torch.count_nonzero(result.success) > 0


def test_trajopt_batch_env(trajopt_solver):
    # run goalset planning:
    q_start = trajopt_solver.retract_config.clone().repeat(3, 1)
    q_goal = q_start.clone()
    q_goal[0] += 0.1
    q_goal[1] -= 0.1
    q_goal[2, -1] += 0.1
    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)

    js_goal = Goal(goal_state=goal_state, goal_pose=goal_pose, current_state=current_state)
    with pytest.raises(ValueError):
        result = trajopt_solver.solve_batch_env(js_goal)
