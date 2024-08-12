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
import numpy as np
import pytest
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig, PoseCostMetric


@pytest.fixture(scope="module")
def mpc_single_env():
    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    world_file = "collision_test.yml"

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_file,
        use_cuda_graph=False,
        use_cuda_graph_metrics=False,
        use_cuda_graph_full_step=False,
    )
    mpc = MpcSolver(mpc_config)
    retract_cfg = robot_cfg.cspace.retract_config.view(1, -1)

    return [mpc, retract_cfg]


@pytest.fixture(scope="function")
def mpc_single_env_lbfgs():
    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    world_file = "collision_test.yml"

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_file,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        use_lbfgs=True,
        use_mppi=False,
        step_dt=0.5,
    )
    mpc = MpcSolver(mpc_config)
    retract_cfg = robot_cfg.cspace.retract_config.view(1, -1)

    return [mpc, retract_cfg]


@pytest.fixture(scope="module")
def mpc_batch_env():
    tensor_args = TensorDeviceType()

    world_files = ["collision_table.yml", "collision_test.yml"]
    world_cfg = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        for world_file in world_files
    ]

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=False,
        use_cuda_graph_metrics=False,
        use_cuda_graph_full_step=False,
    )
    mpc = MpcSolver(mpc_config)
    retract_cfg = robot_cfg.cspace.retract_config.view(1, -1)

    return [mpc, retract_cfg]


@pytest.mark.parametrize(
    "mpc_str, expected",
    [
        ("mpc_single_env", True),
        # ("mpc_single_env_lbfgs", True), unstable
    ],
)
def test_mpc_single(mpc_str, expected, request):
    mpc_val = request.getfixturevalue(mpc_str)
    mpc = mpc_val[0]
    retract_cfg = mpc_val[1]
    start_state = retract_cfg
    state = mpc.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg))
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=JointState.from_position(retract_cfg + 0.5),
        goal_state=JointState.from_position(retract_cfg),
        goal_pose=retract_pose,
    )
    goal_buffer = mpc.setup_solve_single(goal, 1)

    start_state = JointState.from_position(retract_cfg + 0.5, joint_names=mpc.joint_names)

    converged = False
    tstep = 0
    mpc.update_goal(goal_buffer)
    current_state = start_state.clone()
    while not converged:
        result = mpc.step(current_state, max_attempts=1)
        torch.cuda.synchronize()
        current_state.copy_(result.action)
        tstep += 1
        if result.metrics.pose_error.item() < 0.05:
            converged = True
            break
        if tstep > 200:
            break
    assert converged == expected


def test_mpc_goalset(mpc_single_env):
    mpc = mpc_single_env[0]
    retract_cfg = mpc_single_env[1]
    start_state = retract_cfg
    state = mpc.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg))
    retract_pose = Pose(
        state.ee_pos_seq.repeat(2, 1).unsqueeze(0),
        quaternion=state.ee_quat_seq.repeat(2, 1).unsqueeze(0),
    )
    goal = Goal(
        current_state=JointState.from_position(retract_cfg + 0.5),
        goal_state=JointState.from_position(retract_cfg),
        goal_pose=retract_pose,
    )
    goal_buffer = mpc.setup_solve_goalset(goal, 1)

    start_state = JointState.from_position(retract_cfg + 0.5, joint_names=mpc.joint_names)

    converged = False
    tstep = 0
    mpc.update_goal(goal_buffer)
    current_state = start_state.clone()
    while not converged:
        result = mpc.step(current_state)
        torch.cuda.synchronize()
        current_state.copy_(result.action)
        tstep += 1
        if result.metrics.pose_error.item() < 0.01:
            converged = True
            break
        if tstep > 1000:
            break
    assert converged


def test_mpc_batch(mpc_single_env):
    mpc = mpc_single_env[0]
    retract_cfg = mpc_single_env[1].repeat(2, 1)
    start_state = retract_cfg
    state = mpc.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg))
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    retract_pose.position[0, 0] -= 0.02
    goal = Goal(
        current_state=JointState.from_position(retract_cfg + 0.5),
        goal_state=JointState.from_position(retract_cfg),
        goal_pose=retract_pose,
    )
    goal_buffer = mpc.setup_solve_batch(goal, 1)

    start_state = JointState.from_position(retract_cfg + 0.5, joint_names=mpc.joint_names)

    converged = False
    tstep = 0
    mpc.update_goal(goal_buffer)
    current_state = start_state.clone()
    while not converged:
        result = mpc.step(current_state)
        torch.cuda.synchronize()
        current_state.copy_(result.action)
        tstep += 1
        if torch.max(result.metrics.pose_error) < 0.05:
            converged = True
            break
        if tstep > 1000:
            break
    assert converged


def test_mpc_batch_env(mpc_batch_env):
    mpc = mpc_batch_env[0]
    retract_cfg = mpc_batch_env[1].repeat(2, 1)
    start_state = retract_cfg
    state = mpc.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg))
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=JointState.from_position(retract_cfg + 0.5, joint_names=mpc.joint_names),
        goal_state=JointState.from_position(retract_cfg, joint_names=mpc.joint_names),
        goal_pose=retract_pose,
    )
    goal_buffer = mpc.setup_solve_batch_env(goal, 1)

    start_state = JointState.from_position(retract_cfg + 0.5, joint_names=mpc.joint_names)

    converged = False
    tstep = 0
    mpc.update_goal(goal_buffer)
    current_state = start_state.clone()
    while not converged:
        result = mpc.step(current_state)
        torch.cuda.synchronize()
        current_state.copy_(result.action)
        tstep += 1
        if torch.max(result.metrics.pose_error) < 0.05:
            converged = True
            break
        if tstep > 1000:
            break
    assert converged


@pytest.mark.parametrize(
    "mpc_str, expected",
    [
        ("mpc_single_env", True),
        # ("mpc_single_env_lbfgs", True), unstable
    ],
)
def test_mpc_single_pose_metric(mpc_str, expected, request):
    mpc_val = request.getfixturevalue(mpc_str)
    mpc = mpc_val[0]
    retract_cfg = mpc_val[1]
    start_state = retract_cfg
    state = mpc.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg))

    state = mpc.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()

    start_state = JointState.from_position(
        retract_cfg.view(1, -1) + 0.3, joint_names=mpc.joint_names
    )

    start_pose = mpc.compute_kinematics(start_state).ee_pose.clone()
    goal_pose.position = start_pose.position.clone()
    goal_pose.quaternion = start_pose.quaternion.clone()

    if mpc.project_pose_to_goal_frame:
        offset_pose = Pose.from_list([0, 0.1, 0, 1, 0, 0, 0])
        goal_pose = goal_pose.multiply(offset_pose)
    else:
        goal_pose.position[0, 1] += 0.2

    goal = Goal(
        current_state=JointState.from_position(retract_cfg + 0.5),
        goal_state=JointState.from_position(retract_cfg),
        goal_pose=goal_pose,
    )
    goal_buffer = mpc.setup_solve_single(goal, 1)

    converged = False
    tstep = 0
    mpc.update_goal(goal_buffer)
    current_state = start_state.clone()

    pose_cost_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=mpc.tensor_args.to_device([1, 1, 1, 1, 0, 1]),
    )
    mpc.update_pose_cost_metric(pose_cost_metric, start_state, goal_pose)
    while not converged:
        result = mpc.step(current_state, max_attempts=1)
        torch.cuda.synchronize()
        current_state.copy_(result.action)
        tstep += 1
        if result.metrics.pose_error.item() < 0.05:
            converged = True
            break
        if tstep > 200:
            break
    assert converged == expected
