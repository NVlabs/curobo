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
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


@pytest.fixture(scope="module")
def motion_gen():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "ur5e_robotiq_2f_140.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        use_cuda_graph=True,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    return motion_gen_instance


@pytest.fixture(scope="function")
def motion_gen_batch():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        use_cuda_graph=True,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    return motion_gen_instance


def test_goalset_padded(motion_gen):
    # run goalset planning
    motion_gen.warmup(n_goalset=10)
    motion_gen.reset()
    m_config = MotionGenPlanConfig(False, True)

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(10, 1).view(1, -1, 3),
        quaternion=state.ee_quat_seq.repeat(10, 1).view(1, -1, 4),
    )
    goal_pose.position[0, 0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    result = motion_gen.plan_goalset(start_state, goal_pose, m_config.clone())

    assert torch.count_nonzero(result.success) == 1

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(2, 1).view(1, -1, 3),
        quaternion=state.ee_quat_seq.repeat(2, 1).view(1, -1, 4),
    )
    goal_pose.position[0, 0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    result = motion_gen.plan_goalset(start_state, goal_pose, m_config.clone())

    # run goalset with less goals:

    # run single planning

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())

    assert torch.count_nonzero(result.success) == 1


def test_batch_goalset_padded(motion_gen_batch):
    motion_gen = motion_gen_batch
    motion_gen.warmup(n_goalset=10, batch=3, enable_graph=False)
    # run goalset planning
    motion_gen.reset()

    retract_cfg = motion_gen.get_retract_config().clone()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(3 * 3, 1).view(3, -1, 3).contiguous(),
        quaternion=state.ee_quat_seq.repeat(3 * 3, 1).view(3, -1, 4).contiguous(),
    ).clone()
    goal_pose.position[0, 1, 1] = 0.2
    goal_pose.position[1, 0, 1] = 0.2
    goal_pose.position[2, 1, 1] = 0.2
    retract_cfg = motion_gen.get_retract_config().clone()

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.2).repeat_seeds(3)

    m_config = MotionGenPlanConfig(enable_graph_attempt=100, max_attempts=2)
    result = motion_gen.plan_batch_goalset(start_state, goal_pose.clone(), m_config.clone())

    # get final solutions:
    assert torch.count_nonzero(result.success) == result.success.shape[0]

    reached_state = motion_gen.compute_kinematics(
        result.optimized_plan.trim_trajectory(-1).squeeze(1)
    )

    #
    goal_position = torch.cat(
        [
            goal_pose.position[x, result.goalset_index[x], :].clone().unsqueeze(0)
            for x in range(len(result.goalset_index))
        ]
    )

    assert result.goalset_index is not None

    assert torch.max(torch.norm(goal_position - reached_state.ee_pos_seq, dim=-1)) < 0.005

    # run goalset with less goals:

    motion_gen.reset()

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(6, 1).view(3, -1, 3).contiguous(),
        quaternion=state.ee_quat_seq.repeat(6, 1).view(3, -1, 4).contiguous(),
    )
    goal_pose.position[0, 1, 1] = 0.2
    goal_pose.position[1, 0, 1] = 0.2
    goal_pose.position[2, 1, 1] = 0.2

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.2).repeat_seeds(3)

    result = motion_gen.plan_batch_goalset(start_state, goal_pose.clone(), m_config)

    # get final solutions:
    assert torch.count_nonzero(result.success) == result.success.shape[0]

    reached_state = motion_gen.compute_kinematics(result.optimized_plan.trim_trajectory(-1))

    #
    goal_position = torch.cat(
        [
            goal_pose.position[x, result.goalset_index[x], :].unsqueeze(0)
            for x in range(len(result.goalset_index))
        ]
    )

    assert result.goalset_index is not None

    assert torch.max(torch.norm(goal_position - reached_state.ee_pos_seq, dim=-1)) < 0.005
    # run single planning

    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze()
    ).repeat_seeds(3)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3).repeat_seeds(3)

    goal_pose.position[1, 0] -= 0.1

    result = motion_gen.plan_batch(start_state, goal_pose, m_config)
    assert torch.count_nonzero(result.success) == 3

    # get final solutions:
    q = result.optimized_plan.trim_trajectory(-1).squeeze(1)
    reached_state = motion_gen.compute_kinematics(q)
    assert torch.norm(goal_pose.position - reached_state.ee_pos_seq) < 0.005


def test_grasp_goalset(motion_gen):
    motion_gen.reset()
    m_config = MotionGenPlanConfig(False, True)

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(10, 1).view(1, -1, 3),
        quaternion=state.ee_quat_seq.repeat(10, 1).view(1, -1, 4),
    )
    goal_pose.position[0, 0, 0] += 0.2

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    result = motion_gen.plan_grasp(
        start_state,
        goal_pose,
        m_config.clone(),
        disable_collision_links=[
            "left_outer_knuckle",
            "left_inner_knuckle",
            "left_outer_finger",
            "left_inner_finger",
            "left_inner_finger_pad",
            "right_outer_knuckle",
            "right_inner_knuckle",
            "right_outer_finger",
            "right_inner_finger",
            "right_inner_finger_pad",
        ],
    )

    assert torch.count_nonzero(result.success) == 1
