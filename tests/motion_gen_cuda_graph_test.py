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
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.trajectory import InterpolateType
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


@pytest.fixture(scope="function")
def motion_gen():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        use_cuda_graph=False,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    return motion_gen_instance


@pytest.fixture(scope="function")
def motion_gen_batch_env():
    tensor_args = TensorDeviceType()
    world_files = ["collision_table.yml", "collision_test.yml"]
    world_cfg = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        for world_file in world_files
    ]
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
    )
    motion_gen_instance = MotionGen(motion_gen_config)

    return motion_gen_instance


@pytest.mark.parametrize(
    "motion_gen_str,interpolation",
    [
        ("motion_gen", InterpolateType.LINEAR),
        ("motion_gen", InterpolateType.CUBIC),
        # ("motion_gen", InterpolateType.KUNZ_STILMAN_OPTIMAL),
        ("motion_gen", InterpolateType.LINEAR_CUDA),
    ],
)
def test_motion_gen_single(motion_gen_str, interpolation, request):
    motion_gen = request.getfixturevalue(motion_gen_str)
    motion_gen.update_interpolation_type(interpolation)
    motion_gen.warmup()

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    m_config = MotionGenPlanConfig(False, True)

    result = motion_gen.plan_single(start_state, goal_pose, m_config)

    # get final solutions:
    assert torch.count_nonzero(result.success) == 1
    reached_state = motion_gen.compute_kinematics(result.optimized_plan[-1])
    assert torch.norm(goal_pose.position - reached_state.ee_pos_seq) < 0.005


def test_motion_gen_goalset(motion_gen):
    motion_gen.warmup(n_goalset=2)

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(2, 1).view(1, -1, 3),
        quaternion=state.ee_quat_seq.repeat(2, 1).view(1, -1, 4),
    )
    goal_pose.position[0, 0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    m_config = MotionGenPlanConfig(False, True)

    result = motion_gen.plan_goalset(start_state, goal_pose, m_config)

    # get final solutions:
    assert torch.count_nonzero(result.success) == 1

    reached_state = motion_gen.compute_kinematics(result.optimized_plan[-1])

    assert (
        torch.min(
            torch.norm(goal_pose.position[:, 0, :] - reached_state.ee_pos_seq),
            torch.norm(goal_pose.position[:, 1, :] - reached_state.ee_pos_seq),
        )
        < 0.005
    )

    assert result.goalset_index is not None

    assert (
        torch.norm(goal_pose.position[:, result.goalset_index, :] - reached_state.ee_pos_seq)
        < 0.005
    )


def test_motion_gen_batch_goalset(motion_gen):
    motion_gen.warmup(n_goalset=3, batch=3, warmup_js_trajopt=False, enable_graph=False)
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.repeat(6, 1).view(3, -1, 3).clone(),
        quaternion=state.ee_quat_seq.repeat(6, 1).view(3, -1, 4).clone(),
    )
    goal_pose.position[0, 1, 1] = 0.2
    goal_pose.position[1, 0, 1] = 0.2
    goal_pose.position[2, 1, 1] = 0.2

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.2).repeat_seeds(3)

    m_config = MotionGenPlanConfig(False, True, max_attempts=1, enable_graph_attempt=None)
    result = motion_gen.plan_batch_goalset(start_state, goal_pose, m_config)

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


def test_motion_gen_batch(motion_gen):
    motion_gen.warmup(batch=2)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze()
    ).repeat_seeds(2)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3).repeat_seeds(2)

    goal_pose.position[1, 0] -= 0.1

    m_config = MotionGenPlanConfig(False, True)

    result = motion_gen.plan_batch(start_state, goal_pose.clone(), m_config)
    assert torch.count_nonzero(result.success) == 2

    # get final solutions:
    q = result.optimized_plan.trim_trajectory(-1).squeeze(1)
    reached_state = motion_gen.compute_kinematics(q)
    assert torch.norm(goal_pose.position - reached_state.ee_pos_seq) < 0.005


@pytest.mark.parametrize(
    "motion_gen_str,interpolation",
    [
        ("motion_gen", InterpolateType.LINEAR),
        ("motion_gen", InterpolateType.CUBIC),
        # ("motion_gen", InterpolateType.KUNZ_STILMAN_OPTIMAL),
        ("motion_gen", InterpolateType.LINEAR_CUDA),
    ],
)
def test_motion_gen_batch_graph(motion_gen_str: str, interpolation: InterpolateType, request):
    motion_gen = request.getfixturevalue(motion_gen_str)

    motion_gen.graph_planner.interpolation_type = interpolation
    motion_gen.reset()
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(
        state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze()
    ).repeat_seeds(5)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3).repeat_seeds(5)

    goal_pose.position[1, 0] -= 0.05

    m_config = MotionGenPlanConfig(True, False)

    result = motion_gen.plan_batch(start_state, goal_pose, m_config)
    assert torch.count_nonzero(result.success) > 0

    # get final solutions:
    q = result.interpolated_plan.trim_trajectory(-1).squeeze(1)
    reached_state = motion_gen.compute_kinematics(q)
    assert torch.norm(goal_pose.position - reached_state.ee_pos_seq) < 0.005


def test_motion_gen_batch_env(motion_gen_batch_env):
    motion_gen_batch_env.warmup(batch=2, batch_env_mode=True, enable_graph=False)

    # motion_gen_batch_env.reset()
    retract_cfg = motion_gen_batch_env.get_retract_config()
    state = motion_gen_batch_env.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    goal_pose = Pose(
        state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze()
    ).repeat_seeds(2)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3).repeat_seeds(2)

    goal_pose.position[1, 0] -= 0.1

    m_config = MotionGenPlanConfig(False, True, max_attempts=1)

    result = motion_gen_batch_env.plan_batch_env(start_state, goal_pose, m_config)
    assert torch.count_nonzero(result.success) == 2

    # get final solutions:
    reached_state = motion_gen_batch_env.compute_kinematics(
        result.optimized_plan.trim_trajectory(-1).squeeze(1)
    )
    assert torch.norm(goal_pose.position - reached_state.ee_pos_seq) < 0.005


def test_motion_gen_batch_env_goalset(motion_gen_batch_env):
    motion_gen_batch_env.warmup(batch=2, batch_env_mode=True, n_goalset=2, enable_graph=False)
    retract_cfg = motion_gen_batch_env.get_retract_config()
    state = motion_gen_batch_env.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    goal_pose = Pose(
        state.ee_pos_seq.repeat(4, 1).view(2, -1, 3),
        quaternion=state.ee_quat_seq.repeat(4, 1).view(2, -1, 4),
    )

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3).repeat_seeds(2)

    goal_pose.position[1, 0] -= 0.2

    m_config = MotionGenPlanConfig(False, True, enable_graph_attempt=None)

    result = motion_gen_batch_env.plan_batch_env_goalset(start_state, goal_pose, m_config)
    assert torch.count_nonzero(result.success) > 0

    # get final solutions:
    reached_state = motion_gen_batch_env.compute_kinematics(
        result.optimized_plan.trim_trajectory(-1).squeeze(1)
    )
    assert (
        torch.min(
            torch.norm(goal_pose.position[:, 0, :] - reached_state.ee_pos_seq),
            torch.norm(goal_pose.position[:, 1, :] - reached_state.ee_pos_seq),
        )
        < 0.005
    )

    goal_position = torch.cat(
        [
            goal_pose.position[x, result.goalset_index[x], :].unsqueeze(0)
            for x in range(len(result.goalset_index))
        ]
    )

    assert result.goalset_index is not None

    assert torch.max(torch.norm(goal_position - reached_state.ee_pos_seq, dim=-1)) < 0.005


@pytest.mark.parametrize(
    "motion_gen_str,enable_graph",
    [
        ("motion_gen", True),
        ("motion_gen", False),
    ],
)
def test_motion_gen_single_js(motion_gen_str, enable_graph, request):
    motion_gen = request.getfixturevalue(motion_gen_str)

    motion_gen.warmup(warmup_js_trajopt=True)

    retract_cfg = motion_gen.get_retract_config()

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    m_config = MotionGenPlanConfig(enable_graph=enable_graph, max_attempts=2)
    goal_state = start_state.clone()
    goal_state.position -= 0.3

    result = motion_gen.plan_single_js(start_state, goal_state, m_config)

    assert torch.count_nonzero(result.success) == 1

    reached_state = result.optimized_plan[-1]

    assert torch.norm(goal_state.position - reached_state.position) < 0.05
