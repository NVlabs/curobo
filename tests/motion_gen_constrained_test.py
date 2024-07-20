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
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenStatus,
    PoseCostMetric,
)


@pytest.fixture(scope="function")
def motion_gen(request):
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        use_cuda_graph=True,
        project_pose_to_goal_frame=request.param[0],
    )
    motion_gen_instance = MotionGen(motion_gen_config)

    motion_gen_instance.warmup(
        enable_graph=False, warmup_js_trajopt=False, n_goalset=request.param[1]
    )
    return motion_gen_instance


@pytest.mark.parametrize(
    "motion_gen",
    [
        ([True, -1]),
        ([False, -1]),
        ([True, 10]),
        ([False, 10]),
    ],
    indirect=True,
)
def test_approach_grasp_pose(motion_gen):
    # run full pose planning
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()
    goal_pose.position[0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    m_config = MotionGenPlanConfig(max_attempts=1)

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())

    assert torch.count_nonzero(result.success) == 1

    # run grasp pose planning:
    m_config.pose_cost_metric = PoseCostMetric.create_grasp_approach_metric()
    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())

    assert torch.count_nonzero(result.success) == 1


@pytest.mark.parametrize(
    "motion_gen",
    [
        ([True, -1]),
        ([False, -1]),
        ([True, 10]),
        ([False, 10]),
    ],
    indirect=True,
)
def test_reach_only_position(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()
    goal_pose.position[0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)
    m_config = MotionGenPlanConfig(
        max_attempts=1,
        pose_cost_metric=PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=motion_gen.tensor_args.to_device([0, 0, 0, 1, 1, 1]),
        ),
    )

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())
    assert result.status != MotionGenStatus.INVALID_PARTIAL_POSE_COST_METRIC
    assert torch.count_nonzero(result.success) == 1

    reached_state = result.optimized_plan[-1]
    reached_pose = motion_gen.compute_kinematics(reached_state).ee_pose.clone()
    assert goal_pose.angular_distance(reached_pose) > 0.0
    assert goal_pose.linear_distance(reached_pose) <= 0.005


@pytest.mark.parametrize(
    "motion_gen",
    [
        ([True, -1]),
        ([False, -1]),
        ([True, 10]),
        ([False, 10]),
    ],
    indirect=True,
)
def test_reach_only_orientation(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()
    goal_pose.position[0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)
    m_config = MotionGenPlanConfig(
        max_attempts=1,
        pose_cost_metric=PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 0, 0, 0]),
        ),
    )

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())

    assert torch.count_nonzero(result.success) == 1

    reached_state = result.optimized_plan[-1]
    reached_pose = motion_gen.compute_kinematics(reached_state).ee_pose.clone()
    assert goal_pose.linear_distance(reached_pose) > 0.0
    assert goal_pose.angular_distance(reached_pose) < 0.05


@pytest.mark.parametrize(
    "motion_gen",
    [
        ([True, -1]),
        ([False, -1]),
        ([True, 10]),
        ([False, 10]),
    ],
    indirect=True,
)
def test_hold_orientation(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()
    goal_pose.position[0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    start_pose = motion_gen.compute_kinematics(start_state).ee_pose.clone()
    goal_pose.quaternion = start_pose.quaternion.clone()
    m_config = MotionGenPlanConfig(
        max_attempts=1,
        pose_cost_metric=PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 0, 0, 0]),
        ),
    )

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())

    assert torch.count_nonzero(result.success) == 1

    traj_pose = motion_gen.compute_kinematics(result.optimized_plan).ee_pose.clone()

    # assert goal_pose.linear_distance(traj_pose) > 0.0
    goal_pose = goal_pose.repeat(traj_pose.shape[0])
    assert torch.max(goal_pose.angular_distance(traj_pose)) < 0.05


@pytest.mark.parametrize(
    "motion_gen",
    [
        ([True, -1]),
        ([False, -1]),
        ([True, 10]),
        ([False, 10]),
    ],
    indirect=True,
)
def test_hold_position(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()
    goal_pose.position[0, 0] -= 0.1

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    start_pose = motion_gen.compute_kinematics(start_state).ee_pose.clone()
    goal_pose.position = start_pose.position.clone()
    m_config = MotionGenPlanConfig(
        max_attempts=1,
        pose_cost_metric=PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=motion_gen.tensor_args.to_device([0, 0, 0, 1, 1, 1]),
        ),
    )

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())

    assert torch.count_nonzero(result.success) == 1

    traj_pose = motion_gen.compute_kinematics(result.optimized_plan).ee_pose.clone()

    goal_pose = goal_pose.repeat(traj_pose.shape[0])
    assert torch.max(goal_pose.linear_distance(traj_pose)) < 0.005


@pytest.mark.parametrize(
    "motion_gen",
    [
        ([True, -1]),
        ([False, -1]),
        ([True, 10]),
        ([False, 10]),
    ],
    indirect=True,
)
def test_hold_partial_pose(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    start_pose = motion_gen.compute_kinematics(start_state).ee_pose.clone()
    goal_pose.position = start_pose.position.clone()
    goal_pose.quaternion = start_pose.quaternion.clone()

    if motion_gen.project_pose_to_goal_frame:
        offset_pose = Pose.from_list([0, 0.1, 0, 1, 0, 0, 0])
        goal_pose = goal_pose.multiply(offset_pose)
    else:
        goal_pose.position[0, 1] += 0.2

    m_config = MotionGenPlanConfig(
        max_attempts=1,
        pose_cost_metric=PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 1, 0, 1]),
        ),
    )

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())
    assert torch.count_nonzero(result.success) == 1

    traj_pose = motion_gen.compute_kinematics(result.optimized_plan).ee_pose.clone()

    goal_pose = goal_pose.repeat(traj_pose.shape[0])
    if motion_gen.project_pose_to_goal_frame:
        traj_pose = goal_pose.compute_local_pose(traj_pose)
        traj_pose.position[:, 1] = 0.0
        assert torch.max(traj_pose.position) < 0.005

    else:
        goal_pose.position[:, 1] = 0.0
        traj_pose.position[:, 1] = 0.0
        assert torch.max(goal_pose.linear_distance(traj_pose)) < 0.005


@pytest.mark.parametrize(
    "motion_gen",
    [
        ([True, -1]),
        ([False, -1]),
        ([True, 10]),
        ([False, 10]),
    ],
    indirect=True,
)
def test_hold_partial_pose_fail(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = state.ee_pose.clone()

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    start_pose = motion_gen.compute_kinematics(start_state).ee_pose.clone()
    goal_pose.position = start_pose.position.clone()
    goal_pose.quaternion = start_pose.quaternion.clone()

    if motion_gen.project_pose_to_goal_frame:
        offset_pose = Pose.from_list([0, 0.1, 0.1, 1, 0, 0, 0])
        goal_pose = goal_pose.multiply(offset_pose)
    else:
        goal_pose.position[0, 1] += 0.2
        goal_pose.position[0, 0] += 0.2

    m_config = MotionGenPlanConfig(
        max_attempts=1,
        pose_cost_metric=PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 1, 0, 1]),
        ),
    )

    result = motion_gen.plan_single(start_state, goal_pose, m_config.clone())
    assert torch.count_nonzero(result.success) == 0
    assert result.valid_query == False
