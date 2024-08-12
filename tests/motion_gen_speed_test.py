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


@pytest.fixture(scope="function")
def motion_gen(request):
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        velocity_scale=request.param,
        interpolation_steps=10000,
        interpolation_dt=0.02,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    return motion_gen_instance


@pytest.fixture(scope="module")
def motion_gen_ur5e():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "ur5e.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_steps=10000,
        interpolation_dt=0.05,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    motion_gen_instance.warmup(warmup_js_trajopt=False, enable_graph=False)

    return motion_gen_instance


@pytest.fixture(scope="module")
def motion_gen_ur5e_small_interpolation():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "ur5e.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        interpolation_steps=10,
        interpolation_dt=0.05,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    motion_gen_instance.warmup(warmup_js_trajopt=False, enable_graph=False)

    return motion_gen_instance


@pytest.mark.parametrize(
    "motion_gen",
    [
        (1.0),
        (0.75),
        (0.5),
        (0.25),
        (0.15),
        (0.1),
    ],
    indirect=True,
)
def test_motion_gen_velocity_scale(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

    goal_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)

    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)

    m_config = MotionGenPlanConfig(False, True, max_attempts=10)

    result = motion_gen.plan_single(start_state, goal_pose, m_config)

    assert torch.count_nonzero(result.success) == 1


@pytest.mark.parametrize(
    "velocity_scale, acceleration_scale",
    [
        (1.0, 1.0),
        (0.75, 1.0),
        (0.5, 1.0),
        (0.25, 1.0),
        (0.15, 1.0),
        (0.1, 1.0),
        (1.0, 0.1),
        (0.75, 0.1),
        (0.5, 0.1),
        (0.25, 0.1),
        (0.15, 0.1),
        (0.1, 0.1),
    ],
)
def test_pose_sequence_speed_ur5e_scale(velocity_scale, acceleration_scale):
    # load ur5e motion gen:

    world_file = "collision_table.yml"
    robot_file = "ur5e.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        interpolation_dt=(1.0 / 5.0),
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(warmup_js_trajopt=False, enable_graph=False)
    retract_cfg = motion_gen.get_retract_config()
    start_state = JointState.from_position(retract_cfg.view(1, -1))

    # poses for ur5e:
    home_pose = [-0.431, 0.172, 0.348, 0, 1, 0, 0]
    pose_1 = [0.157, -0.443, 0.427, 0, 1, 0, 0]
    pose_2 = [0.126, -0.443, 0.729, 0, 0, 1, 0]
    pose_3 = [-0.449, 0.339, 0.414, -0.681, -0.000, 0.000, 0.732]
    pose_4 = [-0.449, 0.339, 0.414, 0.288, 0.651, -0.626, -0.320]
    pose_5 = [-0.218, 0.508, 0.670, 0.529, 0.169, 0.254, 0.792]
    pose_6 = [-0.865, 0.001, 0.411, 0.286, 0.648, -0.628, -0.321]

    pose_list = [home_pose, pose_1, pose_2, pose_3, pose_4, pose_5, pose_6, home_pose]
    trajectory = start_state
    motion_time = 0
    fail = 0
    for i, pose in enumerate(pose_list):
        goal_pose = Pose.from_list(pose, q_xyzw=False)
        start_state = trajectory[-1].unsqueeze(0).clone()
        start_state.velocity[:] = 0.0
        start_state.acceleration[:] = 0.0
        result = motion_gen.plan_single(
            start_state.clone(),
            goal_pose,
            plan_config=MotionGenPlanConfig(
                max_attempts=5,
            ),
        )
        if result.success.item():
            plan = result.get_interpolated_plan()
            trajectory = trajectory.stack(plan.clone())
            motion_time += result.motion_time
        else:
            fail += 1
    assert fail == 0


@pytest.mark.parametrize(
    "motion_gen_str, time_dilation_factor",
    [
        ("motion_gen_ur5e", 1.0),
        ("motion_gen_ur5e", 0.75),
        ("motion_gen_ur5e", 0.5),
        ("motion_gen_ur5e", 0.25),
        ("motion_gen_ur5e", 0.15),
        ("motion_gen_ur5e", 0.1),
        ("motion_gen_ur5e", 0.001),
        ("motion_gen_ur5e_small_interpolation", 0.01),
    ],
)
def test_pose_sequence_speed_ur5e_time_dilation(motion_gen_str, time_dilation_factor, request):
    # load ur5e motion gen:
    motion_gen = request.getfixturevalue(motion_gen_str)

    retract_cfg = motion_gen.get_retract_config()
    start_state = JointState.from_position(retract_cfg.view(1, -1))

    # poses for ur5e:
    home_pose = [-0.431, 0.172, 0.348, 0, 1, 0, 0]
    pose_1 = [0.157, -0.443, 0.427, 0, 1, 0, 0]
    pose_2 = [0.126, -0.443, 0.729, 0, 0, 1, 0]
    pose_3 = [-0.449, 0.339, 0.414, -0.681, -0.000, 0.000, 0.732]
    pose_4 = [-0.449, 0.339, 0.414, 0.288, 0.651, -0.626, -0.320]
    pose_5 = [-0.218, 0.508, 0.670, 0.529, 0.169, 0.254, 0.792]
    pose_6 = [-0.865, 0.001, 0.411, 0.286, 0.648, -0.628, -0.321]

    pose_list = [home_pose, pose_1, pose_2, pose_3, pose_4, pose_5, pose_6, home_pose]
    trajectory = start_state
    motion_time = 0
    fail = 0
    for i, pose in enumerate(pose_list):
        goal_pose = Pose.from_list(pose, q_xyzw=False)
        start_state = trajectory[-1].unsqueeze(0).clone()
        start_state.velocity[:] = 0.0
        start_state.acceleration[:] = 0.0
        result = motion_gen.plan_single(
            start_state.clone(),
            goal_pose,
            plan_config=MotionGenPlanConfig(
                max_attempts=5,
                time_dilation_factor=time_dilation_factor,
            ),
        )
        if result.success.item():
            plan = result.get_interpolated_plan()
            augmented_js = motion_gen.get_full_js(plan)
            trajectory = trajectory.stack(plan.clone())
            motion_time += result.motion_time
        else:
            fail += 1
    assert fail == 0
    assert motion_time < 15 * (1 / time_dilation_factor)
