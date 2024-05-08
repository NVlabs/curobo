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

# CuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


@pytest.mark.parametrize(
    "parallel_finetune, force_graph, expected_motion_time",
    [
        (True, False, 12),
        (True, True, 12),
    ],
)
def test_pose_sequence_ur5e(parallel_finetune, force_graph, expected_motion_time):
    # load ur5e motion gen:

    world_file = "collision_table.yml"
    robot_file = "ur5e.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        interpolation_dt=(1 / 30),
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(parallel_finetune=parallel_finetune)
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
                parallel_finetune=parallel_finetune, max_attempts=10, enable_graph=force_graph
            ),
        )
        if result.success.item():
            plan = result.get_interpolated_plan()
            trajectory = trajectory.stack(plan.clone())
            motion_time += result.motion_time
        else:
            fail += 1
    assert fail == 0
    assert motion_time <= expected_motion_time
