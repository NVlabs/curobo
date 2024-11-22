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
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


@pytest.fixture(scope="module")
def motion_gen():
    world_file = "collision_table.yml"
    robot_file = "ur5e.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    motion_gen_instance.warmup(warmup_js_trajopt=True)
    return motion_gen_instance


def test_motion_gen_plan_js(motion_gen):
    retract_cfg = motion_gen.get_retract_config()

    start_state = JointState.from_position(retract_cfg.view(1, -1).clone())
    goal_state = JointState.from_position(retract_cfg.view(1, -1).clone())
    goal_state.position[:] = torch.as_tensor(
        [1.000, -2.2000, 1.9000, -1.3830, -1.5700, 0.0000], device=motion_gen.tensor_args.device
    )
    result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(max_attempts=1))
    assert result.success.item()


@pytest.mark.parametrize(
    "motion_gen_str, delta",
    [
        ("motion_gen", 0.1),
        ("motion_gen", 0.2),
        ("motion_gen", 0.3),
        ("motion_gen", 0.4),
        ("motion_gen", 0.5),
    ],
)
def test_motion_gen_plan_js_delta(motion_gen_str, delta, request):
    motion_gen = request.getfixturevalue(motion_gen_str)
    start_state = JointState.from_position(
        motion_gen.get_retract_config().view(1, -1).clone(),
        joint_names=motion_gen.joint_names,
    )
    goal_state = JointState.from_position(
        motion_gen.get_retract_config().view(1, -1).clone() + delta,
        joint_names=motion_gen.joint_names,
    )
    result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(max_attempts=1))
    assert result.success.item()
    assert result.cspace_error < 0.0001
