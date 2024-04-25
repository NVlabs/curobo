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
