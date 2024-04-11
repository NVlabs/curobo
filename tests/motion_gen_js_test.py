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
import torch

# CuRobo
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def test_motion_gen_plan_js():
    world_file = "collision_table.yml"
    robot_file = "ur5e.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        trajopt_tsteps=32,
        use_cuda_graph=False,
        num_trajopt_seeds=4,
        fixed_iters_trajopt=True,
        evaluate_interpolated_trajectory=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(warmup_js_trajopt=True)

    retract_cfg = motion_gen.get_retract_config()

    start_state = JointState.from_position(retract_cfg.view(1, -1).clone())
    goal_state = JointState.from_position(retract_cfg.view(1, -1).clone())
    goal_state.position[:] = torch.as_tensor(
        [1.000, -2.2000, 1.9000, -1.3830, -1.5700, 0.0000], device=motion_gen.tensor_args.device
    )
    result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(max_attempts=1))
    assert result.success.item()
