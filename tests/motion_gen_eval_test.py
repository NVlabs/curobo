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
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.evaluator import TrajEvaluatorConfig

@pytest.fixture(scope="module", params=[True, False])
def evaluate_interpolated_trajectory(request):
    return request.param

@pytest.fixture(scope="module")
def motion_gen(evaluate_interpolated_trajectory):
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    robot_file = "franka.yml"
    dof = 9
    traj_evaluator_config = TrajEvaluatorConfig(
         max_acc=torch.ones((dof), device=tensor_args.device, dtype=tensor_args.dtype),
         max_jerk=torch.ones((dof), device=tensor_args.device, dtype=tensor_args.dtype),
         min_dt=torch.tensor(0.01, device=tensor_args.device, dtype=tensor_args.dtype),
         max_dt=torch.tensor(1.5, device=tensor_args.device, dtype=tensor_args.dtype)
    )
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=26,
        use_cuda_graph=False,
        num_trajopt_seeds=50,
        fixed_iters_trajopt=True,
        evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
        traj_evaluator_config=traj_evaluator_config,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(warmup_js_trajopt=True)

    retract_cfg = motion_gen.get_retract_config()

    start_state = JointState.from_position(retract_cfg.view(1, -1).clone())
    goal_state = JointState.from_position(retract_cfg.view(1, -1).clone())

    result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(max_attempts=1))

def test_motion_gen(motion_gen):
    return True