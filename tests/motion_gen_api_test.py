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
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


@pytest.fixture(scope="module")
def motion_gen():
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=26,
        use_cuda_graph=False,
        num_trajopt_seeds=50,
        fixed_iters_trajopt=True,
        evaluate_interpolated_trajectory=True,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    return motion_gen_instance


def test_motion_gen_attach_obstacle(motion_gen):
    obstacle = motion_gen.world_model.objects[-1].name
    retract_cfg = motion_gen.get_retract_config()

    start_state = JointState.from_position(retract_cfg.view(1, -1))
    motion_gen.attach_objects_to_robot(start_state, [obstacle])
    assert True


def test_motion_gen_attach_obstacle_offset(motion_gen):
    obstacle = motion_gen.world_model.objects[-1].name
    retract_cfg = motion_gen.get_retract_config()

    start_state = JointState.from_position(retract_cfg.view(1, -1))
    offset_pose = Pose.from_list([0, 0, 0.005, 1, 0, 0, 0], motion_gen.tensor_args)
    motion_gen.attach_objects_to_robot(
        start_state, [obstacle], world_objects_pose_offset=offset_pose
    )
    assert True


def test_motion_gen_lock_js_update():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka_mobile.yml"
    robot_config = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_config["kinematics"]["lock_joints"] = {"base_x": 0.0, "base_y": 0.0, "base_z": 0.0}
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_config,
        world_file,
        tensor_args,
        use_cuda_graph=True,
    )
    motion_gen_instance = MotionGen(motion_gen_config)
    motion_gen_instance.warmup()
    retract_cfg = motion_gen_instance.get_retract_config()
    start_state = JointState.from_position(retract_cfg.view(1, -1))

    kin_state = motion_gen_instance.compute_kinematics(start_state)
    ee_pose = kin_state.ee_pose.clone()
    # test motion gen:
    plan_start = start_state.clone()
    plan_start.position[..., :-2] += 0.1
    result = motion_gen_instance.plan_single(plan_start, ee_pose.clone())

    assert result.success.item()
    lock_js = {"base_x": 2.0, "base_y": 0.0, "base_z": 0.0}
    motion_gen_instance.update_locked_joints(lock_js, robot_config)

    kin_state_new = motion_gen_instance.compute_kinematics(start_state)
    ee_pose_shift = kin_state_new.ee_pose.clone()

    assert 2 - torch.norm(ee_pose.position[..., 0] - ee_pose_shift.position[..., 0]).item() <= 1e-5
    assert torch.norm(ee_pose.position[..., 1:] - ee_pose_shift.position[..., 1:]).item() == 0.0

    # test motion gen with new lock state:
    result = motion_gen_instance.plan_single(plan_start, ee_pose_shift.clone())
    assert result.success.item()
    result = motion_gen_instance.plan_single(
        plan_start, ee_pose.clone(), MotionGenPlanConfig(max_attempts=3)
    )
    assert result.success.item() == False
