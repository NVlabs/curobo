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


# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig


def test_motion_gen_mpc():
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    robot_file = "ur5e.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        use_cuda_graph=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )
    motion_gen.warmup(warmup_js_trajopt=False)

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.5)

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    world_file = "collision_test.yml"

    if True:
        new_motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            world_file,
            tensor_args,
            use_cuda_graph=True,
        )
        new_motion_gen = MotionGen(new_motion_gen_config)
    new_motion_gen.warmup()

    if True:
        mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_file,
            world_file,
            use_cuda_graph=True,
            use_cuda_graph_metrics=False,
            use_cuda_graph_full_step=False,
        )
        mpc = MpcSolver(mpc_config)
        retract_cfg = robot_cfg.cspace.retract_config.view(1, -1)

    result = motion_gen.plan(start_state, retract_pose, enable_graph=False)
    assert result.success
    result = new_motion_gen.plan(start_state, retract_pose, enable_graph=False)
    assert result.success
