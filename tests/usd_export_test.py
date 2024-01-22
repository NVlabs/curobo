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

try:
    # CuRobo
    from curobo.util.usd_helper import UsdHelper
except ImportError:
    pytest.skip("usd-core not found, skipping usd tests", allow_module_level=True)

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


@pytest.mark.skip(reason="Takes 60+ seconds and is not a core functionality")
def test_write_motion_gen_log(robot_file: str = "franka.yml"):
    # load motion generation with debug mode:
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    # robot_cfg["kinematics"]["collision_link_names"] = None
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_obb_world()

    c_cache = {"obb": 10}

    robot_cfg_instance = RobotConfig.from_dict(robot_cfg, tensor_args=TensorDeviceType())

    enable_debug = True
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_instance,
        world_cfg,
        collision_cache=c_cache,
        store_ik_debug=enable_debug,
        store_trajopt_debug=enable_debug,
    )
    mg = MotionGen(motion_gen_config)
    motion_gen = mg
    # generate a plan:
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    link_poses = state.link_pose

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1).clone() + 0.5)

    # get link poses if they exist:

    result = motion_gen.plan_single(
        start_state,
        retract_pose,
        link_poses=link_poses,
        plan_config=MotionGenPlanConfig(max_attempts=1, partial_ik_opt=False),
    )
    UsdHelper.write_motion_gen_log(
        result,
        robot_cfg,
        world_cfg,
        start_state,
        retract_pose,
        join_path("log/usd/", "debug"),
        write_robot_usd_path=join_path("log/usd/", "debug/assets/"),
        write_ik=True,
        write_trajopt=True,
        visualize_robot_spheres=False,
        grid_space=2,
        link_poses=link_poses,
    )
    assert True


def test_write_trajectory_usd(robot_file="franka.yml"):
    world_file = "collision_test.yml"
    world_model = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    ).get_obb_world()
    dt = 1 / 60.0
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=24,
        use_cuda_graph=True,
        num_trajopt_seeds=2,
        num_graph_seeds=2,
        evaluate_interpolated_trajectory=True,
        interpolation_dt=dt,
        self_collision_check=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1).clone() + 0.5)
    result = motion_gen.plan_single(start_state, retract_pose)
    q_traj = result.get_interpolated_plan()  # optimized plan
    if q_traj is not None:
        q_start = q_traj[0]

        UsdHelper.write_trajectory_animation_with_robot_usd(
            robot_file,
            world_model,
            q_start,
            q_traj,
            save_path="test.usda",
            robot_color=[0.5, 0.5, 0.2, 1.0],
            base_frame="/grid_world_1",
            dt=result.interpolation_dt,
        )
    else:
        assert False
