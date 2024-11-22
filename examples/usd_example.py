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
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger, log_error
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def save_curobo_world_to_usd():
    world_file = "collision_table.yml"
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    ).get_mesh_world(process=False)
    usd_helper = UsdHelper()
    usd_helper.create_stage()

    usd_helper.add_obstacles_to_stage(world_cfg)

    usd_helper.write_stage_to_file("test.usd")


def get_trajectory(robot_file="franka.yml", dt=1.0 / 60.0, plan_grasp: bool = False):
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        num_trajopt_seeds=2,
        num_graph_seeds=2,
        evaluate_interpolated_trajectory=True,
        interpolation_dt=dt,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(n_goalset=2)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )
    if plan_grasp:
        retract_pose = Pose(
            state.ee_pos_seq.view(1, -1, 3), quaternion=state.ee_quat_seq.view(1, -1, 4)
        )
        start_state = JointState.from_position(retract_cfg.view(1, -1).clone())
        start_state.position[..., :-2] += 0.5
        m_config = MotionGenPlanConfig(False, True)

        result = motion_gen.plan_grasp(start_state, retract_pose, m_config.clone())
        if not result.success:
            log_error("Failed to plan grasp: " + result.status)
        traj = result.grasp_interpolated_trajectory
        traj2 = result.retract_interpolated_trajectory
        traj = traj.stack(traj2).clone()
        # result = motion_gen.plan_single(start_state, retract_pose)
        # traj = result.get_interpolated_plan()  # optimized plan

    else:
        retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
        start_state = JointState.from_position(retract_cfg.view(1, -1).clone())
        start_state.position[..., :-2] += 0.5
        result = motion_gen.plan_single(start_state, retract_pose)
        traj = result.get_interpolated_plan()  # optimized plan
    return traj


def save_curobo_robot_world_to_usd(robot_file="franka.yml", plan_grasp: bool = False):
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    world_model = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    ).get_obb_world()
    dt = 1 / 60.0

    q_traj = get_trajectory(robot_file, dt, plan_grasp)
    if q_traj is not None:
        q_start = q_traj[0]
        UsdHelper.write_trajectory_animation_with_robot_usd(
            robot_file,
            world_model,
            q_start,
            q_traj,
            save_path="test.usd",
            robot_color=[0.5, 0.5, 0.2, 1.0],
            base_frame="/grid_world_1",
            flatten_usd=True,
        )
    else:
        print("failed")


def save_curobo_robot_to_usd(robot_file="franka.yml"):
    # print(robot_file)
    tensor_args = TensorDeviceType()
    robot_cfg_y = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg_y["kinematics"]["use_usd_kinematics"] = True
    print(
        len(robot_cfg_y["kinematics"]["cspace"]["null_space_weight"]),
        len(robot_cfg_y["kinematics"]["cspace"]["retract_config"]),
        len(robot_cfg_y["kinematics"]["cspace"]["joint_names"]),
    )
    # print(robot_cfg_y)
    robot_cfg = RobotConfig.from_dict(robot_cfg_y, tensor_args)
    start = JointState.from_position(robot_cfg.cspace.retract_config)
    retract_cfg = robot_cfg.cspace.retract_config.clone()
    retract_cfg[0] = 0.5

    # print(retract_cfg)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    position = retract_cfg
    q_traj = JointState.from_position(position.unsqueeze(0))
    q_traj.joint_names = kin_model.joint_names
    # print(q_traj.joint_names)
    usd_helper = UsdHelper()
    # usd_helper.create_stage(
    #    "test.usd", timesteps=q_traj.position.shape[0] + 1, dt=dt, interpolation_steps=10
    # )
    world_file = "collision_table.yml"
    world_model = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    ).get_obb_world()

    # print(q_traj.position.shape)
    # usd_helper.load_robot_usd(robot_cfg.kinematics.usd_path, js)
    usd_helper.write_trajectory_animation_with_robot_usd(
        {"robot_cfg": robot_cfg_y},
        world_model,
        start,
        q_traj,
        save_path="test.usd",
        # robot_asset_prim_path="/robot"
    )

    # usd_helper.save()
    # usd_helper.write_stage_to_file("test.usda")


def read_world_from_usd(file_path: str):
    usd_helper = UsdHelper()
    usd_helper.load_stage_from_file(file_path)
    # world_model = usd_helper.get_obstacles_from_stage(reference_prim_path="/Root/world_obstacles")
    world_model = usd_helper.get_obstacles_from_stage(
        only_paths=["/world/obstacles"], reference_prim_path="/world"
    )
    # print(world_model)
    for x in world_model.cuboid:
        print(x.name + ":")
        print("  pose: ", x.pose)
        print("  dims: ", x.dims)


def read_robot_from_usd(robot_file: str = "franka.yml"):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg["kinematics"]["use_usd_kinematics"] = True
    robot_cfg = RobotConfig.from_dict(robot_cfg, TensorDeviceType())


def save_log_motion_gen(robot_file: str = "franka.yml"):
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
        # num_ik_seeds=2,
        # num_trajopt_seeds=1,
        # ik_opt_iters=20,
        # ik_particle_opt=False,
    )
    mg = MotionGen(motion_gen_config)
    # mg.warmup(enable_graph=True, warmup_js_trajopt=False)
    motion_gen = mg
    # generate a plan:
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )
    link_chain = motion_gen.kinematics.kinematics_config.link_chain_map[
        motion_gen.kinematics.kinematics_config.store_link_map.to(dtype=torch.long)
    ]

    # exit()
    link_poses = state.link_pose
    # print(link_poses)
    # del link_poses["tool0"]
    # del link_poses["tool1"]
    # del link_poses["tool2"]

    # del link_poses["tool3"]
    # print(link_poses)

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


if __name__ == "__main__":
    # save_curobo_world_to_usd()
    setup_curobo_logger("error")
    # save_log_motion_gen("franka.yml")
    save_curobo_robot_world_to_usd("ur5e_robotiq_2f_140.yml")
