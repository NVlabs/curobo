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


try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library

# Standard Library
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
args = parser.parse_args()
# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.kit import SimulationApp

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def main():
    usd_help = UsdHelper()
    act_distance = 0.2

    n_envs = 2
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    # Make a target to follow
    target_list = []
    target_material_list = []
    offset_y = 2.5
    radius = 0.1
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    robot_list = []

    for i in range(n_envs):
        if i > 0:
            pose.position[0, 1] += offset_y
        usd_help.add_subroot("/World", "/World/world_" + str(i), pose)

        target = cuboid.VisualCuboid(
            "/World/world_" + str(i) + "/target",
            position=np.array([0.5, 0, 0.5]) + pose.position[0].cpu().numpy(),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)
        r = add_robot_to_scene(
            robot_cfg,
            my_world,
            "/World/world_" + str(i) + "/",
            robot_name="robot_" + str(i),
            position=pose.position[0].cpu().numpy(),
            initialize_world=False,
        )
        robot_list.append(r[0])
    setup_curobo_logger("warn")
    my_world.initialize_physics()

    # warmup curobo instance

    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"

    world_file = ["collision_test.yml", "collision_thin_walls.yml"]
    world_cfg_list = []
    for i in range(n_envs):
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file[i]))
        )  # .get_mesh_world()
        world_cfg.objects[0].pose[2] -= 0.02
        world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
        usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
        world_cfg_list.append(world_cfg)

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg_list,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
        interpolation_dt=0.03,
        collision_cache={"obb": 10, "mesh": 10},
        collision_activation_distance=0.025,
        maximum_trajectory_dt=0.25,
    )
    motion_gen = MotionGen(motion_gen_config)
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    print("warming up...")
    # motion_gen.warmup(
    #     batch=n_envs,
    #     batch_env_mode=True,
    #     warmup_js_trajopt=False,
    # )

    add_extensions(simulation_app, args.headless_mode)
    config = RobotWorldConfig.load_from_config(
        robot_file, world_cfg_list, collision_activation_distance=act_distance
    )
    model = RobotWorld(config)
    i = 0
    max_distance = 0.5
    x_sph = torch.zeros((n_envs, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius
    env_query_idx = torch.arange(n_envs, device=tensor_args.device, dtype=torch.int32)
    plan_config = MotionGenPlanConfig(
        enable_graph=False, max_attempts=2, enable_finetune_trajopt=True
    )
    prev_goal = None
    cmd_plan = [None, None]
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    cmd_idx = 0
    past_goal = None
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        step_index = my_world.current_time_step_index

        if step_index <= 10:
            # my_world.reset()
            for robot in robot_list:
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)

                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                )
        if step_index < 20:
            continue
        sp_buffer = []
        sq_buffer = []
        for k in target_list:
            sph_position, sph_orientation = k.get_local_pose()
            sp_buffer.append(sph_position)
            sq_buffer.append(sph_orientation)

        ik_goal = Pose(
            position=tensor_args.to_device(sp_buffer),
            quaternion=tensor_args.to_device(sq_buffer),
        )
        if prev_goal is None:
            prev_goal = ik_goal.clone()
        if past_goal is None:
            past_goal = ik_goal.clone()
        sim_js_names = robot_list[0].dof_names
        sim_js = robot_list[0].get_joints_state()
        if sim_js is None:
            continue
        full_js = JointState(
            position=tensor_args.to_device(sim_js.positions).view(1, -1),
            velocity=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            joint_names=sim_js_names,
        )
        for i in range(1, len(robot_list)):
            sim_js = robot_list[i].get_joints_state()
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions).view(1, -1),
                velocity=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
                joint_names=sim_js_names,
            )
            full_js = full_js.stack(cu_js)

        prev_distance = ik_goal.distance(prev_goal)
        past_distance = ik_goal.distance(past_goal)

        if (
            (torch.sum(prev_distance[0] > 1e-2) or torch.sum(prev_distance[1] > 1e-2))
            and (torch.sum(past_distance[0]) == 0.0 and torch.sum(past_distance[1] == 0.0))
            and torch.max(torch.abs(full_js.velocity)) < 0.2
            and cmd_plan[0] is None
            and cmd_plan[1] is None
        ):
            full_js = full_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
            result = motion_gen.plan_batch_env(full_js, ik_goal, plan_config.clone())

            prev_goal.copy_(ik_goal)
            if torch.count_nonzero(result.success) > 0:
                trajs = result.get_paths()
                for s in range(len(result.success)):
                    if result.success[s]:
                        cmd_plan[s] = motion_gen.get_full_js(trajs[s])
                        # cmd_plan = result.get_interpolated_plan()
                        # cmd_plan = motion_gen.get_full_js(cmd_plan)
                        # get only joint names that are in both:
                        idx_list = []
                        common_js_names = []
                        for x in sim_js_names:
                            if x in cmd_plan[s].joint_names:
                                idx_list.append(robot_list[s].get_dof_index(x))
                                common_js_names.append(x)

                        cmd_plan[s] = cmd_plan[s].get_ordered_joint_state(common_js_names)

                    cmd_idx = 0
        # print(cmd_plan)

        for s in range(len(cmd_plan)):
            if cmd_plan[s] is not None and cmd_idx < len(cmd_plan[s].position):
                cmd_state = cmd_plan[s][cmd_idx]

                # get full dof state
                art_action = ArticulationAction(
                    cmd_state.position.cpu().numpy(),
                    cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                )
                # print(cmd_state.position)
                # set desired joint angles obtained from IK:
                art_controllers[s].apply_action(art_action)
            else:
                cmd_plan[s] = None
        cmd_idx += 1
        past_goal.copy_(ik_goal)

        for _ in range(2):
            my_world.step(render=False)


if __name__ == "__main__":
    main()
