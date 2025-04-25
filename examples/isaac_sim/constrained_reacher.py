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
# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
import numpy as np
import torch
from helper import add_extensions, add_robot_to_scene

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

simulation_app.update()
# Standard Library
import argparse

# Third Party
import carb
from omni.isaac.core import World

try:
    from omni.isaac.core.materials import OmniGlass, OmniPBR
except ImportError:
    from isaacsim.core.api.materials import OmniGlass, OmniPBR
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.usd_helper import UsdHelper

parser = argparse.ArgumentParser()


parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

args = parser.parse_args()


if __name__ == "__main__":
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    n_obstacle_cuboids = 10
    n_obstacle_mesh = 10

    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    target_material_2 = OmniPBR("/World/looks/t2", color=np.array([0, 1, 0]))
    target_material_plane = OmniGlass(
        "/World/looks/t3", color=np.array([0, 1, 0]), ior=1.25, depth=0.001, thin_walled=False
    )
    target_material_line = OmniGlass(
        "/World/looks/t4", color=np.array([0, 1, 0]), ior=1.25, depth=0.001, thin_walled=True
    )

    # target_orient = [0,0,0.707,0.707]
    target_orient = [0.5, -0.5, 0.5, 0.5]

    target = cuboid.VisualCuboid(
        "/World/target_1",
        position=np.array([0.55, -0.3, 0.5]),
        orientation=np.array(target_orient),
        size=0.04,
        visual_material=target_material,
    )

    # Make a target to follow
    target_2 = cuboid.VisualCuboid(
        "/World/target_2",
        position=np.array([0.55, 0.4, 0.5]),
        orientation=np.array(target_orient),
        size=0.04,
        visual_material=target_material_2,
    )

    x_plane = cuboid.VisualCuboid(
        "/World/constraint_plane",
        position=np.array([0.55, 0.05, 0.5]),
        orientation=np.array(target_orient),
        scale=[1.1, 0.001, 1.0],
        visual_material=target_material_plane,
    )
    xz_line = cuboid.VisualCuboid(
        "/World/constraint_line",
        position=np.array([0.55, 0.05, 0.5]),
        orientation=np.array(target_orient),
        scale=[0.04, 0.04, 0.65],
        visual_material=target_material_line,
    )

    collision_checker_type = CollisionCheckerType.BLOX

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world, "/World/world_robot/")

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )

    world_cfg_table.cuboid[0].pose[2] -= 0.01
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    usd_help = UsdHelper()

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg_table.get_mesh_world(), base_frame="/World")
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        interpolation_dt=0.02,
        ee_link_name="right_gripper",
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up..")
    motion_gen.warmup(warmup_js_trajopt=False)

    world_model = motion_gen.world_collision

    i = 0
    tensor_args = TensorDeviceType()
    target_list = [target, target_2]
    target_material_list = [target_material, target_material_2]
    for material in target_material_list:
        material.set_color(np.array([0.1, 0.1, 0.1]))
    target_material_plane.set_color(np.array([1, 1, 1]))
    target_material_line.set_color(np.array([1, 1, 1]))

    target_idx = 0
    cmd_idx = 0
    cmd_plan = None
    articulation_controller = robot.get_articulation_controller()
    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=2,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    plan_idx = 0
    cmd_step_idx = 0
    pose_cost_metric = None
    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue
        step_index = my_world.current_time_step_index

        if step_index <= 10:
            # my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if False and step_index % 50 == 0.0:  # No obstacle update
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            # print(len(obstacles.objects))

            motion_gen.update_world(obstacles)
            # print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        linear_color = np.ravel([249, 87, 56]) / 255.0
        orient_color = np.ravel([103, 148, 54]) / 255.0

        disable_color = np.ravel([255, 255, 255]) / 255.0

        if cmd_plan is None and step_index % 10 == 0:

            if plan_idx == 4:
                print("Constrained: Holding tool linear-y")
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device([0, 0, 0, 0, 1, 0]),
                )
                target_material_plane.set_color(linear_color)
            if plan_idx == 8:
                print("Constrained: Holding tool Orientation and linear-y")
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 0, 1, 0]),
                )
                target_material_plane.set_color(orient_color)
            if plan_idx == 12:
                print("Constrained: Holding tool linear-y, linear-x")
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device([0, 0, 0, 1, 1, 0]),
                )
                target_material_line.set_color(linear_color)
                target_material_plane.set_color(disable_color)

            if plan_idx == 16:
                print("Constrained: Holding tool Orientation and linear-y, linear-x")
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
                )
                target_material_line.set_color(orient_color)
                target_material_plane.set_color(disable_color)

            if plan_idx > 20:
                plan_idx = 0

            if plan_idx == 0:
                print("Constrained: Reset")
                target_material_line.set_color(disable_color)
                target_material_plane.set_color(disable_color)

                pose_cost_metric = None

            plan_config.pose_cost_metric = pose_cost_metric

            # motion generation:
            for ks in range(len(target_material_list)):
                if ks == target_idx:
                    target_material_list[ks].set_color(np.ravel([0, 1.0, 0]))
                else:
                    target_material_list[ks].set_color(np.ravel([0.1, 0.1, 0.1]))

            sim_js = robot.get_joints_state()
            sim_js_names = robot.dof_names
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

            cube_position, cube_orientation = target_list[target_idx].get_world_pose()

            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            plan_idx += 1
            if succ:
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0
                target_idx += 1
                if target_idx >= len(target_list):
                    target_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]

            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_step_idx += 1
            if cmd_step_idx == 2:
                cmd_idx += 1
                cmd_step_idx = 0
            # for _ in range(2):
            #    my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
    print("finished program")

    simulation_app.close()
