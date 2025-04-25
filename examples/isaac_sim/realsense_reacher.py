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
import cv2
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
from matplotlib import cm
from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

simulation_app.update()
# Standard Library
import argparse

# Third Party
import carb
from helper import VoxelManager, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.usd_helper import UsdHelper

parser = argparse.ArgumentParser()

parser.add_argument(
    "--show-window",
    action="store_true",
    help="When True, shows camera image in a CV window",
    default=False,
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--use-debug-draw",
    action="store_true",
    help="When True, sets robot in static mode",
    default=False,
)
args = parser.parse_args()


def draw_points(voxels):
    # Third Party

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_points()
    if len(voxels) == 0:
        return

    jet = cm.get_cmap("plasma").reversed()

    cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
    z_val = cpu_pos[:, 0]

    jet_colors = jet(z_val)

    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 0.8)]
    sizes = [20.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)


def clip_camera(camera_data):
    # clip camera image to bounding box:
    h_ratio = 0.05
    w_ratio = 0.05
    depth = camera_data["raw_depth"]
    depth_tensor = camera_data["depth"]
    h, w = depth_tensor.shape
    depth[: int(h_ratio * h), :] = 0.0
    depth[int((1 - h_ratio) * h) :, :] = 0.0
    depth[:, : int(w_ratio * w)] = 0.0
    depth[:, int((1 - w_ratio) * w) :] = 0.0

    depth_tensor[: int(h_ratio * h), :] = 0.0
    depth_tensor[int(1 - h_ratio * h) :, :] = 0.0
    depth_tensor[:, : int(w_ratio * w)] = 0.0
    depth_tensor[:, int(1 - w_ratio * w) :] = 0.0


def draw_line(start, gradient):
    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_lines()
    start_list = [start]
    end_list = [start + gradient]

    colors = [(0.0, 0, 0.8, 0.9)]

    sizes = [10.0]
    draw.draw_lines(start_list, end_list, colors, sizes)


if __name__ == "__main__":
    radius = 0.05
    act_distance = 0.4
    voxel_size = 0.05
    render_voxel_size = 0.02
    clipping_distance = 0.7
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    target_material_2 = OmniPBR("/World/looks/t2", color=np.array([0, 1, 0]))

    target = cuboid.VisualCuboid(
        "/World/target_1",
        position=np.array([0.4, -0.5, 0.2]),
        orientation=np.array([0, 1.0, 0, 0]),
        size=0.04,
        visual_material=target_material,
    )

    # Make a target to follow
    target_2 = cuboid.VisualCuboid(
        "/World/target_2",
        position=np.array([0.4, 0.5, 0.2]),
        orientation=np.array([0.0, 1, 0.0, 0.0]),
        size=0.04,
        visual_material=target_material_2,
    )

    # Make a target to follow
    camera_marker = cuboid.VisualCuboid(
        "/World/camera_nvblox",
        position=np.array([-0.05, 0.0, 0.45]),
        # orientation=np.array([0.793, 0, 0.609,0.0]),
        orientation=np.array([0.5, -0.5, 0.5, -0.5]),
        # orientation=np.array([0.561, -0.561, 0.431,-0.431]),
        color=np.array([0, 0, 1]),
        size=0.01,
    )
    camera_marker.set_visibility(False)

    collision_checker_type = CollisionCheckerType.BLOX
    world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.02,
                }
            }
        }
    )
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, _ = add_robot_to_scene(robot_cfg, my_world, "/World/world_robot/")

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_wall.yml"))
    )

    world_cfg_table.cuboid[0].pose[2] -= 0.01
    usd_help = UsdHelper()

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg_table.get_mesh_world(), base_frame="/World")
    world_cfg.add_obstacle(world_cfg_table.cuboid[0])
    world_cfg.add_obstacle(world_cfg_table.cuboid[1])
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_cuda_graph=True,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.03,
        collision_activation_distance=0.025,
        acceleration_scale=1.0,
        self_collision_check=True,
        maximum_trajectory_dt=0.25,
        finetune_dt_scale=1.05,
        fixed_iters_trajopt=True,
        finetune_trajopt_iters=300,
        minimize_jerk=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up..")
    motion_gen.warmup(warmup_js_trajopt=False)

    world_model = motion_gen.world_collision
    realsense_data = RealsenseDataloader(clipping_distance_m=clipping_distance)
    data = realsense_data.get_data()

    camera_pose = Pose.from_list([0, 0, 0, 0.707, 0.707, 0, 0])
    i = 0
    tensor_args = TensorDeviceType()
    target_list = [target, target_2]
    target_material_list = [target_material, target_material_2]
    for material in target_material_list:
        material.set_color(np.array([0.1, 0.1, 0.1]))
    target_idx = 0
    cmd_idx = 0
    cmd_plan = None
    articulation_controller = robot.get_articulation_controller()
    plan_config = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=4, max_attempts=2, enable_finetune_trajopt=True
    )
    if not args.use_debug_draw:
        voxel_viewer = VoxelManager(100, size=render_voxel_size)
    cmd_step_idx = 0
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

        if step_index % 5 == 0.0:
            # camera data updation
            world_model.decay_layer("world")
            data = realsense_data.get_data()
            clip_camera(data)
            cube_position, cube_orientation = camera_marker.get_local_pose()
            camera_pose = Pose(
                position=tensor_args.to_device(cube_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )

            data_camera = CameraObservation(  # rgb_image = data["rgba_nvblox"],
                depth_image=data["depth"], intrinsics=data["intrinsics"], pose=camera_pose
            )
            data_camera = data_camera.to(device=tensor_args.device)
            world_model.add_camera_frame(data_camera, "world")
            world_model.process_camera_frames("world", False)
            torch.cuda.synchronize()
            world_model.update_blox_hashes()

            bounding = Cuboid("t", dims=[1, 1, 1.0], pose=[0, 0, 0, 1, 0, 0, 0])
            voxels = world_model.get_voxels_in_bounding_box(bounding, voxel_size)
            if voxels.shape[0] > 0:
                voxels = voxels[voxels[:, 2] > voxel_size]
                voxels = voxels[voxels[:, 0] > 0.0]
                if args.use_debug_draw:
                    draw_points(voxels)
                else:
                    voxels = voxels.cpu().numpy()
                    voxel_viewer.update_voxels(voxels[:, :3])

                voxel_viewer.update_voxels(voxels[:, :3])
            else:
                if not args.use_debug_draw:
                    voxel_viewer.clear()
            # draw_points(voxels)

        if args.show_window:
            depth_image = data["raw_depth"]
            color_image = data["raw_rgb"]
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_VIRIDIS
            )
            color_image = cv2.flip(color_image, 1)
            depth_colormap = cv2.flip(depth_colormap, 1)

            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow("NVBLOX Example", cv2.WINDOW_NORMAL)
            cv2.imshow("NVBLOX Example", images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break

        if cmd_plan is None and step_index % 10 == 0:
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
    realsense_data.stop_device()
    print("finished program")

    simulation_app.close()
