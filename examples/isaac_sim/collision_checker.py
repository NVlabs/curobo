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

# Third Party
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nvblox", action="store_true", help="When True, enables headless mode", default=False
)

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
args = parser.parse_args()

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
from helper import add_extensions
from omni.isaac.core import World

try:
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    from isaacsim.core.api.materials import OmniPBR
from omni.isaac.core.objects import sphere

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose

########### OV #################
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


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

    colors = [(1, 0, 0, 0.8)]

    sizes = [10.0]
    draw.draw_lines(start_list, end_list, colors, sizes)


def main():
    usd_help = UsdHelper()
    act_distance = 0.4
    ignore_list = ["/World/target", "/World/defaultGroundPlane"]

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
    radius = 0.1
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])

    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))

    target = sphere.VisualSphere(
        "/World/target",
        position=np.array([0.5, 0, 1.0]) + pose.position[0].cpu().numpy(),
        orientation=np.array([1, 0, 0, 0]),
        radius=radius,
        visual_material=target_material,
    )

    setup_curobo_logger("warn")

    # warmup curobo instance

    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    world_file = ["collision_thin_walls.yml", "collision_test.yml"][-1]
    collision_checker_type = CollisionCheckerType.MESH
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    world_cfg.objects[0].pose[2] += 0.2
    vis_world_cfg = world_cfg

    if args.nvblox:
        world_file = "collision_nvblox.yml"
        collision_checker_type = CollisionCheckerType.BLOX
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        world_cfg.objects[0].pose[2] += 0.4
        ignore_list.append(world_cfg.objects[0].name)
        vis_world_cfg = world_cfg.get_mesh_world()
        # world_cfg = vis_world_cfg

    usd_help.add_world_to_stage(vis_world_cfg, base_frame="/World")
    config = RobotWorldConfig.load_from_config(
        robot_file,
        world_cfg,
        collision_activation_distance=act_distance,
        collision_checker_type=collision_checker_type,
    )
    model = RobotWorld(config)
    i = 0
    x_sph = torch.zeros((1, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius

    add_extensions(simulation_app, args.headless_mode)
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        step_index = my_world.current_time_step_index

        if step_index == 0:
            my_world.reset()

        if step_index < 20:
            continue
        if step_index % 1000 == 0.0:
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                reference_prim_path="/World",
                ignore_substring=ignore_list,
            ).get_collision_check_world()

            model.update_world(obstacles)
            print("Updated World")

        sp_buffer = []
        sph_position, _ = target.get_local_pose()

        x_sph[..., :3] = tensor_args.to_device(sph_position).view(1, 1, 1, 3)

        d, d_vec = model.get_collision_vector(x_sph)

        d = d.view(-1).cpu()

        p = d.item()
        p = max(1, p * 5)
        if d.item() != 0.0:
            draw_line(sph_position, d_vec[..., :3].view(3).cpu().numpy())
            print(d, d_vec)

        else:
            # Third Party
            try:
                from omni.isaac.debug_draw import _debug_draw
            except ImportError:
                from isaacsim.util.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
            # if draw.get_num_points() > 0:
            draw.clear_lines()
        if d.item() == 0.0:
            target_material.set_color(np.ravel([0, 1, 0]))
        elif d.item() <= model.contact_distance:
            target_material.set_color(np.array([0, 0, p]))
        elif d.item() >= model.contact_distance:
            target_material.set_color(np.array([p, 0, 0]))


if __name__ == "__main__":
    main()
