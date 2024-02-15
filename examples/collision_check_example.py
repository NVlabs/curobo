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
""" Example computing collisions using curobo

"""
# Third Party
import torch
import argparse

# Third Party
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()

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

from isaac_sim.helper import add_extensions, add_robot_to_scene

# Third Party
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import sphere


# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import WorldConfig
from curobo.util.usd_helper import UsdHelper
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util_file import get_world_configs_path, join_path, load_yaml, get_robot_configs_path

if __name__ == "__main__":
    usd_help = UsdHelper()

    my_world = World(stage_units_in_meters=1.0)
    robot_file = "franka.yml"
    world_file = "collision_test.yml"
    tensor_args = TensorDeviceType()
    # config = RobotWorldConfig.load_from_config(robot_file, world_file, pose_weight=[10, 200, 1, 10],
    #                                           collision_activation_distance=0.0)
    # curobo_fn = RobotWorld(config)

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    robot_file = "franka.yml"
    world_file = "benchmark_shelf.yml"
    collision_checker_type = CollisionCheckerType.MESH
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    tensor_args = TensorDeviceType()

    vis_world_cfg = world_cfg
    usd_help.add_world_to_stage(vis_world_cfg, base_frame="/World")

    config = RobotWorldConfig.load_from_config(
        robot_file, world_file, collision_activation_distance=0.0
    )

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]

    add_robot_to_scene(
        robot_cfg,
        my_world,
        "/World/world_0" + "/",
        robot_name="Pandas",
        position=Pose.from_list([0, 0, 0, 1, 0, 0, 0]).position[0].cpu().numpy(),
    )

    curobo_fn = RobotWorld(config)

    q_sph = torch.randn((10, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    q_sph[..., 3] = 0.2
    d = curobo_fn.get_collision_distance(q_sph)
    print(d)

    q_s = curobo_fn.sample(5, mask_valid=False)

    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)
    print("Collision Distance:")
    print("World:", d_world)
    print("Self:", d_self)

    while simulation_app.is_running():
        my_world.step(render=True)

