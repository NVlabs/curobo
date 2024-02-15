# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

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

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import sphere

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.math import Pose
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path, join_path, load_yaml, get_robot_configs_path
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

def main():
    usd_help = UsdHelper()
    act_distance = 0.2

    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    # my_world.scene.add_default_ground_plane()

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    world_file = "benchmark_shelf.yml"
    collision_checker_type = CollisionCheckerType.MESH
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    vis_world_cfg = world_cfg


    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    usd_help.add_world_to_stage(vis_world_cfg, base_frame="/World")
    
    add_robot_to_scene(
        robot_cfg,
        my_world,
        "/World/world_0" + "/",
        robot_name="Pandas",
        position=Pose.from_list([0, 0, 0, 1, 0, 0, 0]).position[0].cpu().numpy(),
    )

    while simulation_app.is_running():
        my_world.step(render=True)


if __name__ == "__main__":
    main()
