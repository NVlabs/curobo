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

# This script downloads robot usd assets from isaac sim for using in CuRobo.


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

simulation_app = SimulationApp({"headless": True})
# Third Party
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path as nucleus_path
from omni.isaac.core.utils.stage import add_reference_to_stage

# CuRobo
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml

# supported robots:
robots = ["franka.yml", "ur10.yml"]
# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration to download",
)
args = parser.parse_args()

if __name__ == "__main__":
    r = args.robot
    my_world = World(stage_units_in_meters=1.0)
    robot_config = load_yaml(join_path(get_robot_configs_path(), r))
    usd_path = nucleus_path() + robot_config["robot_cfg"]["kinematics"]["isaac_usd_path"]

    usd_help = UsdHelper()
    robot_name = r
    prim_path = robot_config["robot_cfg"]["kinematics"]["usd_robot_root"]
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    robot = my_world.scene.add(Robot(prim_path=prim_path, name=robot_name))
    usd_help.load_stage(my_world.stage)

    my_world.reset()
    articulation_controller = robot.get_articulation_controller()

    # create a new stage and add robot to usd path:
    save_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["usd_path"])
    usd_help.write_stage_to_file(save_path, True)
    my_world.clear()
    my_world.clear_instance()
    simulation_app.close()
