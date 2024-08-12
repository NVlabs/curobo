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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration to download",
)
parser.add_argument("--save_usd", default=False, action="store_true")
args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": args.save_usd})

# Third Party
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction

try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    from omni.importer.urdf import _urdf  # isaac sim 2023.1

# CuRobo
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    join_path,
    load_yaml,
)


def save_usd():
    my_world = World(stage_units_in_meters=1.0)

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 10000
    import_config.default_position_drive_damping = 100
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0
    # Get the urdf file path
    robot_config = load_yaml(join_path(get_robot_configs_path(), args.robot))
    urdf_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["urdf_path"])
    asset_path = join_path(
        get_assets_path(), robot_config["robot_cfg"]["kinematics"]["asset_root_path"]
    )
    urdf_interface = _urdf.acquire_urdf_interface()
    full_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["urdf_path"])
    default_config = robot_config["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
    j_names = robot_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]

    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
    robot_path = urdf_interface.import_robot(
        robot_path, filename, imported_robot, import_config, ""
    )
    robot = my_world.scene.add(Robot(prim_path=robot_path, name="robot"))
    # robot.disable_gravity()
    i = 0

    my_world.reset()

    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    save_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["usd_path"])
    usd_help.write_stage_to_file(save_path, True)
    print("Wrote usd file to " + save_path)
    simulation_app.close()


def debug_usd():
    my_world = World(stage_units_in_meters=1.0)

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 10000
    import_config.default_position_drive_damping = 100
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0
    # Get the urdf file path
    robot_config = load_yaml(join_path(get_robot_configs_path(), args.robot))
    urdf_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["urdf_path"])
    asset_path = join_path(
        get_assets_path(), robot_config["robot_cfg"]["kinematics"]["asset_root_path"]
    )
    urdf_interface = _urdf.acquire_urdf_interface()
    full_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["urdf_path"])
    default_config = robot_config["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
    j_names = robot_config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]

    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
    robot_path = urdf_interface.import_robot(
        robot_path, filename, imported_robot, import_config, ""
    )
    robot = my_world.scene.add(Robot(prim_path=robot_path, name="robot"))
    # robot.disable_gravity()
    i = 0

    articulation_controller = robot.get_articulation_controller()
    my_world.reset()

    while simulation_app.is_running():
        my_world.step(render=True)
        if i == 0:
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            i += 1
        # if dof_n is not None:
        #    dof_i = [robot.get_dof_index(x) for x in j_names]
        #
        #    robot.set_joint_positions(default_config, dof_i)
        if robot.is_valid():
            art_action = ArticulationAction(default_config, joint_indices=idx_list)
            articulation_controller.apply_action(art_action)
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    save_path = join_path(get_assets_path(), robot_config["robot_cfg"]["kinematics"]["usd_path"])
    usd_help.write_stage_to_file(save_path, True)
    simulation_app.close()


if __name__ == "__main__":
    if args.save_usd:
        save_usd()
    else:
        debug_usd()
