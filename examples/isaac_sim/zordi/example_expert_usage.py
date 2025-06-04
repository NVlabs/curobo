"""
Copyright 2024 Zordi, Inc. All rights reserved.

Example usage of ZordiMotionExpert for two-phase motion planning.
This script demonstrates how to integrate the expert policy into a simulation loop.

Usage:
    python example_expert_usage.py [--headless]
"""

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Standard Library
import argparse

import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description="Example usage of ZordiMotionExpert")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from helper import add_robot_to_scene

# Isaac Sim imports
from omni.isaac.core import World
from omni.isaac.core.simulation_context import SimulationContext

# USD imports
from pxr import Gf, UsdGeom

# Import the expert policy
from zordi_motion_expert import create_motion_expert

# Constants
PLANT_ROOT = "/home/gilwoo/workspace/zordi_sim_assets/lightwheel"
PLANT_USD = f"{PLANT_ROOT}/Scene001_kinematics.usd"
right_home_deg = [5, -30, 25, 0, 0, -60, -5, 0.5]
right_home_q = np.deg2rad(right_home_deg)


def setup_simulation() -> tuple:
    """Setup Isaac Sim world and robot for motion expert demo."""
    # Setup logging
    setup_curobo_logger("error")

    # Create world
    my_world = World(
        stage_units_in_meters=1.0, physics_dt=1.0 / 50.0, rendering_dt=1.0 / 50.0
    )
    sim_context = SimulationContext.instance()
    physics_context = sim_context.get_physics_context()
    physics_context.set_gravity(value=0.0)

    # Load plant scene
    stage = my_world.stage
    plant_prim_path = "/World/PlantScene"
    plant_prim = stage.DefinePrim(plant_prim_path, "Xform")
    plant_prim.GetReferences().AddReference(PLANT_USD)

    # Position plant
    xformable = UsdGeom.Xformable(plant_prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.55, 0.45, 0.7))

    # Setup robot
    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, "xarm7.yml"))["robot_cfg"]

    all_joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    j_names = [name for name in all_joint_names if "gripper" not in name.lower()]
    default_config = right_home_q[: len(j_names)]

    robot, robot_prim_path = add_robot_to_scene(
        robot_cfg, my_world, position=np.array([0, 0, 0.0])
    )

    # Setup motion generation
    robot_cfg["kinematics"]["ee_link"] = "tool_pose"
    robot_cfg["kinematics"]["cspace"]["joint_names"] = j_names
    robot_cfg["kinematics"]["cspace"]["retract_config"] = default_config
    robot_cfg["kinematics"]["cspace"]["null_space_weight"] = [1] * len(j_names)
    robot_cfg["kinematics"]["cspace"]["cspace_distance_weight"] = [1] * len(j_names)

    world_cfg = WorldConfig()
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        self_collision_check=False,
        self_collision_opt=False,
        position_threshold=0.001,
        rotation_threshold=15.0,
        cspace_threshold=0.2,
        num_trajopt_seeds=4,
        num_graph_seeds=32,
        interpolation_dt=0.06,
        interpolation_steps=2000,
        collision_cache={"obb": 20, "mesh": 50},
        num_ik_seeds=20,
        use_cuda_graph=True,
        store_trajopt_debug=True,
    )

    motion_gen = MotionGen(motion_gen_config)

    # Update world obstacles
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    obstacles = usd_help.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[robot_prim_path, "/World/defaultGroundPlane"],
    ).get_collision_check_world()

    if obstacles is not None and obstacles.objects is not None:
        motion_gen.clear_world_cache()
        motion_gen.update_world(obstacles)

    return my_world, robot, motion_gen, tensor_args, j_names, default_config


def initialize_robot(robot, j_names, default_config):
    """Initialize robot to home position."""
    sim_js_names = robot.dof_names
    # Set arm joints
    arm_indices = [robot.get_dof_index(x) for x in j_names if x in sim_js_names]
    robot.set_joint_positions(default_config, arm_indices)

    # Set gripper joints
    gripper_indices = [
        robot.get_dof_index(x) for x in sim_js_names if "gripper" in x.lower()
    ]
    gripper_positions = [0.5] * len(gripper_indices)
    robot.set_joint_positions(gripper_positions, gripper_indices)


def main():
    """Main demonstration function."""
    print("[DEMO] Starting ZordiMotionExpert demonstration...")

    # Setup simulation
    my_world, robot, motion_gen, tensor_args, j_names, default_config = (
        setup_simulation()
    )

    # CRITICAL: Reset and initialize the world properly
    my_world.reset()

    # Initialize robot AFTER world reset
    initialize_robot(robot, j_names, default_config)

    # Allow robot to settle to initial joint positions
    for _ in range(50):  # Step for 50 physics frames (1 second at 50Hz)
        my_world.step(render=False)  # No need to render these settling steps

    # Create motion expert
    expert = create_motion_expert(
        motion_gen=motion_gen,
        robot=robot,
        tensor_args=tensor_args,
        orientation_index=0,  # Use first gripper orientation
        pre_grasp_offset=0.15,
        approach_step_size=0.02,
    )

    # Target strawberry sphere
    target_prim_path = "/World/PlantScene/plant_003/plant_003/stem_Unit003_13/Strawberry003/stem/Stem_20/Sphere"
    associated_stem_path = "/World/PlantScene/plant_003/plant_003/stem_Unit003_13/Strawberry003/stem/Stem_20"

    # Set target
    success = expert.set_target(target_prim_path, my_world.stage, associated_stem_path)
    if not success:
        print("[DEMO] Failed to set target - exiting")
        simulation_app.close()
        return

    # Main simulation loop
    step_count = 0
    articulation_controller = robot.get_articulation_controller()

    while simulation_app.is_running():
        my_world.step(render=True)
        step_count += 1

        # Get action from expert
        action = expert.get_action()

        # Apply action if available
        if action is not None:
            articulation_controller.apply_action(action)
        else:
            pass

    simulation_app.close()


if __name__ == "__main__":
    main()
