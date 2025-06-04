"""
Isaac Sim example for two-phase reactive collision-free motion generation using CuRobo.
This script demonstrates a XArm7 robot reaching for a target sphere in a strawberry plant
while avoiding the rest of the plant structure using collision avoidance.

Two-Phase Approach:
- Phase 1: Move to pre-grasp position (20cm offset from target) and wait for user input
- Phase 2: Incremental approach along +X direction to target

Gripper Orientation (tool_pose):
- Different orientations can be cycled through using 'z' key
- Each orientation is maintained during both phases
- After reaching target, press 'z' to reset and try the next orientation

Usage:
    python zordi_motion_gen.py [--headless_mode native|websocket]
"""

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Standard Library
import argparse
import os
import sys
from typing import Optional

# Add parent directory to Python path for importing helper
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np

# Third Party
import torch

# Test CUDA availability
a = torch.zeros(4, device="cuda:0")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Zordi reactive motion generation example with plant obstacles"
)
parser.add_argument(
    "--headless",
    action="store_true",
)
parser.add_argument(
    "--orientation_index",
    type=int,
    default=0,
    choices=range(0, 5),  # Assuming there are 5 orientations
    help="Index of the gripper orientation to use (0-4)",
)

args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": "1920",
    "height": "1080",
})

# CuRobo

# Import configuration utilities
from config_utils import get_plant_usd_path, load_robot_config_with_zordi_paths
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper  # noqa: E402
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (  # noqa: E402
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from helper import add_robot_to_scene

# Isaac Sim imports
from omni.isaac.core import World
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.types import ArticulationAction

# USD imports for plant manipulation
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

from isaacsim.core.utils.xforms import get_world_pose

# Robot configuration from bimanual environment
# Using right arm configuration from plant_zordi_bimanual_env_cfg.py
right_home_deg = [5, -30, 25, 0, 0, -60, -5, 0.5]
right_home_q = np.deg2rad(right_home_deg)
GRIPPER_ORIENTATION_LIST = [
    [0.0, 0.7071, 0.0, 0.7071],  # ( 1, 0 ,0)  (90 deg)
    [0.1379, -0.6935, -0.1379, -0.6935],
    [0.2706, -0.6533, -0.2706, -0.6533],  # 45 deg
    [0.3928, -0.5879, -0.3928, -0.5879],
    [0.5000, -0.5000, -0.5000, -0.5000],  # ( 0, 1 ,0)  (0 deg)
]

GRIPPER_VECTOR_LIST = [
    np.array([1.0000, 0.0000, 0]),
    np.array([0.9239, 0.3827, 0]),
    np.array([0.7071, 0.7071, 0]),
    np.array([0.3827, 0.9239, 0]),
    np.array([0.0000, 1.0000, 0]),
]


def debug_print(message: str, level: str = "info") -> None:
    """Print debug message if debug mode is enabled and message level is important enough."""
    if level in ["error", "warning", "info"]:
        print(f"[DEBUG {level.upper()}] {message}")


class ZordiPlantMotionGenExample:
    """Reactive motion generation example with strawberry plant obstacle avoidance."""

    def __init__(self) -> None:
        """Initialize the motion generation example with plant loading."""
        print("[INIT] Starting ZordiPlantMotionGenExample initialization...")
        debug_print("Initializing ZordiPlantMotionGenExample...", "info")
        # Initialize attributes
        self.cmd_plan: Optional[JointState] = None
        self.cmd_idx = 0
        self.past_cmd: Optional[JointState] = None
        self.spheres: Optional[list] = None
        self.articulation_controller = None

        # Plant-specific attributes
        self.target_position: Optional[np.ndarray] = None
        self.target_prim_path = "/World/PlantScene/plant_003/plant_003/stem_Unit003_13/Strawberry003/stem/Stem_20/Sphere"
        self.associated_stem_path = "/World/PlantScene/plant_003/plant_003/stem_Unit003_13/Strawberry003/stem/Stem_20/Stem_20"

        # Use orientation index from command-line argument
        self.orientation_index = args.orientation_index
        self.total_orientations = len(GRIPPER_ORIENTATION_LIST)
        if self.orientation_index >= self.total_orientations:
            print(
                f"WARNING: Orientation index {self.orientation_index} is out of range. Using index 0."
            )
            self.orientation_index = 0
        debug_print(
            f"Using orientation index {self.orientation_index} from command-line argument",
            "info",
        )

        self.stems_to_ignore = [
            "/World/PlantScene/plant_003/plant_003/stem_Unit003_13/Strawberry003/stem/Stem_14/Stem_14",
        ]
        self.plant_prim_path = "/World/PlantScene"

        # Two-phase motion planning attributes
        self.approach_planned = False
        self.target_reached = False
        self.hold_position = False

        # Phase 2 incremental approach attributes
        self.approach_step_size = 0.02
        self.approach_total_distance = 0.0
        self.approach_steps_taken = 0
        self.approach_total_steps = 0
        self.last_remaining_distance: Optional[float] = None

        # Setup logging
        setup_curobo_logger("error")

        # Initialize components
        self._setup_world()
        self._setup_robot()
        self._setup_motion_gen()

        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.my_world.stage)
        self._update_world_obstacles()

    def _setup_world(self) -> None:
        """Setup the Isaac Sim world and stage with plant-optimized physics settings."""
        self.my_world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 50.0,
            rendering_dt=1.0 / 50.0,
        )

        sim_context = SimulationContext.instance()
        physics_context = sim_context.get_physics_context()
        physics_context.set_gravity(value=0.0)

        stage = self.my_world.stage
        plant_prim = stage.DefinePrim(self.plant_prim_path, "Xform")
        plant_prim.GetReferences().AddReference(get_plant_usd_path())

        # Set plant position and orientation - KEEP ORIGINAL Z=0.6 as requested
        xformable = UsdGeom.Xformable(plant_prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.55, 0.45, 0.7))

        # Use the unified helper function to disable physics for all plants
        debug_print("Plant loaded and configured successfully", "info")

        debug_print(f"Setting target to prim: {self.target_prim_path}", "info")

        self.target_position, _ = get_world_pose(self.target_prim_path)
        # disable collision between robot and plant

    def _setup_robot(self) -> None:
        """Setup the XArm7 robot with configuration from bimanual environment."""
        debug_print("Setting up robot...", "info")
        self.tensor_args = TensorDeviceType()
        robot_cfg = load_robot_config_with_zordi_paths("xarm7.yml")["robot_cfg"]

        all_joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.j_names = [
            name for name in all_joint_names if "gripper" not in name.lower()
        ]

        self.default_config = right_home_q[: len(self.j_names)]
        self.gripper_position = right_home_deg[-1]

        debug_print(f"Arm joint names for planning: {self.j_names}", "info")
        debug_print(f"Default arm config: {self.default_config}", "info")
        debug_print(f"Fixed gripper position: {self.gripper_position}", "info")

        self.robot, self.robot_prim_path = add_robot_to_scene(
            robot_cfg, self.my_world, position=np.array([0, 0, 0.0])
        )
        self.robot_tool_prim_path = f"{self.robot_prim_path}/tool_pose"
        debug_print(f"Robot added to scene at path: {self.robot_prim_path}", "info")

    def _setup_motion_gen(self) -> None:
        """Setup CuRobo motion generation with plant obstacle configuration."""
        debug_print("Setting up motion generation...", "info")

        robot_cfg = load_robot_config_with_zordi_paths("xarm7.yml")["robot_cfg"]
        robot_cfg["kinematics"]["ee_link"] = "tool_pose"

        robot_cfg["kinematics"]["cspace"]["joint_names"] = self.j_names
        robot_cfg["kinematics"]["cspace"]["retract_config"] = self.default_config
        robot_cfg["kinematics"]["cspace"]["null_space_weight"] = [1] * len(self.j_names)
        robot_cfg["kinematics"]["cspace"]["cspace_distance_weight"] = [1] * len(
            self.j_names
        )

        debug_print(
            f"Modified robot config for arm-only planning ({len(self.j_names)} DOF)",
            "info",
        )

        world_cfg = WorldConfig()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,  # PRIMITIVE,
            self_collision_check=False,
            self_collision_opt=False,
            position_threshold=0.001,
            rotation_threshold=15.0,  # More relaxed orientation tolerance (was 5.0)
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

        self.motion_gen = MotionGen(motion_gen_config)
        debug_print("CuRobo motion generator initialized", "info")

        if hasattr(self.motion_gen, "world_coll_checker"):
            if self.motion_gen.world_coll_checker is not None:
                debug_print(
                    "✅ World collision checker initialized successfully", "info"
                )
            else:
                debug_print(
                    "❌ World collision checker is None after MotionGen init",
                    "error",
                )
                raise RuntimeError("Collision checker failed to initialize")
        else:
            debug_print("❌ MotionGen has no world_coll_checker attribute", "error")
            raise RuntimeError("MotionGen missing collision checker attribute")

        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=0,
            max_attempts=10,
            enable_finetune_trajopt=False,
            time_dilation_factor=1.0,
        )
        debug_print("Motion generation plan config set for graph planning", "info")

        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.my_world.stage)

        debug_print(
            "Motion generation setup complete - collision checking enabled", "info"
        )

    def _update_world_obstacles(self) -> None:
        """Update plant obstacles from USD stage for collision avoidance."""
        debug_print("Updating world obstacles...", "info")

        if (
            hasattr(self.motion_gen, "world_coll_checker")
            and self.motion_gen.world_coll_checker is not None
        ):
            self.motion_gen.world_coll_checker.clear_cache()
            debug_print("Cleared collision cache due to geometry change", "info")
        else:
            debug_print(
                "World collision checker not available for cache clear",
                "warning",
            )
        self.usd_help.load_stage(self.my_world.stage)

        debug_print("Scanning USD stage for obstacles...", "info")
        obstacles = self.usd_help.get_obstacles_from_stage(
            only_paths=["/World"],
            reference_prim_path=self.robot_prim_path,
            ignore_substring=[
                self.robot_prim_path,
                "/World/defaultGroundPlane",
            ],
        ).get_collision_check_world()

        if obstacles is not None and obstacles.objects is not None:
            num_obstacles = len(obstacles.objects)
            debug_print(f"Found {num_obstacles} plant obstacles in USD stage", "info")

            self.motion_gen.clear_world_cache()
            self.motion_gen.update_world(obstacles)

    def _plan_pre_grasp(
        self,
        target_position: np.ndarray,
    ) -> bool:
        debug_print(f"Planning motion to target position: {target_position}", "info")

        # Get the updated approach vector and orientation after stem alignment
        gripper_orientation_constraint, approach_vector = self._update_approach_vector()

        # Calculate pre-grasp position based on the gripper orientation
        pre_grasp_offset = 0.15
        pre_grasp_position = target_position - pre_grasp_offset * approach_vector

        pre_grasp_goal = Pose(
            position=self.tensor_args.to_device(pre_grasp_position),
            quaternion=self.tensor_args.to_device(gripper_orientation_constraint),
        )

        # Create planning configuration with relaxed orientation constraints for pre-grasp
        pre_grasp_planning_metric = PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=self.tensor_args.to_device([1, 1, 1, 0.01, 0.01, 0.01]),
        )

        plan_config_pre_grasp = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=0,
            max_attempts=10,
            enable_finetune_trajopt=False,
            time_dilation_factor=1.0,
            pose_cost_metric=pre_grasp_planning_metric,
        )

        cu_js = self._get_current_joint_state()

        result = self.motion_gen.plan_single(
            cu_js.unsqueeze(0),
            pre_grasp_goal,
            plan_config_pre_grasp,
        )

        if result is not None and result.success is not None and result.success.item():
            debug_print("Pre-grasp motion planning SUCCESSFUL!", "info")
            # Get the pre-grasp trajectory ONLY - no approach planning yet
            pre_grasp_plan = self.motion_gen.get_full_js(result.get_interpolated_plan())

            self.cmd_plan = pre_grasp_plan
            self.stored_pre_grasp_position = pre_grasp_position

            sim_js_names = self.robot.dof_names
            idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in self.cmd_plan.joint_names and x in self.j_names:
                    idx_list.append(self.robot.get_dof_index(x))
                    common_js_names.append(x)

            self.cmd_plan = self.cmd_plan.get_ordered_joint_state(common_js_names)
            self.cmd_idx = 0
            return True
        else:
            debug_print(
                f"Pre-grasp motion planning FAILED. Status: {result.status if result else 'None'}",
                "error",
            )
            return False

    def run(self) -> None:
        """Run the main simulation loop."""
        debug_print("Starting main simulation loop...", "info")

        print("[ROBOT] Initializing robot joint state...")
        sim_js_names = self.robot.dof_names
        print(f"sim_js_names: {sim_js_names}")
        arm_indices = [
            self.robot.get_dof_index(x) for x in self.j_names if x in sim_js_names
        ]
        arm_home_positions = self.default_config
        self.phase1_completed = False

        gripper_indices = [
            self.robot.get_dof_index(x) for x in sim_js_names if "gripper" in x.lower()
        ]
        gripper_positions = [self.gripper_position] * len(gripper_indices)
        self.robot.set_joint_positions(arm_home_positions, arm_indices)
        self.robot.set_joint_positions(gripper_positions, gripper_indices)

        self.articulation_controller = self.robot.get_articulation_controller()

        print("[ROBOT] ✅ Robot initialized to home position")

        sim_js_names = self.robot.dof_names

        while simulation_app.is_running():
            self.my_world.step(render=True)

            if self.cmd_plan is None and not self.phase1_completed:
                success = self._plan_pre_grasp(self.target_position)
                if success:
                    debug_print("Motion plan SUCCESSFUL", "info")
                else:
                    debug_print("Motion planning FAILED", "error")

            if self.cmd_plan is not None:
                cmd_state = self.cmd_plan[self.cmd_idx]
                arm_positions_cmd = cmd_state.position.cpu().numpy()
                arm_velocities_cmd = cmd_state.velocity.cpu().numpy()

                gripper_indices_cmd = [
                    self.robot.get_dof_index(x)
                    for x in sim_js_names
                    if "gripper" in x.lower()
                ]
                all_positions_cmd = list(arm_positions_cmd) + [
                    self.gripper_position
                ] * len(gripper_indices_cmd)
                all_velocities_cmd = list(arm_velocities_cmd) + [0.0] * len(
                    gripper_indices_cmd
                )

                arm_idx_list_cmd = [
                    self.robot.get_dof_index(name) for name in self.cmd_plan.joint_names
                ]
                all_idx_list_cmd = arm_idx_list_cmd + gripper_indices_cmd

                if not self.hold_position:
                    art_action = ArticulationAction(
                        np.array(all_positions_cmd),
                        np.array(all_velocities_cmd),
                        joint_indices=all_idx_list_cmd,
                    )
                else:
                    print("Holding position")
                    curr_js = self.robot.get_joints_state()
                    all_positions_cmd = curr_js.positions[arm_idx_list_cmd]
                    all_velocities_cmd = curr_js.velocities[arm_idx_list_cmd] * 0.0
                    art_action = ArticulationAction(
                        np.array(all_positions_cmd),
                        np.array(all_velocities_cmd),
                        joint_indices=all_idx_list_cmd,
                    )

                self.articulation_controller.apply_action(art_action)
                self.cmd_idx += 1

                if (
                    self.cmd_idx >= len(self.cmd_plan.position)
                    and self.cmd_plan is not None
                ):
                    self.cmd_plan = None
                    self.cmd_idx = 0

                    if not self.phase1_completed:
                        debug_print("Phase 1 completed", "info")
                        curr_tool_position, _ = get_world_pose(
                            self.robot_tool_prim_path
                        )
                        dist = np.linalg.norm(
                            self.stored_pre_grasp_position - curr_tool_position
                        )
                        if dist < 0.01:
                            debug_print(
                                f"Pre-grasp position reached: {dist:.4f}m", "info"
                            )
                            self.phase1_completed = True
                            self.approach_planned = False
                            self._plan_approach_motion()
                        else:
                            debug_print(
                                f"Dist to pre-grasp position not reached: {dist:.4f}m",
                                "info",
                            )
                            self.phase1_completed = False
                            self.approach_planned = False
                            self._plan_pre_grasp(self.target_position)
                        continue

                    if self.approach_planned and self.phase1_completed:
                        curr_tool_position, _ = get_world_pose(
                            self.robot_tool_prim_path
                        )
                        dist = np.linalg.norm(self.target_position - curr_tool_position)
                        if dist < 0.005:
                            debug_print(f"Phase 2 completed: {dist:.4f}m", "info")
                            self.phase2_completed = True
                            self.cmd_plan = None
                            self.hold_position = True
                        else:
                            debug_print(
                                f"Dist to target position not reached: {dist:.4f}m",
                                "info",
                            )
                            self.phase2_completed = False
                            self._plan_approach_motion()

    def _get_current_joint_state(self) -> JointState:
        sim_js = self.robot.get_joints_state()
        sim_js_names = self.robot.dof_names
        arm_positions = []
        for joint_name in self.j_names:
            if joint_name in sim_js_names:
                joint_idx = sim_js_names.index(joint_name)
                arm_positions.append(sim_js.positions[joint_idx])
        return JointState(
            position=self.tensor_args.to_device(arm_positions),
            velocity=self.tensor_args.to_device([0.0] * len(arm_positions)),
            acceleration=self.tensor_args.to_device([0.0] * len(arm_positions)),
            jerk=self.tensor_args.to_device([0.0] * len(arm_positions)),
            joint_names=self.j_names,
        ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def _plan_approach_motion(self) -> bool:
        """Plan incremental collision-free motion along the gripper's forward direction toward target."""
        tool_position, _ = get_world_pose(self.robot_tool_prim_path)
        approach_vector = self.target_position - tool_position
        approach_vector = approach_vector / np.linalg.norm(approach_vector)
        gripper_orientation_constraint, _ = self._update_approach_vector()

        distance_to_target = np.linalg.norm(self.target_position - tool_position)
        if self.approach_step_size > distance_to_target:
            step_size = distance_to_target * 0.5
            print(f"Step size: {step_size}")
        else:
            step_size = self.approach_step_size

        # Calculate expected waypoints
        first_step_position = tool_position + approach_vector * step_size

        step_goal = Pose(
            position=self.tensor_args.to_device(first_step_position),
            quaternion=self.tensor_args.to_device(gripper_orientation_constraint),
        )

        step_planning_config = MotionGenPlanConfig(
            enable_graph=False,  # Use graph planning for collision avoidance
            enable_graph_attempt=0,
            max_attempts=2,
            enable_finetune_trajopt=False,  # Keep it simple
            time_dilation_factor=1.0,
            pose_cost_metric=PoseCostMetric(
                reach_partial_pose=True,  # Allow partial pose constraints
                reach_vec_weight=self.tensor_args.to_device([
                    1.0,
                    1.0,
                    1.0,
                    0.05,
                    0.05,
                    0.05,
                ]),
            ),
        )

        step_result = self.motion_gen.plan_single(
            self._get_current_joint_state().unsqueeze(0),
            step_goal,
            step_planning_config,
        )

        if (
            step_result is not None
            and step_result.success is not None
            and step_result.success.item()
        ):
            debug_print("✅ Step planning SUCCESSFUL!", "info")

            step_plan = self.motion_gen.get_full_js(step_result.get_interpolated_plan())
            self.cmd_plan = step_plan
            self.cmd_idx = 0
            self.approach_planned = True
            return True
        else:
            debug_print("❌ Step planning FAILED", "error")
            return False

    def _update_approach_vector(self):
        """Update the approach vector for the gripper."""
        # Get the original approach vector (preserve its xy-direction)
        original_approach = np.array(GRIPPER_VECTOR_LIST[self.orientation_index])
        original_xy_direction = original_approach[:2]  # Keep (x, y) components
        original_xy_magnitude = np.linalg.norm(original_xy_direction)

        # Get the stem's transform and extract rotation matrix
        stage = self.my_world.stage
        stem_prim = stage.GetPrimAtPath(self.associated_stem_path)

        if not stem_prim or not stem_prim.IsValid():
            debug_print(f"Invalid stem prim: {self.associated_stem_path}", "warning")
            return None, None

        stem_xformable = UsdGeom.Xformable(stem_prim)
        world_transform = stem_xformable.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rotation_matrix = np.array(world_transform)[:3, :3]

        # Calculate desired approach direction (negative stem z-direction)
        desired_approach_direction = -rotation_matrix[:, 2]

        # Calculate optimal z-component to maximize alignment while preserving xy-direction
        a_dot_b_xy = np.dot(original_xy_direction, desired_approach_direction[:2])
        optimal_z = (a_dot_b_xy * desired_approach_direction[2]) / (
            original_xy_magnitude**2
        )

        # Construct and normalize the new approach vector
        new_approach = np.array([*original_xy_direction, optimal_z])
        gripper_z_axis = new_approach / np.linalg.norm(new_approach)

        # Calculate gripper x-axis orthogonal to z-axis, closest to world +z
        world_up = np.array([0.0, 0.0, 1.0])
        perpendicular_component = (
            world_up - np.dot(world_up, gripper_z_axis) * gripper_z_axis
        )

        gripper_x_axis = perpendicular_component / np.linalg.norm(
            perpendicular_component
        )

        # Calculate gripper y-axis to complete right-handed coordinate system
        gripper_y_axis = np.cross(gripper_z_axis, gripper_x_axis)

        # Construct rotation matrix and convert to quaternion
        rotation_matrix = np.column_stack((
            gripper_x_axis,
            gripper_y_axis,
            gripper_z_axis,
        ))
        rotation = R.from_matrix(rotation_matrix)
        new_quaternion = rotation.as_quat()[
            [3, 0, 1, 2]
        ]  # Convert to [w, x, y, z] format

        # appraoch vector is the z-axis of the rotatoin matrix
        new_approach_vector = rotation_matrix[:, 2]
        return new_quaternion, new_approach_vector


def main() -> None:
    """Main function to run the motion generation example."""
    example = None
    try:
        example = ZordiPlantMotionGenExample()
        example.run()
    finally:
        # Always ensure cleanup happens
        simulation_app.close()


if __name__ == "__main__":
    main()
