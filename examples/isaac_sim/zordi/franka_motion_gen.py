try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Standard Library
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# Third Party
import torch

# Test CUDA availability
a = torch.zeros(4, device="cuda:0")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Franka reactive motion generation example with plant obstacles"
)
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot collision spheres",
    default=False,
)
parser.add_argument(
    "--visualize_collisions",
    action="store_true",
    help="When True, visualizes collision meshes for debugging",
    default=False,
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable verbose debug output",
    default=True,
)

args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless_mode is not None,
    "width": "1920",
    "height": "1080",
})

# CuRobo and Isaac Sim imports
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from omni.isaac.core import World
from omni.isaac.core.objects import sphere
from omni.isaac.core.utils.types import ArticulationAction

# USD imports for plant manipulation
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from ..helper import add_extensions, add_robot_to_scene

# Plant configuration constants
PLANT_ROOT = "/home/gilwoo/workspace/zordi_sim_assets/lightwheel"
PLANT_USD = f"{PLANT_ROOT}/plant_004_approx.usd"

# Robot configuration for Franka Panda (from franka.yml)
franka_home_deg = [
    0.0,
    -74.5,
    0.0,
    -143.2,
    0.0,
    57.3,
    0.0,
    2.3,
    2.3,
]  # Franka default in degrees
franka_home_q = np.deg2rad(franka_home_deg)


def debug_print(message: str, level: str = "info") -> None:
    """Print debug message if debug mode is enabled and message level is important enough."""
    if level in ["error", "warning", "info"]:
        print(f"[DEBUG {level.upper()}] {message}")


def draw_collision_line(start, gradient):
    """Draw collision direction line from start position in direction of gradient."""
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        try:
            from isaacsim.util.debug_draw import _debug_draw
        except ImportError:
            return  # Skip if debug draw not available

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()

    start_list = [start]
    end_list = [start + gradient * 0.5]  # Scale gradient for visibility
    colors = [(1, 0, 0, 0.8)]  # Red line
    sizes = [5.0]

    draw.draw_lines(start_list, end_list, colors, sizes)


def clear_collision_lines():
    """Clear all collision direction lines."""
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        try:
            from isaacsim.util.debug_draw import _debug_draw
        except ImportError:
            return

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()


class FrankaPlantMotionGenExample:
    """Reactive motion generation example with strawberry plant obstacle avoidance using Franka Panda."""

    def __init__(self) -> None:
        """Initialize the motion generation example with plant loading."""
        print("[INIT] Starting FrankaPlantMotionGenExample initialization...")
        debug_print("Initializing FrankaPlantMotionGenExample...", "info")

        self.num_targets = 0
        self.past_pose: Optional[np.ndarray] = None
        self.past_orientation: Optional[np.ndarray] = None
        self.target_pose: Optional[np.ndarray] = None
        self.target_orientation: Optional[np.ndarray] = None
        self.cmd_plan: Optional[JointState] = None
        self.cmd_idx = 0
        self.past_cmd: Optional[JointState] = None
        self.spheres: Optional[list] = None
        self.articulation_controller = None

        # Plant-specific attributes
        self.target_position: Optional[np.ndarray] = None
        self.target_orientation: Optional[np.ndarray] = None
        self.target_prim_path = "/World/PlantScene/plant_004/stem_Unit001_02/Strawberry003/stem/Stem_20/Sphere"

        self.stems_to_ignore = [
            "/World/PlantScene/plant_004/stem_Unit001_02/Strawberry003/stem/Stem_20/Stem_20",
            "/World/PlantScene/plant_004/stem_Unit001_02/Strawberry003/stem/Stem_14/Stem_14",
        ]
        self.plant_prim_path = "/World/PlantScene"

        # Geometry change tracking
        self.geometry_changed = True
        self.last_target_position: Optional[np.ndarray] = None

        # User input and approach planning attributes
        self.waiting_for_user_input = False
        self.approach_planned = False
        self.stored_target_position: Optional[np.ndarray] = None
        self.stored_target_orientation: Optional[np.ndarray] = None
        self.stored_pre_grasp_position: Optional[np.ndarray] = None

        # **NEW: Target reached state**
        self.target_reached = False
        self.hold_position_plan: Optional[JointState] = None

        print("[INIT] Setting up logging...")
        setup_curobo_logger("error")

        # Suppress additional logging for speed
        import logging

        logging.getLogger("curobo").setLevel(logging.ERROR)
        logging.getLogger("omni.isaac.core").setLevel(logging.ERROR)

        # Initialize world and stage
        print("[INIT] Setting up world...")
        self._setup_world()
        print("[INIT] Configuring simulation...")
        self._configure_simulation()
        print("[INIT] Loading plant...")
        self._load_plant()
        print("[INIT] Setting up target...")
        self._setup_target()
        print("[INIT] Setting up robot...")
        self._setup_robot()
        print("[INIT] Setting up motion generation...")
        self._setup_motion_gen()

        if args.visualize_collisions:
            print("[INIT] Enabling collision visualization...")
            self._enable_collision_visualization()

        print("[INIT] Setting up USD helper...")
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.my_world.stage)

        print("[INIT] Loading plant obstacles into CuRobo...")
        self._update_world_obstacles()

        print("[INIT] Initialization complete!")

    def _setup_world(self) -> None:
        """Setup the Isaac Sim world and stage with plant-optimized physics settings."""
        self.my_world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 512.0,
            rendering_dt=1.0 / 512.0,
        )
        stage = self.my_world.stage

        xform = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(xform)
        stage.DefinePrim("/curobo", "Xform")

        self.my_world.scene.add_default_ground_plane()
        debug_print("World setup complete", "info")

    def _configure_simulation(self) -> None:
        """Configure simulation settings for plant stability."""
        try:
            from omni.isaac.core.simulation_context import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                physics_context = sim_context.get_physics_context()
                if physics_context:
                    # Set gravity to zero for plant stability
                    physics_context.set_gravity(value=[0.0, 0.0, 0.0])

                    physics_scene = physics_context.get_current_physics_scene()
                    if physics_scene:
                        physics_scene.set_solver_type(1)  # TGS solver
                        physics_scene.set_bounce_threshold_velocity(0.05)
                        physics_scene.set_friction_offset_threshold(0.04)
                        physics_scene.set_stabilization_enabled(True)
                        physics_scene.set_ccd_enabled(True)

                debug_print("Simulation configured with stability settings", "info")
        except Exception as e:
            debug_print(
                f"Warning: Could not configure advanced simulation settings: {e}",
                "warning",
            )

    def _setup_robot(self) -> None:
        """Setup the Franka Panda robot with configuration from franka.yml."""
        debug_print("Setting up Franka robot...", "info")
        self.tensor_args = TensorDeviceType()
        robot_cfg_path = get_robot_configs_path()
        robot_cfg = load_yaml(join_path(robot_cfg_path, "franka.yml"))["robot_cfg"]

        # Get Franka joint names (arm only, exclude gripper for planning)
        all_joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.j_names = [
            name for name in all_joint_names if "finger" not in name.lower()
        ]  # Only arm joints (panda_joint1-7)

        # Use Franka default configuration (arm only)
        self.default_config = robot_cfg["kinematics"]["cspace"]["retract_config"][
            : len(self.j_names)
        ]  # First 7 joints

        # Fixed gripper position (open)
        self.gripper_position = 0.04  # Open gripper position

        debug_print(f"Franka arm joint names: {self.j_names}", "info")
        debug_print(f"Default Franka config: {self.default_config}", "info")
        debug_print(f"Fixed gripper position: {self.gripper_position}", "info")

        self.robot, self.robot_prim_path = add_robot_to_scene(robot_cfg, self.my_world)
        debug_print(
            f"Franka robot added to scene at path: {self.robot_prim_path}", "info"
        )

    def _load_plant(self) -> None:
        """Load the strawberry plant USD file with stability settings."""
        if not Path(PLANT_USD).exists():
            debug_print(f"ERROR: Plant USD file not found: {PLANT_USD}", "error")
            raise FileNotFoundError(f"Plant USD file not found: {PLANT_USD}")

        debug_print(f"Loading plant from: {PLANT_USD}", "info")
        stage = self.my_world.stage

        # Create plant prim and reference the USD file
        plant_prim = stage.DefinePrim(self.plant_prim_path, "Xform")
        plant_prim.GetReferences().AddReference(PLANT_USD)

        # Set plant position and orientation
        xformable = UsdGeom.Xformable(plant_prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.85, 0.35, 0.6))  # Position in front of robot

        # Apply plant physics settings
        debug_print("Applying physics settings to plant...", "info")
        self._apply_plant_physics_settings(plant_prim)

        debug_print("Plant loaded and configured successfully", "info")

    def _apply_plant_physics_settings(self, plant_prim) -> None:
        """Apply physics settings for plant stability and enable collisions."""
        for prim in plant_prim.GetAllChildren():
            self._configure_prim_physics(prim)

    def _configure_prim_physics(self, prim) -> None:
        """Configure physics properties for a single prim and its children."""
        if prim.IsValid():
            prim_path = prim.GetPath().pathString
            is_target_prim = prim_path == self.target_prim_path
            is_stem_prim_to_ignore = prim_path in self.stems_to_ignore
            is_leaf_prim = "leaf" in prim_path.lower()

            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
                # Apply collision API for obstacle detection
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                else:
                    collision_api = UsdPhysics.CollisionAPI(prim)

                # Enable or disable collision based on prim type
                if is_target_prim or is_leaf_prim or is_stem_prim_to_ignore:
                    collision_api.GetCollisionEnabledAttr().Set(False)
                    if is_target_prim:
                        print(f"🎯 IGNORING (target): {prim_path}")
                    elif is_leaf_prim:
                        print(f"🍃 IGNORING (leaf): {prim_path}")
                else:
                    collision_api.GetCollisionEnabledAttr().Set(True)
                    print(f"🚫 AVOIDING: {prim_path}")

                # For mesh prims, add mesh collision API
                if prim.IsA(UsdGeom.Mesh):
                    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    else:
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
                    mesh_collision_api.GetApproximationAttr().Set("convexHull")

        # Recursively apply to all children
        for child in prim.GetAllChildren():
            self._configure_prim_physics(child)

    def _setup_target(self) -> None:
        """Setup target position and orientation from the specified prim path."""
        debug_print(f"Setting target to prim: {self.target_prim_path}", "info")
        self._update_target_position()

        if self.target_position is not None and self.target_orientation is not None:
            debug_print(f"Target position set to: {self.target_position}", "info")
            debug_print(f"Target orientation set to: {self.target_orientation}", "info")
        else:
            debug_print(
                "WARNING: Could not get target position or orientation from prim",
                "warning",
            )

    def _update_target_position(self) -> None:
        """Update the target position and orientation from the specified prim."""
        stage = self.my_world.stage
        target_prim = stage.GetPrimAtPath(self.target_prim_path)

        if target_prim and target_prim.IsValid():
            # Get world transform matrix
            target_xformable = UsdGeom.Xformable(target_prim)
            world_transform = target_xformable.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            translation = world_transform.ExtractTranslation()
            rotation = world_transform.ExtractRotationQuat()

            # Store actual target position
            self.target_position = np.array([
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
            ])

            # Check if geometry has changed significantly
            if self.last_target_position is not None:
                distance_moved = np.linalg.norm(
                    self.target_position - self.last_target_position
                )
                if distance_moved > 0.15:  # 15cm threshold
                    debug_print(
                        f"Target moved {distance_moved:.3f}m > 0.15m - geometry changed",
                        "info",
                    )
                    self.geometry_changed = True

            self.last_target_position = self.target_position.copy()

            # Store orientation as quaternion (w, x, y, z)
            extracted_orientation = np.array([
                float(rotation.GetReal()),  # w
                float(rotation.GetImaginary()[0]),  # x
                float(rotation.GetImaginary()[1]),  # y
                float(rotation.GetImaginary()[2]),  # z
            ])

            # Validate and normalize quaternion
            quat_norm = np.linalg.norm(extracted_orientation)
            if quat_norm > 0.0001:
                self.target_orientation = extracted_orientation / quat_norm
            else:
                debug_print(
                    "Invalid orientation extracted, using identity quaternion",
                    "warning",
                )
                self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        else:
            debug_print(
                f"WARNING: Invalid target prim: {self.target_prim_path}", "warning"
            )

    def _setup_motion_gen(self) -> None:
        """Setup CuRobo motion generation with plant obstacle configuration."""
        debug_print("Setting up motion generation...", "info")

        # Load and modify robot configuration
        robot_cfg_path = get_robot_configs_path()
        robot_cfg = load_yaml(join_path(robot_cfg_path, "franka.yml"))["robot_cfg"]
        robot_cfg["kinematics"]["ee_link"] = "ee_link"  # Franka end-effector

        robot_cfg["kinematics"]["cspace"]["joint_names"] = self.j_names
        robot_cfg["kinematics"]["cspace"]["retract_config"] = self.default_config
        robot_cfg["kinematics"]["cspace"]["null_space_weight"] = [1] * 7
        robot_cfg["kinematics"]["cspace"]["cspace_distance_weight"] = [1] * 7

        debug_print(
            "Modified robot config for Franka arm-only planning (7 DOF)", "info"
        )

        # Create a proper world configuration to ensure collision checker initialization

        # Create an empty world config that can be populated with obstacles dynamically
        # This is the correct approach instead of trying to use create_obb_world() with None
        world_cfg = WorldConfig(
            cuboid=[],  # Empty list of cuboids (oriented bounding boxes)
            mesh=[],  # Empty list of meshes
            sphere=[],  # Empty list of spheres
            capsule=[],  # Empty list of capsules
            cylinder=[],  # Empty list of cylinders
            blox=[],  # Empty list of blox objects
            voxel=[],  # Empty list of voxel grids
        )
        debug_print(
            "Created empty world configuration for dynamic obstacle loading", "info"
        )

        # Configure for reactive mode with USD collision meshes and self-collision
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            self_collision_check=True,
            self_collision_opt=False,
            position_threshold=0.05,  # 5cm tolerance
            rotation_threshold=5.0,  # 5 degree tolerance
            cspace_threshold=0.2,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            interpolation_dt=0.06,
            interpolation_steps=2000,
            collision_cache={"obb": 20, "mesh": 50},
            num_ik_seeds=20,
            use_cuda_graph=True,
            store_trajopt_debug=True,
        )

        # Initialize CuRobo motion generator
        self.motion_gen = MotionGen(motion_gen_config)
        debug_print("CuRobo motion generator initialized", "info")

        # Verify collision checker was properly initialized
        if hasattr(self.motion_gen, "world_coll_checker"):
            if self.motion_gen.world_coll_checker is not None:
                debug_print(
                    "✅ World collision checker initialized successfully", "info"
                )

                # Test basic collision checker functionality
                try:
                    # Try to check collision for a dummy state
                    dummy_js = JointState(
                        position=self.tensor_args.to_device(self.default_config),
                        velocity=self.tensor_args.to_device(
                            [0.0] * len(self.default_config)
                        ),
                        acceleration=self.tensor_args.to_device(
                            [0.0] * len(self.default_config)
                        ),
                        jerk=self.tensor_args.to_device(
                            [0.0] * len(self.default_config)
                        ),
                        joint_names=self.j_names,
                    ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

                    test_result = self.motion_gen.world_coll_checker.check_collision(
                        dummy_js.unsqueeze(0)
                    )
                    if test_result is not None:
                        debug_print(
                            "✅ Collision checker is functional and ready", "info"
                        )
                    else:
                        debug_print(
                            "⚠️ Collision checker returns None - may need obstacles loaded",
                            "warning",
                        )

                except Exception as test_error:
                    debug_print(
                        f"⚠️ Collision checker test failed: {test_error}", "warning"
                    )
            else:
                debug_print(
                    "❌ World collision checker is None after MotionGen init", "error"
                )
                raise RuntimeError(
                    "Collision checker failed to initialize - cannot proceed safely"
                )
        else:
            debug_print("❌ MotionGen has no world_coll_checker attribute", "error")
            raise RuntimeError("MotionGen missing collision checker attribute")

        # Planning config for graph planner
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=0,
            max_attempts=1,
            enable_finetune_trajopt=False,
            time_dilation_factor=1.0,
        )
        debug_print("Motion generation plan config set for graph planning", "info")

        # USD stage helper
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.my_world.stage)

        debug_print(
            "Motion generation setup complete - collision checking enabled", "info"
        )

        # Load plant obstacles into CuRobo after collision checker is initialized
        debug_print("Loading plant obstacles into CuRobo...", "info")
        self._update_world_obstacles()

    def _enable_collision_visualization(self) -> None:
        """Enable collision mesh visualization in Isaac Sim viewport."""
        try:
            import omni.kit.commands

            omni.kit.commands.execute(
                "ToggleVisibilityCommand", prim_paths=[], physics_collision=True
            )
            debug_print("Collision visualization enabled", "info")
        except Exception:
            try:
                import carb.settings

                settings = carb.settings.get_settings()
                settings.set("/physics/visualization/collision", True)
                settings.set("/physics/visualization/collisionMeshes", True)
                debug_print("Collision visualization enabled via carb.settings", "info")
            except Exception:
                debug_print("Failed to enable collision visualization", "error")

    def _update_world_obstacles(self) -> None:
        """Update plant obstacles from USD stage for collision avoidance."""
        debug_print("Updating world obstacles...", "info")
        try:
            # Clear collision cache if geometry changed (with null check)
            if self.geometry_changed:
                if (
                    hasattr(self.motion_gen, "world_coll_checker")
                    and self.motion_gen.world_coll_checker is not None
                ):
                    self.motion_gen.world_coll_checker.clear_cache()
                    debug_print(
                        "Cleared collision cache due to geometry change", "info"
                    )
                else:
                    debug_print(
                        "World collision checker not available for cache clear",
                        "warning",
                    )
                self.geometry_changed = False

            stage = self.my_world.stage
            plant_prim = stage.GetPrimAtPath(self.plant_prim_path)

            if not plant_prim or not plant_prim.IsValid():
                debug_print(f"Plant prim not found at {self.plant_prim_path}", "error")
                return

            # Let USD helper automatically detect plant obstacles
            self.usd_help.load_stage(self.my_world.stage)

            # Get obstacles with detailed debugging
            debug_print("Scanning USD stage for obstacles...", "info")
            obstacles = self.usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=self.robot_prim_path,
                ignore_substring=[
                    self.robot_prim_path,
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()

            if obstacles is not None and obstacles.objects is not None:
                num_obstacles = len(obstacles.objects)
                debug_print(
                    f"Found {num_obstacles} plant obstacles in USD stage", "info"
                )

                # List obstacles for debugging
                if args.debug and num_obstacles > 0:
                    debug_print("Plant obstacles detected:", "info")
                    for i, obj in enumerate(obstacles.objects[:5]):  # Show first 5
                        if hasattr(obj, "name"):
                            debug_print(f"  - Obstacle {i + 1}: {obj.name}", "info")
                        else:
                            debug_print(
                                f"  - Obstacle {i + 1}: {type(obj).__name__}", "info"
                            )
                    if num_obstacles > 5:
                        debug_print(
                            f"  ... and {num_obstacles - 5} more obstacles", "info"
                        )

                # Update CuRobo world with obstacles
                try:
                    self.motion_gen.clear_world_cache()
                except AttributeError as e:
                    if "'NoneType' object has no attribute 'clear_cache'" in str(e):
                        debug_print(
                            "World collision checker not initialized yet, skipping cache clear",
                            "warning",
                        )
                    else:
                        debug_print(f"Error clearing world cache: {e}", "warning")
                except Exception as e:
                    debug_print(
                        f"Unexpected error clearing world cache: {e}", "warning"
                    )

                # Update world with collision obstacles
                if self._safe_update_world(obstacles):
                    debug_print(
                        f"✅ Updated CuRobo world with {num_obstacles} plant obstacles",
                        "info",
                    )
                else:
                    debug_print(
                        f"❌ Failed to update CuRobo world with {num_obstacles} plant obstacles",
                        "error",
                    )

                # Update target position
                self._update_target_position()

            else:
                debug_print(
                    "❌ No obstacles found in USD stage - collision avoidance may not work",
                    "warning",
                )

                # Debug: Check if plant has collision-enabled prims
                if args.debug:
                    debug_print("Checking plant for collision-enabled prims...", "info")
                    collision_count = 0
                    for prim in plant_prim.GetAllChildren():
                        if self._prim_has_collision_enabled(prim):
                            collision_count += 1
                    debug_print(
                        f"Found {collision_count} collision-enabled prims in plant",
                        "info",
                    )

        except Exception as e:
            debug_print(f"Failed to get plant obstacles from USD stage: {e}", "error")
            import traceback

            if args.debug:
                debug_print(f"Full error traceback: {traceback.format_exc()}", "error")

    def _prim_has_collision_enabled(self, prim) -> bool:
        """Check if a prim has collision enabled (for debugging)."""
        try:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                if collision_api.GetCollisionEnabledAttr():
                    enabled = collision_api.GetCollisionEnabledAttr().Get()
                    return enabled if enabled is not None else False
            return False
        except:
            return False

    def _visualize_robot_spheres(self, cu_js: JointState, step_index: int) -> None:
        """Visualize robot collision spheres if enabled."""
        if not args.visualize_spheres or step_index % 10 != 0:
            return

        sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

        if self.spheres is None:
            self.spheres = []
            if len(sph_list) > 0 and hasattr(sph_list[0], "__iter__"):
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path=f"/curobo/robot_sphere_{si}",
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0.0, 0.8, 0.2]),
                    )
                    self.spheres.append(sp)
        elif len(sph_list) > 0 and hasattr(sph_list[0], "__iter__"):
            for si, s in enumerate(sph_list[0]):
                if si < len(self.spheres) and not np.isnan(s.position[0]):
                    self.spheres[si].set_world_pose(position=np.ravel(s.position))
                    self.spheres[si].set_radius(float(s.radius))

    def _plan_motion(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        cu_js: JointState,
    ) -> bool:
        """Plan motion with pre-grasp approach: move to offset position, then wait for user input."""
        debug_print(f"Planning motion to target position: {target_position}", "info")

        # SAFETY CHECK: Verify collision detection is working before planning
        if (
            not hasattr(self.motion_gen, "world_coll_checker")
            or self.motion_gen.world_coll_checker is None
        ):
            debug_print(
                "❌ CRITICAL: Motion planning attempted without collision checking!",
                "error",
            )
            debug_print(
                "❌ Robot would move in FREE SPACE without obstacle avoidance!", "error"
            )
            return False

        # Quick collision check to verify system is working
        try:
            # Get robot spheres for collision checking
            robot_spheres = self.motion_gen.kinematics.get_robot_as_spheres(
                cu_js.position
            )

            debug_print(f"Robot spheres type: {type(robot_spheres)}", "info")
            if robot_spheres and len(robot_spheres) > 0:
                debug_print(f"Robot spheres[0] type: {type(robot_spheres[0])}", "info")
                debug_print(f"Robot spheres length: {len(robot_spheres)}", "info")

                # Handle different possible structures
                sphere_list = robot_spheres[0]
                if isinstance(sphere_list, list):
                    debug_print(
                        f"Robot spheres[0] is list with length: {len(sphere_list)}",
                        "info",
                    )
                    if len(sphere_list) > 0:
                        debug_print(
                            f"Robot spheres[0][0] type: {type(sphere_list[0])}", "info"
                        )

                        # Convert Sphere objects to tensor format
                        sphere_data = []
                        for sphere in sphere_list:
                            if hasattr(sphere, "position") and hasattr(
                                sphere, "radius"
                            ):
                                # Convert position and radius to tensor format
                                pos = (
                                    sphere.position
                                    if hasattr(sphere.position, "shape")
                                    else torch.tensor(
                                        sphere.position, device=self.tensor_args.device
                                    )
                                )
                                rad = (
                                    sphere.radius
                                    if hasattr(sphere.radius, "shape")
                                    else torch.tensor(
                                        sphere.radius, device=self.tensor_args.device
                                    )
                                )

                                # Ensure both are tensors on the correct device
                                if not isinstance(pos, torch.Tensor):
                                    pos = torch.tensor(
                                        pos,
                                        device=self.tensor_args.device,
                                        dtype=self.tensor_args.dtype,
                                    )
                                else:
                                    pos = pos.to(
                                        device=self.tensor_args.device,
                                        dtype=self.tensor_args.dtype,
                                    )

                                if not isinstance(rad, torch.Tensor):
                                    rad = torch.tensor(
                                        rad,
                                        device=self.tensor_args.device,
                                        dtype=self.tensor_args.dtype,
                                    )
                                else:
                                    rad = rad.to(
                                        device=self.tensor_args.device,
                                        dtype=self.tensor_args.dtype,
                                    )

                                # Combine position [x,y,z] and radius [r] -> [x,y,z,r]
                                sphere_tensor_item = torch.cat([
                                    pos.flatten(),
                                    rad.flatten(),
                                ])
                                sphere_data.append(sphere_tensor_item)

                        if sphere_data:
                            # Stack all spheres into tensor [n_spheres, 4]
                            sphere_tensor = torch.stack(sphere_data, dim=0)
                            # Add batch and horizon dimensions [1, 1, n_spheres, 4]
                            sphere_tensor = sphere_tensor.unsqueeze(0).unsqueeze(0)
                        else:
                            debug_print(
                                "❌ COLLISION TEST FAILED: No valid sphere data",
                                "error",
                            )
                            return False
                else:
                    # If it's already a tensor, use it directly
                    sphere_tensor = sphere_list
                    if len(sphere_tensor.shape) == 2:  # [n_spheres, 4]
                        sphere_tensor = sphere_tensor.unsqueeze(0).unsqueeze(
                            0
                        )  # [1, 1, n_spheres, 4]
                    elif len(sphere_tensor.shape) == 3:  # [1, n_spheres, 4]
                        sphere_tensor = sphere_tensor.unsqueeze(
                            1
                        )  # [1, 1, n_spheres, 4]

                debug_print(f"Final sphere tensor shape: {sphere_tensor.shape}", "info")

                # Create collision query buffer
                from curobo.geom.sdf.world import CollisionQueryBuffer

                query_buffer = CollisionQueryBuffer.initialize_from_shape(
                    sphere_tensor.shape,
                    self.tensor_args,
                    self.motion_gen.world_coll_checker.collision_types,
                )

                # Test collision using correct method
                weight = self.tensor_args.to_device([1.0])
                activation_distance = self.tensor_args.to_device([0.02])

                current_collision = (
                    self.motion_gen.world_coll_checker.get_sphere_collision(
                        sphere_tensor,
                        query_buffer,
                        weight,
                        activation_distance,
                    )
                )

                if current_collision is None:
                    debug_print(
                        "❌ CRITICAL: Collision checker returns None - collision avoidance not working!",
                        "error",
                    )
                    return False
                debug_print(
                    "✅ Collision checking verified before motion planning", "info"
                )
            else:
                debug_print(
                    "⚠️ Could not get robot spheres for collision test", "warning"
                )
        except Exception as e:
            debug_print(
                f"❌ CRITICAL: Collision check failed before planning: {e}", "error"
            )
            return False

        # Define specific orientation constraint:
        # - Local x-axis [1,0,0] aligned with world -z [0,0,-1] (downward)
        # - Local z-axis [0,0,1] aligned with world x [1,0,0]
        # - Local y-axis [0,1,0] aligned with world y [0,1,0] (by right-hand rule)
        ee_orientation_constraint = np.array([0.7071068, 0.0, 0.7071068, 0.0])

        # Calculate pre-grasp position: offset target in negative x-direction
        pre_grasp_offset = 0.3  # 45cm offset in negative x-direction
        pre_grasp_position = target_position.copy()
        pre_grasp_position[0] -= pre_grasp_offset  # Move back in x-direction

        debug_print(
            f"Pre-grasp position: {pre_grasp_position} (offset by -{pre_grasp_offset}m in x)",
            "info",
        )
        debug_print("Phase 1: Planning motion to pre-grasp position...", "info")

        # Phase 1: Plan motion to pre-grasp position ONLY
        pre_grasp_goal = Pose(
            position=self.tensor_args.to_device(pre_grasp_position),
            quaternion=self.tensor_args.to_device(ee_orientation_constraint),
        )

        # Create planning configuration with full constraints for pre-grasp
        pre_grasp_planning = PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=self.tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )

        plan_config_pre_grasp = self.plan_config.clone()
        plan_config_pre_grasp.pose_cost_metric = pre_grasp_planning

        try:
            result = self.motion_gen.plan_single(
                cu_js.unsqueeze(0), pre_grasp_goal, plan_config_pre_grasp
            )
        except Exception as e:
            debug_print(f"Pre-grasp motion planning error: {e}", "error")
            return False

        # If strict constraint fails, try with relaxed orientation weights
        if result is None or not (result.success is not None and result.success.item()):
            debug_print(
                "Strict pre-grasp constraint failed, trying with relaxed weights...",
                "info",
            )

            relaxed_planning = PoseCostMetric(
                reach_partial_pose=True,
                reach_vec_weight=self.tensor_args.to_device([
                    0.5,
                    0.5,
                    0.5,
                    1.0,
                    1.0,
                    1.0,
                ]),
            )

            relaxed_config = self.plan_config.clone()
            relaxed_config.pose_cost_metric = relaxed_planning
            relaxed_config.max_attempts = 4

            try:
                result = self.motion_gen.plan_single(
                    cu_js.unsqueeze(0), pre_grasp_goal, relaxed_config
                )
            except Exception as e:
                debug_print(f"Relaxed pre-grasp motion planning error: {e}", "error")
                return False

        if result is not None and result.success is not None and result.success.item():
            debug_print("Pre-grasp motion planning SUCCESSFUL!", "info")

            # Get the pre-grasp trajectory ONLY - no approach planning yet
            pre_grasp_plan = result.get_interpolated_plan()
            pre_grasp_plan = self.motion_gen.get_full_js(pre_grasp_plan)

            # Store target info for later approach planning
            self.stored_target_position = target_position
            self.stored_target_orientation = ee_orientation_constraint
            self.stored_pre_grasp_position = pre_grasp_position
            self.waiting_for_user_input = True
            self.approach_planned = False

            debug_print(
                "Pre-grasp trajectory ready. Robot will stop at pre-grasp position and wait for user input.",
                "info",
            )
            print("\n" + "=" * 80)
            print("🤖 ROBOT WILL STOP AT PRE-GRASP POSITION")
            print(
                "⏸️  Press ENTER in the terminal after robot reaches pre-grasp to continue with approach..."
            )
            print("=" * 80 + "\n")

            # Use only pre-grasp plan for now
            self.cmd_plan = pre_grasp_plan
            self.num_targets += 1

            # Get joint mapping for arm joints only
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
            debug_print("Pre-grasp motion planning FAILED", "error")
            return False

    def _plan_approach_motion(self) -> bool:
        """Plan incremental collision-free motion along +X direction toward target."""
        if not self.waiting_for_user_input or self.approach_planned:
            return False

        debug_print("Phase 2: Planning incremental +X motion toward target...", "info")

        # Get current robot joint state (should be at pre-grasp position)
        try:
            sim_js = self.robot.get_joints_state()
            if sim_js is None:
                debug_print(
                    "Cannot get robot joint state for approach planning", "error"
                )
                return False

            # Create joint state with only arm joints
            arm_positions = []
            sim_js_names = self.robot.dof_names

            for joint_name in self.j_names:
                if joint_name in sim_js_names:
                    joint_idx = sim_js_names.index(joint_name)
                    arm_positions.append(sim_js.positions[joint_idx])

            current_js = JointState(
                position=self.tensor_args.to_device(arm_positions),
                velocity=self.tensor_args.to_device([0.0] * len(arm_positions)),
                acceleration=self.tensor_args.to_device([0.0] * len(arm_positions)),
                jerk=self.tensor_args.to_device([0.0] * len(arm_positions)),
                joint_names=self.j_names,
            ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

            # Get current end-effector pose
            ee_state = self.motion_gen.kinematics.compute_kinematics(current_js)
            start_position = ee_state.ee_pose.position.cpu().numpy().flatten()
            start_orientation = ee_state.ee_pose.quaternion.cpu().numpy().flatten()

            target_position = self.stored_target_position

            # Calculate approach parameters
            approach_distance = np.linalg.norm(target_position - start_position)
            approach_direction = (target_position - start_position) / approach_distance

            debug_print("PHASE 2 APPROACH PLANNING:", "info")
            debug_print(f"  Current EE position: {start_position}", "info")
            debug_print(f"  Target position: {target_position}", "info")
            debug_print(f"  Total approach distance: {approach_distance:.3f}m", "info")
            debug_print(
                f"  Approach direction (normalized): {approach_direction}", "info"
            )
            debug_print(
                f"  X-component (should be ~1.0): {approach_direction[0]:.3f}", "info"
            )

            # **ROBUST STEP-BY-STEP APPROACH:**
            # Instead of planning the entire trajectory, plan one small step at a time
            step_size = 0.02  # 2cm steps for safety
            num_steps = max(1, int(approach_distance / step_size))

            debug_print(
                f"Planning {num_steps} incremental steps of {step_size * 1000:.1f}mm each",
                "info",
            )

            # Plan just the first step toward target
            first_step_position = start_position + approach_direction * step_size

            debug_print(f"First step target: {first_step_position}", "info")
            debug_print(f"Step direction: {approach_direction * step_size}", "info")

            # Use simple IK planning for the first step
            step_goal = Pose(
                position=self.tensor_args.to_device(first_step_position),
                quaternion=self.tensor_args.to_device(
                    start_orientation
                ),  # Maintain orientation
            )

            # Create simple planning config for incremental motion
            step_planning_config = MotionGenPlanConfig(
                enable_graph=True,  # Use graph planning for collision avoidance
                enable_graph_attempt=0,
                max_attempts=2,
                enable_finetune_trajopt=False,  # Keep it simple
                time_dilation_factor=1.0,
                pose_cost_metric=PoseCostMetric(
                    reach_partial_pose=False,  # Full pose constraint
                    reach_vec_weight=self.tensor_args.to_device([
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]),
                ),
            )

            debug_print("Planning collision-free step toward target...", "info")

            try:
                # Plan collision-free motion for just this step
                step_result = self.motion_gen.plan_single(
                    current_js.unsqueeze(0), step_goal, step_planning_config
                )

                if (
                    step_result is not None
                    and step_result.success is not None
                    and step_result.success.item()
                ):
                    debug_print("✅ Step planning SUCCESSFUL!", "info")

                    # Get the planned step trajectory
                    step_plan = step_result.get_interpolated_plan()
                    step_plan = self.motion_gen.get_full_js(step_plan)

                    # Validate the step moves in correct direction
                    final_js = step_plan[-1]
                    final_js_reordered = final_js.get_ordered_joint_state(
                        self.motion_gen.kinematics.joint_names
                    )
                    final_ee_state = self.motion_gen.kinematics.compute_kinematics(
                        final_js_reordered
                    )
                    final_position = (
                        final_ee_state.ee_pose.position.cpu().numpy().flatten()
                    )

                    actual_step_direction = final_position - start_position
                    actual_step_distance = np.linalg.norm(actual_step_direction)

                    debug_print("STEP VALIDATION:", "info")
                    debug_print(
                        f"  Expected step: {approach_direction * step_size}", "info"
                    )
                    debug_print(f"  Actual step: {actual_step_direction}", "info")
                    debug_print(f"  Expected distance: {step_size:.3f}m", "info")
                    debug_print(
                        f"  Actual distance: {actual_step_distance:.3f}m", "info"
                    )

                    # Check if step is in correct direction (positive X)
                    if actual_step_direction[0] > 0.001:  # Moving in +X direction
                        debug_print("✅ Step moves in correct +X direction!", "info")

                        # Store step information for continuous planning
                        self.approach_step_size = step_size
                        self.approach_total_distance = approach_distance
                        self.approach_steps_taken = 0
                        self.approach_total_steps = num_steps

                        # Get joint mapping for arm joints only
                        sim_js_names = self.robot.dof_names
                        idx_list = []
                        common_js_names = []
                        for x in sim_js_names:
                            if x in step_plan.joint_names and x in self.j_names:
                                idx_list.append(self.robot.get_dof_index(x))
                                common_js_names.append(x)

                        self.cmd_plan = step_plan.get_ordered_joint_state(
                            common_js_names
                        )
                        self.cmd_idx = 0
                        self.approach_planned = True
                        self.waiting_for_user_input = False

                        debug_print(
                            "✅ Phase 2 step motion ready - executing incremental approach!",
                            "info",
                        )
                        print("\n" + "=" * 70)
                        print("🎯 INCREMENTAL +X APPROACH STARTED!")
                        print(f"📐 Moving {step_size * 1000:.1f}mm steps toward target")
                        print(f"🔢 Step 1/{num_steps} planned - more steps will follow")
                        print("🔒 Collision-free motion with orientation lock")
                        print("=" * 70 + "\n")

                        return True
                    else:
                        debug_print(
                            f"❌ Step moves in WRONG direction: {actual_step_direction}",
                            "error",
                        )
                        debug_print("This suggests coordinate frame issues", "error")
                        return False

                else:
                    debug_print("❌ Step planning FAILED", "error")
                    if step_result is not None and hasattr(step_result, "status"):
                        debug_print(f"Planning status: {step_result.status}", "error")

                    # Fallback: try even smaller step
                    smaller_step = step_size * 0.5  # 1cm step
                    debug_print(
                        f"Trying smaller step size: {smaller_step * 1000:.1f}mm", "info"
                    )

                    smaller_step_position = (
                        start_position + approach_direction * smaller_step
                    )
                    smaller_goal = Pose(
                        position=self.tensor_args.to_device(smaller_step_position),
                        quaternion=self.tensor_args.to_device(start_orientation),
                    )

                    fallback_result = self.motion_gen.plan_single(
                        current_js.unsqueeze(0), smaller_goal, step_planning_config
                    )

                    if (
                        fallback_result is not None
                        and fallback_result.success is not None
                        and fallback_result.success.item()
                    ):
                        debug_print("✅ Smaller step planning SUCCESSFUL!", "info")

                        # Use the smaller step
                        fallback_plan = fallback_result.get_interpolated_plan()
                        fallback_plan = self.motion_gen.get_full_js(fallback_plan)

                        self.approach_step_size = smaller_step
                        self.approach_total_distance = approach_distance
                        self.approach_steps_taken = 0
                        self.approach_total_steps = max(
                            1, int(approach_distance / smaller_step)
                        )

                        sim_js_names = self.robot.dof_names
                        common_js_names = []
                        for x in sim_js_names:
                            if x in fallback_plan.joint_names and x in self.j_names:
                                common_js_names.append(x)

                        self.cmd_plan = fallback_plan.get_ordered_joint_state(
                            common_js_names
                        )
                        self.cmd_idx = 0
                        self.approach_planned = True
                        self.waiting_for_user_input = False

                        debug_print("✅ Fallback step motion ready!", "info")
                        return True
                    else:
                        debug_print("❌ Even smaller step planning failed", "error")
                        return False

            except Exception as e:
                debug_print(f"Error in step planning: {e}", "error")
                return False

        except Exception as e:
            debug_print(f"Error in approach motion planning: {e}", "error")
            return False

    def _plan_next_approach_step(self) -> bool:
        """Plan the next incremental step toward target during Phase 2."""
        if not hasattr(self, "approach_steps_taken") or not hasattr(
            self, "approach_total_steps"
        ):
            debug_print("No approach step tracking available", "warning")
            return False

        try:
            # Get current robot state
            sim_js = self.robot.get_joints_state()
            if sim_js is None:
                debug_print("Cannot get robot joint state for next step", "error")
                return False

            # Create joint state with only arm joints
            arm_positions = []
            sim_js_names = self.robot.dof_names

            for joint_name in self.j_names:
                if joint_name in sim_js_names:
                    joint_idx = sim_js_names.index(joint_name)
                    arm_positions.append(sim_js.positions[joint_idx])

            current_js = JointState(
                position=self.tensor_args.to_device(arm_positions),
                velocity=self.tensor_args.to_device([0.0] * len(arm_positions)),
                acceleration=self.tensor_args.to_device([0.0] * len(arm_positions)),
                jerk=self.tensor_args.to_device([0.0] * len(arm_positions)),
                joint_names=self.j_names,
            ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

            # Get current end-effector position
            ee_state = self.motion_gen.kinematics.compute_kinematics(current_js)
            current_position = ee_state.ee_pose.position.cpu().numpy().flatten()
            current_orientation = ee_state.ee_pose.quaternion.cpu().numpy().flatten()

            target_position = self.stored_target_position

            # Calculate remaining distance and direction
            remaining_direction = target_position - current_position
            remaining_distance = np.linalg.norm(remaining_direction)

            # **FIX: Use distance-based termination instead of step count**
            if remaining_distance < 0.02:  # Within 2cm of target (as requested)
                debug_print(
                    f"✅ Reached target within 2cm tolerance! Distance: {remaining_distance * 1000:.1f}mm",
                    "info",
                )

                # **NEW: Set target reached state and create position holding plan**
                self.target_reached = True
                self.approach_planned = False

                # Create a position holding plan to maintain current pose
                try:
                    # Create a static plan that holds the current position
                    current_positions = current_js.position.unsqueeze(
                        0
                    )  # Add batch dimension

                    # Create a holding plan with multiple timesteps of the same position
                    hold_duration = 50  # Hold for 50 timesteps
                    hold_positions = current_positions.repeat(
                        hold_duration, 1
                    )  # [50, 7]
                    hold_velocities = torch.zeros_like(hold_positions)
                    hold_accelerations = torch.zeros_like(hold_positions)
                    hold_jerks = torch.zeros_like(hold_positions)

                    self.hold_position_plan = JointState(
                        position=hold_positions,
                        velocity=hold_velocities,
                        acceleration=hold_accelerations,
                        jerk=hold_jerks,
                        joint_names=current_js.joint_names,
                    )

                    debug_print(
                        "✅ Created position holding plan to maintain target pose",
                        "info",
                    )

                except Exception as hold_error:
                    debug_print(
                        f"Warning: Could not create position holding plan: {hold_error}",
                        "warning",
                    )
                    self.hold_position_plan = None

                return False

            # Safety check: prevent infinite steps
            if self.approach_steps_taken >= 50:  # Max 50 steps (1m total distance)
                debug_print(
                    f"⚠️ Reached maximum step limit (50 steps), stopping approach. Distance: {remaining_distance * 1000:.1f}mm",
                    "warning",
                )
                debug_print(
                    "Target may be unreachable or step size too small", "warning"
                )
                self.approach_planned = False
                return False

            # **NEW: Special handling when very close to target to prevent collision oscillation**
            if remaining_distance < 0.1:  # Within 10cm of target
                debug_print(
                    f"🎯 CLOSE TO TARGET: {remaining_distance * 1000:.1f}mm - using special close-approach mode",
                    "info",
                )

                # Check for collision oscillation (distance increasing)
                if hasattr(self, "last_remaining_distance"):
                    distance_change = remaining_distance - self.last_remaining_distance
                    if distance_change > 0.02:  # Distance increased by more than 2cm
                        debug_print(
                            f"⚠️ COLLISION OSCILLATION DETECTED: Distance increased by {distance_change * 1000:.1f}mm",
                            "warning",
                        )
                        debug_print(
                            f"Previous distance: {self.last_remaining_distance * 1000:.1f}mm, Current: {remaining_distance * 1000:.1f}mm",
                            "warning",
                        )

                        # **SOLUTION: Use collision-free IK mode when very close**
                        debug_print(
                            "🔧 Switching to collision-free IK mode for final approach",
                            "info",
                        )

                        # Plan directly to target without collision checking for final approach
                        direct_target_goal = Pose(
                            position=self.tensor_args.to_device(target_position),
                            quaternion=self.tensor_args.to_device(current_orientation),
                        )

                        # Use IK-only planning (no collision checking) for final approach
                        collision_free_config = MotionGenPlanConfig(
                            enable_graph=False,  # Disable graph planning
                            enable_graph_attempt=0,
                            max_attempts=1,
                            enable_finetune_trajopt=False,
                            time_dilation_factor=1.0,
                            pose_cost_metric=PoseCostMetric(
                                reach_partial_pose=False,
                                reach_vec_weight=self.tensor_args.to_device([
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                ]),
                            ),
                        )

                        # Try direct target approach
                        try:
                            direct_result = self.motion_gen.plan_single(
                                current_js.unsqueeze(0),
                                direct_target_goal,
                                collision_free_config,
                            )

                            if (
                                direct_result is not None
                                and direct_result.success is not None
                                and direct_result.success.item()
                            ):
                                debug_print(
                                    "✅ Direct target approach SUCCESSFUL!", "info"
                                )

                                direct_plan = direct_result.get_interpolated_plan()
                                direct_plan = self.motion_gen.get_full_js(direct_plan)

                                sim_js_names = self.robot.dof_names
                                common_js_names = []
                                for x in sim_js_names:
                                    if (
                                        x in direct_plan.joint_names
                                        and x in self.j_names
                                    ):
                                        common_js_names.append(x)

                                self.cmd_plan = direct_plan.get_ordered_joint_state(
                                    common_js_names
                                )
                                self.cmd_idx = 0
                                self.approach_steps_taken += 1

                                debug_print(
                                    f"✅ Direct approach step {self.approach_steps_taken} ready!",
                                    "info",
                                )
                                return True
                            else:
                                debug_print(
                                    "❌ Direct target approach failed, continuing with small steps",
                                    "warning",
                                )

                        except Exception as direct_error:
                            debug_print(
                                f"❌ Direct approach error: {direct_error}", "error"
                            )

                # Store current distance for oscillation detection
                self.last_remaining_distance = remaining_distance

                # Use smaller steps when close to target
                close_step_size = min(
                    0.01, remaining_distance * 0.5
                )  # 1cm or half remaining distance
                debug_print(
                    f"Using smaller step size when close: {close_step_size * 1000:.1f}mm",
                    "info",
                )

            else:
                # Normal step size when far from target
                close_step_size = self.approach_step_size

            # Normalize direction and plan next step
            remaining_direction_normalized = remaining_direction / remaining_distance
            next_step_size = min(close_step_size, remaining_distance)  # Don't overshoot
            next_step_position = (
                current_position + remaining_direction_normalized * next_step_size
            )

            self.approach_steps_taken += 1

            debug_print(
                f"Planning step {self.approach_steps_taken} (distance-based approach)",
                "info",
            )
            debug_print(f"  Current position: {current_position}", "info")
            debug_print(f"  Next step target: {next_step_position}", "info")
            debug_print(
                f"  Remaining distance: {remaining_distance * 1000:.1f}mm (target: <20mm)",
                "info",
            )
            debug_print(f"  Step size: {next_step_size * 1000:.1f}mm", "info")

            # **NEW: Show collision oscillation status**
            if hasattr(self, "last_remaining_distance"):
                distance_change = remaining_distance - self.last_remaining_distance
                debug_print(
                    f"  Distance change: {distance_change * 1000:+.1f}mm ({'approaching' if distance_change < 0 else 'retreating'})",
                    "info",
                )

            # Plan next step
            next_step_goal = Pose(
                position=self.tensor_args.to_device(next_step_position),
                quaternion=self.tensor_args.to_device(
                    current_orientation
                ),  # Maintain orientation
            )

            step_planning_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_graph_attempt=0,
                max_attempts=2,
                enable_finetune_trajopt=False,
                time_dilation_factor=1.0,
                pose_cost_metric=PoseCostMetric(
                    reach_partial_pose=False,
                    reach_vec_weight=self.tensor_args.to_device([
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]),
                ),
            )

            step_result = self.motion_gen.plan_single(
                current_js.unsqueeze(0), next_step_goal, step_planning_config
            )

            if (
                step_result is not None
                and step_result.success is not None
                and step_result.success.item()
            ):
                debug_print(
                    f"✅ Step {self.approach_steps_taken} planning SUCCESSFUL!", "info"
                )

                # Get the planned step trajectory
                step_plan = step_result.get_interpolated_plan()
                step_plan = self.motion_gen.get_full_js(step_plan)

                # Get joint mapping for arm joints only
                sim_js_names = self.robot.dof_names
                common_js_names = []
                for x in sim_js_names:
                    if x in step_plan.joint_names and x in self.j_names:
                        common_js_names.append(x)

                self.cmd_plan = step_plan.get_ordered_joint_state(common_js_names)
                self.cmd_idx = 0

                debug_print(
                    f"✅ Step {self.approach_steps_taken} ready for execution!", "info"
                )
                return True
            else:
                debug_print(
                    f"❌ Step {self.approach_steps_taken} planning FAILED", "error"
                )

                # Try smaller step if regular step fails
                if next_step_size > 0.005:  # If step is larger than 5mm
                    smaller_step_size = next_step_size * 0.5
                    smaller_step_position = (
                        current_position
                        + remaining_direction_normalized * smaller_step_size
                    )

                    debug_print(
                        f"Trying smaller step: {smaller_step_size * 1000:.1f}mm", "info"
                    )

                    smaller_goal = Pose(
                        position=self.tensor_args.to_device(smaller_step_position),
                        quaternion=self.tensor_args.to_device(current_orientation),
                    )

                    smaller_result = self.motion_gen.plan_single(
                        current_js.unsqueeze(0), smaller_goal, step_planning_config
                    )

                    if (
                        smaller_result is not None
                        and smaller_result.success is not None
                        and smaller_result.success.item()
                    ):
                        debug_print(
                            f"✅ Smaller step {self.approach_steps_taken} SUCCESSFUL!",
                            "info",
                        )

                        smaller_plan = smaller_result.get_interpolated_plan()
                        smaller_plan = self.motion_gen.get_full_js(smaller_plan)

                        common_js_names = []
                        for x in sim_js_names:
                            if x in smaller_plan.joint_names and x in self.j_names:
                                common_js_names.append(x)

                        self.cmd_plan = smaller_plan.get_ordered_joint_state(
                            common_js_names
                        )
                        self.cmd_idx = 0

                        # **UPDATED: Update the close step size instead of self.approach_step_size**
                        if remaining_distance < 0.1:  # Only update when close
                            debug_print(
                                f"Updated close step size to {smaller_step_size * 1000:.1f}mm for close approach",
                                "info",
                            )
                        else:
                            self.approach_step_size = smaller_step_size
                            debug_print(
                                f"Updated regular step size to {smaller_step_size * 1000:.1f}mm for future steps",
                                "info",
                            )

                        return True

                debug_print("❌ Both regular and smaller step planning failed", "error")
                return False

        except Exception as e:
            debug_print(f"Error planning next approach step: {e}", "error")
            return False

    def _verify_collision_detection_working(self) -> bool:
        """Comprehensive test to verify collision detection is actually working with plant obstacles."""
        try:
            debug_print(
                "🔍 TESTING: Verifying collision detection is working...", "info"
            )

            # Check if collision checker exists
            if (
                not hasattr(self.motion_gen, "world_coll_checker")
                or self.motion_gen.world_coll_checker is None
            ):
                debug_print(
                    "❌ COLLISION TEST FAILED: No collision checker available", "error"
                )
                return False

            # Get current robot state
            sim_js = self.robot.get_joints_state()
            if sim_js is None:
                debug_print("❌ COLLISION TEST FAILED: Cannot get robot state", "error")
                return False

            # Test 1: Current robot position (should be collision-free)
            sim_js_names = self.robot.dof_names
            current_js = JointState(
                position=self.tensor_args.to_device([
                    sim_js.positions[sim_js_names.index(x)]
                    for x in self.j_names
                    if x in sim_js_names
                ]),
                velocity=self.tensor_args.to_device([0.0] * len(self.j_names)),
                acceleration=self.tensor_args.to_device([0.0] * len(self.j_names)),
                jerk=self.tensor_args.to_device([0.0] * len(self.j_names)),
                joint_names=self.j_names,
            ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

            # Get robot spheres for current position
            robot_spheres = self.motion_gen.kinematics.get_robot_as_spheres(
                current_js.position
            )
            if not robot_spheres or len(robot_spheres) == 0:
                debug_print(
                    "❌ COLLISION TEST FAILED: No robot spheres available", "error"
                )
                return False

            # Convert Sphere objects to tensor format
            sphere_list = robot_spheres[0]
            if isinstance(sphere_list, list):
                # Convert Sphere objects to tensor format
                sphere_data = []
                for sphere in sphere_list:
                    if hasattr(sphere, "position") and hasattr(sphere, "radius"):
                        # Convert position and radius to tensor format
                        pos = (
                            sphere.position
                            if hasattr(sphere.position, "shape")
                            else torch.tensor(
                                sphere.position, device=self.tensor_args.device
                            )
                        )
                        rad = (
                            sphere.radius
                            if hasattr(sphere.radius, "shape")
                            else torch.tensor(
                                sphere.radius, device=self.tensor_args.device
                            )
                        )

                        # Ensure both are tensors on the correct device
                        if not isinstance(pos, torch.Tensor):
                            pos = torch.tensor(
                                pos,
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )
                        else:
                            pos = pos.to(
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )

                        if not isinstance(rad, torch.Tensor):
                            rad = torch.tensor(
                                rad,
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )
                        else:
                            rad = rad.to(
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )

                        # Combine position [x,y,z] and radius [r] -> [x,y,z,r]
                        sphere_tensor_item = torch.cat([pos.flatten(), rad.flatten()])
                        sphere_data.append(sphere_tensor_item)

                if sphere_data:
                    # Stack all spheres into tensor [n_spheres, 4]
                    sphere_tensor = torch.stack(sphere_data, dim=0)
                    # Add batch and horizon dimensions [1, 1, n_spheres, 4]
                    sphere_tensor = sphere_tensor.unsqueeze(0).unsqueeze(0)
                else:
                    debug_print(
                        "❌ COLLISION TEST FAILED: No valid sphere data", "error"
                    )
                    return False
            else:
                # If it's already a tensor, use it directly
                sphere_tensor = sphere_list
                if len(sphere_tensor.shape) == 2:  # [n_spheres, 4]
                    sphere_tensor = sphere_tensor.unsqueeze(0).unsqueeze(
                        0
                    )  # [1, 1, n_spheres, 4]
                elif len(sphere_tensor.shape) == 3:  # [1, n_spheres, 4]
                    sphere_tensor = sphere_tensor.unsqueeze(1)  # [1, 1, n_spheres, 4]

            # Create collision query buffer
            from curobo.geom.sdf.world import CollisionQueryBuffer

            query_buffer = CollisionQueryBuffer.initialize_from_shape(
                sphere_tensor.shape,
                self.tensor_args,
                self.motion_gen.world_coll_checker.collision_types,
            )

            # Test collision checking parameters
            weight = self.tensor_args.to_device([1.0])
            activation_distance = self.tensor_args.to_device([0.02])

            current_collision = self.motion_gen.world_coll_checker.get_sphere_collision(
                sphere_tensor, query_buffer, weight, activation_distance
            )

            if current_collision is None:
                debug_print(
                    "❌ COLLISION TEST FAILED: Collision checker returns None", "error"
                )
                return False

            # Check if result has collision information
            if hasattr(current_collision, "any"):
                current_in_collision = current_collision.any().item()
            else:
                # If it's just a tensor, check if any values are positive (indicating collision)
                current_in_collision = (current_collision > 0).any().item()

            debug_print(
                f"✅ Current robot position collision: {current_in_collision}", "info"
            )

            # Test 2: Try a configuration that should definitely be in collision
            # (move robot joints to extreme positions that would likely hit plant)
            collision_config = [
                1.5,
                -0.5,
                1.0,
                -1.5,
                1.0,
                1.5,
                0.5,
            ]  # Extreme joint positions

            collision_js = JointState(
                position=self.tensor_args.to_device(collision_config),
                velocity=self.tensor_args.to_device([0.0] * len(collision_config)),
                acceleration=self.tensor_args.to_device([0.0] * len(collision_config)),
                jerk=self.tensor_args.to_device([0.0] * len(collision_config)),
                joint_names=self.j_names,
            ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

            # Get robot spheres for extreme position
            extreme_spheres = self.motion_gen.kinematics.get_robot_as_spheres(
                collision_js.position
            )

            # Convert Sphere objects to tensor format (same as above)
            extreme_sphere_list = extreme_spheres[0]
            if isinstance(extreme_sphere_list, list):
                # Convert Sphere objects to tensor format
                extreme_sphere_data = []
                for sphere in extreme_sphere_list:
                    if hasattr(sphere, "position") and hasattr(sphere, "radius"):
                        # Convert position and radius to tensor format
                        pos = (
                            sphere.position
                            if hasattr(sphere.position, "shape")
                            else torch.tensor(
                                sphere.position, device=self.tensor_args.device
                            )
                        )
                        rad = (
                            sphere.radius
                            if hasattr(sphere.radius, "shape")
                            else torch.tensor(
                                sphere.radius, device=self.tensor_args.device
                            )
                        )

                        # Ensure both are tensors on the correct device
                        if not isinstance(pos, torch.Tensor):
                            pos = torch.tensor(
                                pos,
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )
                        else:
                            pos = pos.to(
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )

                        if not isinstance(rad, torch.Tensor):
                            rad = torch.tensor(
                                rad,
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )
                        else:
                            rad = rad.to(
                                device=self.tensor_args.device,
                                dtype=self.tensor_args.dtype,
                            )

                        # Combine position [x,y,z] and radius [r] -> [x,y,z,r]
                        sphere_tensor_item = torch.cat([pos.flatten(), rad.flatten()])
                        extreme_sphere_data.append(sphere_tensor_item)

                if extreme_sphere_data:
                    # Stack all spheres into tensor [n_spheres, 4]
                    extreme_sphere_tensor = torch.stack(extreme_sphere_data, dim=0)
                    # Add batch and horizon dimensions [1, 1, n_spheres, 4]
                    extreme_sphere_tensor = extreme_sphere_tensor.unsqueeze(
                        0
                    ).unsqueeze(0)
                else:
                    debug_print(
                        "❌ COLLISION TEST FAILED: No valid extreme sphere data",
                        "error",
                    )
                    return False
            else:
                # If it's already a tensor, use it directly
                extreme_sphere_tensor = extreme_sphere_list
                if len(extreme_sphere_tensor.shape) == 2:  # [n_spheres, 4]
                    extreme_sphere_tensor = extreme_sphere_tensor.unsqueeze(
                        0
                    ).unsqueeze(0)  # [1, 1, n_spheres, 4]
                elif len(extreme_sphere_tensor.shape) == 3:  # [1, n_spheres, 4]
                    extreme_sphere_tensor = extreme_sphere_tensor.unsqueeze(
                        1
                    )  # [1, 1, n_spheres, 4]

            extreme_collision = self.motion_gen.world_coll_checker.get_sphere_collision(
                extreme_sphere_tensor, query_buffer, weight, activation_distance
            )

            if hasattr(extreme_collision, "any"):
                extreme_in_collision = extreme_collision.any().item()
            else:
                extreme_in_collision = (extreme_collision > 0).any().item()

            debug_print(
                f"✅ Extreme position collision: {extreme_in_collision}", "info"
            )

            # Test 3: Check minimum distance to obstacles
            if hasattr(current_collision, "distance"):
                min_distance = current_collision.distance.min().item()
                debug_print(
                    f"✅ Minimum distance to obstacles: {min_distance:.3f}m", "info"
                )

                # If distance is very large, obstacles might not be loaded
                if min_distance > 10.0:
                    debug_print(
                        "⚠️ WARNING: Very large distance to obstacles suggests they may not be loaded",
                        "warning",
                    )
                    return False

            # Test 4: Verify we have obstacles loaded in world
            if hasattr(self.motion_gen.world_coll_checker, "world_model"):
                world_model = self.motion_gen.world_coll_checker.world_model
                if hasattr(world_model, "objects") and world_model.objects:
                    debug_print(
                        f"✅ World model has {len(world_model.objects)} collision objects loaded",
                        "info",
                    )

                    # List some obstacle info
                    for i, obj in enumerate(world_model.objects[:3]):  # Show first 3
                        if hasattr(obj, "name"):
                            debug_print(f"  - Obstacle {i + 1}: {obj.name}", "info")
                        else:
                            debug_print(
                                f"  - Obstacle {i + 1}: {type(obj).__name__}", "info"
                            )
                else:
                    debug_print(
                        "❌ COLLISION TEST FAILED: No collision objects in world model",
                        "error",
                    )
                    return False

            debug_print("✅ COLLISION DETECTION VERIFICATION PASSED", "info")
            print("\n" + "🟢" * 60)
            print("✅ COLLISION DETECTION IS WORKING!")
            print(f"   Current position in collision: {current_in_collision}")
            print(f"   Extreme position in collision: {extreme_in_collision}")
            if hasattr(current_collision, "distance"):
                print(f"   Minimum distance to obstacles: {min_distance:.3f}m")
            print("🟢" * 60 + "\n")

            return True

        except Exception as e:
            debug_print(f"❌ COLLISION TEST FAILED with error: {e}", "error")
            import traceback

            debug_print(f"Full error: {traceback.format_exc()}", "error")
            return False

    def _safe_update_world(self, obstacles) -> bool:
        """Safely update world obstacles, handling collision checker initialization."""
        try:
            # Check if collision checker needs to be initialized
            if (
                not hasattr(self.motion_gen, "world_coll_checker")
                or self.motion_gen.world_coll_checker is None
            ):
                debug_print("Initializing collision checker with obstacles...", "info")

                # For MotionGen initialized with world_cfg=None, the collision checker
                # is created during the first update_world() call
                self.motion_gen.update_world(obstacles)
                debug_print("✅ Collision checker initialized successfully", "info")
                return True
            else:
                # Normal update with existing collision checker
                self.motion_gen.update_world(obstacles)
                debug_print("✅ World obstacles updated successfully", "info")
                return True

        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                debug_print("❌ Collision checker initialization failed", "error")
                debug_print(
                    "This may be due to CuRobo version compatibility or configuration issues",
                    "error",
                )
                debug_print(f"Specific error: {e}", "error")
                return False
            else:
                debug_print(f"Unexpected AttributeError in world update: {e}", "error")
                return False
        except Exception as e:
            debug_print(f"Unexpected error updating world: {e}", "error")
            return False

    def run(self) -> None:
        """Run the main simulation loop."""
        debug_print("Starting main simulation loop...", "info")
        add_extensions(simulation_app, args.headless_mode)

        i = 0
        robot_initialized = False
        robot_not_ready_count = 0
        last_plan_complete_step = 0

        while simulation_app.is_running():
            self.my_world.step(render=True)

            if not self.my_world.is_playing():
                if i % 100 == 0:
                    print("**** Click Play to start simulation *****")
                i += 1
                continue

            step_index = self.my_world.current_time_step_index

            # Robot initialization
            if not robot_initialized:
                try:
                    if self.articulation_controller is None:
                        self.articulation_controller = (
                            self.robot.get_articulation_controller()
                        )
                        debug_print(
                            f"Articulation controller initialized at step {step_index}",
                            "info",
                        )

                    if step_index >= 5:
                        self.robot._articulation_view.initialize()
                        debug_print(
                            f"Robot articulation view initialized at step {step_index}",
                            "info",
                        )

                        # Set initial joint positions (arm + gripper)
                        all_joint_names = self.robot.dof_names
                        arm_indices = [
                            self.robot.get_dof_index(x) for x in self.j_names
                        ]
                        gripper_indices = [
                            self.robot.get_dof_index(x)
                            for x in all_joint_names
                            if "finger" in x.lower()
                        ]

                        # Set arm joints to default config
                        self.robot.set_joint_positions(self.default_config, arm_indices)

                        # Set gripper to fixed open position
                        if gripper_indices:
                            gripper_positions = [self.gripper_position] * len(
                                gripper_indices
                            )
                            self.robot.set_joint_positions(
                                gripper_positions, gripper_indices
                            )
                            debug_print(
                                f"Set gripper to position: {self.gripper_position}",
                                "info",
                            )

                        # Set max efforts for all joints
                        all_indices = arm_indices + gripper_indices
                        self.robot._articulation_view.set_max_efforts(
                            values=np.array([5000 for _ in range(len(all_indices))]),
                            joint_indices=all_indices,
                        )

                        # Mark as initialized
                        test_js = self.robot.get_joints_state()
                        if test_js is not None:
                            robot_initialized = True
                            debug_print(
                                f"Robot fully initialized at step {step_index}", "info"
                            )

                except Exception as init_error:
                    debug_print(
                        f"Robot initialization error at step {step_index}: {init_error}",
                        "error",
                    )
                    if step_index > 50:
                        robot_initialized = False
                        self.articulation_controller = None

                if not robot_initialized:
                    continue

            # Only proceed with motion planning if robot is ready
            if step_index < 30:
                continue

            # Update plant obstacles less frequently
            if step_index == 50 or step_index % 500 == 0:
                debug_print(f"Updating plant obstacles at step {step_index}", "info")
                self._update_world_obstacles()

                # Test collision detection after initial obstacle load (debug mode only)
                if args.debug and step_index == 50:
                    self._verify_collision_detection_working()

            # Get target position
            if self.target_position is None:
                debug_print(
                    "Target position not available, skipping planning", "warning"
                )
                continue

            target_position = self.target_position
            target_orientation = (
                self.target_orientation
                if self.target_orientation is not None
                else np.array([1.0, 0.0, 0.0, 0.0])
            )

            # Initialize tracking variables
            if self.past_pose is None:
                debug_print("Initializing tracking variables", "info")
                self.past_pose = target_position
                self.target_pose = target_position
                self.stored_target_orientation = target_orientation
                self.past_orientation = target_orientation

            # Get robot joint state with error handling
            try:
                sim_js = self.robot.get_joints_state()
                if sim_js is None:
                    debug_print(
                        "Robot joint state is None - robot may need re-initialization",
                        "warning",
                    )
                    robot_initialized = False
                    continue

            except Exception as js_error:
                debug_print(f"Error getting robot joint state: {js_error}", "error")
                robot_initialized = False
                continue

            sim_js_names = self.robot.dof_names
            if np.any(np.isnan(sim_js.positions)):
                debug_print(
                    "ERROR: Isaac Sim returned NaN joint position values", "error"
                )
                continue

            # Create joint state with only arm joints for motion planning
            arm_positions = []
            arm_velocities = []

            for joint_name in self.j_names:
                if joint_name in sim_js_names:
                    joint_idx = sim_js_names.index(joint_name)
                    arm_positions.append(sim_js.positions[joint_idx])
                    arm_velocities.append(sim_js.velocities[joint_idx])

            cu_js = JointState(
                position=self.tensor_args.to_device(arm_positions),
                velocity=self.tensor_args.to_device(arm_velocities),
                acceleration=self.tensor_args.to_device([0.0] * len(arm_positions)),
                jerk=self.tensor_args.to_device([0.0] * len(arm_positions)),
                joint_names=self.j_names,
            )

            # Use previous command for reactive mode
            if self.past_cmd is not None:
                cu_js.position[:] = self.past_cmd.position
                cu_js.velocity[:] = self.past_cmd.velocity
                cu_js.acceleration[:] = self.past_cmd.acceleration

            cu_js = cu_js.get_ordered_joint_state(
                self.motion_gen.kinematics.joint_names
            )

            # Visualize robot collision spheres
            self._visualize_robot_spheres(cu_js, step_index)

            # Calculate robot velocity metrics
            max_velocity = (
                max(abs(v) for v in arm_velocities) if arm_velocities else 0.0
            )
            robot_ready_strict = max_velocity < 0.1

            # Track robot not ready state
            if not robot_ready_strict:
                robot_not_ready_count += 1
            else:
                robot_not_ready_count = 0

            # Planning conditions
            target_moved = np.linalg.norm(target_position - self.target_pose) > 0.02
            should_plan = False
            plan_reason = ""

            dwell_time = 200
            in_dwell_period = (step_index - last_plan_complete_step) < dwell_time

            # Check distance to target
            distance_to_target = float("inf")
            robot_at_target = False

            if hasattr(self.motion_gen, "kinematics") and hasattr(self, "target_pose"):
                try:
                    current_js = JointState(
                        position=self.tensor_args.to_device([
                            sim_js.positions[sim_js_names.index(x)]
                            for x in self.j_names
                            if x in sim_js_names
                        ]),
                        velocity=self.tensor_args.to_device([0.0] * len(self.j_names)),
                        acceleration=self.tensor_args.to_device(
                            [0.0] * len(self.j_names)
                        ),
                        jerk=self.tensor_args.to_device([0.0] * len(self.j_names)),
                        joint_names=self.j_names,
                    ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

                    ee_state = self.motion_gen.kinematics.compute_kinematics(current_js)
                    current_ee_position = (
                        ee_state.ee_pose.position.cpu().numpy().flatten()
                    )
                    distance_to_target = np.linalg.norm(
                        current_ee_position - target_position
                    )
                    robot_at_target = (
                        distance_to_target < 0.02
                    )  # Changed from 0.15 to 0.02 (2cm)

                except Exception:
                    robot_at_target = False
                    distance_to_target = float("inf")

            # Planning decision logic
            if step_index == 50:
                should_plan = True
                plan_reason = "Initial startup planning"
            elif (
                self.cmd_plan is None
                and not robot_at_target
                and robot_ready_strict
                and not in_dwell_period
                and not getattr(
                    self, "target_reached", False
                )  # **NEW: Don't plan if target reached**
            ):
                should_plan = True
                plan_reason = f"No active plan, robot far from target ({distance_to_target:.3f}m) and robot ready"

            if step_index % 500 == 0:
                target_reached_status = getattr(self, "target_reached", False)
                debug_print(
                    f"Step {step_index}: Robot ready={robot_ready_strict}, Planning={should_plan}, "
                    f"Active plan={self.cmd_plan is not None}, At target={robot_at_target} (dist={distance_to_target:.3f}m), "
                    f"Target reached={target_reached_status}"
                )

            if should_plan:
                debug_print(f"Planning at step {step_index}: {plan_reason}", "info")

                # Stop robot before planning if velocities are too high
                if max_velocity > 1.0:
                    debug_print(
                        "Robot velocities too high, stopping robot before planning...",
                        "info",
                    )
                    try:
                        arm_indices = [
                            self.robot.get_dof_index(x) for x in self.j_names
                        ]
                        gripper_indices = [
                            self.robot.get_dof_index(x)
                            for x in sim_js_names
                            if "finger" in x.lower()
                        ]

                        arm_positions = [
                            sim_js.positions[sim_js_names.index(x)]
                            for x in self.j_names
                            if x in sim_js_names
                        ]
                        zero_arm_velocities = [0.0] * len(arm_positions)

                        all_positions = arm_positions + [self.gripper_position] * len(
                            gripper_indices
                        )
                        all_velocities = zero_arm_velocities + [0.0] * len(
                            gripper_indices
                        )
                        all_indices = arm_indices + gripper_indices

                        stop_action = ArticulationAction(
                            np.array(all_positions),
                            np.array(all_velocities),
                            joint_indices=all_indices,
                        )
                        self.articulation_controller.apply_action(stop_action)

                        debug_print(
                            "Stopping robot, will try planning next step", "info"
                        )
                        continue

                    except Exception as stop_error:
                        debug_print(f"Failed to stop robot: {stop_error}", "error")

                success = self._plan_motion(target_position, target_orientation, cu_js)
                if success:
                    debug_print(
                        "Motion plan SUCCESSFUL - navigating to target while avoiding plant",
                        "info",
                    )
                    robot_not_ready_count = 0
                else:
                    debug_print(
                        "Motion planning FAILED - plant obstacles may be blocking path",
                        "error",
                    )

                self.target_pose = target_position.copy()
                self.stored_target_orientation = target_orientation.copy()

            # Update tracking
            self.past_pose = target_position.copy()
            self.past_orientation = target_orientation.copy()

            # Execute motion plan
            if self.cmd_plan is not None:
                cmd_state = self.cmd_plan[self.cmd_idx]
                past_cmd = cmd_state.clone()

                # Prepare action for arm + gripper
                arm_positions = cmd_state.position.cpu().numpy()
                arm_velocities = cmd_state.velocity.cpu().numpy()

                # Get gripper indices
                gripper_indices = [
                    self.robot.get_dof_index(x)
                    for x in sim_js_names
                    if "finger" in x.lower()
                ]

                # Add gripper positions
                all_positions = list(arm_positions) + [self.gripper_position] * len(
                    gripper_indices
                )
                all_velocities = list(arm_velocities) + [0.0] * len(gripper_indices)

                # Get all joint indices (arm + gripper)
                all_idx_list = [
                    self.robot.get_dof_index(x)
                    for x in sim_js_names
                    if x in self.j_names
                ] + gripper_indices

                art_action = ArticulationAction(
                    np.array(all_positions),
                    np.array(all_velocities),
                    joint_indices=all_idx_list,
                )
                print("ART_ACTION", art_action)
                self.articulation_controller.apply_action(art_action)

                # **NEW: Real-time collision visualization during motion execution**
                if step_index % 10 == 0:  # Update every 10 steps for performance
                    try:
                        # Get current end-effector position for collision visualization
                        current_cmd_js = JointState(
                            position=self.tensor_args.to_device(arm_positions),
                            velocity=self.tensor_args.to_device(
                                [0.0] * len(arm_positions)
                            ),
                            acceleration=self.tensor_args.to_device(
                                [0.0] * len(arm_positions)
                            ),
                            jerk=self.tensor_args.to_device([0.0] * len(arm_positions)),
                            joint_names=self.j_names,
                        ).get_ordered_joint_state(
                            self.motion_gen.kinematics.joint_names
                        )

                        current_ee_state = (
                            self.motion_gen.kinematics.compute_kinematics(
                                current_cmd_js
                            )
                        )
                        current_ee_position = (
                            current_ee_state.ee_pose.position.cpu().numpy().flatten()
                        )

                        # Visualize collision direction from current EE position
                        self._visualize_collision_direction(current_ee_position)
                    except Exception as viz_error:
                        debug_print(
                            f"Collision visualization error: {viz_error}", "warning"
                        )

                self.cmd_idx += 1

                # Reset plan when complete
                if self.cmd_idx >= len(self.cmd_plan.position):
                    self.cmd_idx = 0
                    self.cmd_plan = None
                    past_cmd = None
                    last_plan_complete_step = step_index

                    # **NEW: Check if we're in target reached mode and restart position holding**
                    if getattr(self, "target_reached", False):
                        debug_print(
                            "🔒 Position holding cycle complete, restarting...", "info"
                        )

                        if (
                            hasattr(self, "hold_position_plan")
                            and self.hold_position_plan is not None
                        ):
                            # Restart the position holding plan
                            sim_js_names = self.robot.dof_names
                            common_js_names = []
                            for x in sim_js_names:
                                if (
                                    x in self.hold_position_plan.joint_names
                                    and x in self.j_names
                                ):
                                    common_js_names.append(x)

                            self.cmd_plan = (
                                self.hold_position_plan.get_ordered_joint_state(
                                    common_js_names
                                )
                            )
                            self.cmd_idx = 0
                            debug_print("🔒 Restarted position holding plan", "info")
                        else:
                            debug_print(
                                "⚠️ No position holding plan to restart", "warning"
                            )
                    # Check if we just completed pre-grasp and should wait for user input
                    elif self.waiting_for_user_input and not self.approach_planned:
                        debug_print(
                            "✅ Pre-grasp motion COMPLETE! Robot stopped at pre-grasp position.",
                            "info",
                        )
                        print("\n" + "🟡" * 60)
                        print("🛑 ROBOT STOPPED AT PRE-GRASP POSITION")
                        print("👀 Robot is positioned and ready for approach")
                        print(
                            "⌨️  Press ENTER in terminal to continue with final approach..."
                        )
                        print("🟡" * 60 + "\n")
                    # **NEW: Check if we completed a Phase 2 approach step and plan next step**
                    elif self.approach_planned and hasattr(
                        self, "approach_steps_taken"
                    ):
                        debug_print(
                            f"✅ Phase 2 step {self.approach_steps_taken} COMPLETE!",
                            "info",
                        )

                        # Automatically plan next step
                        if self._plan_next_approach_step():
                            debug_print(
                                "✅ Next approach step planned automatically", "info"
                            )
                        else:
                            debug_print(
                                "✅ Phase 2 approach COMPLETE - reached target!", "info"
                            )
                            print("\n" + "🎯" * 60)
                            print("🎉 TARGET REACHED!")
                            print(
                                "✅ Phase 2 incremental approach completed successfully"
                            )
                            print("🤖 Robot has reached the target position")
                            print("🔒 Holding position at target...")
                            print("🎯" * 60 + "\n")

                            # **NEW: Start position holding if available**
                            if (
                                hasattr(self, "hold_position_plan")
                                and self.hold_position_plan is not None
                            ):
                                # Get joint mapping for position holding plan
                                sim_js_names = self.robot.dof_names
                                common_js_names = []
                                for x in sim_js_names:
                                    if (
                                        x in self.hold_position_plan.joint_names
                                        and x in self.j_names
                                    ):
                                        common_js_names.append(x)

                                self.cmd_plan = (
                                    self.hold_position_plan.get_ordered_joint_state(
                                        common_js_names
                                    )
                                )
                                self.cmd_idx = 0
                                debug_print("✅ Started position holding plan", "info")
                            else:
                                debug_print(
                                    "⚠️ No position holding plan available", "warning"
                                )
                    else:
                        debug_print("Motion plan execution COMPLETE", "info")

            # Handle user input when waiting at pre-grasp position
            if (
                self.waiting_for_user_input
                and not self.approach_planned
                and self.cmd_plan is None
            ):
                # Check for user input (non-blocking)
                import select
                import sys

                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    user_input = sys.stdin.readline().strip()
                    debug_print(
                        "User input received! Planning approach motion...", "info"
                    )

                    # Plan and execute approach motion
                    if self._plan_approach_motion():
                        debug_print("Approach motion planned successfully!", "info")
                    else:
                        debug_print("Failed to plan approach motion", "error")
                        # Reset waiting state on failure
                        self.waiting_for_user_input = False

            self._visualize_collision_direction(target_position)

    def _visualize_collision_direction(self, ee_position: np.ndarray) -> None:
        """Visualize collision direction from end-effector to nearest obstacle."""
        try:
            # During Phase 2 approach, show direction toward target
            if self.approach_planned and hasattr(self, "stored_target_position"):
                # Show direction toward target (Phase 2 approach direction)
                target_direction = self.stored_target_position - ee_position
                target_direction_norm = np.linalg.norm(target_direction)

                if target_direction_norm > 0.01:  # Valid direction
                    normalized_direction = target_direction / target_direction_norm
                    draw_collision_line(
                        ee_position, normalized_direction * 0.2
                    )  # Scale for visibility
                    debug_print(
                        f"Phase 2: EE moving toward target, direction: {normalized_direction}, distance: {target_direction_norm:.3f}m",
                        "info",
                    )
                else:
                    clear_collision_lines()
                return

            # For Phase 1 or general collision detection, use collision-based visualization
            # Create sphere at end-effector position for collision checking
            sphere_radius = 0.05  # 5cm radius for end-effector sphere
            x_sph = torch.zeros(
                (1, 1, 1, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            x_sph[..., :3] = self.tensor_args.to_device(ee_position).view(1, 1, 1, 3)
            x_sph[..., 3] = sphere_radius

            # Get collision distance and direction vector
            if (
                hasattr(self.motion_gen, "world_coll_checker")
                and self.motion_gen.world_coll_checker is not None
            ):
                # Check if collision checker has get_collision_vector method
                if hasattr(self.motion_gen.world_coll_checker, "get_collision_vector"):
                    # Use the collision checker to get distance and direction
                    d, d_vec = self.motion_gen.world_coll_checker.get_collision_vector(
                        x_sph
                    )

                    if d is not None and d_vec is not None:
                        d_value = d.view(-1).cpu().item()

                        if (
                            d_value > 0.0 and d_value < 1.0
                        ):  # Only show if reasonably close to obstacles
                            collision_direction = d_vec[..., :3].view(3).cpu().numpy()
                            draw_collision_line(ee_position, collision_direction)
                            debug_print(
                                f"EE collision distance: {d_value:.3f}m, direction: {collision_direction}",
                                "info",
                            )
                        else:
                            clear_collision_lines()
                    else:
                        debug_print("No collision vector data available", "warning")
                        clear_collision_lines()
                # Alternative approach: Show direction toward target if available
                elif (
                    hasattr(self, "target_position")
                    and self.target_position is not None
                ):
                    target_direction = self.target_position - ee_position
                    target_direction_norm = np.linalg.norm(target_direction)

                    if target_direction_norm > 0.01:  # Valid direction
                        normalized_direction = target_direction / target_direction_norm
                        draw_collision_line(
                            ee_position, normalized_direction * 0.2
                        )  # Scale for visibility
                        debug_print(
                            f"EE moving toward target, direction: {normalized_direction}, distance: {target_direction_norm:.3f}m",
                            "info",
                        )
                    else:
                        clear_collision_lines()
            else:
                debug_print(
                    "No collision checker available for visualization", "warning"
                )
                clear_collision_lines()

        except Exception as e:
            debug_print(f"Error visualizing collision direction: {e}", "error")
            clear_collision_lines()

    def _sample_collision_direction(
        self, ee_position: np.ndarray, radius: float
    ) -> np.ndarray:
        """Sample directions around the end-effector to find collision direction."""
        # NOTE: This method is no longer used - we now show direction toward target
        # instead of direction away from collision
        debug_print(
            "Note: Collision sampling method deprecated - showing target direction instead",
            "info",
        )
        return None


def main() -> None:
    """Main function to run the motion generation example."""
    try:
        print("=== STARTING FRANKA MOTION GENERATION ===")
        debug_print("Initializing motion generation example...", "info")
        example = FrankaPlantMotionGenExample()
        print("=== INITIALIZATION COMPLETE ===")
        print("Starting Franka plant motion generation example...")
        print(
            "Robot will plan collision-free paths to target prim while avoiding plant structure!"
        )

        if args.visualize_collisions:
            print("🔍 Collision mesh visualization is ENABLED")
        else:
            print("💡 Run with --visualize_collisions to see collision meshes")

        if args.visualize_spheres:
            print("🟢 Robot collision sphere visualization is ENABLED")
        else:
            print("💡 Run with --visualize_spheres to see robot collision spheres")

        print("\n🚀 Starting simulation...")
        example.run()
    except Exception as e:
        print(f"❌ ERROR in motion generation example: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        print("=== CLOSING SIMULATION ===")
        simulation_app.close()


if __name__ == "__main__":
    main()
