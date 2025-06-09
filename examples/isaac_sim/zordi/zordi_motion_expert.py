"""
Expert motion policy for two-phase reactive collision-free motion generation using CuRobo.
This module provides a reusable expert policy that can be called to generate motions
for reaching target primitives in a plant environment with collision avoidance.

Two-Phase Approach:
- Phase 1: Move to pre-grasp position (15cm offset from target) - FAST
- Phase 2: Incremental approach toward target - SLOW & PRECISE

Usage:
    expert = ZordiMotionExpert(motion_gen, robot, tensor_args)
    action = expert.get_action(target_prim_path, current_world_stage)
"""

# Standard Library
from enum import Enum
from typing import Optional, Tuple

import numpy as np

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenPlanConfig,
    PoseCostMetric,
)

# Isaac Sim imports
from omni.isaac.core.utils.types import ArticulationAction

# USD imports
from pxr import Usd, UsdGeom

# Third Party
from scipy.spatial.transform import Rotation as R

from isaacsim.core.utils.xforms import get_world_pose


class MotionPhase(Enum):
    """Enumeration of motion phases for the expert policy."""

    IDLE = "idle"
    PLANNING_PREGRASP = "planning_pregrasp"
    EXECUTING_PREGRASP = "executing_pregrasp"
    PLANNING_APPROACH = "planning_approach"
    EXECUTING_APPROACH = "executing_approach"
    COMPLETED = "completed"
    FAILED = "failed"


# Gripper orientation configurations (from original example)
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


class ZordiMotionExpert:
    """Expert policy for two-phase motion planning toward target primitives."""

    def __init__(
        self,
        motion_gen: MotionGen,
        robot,
        tensor_args: TensorDeviceType,
        orientation_index: int = 0,
        pre_grasp_offset: float = 0.15,
        approach_step_size: float = 0.02,
        position_threshold: float = 0.01,
        target_threshold: float = 0.005,
    ):
        """Initialize the motion expert policy.

        Args:
            motion_gen: CuRobo motion generator instance
            robot: Isaac Sim robot articulation
            tensor_args: Tensor device configuration
            orientation_index: Index for gripper orientation (0-4)
            pre_grasp_offset: Distance offset for pre-grasp position (meters)
            approach_step_size: Step size for incremental approach (meters)
            position_threshold: Threshold for reaching pre-grasp position (meters)
            target_threshold: Threshold for reaching final target (meters)
        """
        self.motion_gen = motion_gen
        self.robot = robot
        self.tensor_args = tensor_args
        self.orientation_index = min(
            orientation_index, len(GRIPPER_ORIENTATION_LIST) - 1
        )
        self.pre_grasp_offset = pre_grasp_offset
        self.approach_step_size = approach_step_size
        self.position_threshold = position_threshold
        self.target_threshold = target_threshold

        # Motion planning state
        self.current_phase = MotionPhase.IDLE
        self.cmd_plan: Optional[JointState] = None
        self.cmd_idx = 0
        self.target_prim_path: Optional[str] = None
        self.target_position: Optional[np.ndarray] = None
        self.associated_stem_path: Optional[str] = None
        self.stored_pre_grasp_position: Optional[np.ndarray] = None

        # Robot configuration - get robot prim path correctly
        self.robot_prim_path = str(self.robot.prim_path).rsplit("/", 1)[0]
        self.robot_tool_prim_path = f"{self.robot_prim_path}/tool_pose"

        # Get gripper position from robot config (like working code)
        from config_utils import load_robot_config_with_zordi_paths

        robot_cfg = load_robot_config_with_zordi_paths("xarm7.yml")["robot_cfg"]

        all_joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.j_names = [
            name for name in all_joint_names if "gripper" not in name.lower()
        ]

        # Use same gripper position as working code
        right_home_deg = [5, -30, 25, 0, 0, -60, -5, 0.5]
        self.gripper_position = right_home_deg[-1]  # 0.5

    def set_target(
        self,
        target_prim_path: str,
        world_stage: Usd.Stage,
        associated_stem_path: str,
    ) -> bool:
        """Set a new target primitive for motion planning.

        Args:
            target_prim_path: USD path to the target primitive
            world_stage: USD stage containing the target
            associated_stem_path: USD path to the associated stem

        Returns:
            True if target was set successfully, False otherwise
        """
        self.target_prim_path = target_prim_path
        self.target_position, _ = get_world_pose(target_prim_path)
        print(f"[EXPERT] Target set to: {target_prim_path}")
        print(f"[EXPERT] Target position: {self.target_position}")

        # Set associated stem path
        self.associated_stem_path = associated_stem_path
        print(f"[EXPERT] Associated stem: {associated_stem_path}")

        # Reset phase to planning
        self.current_phase = MotionPhase.PLANNING_PREGRASP
        self.cmd_plan = None
        self.cmd_idx = 0

        return True

    def get_action(self) -> Optional[ArticulationAction]:
        """Get the next action for the robot based on current phase.

        Returns:
            ArticulationAction for robot control, or None if no action needed
        """
        if self.current_phase == MotionPhase.IDLE:
            return None

        elif self.current_phase == MotionPhase.PLANNING_PREGRASP:
            success = self._plan_pre_grasp()
            if success:
                self.current_phase = MotionPhase.EXECUTING_PREGRASP
                print("[EXPERT] Phase 1: Pre-grasp planning successful")
                action = self._execute_current_plan()
                return action
            else:
                self.current_phase = MotionPhase.FAILED
                print("[EXPERT] Phase 1: Pre-grasp planning failed")
                return None

        elif self.current_phase == MotionPhase.EXECUTING_PREGRASP:
            action = self._execute_current_plan()
            if action is None:  # Plan execution completed
                if self._check_pre_grasp_reached():
                    self.current_phase = MotionPhase.PLANNING_APPROACH
                    print("[EXPERT] Phase 1 completed, starting Phase 2")
                else:
                    # Re-plan if position not reached
                    self.current_phase = MotionPhase.PLANNING_PREGRASP
                    print("[EXPERT] Pre-grasp position not reached, re-planning")
            return action

        elif self.current_phase == MotionPhase.PLANNING_APPROACH:
            success = self._plan_approach_motion()
            if success:
                self.current_phase = MotionPhase.EXECUTING_APPROACH
            else:
                self.current_phase = MotionPhase.FAILED
                print("[EXPERT] Phase 2: Approach planning failed")
            return None

        elif self.current_phase == MotionPhase.EXECUTING_APPROACH:
            action = self._execute_current_plan()
            if action is None:  # Plan execution completed
                if self._check_target_reached():
                    self.current_phase = MotionPhase.COMPLETED
                else:
                    # Continue approaching with another step
                    self.current_phase = MotionPhase.PLANNING_APPROACH
            return action

        elif self.current_phase in [MotionPhase.COMPLETED, MotionPhase.FAILED]:
            return None

        return None

    def reset(self) -> None:
        """Reset the expert policy to idle state."""
        self.current_phase = MotionPhase.IDLE
        self.cmd_plan = None
        self.cmd_idx = 0
        self.target_prim_path = None
        self.target_position = None
        self.associated_stem_path = None
        self.stored_pre_grasp_position = None
        print("[EXPERT] Policy reset to idle state")

    def get_status(self) -> Tuple[MotionPhase, float]:
        """Get current status and progress.

        Returns:
            Tuple of (current_phase, distance_to_target)
        """
        distance = float("inf")
        if self.target_position is not None:
            current_pos, _ = get_world_pose(self.robot_tool_prim_path)
            distance = float(np.linalg.norm(self.target_position - current_pos))
        return self.current_phase, distance

    def _find_associated_stem(
        self, target_prim_path: str, world_stage: Usd.Stage
    ) -> Optional[str]:
        """Find the associated stem path for orientation calculation."""
        # Look for stem in the path hierarchy
        path_parts = target_prim_path.split("/")
        for i, part in enumerate(path_parts):
            if "stem" in part.lower():
                # Try to find the Stem_XX pattern in the path
                stem_path = "/".join(path_parts[: i + 2])  # Include the stem directory
                if world_stage.GetPrimAtPath(stem_path).IsValid():
                    return stem_path

        # Fallback: try parent directories
        parent_path = "/".join(target_prim_path.split("/")[:-1])
        if "stem" in parent_path.lower():
            return parent_path

        return None

    def _plan_pre_grasp(self) -> bool:
        """Plan motion to pre-grasp position."""
        if self.target_position is None:
            return False

        # Get gripper orientation and approach vector
        gripper_orientation, approach_vector = self._update_approach_vector()
        if gripper_orientation is None:
            # Use default orientation if stem-based calculation fails
            gripper_orientation = GRIPPER_ORIENTATION_LIST[self.orientation_index]
            approach_vector = GRIPPER_VECTOR_LIST[self.orientation_index]

        # Calculate pre-grasp position
        pre_grasp_position = (
            self.target_position - self.pre_grasp_offset * approach_vector
        )
        self.stored_pre_grasp_position = pre_grasp_position
        # Create pose goal
        pre_grasp_goal = Pose(
            position=self.tensor_args.to_device(pre_grasp_position),
            quaternion=self.tensor_args.to_device(gripper_orientation),
        )

        # Planning configuration with relaxed orientation constraints
        planning_metric = PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=self.tensor_args.to_device([1, 1, 1, 0.01, 0.01, 0.01]),
        )

        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=0,
            max_attempts=10,
            enable_finetune_trajopt=False,
            time_dilation_factor=1.0,
            pose_cost_metric=planning_metric,
        )

        # Execute planning
        current_js = self._get_current_joint_state()
        result = self.motion_gen.plan_single(
            current_js.unsqueeze(0),
            pre_grasp_goal,
            plan_config,
        )

        if result is not None and result.success is not None and result.success.item():
            # Get the trajectory and prepare for execution
            plan = self.motion_gen.get_full_js(result.get_interpolated_plan())

            self.cmd_plan = plan

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
            return False

    def _plan_approach_motion(self) -> bool:
        """Plan incremental approach motion toward target."""
        if self.target_position is None:
            return False

        # Get current tool position
        tool_position, _ = get_world_pose(self.robot_tool_prim_path)

        # Calculate approach vector toward target
        approach_vector = self.target_position - tool_position
        approach_vector_norm = np.linalg.norm(approach_vector)

        if approach_vector_norm < self.target_threshold:
            return False  # Already at target

        approach_vector = approach_vector / approach_vector_norm

        # Get gripper orientation
        gripper_orientation, _ = self._update_approach_vector()
        if gripper_orientation is None:
            gripper_orientation = GRIPPER_ORIENTATION_LIST[self.orientation_index]

        # Calculate step size (adaptive based on distance to target)
        distance_to_target = approach_vector_norm
        if self.approach_step_size > distance_to_target:
            step_size = distance_to_target * 0.5
        else:
            step_size = self.approach_step_size

        # Calculate next step position
        step_position = tool_position + approach_vector * step_size

        # Create step goal
        step_goal = Pose(
            position=self.tensor_args.to_device(step_position),
            quaternion=self.tensor_args.to_device(gripper_orientation),
        )

        # Planning configuration for precise approach
        step_planning_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=0,
            max_attempts=2,
            enable_finetune_trajopt=False,
            time_dilation_factor=1.0,
            pose_cost_metric=PoseCostMetric(
                reach_partial_pose=True,
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

        # Execute planning
        current_js = self._get_current_joint_state()
        result = self.motion_gen.plan_single(
            current_js.unsqueeze(0),
            step_goal,
            step_planning_config,
        )

        if result is not None and result.success is not None and result.success.item():
            # Get the trajectory and prepare for execution
            plan = self.motion_gen.get_full_js(result.get_interpolated_plan())

            # CRITICAL: Use exact same joint ordering logic as working code
            self.cmd_plan = plan

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
            print(
                f"[EXPERT] Approach planning failed: {result.status if result else 'None'}"
            )
            return False

    def _execute_current_plan(self) -> Optional[ArticulationAction]:
        """Execute the current motion plan."""
        if self.cmd_plan is None or self.cmd_idx >= len(self.cmd_plan.position):
            return None

        # Get current command
        cmd_state = self.cmd_plan[self.cmd_idx]

        # Prepare joint positions and velocities
        arm_positions = cmd_state.position.cpu().numpy()
        arm_velocities = cmd_state.velocity.cpu().numpy()

        # Add gripper positions
        sim_js_names = self.robot.dof_names
        gripper_indices = [
            self.robot.get_dof_index(x) for x in sim_js_names if "gripper" in x.lower()
        ]

        all_positions = list(arm_positions) + [self.gripper_position] * len(
            gripper_indices
        )
        all_velocities = list(arm_velocities) + [0.0] * len(gripper_indices)

        # Get joint indices - CRITICAL: Use exact same logic as working code
        arm_indices = [
            self.robot.get_dof_index(name) for name in self.cmd_plan.joint_names
        ]
        all_indices = arm_indices + gripper_indices

        # Create action
        action = ArticulationAction(
            np.array(all_positions),
            np.array(all_velocities),
            joint_indices=all_indices,
        )

        # Advance command index
        self.cmd_idx += 1

        # Check if plan is completed
        if self.cmd_idx >= len(self.cmd_plan.position):
            print(f"[EXPERT] Plan execution completed ({self.cmd_idx} steps)")
            self.cmd_plan = None
            self.cmd_idx = 0

        return action

    def _check_pre_grasp_reached(self) -> bool:
        """Check if pre-grasp position has been reached."""
        if self.stored_pre_grasp_position is None:
            return False

        current_pos, _ = get_world_pose(self.robot_tool_prim_path)
        distance = np.linalg.norm(self.stored_pre_grasp_position - current_pos)
        return distance < self.position_threshold

    def _check_target_reached(self) -> bool:
        """Check if final target has been reached."""
        if self.target_position is None:
            return False

        current_pos, _ = get_world_pose(self.robot_tool_prim_path)
        distance = np.linalg.norm(self.target_position - current_pos)
        return distance < self.target_threshold

    def _get_current_joint_state(self) -> JointState:
        """Get current joint state for motion planning."""
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

    def _update_approach_vector(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Update approach vector based on stem orientation."""
        if self.associated_stem_path is None:
            return None, None

        # Get the original approach vector
        original_approach = np.array(GRIPPER_VECTOR_LIST[self.orientation_index])
        original_xy_direction = original_approach[:2]
        original_xy_magnitude = np.linalg.norm(original_xy_direction)

        # Get stem transform - use robot's stage instead of self.robot.prim.GetStage()
        stage = self.robot.prim.GetStage()
        stem_prim = stage.GetPrimAtPath(self.associated_stem_path)
        if not stem_prim or not stem_prim.IsValid():
            return None, None

        stem_xformable = UsdGeom.Xformable(stem_prim)
        world_transform = stem_xformable.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rotation_matrix = np.array(world_transform)[:3, :3]

        # Calculate desired approach direction (negative stem z-direction)
        desired_approach_direction = -rotation_matrix[:, 2]

        # Calculate optimal z-component to maximize alignment
        a_dot_b_xy = np.dot(original_xy_direction, desired_approach_direction[:2])
        optimal_z = (a_dot_b_xy * desired_approach_direction[2]) / (
            original_xy_magnitude**2
        )

        # Construct new approach vector
        new_approach = np.array([*original_xy_direction, optimal_z])
        gripper_z_axis = new_approach / np.linalg.norm(new_approach)

        # Calculate gripper x-axis orthogonal to z-axis
        world_up = np.array([0.0, 0.0, 1.0])
        perpendicular_component = (
            world_up - np.dot(world_up, gripper_z_axis) * gripper_z_axis
        )
        gripper_x_axis = perpendicular_component / np.linalg.norm(
            perpendicular_component
        )

        # Calculate gripper y-axis
        gripper_y_axis = np.cross(gripper_z_axis, gripper_x_axis)

        # Construct rotation matrix and convert to quaternion
        rotation_matrix = np.column_stack((
            gripper_x_axis,
            gripper_y_axis,
            gripper_z_axis,
        ))
        rotation = R.from_matrix(rotation_matrix)
        new_quaternion = rotation.as_quat()[[3, 0, 1, 2]]  # Convert to [w, x, y, z]

        # Return quaternion and approach vector
        new_approach_vector = rotation_matrix[:, 2]
        return new_quaternion, new_approach_vector


def create_motion_expert(
    motion_gen: MotionGen, robot, tensor_args: TensorDeviceType, **kwargs
) -> ZordiMotionExpert:
    """Factory function to create a motion expert policy.

    Args:
        motion_gen: CuRobo motion generator instance
        robot: Isaac Sim robot articulation
        tensor_args: Tensor device configuration
        **kwargs: Additional configuration parameters

    Returns:
        Configured ZordiMotionExpert instance
    """
    return ZordiMotionExpert(motion_gen, robot, tensor_args, **kwargs)
