"""
Zordi Multi-Candidate Motion Planner

This planner generates multiple candidate trajectories using different gripper orientations,
evaluates them for collision and cost, and selects the optimal trajectory using an
MPC/MPPI-style approach.

Key Features:
- Batch trajectory generation for all 5 orientations
- Collision-aware trajectory evaluation
- Cost-based trajectory selection
- Rolling horizon re-planning capability
- Trajectory rollout and simulation
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

# Reuse imports from original file
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenPlanConfig,
    PoseCostMetric,
)

# Reuse orientation definitions
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


@dataclass
class TrajectoryCandidate:
    """Container for a candidate trajectory and its evaluation metrics."""

    orientation_index: int
    pre_grasp_pose: Pose
    target_pose: Pose
    pre_grasp_trajectory: Optional[JointState] = None
    approach_trajectory: Optional[JointState] = None
    collision_cost: float = float("inf")
    path_length_cost: float = float("inf")
    smoothness_cost: float = float("inf")
    manipulability_cost: float = float("inf")
    total_cost: float = float("inf")
    is_valid: bool = False
    planning_success: bool = False
    collision_free: bool = False
    final_target_distance: float = float("inf")


class TrajectoryEvaluator:
    """Evaluates trajectory candidates using multiple cost metrics."""

    def __init__(self, motion_gen: MotionGen, cost_weights: Dict[str, float] = None):
        self.motion_gen = motion_gen
        self.tensor_args = motion_gen.tensor_args

        # Default cost weights
        self.cost_weights = cost_weights or {
            "collision": 1000.0,  # Heavy penalty for collision
            "path_length": 1.0,  # Path length in meters
            "smoothness": 10.0,  # Jerk penalty
            "manipulability": 5.0,  # Manipulability penalty
            "proximity": 100.0,  # Proximity to obstacles
        }

    def evaluate_trajectory(
        self, candidate: TrajectoryCandidate
    ) -> TrajectoryCandidate:
        """Evaluate a single trajectory candidate across multiple metrics."""
        try:
            if not candidate.planning_success:
                return candidate

            # Combine pre-grasp and approach trajectories
            full_trajectory = self._combine_trajectories(
                candidate.pre_grasp_trajectory, candidate.approach_trajectory
            )

            if full_trajectory is None:
                return candidate

            # Collision evaluation
            candidate.collision_cost, candidate.collision_free = (
                self._evaluate_collision(full_trajectory)
            )

            # Path length evaluation
            candidate.path_length_cost = self._evaluate_path_length(full_trajectory)

            # Smoothness evaluation
            candidate.smoothness_cost = self._evaluate_smoothness(full_trajectory)

            # Manipulability evaluation
            candidate.manipulability_cost = self._evaluate_manipulability(
                full_trajectory
            )

            # ✅ NEW: Calculate final target distance
            candidate.final_target_distance = self._evaluate_final_target_distance(
                full_trajectory, candidate.target_pose
            )

            # Compute total cost with weights
            candidate.total_cost = (
                self.cost_weights["collision"] * candidate.collision_cost
                + self.cost_weights["path_length"] * candidate.path_length_cost
                + self.cost_weights["smoothness"] * candidate.smoothness_cost
                + self.cost_weights["manipulability"] * candidate.manipulability_cost
            )

            # A trajectory is valid if:
            # 1. It's collision-free (collision_cost == 0)
            # 2. All metrics have finite values (no inf or NaN)
            # 3. Total cost is finite and not excessive
            candidate.is_valid = (
                candidate.collision_free
                and np.isfinite(candidate.path_length_cost)
                and np.isfinite(candidate.smoothness_cost)
                and np.isfinite(candidate.total_cost)
                and candidate.total_cost < 1e6  # Set a reasonable upper limit
            )

            # Print trajectory stats for debugging
            print(f"Trajectory stats for orientation {candidate.orientation_index}:")
            print(f"  Collision cost: {candidate.collision_cost:.3f}")
            print(f"  Path length cost: {candidate.path_length_cost:.3f}")
            print(f"  Smoothness cost: {candidate.smoothness_cost:.3f}")
            print(f"  Manipulability cost: {candidate.manipulability_cost:.3f}")
            print(f"  Final target distance: {candidate.final_target_distance:.3f}m")
            print(f"  Total cost: {candidate.total_cost:.3f}")
            print(f"  Collision-free: {candidate.collision_free}")
            print(f"  Valid: {candidate.is_valid}")

        except Exception as e:
            print(f"Error evaluating trajectory {candidate.orientation_index}: {e}")
            candidate.is_valid = False

        return candidate

    def _combine_trajectories(
        self, pre_grasp_traj: JointState, approach_traj: JointState
    ) -> Optional[JointState]:
        """Combine pre-grasp and approach trajectories into a single trajectory."""
        if pre_grasp_traj is None:
            return approach_traj
        if approach_traj is None:
            return pre_grasp_traj

        try:
            # Concatenate trajectories
            combined_position = torch.cat(
                [pre_grasp_traj.position, approach_traj.position], dim=0
            )
            combined_velocity = torch.cat(
                [pre_grasp_traj.velocity, approach_traj.velocity], dim=0
            )
            combined_acceleration = torch.cat(
                [pre_grasp_traj.acceleration, approach_traj.acceleration], dim=0
            )
            combined_jerk = torch.cat([pre_grasp_traj.jerk, approach_traj.jerk], dim=0)

            return JointState(
                position=combined_position,
                velocity=combined_velocity,
                acceleration=combined_acceleration,
                jerk=combined_jerk,
                joint_names=pre_grasp_traj.joint_names,
            )
        except Exception:
            return pre_grasp_traj  # Fallback to pre-grasp only

    def _evaluate_collision(self, trajectory: JointState) -> Tuple[float, bool]:
        """Evaluate collision cost and collision-free status."""
        try:
            collision_cost = 0.0
            min_distance = float("inf")

            # ✅ IMPROVED: Increase sampling density for better collision detection
            num_samples = min(len(trajectory.position), 50)  # Increased from 20 to 50
            indices = torch.linspace(
                0, len(trajectory.position) - 1, num_samples, dtype=torch.long
            )

            # ✅ NEW: Add debug info
            print(
                f"  Collision check: sampling {num_samples} points from {len(trajectory.position)} waypoints"
            )

            collision_points_found = 0
            for idx in indices:
                joint_state = JointState(
                    position=trajectory.position[idx : idx + 1],
                    velocity=trajectory.velocity[idx : idx + 1]
                    if trajectory.velocity is not None
                    else None,
                    acceleration=trajectory.acceleration[idx : idx + 1]
                    if trajectory.acceleration is not None
                    else None,
                    jerk=trajectory.jerk[idx : idx + 1]
                    if trajectory.jerk is not None
                    else None,
                    joint_names=trajectory.joint_names,
                )

                # Reorder joints to match kinematics model joint names
                joint_state = joint_state.get_ordered_joint_state(
                    self.motion_gen.kinematics.joint_names
                )

                # Get robot spheres for collision checking
                robot_spheres = self.motion_gen.kinematics.get_robot_as_spheres(
                    joint_state.position
                )

                if robot_spheres and len(robot_spheres) > 0:
                    sphere_tensor = self._convert_spheres_to_tensor(robot_spheres[0])

                    if sphere_tensor is not None:
                        # Check collision
                        collision_result = self._check_sphere_collision(sphere_tensor)

                        if collision_result is not None:
                            if hasattr(collision_result, "distance"):
                                min_dist = collision_result.distance.min().item()
                                min_distance = min(min_distance, min_dist)

                                # ✅ IMPROVED: More aggressive collision detection
                                if min_dist < 0.0:  # In collision
                                    collision_cost += (
                                        abs(min_dist) * 1000
                                    )  # Increased penalty
                                    collision_points_found += 1
                                elif (
                                    min_dist < 0.08
                                ):  # ✅ IMPROVED: Increased safety margin from 0.05 to 0.08
                                    collision_cost += (
                                        0.08 - min_dist
                                    ) * 100  # Increased penalty
                            else:
                                # ✅ NEW: Handle case where distance attribute is missing
                                print(
                                    f"    Warning: Collision result missing distance attribute at point {idx}"
                                )

            # ✅ NEW: Add detailed debug info
            collision_free = collision_cost == 0.0
            print(
                f"    Collision summary: cost={collision_cost:.3f}, min_dist={min_distance:.4f}m, collision_points={collision_points_found}"
            )

            return collision_cost, collision_free

        except Exception as e:
            print(f"Collision evaluation error: {e}")
            import traceback

            traceback.print_exc()  # ✅ NEW: Better error debugging
            return float("inf"), False

    def _convert_spheres_to_tensor(self, sphere_list) -> Optional[torch.Tensor]:
        """Convert robot spheres to tensor format for collision checking."""
        try:
            if isinstance(sphere_list, torch.Tensor):
                return (
                    sphere_list.unsqueeze(0).unsqueeze(0)
                    if len(sphere_list.shape) == 2
                    else sphere_list
                )

            sphere_data = []
            for sphere_obj in sphere_list:
                if hasattr(sphere_obj, "position") and hasattr(sphere_obj, "radius"):
                    pos = sphere_obj.position
                    rad = sphere_obj.radius

                    if not isinstance(pos, torch.Tensor):
                        pos = torch.tensor(
                            pos,
                            device=self.tensor_args.device,
                            dtype=self.tensor_args.dtype,
                        )
                    if not isinstance(rad, torch.Tensor):
                        rad = torch.tensor(
                            rad,
                            device=self.tensor_args.device,
                            dtype=self.tensor_args.dtype,
                        )

                    sphere_data.append(torch.cat([pos.flatten(), rad.flatten()]))

            if sphere_data:
                return torch.stack(sphere_data, dim=0).unsqueeze(0).unsqueeze(0)
            return None

        except Exception:
            return None

    def _check_sphere_collision(self, sphere_tensor: torch.Tensor):
        """Check collision for sphere tensor."""
        try:
            if (
                not hasattr(self.motion_gen, "world_coll_checker")
                or self.motion_gen.world_coll_checker is None
            ):
                print("    Warning: No world collision checker available")
                return None

            # ✅ NEW: Validate collision checker has obstacles
            if hasattr(self.motion_gen.world_coll_checker, "get_obstacle_names"):
                obstacle_names = self.motion_gen.world_coll_checker.get_obstacle_names()
                if not obstacle_names:
                    print("    Warning: No obstacles loaded in collision checker")
                    return None
                else:
                    print(f"    Collision checker has {len(obstacle_names)} obstacles")

            from curobo.geom.sdf.world import CollisionQueryBuffer

            query_buffer = CollisionQueryBuffer.initialize_from_shape(
                sphere_tensor.shape,
                self.tensor_args,
                self.motion_gen.world_coll_checker.collision_types,
            )

            weight = self.tensor_args.to_device([1.0])
            # ✅ IMPROVED: Increase activation distance for better safety margin
            activation_distance = self.tensor_args.to_device(
                [0.05]
            )  # Increased from 0.02 to 0.05

            result = self.motion_gen.world_coll_checker.get_sphere_collision(
                sphere_tensor, query_buffer, weight, activation_distance
            )

            # ✅ NEW: Add validation of collision result
            if result is not None and hasattr(result, "distance"):
                min_dist = result.distance.min().item()
                max_dist = result.distance.max().item()
                if (
                    min_dist < -0.5 or max_dist > 10.0
                ):  # Sanity check for unrealistic distances
                    print(
                        f"    Warning: Suspicious collision distances: min={min_dist:.3f}, max={max_dist:.3f}"
                    )

            return result
        except Exception as e:
            print(f"    Collision check error: {e}")
            return None

    def _evaluate_path_length(self, trajectory: JointState) -> float:
        """Evaluate path length in Cartesian space."""
        try:
            total_length = 0.0
            prev_pos = None

            # Sample fewer points for performance
            num_samples = min(len(trajectory.position), 10)
            indices = torch.linspace(
                0, len(trajectory.position) - 1, num_samples, dtype=torch.long
            )

            for idx in indices:
                joint_state = JointState(
                    position=trajectory.position[idx : idx + 1],
                    joint_names=trajectory.joint_names,
                )

                # Reorder joints to match kinematics model joint names
                joint_state = joint_state.get_ordered_joint_state(
                    self.motion_gen.kinematics.joint_names
                )

                ee_state = self.motion_gen.kinematics.compute_kinematics(joint_state)
                current_pos = ee_state.ee_pose.position.cpu().numpy().flatten()

                if prev_pos is not None:
                    total_length += np.linalg.norm(current_pos - prev_pos)
                prev_pos = current_pos

            return total_length

        except Exception as e:
            print(f"Path length evaluation error: {e}")
            return float("inf")

    def _evaluate_smoothness(self, trajectory: JointState) -> float:
        """Evaluate trajectory smoothness using jerk."""
        try:
            if trajectory.jerk is None:
                return 0.0

            # RMS jerk as smoothness metric
            jerk_squared = trajectory.jerk**2
            mean_jerk_squared = torch.mean(jerk_squared)
            return torch.sqrt(mean_jerk_squared).item()

        except Exception as e:
            print(f"Smoothness evaluation error: {e}")
            return float("inf")

    def _evaluate_manipulability(self, trajectory: JointState) -> float:
        """Evaluate manipulability along trajectory."""
        try:
            # Check if motion_gen.kinematics has the get_jacobian method
            if not hasattr(self.motion_gen.kinematics, "get_jacobian"):
                print(
                    "Skipping manipulability evaluation - get_jacobian method not available"
                )
                return 0.0  # Return 0 cost if manipulability can't be evaluated

            manipulability_sum = 0.0
            count = 0

            # Sample fewer points for performance
            num_samples = min(len(trajectory.position), 5)
            indices = torch.linspace(
                0, len(trajectory.position) - 1, num_samples, dtype=torch.long
            )

            for idx in indices:
                joint_state = JointState(
                    position=trajectory.position[idx : idx + 1],
                    joint_names=trajectory.joint_names,
                )

                # Reorder joints to match kinematics model joint names
                joint_state = joint_state.get_ordered_joint_state(
                    self.motion_gen.kinematics.joint_names
                )

                # Get Jacobian
                jacobian = self.motion_gen.kinematics.get_jacobian(joint_state)
                if jacobian is not None:
                    jac_np = jacobian.cpu().numpy()[0][:3]  # Position part only

                    # Calculate manipulability index
                    manipulability = np.sqrt(np.linalg.det(jac_np @ jac_np.T))
                    manipulability_sum += manipulability
                    count += 1

            if count > 0:
                avg_manipulability = manipulability_sum / count
                # Return inverse manipulability as cost (lower manipulability = higher cost)
                return 1.0 / (avg_manipulability + 1e-6)
            else:
                return float("inf")

        except Exception as e:
            print(f"Manipulability evaluation error: {e}")
            return 0.0  # Return 0 cost if manipulability evaluation fails

    def _evaluate_final_target_distance(
        self, trajectory: JointState, target_pose: Pose
    ) -> float:
        """Calculate distance from final trajectory point to target."""
        try:
            if trajectory is None or len(trajectory.position) == 0:
                return float("inf")

            # Get final joint state
            final_joint_state = JointState(
                position=trajectory.position[-1:],
                joint_names=trajectory.joint_names,
            )

            # Reorder joints to match kinematics model joint names
            final_joint_state = final_joint_state.get_ordered_joint_state(
                self.motion_gen.kinematics.joint_names
            )

            # Compute forward kinematics for final position
            ee_state = self.motion_gen.kinematics.compute_kinematics(final_joint_state)
            final_ee_pos = ee_state.ee_pose.position.cpu().numpy().flatten()

            # Get target position
            target_pos = target_pose.position.cpu().numpy().flatten()

            # Calculate Euclidean distance
            distance = np.linalg.norm(final_ee_pos - target_pos)
            return distance

        except Exception as e:
            print(f"Error calculating final target distance: {e}")
            return float("inf")


class ZordiMultiCandidatePlanner:
    """Multi-candidate motion planner with MPC/MPPI-style trajectory selection."""

    def __init__(
        self,
        motion_gen: MotionGen,
        target_position: torch.Tensor,
        planning_config: Dict = None,
    ):
        """Initialize the multi-candidate planner.

        Args:
            motion_gen: The motion generator
            target_position: Target position tensor [batch, 3]
            planning_config: Optional configuration parameters for planning
        """
        self.motion_gen = motion_gen
        self.tensor_args = motion_gen.tensor_args

        # Set default planning configuration
        default_planning_config = {
            "pre_grasp_offset": 0.1,
            "approach_step_size": 0.02,
            "max_planning_attempts": 3,
            "enable_re_planning": True,
            "planning_timeout": 15.0,
            "max_candidates": 5,
            "stem_alignment_enabled": True,  # New flag to enable/disable stem alignment
            "associated_stem_path": "/World/PlantScene/plant_004/stem_Unit001_03/Strawberry003/stem/Stem_20/Stem_20",  # Path to the stem
        }

        # Update with user-provided configuration if available
        self.planning_config = default_planning_config
        if planning_config is not None:
            self.planning_config.update(planning_config)

        # Set orientation index (if specified, only plan for this orientation)
        self.orientation_index = None

        # Initialize candidate trajectories
        self.candidates: List[TrajectoryCandidate] = []
        self.best_candidate: Optional[TrajectoryCandidate] = None

        # Store target position
        self.target_position = target_position

        # Initialize evaluator
        self.evaluator = TrajectoryEvaluator(motion_gen)

        # Make backup copies of original orientations and approach vectors
        self.original_orientations = [list(q) for q in GRIPPER_ORIENTATION_LIST]
        self.original_approach_vectors = [np.array(v) for v in GRIPPER_VECTOR_LIST]

    def _update_orientation_for_stem_alignment(self, orientation_idx: int) -> None:
        """Update gripper orientation to align with stem's negative z-direction while preserving xy-direction.

        Args:
            orientation_idx: Index of the orientation to update
        """
        # Skip if stem alignment is disabled
        if not self.planning_config.get("stem_alignment_enabled", True):
            return

        # Get stem path from config
        stem_path = self.planning_config.get("associated_stem_path")
        if not stem_path:
            print("No associated stem path specified, using default orientation")
            return

        try:
            # Access the stage
            from omni.isaac.core.utils.stage import get_current_stage
            from pxr import UsdGeom, Usd

            stage = get_current_stage()
            stem_prim = stage.GetPrimAtPath(stem_path)

            if not stem_prim or not stem_prim.IsValid():
                print(f"Invalid stem prim: {stem_path}")
                return

            # Get the original approach vector (preserve its xy-direction)
            original_approach = np.array(
                self.original_approach_vectors[orientation_idx]
            )
            original_xy_direction = original_approach[:2]  # Keep (x, y) components
            original_xy_magnitude = np.linalg.norm(original_xy_direction)

            if original_xy_magnitude < 1e-6:
                print(
                    "Original approach vector has no xy-component, cannot preserve direction"
                )
                return

            # Get the stem's transform
            stem_xformable = UsdGeom.Xformable(stem_prim)
            world_transform = stem_xformable.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )

            # Convert transform matrix to numpy
            transform_matrix = np.array(
                [
                    [
                        world_transform[0][0],
                        world_transform[0][1],
                        world_transform[0][2],
                        world_transform[0][3],
                    ],
                    [
                        world_transform[1][0],
                        world_transform[1][1],
                        world_transform[1][2],
                        world_transform[1][3],
                    ],
                    [
                        world_transform[2][0],
                        world_transform[2][1],
                        world_transform[2][2],
                        world_transform[2][3],
                    ],
                    [
                        world_transform[3][0],
                        world_transform[3][1],
                        world_transform[3][2],
                        world_transform[3][3],
                    ],
                ]
            )

            # Extract stem's z-direction (negative for approach)
            rotation_matrix = transform_matrix[:3, :3]
            stem_z_direction = rotation_matrix[:, 2]
            desired_stem_direction = -stem_z_direction

            # Calculate optimal z-component for alignment
            a_dot_b_xy = np.dot(original_xy_direction, desired_stem_direction[:2])
            b_z = desired_stem_direction[2]
            a_magnitude_sq = original_xy_magnitude**2

            # Check for perpendicular case
            if abs(a_dot_b_xy) < 1e-6:
                # Special case: xy-components are perpendicular
                optimal_z = b_z
            else:
                # Normal case: calculus-derived formula for maximum alignment
                optimal_z = (a_dot_b_xy * b_z) / a_magnitude_sq

            # Construct the new approach vector
            new_approach = np.array(
                [original_xy_direction[0], original_xy_direction[1], optimal_z]
            )
            new_approach = new_approach / np.linalg.norm(new_approach)  # Normalize

            # Calculate alignment metrics
            alignment_with_stem = np.dot(new_approach, desired_stem_direction)
            z_tilt_angle = (
                np.arctan2(abs(optimal_z), original_xy_magnitude) * 180.0 / np.pi
            )

            # Build rotation matrix for the gripper
            gripper_z_axis = new_approach
            world_up = np.array([0.0, 0.0, 1.0])

            # Project world_up onto the plane perpendicular to gripper_z_axis
            parallel_component = np.dot(world_up, gripper_z_axis) * gripper_z_axis
            perpendicular_component = world_up - parallel_component

            perp_magnitude = np.linalg.norm(perpendicular_component)
            if perp_magnitude < 1e-6:
                # If world +z is nearly parallel to approach, use world +x as fallback
                world_ref = np.array([1.0, 0.0, 0.0])
                parallel_component = np.dot(world_ref, gripper_z_axis) * gripper_z_axis
                perpendicular_component = world_ref - parallel_component
                perp_magnitude = np.linalg.norm(perpendicular_component)

            gripper_x_axis = perpendicular_component / perp_magnitude

            # Compute gripper +y to maintain right-handed coordinate system
            gripper_y_axis = np.cross(gripper_z_axis, gripper_x_axis)
            gripper_y_axis = gripper_y_axis / np.linalg.norm(gripper_y_axis)

            # Double-check and ensure consistency
            gripper_x_check = np.cross(gripper_y_axis, gripper_z_axis)
            gripper_x_check = gripper_x_check / np.linalg.norm(gripper_x_check)

            if np.dot(gripper_x_axis, gripper_x_check) < 0:
                gripper_x_axis = -gripper_x_check
            else:
                gripper_x_axis = gripper_x_check

            # Calculate final upright alignment
            upright_alignment = np.dot(gripper_x_axis, world_up)
            gripper_tilt_angle = (
                np.arccos(np.clip(upright_alignment, -1.0, 1.0)) * 180.0 / np.pi
            )

            # Create rotation matrix and convert to quaternion
            new_rotation_matrix = np.column_stack(
                [gripper_x_axis, gripper_y_axis, gripper_z_axis]
            )
            new_quaternion = self._rotation_matrix_to_quaternion(new_rotation_matrix)

            # Update the orientation and approach vector
            GRIPPER_ORIENTATION_LIST[orientation_idx] = new_quaternion.tolist()
            GRIPPER_VECTOR_LIST[orientation_idx] = new_approach

            print(f"\n🎯 STEM ALIGNMENT (Orientation {orientation_idx}):")
            print(f"   Z-component tilted {z_tilt_angle:.1f}° for stem alignment")
            print(f"   Stem alignment: {alignment_with_stem * 100:.1f}%")
            print(f"   Gripper tilted {gripper_tilt_angle:.1f}° from upright\n")

        except Exception as e:
            print(f"Error updating orientation for stem alignment: {e}")
            import traceback

            print(traceback.format_exc())

    def _rotation_matrix_to_quaternion(self, R):
        """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        # Normalize and return as numpy array
        quat = np.array([w, x, y, z])
        return quat / np.linalg.norm(quat)

    def _generate_candidate(
        self, orientation_idx: int, current_joint_state: JointState
    ) -> Optional[TrajectoryCandidate]:
        """Generate a single trajectory candidate for the given orientation."""
        try:
            # Apply stem alignment to update orientation and approach vector
            self._update_orientation_for_stem_alignment(orientation_idx)

            # Get orientation and approach vector (now potentially updated with stem alignment)
            orientation = np.array(GRIPPER_ORIENTATION_LIST[orientation_idx])
            approach_vector = GRIPPER_VECTOR_LIST[orientation_idx]

            # Convert target position to numpy array - handle all input types
            if isinstance(self.target_position, torch.Tensor):
                target_position_np = self.target_position.cpu().numpy().flatten()
            elif isinstance(self.target_position, (list, tuple)):
                target_position_np = np.array(self.target_position)
            elif isinstance(self.target_position, np.ndarray):
                target_position_np = self.target_position
            else:
                raise ValueError(
                    f"Unsupported target_position type: {type(self.target_position)}"
                )

            # Ensure it's a 1D array with 3 elements
            target_position_np = target_position_np.flatten()[:3]

            # Calculate pre-grasp pose
            pre_grasp_offset = self.planning_config["pre_grasp_offset"]
            pre_grasp_position = target_position_np - pre_grasp_offset * approach_vector

            # Convert to tensors for poses
            pre_grasp_pose = Pose(
                position=torch.tensor(
                    pre_grasp_position,
                    dtype=torch.float32,
                    device=self.tensor_args.device,
                ).unsqueeze(0),
                quaternion=torch.tensor(
                    orientation, dtype=torch.float32, device=self.tensor_args.device
                ).unsqueeze(0),
            )

            target_pose = self._create_target_pose(orientation_idx)

            candidate = TrajectoryCandidate(
                orientation_index=orientation_idx,
                pre_grasp_pose=pre_grasp_pose,
                target_pose=target_pose,
            )

            # Plan pre-grasp trajectory
            pre_grasp_success = self._plan_pre_grasp_trajectory(
                candidate, current_joint_state
            )

            if pre_grasp_success:
                # Plan approach trajectory
                approach_success = self._plan_approach_trajectory(candidate)
                candidate.planning_success = approach_success
            else:
                candidate.planning_success = False

            return candidate

        except Exception as e:
            print(f"Error generating candidate {orientation_idx}: {e}")
            return None

    def _create_target_pose(self, orientation_idx: int) -> Pose:
        """Create a target pose with the specified orientation index."""
        # Create quaternion from orientation index
        quaternion = GRIPPER_ORIENTATION_LIST[orientation_idx]

        # Convert target position to numpy array - handle all input types
        if isinstance(self.target_position, torch.Tensor):
            target_position_np = self.target_position.cpu().numpy().flatten()
        elif isinstance(self.target_position, (list, tuple)):
            target_position_np = np.array(self.target_position)
        elif isinstance(self.target_position, np.ndarray):
            target_position_np = self.target_position
        else:
            raise ValueError(
                f"Unsupported target_position type: {type(self.target_position)}"
            )

        # Ensure it's a 1D array with 3 elements
        target_position_np = target_position_np.flatten()[:3]

        # Create target pose
        return Pose(
            position=torch.tensor(
                target_position_np, dtype=torch.float32, device=self.tensor_args.device
            ).unsqueeze(0),
            quaternion=torch.tensor(
                quaternion, dtype=torch.float32, device=self.tensor_args.device
            ).unsqueeze(0),
        )

    def _plan_pre_grasp_trajectory(
        self, candidate: TrajectoryCandidate, current_joint_state: JointState
    ) -> bool:
        """Plan trajectory to pre-grasp pose."""
        try:
            # Filter out gripper joints - only use arm joints for planning
            # This is the key fix: ensure we're using only the arm joints for planning
            arm_joint_names = [
                name
                for name in current_joint_state.joint_names
                if "gripper" not in name.lower()
            ]

            # Get positions for arm joints only
            arm_joint_indices = [
                i
                for i, name in enumerate(current_joint_state.joint_names)
                if name in arm_joint_names
            ]

            # Properly handle tensor dimensions based on shape
            if len(current_joint_state.position.shape) > 1:
                # Batch dimension exists - select indices along last dimension
                arm_positions = current_joint_state.position[:, arm_joint_indices]
                arm_velocities = (
                    None
                    if current_joint_state.velocity is None
                    else current_joint_state.velocity[:, arm_joint_indices]
                )
                arm_accelerations = (
                    None
                    if current_joint_state.acceleration is None
                    else current_joint_state.acceleration[:, arm_joint_indices]
                )
                arm_jerks = (
                    None
                    if current_joint_state.jerk is None
                    else current_joint_state.jerk[:, arm_joint_indices]
                )
            else:
                # No batch dimension - select indices directly
                arm_positions = current_joint_state.position[arm_joint_indices]
                arm_velocities = (
                    None
                    if current_joint_state.velocity is None
                    else current_joint_state.velocity[arm_joint_indices]
                )
                arm_accelerations = (
                    None
                    if current_joint_state.acceleration is None
                    else current_joint_state.acceleration[arm_joint_indices]
                )
                arm_jerks = (
                    None
                    if current_joint_state.jerk is None
                    else current_joint_state.jerk[arm_joint_indices]
                )

            # Create arm-only joint state
            arm_joint_state = JointState(
                position=arm_positions,
                velocity=arm_velocities,
                acceleration=arm_accelerations,
                jerk=arm_jerks,
                joint_names=arm_joint_names,
            )

            # Ensure joint state is batched - add batch dimension if needed
            if len(arm_joint_state.position.shape) == 1:
                arm_joint_state = arm_joint_state.unsqueeze(0)

            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_graph_attempt=0,
                max_attempts=self.planning_config["max_planning_attempts"],
                enable_finetune_trajopt=False,
                time_dilation_factor=1.0,
                pose_cost_metric=PoseCostMetric(
                    reach_partial_pose=False,
                    reach_vec_weight=self.tensor_args.to_device(
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    ),
                ),
            )

            result = self.motion_gen.plan_single(
                arm_joint_state, candidate.pre_grasp_pose, plan_config
            )

            if (
                result is not None
                and result.success is not None
                and result.success.item()
            ):
                candidate.pre_grasp_trajectory = result.get_interpolated_plan()
                candidate.pre_grasp_trajectory = self.motion_gen.get_full_js(
                    candidate.pre_grasp_trajectory
                )
                return True
            else:
                return False

        except Exception as e:
            print(
                f"Pre-grasp planning error for orientation {candidate.orientation_index}: {e}"
            )
            return False

    def _plan_approach_trajectory(self, candidate: TrajectoryCandidate) -> bool:
        """Plan approach trajectory from pre-grasp to target."""
        try:
            if candidate.pre_grasp_trajectory is None:
                return False

            # Get final state from pre-grasp trajectory
            final_pre_grasp_state = candidate.pre_grasp_trajectory[-1]

            # Filter out gripper joints - only use arm joints for planning
            arm_joint_names = [
                name
                for name in final_pre_grasp_state.joint_names
                if "gripper" not in name.lower()
            ]

            # Get positions for arm joints only
            arm_joint_indices = [
                i
                for i, name in enumerate(final_pre_grasp_state.joint_names)
                if name in arm_joint_names
            ]

            # Extract arm joint positions with proper tensor dimension handling
            if len(final_pre_grasp_state.position.shape) > 1:
                # Batch dimension exists
                arm_positions = final_pre_grasp_state.position[:, arm_joint_indices]
                arm_velocities = (
                    None
                    if final_pre_grasp_state.velocity is None
                    else final_pre_grasp_state.velocity[:, arm_joint_indices]
                )
                arm_accelerations = (
                    None
                    if final_pre_grasp_state.acceleration is None
                    else final_pre_grasp_state.acceleration[:, arm_joint_indices]
                )
                arm_jerks = (
                    None
                    if final_pre_grasp_state.jerk is None
                    else final_pre_grasp_state.jerk[:, arm_joint_indices]
                )
            else:
                # No batch dimension
                arm_positions = final_pre_grasp_state.position[arm_joint_indices]
                arm_velocities = (
                    None
                    if final_pre_grasp_state.velocity is None
                    else final_pre_grasp_state.velocity[arm_joint_indices]
                )
                arm_accelerations = (
                    None
                    if final_pre_grasp_state.acceleration is None
                    else final_pre_grasp_state.acceleration[arm_joint_indices]
                )
                arm_jerks = (
                    None
                    if final_pre_grasp_state.jerk is None
                    else final_pre_grasp_state.jerk[arm_joint_indices]
                )

            # Create arm-only joint state
            arm_joint_state = JointState(
                position=arm_positions,
                velocity=arm_velocities,
                acceleration=arm_accelerations,
                jerk=arm_jerks,
                joint_names=arm_joint_names,
            )

            # Ensure joint state is batched - add batch dimension if needed
            if len(arm_joint_state.position.shape) == 1:
                arm_joint_state = arm_joint_state.unsqueeze(0)

            # Plan approach trajectory
            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_graph_attempt=0,
                max_attempts=self.planning_config["max_planning_attempts"],
                enable_finetune_trajopt=False,
                time_dilation_factor=1.0,
                pose_cost_metric=PoseCostMetric(
                    reach_partial_pose=False,
                    reach_vec_weight=self.tensor_args.to_device(
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    ),
                ),
            )

            result = self.motion_gen.plan_single(
                arm_joint_state, candidate.target_pose, plan_config
            )

            if (
                result is not None
                and result.success is not None
                and result.success.item()
            ):
                candidate.approach_trajectory = result.get_interpolated_plan()
                candidate.approach_trajectory = self.motion_gen.get_full_js(
                    candidate.approach_trajectory
                )
                return True
            else:
                return False

        except Exception as e:
            print(
                f"Approach planning error for orientation {candidate.orientation_index}: {e}"
            )
            return False

    def _plan_next_approach_step(
        self,
        current_joint_state: JointState,
        current_position: np.ndarray,
        target_position: np.ndarray,
        approach_vector: np.ndarray,
        current_orientation: np.ndarray,
    ) -> Optional[JointState]:
        """Plan next step with adaptive step size based on remaining distance to target."""
        try:
            # Calculate remaining distance to target
            vector_to_target = target_position - current_position
            remaining_distance = np.linalg.norm(vector_to_target)

            # Adaptive step size adjustment - smaller steps as we get closer
            base_step_size = self.planning_config.get(
                "approach_step_size", 0.02
            )  # Default 2cm

            # Use adaptive step size based on remaining distance
            if remaining_distance < 0.01:  # Very close (< 1cm)
                next_step_size = min(
                    0.002, remaining_distance / 2
                )  # 2mm or half remaining distance
                print(
                    f"  Using precision step size: {next_step_size * 1000:.2f}mm (remaining: {remaining_distance * 1000:.2f}mm)"
                )
            elif remaining_distance < 0.03:  # Close (< 3cm)
                next_step_size = min(
                    0.005, remaining_distance / 3
                )  # 5mm or third of remaining distance
                print(
                    f"  Using fine step size: {next_step_size * 1000:.2f}mm (remaining: {remaining_distance * 1000:.2f}mm)"
                )
            elif remaining_distance < 0.08:  # Medium distance (< 8cm)
                next_step_size = min(
                    0.01, remaining_distance / 4
                )  # 1cm or quarter of remaining distance
                print(
                    f"  Using medium step size: {next_step_size * 1000:.2f}mm (remaining: {remaining_distance * 1000:.2f}mm)"
                )
            else:  # Far away
                next_step_size = min(
                    base_step_size, remaining_distance / 5
                )  # Base step size or fifth of remaining distance
                print(
                    f"  Using standard step size: {next_step_size * 1000:.2f}mm (remaining: {remaining_distance * 1000:.2f}mm)"
                )

            # Don't overshoot the target
            if next_step_size > remaining_distance:
                next_step_size = remaining_distance * 0.9  # 90% of remaining distance
                print(f"  Adjusted to avoid overshoot: {next_step_size * 1000:.2f}mm")

            # Calculate next step position
            next_step_position = current_position + approach_vector * next_step_size

            # Create pose for next step
            next_step_goal = Pose(
                position=self.tensor_args.to_device(next_step_position),
                quaternion=self.tensor_args.to_device(current_orientation),
            )

            # Plan with this step size
            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_graph_attempt=0,
                max_attempts=self.planning_config["max_planning_attempts"],
                enable_finetune_trajopt=False,
                time_dilation_factor=1.0,
                pose_cost_metric=PoseCostMetric(
                    reach_partial_pose=False,
                    reach_vec_weight=self.tensor_args.to_device(
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    ),
                ),
            )

            result = self.motion_gen.plan_single(
                current_joint_state.unsqueeze(0), next_step_goal, plan_config
            )

            if (
                result is not None
                and result.success is not None
                and result.success.item()
            ):
                step_plan = result.get_interpolated_plan()
                step_plan = self.motion_gen.get_full_js(step_plan)
                return step_plan
            else:
                # If planning fails, try with an even smaller step size
                if (
                    next_step_size > 0.003
                ):  # Only try fallback for steps larger than 3mm
                    print("  Trying fallback with smaller step...")
                    fallback_step_size = next_step_size * 0.5
                    fallback_position = (
                        current_position + approach_vector * fallback_step_size
                    )

                    fallback_goal = Pose(
                        position=self.tensor_args.to_device(fallback_position),
                        quaternion=self.tensor_args.to_device(current_orientation),
                    )

                    fallback_result = self.motion_gen.plan_single(
                        current_joint_state.unsqueeze(0), fallback_goal, plan_config
                    )

                    if (
                        fallback_result is not None
                        and fallback_result.success is not None
                        and fallback_result.success.item()
                    ):
                        print(
                            f"  ✅ Fallback planning successful with {fallback_step_size * 1000:.2f}mm step"
                        )
                        fallback_plan = fallback_result.get_interpolated_plan()
                        fallback_plan = self.motion_gen.get_full_js(fallback_plan)
                        return fallback_plan

                print("  ❌ Step planning failed with multiple step sizes")
                return None

        except Exception as e:
            print(f"Error planning next approach step: {e}")
            return None

    def get_execution_trajectory(self) -> Optional[JointState]:
        """Get the full trajectory for execution from the best candidate."""
        if self.best_candidate is None or not self.best_candidate.is_valid:
            return None

        return self.evaluator._combine_trajectories(
            self.best_candidate.pre_grasp_trajectory,
            self.best_candidate.approach_trajectory,
        )

    def replan_if_needed(
        self, current_joint_state: JointState, execution_progress: float = 0.0
    ) -> Optional[TrajectoryCandidate]:
        """Re-plan if the current trajectory becomes invalid or sub-optimal, or for incremental approach."""
        if not self.planning_config["enable_re_planning"]:
            return self.best_candidate

        # If we don't have a best candidate yet, return None
        if self.best_candidate is None:
            return None

        # If we're in the approach phase and have completed pre-grasp trajectory
        if (
            execution_progress > 0.5
            and self.best_candidate.pre_grasp_trajectory is not None
        ):
            print("🔄 Planning next approach step with adaptive step size...")

            try:
                # Get current end-effector position using FK
                current_js_for_fk = current_joint_state.get_ordered_joint_state(
                    self.motion_gen.kinematics.joint_names
                )
                ee_state = self.motion_gen.kinematics.compute_kinematics(
                    current_js_for_fk
                )
                current_position = ee_state.ee_pose.position.cpu().numpy().flatten()

                # Get target position and orientation from best candidate
                target_position = (
                    self.best_candidate.target_pose.position.cpu().numpy().flatten()
                )
                current_orientation = (
                    self.best_candidate.target_pose.quaternion.cpu().numpy().flatten()
                )

                # Get approach vector for this orientation
                orientation_idx = self.best_candidate.orientation_index

                # Use locally defined GRIPPER_VECTOR_LIST instead of importing
                approach_vector = GRIPPER_VECTOR_LIST[orientation_idx]

                # Calculate distance to target
                vector_to_target = target_position - current_position
                remaining_distance = np.linalg.norm(vector_to_target)

                # Check if we've reached the target (very close)
                if remaining_distance < 0.003:  # Within 3mm of target
                    print(
                        f"🎯 TARGET REACHED! Final distance: {remaining_distance * 1000:.2f}mm"
                    )
                    print(f"Current position: {current_position}")
                    print(f"Target position: {target_position}")
                    return self.best_candidate

                # Plan the next step with adaptive sizing
                next_step_plan = self._plan_next_approach_step(
                    current_joint_state,
                    current_position,
                    target_position,
                    approach_vector,
                    current_orientation,
                )

                if next_step_plan is not None:
                    # Create a new candidate with just this step
                    step_candidate = TrajectoryCandidate(
                        orientation_index=self.best_candidate.orientation_index,
                        pre_grasp_pose=self.best_candidate.pre_grasp_pose,
                        target_pose=self.best_candidate.target_pose,
                        # Use the next step as the approach trajectory
                        approach_trajectory=next_step_plan,
                    )

                    step_candidate.planning_success = True
                    step_candidate.is_valid = True

                    return step_candidate
                else:
                    print("❌ Failed to plan next approach step")
                    return self.best_candidate

            except Exception as e:
                print(f"Error planning next approach step: {e}")
                import traceback

                print(traceback.format_exc())
                return self.best_candidate

        return self.best_candidate

    def get_planning_statistics(self) -> Dict:
        """Get statistics about the last planning iteration."""
        if not self.candidates:
            return {}

        valid_count = sum(1 for c in self.candidates if c.is_valid)
        collision_free_count = sum(1 for c in self.candidates if c.collision_free)

        costs = [c.total_cost for c in self.candidates if c.is_valid]

        return {
            "total_candidates": len(self.candidates),
            "valid_candidates": valid_count,
            "collision_free_candidates": collision_free_count,
            "best_cost": min(costs) if costs else float("inf"),
            "average_cost": np.mean(costs) if costs else float("inf"),
            "best_orientation": self.best_candidate.orientation_index
            if self.best_candidate
            else None,
        }

    def plan_multi_candidate_trajectory(
        self, current_joint_state: JointState
    ) -> Optional[TrajectoryCandidate]:
        """Plan and evaluate multiple candidate trajectories, return the best one."""
        print("🔄 Planning multiple candidate trajectories...")

        start_time = time.time()
        self.candidates = []

        # Determine max candidates to evaluate
        max_candidates = self.planning_config.get(
            "max_candidates", len(GRIPPER_ORIENTATION_LIST)
        )
        max_candidates = min(max_candidates, len(GRIPPER_ORIENTATION_LIST))

        print(f"Planning for up to {max_candidates} candidate orientations")

        # Generate candidates for all orientations (up to max_candidates)
        for orientation_idx in range(max_candidates):
            if time.time() - start_time > self.planning_config["planning_timeout"]:
                print(
                    f"⏰ Planning timeout reached, using {len(self.candidates)} candidates"
                )
                break

            candidate = self._generate_candidate(orientation_idx, current_joint_state)
            if candidate is not None:
                self.candidates.append(candidate)

        print(f"📊 Generated {len(self.candidates)} candidate trajectories")

        # Evaluate all candidates
        valid_candidates = []
        for candidate in self.candidates:
            evaluated_candidate = self.evaluator.evaluate_trajectory(candidate)
            if evaluated_candidate.is_valid:
                valid_candidates.append(evaluated_candidate)
                print(
                    f"✅ Orientation {evaluated_candidate.orientation_index}: "
                    f"Cost={evaluated_candidate.total_cost:.3f}, "
                    f"Collision-free={evaluated_candidate.collision_free}"
                )
            else:
                print(
                    f"❌ Orientation {evaluated_candidate.orientation_index}: Invalid"
                )

        if not valid_candidates:
            print("❌ No valid candidate trajectories found!")
            return None

        # Select best candidate
        self.best_candidate = min(valid_candidates, key=lambda c: c.total_cost)

        print(
            f"🏆 Best candidate: Orientation {self.best_candidate.orientation_index} "
            f"with cost {self.best_candidate.total_cost:.3f}"
        )

        return self.best_candidate


# Example usage function
def create_multi_candidate_planner(
    motion_gen: MotionGen, target_position: List[float], planning_config: Dict = None
) -> ZordiMultiCandidatePlanner:
    """Create a multi-candidate motion planner.

    Args:
        motion_gen: The motion generator
        target_position: Target position [x, y, z]
        planning_config: Optional configuration parameters for planning

    Returns:
        The multi-candidate planner

    Version: Updated 2025-06-03 to properly handle planning_config parameter
    """
    # Convert target position to tensor
    target_position_tensor = torch.tensor(
        target_position, dtype=torch.float32, device=motion_gen.tensor_args.device
    ).unsqueeze(0)

    planner = ZordiMultiCandidatePlanner(
        motion_gen=motion_gen,
        target_position=target_position_tensor,
        planning_config=planning_config,
    )

    return planner
