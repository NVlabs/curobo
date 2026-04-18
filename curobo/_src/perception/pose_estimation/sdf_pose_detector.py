# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""SDF-based pose detector using mesh SDF queries and Levenberg-Marquardt optimization.

Uses ground-truth mesh SDF for implicit correspondence finding and analytic gradients.
More accurate than ICP for meshes with known geometry as it doesn't require explicit
correspondence search.

Key features:
- Mesh SDF queries for distance and gradient computation
- Levenberg-Marquardt with trust-region updates
- Huber loss for outlier robustness
- CUDA graph compatible (mesh_id captured at init)
- Two-pass kernel design for better GPU occupancy

Usage:
    robot_mesh = RobotMesh.from_trimesh(mesh, device="cuda:0")
    config = SDFDetectorCfg()
    detector = SDFPoseDetector(robot_mesh, config)
    result = detector.detect(observed_points, initial_pose)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.perception.optim_pose_lm import (
    compute_predicted_reduction,
    solve_lm_step,
    trust_region_update,
)
from curobo._src.perception.pose_estimation.detection_result import DetectionResult
from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo._src.perception.pose_estimation.util import extract_observed_points
from curobo._src.perception.pose_estimation.wp_mesh_sdf_alignment import (
    jacobian_reduce_kernel,
    mesh_surface_distance_query_kernel,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_profiler_decorator
from curobo._src.util.warp import get_warp_device_stream


@dataclass
class SDFRefinementState:
    """State for one SDF refinement session."""

    observed_points: torch.Tensor  # [N, 3] observed points in world frame
    n_points: int

    best_position: torch.Tensor  # [3]
    best_quaternion: torch.Tensor  # [4] wxyz

    best_error: torch.Tensor
    best_sum_sq: torch.Tensor  # Sum of squared residuals (for trust ratio)
    best_n_valid: torch.Tensor
    best_JtJ: torch.Tensor  # [6, 6]
    best_Jtr: torch.Tensor  # [6]

    lambda_damping: torch.Tensor

    translation_change: torch.Tensor
    rotation_change: torch.Tensor

    def clone(self) -> "SDFRefinementState":
        """Create a deep copy of this state."""
        return SDFRefinementState(
            observed_points=self.observed_points.clone(),
            n_points=self.n_points,
            best_position=self.best_position.clone(),
            best_quaternion=self.best_quaternion.clone(),
            best_error=self.best_error.clone(),
            best_sum_sq=self.best_sum_sq.clone(),
            best_n_valid=self.best_n_valid.clone(),
            best_JtJ=self.best_JtJ.clone(),
            best_Jtr=self.best_Jtr.clone(),
            lambda_damping=self.lambda_damping.clone(),
            translation_change=self.translation_change.clone(),
            rotation_change=self.rotation_change.clone(),
        )

    def copy_(self, other: "SDFRefinementState") -> "SDFRefinementState":
        """Copy values from another state in-place (for CUDA graph compatibility)."""
        self.observed_points.copy_(other.observed_points)
        self.n_points = other.n_points
        self.best_position.copy_(other.best_position)
        self.best_quaternion.copy_(other.best_quaternion)
        self.best_error.copy_(other.best_error)
        self.best_sum_sq.copy_(other.best_sum_sq)
        self.best_n_valid.copy_(other.best_n_valid)
        self.best_JtJ.copy_(other.best_JtJ)
        self.best_Jtr.copy_(other.best_Jtr)
        self.lambda_damping.copy_(other.lambda_damping)
        self.translation_change.copy_(other.translation_change)
        self.rotation_change.copy_(other.rotation_change)
        return self


class SDFPoseDetector:
    """SDF-based pose detector using mesh SDF queries and Levenberg-Marquardt.

    Uses ground-truth mesh SDF for implicit correspondence finding.
    Supports both rigid objects and articulated robots via RobotMesh.

    Uses two-pass kernel design for reduced register pressure and better GPU occupancy:
    - Pass 1: Query mesh SDF (BVH traversal)
    - Pass 2: Compute Jacobian and block-reduce
    """

    def __init__(
        self,
        robot_mesh: RobotMesh,
        config: Optional[SDFDetectorCfg] = None,
    ):
        """Initialize SDF pose detector.

        Args:
            robot_mesh: RobotMesh instance providing mesh for SDF queries.
            config: Detector configuration.
        """
        self.robot_mesh = robot_mesh
        self.config = config or SDFDetectorCfg()
        self.device = robot_mesh.device

        # Store mesh_id for CUDA graph compatibility
        self._mesh_id = robot_mesh.mesh_id

        # Pre-allocate buffers
        self._allocate_buffers()

        # Identity matrix for LM damping
        self._eye6 = torch.eye(6, dtype=torch.float32, device=self.device)

        # CUDA graph executor for inner iterations
        self._refine_inner_executor = GraphExecutor(
            capture_fn=self._refine_inner_iterations,
            device=torch.device(self.device),
            use_cuda_graph=self.config.use_cuda_graph,
            clone_outputs=True,
        )

    def _allocate_buffers(self):
        """Pre-allocate GPU buffers for reuse."""
        n = self.config.n_points

        # Output buffers for reduction
        self._JtJ_out = torch.zeros(36, dtype=torch.float32, device=self.device)
        self._Jtr_out = torch.zeros(6, dtype=torch.float32, device=self.device)
        self._sum_sq_residuals = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._valid_count = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Intermediate buffers for two-pass kernel (~10KB for 500 points)
        self._sdf_values = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._gradients_world = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        self._valid_mask = torch.zeros(n, dtype=torch.int32, device=self.device)

        # Pose buffers
        self._cand_position = torch.zeros(3, dtype=torch.float32, device=self.device)
        self._cand_quaternion = torch.zeros(4, dtype=torch.float32, device=self.device)

        # Infinity tensor for trust region update (CUDA graph compatible)
        self._inf_tensor = torch.tensor(float("inf"), dtype=torch.float32, device=self.device)

        # Pre-allocated tensor for n_total_valid (CUDA graph compatible)
        self._n_total_valid = torch.tensor(
            float(self.config.n_points), dtype=torch.float32, device=self.device
        )

        # Pre-allocated tensor for lambda_damping (CUDA graph compatible)
        self._lambda_damping_init = torch.tensor(
            self.config.lambda_initial, dtype=torch.float32, device=self.device
        )
        self._translation_change_init = torch.zeros(3, dtype=torch.float32, device=self.device)
        self._rotation_change_init = torch.zeros(3, dtype=torch.float32, device=self.device)

    @get_profiler_decorator("sdf_pose_detector/evaluate_at_pose")
    def _evaluate_at_pose(
        self,
        observed_points: torch.Tensor,
        n_points: int,
        position: torch.Tensor,
        quaternion: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate cost and Jacobian at given pose using two-pass kernels.

        Pass 1: Query mesh SDF, store distances and gradients
        Pass 2: Compute Jacobian and block-reduce

        Returns:
            Tuple of (JtJ, Jtr, sum_sq_residuals, valid_count).
        """
        # Reset output buffers
        self._JtJ_out.zero_()
        self._Jtr_out.zero_()
        self._sum_sq_residuals.zero_()
        self._valid_count.zero_()
        self._sdf_values.zero_()
        self._gradients_world.zero_()
        self._valid_mask.zero_()
        # Get CUDA stream for async execution
        _, stream = get_warp_device_stream(observed_points)

        check_float32_tensors(position.device, position=position, quaternion=quaternion)
        # Pass 1: Mesh surface distance query (high register, low occupancy but isolated)
        wp.launch(
            kernel=mesh_surface_distance_query_kernel,
            dim=n_points,
            inputs=[
                wp.from_torch(observed_points, dtype=wp.vec3),
                n_points,
                wp.from_torch(position, dtype=wp.float32),
                wp.from_torch(quaternion, dtype=wp.float32),
                self._mesh_id,
                self.config.max_distance,
                self.config.distance_threshold,
                wp.from_torch(self._sdf_values, dtype=wp.float32),
                wp.from_torch(self._gradients_world, dtype=wp.vec3),
                wp.from_torch(self._valid_mask, dtype=wp.int32),
            ],
            block_dim=256,  # Higher block dim for better occupancy
            device=self.device,
            stream=stream,
        )

        # Pass 2: Jacobian computation and reduction (lower register)
        wp.launch(
            kernel=jacobian_reduce_kernel,
            dim=n_points,
            inputs=[
                wp.from_torch(observed_points, dtype=wp.vec3),
                wp.from_torch(self._sdf_values, dtype=wp.float32),
                wp.from_torch(self._gradients_world, dtype=wp.vec3),
                wp.from_torch(self._valid_mask, dtype=wp.int32),
                n_points,
                1 if self.config.use_huber else 0,
                self.config.huber_delta,
                wp.from_torch(self._JtJ_out, dtype=wp.float32),
                wp.from_torch(self._Jtr_out, dtype=wp.float32),
                wp.from_torch(self._sum_sq_residuals, dtype=wp.float32),
                wp.from_torch(self._valid_count, dtype=wp.int32),
            ],
            block_dim=256,  # Match tile operations
            device=self.device,
            stream=stream,
        )

        return (
            self._JtJ_out.view(6, 6),
            self._Jtr_out,
            self._sum_sq_residuals,
            self._valid_count,
        )

    @get_profiler_decorator("sdf_pose_detector/setup_refinement")
    def _setup_refinement(
        self,
        observed_points: torch.Tensor,
        initial_pose: Pose,
    ) -> SDFRefinementState:
        """Setup refinement state from initial pose."""
        n_points = len(observed_points)

        # Update pre-allocated n_total_valid (in-place for CUDA graph compatibility)
        self._n_total_valid.fill_(float(n_points))

        # Extract position and quaternion from pose
        position = initial_pose.position[0].to(self.device)  # [3]
        quaternion = initial_pose.quaternion[0].to(self.device)  # [4] wxyz

        # Evaluate at initial pose
        JtJ, Jtr, sum_sq, n_valid = self._evaluate_at_pose(
            observed_points, n_points, position, quaternion
        )

        # Compute initial error
        error = torch.sqrt(sum_sq / (n_valid + 1e-8))

        return SDFRefinementState(
            observed_points=observed_points,
            n_points=n_points,
            best_position=position,
            best_quaternion=quaternion,
            best_error=error.squeeze(),
            best_sum_sq=sum_sq.squeeze(),
            best_n_valid=n_valid.squeeze(),
            best_JtJ=JtJ,
            best_Jtr=Jtr,
            lambda_damping=self._lambda_damping_init,
            translation_change=self._translation_change_init,
            rotation_change=self._rotation_change_init,
        )

    @get_profiler_decorator("sdf_pose_detector/refine_iteration")
    def _refine_iteration(
        self,
        state: SDFRefinementState,
    ) -> Tuple[SDFRefinementState, torch.Tensor, torch.Tensor]:
        """Perform one LM iteration.

        Returns:
            Tuple of (updated_state, translation_change, rotation_change).
        """
        # Solve LM step
        delta = solve_lm_step(
            state.best_JtJ,
            state.best_Jtr,
            state.lambda_damping,
            self._eye6,
        )

        # Extract translation and rotation updates
        delta_t = delta[:3]
        delta_r = delta[3:]

        # Compute candidate pose using Pose.from_euler_xyz and multiply
        delta_pose = Pose.from_euler_xyz(delta_r, delta_t)
        current_pose = Pose(
            position=state.best_position.unsqueeze(0),
            quaternion=state.best_quaternion.unsqueeze(0),
        )
        cand_pose = delta_pose.multiply(current_pose, out_position=self._cand_position, out_quaternion=self._cand_quaternion)
        cand_position = cand_pose.position.squeeze(0)
        cand_quaternion = cand_pose.quaternion.squeeze(0)

        # Compute predicted reduction for trust ratio
        pred_reduction = compute_predicted_reduction(delta, state.best_Jtr, state.best_JtJ)

        # Evaluate at candidate pose
        cand_JtJ, cand_Jtr, sum_sq, cand_n_valid = self._evaluate_at_pose(
            state.observed_points, state.n_points, cand_position, cand_quaternion
        )

        # Trust region update
        (
            new_position,
            new_quaternion,
            new_error,
            new_sum_sq,
            new_n_valid,
            new_JtJ,
            new_Jtr,
            new_lambda,
        ) = trust_region_update(
            cand_n_valid=cand_n_valid.squeeze(),
            sum_sq_residuals=sum_sq.squeeze(),
            cand_JtJ=cand_JtJ,
            cand_Jtr=cand_Jtr,
            cand_position=cand_position,
            cand_quaternion=cand_quaternion,
            best_error=state.best_error,
            best_sum_sq=state.best_sum_sq,
            best_n_valid=state.best_n_valid,
            best_JtJ=state.best_JtJ,
            best_Jtr=state.best_Jtr,
            best_position=state.best_position,
            best_quaternion=state.best_quaternion,
            pred_reduction=pred_reduction,
            lambda_damping=state.lambda_damping,
            n_total_valid=self._n_total_valid,
            min_valid_ratio=self.config.min_valid_ratio,
            rho_min=self.config.rho_min,
            lambda_factor=self.config.lambda_factor,
            lambda_min=self.config.lambda_min,
            lambda_max=self.config.lambda_max,
            inf_tensor=self._inf_tensor,
        )

        # Compute convergence metrics
        translation_change = delta_t
        rotation_change = delta_r

        # Update state
        new_state = SDFRefinementState(
            observed_points=state.observed_points,
            n_points=state.n_points,
            best_position=new_position,
            best_quaternion=new_quaternion,
            best_error=new_error,
            best_sum_sq=new_sum_sq,
            best_n_valid=new_n_valid,
            best_JtJ=new_JtJ,
            best_Jtr=new_Jtr,
            lambda_damping=new_lambda,
            translation_change=translation_change,
            rotation_change=rotation_change,
        )

        return new_state



    @get_profiler_decorator("sdf_pose_detector/refine_inner_iterations")
    def _refine_inner_iterations(
        self,
        state: SDFRefinementState,
    ) -> SDFRefinementState:
        """Perform inner iterations of refinement (no convergence check).

        This method is captured by CUDA graph for acceleration.
        Convergence is checked in the outer loop only.

        Args:
            state: Current refinement state.

        Returns:
            Updated refinement state.
        """
        current_state = state

        for _ in range(self.config.inner_iterations):
            current_state = self._refine_iteration(current_state)

        return current_state

    @get_profiler_decorator("sdf_pose_detector/detect")
    def detect(
        self,
        camera_obs: CameraObservation,
        config: Optional[torch.Tensor] = None,
        initial_pose: Optional[Pose] = None,
    ) -> DetectionResult:
        """Detect pose from camera observation.

        Args:
            camera_obs: Camera observation with depth and segmentation.
            config: Joint angles for articulated robots (updates mesh). Ignored for rigid.
            initial_pose: Initial pose estimate. Required for SDF detector.

        Returns:
            DetectionResult with refined pose.
        """
        observed_points = self._extract_observed_points(camera_obs)
        return self.detect_from_points(observed_points, config, initial_pose)

    @get_profiler_decorator("sdf_pose_detector/detect_from_points")
    def detect_from_points(
        self,
        observed_points: torch.Tensor,
        config: Optional[torch.Tensor] = None,
        initial_pose: Optional[Pose] = None,
    ) -> DetectionResult:
        """Detect object pose from pre-segmented 3D points.

        Args:
            observed_points: [N, 3] observed points in world frame.
            config: Joint angles for articulated robots (updates mesh). Ignored for rigid.
            initial_pose: Initial pose estimate. Required for SDF detector (no random sampling).

        Returns:
            DetectionResult with refined pose.
        """
        # Start timing with CUDA events (non-blocking)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if initial_pose is None:
            log_and_raise("SDFPoseDetector requires an initial_pose estimate")

        # Update mesh if articulated
        if config is not None:
            self.robot_mesh.update(config)

        # Ensure points on device
        points = observed_points.to(device=self.device, dtype=torch.float32)

        # Subsample if needed
        if len(points) > self.config.n_points:
            indices = torch.randperm(len(points), device=self.device)[: self.config.n_points]
            points = points[indices]

        # Setup refinement
        state = self._setup_refinement(points, initial_pose)

        # Outer/inner loop pattern for CUDA graph compatibility
        # Inner iterations are captured in graph, outer loop checks convergence
        outer_iterations = self.config.max_iterations // self.config.inner_iterations
        n_iterations = 0

        for outer_i in range(outer_iterations):
            # Run inner iterations (potentially accelerated by CUDA graph)
            state = self._refine_inner_executor(state)
            n_iterations = (outer_i + 1) * self.config.inner_iterations

            if (
                state.translation_change.norm() < self.config.convergence_threshold
                and state.rotation_change.norm() < self.config.rotation_convergence_threshold
            ):
                break

        # Build result
        final_pose = Pose(
            position=state.best_position.unsqueeze(0),
            quaternion=state.best_quaternion.unsqueeze(0),
        )

        # Compute confidence from valid ratio
        valid_ratio = state.best_n_valid.item() / state.n_points
        confidence = min(1.0, valid_ratio / self.config.min_valid_ratio)

        # End timing (only syncs this event, not full device)
        end_event.record()
        end_event.synchronize()
        compute_time = start_event.elapsed_time(end_event) / 1000.0  # ms -> seconds

        return DetectionResult(
            pose=final_pose,
            config=config,
            confidence=confidence,
            alignment_error=state.best_error.item(),
            n_iterations=n_iterations,
            compute_time=compute_time,
        )

    @get_profiler_decorator("sdf_pose_detector/extract_observed_points")
    def _extract_observed_points(self, camera_obs: CameraObservation) -> torch.Tensor:
        """Extract and filter observed points from camera observation."""
        return extract_observed_points(camera_obs)
