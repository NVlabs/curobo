# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Raycast-based camera pose refinement for block-sparse TSDF.

Uses KinectFusion-style raycasting through the block-sparse TSDF for
correspondence finding. More robust than EDT lookup for large pose errors
and complex geometry.

Key differences from dense pose refiner:
- Uses hash table lookups instead of direct grid indexing
- Center-origin bounded TSDF grid with sparse block allocation

Reference:
- "KinectFusion: Real-time Dense Surface Mapping and Tracking" (Newcombe et al.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import warp as wp

from curobo._src.curobolib.backends.cuda_core_backend.pose_refiner import (
    launch_pose_refiner_accept_candidate,
    launch_pose_refiner_count_valid_depth,
    launch_pose_refiner_initialize_state,
    launch_pose_refiner_solve_candidate,
)
from curobo._src.perception.mapper.kernel.wp_raycast_pose_refine import (
    create_ray_sdf_alignment_block_sparse_tiled_kernel,
)
from curobo._src.types.pose import Pose
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.torch_util import get_profiler_decorator
from curobo._src.util.warp import get_warp_device_stream

if TYPE_CHECKING:
    from curobo._src.perception.mapper.integrator_esdf import (
        BlockSparseESDFIntegrator,
    )
    from curobo._src.perception.mapper.integrator_tsdf import (
        BlockSparseTSDFIntegrator,
    )


@dataclass
class BlockSparseRefinementState:
    """State for one refinement session using trust-region Levenberg-Marquardt.

    Created by _setup_refinement, used by _refine_iteration_tiled.
    The best_* fields represent the current accepted pose that we iterate from.

    Attributes:
        depth: [H, W] depth image on device.
        intrinsics: [3, 3] camera intrinsic matrix on device.
        n_valid_depth_pixels: Number of valid depth pixels.
        stride: Sampling stride for depth image.
        out_W: Output grid width.
        n_pixels: Number of sampled pixels.
        origin: [3] TSDF grid origin.
        voxel_size: TSDF voxel size in meters.
        block_size: Voxels per block edge.
        hash_capacity: Hash table capacity.
        truncation_distance: TSDF truncation distance.
        warp_data: Reference to block-sparse Warp data.
        best_position: [1, 3] current accepted position.
        best_quaternion: [1, 4] current accepted quaternion.
        best_error: Error at current accepted pose.
        best_n_valid: Number of valid correspondences at current accepted pose.
        best_JtJ: [6, 6] J^T @ J at current accepted pose.
        best_Jtr: [6] J^T @ r at current accepted pose.
        lambda_damping: LM damping parameter.
        result_float: [2] final error and damping result storage.
        result_int: [4] iteration, valid-count, accepted-step, and status storage.
    """

    # Input data
    depth: torch.Tensor
    intrinsics: torch.Tensor
    n_valid_depth_pixels: torch.Tensor
    stride: int
    out_W: int
    n_pixels: int

    # Block-sparse TSDF params
    origin: torch.Tensor
    voxel_size: float
    block_size: int
    hash_capacity: int
    truncation_distance: float
    warp_data: object  # BlockSparseTSDFWarp
    grid_W: int
    grid_H: int
    grid_D: int

    # Current accepted pose
    best_position: torch.Tensor
    best_quaternion: torch.Tensor

    # Jacobian info at current accepted pose
    best_error: torch.Tensor
    best_sum_sq: torch.Tensor  # Sum of squared residuals for trust ratio
    best_n_valid: torch.Tensor
    best_JtJ: torch.Tensor
    best_Jtr: torch.Tensor

    # LM damping parameter
    lambda_damping: torch.Tensor
    result_float: torch.Tensor
    result_int: torch.Tensor

    def clone(self) -> BlockSparseRefinementState:
        """Create a deep copy of this state."""
        return BlockSparseRefinementState(
            depth=self.depth.clone(),
            intrinsics=self.intrinsics.clone(),
            n_valid_depth_pixels=self.n_valid_depth_pixels.clone(),
            stride=self.stride,
            out_W=self.out_W,
            n_pixels=self.n_pixels,
            origin=self.origin.clone(),
            voxel_size=self.voxel_size,
            block_size=self.block_size,
            hash_capacity=self.hash_capacity,
            truncation_distance=self.truncation_distance,
            warp_data=self.warp_data,
            grid_W=self.grid_W,
            grid_H=self.grid_H,
            grid_D=self.grid_D,
            best_position=self.best_position.clone(),
            best_quaternion=self.best_quaternion.clone(),
            best_error=self.best_error.clone(),
            best_sum_sq=self.best_sum_sq.clone(),
            best_n_valid=self.best_n_valid.clone(),
            best_JtJ=self.best_JtJ.clone(),
            best_Jtr=self.best_Jtr.clone(),
            lambda_damping=self.lambda_damping.clone(),
            result_float=self.result_float.clone(),
            result_int=self.result_int.clone(),
        )

    def copy_(self, other: BlockSparseRefinementState) -> BlockSparseRefinementState:
        """Copy values from another state in-place."""
        self.depth.copy_(other.depth)
        self.intrinsics.copy_(other.intrinsics)
        self.n_valid_depth_pixels.copy_(other.n_valid_depth_pixels)
        self.stride = other.stride
        self.out_W = other.out_W
        self.n_pixels = other.n_pixels
        self.origin.copy_(other.origin)
        self.voxel_size = other.voxel_size
        self.block_size = other.block_size
        self.hash_capacity = other.hash_capacity
        self.truncation_distance = other.truncation_distance
        self.warp_data = other.warp_data
        self.grid_W = other.grid_W
        self.grid_H = other.grid_H
        self.grid_D = other.grid_D
        self.best_position.copy_(other.best_position)
        self.best_quaternion.copy_(other.best_quaternion)
        self.best_error.copy_(other.best_error)
        self.best_sum_sq.copy_(other.best_sum_sq)
        self.best_n_valid.copy_(other.best_n_valid)
        self.best_JtJ.copy_(other.best_JtJ)
        self.best_Jtr.copy_(other.best_Jtr)
        self.lambda_damping.copy_(other.lambda_damping)
        self.result_float.copy_(other.result_float)
        self.result_int.copy_(other.result_int)
        return self


@dataclass
class BlockSparseRaycastRefinerCfg:
    """Configuration for BlockSparseRaycastPoseRefiner.

    Attributes:
        n_points: Number of points to use for ICP (subsampled from depth).
        max_iterations: Maximum ICP iterations.
        minimum_valid_depth_pixels: Minimum valid depth pixels to proceed.
        distance_threshold: Reject correspondences beyond this distance (meters).
        min_valid_ratio: Minimum ratio of valid correspondences to proceed.
        n_samples_per_ray: Number of SDF samples per ray.
        tile_block_dim: Block size for tiled kernel.
        depth_minimum_distance: Minimum valid depth (meters).
        depth_maximum_distance: Maximum valid depth (meters).
        minimum_tsdf_weight: Minimum TSDF weight for valid voxel.
            Since weight = 1/depth² (clamped to [0.001, 2.0]), this can be interpreted
            as the number of observations at 1m depth. E.g., 0.5 = half an observation
            at 1m, or one observation at ~1.4m, or two observations at 2m.
        lambda_initial: Initial LM damping parameter.
        lambda_factor: Factor to multiply/divide lambda on reject/accept.
        lambda_min: Minimum lambda value.
        lambda_max: Maximum lambda value.
        rho_min: Minimum trust ratio for step acceptance.
    """

    n_points: int = (1280 * 720) // 10
    minimum_valid_depth_pixels: int = 100
    max_iterations: int = 100
    inner_iterations: int = 4
    distance_threshold: float = 0.1
    min_valid_ratio: float = 0.1
    n_samples_per_ray: int = 1
    tile_block_dim: int = 256

    # Depth filtering
    depth_minimum_distance: float = 0.1
    depth_maximum_distance: float = 10.0
    minimum_tsdf_weight: float = 2.0

    # Levenberg-Marquardt trust-region parameters
    lambda_initial: float = 1e-3
    lambda_factor: float = 10.0
    lambda_min: float = 1e-7
    lambda_max: float = 1e7
    rho_min: float = 0.25

    @property
    def outer_iterations(self) -> int:
        """Number of outer iterations."""
        return self.max_iterations // self.inner_iterations


class BlockSparseRaycastPoseRefiner:
    """Camera pose refinement using block-sparse TSDF raycasting.

    Uses KinectFusion-style raycasting for correspondence finding with
    block-sparse TSDF storage. More memory-efficient than dense version
    for large-scale environments.

    Example:
        refiner = BlockSparseRaycastPoseRefiner(integrator)
        refined_pose, error, num_iters = refiner.refine_pose(
            depth, intrinsics, estimated_pose
        )
        observation.pose = refined_pose
        integrator.integrate(observation)
    """

    def __init__(
        self,
        integrator: "BlockSparseESDFIntegrator | BlockSparseTSDFIntegrator",
        config: Optional[BlockSparseRaycastRefinerCfg] = None,
    ) -> None:
        """Initialize BlockSparseRaycastPoseRefiner.

        Args:
            integrator: Block-sparse integrator instance with TSDF.
            config: Refinement configuration.
        """
        self.integrator = integrator
        self.config = config or BlockSparseRaycastRefinerCfg()
        self.device = integrator.device

        # Pre-allocate buffers
        self._allocate_buffers()

        # Create tiled kernel bound to the integrator's TSDF
        # specialization. The sampler helpers closure-capture the
        # matching ``block_size``; without this the tiled kernel would
        # silently use BS=8 samplers against a BS≠8 TSDF.
        self._tiled_kernel = create_ray_sdf_alignment_block_sparse_tiled_kernel(
            n_samples_per_ray=self.config.n_samples_per_ray,
            kernels=integrator._tsdf.kernels,
        )

        # CUDA graph executor for inner iterations
        self._refine_inner_iteration_executor = GraphExecutor(
            capture_fn=self._refine_inner_iterations,
            device=self.device,
            use_cuda_graph=True,
            clone_outputs=False,
        )

    def _allocate_buffers(self) -> None:
        """Pre-allocate GPU buffers for reuse."""
        # Tiled kernel output buffers
        self._JtJ_flat = torch.zeros(36, dtype=torch.float32, device=self.device)
        self._Jtr = torch.zeros(6, dtype=torch.float32, device=self.device)
        self._sum_sq_residuals = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._valid_count = torch.zeros(1, dtype=torch.int32, device=self.device)

        # State buffers
        self._state_JtJ = torch.zeros((6, 6), dtype=torch.float32, device=self.device)
        self._state_Jtr = torch.zeros(6, dtype=torch.float32, device=self.device)
        self._state_position = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self._state_quaternion = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        self._state_error = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._state_sum_sq = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._state_valid_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._state_lambda = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._state_result_float = torch.zeros(2, dtype=torch.float32, device=self.device)
        self._state_result_int = torch.zeros(4, dtype=torch.int32, device=self.device)

        # Candidate and setup buffers remain fixed across CUDA graph capture.
        self._candidate_position = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self._candidate_quaternion = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        self._predicted_reduction = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._valid_depth_count = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Cached state
        self._cached_state: Optional[BlockSparseRefinementState] = None

    @get_profiler_decorator("block_sparse_pose_refiner/refine_pose")
    def refine_pose(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        estimated_pose: Pose,
    ) -> Tuple[Pose, float, int]:
        """Refine camera pose using raycast-based ICP.

        Args:
            depth: [H, W] depth image in meters.
            intrinsics: [3, 3] camera intrinsic matrix.
            estimated_pose: Initial pose estimate (camera to world).

        Returns:
            refined_pose: Refined pose.
            final_error: Mean alignment error (meters).
            n_iterations: Number of ICP iterations performed.
        """
        return self._refine_pose_along_ray(depth, intrinsics, estimated_pose)

    @get_profiler_decorator("block_sparse_pose_refiner/_setup_refinement")
    def _setup_refinement(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        estimated_pose: Pose,
    ) -> Optional[BlockSparseRefinementState]:
        """Setup state for refinement iterations.

        Args:
            depth: [H, W] depth image in meters.
            intrinsics: [3, 3] camera intrinsic matrix.
            estimated_pose: Initial pose estimate.

        Returns:
            BlockSparseRefinementState if setup succeeded, None otherwise.
        """
        # Get TSDF parameters
        tsdf = self.integrator.tsdf

        # Calculate valid depth pixels
        depth_on_device = depth.to(self.device, dtype=torch.float32).contiguous()
        self._valid_depth_count.zero_()
        launch_pose_refiner_count_valid_depth(
            depth_on_device,
            self.config.depth_minimum_distance,
            self.config.depth_maximum_distance,
            self._valid_depth_count,
        )
        if self._valid_depth_count.item() < self.config.minimum_valid_depth_pixels:
            return None

        # Compute sampling grid params
        img_H, img_W = depth.shape
        stride = max(1, int((img_H * img_W / self.config.n_points) ** 0.5))
        out_H = (img_H + stride - 1) // stride
        out_W = (img_W + stride - 1) // stride
        n_pixels = out_H * out_W

        # Prepare pose tensors
        intrinsics_on_device = intrinsics.to(self.device, dtype=torch.float32).contiguous()
        position = (
            estimated_pose.position.squeeze(0).to(self.device, dtype=torch.float32).view(1, 3)
        )
        quaternion = (
            estimated_pose.quaternion.squeeze(0).to(self.device, dtype=torch.float32).view(1, 4)
        )

        # Get block-sparse data
        warp_data = tsdf.get_warp_data()
        origin = tsdf.data.origin
        voxel_size = tsdf.data.voxel_size
        block_size = tsdf.data.block_size
        hash_capacity = tsdf.config.hash_capacity
        truncation_distance = tsdf.config.truncation_distance

        # Get grid dimensions for center-origin convention
        grid_D, grid_H_dim, grid_W_dim = tsdf.data.grid_shape  # nz, ny, nx -> D, H, W

        # Compute initial JtJ/Jtr/error
        n_total_samples = n_pixels * self.config.n_samples_per_ray

        self._JtJ_flat.zero_()
        self._Jtr.zero_()
        self._sum_sq_residuals.zero_()
        self._valid_count.zero_()

        _, stream = get_warp_device_stream(depth_on_device)

        wp.launch(
            kernel=self._tiled_kernel,
            dim=n_total_samples,
            inputs=[
                wp.from_torch(depth_on_device, dtype=wp.float32),
                wp.from_torch(intrinsics_on_device, dtype=wp.float32),
                wp.from_torch(position.view(3), dtype=wp.float32),
                wp.from_torch(quaternion.view(4), dtype=wp.float32),
                warp_data,
                self.config.minimum_tsdf_weight,
                self.config.depth_minimum_distance,
                self.config.depth_maximum_distance,
                self.config.distance_threshold,
                stride,
                out_W,
                n_pixels,
                wp.from_torch(self._JtJ_flat, dtype=wp.float32),
                wp.from_torch(self._Jtr, dtype=wp.float32),
                wp.from_torch(self._sum_sq_residuals, dtype=wp.float32),
                wp.from_torch(self._valid_count, dtype=wp.int32),
            ],
            block_dim=self.config.tile_block_dim,
            device=wp.device_from_torch(self.device),
            stream=stream,
            adjoint=False,
        )

        launch_pose_refiner_initialize_state(
            position,
            quaternion,
            self._JtJ_flat,
            self._Jtr,
            self._sum_sq_residuals,
            self._valid_count,
            self._valid_depth_count,
            self.config.minimum_valid_depth_pixels,
            self.config.minimum_valid_depth_pixels,
            self.config.lambda_initial,
            self._state_position,
            self._state_quaternion,
            self._state_JtJ,
            self._state_Jtr,
            self._state_sum_sq,
            self._state_error,
            self._state_valid_count,
            self._state_lambda,
            self._state_result_float,
            self._state_result_int,
        )

        # Create state
        self._cached_state = BlockSparseRefinementState(
            depth=depth_on_device.clone(),
            intrinsics=intrinsics_on_device.clone(),
            n_valid_depth_pixels=self._valid_depth_count.clone(),
            stride=stride,
            out_W=out_W,
            n_pixels=n_pixels,
            origin=origin.clone(),
            voxel_size=voxel_size,
            block_size=block_size,
            hash_capacity=hash_capacity,
            truncation_distance=truncation_distance,
            warp_data=warp_data,
            grid_W=grid_W_dim,
            grid_H=grid_H_dim,
            grid_D=grid_D,
            best_position=self._state_position.clone(),
            best_quaternion=self._state_quaternion.clone(),
            best_error=self._state_error.clone(),
            best_sum_sq=self._state_sum_sq.clone(),
            best_n_valid=self._state_valid_count.clone(),
            best_JtJ=self._state_JtJ.clone(),
            best_Jtr=self._state_Jtr.clone(),
            lambda_damping=self._state_lambda.clone(),
            result_float=self._state_result_float.clone(),
            result_int=self._state_result_int.clone(),
        )

        return self._cached_state

    @get_profiler_decorator("block_sparse_pose_refiner/_refine_iteration_tiled")
    def _refine_iteration_tiled(
        self, state: BlockSparseRefinementState
    ) -> BlockSparseRefinementState:
        """Perform one trust-region LM iteration.

        Args:
            state: Current refinement state.

        Returns:
            Updated refinement state.
        """
        # Step 1 and 2: Solve for the LM step and compose the candidate pose.
        launch_pose_refiner_solve_candidate(
            state.best_position,
            state.best_quaternion,
            state.best_JtJ,
            state.best_Jtr,
            state.lambda_damping,
            self._candidate_position,
            self._candidate_quaternion,
            self._predicted_reduction,
            state.result_int,
        )

        # Step 3: Evaluate candidate
        n_total_samples = state.n_pixels * self.config.n_samples_per_ray

        self._JtJ_flat.zero_()
        self._Jtr.zero_()
        self._sum_sq_residuals.zero_()
        self._valid_count.zero_()

        _, stream = get_warp_device_stream(state.depth)

        wp.launch(
            kernel=self._tiled_kernel,
            dim=n_total_samples,
            inputs=[
                wp.from_torch(state.depth, dtype=wp.float32),
                wp.from_torch(state.intrinsics, dtype=wp.float32),
                wp.from_torch(self._candidate_position.view(3), dtype=wp.float32),
                wp.from_torch(self._candidate_quaternion.view(4), dtype=wp.float32),
                state.warp_data,
                self.config.minimum_tsdf_weight,
                self.config.depth_minimum_distance,
                self.config.depth_maximum_distance,
                self.config.distance_threshold,
                state.stride,
                state.out_W,
                state.n_pixels,
                wp.from_torch(self._JtJ_flat, dtype=wp.float32),
                wp.from_torch(self._Jtr, dtype=wp.float32),
                wp.from_torch(self._sum_sq_residuals, dtype=wp.float32),
                wp.from_torch(self._valid_count, dtype=wp.int32),
            ],
            block_dim=self.config.tile_block_dim,
            device=wp.device_from_torch(self.device),
            stream=stream,
            adjoint=False,
        )

        # Step 4 and 5: Update the trust region and accepted state in-place.
        launch_pose_refiner_accept_candidate(
            self._candidate_position,
            self._candidate_quaternion,
            self._JtJ_flat,
            self._Jtr,
            self._sum_sq_residuals,
            self._valid_count,
            state.best_position,
            state.best_quaternion,
            state.best_JtJ,
            state.best_Jtr,
            state.best_sum_sq,
            state.best_error,
            state.best_n_valid,
            state.lambda_damping,
            self._predicted_reduction,
            self.config.minimum_valid_depth_pixels,
            self.config.lambda_factor,
            self.config.lambda_min,
            self.config.lambda_max,
            state.result_float,
            state.result_int,
        )
        return state

    @get_profiler_decorator("block_sparse_pose_refiner/_refine_inner_iterations")
    def _refine_inner_iterations(
        self, state: BlockSparseRefinementState
    ) -> BlockSparseRefinementState:
        """Perform inner iterations of refinement.

        Args:
            state: Current refinement state.

        Returns:
            Updated refinement state.
        """
        for i in range(self.config.inner_iterations):
            state = self._refine_iteration_tiled(state)
        return state

    @get_profiler_decorator("block_sparse_pose_refiner/_refine_pose_along_ray")
    def _refine_pose_along_ray(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        estimated_pose: Pose,
    ) -> Tuple[Pose, float, int]:
        """Along-ray SDF alignment for pose refinement.

        Args:
            depth: [H, W] depth image in meters.
            intrinsics: [3, 3] camera intrinsic matrix.
            estimated_pose: Initial pose estimate.

        Returns:
            refined_pose: Refined pose.
            final_error: RMS SDF error.
            n_iterations: Number of iterations performed.
        """
        state = self._setup_refinement(depth, intrinsics, estimated_pose)
        if state is None:
            print(
                f"[BLOCK-SPARSE ALONG-RAY] Too few valid depth pixels "
                f"< {self.config.minimum_valid_depth_pixels}"
            )
            return estimated_pose, float("inf"), 0

        iteration = 0
        for iteration in range(self.config.outer_iterations):
            state = self._refine_inner_iteration_executor(state)

        return (
            Pose(
                position=state.best_position.view(1, 3),
                quaternion=state.best_quaternion.view(1, 4),
            ),
            state.best_error.item(),
            iteration + 1,
        )
