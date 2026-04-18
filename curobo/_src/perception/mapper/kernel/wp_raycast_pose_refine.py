# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF pose refinement kernels.

Implements tiled kernel for ray-SDF alignment using block-sparse TSDF storage.
Uses Warp tile operations for efficient block-level JtJ/Jtr accumulation.

Key differences from dense version:
- Uses hash table lookups instead of direct array indexing
- Corner-origin convention (no grid center offset)
- No fixed grid bounds (hash returns -1 for missing blocks)
"""

from typing import Callable

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_raycast_common import (
    compute_gradient,
    sample_tsdf_trilinear,
)

# =============================================================================
# Helper Functions
# =============================================================================


@wp.func
def quat_from_wxyz_array(q: wp.array(dtype=wp.float32)) -> wp.quat:
    """Convert wxyz quaternion array to warp quaternion (xyzw internal)."""
    return wp.quat(q[1], q[2], q[3], q[0])


@wp.func
def vec3_from_array(v: wp.array(dtype=wp.float32)) -> wp.vec3:
    """Convert float array to vec3."""
    return wp.vec3(v[0], v[1], v[2])


# =============================================================================
# Tiled Kernel Factory
# =============================================================================


def create_ray_sdf_alignment_block_sparse_tiled_kernel(n_samples_per_ray: int = 5) -> Callable:
    """Create a tiled kernel for ray-SDF alignment with block-sparse TSDF.

    This kernel computes JtJ and Jtr directly using block reduction, avoiding
    storage of the full Jacobian matrix. Each block:
    1. Each thread computes one sample's Jacobian (1x6) and residual
    2. Block-level reduction via wp.tile_sum() accumulates JtJ (6x6) and Jtr (6)
    3. One atomic add per block to global accumulators

    Args:
        n_samples_per_ray: Number of SDF samples per ray (e.g., 5).

    Returns:
        Compiled Warp kernel for use with wp.launch(..., block_dim=TILE_SIZE).
    """
    N_SAMPLES = wp.constant(n_samples_per_ray)
    HALF_SAMPLES = wp.constant(n_samples_per_ray // 2)

    def _ray_sdf_alignment_block_sparse_tiled_kernel(
        # Inputs
        depth: wp.array2d(dtype=wp.float32),
        intrinsics: wp.array2d(dtype=wp.float32),
        cam_position: wp.array(dtype=wp.float32),
        cam_quaternion: wp.array(dtype=wp.float32),
        # Block-sparse TSDF (struct)
        tsdf: BlockSparseTSDFWarp,
        # Ray marching parameters
        minimum_tsdf_weight: float,
        depth_minimum_distance: float,
        depth_maximum_distance: float,
        distance_threshold: float,
        # Sampling
        stride: int,
        out_W: int,
        n_pixels: int,
        # Outputs: 1D global accumulators (zero-initialized before launch)
        JtJ_out: wp.array(dtype=wp.float32),  # [36] flattened 6x6
        Jtr_out: wp.array(dtype=wp.float32),  # [6]
        sum_sq_residuals: wp.array(dtype=wp.float32),  # [1]
        valid_count: wp.array(dtype=wp.int32),  # [1]
    ):
        """Tiled kernel for ray-SDF alignment with block-sparse TSDF.

        Launch with dim = n_pixels * n_samples_per_ray, block_dim = 256.
        Each thread handles exactly one sample.
        """
        # Global sample index
        sample_idx = wp.tid()
        n_total_samples = n_pixels * N_SAMPLES

        # Initialize per-thread values (invalid samples contribute zeros)
        j0 = float(0.0)
        j1 = float(0.0)
        j2 = float(0.0)
        j3 = float(0.0)
        j4 = float(0.0)
        j5 = float(0.0)
        r = float(0.0)
        valid = wp.int32(0)

        if sample_idx < n_total_samples:
            # Map sample to pixel and k-offset
            pixel_idx = sample_idx // N_SAMPLES
            k_idx = sample_idx % N_SAMPLES
            k = k_idx - HALF_SAMPLES

            # Compute pixel coordinates
            out_i = pixel_idx // out_W
            out_j = pixel_idx % out_W
            pi = out_i * stride
            pj = out_j * stride

            H = depth.shape[0]
            W = depth.shape[1]

            if pi < H and pj < W:
                d = depth[pi, pj]
                if d >= depth_minimum_distance and d <= depth_maximum_distance and wp.isfinite(d):
                    # Camera pose
                    cam_pos = vec3_from_array(cam_position)
                    cam_quat = quat_from_wxyz_array(cam_quaternion)

                    # Intrinsics
                    fx = intrinsics[0, 0]
                    fy = intrinsics[1, 1]
                    cx = intrinsics[0, 2]
                    cy = intrinsics[1, 2]

                    # Ray direction
                    ray_cam_x = (float(pj) - cx) / fx
                    ray_cam_y = (float(pi) - cy) / fy
                    ray_cam_z = 1.0
                    ray_len = wp.sqrt(
                        ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z
                    )
                    ray_cam = wp.vec3(
                        ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len
                    )
                    ray_world = wp.quat_rotate(cam_quat, ray_cam)
                    obs_t = d * ray_len

                    # Check surface sample (k=0) for early rejection
                    p_surface = cam_pos + ray_world * obs_t
                    sdf_k0 = sample_tsdf_trilinear(tsdf, p_surface, minimum_tsdf_weight)

                    if sdf_k0[1] >= 0.5 and wp.abs(sdf_k0[0]) <= distance_threshold:
                        # Ray is valid - compute this sample
                        sample_t = obs_t + float(k) * tsdf.voxel_size

                        if sample_t >= 0.1:
                            p_sample = cam_pos + ray_world * sample_t
                            sdf_result = sample_tsdf_trilinear(tsdf, p_sample, minimum_tsdf_weight)
                            sdf_actual = sdf_result[0]
                            sdf_valid = sdf_result[1]
                            sdf_expected = float(-k) * tsdf.voxel_size

                            if sdf_valid >= 0.5 and wp.abs(sdf_expected) <= tsdf.truncation_distance:
                                r = sdf_actual - sdf_expected

                                # Distance-dependent Huber weighting for robustness
                                abs_r = wp.abs(r)
                                huber_base = 0.05  # base threshold at 1m
                                depth_scale = sample_t * sample_t
                                huber_delta = huber_base * depth_scale
                                huber_scale = float(1.0)
                                if abs_r > huber_delta:
                                    huber_scale = wp.sqrt(huber_delta / abs_r)
                                r = r * huber_scale

                                # Gradient using combined sampling
                                grad = compute_gradient(tsdf, p_sample, minimum_tsdf_weight)
                                gx = grad[0]
                                gy = grad[1]
                                gz = grad[2]

                                px = p_sample[0]
                                py = p_sample[1]
                                pz = p_sample[2]

                                # 6-DOF Jacobian with Huber scaling
                                # Negated because SDF gradient points away from surface
                                # (increasing SDF), but we want to move toward surface
                                # (decreasing residual).
                                j0 = gx * huber_scale
                                j1 = gy * huber_scale
                                j2 = gz * huber_scale
                                j3 = (gz * py - gy * pz) * huber_scale
                                j4 = (gx * pz - gz * px) * huber_scale
                                j5 = (gy * px - gx * py) * huber_scale
                                valid = wp.int32(1)

        # =====================================================================
        # Block reduction using tile operations
        # =====================================================================

        # Create tiles from per-thread scalar values
        t_j0 = wp.tile(j0)
        t_j1 = wp.tile(j1)
        t_j2 = wp.tile(j2)
        t_j3 = wp.tile(j3)
        t_j4 = wp.tile(j4)
        t_j5 = wp.tile(j5)
        t_r = wp.tile(r)
        t_valid = wp.tile(valid)

        # Reduce scalars: sum(valid) first for early exit check
        sum_valid = wp.tile_sum(t_valid)

        # Skip all atomic operations if block has no valid samples
        if sum_valid[0] > 0.0:
            # Reduce sum(r^2)
            sum_r_sq = wp.tile_sum(wp.tile_map(wp.mul, t_r, t_r))

            # Atomic add to global accumulators
            wp.tile_atomic_add(sum_sq_residuals, sum_r_sq, 0, False)
            wp.tile_atomic_add(valid_count, sum_valid, 0, False)

            # Reduce J^T @ r (6 elements)
            Jtr_0 = wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_r))
            Jtr_1 = wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_r))
            Jtr_2 = wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_r))
            Jtr_3 = wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_r))
            Jtr_4 = wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_r))
            Jtr_5 = wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_r))

            wp.tile_atomic_add(Jtr_out, Jtr_0, 0, False)
            wp.tile_atomic_add(Jtr_out, Jtr_1, 1, False)
            wp.tile_atomic_add(Jtr_out, Jtr_2, 2, False)
            wp.tile_atomic_add(Jtr_out, Jtr_3, 3, False)
            wp.tile_atomic_add(Jtr_out, Jtr_4, 4, False)
            wp.tile_atomic_add(Jtr_out, Jtr_5, 5, False)

            # Reduce J^T @ J (6x6 flattened to 36)
            # Row 0
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j0)), 0, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j1)), 1, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j2)), 2, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j3)), 3, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j4)), 4, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j0, t_j5)), 5, False)
            # Row 1
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j0)), 6, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j1)), 7, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j2)), 8, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j3)), 9, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j4)), 10, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j1, t_j5)), 11, False)
            # Row 2
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j0)), 12, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j1)), 13, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j2)), 14, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j3)), 15, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j4)), 16, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j2, t_j5)), 17, False)
            # Row 3
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j0)), 18, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j1)), 19, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j2)), 20, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j3)), 21, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j4)), 22, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j3, t_j5)), 23, False)
            # Row 4
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j0)), 24, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j1)), 25, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j2)), 26, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j3)), 27, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j4)), 28, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j4, t_j5)), 29, False)
            # Row 5
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j0)), 30, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j1)), 31, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j2)), 32, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j3)), 33, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j4)), 34, False)
            wp.tile_atomic_add(JtJ_out, wp.tile_sum(wp.tile_map(wp.mul, t_j5, t_j5)), 35, False)

    # Generate unique name for this configuration
    kernel_name = f"ray_sdf_alignment_block_sparse_tiled_{n_samples_per_ray}"
    _ray_sdf_alignment_block_sparse_tiled_kernel.__name__ = kernel_name
    _ray_sdf_alignment_block_sparse_tiled_kernel.__qualname__ = kernel_name

    return wp.kernel(enable_backward=False, module="unique")(
        _ray_sdf_alignment_block_sparse_tiled_kernel
    )

