# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels and helpers for Jump Flooding EDT.

Contains the JFA propagation kernels and ESDF smoothing kernels
used by JumpFloodingEDT.
"""

from typing import Tuple

import numpy as np
import torch
import warp as wp

from curobo._src.perception.mapper.util.utils_quantization import (
    get_weight_from_float16,
    unpack_site_coords_torch,
    unpack_site_x,
    unpack_site_y,
    unpack_site_z,
)

# =============================================================================
# Helper Functions
# =============================================================================


def validate_grid_size(grid_shape: Tuple[int, int, int], class_name: str) -> int:
    """Validate grid fits in int32 and return n_voxels."""
    D, H, W = grid_shape
    n_voxels = D * H * W
    max_int32 = 2**31 - 1
    if n_voxels > max_int32:
        raise ValueError(
            f"Grid too large for int32 site_index: {D}×{H}×{W} = {n_voxels:,} voxels "
            f"(max {max_int32:,}). Reduce grid dimensions or increase voxel_size."
        )
    return n_voxels


def compute_num_passes(grid_shape: Tuple[int, int, int]) -> int:
    """Compute JFA passes needed for exact EDT."""
    D, H, W = grid_shape
    return max(
        int(np.ceil(np.log2(D))) if D > 1 else 0,
        int(np.ceil(np.log2(H))) if H > 1 else 0,
        int(np.ceil(np.log2(W))) if W > 1 else 0,
    )


def get_nearest_surface_coords(
    voxel_idx: torch.Tensor,
    site_index: torch.Tensor,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Get coordinates of nearest surface voxel for given voxel indices.

    Site format: | 2 unused | 10 bits Z | 10 bits Y | 10 bits X |
    Uses bit shifts for ~3x faster coordinate decode vs division.

    Args:
        voxel_idx: Query voxel indices of shape (N, 3) in (z, y, x) order.
        site_index: Packed site index tensor (D, H, W) from EDT computation.
        grid_shape: Grid dimensions (D, H, W). Not used with packed format.

    Returns:
        Surface voxel coordinates (N, 3) in (z, y, x) order.
    """
    D, H, W = grid_shape
    flat_idx = voxel_idx[:, 0] * H * W + voxel_idx[:, 1] * W + voxel_idx[:, 2]
    site_packed = site_index.view(-1)[flat_idx]
    # Unpack using bit operations (faster than division)
    site_x, site_y, site_z = unpack_site_coords_torch(site_packed)
    return torch.stack([site_z, site_y, site_x], dim=1)


# =============================================================================
# Warp Kernels
# =============================================================================


@wp.func
def dist_sq_to_site_packed(i: int, j: int, k: int, site_packed: wp.int32) -> int:
    """Squared distance from voxel (i,j,k) to site (packed 3×10-bit coordinates).

    Args:
        i: X index (slowest dimension, D)
        j: Y index (middle dimension, H)
        k: Z index (fastest dimension, W)
        site_packed: Packed site coords | 2 unused | 10 bits Z | 10 bits Y | 10 bits X |

    Uses bit shifts instead of division for ~3x faster decode.
    """
    if site_packed < wp.int32(0):
        return 2147483647
    # Unpack using bit operations (faster than division)
    sx = int(unpack_site_x(site_packed))
    sy = int(unpack_site_y(site_packed))
    sz = int(unpack_site_z(site_packed))
    # i=X, j=Y, k=Z indices; sx=X, sy=Y, sz=Z site coords
    dx = i - sx
    dy = j - sy
    dz = k - sz
    return dx * dx + dy * dy + dz * dz


@wp.kernel
def jfa_propagate_kernel_18(
    site_in: wp.array(dtype=wp.int32),
    site_out: wp.array(dtype=wp.int32),
    offset: int,
    D: int,
    H: int,
    W: int,
):
    """JFA propagation with 18-connected neighbors (face + edge, no corners).

    Checks 18 neighbors (6 face + 12 edge) at the given offset, skipping the
    8 corner neighbors (±s,±s,±s) which travel the longest distance and
    contribute least. This saves 31% bandwidth vs 26-connected, and the
    suffix passes in the 1+JFA+2 scheme compensate for the minor accuracy loss.

    Supports both double-buffer (site_in != site_out) and
    single-buffer chaotic relaxation (site_in == site_out).

    Site format: | 2 unused | 10 bits Z | 10 bits Y | 10 bits X |
    """
    tid = wp.tid()
    k = tid % W
    temp = tid // W
    j = temp % H
    i = temp // H

    if i >= D:
        return

    stride_z = H * W

    best_site = site_in[tid]
    best_dist = dist_sq_to_site_packed(i, j, k, best_site)

    # Check 18 neighbors at offset distance (6 face + 12 edge, skip 8 corners)
    # di, dj, dk in {-1, 0, 1}, excluding (0,0,0) and corners (|di|+|dj|+|dk|==3)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                # Skip self
                if di == 0 and dj == 0 and dk == 0:
                    continue
                # Skip 8 corners: all three offsets nonzero
                if di != 0 and dj != 0 and dk != 0:
                    continue

                ni = i + di * offset
                nj = j + dj * offset
                nk = k + dk * offset

                # Bounds check
                if ni >= 0 and ni < D and nj >= 0 and nj < H and nk >= 0 and nk < W:
                    n_idx = ni * stride_z + nj * W + nk
                    n_site = site_in[n_idx]
                    d = dist_sq_to_site_packed(i, j, k, n_site)
                    if d < best_dist:
                        best_dist = d
                        best_site = n_site

    site_out[tid] = best_site


@wp.kernel
def jfa_propagate_kernel_26(
    site_in: wp.array(dtype=wp.int32),
    site_out: wp.array(dtype=wp.int32),
    offset: int,
    D: int,
    H: int,
    W: int,
):
    """JFA propagation with 26-connected neighbors (face + edge + corner).

    Checks all 26 neighbors (6 face + 12 edge + 8 corner) at the given offset.
    Higher accuracy than 18-connected at the cost of 44% more memory reads
    per pass (26 vs 18).

    Supports both double-buffer (site_in != site_out) and
    single-buffer chaotic relaxation (site_in == site_out).

    Site format: | 2 unused | 10 bits Z | 10 bits Y | 10 bits X |
    """
    tid = wp.tid()
    k = tid % W
    temp = tid // W
    j = temp % H
    i = temp // H

    if i >= D:
        return

    stride_z = H * W

    best_site = site_in[tid]
    best_dist = dist_sq_to_site_packed(i, j, k, best_site)

    # Check all 26 neighbors at offset distance
    # di, dj, dk in {-1, 0, 1}, excluding (0, 0, 0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                # Skip self
                if di == 0 and dj == 0 and dk == 0:
                    continue

                ni = i + di * offset
                nj = j + dj * offset
                nk = k + dk * offset

                # Bounds check
                if ni >= 0 and ni < D and nj >= 0 and nj < H and nk >= 0 and nk < W:
                    n_idx = ni * stride_z + nj * W + nk
                    n_site = site_in[n_idx]
                    d = dist_sq_to_site_packed(i, j, k, n_site)
                    if d < best_dist:
                        best_dist = d
                        best_site = n_site

    site_out[tid] = best_site


# =============================================================================
# ESDF Smoothing Kernels (Separable 1D Gaussian)
# =============================================================================
# Uses 3 sequential 1D passes instead of one 3D pass.
# Memory reads per voxel: 9 (3 passes × 3 reads) vs 27 (one 3D pass).
# Mathematically equivalent for Gaussian kernels due to separability.


@wp.kernel
def gaussian_smooth_1d_x_kernel(
    input_field: wp.array(dtype=wp.float16),
    output_field: wp.array(dtype=wp.float16),
    D: int,
    H: int,
    W: int,
):
    """1D Gaussian smoothing along X axis (k dimension) with [1, 2, 1] kernel."""
    tid = wp.tid()
    HW = H * W
    i = tid // HW
    remainder = tid - i * HW
    j = remainder // W
    k = remainder - j * W

    if i >= D:
        return

    center_val = float(input_field[tid])

    # Skip unobserved (very large values)
    if center_val > 1e9 or center_val < -1e9:
        output_field[tid] = input_field[tid]
        return

    # [1, 2, 1] kernel along X (k dimension)
    weighted_sum = 2.0 * center_val
    total_weight = 2.0

    # k-1 neighbor
    if k > 0:
        neighbor_idx = i * HW + j * W + (k - 1)
        neighbor_val = float(input_field[neighbor_idx])
        if neighbor_val > -1e9 and neighbor_val < 1e9:
            weighted_sum += neighbor_val
            total_weight += 1.0

    # k+1 neighbor
    if k < W - 1:
        neighbor_idx = i * HW + j * W + (k + 1)
        neighbor_val = float(input_field[neighbor_idx])
        if neighbor_val > -1e9 and neighbor_val < 1e9:
            weighted_sum += neighbor_val
            total_weight += 1.0

    output_field[tid] = wp.float16(weighted_sum / total_weight)


@wp.kernel
def gaussian_smooth_1d_y_kernel(
    input_field: wp.array(dtype=wp.float16),
    output_field: wp.array(dtype=wp.float16),
    D: int,
    H: int,
    W: int,
):
    """1D Gaussian smoothing along Y axis (j dimension) with [1, 2, 1] kernel."""
    tid = wp.tid()
    HW = H * W
    i = tid // HW
    remainder = tid - i * HW
    j = remainder // W
    k = remainder - j * W

    if i >= D:
        return

    center_val = float(input_field[tid])

    # Skip unobserved (very large values)
    if center_val > 1e9 or center_val < -1e9:
        output_field[tid] = input_field[tid]
        return

    # [1, 2, 1] kernel along Y (j dimension)
    weighted_sum = 2.0 * center_val
    total_weight = 2.0

    # j-1 neighbor
    if j > 0:
        neighbor_idx = i * HW + (j - 1) * W + k
        neighbor_val = float(input_field[neighbor_idx])
        if neighbor_val > -1e9 and neighbor_val < 1e9:
            weighted_sum += neighbor_val
            total_weight += 1.0

    # j+1 neighbor
    if j < H - 1:
        neighbor_idx = i * HW + (j + 1) * W + k
        neighbor_val = float(input_field[neighbor_idx])
        if neighbor_val > -1e9 and neighbor_val < 1e9:
            weighted_sum += neighbor_val
            total_weight += 1.0

    output_field[tid] = wp.float16(weighted_sum / total_weight)


@wp.kernel
def gaussian_smooth_1d_z_kernel(
    input_field: wp.array(dtype=wp.float16),
    output_field: wp.array(dtype=wp.float16),
    weight_grid: wp.array(dtype=wp.float16),  # float16: accumulated weight
    minimum_tsdf_weight: float,
    D: int,
    H: int,
    W: int,
):
    """1D Gaussian smoothing along Z axis (i dimension) with [1, 2, 1] kernel.

    This is the final pass, so it applies weight masking to skip unobserved voxels.
    """
    tid = wp.tid()
    HW = H * W
    i = tid // HW
    remainder = tid - i * HW
    j = remainder // W
    k = remainder - j * W

    if i >= D:
        return

    # Check if this voxel is observed (only on final pass)
    # Get weight from float16 grid
    center_weight = get_weight_from_float16(weight_grid[tid])
    if center_weight < minimum_tsdf_weight:
        output_field[tid] = input_field[tid]
        return

    center_val = float(input_field[tid])

    # Skip unobserved (very large values)
    if center_val > 1e9 or center_val < -1e9:
        output_field[tid] = input_field[tid]
        return

    # [1, 2, 1] kernel along Z (i dimension)
    weighted_sum = 2.0 * center_val
    total_weight = 2.0

    # i-1 neighbor
    if i > 0:
        neighbor_idx = (i - 1) * HW + j * W + k
        neighbor_val = float(input_field[neighbor_idx])
        if neighbor_val > -1e9 and neighbor_val < 1e9:
            weighted_sum += neighbor_val
            total_weight += 1.0

    # i+1 neighbor
    if i < D - 1:
        neighbor_idx = (i + 1) * HW + j * W + k
        neighbor_val = float(input_field[neighbor_idx])
        if neighbor_val > -1e9 and neighbor_val < 1e9:
            weighted_sum += neighbor_val
            total_weight += 1.0

    output_field[tid] = wp.float16(weighted_sum / total_weight)


def smooth_esdf(
    dist_field: torch.Tensor,
    weight_grid: torch.Tensor,
    temp_buffer: torch.Tensor,
    minimum_tsdf_weight: float = 0.5,
    iterations: int = 2,
) -> None:
    """Apply separable 1D Gaussian smoothing to ESDF (in-place, CUDA graph safe).

    Uses 3 sequential 1D passes (X -> Y -> Z) instead of one 3D pass.
    This reduces memory reads from 27 to 9 per voxel while producing
    mathematically identical results (Gaussian kernels are separable).

    Note: Iterations must be even so result ends up in dist_field (no copy needed).
    This makes the function CUDA graph safe with no conditional branches.

    Args:
        dist_field: ESDF tensor (D, H, W), float16. Modified in-place.
        weight_grid: Float16 tensor (D, H, W) with accumulated weights.
        temp_buffer: Pre-allocated temp buffer, same shape as dist_field.
        minimum_tsdf_weight: Minimum weight to consider voxel observed.
        iterations: Number of smoothing passes. Rounded up to even if odd.
    """
    if iterations <= 0:
        return

    # Force even iterations so result is always in dist_field (CUDA graph safe)
    if iterations % 2 == 1:
        iterations += 1

    D, H, W = dist_field.shape
    n_voxels = D * H * W
    device_str = str(dist_field.device)

    buf_a_wp = wp.from_torch(dist_field.view(-1), dtype=wp.float16)
    buf_b_wp = wp.from_torch(temp_buffer.view(-1), dtype=wp.float16)
    weight_wp = wp.from_torch(weight_grid.view(-1), dtype=wp.float16)

    for _ in range(iterations):
        # Pass 1: X-axis (buf_a -> buf_b)
        wp.launch(
            gaussian_smooth_1d_x_kernel,
            dim=n_voxels,
            inputs=[buf_a_wp, buf_b_wp, D, H, W],
            device=device_str,
        )

        # Pass 2: Y-axis (buf_b -> buf_a)
        wp.launch(
            gaussian_smooth_1d_y_kernel,
            dim=n_voxels,
            inputs=[buf_b_wp, buf_a_wp, D, H, W],
            device=device_str,
        )

        # Pass 3: Z-axis with weight masking (buf_a -> buf_b)
        wp.launch(
            gaussian_smooth_1d_z_kernel,
            dim=n_voxels,
            inputs=[buf_a_wp, buf_b_wp, weight_wp, minimum_tsdf_weight, D, H, W],
            device=device_str,
        )

        # Swap for next iteration (result is in buf_b)
        buf_a_wp, buf_b_wp = buf_b_wp, buf_a_wp

    # With even iterations, result is in dist_field (buf_a). No copy needed.
