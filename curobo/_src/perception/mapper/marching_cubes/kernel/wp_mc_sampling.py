# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDF sampling utilities for marching cubes.

Provides Warp functions for sampling, interpolation, gradient estimation,
and vertex refinement on signed distance fields.

Uses float16 weight grid for weight checking.
Computes SDF on-the-fly from sdf_weight_grid / weight_grid for memory efficiency.
"""

import warp as wp

# Large value for unobserved voxels (outside surface)
UNOBSERVED_SDF = wp.constant(1000.0)


@wp.func
def sample_sdf_with_weight(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),  # float16: accumulated weight
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
) -> wp.float32:
    """Sample SDF at voxel, computing on-the-fly from weighted grids.

    Computes sdf = sdf_weight / weight, returning large value if weight is below threshold.
    """
    w = float(weight_grid[i, j, k])
    if w < minimum_tsdf_weight:
        return UNOBSERVED_SDF
    sdf_w = float(sdf_weight_grid[i, j, k])
    return wp.float32(sdf_w / w) - level


@wp.func
def trilinear_sample_sdf_weighted(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),  # float16: accumulated weight
    pos: wp.vec3,
    origin: wp.vec3,
    voxel_size: wp.float32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
) -> wp.float32:
    """Sample SDF with weight masking using trilinear interpolation.

    Computes SDF on-the-fly from sdf_weight_grid / weight_grid.

    Args:
        sdf_weight_grid: 3D accumulated sdf*weight array (z, y, x ordering), float16.
        weight_grid: Float16 weight grid.
        pos: World position to sample.
        origin: World origin of the grid.
        voxel_size: Size of each voxel.
        minimum_tsdf_weight: Minimum weight threshold.
        level: Isosurface level.

    Returns:
        Interpolated SDF value at position (relative to level).
        Returns large value if any corner is unobserved.
    """
    D = weight_grid.shape[0]
    H = weight_grid.shape[1]
    W = weight_grid.shape[2]

    # Convert world position to continuous voxel coordinates (origin is at center/half-extents)
    voxel_x = (pos[0] - origin[0]) / voxel_size + wp.float32(W) * 0.5
    voxel_y = (pos[1] - origin[1]) / voxel_size + wp.float32(H) * 0.5
    voxel_z = (pos[2] - origin[2]) / voxel_size + wp.float32(D) * 0.5

    # Get integer voxel indices (floor)
    k0 = wp.int32(wp.floor(voxel_x))
    j0 = wp.int32(wp.floor(voxel_y))
    i0 = wp.int32(wp.floor(voxel_z))

    # Clamp to valid range
    k0 = wp.clamp(k0, 0, W - 2)
    j0 = wp.clamp(j0, 0, H - 2)
    i0 = wp.clamp(i0, 0, D - 2)

    k1 = k0 + 1
    j1 = j0 + 1
    i1 = i0 + 1

    # Fractional part for interpolation
    fx = wp.clamp(voxel_x - wp.float32(k0), 0.0, 1.0)
    fy = wp.clamp(voxel_y - wp.float32(j0), 0.0, 1.0)
    fz = wp.clamp(voxel_z - wp.float32(i0), 0.0, 1.0)

    # Sample 8 corners with weight check
    c000 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i0, j0, k0, minimum_tsdf_weight, level)
    c100 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i1, j0, k0, minimum_tsdf_weight, level)
    c010 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i0, j1, k0, minimum_tsdf_weight, level)
    c110 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i1, j1, k0, minimum_tsdf_weight, level)
    c001 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i0, j0, k1, minimum_tsdf_weight, level)
    c101 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i1, j0, k1, minimum_tsdf_weight, level)
    c011 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i0, j1, k1, minimum_tsdf_weight, level)
    c111 = sample_sdf_with_weight(sdf_weight_grid, weight_grid, i1, j1, k1, minimum_tsdf_weight, level)

    # Trilinear interpolation
    c00 = c000 * (1.0 - fx) + c001 * fx
    c10 = c100 * (1.0 - fx) + c101 * fx
    c01 = c010 * (1.0 - fx) + c011 * fx
    c11 = c110 * (1.0 - fx) + c111 * fx

    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def estimate_sdf_gradient_weighted(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),  # float16: accumulated weight
    pos: wp.vec3,
    origin: wp.vec3,
    voxel_size: wp.float32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
) -> wp.vec3:
    """Estimate SDF gradient using central finite differences with weight masking."""
    h = voxel_size * 0.5

    dx_pos = trilinear_sample_sdf_weighted(
        sdf_weight_grid, weight_grid, pos + wp.vec3(h, 0.0, 0.0), origin, voxel_size, minimum_tsdf_weight, level
    )
    dx_neg = trilinear_sample_sdf_weighted(
        sdf_weight_grid, weight_grid, pos - wp.vec3(h, 0.0, 0.0), origin, voxel_size, minimum_tsdf_weight, level
    )

    dy_pos = trilinear_sample_sdf_weighted(
        sdf_weight_grid, weight_grid, pos + wp.vec3(0.0, h, 0.0), origin, voxel_size, minimum_tsdf_weight, level
    )
    dy_neg = trilinear_sample_sdf_weighted(
        sdf_weight_grid, weight_grid, pos - wp.vec3(0.0, h, 0.0), origin, voxel_size, minimum_tsdf_weight, level
    )

    dz_pos = trilinear_sample_sdf_weighted(
        sdf_weight_grid, weight_grid, pos + wp.vec3(0.0, 0.0, h), origin, voxel_size, minimum_tsdf_weight, level
    )
    dz_neg = trilinear_sample_sdf_weighted(
        sdf_weight_grid, weight_grid, pos - wp.vec3(0.0, 0.0, h), origin, voxel_size, minimum_tsdf_weight, level
    )

    return wp.vec3(
        (dx_pos - dx_neg) / (2.0 * h),
        (dy_pos - dy_neg) / (2.0 * h),
        (dz_pos - dz_neg) / (2.0 * h),
    )


@wp.func
def refine_vertex_weighted(
    sdf_weight_grid: wp.array3d(dtype=wp.float16),
    weight_grid: wp.array3d(dtype=wp.float16),  # float16: accumulated weight
    vertex: wp.vec3,
    origin: wp.vec3,
    voxel_size: wp.float32,
    minimum_tsdf_weight: wp.float32,
    level: wp.float32,
    iterations: wp.int32,
) -> wp.vec3:
    """Refine vertex position to true SDF zero-crossing using Newton-Raphson.

    Computes SDF on-the-fly from sdf_weight_grid / weight_grid.

    Args:
        sdf_weight_grid: 3D accumulated sdf*weight array (float16).
        weight_grid: Float16 weight grid.
        vertex: Initial vertex position.
        origin: World origin of the grid.
        voxel_size: Size of each voxel.
        minimum_tsdf_weight: Minimum weight threshold.
        level: Isosurface level.
        iterations: Number of refinement iterations.

    Returns:
        Refined vertex position.
    """
    pos = vertex

    for _ in range(iterations):
        sdf_val = trilinear_sample_sdf_weighted(
            sdf_weight_grid, weight_grid, pos, origin, voxel_size, minimum_tsdf_weight, level
        )

        # If at zero or in unobserved region, stop
        if wp.abs(sdf_val) < 1e-6 or sdf_val > 100.0:
            break

        grad = estimate_sdf_gradient_weighted(
            sdf_weight_grid, weight_grid, pos, origin, voxel_size, minimum_tsdf_weight, level
        )
        grad_mag = wp.sqrt(wp.dot(grad, grad))

        if grad_mag < 1e-4:
            break

        step_size = wp.clamp(sdf_val / grad_mag, -voxel_size * 0.5, voxel_size * 0.5)
        pos = pos - step_size * (grad / grad_mag)

    return pos

