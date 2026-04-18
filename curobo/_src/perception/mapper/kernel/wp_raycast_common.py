# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF sampling functions for raycasting.

This module provides device-side functions for sampling SDF and color values
from block-sparse TSDF storage. All functions use the struct-based API with
BlockSparseTSDFWarp for clean and consistent interfaces.

Key functions:
- sample_voxel: Core lowest-level combined SDF sampling at pool_idx/local_idx
- sample_tsdf: Nearest-neighbor SDF sampling at world position
- sample_tsdf_trilinear: Trilinear interpolation across block boundaries
- sample_rgb: RGB color sampling (per-block average)
- compute_gradient: Surface normal from SDF gradient

Call hierarchy:
    sample_tsdf() → sample_voxel()
    sample_tsdf_trilinear() → _sample_voxel_at_block_local() → sample_voxel()
    compute_gradient() → sample_tsdf()
"""

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_coord import (
    world_to_block_and_local,
    world_to_continuous_voxel,
)
from curobo._src.perception.mapper.kernel.wp_hash import (
    compute_avg_rgb_from_block,
    hash_lookup,
)

# Large value representing "no obstacle" or "unobserved"
_SDF_INFINITY = wp.constant(wp.float32(1e10))


# =============================================================================
# Struct-Based Combined Sampling API
# =============================================================================


@wp.func
def sample_voxel(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
    minimum_tsdf_weight: float,
) -> wp.vec2:
    """Sample combined SDF at a specific voxel (lowest-level).

    This is the DRY foundation for all combined SDF sampling. All other
    sampling functions should call this.

    Args:
        tsdf: BlockSparseTSDFWarp struct containing all TSDF data.
        pool_idx: Block pool index from hash lookup.
        local_idx: Local voxel index (0-511).
        minimum_tsdf_weight: Minimum weight for valid dynamic voxel.

    Returns:
        vec2(sdf, valid) where valid=1.0 if observed, 0.0 otherwise.
    """
    dynamic_sdf = _SDF_INFINITY
    static_sdf = _SDF_INFINITY
    has_valid = False

    if tsdf.has_dynamic:
        sdf_w = wp.float32(tsdf.block_data[pool_idx, local_idx, 0])
        w = wp.float32(tsdf.block_data[pool_idx, local_idx, 1])
        if w >= minimum_tsdf_weight:
            dynamic_sdf = sdf_w / w
            has_valid = True

    if tsdf.has_static:
        static_sdf = wp.float32(tsdf.static_block_data[pool_idx, local_idx])
        if static_sdf < 1e9:
            has_valid = True

    if not has_valid:
        return wp.vec2(_SDF_INFINITY, 0.0)

    return wp.vec2(wp.min(dynamic_sdf, static_sdf), 1.0)


@wp.func
def sample_tsdf(
    tsdf: BlockSparseTSDFWarp,
    world_pos: wp.vec3,
    minimum_tsdf_weight: float,
) -> wp.vec2:
    """Sample combined SDF from TSDF struct (nearest-neighbor).

    Args:
        tsdf: BlockSparseTSDFWarp struct containing all TSDF data.
        world_pos: World position to sample.
        minimum_tsdf_weight: Minimum weight for valid dynamic voxel.

    Returns:
        vec2(sdf, valid) where valid=1.0 if observed, 0.0 otherwise.
    """
    # Convert world position to block and local coordinates using common function
    coords = world_to_block_and_local(
        world_pos, tsdf.origin, tsdf.voxel_size, tsdf.block_size,
        tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
    )
    bx = coords[0]
    by = coords[1]
    bz = coords[2]
    local_idx = coords[3]

    # Hash lookup
    pool_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
    if pool_idx < 0:
        return wp.vec2(_SDF_INFINITY, 0.0)

    return sample_voxel(tsdf, pool_idx, local_idx, minimum_tsdf_weight)


@wp.func
def _sample_voxel_at_block_local(
    tsdf: BlockSparseTSDFWarp,
    bx: int,
    by: int,
    bz: int,
    lx: int,
    ly: int,
    lz: int,
    minimum_tsdf_weight: float,
) -> wp.vec2:
    """Sample combined SDF at specific block/local coords with boundary handling.

    Handles block boundary crossings and delegates to sample_voxel() for actual sampling.
    Used by trilinear interpolation to sample the 8 corner voxels.
    """
    actual_bx = bx
    actual_by = by
    actual_bz = bz
    actual_lx = lx
    actual_ly = ly
    actual_lz = lz

    if lx < 0:
        actual_bx = bx - 1
        actual_lx = lx + tsdf.block_size
    elif lx >= tsdf.block_size:
        actual_bx = bx + 1
        actual_lx = 0
    if ly < 0:
        actual_by = by - 1
        actual_ly = ly + tsdf.block_size
    elif ly >= tsdf.block_size:
        actual_by = by + 1
        actual_ly = 0
    if lz < 0:
        actual_bz = bz - 1
        actual_lz = lz + tsdf.block_size
    elif lz >= tsdf.block_size:
        actual_bz = bz + 1
        actual_lz = 0

    pool_idx = hash_lookup(tsdf.hash_table, actual_bx, actual_by, actual_bz, tsdf.hash_capacity)
    if pool_idx < 0:
        return wp.vec2(_SDF_INFINITY, 0.0)

    local_idx = actual_lz * 64 + actual_ly * 8 + actual_lx
    return sample_voxel(tsdf, pool_idx, local_idx, minimum_tsdf_weight)


@wp.func
def sample_tsdf_trilinear(
    tsdf: BlockSparseTSDFWarp,
    world_pos: wp.vec3,
    minimum_tsdf_weight: float,
) -> wp.vec2:
    """Sample combined SDF with trilinear interpolation.

    Args:
        tsdf: BlockSparseTSDFWarp struct containing all TSDF data.
        world_pos: World position to sample.
        minimum_tsdf_weight: Minimum weight for valid dynamic voxel.

    Returns:
        vec2(sdf, valid) where valid=1.0 if interpolation succeeded.
    """
    block_size_f = wp.float32(tsdf.block_size)

    # Continuous voxel coordinates using common function
    voxel_f = world_to_continuous_voxel(
        world_pos, tsdf.origin, tsdf.voxel_size,
        tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
    )
    vx = voxel_f[0]
    vy = voxel_f[1]
    vz = voxel_f[2]

    # Block coordinates
    bx = wp.int32(wp.floor(vx / block_size_f))
    by = wp.int32(wp.floor(vy / block_size_f))
    bz = wp.int32(wp.floor(vz / block_size_f))

    # Local float coordinates for interpolation
    lxf = vx - wp.float32(bx) * block_size_f - 0.5
    lyf = vy - wp.float32(by) * block_size_f - 0.5
    lzf = vz - wp.float32(bz) * block_size_f - 0.5

    lx0 = wp.int32(wp.floor(lxf))
    ly0 = wp.int32(wp.floor(lyf))
    lz0 = wp.int32(wp.floor(lzf))

    tx = lxf - wp.float32(lx0)
    ty = lyf - wp.float32(ly0)
    tz = lzf - wp.float32(lz0)

    # Sample 8 corners
    c000 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0, ly0, lz0, minimum_tsdf_weight)
    c001 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0 + 1, ly0, lz0, minimum_tsdf_weight)
    c010 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0, ly0 + 1, lz0, minimum_tsdf_weight)
    c011 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0 + 1, ly0 + 1, lz0, minimum_tsdf_weight)
    c100 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0, ly0, lz0 + 1, minimum_tsdf_weight)
    c101 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0 + 1, ly0, lz0 + 1, minimum_tsdf_weight)
    c110 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0, ly0 + 1, lz0 + 1, minimum_tsdf_weight)
    c111 = _sample_voxel_at_block_local(tsdf, bx, by, bz, lx0 + 1, ly0 + 1, lz0 + 1, minimum_tsdf_weight)

    # Trilinear interpolation weights
    w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
    w001 = tx * (1.0 - ty) * (1.0 - tz)
    w010 = (1.0 - tx) * ty * (1.0 - tz)
    w011 = tx * ty * (1.0 - tz)
    w100 = (1.0 - tx) * (1.0 - ty) * tz
    w101 = tx * (1.0 - ty) * tz
    w110 = (1.0 - tx) * ty * tz
    w111 = tx * ty * tz

    # Check if all corners are invalid
    n_valid = c000[1] + c001[1] + c010[1] + c011[1] + c100[1] + c101[1] + c110[1] + c111[1]
    if n_valid < 0.5:
        return wp.vec2(_SDF_INFINITY, 0.0)

    # Replace invalid corners with +truncation so they are treated as exterior.
    # This avoids partial-weight renormalization artifacts at observation boundaries
    # where dropping invalid corners shifts the zero-crossing outward.
    fallback = tsdf.truncation_distance
    s000 = wp.where(c000[1] > 0.5, c000[0], fallback)
    s001 = wp.where(c001[1] > 0.5, c001[0], fallback)
    s010 = wp.where(c010[1] > 0.5, c010[0], fallback)
    s011 = wp.where(c011[1] > 0.5, c011[0], fallback)
    s100 = wp.where(c100[1] > 0.5, c100[0], fallback)
    s101 = wp.where(c101[1] > 0.5, c101[0], fallback)
    s110 = wp.where(c110[1] > 0.5, c110[0], fallback)
    s111 = wp.where(c111[1] > 0.5, c111[0], fallback)

    sdf = (
        w000 * s000 + w001 * s001 + w010 * s010 + w011 * s011 +
        w100 * s100 + w101 * s101 + w110 * s110 + w111 * s111
    )

    return wp.vec2(sdf, 1.0)


@wp.func
def sample_rgb(
    tsdf: BlockSparseTSDFWarp,
    world_pos: wp.vec3,
) -> wp.vec3:
    """Sample RGB color from TSDF struct (per-block average).

    Args:
        tsdf: BlockSparseTSDFWarp struct.
        world_pos: World position to sample.

    Returns:
        vec3 with RGB values [0, 255] or (0, 0, 0) if unobserved.
    """
    # Convert world position to block coordinates using common function
    coords = world_to_block_and_local(
        world_pos, tsdf.origin, tsdf.voxel_size, tsdf.block_size,
        tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
    )
    bx = coords[0]
    by = coords[1]
    bz = coords[2]

    pool_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
    if pool_idx < 0:
        return wp.vec3(0.0, 0.0, 0.0)

    return compute_avg_rgb_from_block(tsdf.block_rgb, pool_idx)


@wp.func
def compute_gradient(
    tsdf: BlockSparseTSDFWarp,
    world_pos: wp.vec3,
    minimum_tsdf_weight: float,
) -> wp.vec3:
    """Compute surface normal from combined SDF gradient using trilinear interpolation.

    Uses trilinear interpolation for smooth gradients, producing smooth normals
    even at voxel boundaries.

    Args:
        tsdf: BlockSparseTSDFWarp struct.
        world_pos: World position.
        minimum_tsdf_weight: Minimum weight for valid dynamic voxel.

    Returns:
        Normalized gradient vector (surface normal pointing outward).
    """
    eps = tsdf.voxel_size

    # Use trilinear interpolation for smooth gradients
    sdf_xp = sample_tsdf_trilinear(tsdf, wp.vec3(world_pos[0] + eps, world_pos[1], world_pos[2]), minimum_tsdf_weight)[0]
    sdf_xm = sample_tsdf_trilinear(tsdf, wp.vec3(world_pos[0] - eps, world_pos[1], world_pos[2]), minimum_tsdf_weight)[0]
    sdf_yp = sample_tsdf_trilinear(tsdf, wp.vec3(world_pos[0], world_pos[1] + eps, world_pos[2]), minimum_tsdf_weight)[0]
    sdf_ym = sample_tsdf_trilinear(tsdf, wp.vec3(world_pos[0], world_pos[1] - eps, world_pos[2]), minimum_tsdf_weight)[0]
    sdf_zp = sample_tsdf_trilinear(tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] + eps), minimum_tsdf_weight)[0]
    sdf_zm = sample_tsdf_trilinear(tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] - eps), minimum_tsdf_weight)[0]

    if sdf_xp > 1e9 or sdf_xm > 1e9 or sdf_yp > 1e9 or sdf_ym > 1e9 or sdf_zp > 1e9 or sdf_zm > 1e9:
        return wp.vec3(0.0, 0.0, 1.0)

    grad_x = (sdf_xp - sdf_xm) / (2.0 * eps)
    grad_y = (sdf_yp - sdf_ym) / (2.0 * eps)
    grad_z = (sdf_zp - sdf_zm) / (2.0 * eps)

    grad_mag = wp.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
    if grad_mag < 1e-6:
        return wp.vec3(0.0, 0.0, 1.0)

    return wp.vec3(grad_x / grad_mag, grad_y / grad_mag, grad_z / grad_mag)


@wp.func
def compute_gradient_nearest(
    tsdf: BlockSparseTSDFWarp,
    world_pos: wp.vec3,
    minimum_tsdf_weight: float,
) -> wp.vec3:
    """Compute surface normal from SDF gradient using nearest-neighbor sampling.

    Uses nearest-neighbor sampling for faster gradient computation. This is
    suitable for mesh extraction where smooth normals are less critical and
    compilation speed is more important.

    Args:
        tsdf: BlockSparseTSDFWarp struct.
        world_pos: World position.
        minimum_tsdf_weight: Minimum weight for valid dynamic voxel.

    Returns:
        Normalized gradient vector (surface normal pointing outward).
    """
    eps = tsdf.voxel_size

    # Use nearest-neighbor sampling for faster gradients (6 lookups vs 48)
    sdf_xp = sample_tsdf(tsdf, wp.vec3(world_pos[0] + eps, world_pos[1], world_pos[2]), minimum_tsdf_weight)[0]
    sdf_xm = sample_tsdf(tsdf, wp.vec3(world_pos[0] - eps, world_pos[1], world_pos[2]), minimum_tsdf_weight)[0]
    sdf_yp = sample_tsdf(tsdf, wp.vec3(world_pos[0], world_pos[1] + eps, world_pos[2]), minimum_tsdf_weight)[0]
    sdf_ym = sample_tsdf(tsdf, wp.vec3(world_pos[0], world_pos[1] - eps, world_pos[2]), minimum_tsdf_weight)[0]
    sdf_zp = sample_tsdf(tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] + eps), minimum_tsdf_weight)[0]
    sdf_zm = sample_tsdf(tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] - eps), minimum_tsdf_weight)[0]

    if sdf_xp > 1e9 or sdf_xm > 1e9 or sdf_yp > 1e9 or sdf_ym > 1e9 or sdf_zp > 1e9 or sdf_zm > 1e9:
        return wp.vec3(0.0, 0.0, 1.0)

    grad_x = (sdf_xp - sdf_xm) / (2.0 * eps)
    grad_y = (sdf_yp - sdf_ym) / (2.0 * eps)
    grad_z = (sdf_zp - sdf_zm) / (2.0 * eps)

    grad_mag = wp.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
    if grad_mag < 1e-6:
        return wp.vec3(0.0, 0.0, 1.0)

    return wp.vec3(grad_x / grad_mag, grad_y / grad_mag, grad_z / grad_mag)

