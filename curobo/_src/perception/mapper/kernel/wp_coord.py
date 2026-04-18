# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Common coordinate conversion functions for block-sparse TSDF.

This module provides the single source of truth for all world/voxel/block
coordinate conversions used throughout the mapper kernel code.

All functions use the center-origin convention:
- Origin is at the CENTER of the grid (not a corner)
- voxel_idx = (world_pos - origin) / voxel_size + grid_dims / 2
- world_pos = (voxel_idx - grid_dims / 2) * voxel_size + origin

Block coordinates are derived from voxel coordinates:
- block_idx = floor(voxel_idx / block_size)
- local_idx = voxel_idx - block_idx * block_size
"""

import warp as wp

# =============================================================================
# World <-> Voxel Conversion
# =============================================================================


@wp.func
def world_to_continuous_voxel(
    world_pos: wp.vec3,
    origin: wp.vec3,
    voxel_size: float,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> wp.vec3:
    """Convert world position to continuous voxel coordinates.

    Args:
        world_pos: World position (x, y, z).
        origin: Grid origin (center of the grid).
        voxel_size: Voxel size in meters.
        grid_W: Grid width (nx).
        grid_H: Grid height (ny).
        grid_D: Grid depth (nz).

    Returns:
        Continuous voxel coordinates (vx, vy, vz).
    """
    center_offset_x = wp.float32(grid_W) * 0.5
    center_offset_y = wp.float32(grid_H) * 0.5
    center_offset_z = wp.float32(grid_D) * 0.5

    vx = (world_pos[0] - origin[0]) / voxel_size + center_offset_x
    vy = (world_pos[1] - origin[1]) / voxel_size + center_offset_y
    vz = (world_pos[2] - origin[2]) / voxel_size + center_offset_z
    return wp.vec3(vx, vy, vz)


@wp.func
def voxel_to_world(
    voxel_idx: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> wp.vec3:
    """Convert integer voxel index to world position (voxel center).

    Args:
        voxel_idx: Integer voxel indices (vx, vy, vz).
        origin: Grid origin (center of the grid).
        voxel_size: Voxel size in meters.
        grid_W: Grid width (nx).
        grid_H: Grid height (ny).
        grid_D: Grid depth (nz).

    Returns:
        World position at voxel center.
    """
    center_offset_x = wp.float32(grid_W) * 0.5
    center_offset_y = wp.float32(grid_H) * 0.5
    center_offset_z = wp.float32(grid_D) * 0.5

    # Add 0.5 to get voxel center
    wx = (wp.float32(voxel_idx[0]) + 0.5 - center_offset_x) * voxel_size + origin[0]
    wy = (wp.float32(voxel_idx[1]) + 0.5 - center_offset_y) * voxel_size + origin[1]
    wz = (wp.float32(voxel_idx[2]) + 0.5 - center_offset_z) * voxel_size + origin[2]
    return wp.vec3(wx, wy, wz)


@wp.func
def voxel_to_world_corner(
    voxel_idx: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> wp.vec3:
    """Convert integer voxel index to world position (voxel corner, not center).

    Args:
        voxel_idx: Integer voxel indices (vx, vy, vz).
        origin: Grid origin (center of the grid).
        voxel_size: Voxel size in meters.
        grid_W: Grid width (nx).
        grid_H: Grid height (ny).
        grid_D: Grid depth (nz).

    Returns:
        World position at voxel corner (lower-left).
    """
    center_offset_x = wp.float32(grid_W) * 0.5
    center_offset_y = wp.float32(grid_H) * 0.5
    center_offset_z = wp.float32(grid_D) * 0.5

    wx = (wp.float32(voxel_idx[0]) - center_offset_x) * voxel_size + origin[0]
    wy = (wp.float32(voxel_idx[1]) - center_offset_y) * voxel_size + origin[1]
    wz = (wp.float32(voxel_idx[2]) - center_offset_z) * voxel_size + origin[2]
    return wp.vec3(wx, wy, wz)


# =============================================================================
# World <-> Block Conversion
# =============================================================================


@wp.func
def world_to_block_coords(
    world_pos: wp.vec3,
    origin: wp.vec3,
    voxel_size: float,
    block_size: int,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> wp.vec3i:
    """Convert world position to block coordinates.

    Args:
        world_pos: Position in world frame.
        origin: Grid origin (center of the grid).
        voxel_size: Voxel size in meters.
        block_size: Voxels per block edge (typically 8).
        grid_W, grid_H, grid_D: Grid dimensions.

    Returns:
        Block coordinates (bx, by, bz).
    """
    block_size_f = wp.float32(block_size)

    center_offset_x = wp.float32(grid_W) * 0.5
    center_offset_y = wp.float32(grid_H) * 0.5
    center_offset_z = wp.float32(grid_D) * 0.5

    vx = (world_pos[0] - origin[0]) / voxel_size + center_offset_x
    vy = (world_pos[1] - origin[1]) / voxel_size + center_offset_y
    vz = (world_pos[2] - origin[2]) / voxel_size + center_offset_z

    bx = wp.int32(wp.floor(vx / block_size_f))
    by = wp.int32(wp.floor(vy / block_size_f))
    bz = wp.int32(wp.floor(vz / block_size_f))

    return wp.vec3i(bx, by, bz)


@wp.func
def world_to_block_and_local(
    world_pos: wp.vec3,
    origin: wp.vec3,
    voxel_size: float,
    block_size: int,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> wp.vec4i:
    """Convert world position to block and local voxel coordinates.

    Args:
        world_pos: World position (x, y, z).
        origin: Grid origin (center of the grid).
        voxel_size: Voxel size in meters.
        block_size: Voxels per block edge (typically 8).
        grid_W: Grid width (nx).
        grid_H: Grid height (ny).
        grid_D: Grid depth (nz).

    Returns:
        vec4i with (bx, by, bz, local_idx) where local_idx is linear index [0, 511].
    """
    block_size_f = wp.float32(block_size)

    center_offset_x = wp.float32(grid_W) * 0.5
    center_offset_y = wp.float32(grid_H) * 0.5
    center_offset_z = wp.float32(grid_D) * 0.5

    # Continuous voxel coordinates
    vx = (world_pos[0] - origin[0]) / voxel_size + center_offset_x
    vy = (world_pos[1] - origin[1]) / voxel_size + center_offset_y
    vz = (world_pos[2] - origin[2]) / voxel_size + center_offset_z

    # Block coordinates
    bx = wp.int32(wp.floor(vx / block_size_f))
    by = wp.int32(wp.floor(vy / block_size_f))
    bz = wp.int32(wp.floor(vz / block_size_f))

    # Local coordinates within block
    lx = wp.int32(vx - wp.float32(bx) * block_size_f)
    ly = wp.int32(vy - wp.float32(by) * block_size_f)
    lz = wp.int32(vz - wp.float32(bz) * block_size_f)

    # Clamp to valid range
    lx = wp.clamp(lx, 0, block_size - 1)
    ly = wp.clamp(ly, 0, block_size - 1)
    lz = wp.clamp(lz, 0, block_size - 1)

    # Linear index within block (assumes block_size = 8)
    local_idx = lz * 64 + ly * 8 + lx

    return wp.vec4i(bx, by, bz, local_idx)


@wp.func
def block_local_to_world(
    bx: wp.int32,
    by: wp.int32,
    bz: wp.int32,
    local_idx: wp.int32,
    origin: wp.vec3,
    voxel_size: float,
    block_size: int,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> wp.vec3:
    """Convert block and local index to world position (voxel center).

    Args:
        bx, by, bz: Block coordinates.
        local_idx: Linear local index within block (0-511).
        origin: Grid origin.
        voxel_size: Voxel size in meters.
        block_size: Voxels per block edge (typically 8).
        grid_W, grid_H, grid_D: Grid dimensions.

    Returns:
        World position at voxel center.
    """
    # Convert local linear index to 3D local coords
    lx = local_idx % block_size
    ly = (local_idx // block_size) % block_size
    lz = local_idx // (block_size * block_size)

    # Global voxel index
    vx = bx * block_size + lx
    vy = by * block_size + ly
    vz = bz * block_size + lz

    return voxel_to_world(wp.vec3i(vx, vy, vz), origin, voxel_size, grid_W, grid_H, grid_D)


# =============================================================================
# Local Index Conversion
# =============================================================================


@wp.func
def local_to_linear_index(lx: wp.int32, ly: wp.int32, lz: wp.int32) -> wp.int32:
    """Convert local voxel coordinates to linear index within block.

    Assumes block_size = 8.

    Args:
        lx, ly, lz: Local coordinates [0, 7].

    Returns:
        Linear index [0, 511].
    """
    return lz * 64 + ly * 8 + lx


@wp.func
def linear_to_local_coords(linear_idx: wp.int32, block_size: int) -> wp.vec3i:
    """Convert linear index to local coordinates within block.

    Args:
        linear_idx: Linear index [0, block_size^3).
        block_size: Voxels per block edge.

    Returns:
        Local coordinates (lx, ly, lz).
    """
    lx = linear_idx % block_size
    ly = (linear_idx // block_size) % block_size
    lz = linear_idx // (block_size * block_size)
    return wp.vec3i(lx, ly, lz)
