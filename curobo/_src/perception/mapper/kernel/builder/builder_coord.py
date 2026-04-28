# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""World, voxel, and block coordinate conversions for one block size.

All functions here use the center-origin convention: ``origin`` is the
center of the bounded voxel grid. Block-key coordinates are centered around
the grid's block midpoint, while block-grid coordinates are zero-based.
"""

from __future__ import annotations

import warp as wp

from curobo._src.util.warp import warp_func


def make_coord_kernels(block_size: int) -> dict[str, object]:
    """Build coordinate-conversion Warp functions."""
    BLOCK_SIZE = wp.constant(block_size)

    # =====================================================================
    # World <-> Voxel Conversion
    # =====================================================================

    @warp_func(f"world_to_continuous_voxel_bs{block_size}")
    def world_to_continuous_voxel(
        world_pos: wp.vec3,
        origin: wp.vec3,
        voxel_size: float,
        grid_W: int,
        grid_H: int,
        grid_D: int,
    ) -> wp.vec3:
        center_offset_x = wp.float32(grid_W) * 0.5
        center_offset_y = wp.float32(grid_H) * 0.5
        center_offset_z = wp.float32(grid_D) * 0.5
        vx = (world_pos[0] - origin[0]) / voxel_size + center_offset_x
        vy = (world_pos[1] - origin[1]) / voxel_size + center_offset_y
        vz = (world_pos[2] - origin[2]) / voxel_size + center_offset_z
        return wp.vec3(vx, vy, vz)

    @warp_func(f"voxel_to_world_bs{block_size}")
    def voxel_to_world(
        voxel_idx: wp.vec3i,
        origin: wp.vec3,
        voxel_size: float,
        grid_W: int,
        grid_H: int,
        grid_D: int,
    ) -> wp.vec3:
        center_offset_x = wp.float32(grid_W) * 0.5
        center_offset_y = wp.float32(grid_H) * 0.5
        center_offset_z = wp.float32(grid_D) * 0.5
        wx = (wp.float32(voxel_idx[0]) + 0.5 - center_offset_x) * voxel_size + origin[0]
        wy = (wp.float32(voxel_idx[1]) + 0.5 - center_offset_y) * voxel_size + origin[1]
        wz = (wp.float32(voxel_idx[2]) + 0.5 - center_offset_z) * voxel_size + origin[2]
        return wp.vec3(wx, wy, wz)

    @warp_func(f"voxel_to_world_corner_bs{block_size}")
    def voxel_to_world_corner(
        voxel_idx: wp.vec3i,
        origin: wp.vec3,
        voxel_size: float,
        grid_W: int,
        grid_H: int,
        grid_D: int,
    ) -> wp.vec3:
        center_offset_x = wp.float32(grid_W) * 0.5
        center_offset_y = wp.float32(grid_H) * 0.5
        center_offset_z = wp.float32(grid_D) * 0.5
        wx = (wp.float32(voxel_idx[0]) - center_offset_x) * voxel_size + origin[0]
        wy = (wp.float32(voxel_idx[1]) - center_offset_y) * voxel_size + origin[1]
        wz = (wp.float32(voxel_idx[2]) - center_offset_z) * voxel_size + origin[2]
        return wp.vec3(wx, wy, wz)

    # =====================================================================
    # World <-> Block Conversion
    # =====================================================================

    @warp_func(f"_ceil_div_int_bs{block_size}")
    def _ceil_div_int(value: wp.int32, divisor: wp.int32) -> wp.int32:
        return (value + divisor - wp.int32(1)) // divisor

    @warp_func(f"block_offsets_bs{block_size}")
    def block_offsets(
        grid_W: wp.int32,
        grid_H: wp.int32,
        grid_D: wp.int32,
    ) -> wp.vec3i:
        blocks_x = _ceil_div_int(grid_W, BLOCK_SIZE)
        blocks_y = _ceil_div_int(grid_H, BLOCK_SIZE)
        blocks_z = _ceil_div_int(grid_D, BLOCK_SIZE)
        return wp.vec3i(blocks_x // wp.int32(2), blocks_y // wp.int32(2), blocks_z // wp.int32(2))

    @warp_func(f"block_grid_to_key_coords_bs{block_size}")
    def block_grid_to_key_coords(
        bx_grid: wp.int32,
        by_grid: wp.int32,
        bz_grid: wp.int32,
        grid_W: wp.int32,
        grid_H: wp.int32,
        grid_D: wp.int32,
    ) -> wp.vec3i:
        offsets = block_offsets(grid_W, grid_H, grid_D)
        return wp.vec3i(bx_grid - offsets[0], by_grid - offsets[1], bz_grid - offsets[2])

    @warp_func(f"block_key_to_grid_coords_bs{block_size}")
    def block_key_to_grid_coords(
        bx_key: wp.int32,
        by_key: wp.int32,
        bz_key: wp.int32,
        grid_W: wp.int32,
        grid_H: wp.int32,
        grid_D: wp.int32,
    ) -> wp.vec3i:
        offsets = block_offsets(grid_W, grid_H, grid_D)
        return wp.vec3i(bx_key + offsets[0], by_key + offsets[1], bz_key + offsets[2])

    @warp_func(f"block_key_to_voxel_base_bs{block_size}")
    def block_key_to_voxel_base(
        bx_key: wp.int32,
        by_key: wp.int32,
        bz_key: wp.int32,
        grid_W: wp.int32,
        grid_H: wp.int32,
        grid_D: wp.int32,
    ) -> wp.vec3i:
        grid = block_key_to_grid_coords(bx_key, by_key, bz_key, grid_W, grid_H, grid_D)
        return wp.vec3i(grid[0] * BLOCK_SIZE, grid[1] * BLOCK_SIZE, grid[2] * BLOCK_SIZE)

    @warp_func(f"world_to_block_coords_bs{block_size}")
    def world_to_block_coords(
        world_pos: wp.vec3,
        origin: wp.vec3,
        voxel_size: float,
        grid_W: int,
        grid_H: int,
        grid_D: int,
    ) -> wp.vec3i:
        block_size_f = wp.float32(BLOCK_SIZE)
        center_offset_x = wp.float32(grid_W) * 0.5
        center_offset_y = wp.float32(grid_H) * 0.5
        center_offset_z = wp.float32(grid_D) * 0.5
        vx = (world_pos[0] - origin[0]) / voxel_size + center_offset_x
        vy = (world_pos[1] - origin[1]) / voxel_size + center_offset_y
        vz = (world_pos[2] - origin[2]) / voxel_size + center_offset_z
        bx_grid = wp.int32(wp.floor(vx / block_size_f))
        by_grid = wp.int32(wp.floor(vy / block_size_f))
        bz_grid = wp.int32(wp.floor(vz / block_size_f))
        return block_grid_to_key_coords(
            bx_grid, by_grid, bz_grid, grid_W, grid_H, grid_D
        )

    @warp_func(f"world_to_block_and_local_bs{block_size}")
    def world_to_block_and_local(
        world_pos: wp.vec3,
        origin: wp.vec3,
        voxel_size: float,
        grid_W: int,
        grid_H: int,
        grid_D: int,
    ) -> wp.vec4i:
        block_size_f = wp.float32(BLOCK_SIZE)
        center_offset_x = wp.float32(grid_W) * 0.5
        center_offset_y = wp.float32(grid_H) * 0.5
        center_offset_z = wp.float32(grid_D) * 0.5
        vx = (world_pos[0] - origin[0]) / voxel_size + center_offset_x
        vy = (world_pos[1] - origin[1]) / voxel_size + center_offset_y
        vz = (world_pos[2] - origin[2]) / voxel_size + center_offset_z
        bx_grid = wp.int32(wp.floor(vx / block_size_f))
        by_grid = wp.int32(wp.floor(vy / block_size_f))
        bz_grid = wp.int32(wp.floor(vz / block_size_f))
        lx = wp.int32(vx - wp.float32(bx_grid) * block_size_f)
        ly = wp.int32(vy - wp.float32(by_grid) * block_size_f)
        lz = wp.int32(vz - wp.float32(bz_grid) * block_size_f)
        lx = wp.clamp(lx, 0, BLOCK_SIZE - wp.int32(1))
        ly = wp.clamp(ly, 0, BLOCK_SIZE - wp.int32(1))
        lz = wp.clamp(lz, 0, BLOCK_SIZE - wp.int32(1))
        local_idx = lz * BLOCK_SIZE * BLOCK_SIZE + ly * BLOCK_SIZE + lx
        key = block_grid_to_key_coords(
            bx_grid, by_grid, bz_grid, grid_W, grid_H, grid_D
        )
        return wp.vec4i(key[0], key[1], key[2], local_idx)

    @warp_func(f"block_local_to_world_bs{block_size}")
    def block_local_to_world(
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        local_idx: wp.int32,
        origin: wp.vec3,
        voxel_size: float,
        grid_W: int,
        grid_H: int,
        grid_D: int,
    ) -> wp.vec3:
        lx = local_idx % BLOCK_SIZE
        ly = (local_idx // BLOCK_SIZE) % BLOCK_SIZE
        lz = local_idx // (BLOCK_SIZE * BLOCK_SIZE)
        base = block_key_to_voxel_base(bx, by, bz, grid_W, grid_H, grid_D)
        vx = base[0] + lx
        vy = base[1] + ly
        vz = base[2] + lz
        return voxel_to_world(wp.vec3i(vx, vy, vz), origin, voxel_size, grid_W, grid_H, grid_D)

    # =====================================================================
    # Local Index Conversion
    # =====================================================================

    @warp_func(f"local_to_linear_index_bs{block_size}")
    def local_to_linear_index(lx: wp.int32, ly: wp.int32, lz: wp.int32) -> wp.int32:
        return lz * BLOCK_SIZE * BLOCK_SIZE + ly * BLOCK_SIZE + lx

    @warp_func(f"linear_to_local_coords_bs{block_size}")
    def linear_to_local_coords(linear_idx: wp.int32) -> wp.vec3i:
        lx = linear_idx % BLOCK_SIZE
        ly = (linear_idx // BLOCK_SIZE) % BLOCK_SIZE
        lz = linear_idx // (BLOCK_SIZE * BLOCK_SIZE)
        return wp.vec3i(lx, ly, lz)

    # Register all closures on the instance.
    return {
        "world_to_continuous_voxel": world_to_continuous_voxel,
        "voxel_to_world": voxel_to_world,
        "voxel_to_world_corner": voxel_to_world_corner,
        "block_offsets": block_offsets,
        "block_grid_to_key_coords": block_grid_to_key_coords,
        "block_key_to_grid_coords": block_key_to_grid_coords,
        "block_key_to_voxel_base": block_key_to_voxel_base,
        "world_to_block_coords": world_to_block_coords,
        "world_to_block_and_local": world_to_block_and_local,
        "block_local_to_world": block_local_to_world,
        "local_to_linear_index": local_to_linear_index,
        "linear_to_local_coords": linear_to_local_coords,
    }
