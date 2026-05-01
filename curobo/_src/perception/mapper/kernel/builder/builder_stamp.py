# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Obstacle-stamping kernels for one block size.

``filter_blocks_by_sdf_kernel`` and ``stamp_sdf_kernel`` close over the
fixed TSDF geometry and truncation distance. Runtime arguments are limited
to obstacle data, allocation state, and output tensors.

The obstacle-SDF overload registration remains at module scope in
``kernel/wp_stamp_obstacles.py``. The stamp kernels in this builder call
those overloads via cross-module resolution.
"""

from __future__ import annotations

from typing import Any

import warp as wp

# Overloaded SDF funcs live at module scope in wp_stamp_obstacles.py.
# The overload registry is initialized at that module's import time; we
# just reference the names from this side.
from curobo._src.perception.mapper.kernel.wp_stamp_obstacles import (
    compute_local_sdf,
    is_obs_enabled,
    load_obstacle_transform,
)
from curobo._src.util.warp import warp_constant_suffix, warp_func, warp_kernel


def make_stamp_kernels(
    block_size: int,
    *,
    grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
    truncation_distance: float,
    pack_key_only,
    unpack_block_key,
    block_local_to_world,
    hash_lookup,
    hash_table_insert_with_pool_idx,
    free_list_pop,
) -> dict[str, object]:
    """Build obstacle-stamping kernels."""
    suffix = (
        f"bs{block_size}_cfg"
        f"{warp_constant_suffix(block_size, grid_shape, origin_xyz, voxel_size, truncation_distance)}"
    )
    BLOCK_SIZE = wp.constant(wp.int32(block_size))
    GRID_D = wp.constant(wp.int32(grid_shape[0]))
    GRID_H = wp.constant(wp.int32(grid_shape[1]))
    GRID_W = wp.constant(wp.int32(grid_shape[2]))
    ORIGIN_X = wp.constant(wp.float32(origin_xyz[0]))
    ORIGIN_Y = wp.constant(wp.float32(origin_xyz[1]))
    ORIGIN_Z = wp.constant(wp.float32(origin_xyz[2]))
    VOXEL_SIZE = wp.constant(wp.float32(voxel_size))
    TRUNCATION_DIST = wp.constant(wp.float32(truncation_distance))
    max_sdf_threshold = truncation_distance + (3.0**0.5) * block_size * voxel_size * 0.5
    MAX_SDF_THRESHOLD = wp.constant(wp.float32(max_sdf_threshold))

    # Cross-domain helpers are parameters so Warp sees them as local
    # closure bindings when compiling the kernels below.

    @warp_func(f"stamp_is_block_in_bounds_{suffix}")
    def _is_block_in_bounds(
        bx_key: wp.int32,
        by_key: wp.int32,
        bz_key: wp.int32,
    ) -> wp.bool:
        blocks_W = (GRID_W + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        blocks_H = (GRID_H + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        blocks_D = (GRID_D + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        bx = bx_key + blocks_W // wp.int32(2)
        by = by_key + blocks_H // wp.int32(2)
        bz = bz_key + blocks_D // wp.int32(2)

        if bx < 0 or bx >= blocks_W:
            return False
        if by < 0 or by >= blocks_H:
            return False
        if bz < 0 or bz >= blocks_D:
            return False
        return True

    @warp_func(f"stamp_block_center_to_world_{suffix}")
    def _block_center_to_world(
        bx_key: wp.int32,
        by_key: wp.int32,
        bz_key: wp.int32,
    ) -> wp.vec3:
        block_size_f = wp.float32(BLOCK_SIZE)
        half_block = block_size_f * 0.5
        blocks_W = (GRID_W + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        blocks_H = (GRID_H + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        blocks_D = (GRID_D + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        bx = bx_key + blocks_W // wp.int32(2)
        by = by_key + blocks_H // wp.int32(2)
        bz = bz_key + blocks_D // wp.int32(2)

        vx = wp.float32(bx) * block_size_f + half_block
        vy = wp.float32(by) * block_size_f + half_block
        vz = wp.float32(bz) * block_size_f + half_block

        wx = (vx - wp.float32(GRID_W) * 0.5) * VOXEL_SIZE + ORIGIN_X
        wy = (vy - wp.float32(GRID_H) * 0.5) * VOXEL_SIZE + ORIGIN_Y
        wz = (vz - wp.float32(GRID_D) * 0.5) * VOXEL_SIZE + ORIGIN_Z

        return wp.vec3(wx, wy, wz)

    # =====================================================================
    # Block Pre-allocation (Phase 5)
    # =====================================================================

    @warp_kernel(f"preallocate_unique_blocks_kernel_bs{block_size}")
    def preallocate_unique_blocks_kernel(
        unique_blocks: wp.array(dtype=wp.int64),
        n_unique: wp.int32,
        hash_table: wp.array(dtype=wp.int64),
        hash_capacity: wp.int32,
        block_coords: wp.array(dtype=wp.int32),
        block_to_hash_slot: wp.array(dtype=wp.int32),
        num_allocated: wp.array(dtype=wp.int32),
        allocation_failures: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
        free_list: wp.array(dtype=wp.int32),
        free_count: wp.array(dtype=wp.int32),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        pool_indices: wp.array(dtype=wp.int32),
    ):
        """Pre-allocate unique blocks using CAS for collision handling.

        Each thread handles exactly one unique block key. First
        checks if block already exists, then allocates from free
        list or pool if needed.
        """
        tid = wp.tid()
        if tid >= n_unique:
            return

        key = unique_blocks[tid]
        coords = unpack_block_key(key)
        bx = coords[0]
        by = coords[1]
        bz = coords[2]

        existing_pool_idx = hash_lookup(hash_table, bx, by, bz, hash_capacity)
        if existing_pool_idx >= 0:
            pool_indices[tid] = existing_pool_idx
            return

        pool_idx = free_list_pop(free_list, free_count)
        if pool_idx < wp.int32(0):
            pool_idx = wp.atomic_add(num_allocated, 0, wp.int32(1))
            if pool_idx >= max_blocks:
                wp.atomic_add(num_allocated, 0, wp.int32(-1))
                wp.atomic_add(allocation_failures, 0, wp.int32(1))
                pool_indices[tid] = wp.int32(-1)
                return

        result_pool_idx = hash_table_insert_with_pool_idx(
            hash_table,
            block_coords,
            block_to_hash_slot,
            new_blocks,
            new_block_count,
            bx,
            by,
            bz,
            pool_idx,
            hash_capacity,
            max_blocks,
        )
        pool_indices[tid] = result_pool_idx

    # =====================================================================
    # Phase 2: Block Enumeration
    # =====================================================================

    @warp_kernel(f"enumerate_blocks_from_aabb_kernel_bs{block_size}")
    def enumerate_blocks_from_aabb_kernel(
        aabb_bmin: wp.array2d(dtype=wp.int32),
        aabb_bmax: wp.array2d(dtype=wp.int32),
        n_obs: wp.int32,
        offsets: wp.array(dtype=wp.int32),
        block_keys: wp.array(dtype=wp.int64),
    ):
        """Enumerate blocks from pre-computed AABB bounds."""
        obs_idx = wp.tid()
        if obs_idx >= n_obs:
            return

        bmin_x = aabb_bmin[obs_idx, 0]
        bmin_y = aabb_bmin[obs_idx, 1]
        bmin_z = aabb_bmin[obs_idx, 2]
        bmax_x = aabb_bmax[obs_idx, 0]
        bmax_y = aabb_bmax[obs_idx, 1]
        bmax_z = aabb_bmax[obs_idx, 2]

        write_base = offsets[obs_idx]
        key_idx = wp.int32(0)

        for bz in range(bmin_z, bmax_z + 1):
            for by in range(bmin_y, bmax_y + 1):
                for bx in range(bmin_x, bmax_x + 1):
                    key = pack_key_only(bx, by, bz)
                    block_keys[write_base + key_idx] = key
                    key_idx = key_idx + 1

    # =====================================================================
    # Phase 3: Generic Filter by SDF
    # =====================================================================

    @warp_kernel(f"filter_blocks_by_sdf_kernel_{suffix}", enable_backward=False)
    def filter_blocks_by_sdf_kernel(
        candidate_blocks: wp.array(dtype=wp.int64),
        n_candidates: wp.int32,
        obs_set: Any,
        env_idx: wp.int32,
        filtered_blocks: wp.array(dtype=wp.int64),
        filtered_count: wp.array(dtype=wp.int32),
    ):
        """Filter blocks by SDF at block center using generic compute_sdf_value.

        Uses Warp function overloading: ``compute_local_sdf`` and
        ``is_obs_enabled`` are dispatched based on the concrete type
        of ``obs_set`` at kernel compile time.
        """
        tid = wp.tid()
        if tid >= n_candidates:
            return

        key = candidate_blocks[tid]
        coords = unpack_block_key(key)
        bx = coords[0]
        by = coords[1]
        bz = coords[2]

        if not _is_block_in_bounds(bx, by, bz):
            return

        block_center = _block_center_to_world(bx, by, bz)

        min_sdf = wp.float32(1e10)
        for i in range(obs_set.max_n):
            if is_obs_enabled(obs_set, env_idx, i):
                inv_t = load_obstacle_transform(obs_set, env_idx, i)
                local_pt = wp.transform_point(inv_t, block_center)
                sdf = compute_local_sdf(obs_set, env_idx, i, local_pt)
                min_sdf = wp.min(min_sdf, sdf)

        if wp.abs(min_sdf) <= MAX_SDF_THRESHOLD:
            out_idx = wp.atomic_add(filtered_count, 0, wp.int32(1))
            filtered_blocks[out_idx] = key

    # =====================================================================
    # Phase 6: Stamp SDF per voxel (BLOCK_SIZE-specialized)
    # =====================================================================

    @warp_kernel(f"stamp_sdf_kernel_{suffix}")
    def stamp_sdf_kernel(
        unique_blocks: wp.array(dtype=wp.int64),
        pool_indices: wp.array(dtype=wp.int32),
        n_unique: wp.int32,
        obs_set: Any,
        env_idx: wp.int32,
        static_block_data: wp.array2d(dtype=wp.float16),
        static_block_sums: wp.array(dtype=wp.int32),
    ):
        """Stamp SDF values using generic ``compute_local_sdf``.

        Uses read-min-write.

        Launch with ``dim = (n_unique, BLOCK_SIZE ** 3)``; ``BLOCK_SIZE`` is
        closure-captured so the voxel-per-block count stays
        consistent with the specialized tensor shapes.
        """
        block_list_idx, local_idx = wp.tid()

        if block_list_idx >= n_unique:
            return
        if local_idx >= BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE:
            return

        pool_idx = pool_indices[block_list_idx]
        if pool_idx < 0:
            return

        key = unique_blocks[block_list_idx]
        coords = unpack_block_key(key)
        bx = coords[0]
        by = coords[1]
        bz = coords[2]

        voxel_pos = block_local_to_world(bx, by, bz, local_idx)

        min_sdf = wp.float32(1e10)
        for i in range(obs_set.max_n):
            if is_obs_enabled(obs_set, env_idx, i):
                inv_t = load_obstacle_transform(obs_set, env_idx, i)
                local_pt = wp.transform_point(inv_t, voxel_pos)
                sdf = compute_local_sdf(obs_set, env_idx, i, local_pt)
                min_sdf = wp.min(min_sdf, sdf)

        if wp.abs(min_sdf) <= TRUNCATION_DIST:
            existing = wp.float32(static_block_data[pool_idx, local_idx])
            final_sdf = wp.min(existing, min_sdf)
            clamped_sdf = wp.clamp(final_sdf, -TRUNCATION_DIST, TRUNCATION_DIST)
            was_infinite = existing > 1e9
            static_block_data[pool_idx, local_idx] = wp.float16(clamped_sdf)
            if was_infinite:
                wp.atomic_add(static_block_sums, pool_idx, wp.int32(1))

    # =====================================================================
    # Phase 7: Update Block Colors
    # =====================================================================

    @warp_kernel(f"update_block_rgb_kernel_bs{block_size}")
    def update_block_rgb_kernel(
        block_rgb: wp.array2d(dtype=wp.float16),
        pool_indices: wp.array(dtype=wp.int32),
        n_unique: wp.int32,
        static_color: wp.vec3,
    ):
        """Update block RGB with constant static color for allocated blocks.

        ``static_color`` must be pre-normalized to ``[0, 1]`` (the
        launcher divides the uint8-style config value by 255) to
        match the integration convention and keep the fp16-stored
        weighted sums consistent with the dynamic-channel values.
        """
        tid = wp.tid()
        if tid >= n_unique:
            return

        pool_idx = pool_indices[tid]
        if pool_idx < 0:
            return

        block_rgb[pool_idx, 0] = wp.float16(static_color[0])
        block_rgb[pool_idx, 1] = wp.float16(static_color[1])
        block_rgb[pool_idx, 2] = wp.float16(static_color[2])
        block_rgb[pool_idx, 3] = wp.float16(1.0)

    return {
        "preallocate_unique_blocks_kernel": preallocate_unique_blocks_kernel,
        "enumerate_blocks_from_aabb_kernel": enumerate_blocks_from_aabb_kernel,
        "filter_blocks_by_sdf_kernel": filter_blocks_by_sdf_kernel,
        "stamp_sdf_kernel": stamp_sdf_kernel,
        "update_block_rgb_kernel": update_block_rgb_kernel,
    }
