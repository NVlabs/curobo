# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse hash table operations, per-``block_size`` builder.

Moved from :mod:`curobo._src.perception.mapper.kernel.wp_hash` in the
block-size builder refactor. All device-side functions and kernels
here are defined via closures that capture ``BS = wp.constant(block_size)``
so a single process can hold multiple specializations.

Default bit layout (64 bits total)::

    ┌─────────────────────────────────────────────────┬────────────────┐
    │              Block Key (39 bits)                │ Value (25)     │
    │   X (13 bits)  |  Y (13 bits)  |  Z (13 bits)   │ pool_idx       │
    └─────────────────────────────────────────────────┴────────────────┘

Thread Safety:
    Single atomic CAS on packed entry ensures both key and value are
    written atomically. No race conditions possible.

CUDA Graph Compatibility:
    All operations use pre-allocated memory and atomic primitives.
    No host-device synchronization is required.
"""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    HASH_EMPTY,
    HASH_PRIME_X,
    HASH_PRIME_Y,
    HASH_PRIME_Z,
    HASH_TOMBSTONE,
    HashLayout,
)
from curobo._src.util.warp import warp_func, warp_kernel

# Back-compat aliases — the canonical names live in
# :mod:`curobo._src.perception.mapper.constants` but the historical
# hash-kernel code used ``ENTRY_EMPTY`` / ``ENTRY_TOMBSTONE``.
ENTRY_EMPTY = HASH_EMPTY
ENTRY_TOMBSTONE = HASH_TOMBSTONE


def make_hash_kernels(
    block_size: int,
    hash_layout: HashLayout = DEFAULT_HASH_LAYOUT,
) -> dict[str, object]:
    """Build hash-table Warp functions and block-clear kernels."""
    BS = wp.constant(block_size)
    suffix = f"bs{block_size}_{hash_layout.name}"

    X_SHIFT = wp.constant(wp.int64(hash_layout.x_shift))
    Y_SHIFT = wp.constant(wp.int64(hash_layout.y_shift))
    Z_SHIFT = wp.constant(wp.int64(hash_layout.z_shift))

    X_MASK = wp.constant(wp.int64(hash_layout.coord_masks_xyz[0]))
    Y_MASK = wp.constant(wp.int64(hash_layout.coord_masks_xyz[1]))
    Z_MASK = wp.constant(wp.int64(hash_layout.coord_masks_xyz[2]))

    X_BIAS = wp.constant(wp.int64(hash_layout.coord_bias_xyz[0]))
    Y_BIAS = wp.constant(wp.int64(hash_layout.coord_bias_xyz[1]))
    Z_BIAS = wp.constant(wp.int64(hash_layout.coord_bias_xyz[2]))

    X_MIN = wp.constant(wp.int32(hash_layout.coord_min_xyz[0]))
    Y_MIN = wp.constant(wp.int32(hash_layout.coord_min_xyz[1]))
    Z_MIN = wp.constant(wp.int32(hash_layout.coord_min_xyz[2]))
    X_MAX = wp.constant(wp.int32(hash_layout.coord_max_xyz[0]))
    Y_MAX = wp.constant(wp.int32(hash_layout.coord_max_xyz[1]))
    Z_MAX = wp.constant(wp.int32(hash_layout.coord_max_xyz[2]))

    VALUE_MASK = wp.constant(wp.int64(hash_layout.value_mask))
    KEY_MASK = wp.constant(wp.int64(hash_layout.key_mask_signed))
    PENDING_POOL_IDX_WP = wp.constant(wp.int32(hash_layout.pending_pool_idx))

    # =====================================================================
    # Entry Packing/Unpacking
    # =====================================================================

    @warp_func(f"pack_entry_{suffix}")
    def pack_entry(bx: wp.int32, by: wp.int32, bz: wp.int32, pool_idx: wp.int32) -> wp.int64:
        x = (wp.int64(bx) + X_BIAS) & X_MASK
        y = (wp.int64(by) + Y_BIAS) & Y_MASK
        z = (wp.int64(bz) + Z_BIAS) & Z_MASK
        v = wp.int64(pool_idx) & VALUE_MASK
        return (x << X_SHIFT) | (y << Y_SHIFT) | (z << Z_SHIFT) | v

    @warp_func(f"pack_key_only_{suffix}")
    def pack_key_only(bx: wp.int32, by: wp.int32, bz: wp.int32) -> wp.int64:
        x = (wp.int64(bx) + X_BIAS) & X_MASK
        y = (wp.int64(by) + Y_BIAS) & Y_MASK
        z = (wp.int64(bz) + Z_BIAS) & Z_MASK
        return (x << X_SHIFT) | (y << Y_SHIFT) | (z << Z_SHIFT)

    @warp_func(f"unpack_entry_{suffix}")
    def unpack_entry(entry: wp.int64) -> wp.vec4i:
        x = wp.int32(((entry >> X_SHIFT) & X_MASK) - X_BIAS)
        y = wp.int32(((entry >> Y_SHIFT) & Y_MASK) - Y_BIAS)
        z = wp.int32(((entry >> Z_SHIFT) & Z_MASK) - Z_BIAS)
        v = wp.int32(entry & VALUE_MASK)
        return wp.vec4i(x, y, z, v)

    @warp_func(f"get_pool_idx_{suffix}")
    def get_pool_idx(entry: wp.int64) -> wp.int32:
        return wp.int32(entry & VALUE_MASK)

    @warp_func(f"get_key_part_{suffix}")
    def get_key_part(entry: wp.int64) -> wp.int64:
        return entry & KEY_MASK

    @warp_func(f"is_valid_block_key_{suffix}")
    def is_valid_block_key(bx: wp.int32, by: wp.int32, bz: wp.int32) -> wp.bool:
        return (
            bx >= X_MIN
            and bx <= X_MAX
            and by >= Y_MIN
            and by <= Y_MAX
            and bz >= Z_MIN
            and bz <= Z_MAX
        )

    # =====================================================================
    # Hash Functions
    # =====================================================================

    @warp_func(f"spatial_hash_{suffix}")
    def spatial_hash(bx: wp.int32, by: wp.int32, bz: wp.int32, capacity: wp.int32) -> wp.int32:
        h = wp.int64(bx) * HASH_PRIME_X ^ wp.int64(by) * HASH_PRIME_Y ^ wp.int64(bz) * HASH_PRIME_Z
        h = h & wp.int64(0x7FFFFFFFFFFFFFFF)
        result = wp.int32(h % wp.int64(capacity))
        if result < wp.int32(0):
            result = result + capacity
        return result

    # =====================================================================
    # RGB Packing/Unpacking (for atomic color writes)
    # =====================================================================

    @warp_func(f"pack_rgb_{suffix}")
    def pack_rgb(r: wp.uint8, g: wp.uint8, b: wp.uint8) -> wp.int32:
        return (wp.int32(r) << 16) | (wp.int32(g) << 8) | wp.int32(b)

    @warp_func(f"compute_avg_rgb_from_block_{suffix}")
    def compute_avg_rgb_from_block(
        block_rgb: wp.array2d(dtype=wp.float16),
        pool_idx: wp.int32,
    ) -> wp.vec3:
        """Weighted-mean RGB for a block, rescaled to ``[0, 255]``.

        Weighted sums are stored fp16 with RGB pre-normalized to
        ``[0, 1]`` at the integration site; divide in fp32 and
        multiply by 255 to match the legacy uint8-scaled output
        consumers expect.
        """
        r_sum = wp.float32(block_rgb[pool_idx, 0])
        g_sum = wp.float32(block_rgb[pool_idx, 1])
        b_sum = wp.float32(block_rgb[pool_idx, 2])
        w = wp.float32(block_rgb[pool_idx, 3])
        if w < 1e-6:
            return wp.vec3(0.0, 0.0, 0.0)
        scale = 255.0 / w
        r = wp.clamp(r_sum * scale, 0.0, 255.0)
        g = wp.clamp(g_sum * scale, 0.0, 255.0)
        b = wp.clamp(b_sum * scale, 0.0, 255.0)
        return wp.vec3(r, g, b)

    @warp_func(f"compute_avg_rgb_uint8_from_block_{suffix}")
    def compute_avg_rgb_uint8_from_block(
        block_rgb: wp.array2d(dtype=wp.float16),
        pool_idx: wp.int32,
    ) -> wp.vec3i:
        r_sum = wp.float32(block_rgb[pool_idx, 0])
        g_sum = wp.float32(block_rgb[pool_idx, 1])
        b_sum = wp.float32(block_rgb[pool_idx, 2])
        w = wp.float32(block_rgb[pool_idx, 3])
        if w < 1e-6:
            return wp.vec3i(0, 0, 0)
        scale = 255.0 / w
        r = wp.int32(wp.clamp(r_sum * scale, 0.0, 255.0))
        g = wp.int32(wp.clamp(g_sum * scale, 0.0, 255.0))
        b = wp.int32(wp.clamp(b_sum * scale, 0.0, 255.0))
        return wp.vec3i(r, g, b)

    # =====================================================================
    # Hash Table Lookup
    # =====================================================================

    @warp_func(f"hash_lookup_{suffix}")
    def hash_lookup(
        hash_table: wp.array(dtype=wp.int64),
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        capacity: wp.int32,
    ) -> wp.int32:
        if not is_valid_block_key(bx, by, bz):
            return wp.int32(-1)
        target_key = pack_key_only(bx, by, bz)
        slot = spatial_hash(bx, by, bz, capacity)
        for _ in range(capacity):
            entry = hash_table[slot]
            if entry == ENTRY_EMPTY:
                return wp.int32(-1)
            if get_key_part(entry) == target_key:
                pool_idx = get_pool_idx(entry)
                if pool_idx == PENDING_POOL_IDX_WP:
                    return wp.int32(-1)
                return pool_idx
            slot = (slot + wp.int32(1)) % capacity
        return wp.int32(-1)

    # =====================================================================
    # Free List Operations
    # =====================================================================

    @warp_func(f"free_list_pop_{suffix}")
    def free_list_pop(
        free_list: wp.array(dtype=wp.int32),
        free_count: wp.array(dtype=wp.int32),
    ) -> wp.int32:
        old = wp.atomic_add(free_count, 0, wp.int32(-1))
        if old <= wp.int32(0):
            wp.atomic_add(free_count, 0, wp.int32(1))
            return wp.int32(-1)
        return free_list[old - wp.int32(1)]

    @warp_func(f"free_list_push_{suffix}")
    def free_list_push(
        free_list: wp.array(dtype=wp.int32),
        free_count: wp.array(dtype=wp.int32),
        pool_idx: wp.int32,
        max_blocks: wp.int32,
    ) -> wp.int32:
        slot = wp.atomic_add(free_count, 0, wp.int32(1))
        if slot >= max_blocks:
            wp.atomic_add(free_count, 0, wp.int32(-1))
            return wp.int32(0)
        free_list[slot] = pool_idx
        return wp.int32(1)

    # =====================================================================
    # Find-or-Insert
    # =====================================================================

    @warp_func(f"spin_until_ready_{suffix}")
    def spin_until_ready(
        hash_table: wp.array(dtype=wp.int64),
        slot: wp.int32,
        target_key: wp.int64,
        pending_marker: wp.int32,
    ) -> wp.int32:
        for _spin in range(8192):
            entry = hash_table[slot]
            pool_idx = get_pool_idx(entry)
            if pool_idx != pending_marker:
                if get_key_part(entry) == target_key:
                    return pool_idx
                return wp.int32(-1)
        return wp.int32(-1)

    @warp_func(f"find_or_insert_block_{suffix}")
    def find_or_insert_block(
        hash_table: wp.array(dtype=wp.int64),
        block_coords: wp.array(dtype=wp.int32),
        block_to_hash_slot: wp.array(dtype=wp.int32),
        free_list: wp.array(dtype=wp.int32),
        free_count: wp.array(dtype=wp.int32),
        num_allocated: wp.array(dtype=wp.int32),
        allocation_failures: wp.array(dtype=wp.int32),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        capacity: wp.int32,
        max_blocks: wp.int32,
    ) -> wp.int32:
        if not is_valid_block_key(bx, by, bz):
            wp.atomic_add(allocation_failures, 0, wp.int32(1))
            return wp.int32(-1)
        target_key = pack_key_only(bx, by, bz)
        start_slot = spatial_hash(bx, by, bz, capacity)

        for _retry in range(64):
            slot = start_slot
            insert_slot = wp.int32(-1)

            for _probe in range(64):
                entry = hash_table[slot]

                if entry == ENTRY_EMPTY:
                    if insert_slot < wp.int32(0):
                        insert_slot = slot
                    break

                if entry == ENTRY_TOMBSTONE:
                    if insert_slot < wp.int32(0):
                        insert_slot = slot
                    slot = (slot + wp.int32(1)) % capacity
                    continue

                if get_key_part(entry) == target_key:
                    pool_idx = get_pool_idx(entry)
                    if pool_idx != PENDING_POOL_IDX_WP:
                        return pool_idx
                    pool_idx = spin_until_ready(hash_table, slot, target_key, PENDING_POOL_IDX_WP)
                    if pool_idx >= wp.int32(0):
                        return pool_idx
                    break

                slot = (slot + wp.int32(1)) % capacity

            if insert_slot < wp.int32(0):
                continue

            pending_entry = pack_entry(bx, by, bz, PENDING_POOL_IDX_WP)
            old_entry = hash_table[insert_slot]

            if old_entry != ENTRY_EMPTY and old_entry != ENTRY_TOMBSTONE:
                if get_key_part(old_entry) == target_key:
                    pool_idx = get_pool_idx(old_entry)
                    if pool_idx != PENDING_POOL_IDX_WP:
                        return pool_idx
                    pool_idx = spin_until_ready(
                        hash_table, insert_slot, target_key, PENDING_POOL_IDX_WP
                    )
                    if pool_idx >= wp.int32(0):
                        return pool_idx
                continue

            cas_result = wp.atomic_cas(hash_table, insert_slot, old_entry, pending_entry)

            if cas_result != old_entry:
                if get_key_part(cas_result) == target_key:
                    pool_idx = get_pool_idx(cas_result)
                    if pool_idx != PENDING_POOL_IDX_WP:
                        return pool_idx
                    pool_idx = spin_until_ready(
                        hash_table, insert_slot, target_key, PENDING_POOL_IDX_WP
                    )
                    if pool_idx >= wp.int32(0):
                        return pool_idx
                continue

            pool_idx = free_list_pop(free_list, free_count)
            if pool_idx == wp.int32(-1):
                pool_idx = wp.atomic_add(num_allocated, 0, wp.int32(1))
                if pool_idx >= max_blocks:
                    wp.atomic_add(num_allocated, 0, wp.int32(-1))
                    hash_table[insert_slot] = ENTRY_TOMBSTONE
                    wp.atomic_add(allocation_failures, 0, wp.int32(1))
                    return wp.int32(-1)

            final_entry = pack_entry(bx, by, bz, pool_idx)
            hash_table[insert_slot] = final_entry

            block_coords[pool_idx * wp.int32(3) + wp.int32(0)] = bx
            block_coords[pool_idx * wp.int32(3) + wp.int32(1)] = by
            block_coords[pool_idx * wp.int32(3) + wp.int32(2)] = bz
            block_to_hash_slot[pool_idx] = insert_slot

            clear_idx = wp.atomic_add(new_block_count, 0, wp.int32(1))
            if clear_idx < max_blocks:
                new_blocks[clear_idx] = pool_idx

            return pool_idx

        # Retry exhaustion can happen under high duplicate-key contention
        # while another thread owns a pending insert. It is not pool
        # exhaustion, so do not report it as an allocation failure.
        return wp.int32(-1)

    # =====================================================================
    # Simplified Hash Table Insert (for unique keys with pre-allocated pool_idx)
    # =====================================================================

    @warp_func(f"hash_table_insert_with_pool_idx_{suffix}")
    def hash_table_insert_with_pool_idx(
        hash_table: wp.array(dtype=wp.int64),
        block_coords: wp.array(dtype=wp.int32),
        block_to_hash_slot: wp.array(dtype=wp.int32),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        pool_idx: wp.int32,
        capacity: wp.int32,
        max_blocks: wp.int32,
    ) -> wp.int32:
        if not is_valid_block_key(bx, by, bz):
            return wp.int32(-1)
        target_key = pack_key_only(bx, by, bz)
        slot = spatial_hash(bx, by, bz, capacity)
        final_entry = pack_entry(bx, by, bz, pool_idx)

        for _probe in range(capacity):
            entry = hash_table[slot]

            if entry != ENTRY_EMPTY and entry != ENTRY_TOMBSTONE:
                if get_key_part(entry) == target_key:
                    return get_pool_idx(entry)
                slot = (slot + 1) % capacity
                continue

            old_entry = wp.atomic_cas(hash_table, slot, entry, final_entry)

            if old_entry == entry:
                block_coords[pool_idx * 3 + 0] = bx
                block_coords[pool_idx * 3 + 1] = by
                block_coords[pool_idx * 3 + 2] = bz
                block_to_hash_slot[pool_idx] = slot
                clear_idx = wp.atomic_add(new_block_count, 0, wp.int32(1))
                if clear_idx < max_blocks:
                    new_blocks[clear_idx] = pool_idx
                return pool_idx

            if old_entry != ENTRY_EMPTY and old_entry != ENTRY_TOMBSTONE:
                if get_key_part(old_entry) == target_key:
                    return get_pool_idx(old_entry)
                slot = (slot + 1) % capacity

        return wp.int32(-1)

    # =====================================================================
    # TSDF Voxel Access
    # =====================================================================

    @warp_func(f"read_tsdf_voxel_{suffix}")
    def read_tsdf_voxel(
        block_data: wp.array3d(dtype=wp.float16),
        pool_idx: wp.int32,
        local_idx: wp.int32,
    ) -> wp.vec2:
        sdf_w = wp.float32(block_data[pool_idx, local_idx, 0])
        w = wp.float32(block_data[pool_idx, local_idx, 1])
        return wp.vec2(sdf_w, w)

    @warp_func(f"write_tsdf_voxel_{suffix}")
    def write_tsdf_voxel(
        block_data: wp.array3d(dtype=wp.float16),
        pool_idx: wp.int32,
        local_idx: wp.int32,
        sdf_weight: wp.float32,
        weight: wp.float32,
    ):
        block_data[pool_idx, local_idx, 0] = wp.float16(sdf_weight)
        block_data[pool_idx, local_idx, 1] = wp.float16(weight)

    # =====================================================================
    # Legacy Wrappers — the historical names ``pack_block_key`` and
    # ``unpack_block_key`` are still used by downstream kernels.
    # =====================================================================

    @warp_func(f"pack_block_key_{suffix}")
    def pack_block_key(bx: wp.int32, by: wp.int32, bz: wp.int32) -> wp.int64:
        return pack_key_only(bx, by, bz)

    @warp_func(f"unpack_block_key_{suffix}")
    def unpack_block_key(key: wp.int64) -> wp.vec3i:
        result = unpack_entry(key)
        return wp.vec3i(result[0], result[1], result[2])

    # =====================================================================
    # Block Clearing Kernel (only @wp.kernel in wp_hash; BS-sensitive)
    # =====================================================================

    @warp_kernel(f"clear_new_blocks_kernel_{suffix}")
    def clear_new_blocks_kernel(
        block_data: wp.array3d(dtype=wp.float16),
        block_rgb: wp.array2d(dtype=wp.float16),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
    ):
        slot_idx, local_idx = wp.tid()

        count = new_block_count[0]
        if count > max_blocks:
            count = max_blocks
        if slot_idx >= count:
            return

        pool_idx = new_blocks[slot_idx]

        block_data[pool_idx, local_idx, 0] = wp.float16(0.0)
        block_data[pool_idx, local_idx, 1] = wp.float16(0.0)

        if local_idx == 0:
            block_rgb[pool_idx, 0] = wp.float16(0.0)
            block_rgb[pool_idx, 1] = wp.float16(0.0)
            block_rgb[pool_idx, 2] = wp.float16(0.0)
            block_rgb[pool_idx, 3] = wp.float16(0.0)

    @warp_kernel(f"clear_new_block_features_kernel_{suffix}")
    def clear_new_block_features_kernel(
        block_features: wp.array2d(dtype=wp.float16),
        block_feature_weight: wp.array(dtype=wp.float16),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
    ):
        """Zero the per-block feature accumulator for freshly allocated blocks.

        Launch with ``dim = (max_clearable, feature_dim)`` so one thread
        clears one ``(block, channel)`` cell; the thread with
        ``ch == 0`` also zeroes the per-block feature weight.
        """
        slot_idx, ch = wp.tid()

        count = new_block_count[0]
        if count > max_blocks:
            count = max_blocks
        if slot_idx >= count:
            return

        pool_idx = new_blocks[slot_idx]
        block_features[pool_idx, ch] = wp.float16(0.0)
        if ch == wp.int32(0):
            block_feature_weight[pool_idx] = wp.float16(0.0)

    # Register all closures on the instance for external access.
    return {
        "pack_entry": pack_entry,
        "pack_key_only": pack_key_only,
        "unpack_entry": unpack_entry,
        "get_pool_idx": get_pool_idx,
        "get_key_part": get_key_part,
        "is_valid_block_key": is_valid_block_key,
        "spatial_hash": spatial_hash,
        "pack_rgb": pack_rgb,
        "compute_avg_rgb_from_block": compute_avg_rgb_from_block,
        "compute_avg_rgb_uint8_from_block": compute_avg_rgb_uint8_from_block,
        "hash_lookup": hash_lookup,
        "free_list_pop": free_list_pop,
        "free_list_push": free_list_push,
        "spin_until_ready": spin_until_ready,
        "find_or_insert_block": find_or_insert_block,
        "hash_table_insert_with_pool_idx": hash_table_insert_with_pool_idx,
        "read_tsdf_voxel": read_tsdf_voxel,
        "write_tsdf_voxel": write_tsdf_voxel,
        "pack_block_key": pack_block_key,
        "unpack_block_key": unpack_block_key,
        "clear_new_blocks_kernel": clear_new_blocks_kernel,
        "clear_new_block_features_kernel": clear_new_block_features_kernel,
    }
