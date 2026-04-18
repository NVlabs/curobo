# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp functions for block-sparse hash table operations.

This module provides device-side functions for:
1. Packed entry encoding/decoding (64-bit: 14-bit coords + 22-bit pool_idx)
2. Hash table lookup (linear probing)
3. Find-or-insert with single atomic CAS (race-free)
4. Free list management (atomic pop with CAS)

Bit Layout (64 bits total):
    ┌────────────────────────────────────────────────────┬──────────────┐
    │              Block Key (42 bits)                   │ Value (22)   │
    │   X (14 bits)  |  Y (14 bits)  |  Z (14 bits)     │ pool_idx     │
    └────────────────────────────────────────────────────┴──────────────┘

Range:
    - Block coords: ±8,192 per axis (14-bit signed range)
    - At 8mm block size: ±65m per axis
    - At 16mm block size: ±131m per axis
    - Pool index: 0 to 4,194,303 (22 bits = 4M blocks max)

Thread Safety:
    Single atomic CAS on packed entry ensures both key and value are
    written atomically. No race conditions possible.

CUDA Graph Compatibility:
    All operations use pre-allocated memory and atomic primitives.
    No host-device synchronization is required.
"""

import warp as wp

# =============================================================================
# Constants for 64-bit Packed Entries
# =============================================================================

# Coordinate encoding: 14 bits per axis, symmetric signed range (±8192)
COORD_BITS = wp.constant(14)
COORD_MASK = wp.constant(wp.int64(0x3FFF))  # 14 bits = 0x3FFF
COORD_OFFSET = wp.constant(wp.int64(8192))  # Offset for signed range ±8192

# Value encoding: 22 bits for pool index
VALUE_BITS = wp.constant(22)
VALUE_MASK = wp.constant(wp.int64(0x3FFFFF))  # 22 bits

# Bit positions
Z_SHIFT = wp.constant(wp.int64(22))   # Z starts at bit 22
Y_SHIFT = wp.constant(wp.int64(36))   # Y starts at bit 36
X_SHIFT = wp.constant(wp.int64(50))   # X starts at bit 50

# Key mask (upper 42 bits)
KEY_MASK = wp.constant(wp.int64(0xFFFFFFFFFFC00000))

# Special values
ENTRY_EMPTY = wp.constant(wp.int64(-1))      # All 1s = empty slot
ENTRY_TOMBSTONE = wp.constant(wp.int64(-2))  # Deleted entry

# Hash primes (Teschner spatial hash)
HASH_PRIME_X = wp.constant(wp.int64(73856093))
HASH_PRIME_Y = wp.constant(wp.int64(19349663))
HASH_PRIME_Z = wp.constant(wp.int64(83492791))


# =============================================================================
# Entry Packing/Unpacking
# =============================================================================


@wp.func
def pack_entry(bx: wp.int32, by: wp.int32, bz: wp.int32, pool_idx: wp.int32) -> wp.int64:
    """Pack block coordinates and pool index into 64-bit entry.

    Layout: | X (14 bits) | Y (14 bits) | Z (14 bits) | pool_idx (22 bits) |

    Args:
        bx, by, bz: Signed block coordinates (must be in ±8192 range).
        pool_idx: Pool index (must be < 4M).

    Returns:
        64-bit packed entry.
    """
    x = (wp.int64(bx) + COORD_OFFSET) & COORD_MASK
    y = (wp.int64(by) + COORD_OFFSET) & COORD_MASK
    z = (wp.int64(bz) + COORD_OFFSET) & COORD_MASK
    v = wp.int64(pool_idx) & VALUE_MASK

    return (x << X_SHIFT) | (y << Y_SHIFT) | (z << Z_SHIFT) | v


@wp.func
def pack_key_only(bx: wp.int32, by: wp.int32, bz: wp.int32) -> wp.int64:
    """Pack block coordinates only (pool_idx = 0).

    Used for key comparison during lookup.

    Args:
        bx, by, bz: Signed block coordinates.

    Returns:
        64-bit packed key with zero pool_idx.
    """
    x = (wp.int64(bx) + COORD_OFFSET) & COORD_MASK
    y = (wp.int64(by) + COORD_OFFSET) & COORD_MASK
    z = (wp.int64(bz) + COORD_OFFSET) & COORD_MASK

    return (x << X_SHIFT) | (y << Y_SHIFT) | (z << Z_SHIFT)


@wp.func
def unpack_entry(entry: wp.int64) -> wp.vec4i:
    """Unpack 64-bit entry to (bx, by, bz, pool_idx).

    Args:
        entry: 64-bit packed entry.

    Returns:
        vec4i with (bx, by, bz, pool_idx).
    """
    x = wp.int32(((entry >> X_SHIFT) & COORD_MASK) - COORD_OFFSET)
    y = wp.int32(((entry >> Y_SHIFT) & COORD_MASK) - COORD_OFFSET)
    z = wp.int32(((entry >> Z_SHIFT) & COORD_MASK) - COORD_OFFSET)
    v = wp.int32(entry & VALUE_MASK)
    return wp.vec4i(x, y, z, v)


@wp.func
def get_pool_idx(entry: wp.int64) -> wp.int32:
    """Extract pool index from packed entry."""
    return wp.int32(entry & VALUE_MASK)


@wp.func
def get_key_part(entry: wp.int64) -> wp.int64:
    """Extract key part (upper 42 bits) from entry."""
    return entry & KEY_MASK


# =============================================================================
# Hash Functions
# =============================================================================


@wp.func
def spatial_hash(bx: wp.int32, by: wp.int32, bz: wp.int32, capacity: wp.int32) -> wp.int32:
    """Compute hash slot for block coordinates.

    Uses Teschner spatial hash with prime multipliers.

    Args:
        bx, by, bz: Block coordinates.
        capacity: Hash table capacity.

    Returns:
        Hash slot index [0, capacity).
    """
    h = (
        wp.int64(bx) * HASH_PRIME_X
        ^ wp.int64(by) * HASH_PRIME_Y
        ^ wp.int64(bz) * HASH_PRIME_Z
    )
    # Ensure positive result
    h = h & wp.int64(0x7FFFFFFFFFFFFFFF)
    result = wp.int32(h % wp.int64(capacity))
    if result < wp.int32(0):
        result = result + capacity
    return result


# =============================================================================
# RGB Packing/Unpacking (for atomic color writes)
# =============================================================================


@wp.func
def pack_rgb(r: wp.uint8, g: wp.uint8, b: wp.uint8) -> wp.int32:
    """Pack RGB into single int32: 0x00RRGGBB.

    Enables single atomic write to avoid color tearing when multiple
    threads write to the same voxel.

    Args:
        r, g, b: Color components (0-255).

    Returns:
        Packed int32 with RGB.
    """
    return (wp.int32(r) << 16) | (wp.int32(g) << 8) | wp.int32(b)


@wp.func
def compute_avg_rgb_from_block(
    block_rgb: wp.array2d(dtype=wp.float32),
    pool_idx: wp.int32,
) -> wp.vec3:
    """Compute averaged RGB from per-block RGBW storage.

    Args:
        block_rgb: Per-block array with [R×w, G×w, B×w, W] in channels 0-3.
        pool_idx: Block pool index.

    Returns:
        vec3(R, G, B) with values in 0-255 range.
    """
    r_sum = block_rgb[pool_idx, 0]
    g_sum = block_rgb[pool_idx, 1]
    b_sum = block_rgb[pool_idx, 2]
    w = block_rgb[pool_idx, 3]

    if w < 1e-6:
        return wp.vec3(0.0, 0.0, 0.0)
    inv_w = 1.0 / w
    r = wp.clamp(r_sum * inv_w, 0.0, 255.0)
    g = wp.clamp(g_sum * inv_w, 0.0, 255.0)
    b = wp.clamp(b_sum * inv_w, 0.0, 255.0)
    return wp.vec3(r, g, b)


@wp.func
def compute_avg_rgb_uint8_from_block(
    block_rgb: wp.array2d(dtype=wp.float32),
    pool_idx: wp.int32,
) -> wp.vec3i:
    """Compute averaged RGB from per-block RGBW storage as uint8.

    Args:
        block_rgb: Per-block array with [R×w, G×w, B×w, W] in channels 0-3.
        pool_idx: Block pool index.

    Returns:
        vec3i(R, G, B) with values in 0-255 range.
    """
    r_sum = block_rgb[pool_idx, 0]
    g_sum = block_rgb[pool_idx, 1]
    b_sum = block_rgb[pool_idx, 2]
    w = block_rgb[pool_idx, 3]

    if w < 1e-6:
        return wp.vec3i(0, 0, 0)
    inv_w = 1.0 / w
    r = wp.int32(wp.clamp(r_sum * inv_w, 0.0, 255.0))
    g = wp.int32(wp.clamp(g_sum * inv_w, 0.0, 255.0))
    b = wp.int32(wp.clamp(b_sum * inv_w, 0.0, 255.0))
    return wp.vec3i(r, g, b)


# =============================================================================
# Hash Table Lookup
# =============================================================================


@wp.func
def hash_lookup(
    hash_table: wp.array(dtype=wp.int64),
    bx: wp.int32,
    by: wp.int32,
    bz: wp.int32,
    capacity: wp.int32,
) -> wp.int32:
    """Lookup block in hash table.

    Args:
        hash_table: Packed hash table (key+value in each entry).
        bx, by, bz: Block coordinates to find.
        capacity: Hash table capacity.

    Returns:
        Pool index if found, -1 if not found or entry is pending.
    """
    # Pending marker used during allocation
    PENDING_POOL_IDX = wp.int32(0x3FFFFF)

    target_key = pack_key_only(bx, by, bz)
    slot = spatial_hash(bx, by, bz, capacity)

    for _ in range(64):  # Max probes
        entry = hash_table[slot]

        if entry == ENTRY_EMPTY:
            return wp.int32(-1)  # Not found

        # Compare key parts (ignore pool_idx for comparison)
        if get_key_part(entry) == target_key:
            pool_idx = get_pool_idx(entry)
            # Skip pending entries
            if pool_idx == PENDING_POOL_IDX:
                return wp.int32(-1)  # Treat as not found (being allocated)
            return pool_idx

        # Continue probing (handles tombstones too)
        slot = (slot + wp.int32(1)) % capacity

    return wp.int32(-1)  # Not found after max probes


# =============================================================================
# Free List Operations
# =============================================================================


@wp.func
def free_list_pop(
    free_list: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
) -> wp.int32:
    """Pop a pool index from the free list.

    Thread-safe using CAS loop.

    Args:
        free_list: Stack of free pool indices.
        free_count: Current stack size (single element).

    Returns:
        Pool index if successful, -1 if list is empty.
    """
    for _ in range(32):  # Max retries
        count = free_count[0]
        if count <= wp.int32(0):
            return wp.int32(-1)  # Empty

        old = wp.atomic_cas(free_count, 0, count, count - wp.int32(1))
        if old == count:
            return free_list[count - wp.int32(1)]

    return wp.int32(-1)


@wp.func
def free_list_push(
    free_list: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    pool_idx: wp.int32,
    max_blocks: wp.int32,
) -> wp.int32:
    """Push a pool index back to the free list.

    Args:
        free_list: Stack of free pool indices.
        free_count: Current stack size.
        pool_idx: Pool index to push.
        max_blocks: Maximum pool size.

    Returns:
        1 if successful, 0 if list is full.
    """
    slot = wp.atomic_add(free_count, 0, wp.int32(1))

    if slot >= max_blocks:
        wp.atomic_add(free_count, 0, wp.int32(-1))
        return wp.int32(0)

    free_list[slot] = pool_idx
    return wp.int32(1)


# =============================================================================
# Find-or-Insert (Simplified)
# =============================================================================


@wp.func
def spin_until_ready(
    hash_table: wp.array(dtype=wp.int64),
    slot: wp.int32,
    target_key: wp.int64,
    pending_marker: wp.int32,
) -> wp.int32:
    """Spin on a slot until PENDING resolves to a valid pool_idx.

    Returns pool_idx if successful, -1 if timeout or key changed.
    """
    for _spin in range(8192):
        entry = hash_table[slot]
        pool_idx = get_pool_idx(entry)
        if pool_idx != pending_marker:
            if get_key_part(entry) == target_key:
                return pool_idx
            return wp.int32(-1)  # Key changed (tombstone?)
    return wp.int32(-1)  # Timeout


@wp.func
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
    """Find existing block or allocate new one.

    Simplified algorithm:
    1. Linear probe to find key or empty slot
    2. If key found: return pool_idx (spin if PENDING)
    3. If empty found: CAS to claim slot, allocate pool_idx, finalize
    4. If CAS fails: check if winner has our key, spin if PENDING

    Args:
        hash_table: Packed hash table (key+value per entry).
        block_coords, block_to_hash_slot: Block metadata arrays.
        free_list, free_count: Free list for recycled block reuse.
        num_allocated: Pool allocation counter (high-water mark).
        allocation_failures: Failure counter.
        new_blocks, new_block_count: Per-frame new block tracking.
        bx, by, bz: Block coordinates.
        capacity: Hash table size.
        max_blocks: Maximum pool size.

    Returns:
        Pool index if successful, -1 if failed.
    """
    target_key = pack_key_only(bx, by, bz)
    start_slot = spatial_hash(bx, by, bz, capacity)
    PENDING = wp.int32(0x3FFFFF)

    for _retry in range(64):
        slot = start_slot
        insert_slot = wp.int32(-1)

        # Step 1: Linear probe to find key or first empty/tombstone slot
        for _probe in range(64):
            entry = hash_table[slot]

            # Empty slot - key not in table
            if entry == ENTRY_EMPTY:
                if insert_slot < wp.int32(0):
                    insert_slot = slot
                break

            # Tombstone - remember for insertion, keep probing
            if entry == ENTRY_TOMBSTONE:
                if insert_slot < wp.int32(0):
                    insert_slot = slot
                slot = (slot + wp.int32(1)) % capacity
                continue

            # Check if this is our key
            if get_key_part(entry) == target_key:
                pool_idx = get_pool_idx(entry)
                if pool_idx != PENDING:
                    return pool_idx
                # Spin until PENDING resolves
                pool_idx = spin_until_ready(hash_table, slot, target_key, PENDING)
                if pool_idx >= wp.int32(0):
                    return pool_idx
                # Spin failed - retry from beginning
                break

            slot = (slot + wp.int32(1)) % capacity

        # Step 2: Try to insert at empty/tombstone slot
        if insert_slot < wp.int32(0):
            continue  # No slot found, retry

        pending_entry = pack_entry(bx, by, bz, PENDING)
        old_entry = hash_table[insert_slot]

        # Only CAS if slot is still empty or tombstone
        if old_entry != ENTRY_EMPTY and old_entry != ENTRY_TOMBSTONE:
            # Slot taken - check if it's our key
            if get_key_part(old_entry) == target_key:
                pool_idx = get_pool_idx(old_entry)
                if pool_idx != PENDING:
                    return pool_idx
                pool_idx = spin_until_ready(hash_table, insert_slot, target_key, PENDING)
                if pool_idx >= wp.int32(0):
                    return pool_idx
            continue  # Retry

        # Try to claim the slot
        cas_result = wp.atomic_cas(hash_table, insert_slot, old_entry, pending_entry)

        if cas_result != old_entry:
            # CAS failed - check what's there
            if get_key_part(cas_result) == target_key:
                pool_idx = get_pool_idx(cas_result)
                if pool_idx != PENDING:
                    return pool_idx
                pool_idx = spin_until_ready(hash_table, insert_slot, target_key, PENDING)
                if pool_idx >= wp.int32(0):
                    return pool_idx
            continue  # Retry

        # We won the slot! Allocate pool_idx - try free list first
        pool_idx = free_list_pop(free_list, free_count)
        if pool_idx == wp.int32(-1):
            # Free list empty, allocate new block
            pool_idx = wp.atomic_add(num_allocated, 0, wp.int32(1))

            if pool_idx >= max_blocks:
                # Pool exhausted - release slot and fail
                wp.atomic_add(num_allocated, 0, wp.int32(-1))
                hash_table[insert_slot] = ENTRY_TOMBSTONE
                wp.atomic_add(allocation_failures, 0, wp.int32(1))
                return wp.int32(-1)

        # Finalize the entry with real pool_idx
        final_entry = pack_entry(bx, by, bz, pool_idx)
        hash_table[insert_slot] = final_entry

        # Write metadata
        block_coords[pool_idx * wp.int32(3) + wp.int32(0)] = bx
        block_coords[pool_idx * wp.int32(3) + wp.int32(1)] = by
        block_coords[pool_idx * wp.int32(3) + wp.int32(2)] = bz
        block_to_hash_slot[pool_idx] = insert_slot

        # Track new block for clearing
        clear_idx = wp.atomic_add(new_block_count, 0, wp.int32(1))
        if clear_idx < max_blocks:
            new_blocks[clear_idx] = pool_idx

        return pool_idx

    # All retries exhausted
    wp.atomic_add(allocation_failures, 0, wp.int32(1))
    return wp.int32(-1)



# =============================================================================
# Simplified Hash Table Insert (for unique keys with pre-allocated pool_idx)
# =============================================================================


@wp.func
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
    """Insert block into hash table using pre-allocated pool_idx.

    This function is for use cases where:
    1. Keys are known to be unique (no concurrent insertion of same key)
    2. Pool index is allocated before calling this function
    3. CAS is used to handle hash collisions (different keys, same slot)

    Algorithm:
    1. Probe from hash(key) to find empty slot or existing key
    2. If key exists: return existing pool_idx (caller's pool_idx is wasted)
    3. If empty slot: CAS to claim it
       - CAS success: write metadata, return pool_idx
       - CAS fail: continue probing with same pool_idx

    Args:
        hash_table: Packed hash table (key+value per entry).
        block_coords: Block coordinate storage.
        block_to_hash_slot: Mapping from pool_idx to hash slot.
        new_blocks: Array to track newly allocated blocks.
        new_block_count: Counter for new blocks.
        bx, by, bz: Block coordinates.
        pool_idx: Pre-allocated pool index to use.
        capacity: Hash table capacity.
        max_blocks: Maximum pool size (bounds new_blocks tracking).

    Returns:
        Pool index used (may differ from input if key already existed).
        Returns -1 only if hash table is full (should not happen).
    """
    target_key = pack_key_only(bx, by, bz)
    slot = spatial_hash(bx, by, bz, capacity)
    final_entry = pack_entry(bx, by, bz, pool_idx)

    # Probe until CAS succeeds or key found
    for _probe in range(capacity):
        entry = hash_table[slot]

        # Check if key already exists
        if entry != ENTRY_EMPTY and entry != ENTRY_TOMBSTONE:
            if get_key_part(entry) == target_key:
                # Key exists - return existing pool_idx (ours is wasted)
                return get_pool_idx(entry)
            # Different key - continue probing
            slot = (slot + 1) % capacity
            continue

        # Empty/tombstone slot - try CAS
        old_entry = wp.atomic_cas(hash_table, slot, entry, final_entry)

        if old_entry == entry:
            # CAS succeeded - write metadata
            block_coords[pool_idx * 3 + 0] = bx
            block_coords[pool_idx * 3 + 1] = by
            block_coords[pool_idx * 3 + 2] = bz
            block_to_hash_slot[pool_idx] = slot

            # Track new block
            clear_idx = wp.atomic_add(new_block_count, 0, wp.int32(1))
            if clear_idx < max_blocks:
                new_blocks[clear_idx] = pool_idx

            return pool_idx

        # CAS failed - check what's in slot now
        if old_entry != ENTRY_EMPTY and old_entry != ENTRY_TOMBSTONE:
            if get_key_part(old_entry) == target_key:
                # Another thread inserted our key - use their pool_idx
                return get_pool_idx(old_entry)
            # Different key claimed slot - move to next
            slot = (slot + 1) % capacity
        # else: slot still empty/tombstone, retry CAS on same slot

    # Hash table full (shouldn't happen with proper sizing)
    return wp.int32(-1)


# =============================================================================
# TSDF Voxel Access
# =============================================================================


@wp.func
def read_tsdf_voxel(
    block_data: wp.array3d(dtype=wp.float16),
    pool_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.vec2:
    """Read TSDF voxel data (sdf_weight, weight).

    Args:
        block_data: Block data array (3D: max_blocks, 512, 2).
        pool_idx: Block pool index.
        local_idx: Linear voxel index within block [0, 511].

    Returns:
        vec2 with (sdf_weight, weight).
    """
    sdf_w = wp.float32(block_data[pool_idx, local_idx, 0])
    w = wp.float32(block_data[pool_idx, local_idx, 1])
    return wp.vec2(sdf_w, w)


@wp.func
def write_tsdf_voxel(
    block_data: wp.array3d(dtype=wp.float16),
    pool_idx: wp.int32,
    local_idx: wp.int32,
    sdf_weight: wp.float32,
    weight: wp.float32,
):
    """Write TSDF voxel data.

    Args:
        block_data: Block data array (3D: max_blocks, 512, 2).
        pool_idx: Block pool index.
        local_idx: Linear voxel index within block.
        sdf_weight: Accumulated sdf * weight.
        weight: Accumulated weight.
    """
    block_data[pool_idx, local_idx, 0] = wp.float16(sdf_weight)
    block_data[pool_idx, local_idx, 1] = wp.float16(weight)


# =============================================================================
# Block Clearing Kernel
# =============================================================================


@wp.kernel
def clear_new_blocks_kernel(
    block_data: wp.array3d(dtype=wp.float16),
    block_rgb: wp.array2d(dtype=wp.float32),  # Per-block [R×w, G×w, B×w, W]
    new_blocks: wp.array(dtype=wp.int32),
    new_block_count: wp.array(dtype=wp.int32),
    max_blocks: wp.int32,
):
    """Clear newly allocated blocks.

    Launch with dim = new_block_count * 512.
    """
    tid = wp.tid()
    slot_idx = tid // 512
    local_idx = tid % 512

    count = new_block_count[0]
    if count > max_blocks:
        count = max_blocks
    if slot_idx >= count:
        return

    pool_idx = new_blocks[slot_idx]

    # Clear TSDF data - 3D indexing: [pool_idx, local_idx, channel]
    block_data[pool_idx, local_idx, 0] = wp.float16(0.0)
    block_data[pool_idx, local_idx, 1] = wp.float16(0.0)

    # Clear per-block RGBW (only first thread per block)
    if local_idx == 0:
        block_rgb[pool_idx, 0] = 0.0
        block_rgb[pool_idx, 1] = 0.0
        block_rgb[pool_idx, 2] = 0.0
        block_rgb[pool_idx, 3] = 0.0


# =============================================================================
# Legacy Compatibility (for existing code)
# =============================================================================

# Keep old constants for any code that might reference them
HASH_EMPTY = ENTRY_EMPTY
HASH_TOMBSTONE = ENTRY_TOMBSTONE
BLOCK_KEY_OFFSET = COORD_OFFSET
BLOCK_KEY_MASK = COORD_MASK


@wp.func
def pack_block_key(bx: wp.int32, by: wp.int32, bz: wp.int32) -> wp.int64:
    """Legacy: Pack block key only (for compatibility)."""
    return pack_key_only(bx, by, bz)


@wp.func
def unpack_block_key(key: wp.int64) -> wp.vec3i:
    """Legacy: Unpack block key (for compatibility)."""
    result = unpack_entry(key)
    return wp.vec3i(result[0], result[1], result[2])
