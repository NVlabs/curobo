# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp struct definitions for block-sparse TSDF kernel access.

This module defines the BlockSparseTSDFWarp struct which is passed to Warp
kernels. The corresponding Python dataclass (BlockSparseTSDFData) provides
a to_warp() method for conversion.

Hash Table Entry Layout (64 bits):
    ┌────────────────────────────────────────────────────┬──────────────┐
    │              Block Key (42 bits)                   │ Value (22)   │
    │   X (14 bits)  |  Y (14 bits)  |  Z (14 bits)     │ pool_idx     │
    └────────────────────────────────────────────────────┴──────────────┘

    - Coordinates: ±8,192 per axis (14-bit signed range)
    - Pool index: 0 to 4,194,303 (22 bits = 4M blocks max)
    - Empty slot: -1 (all 1s)
    - Tombstone: -2

Range at different voxel sizes (block_size=8):
    - 1mm voxel: ±65m per axis
    - 2mm voxel: ±131m per axis
    - 4mm voxel: ±262m per axis
"""

import warp as wp


@wp.struct
class BlockSparseTSDFWarp:
    """Warp struct for block-sparse TSDF kernel access.

    This struct is created from BlockSparseTSDFData.to_warp() and should
    be cached for CUDA graph compatibility.

    Memory Layout:
        hash_table: (capacity,) int64
            - Packed entries: key (42 bits) + pool_idx (22 bits)
            - -1 = empty slot, -2 = tombstone

        block_data: (max_blocks, 512, 2) float16
            - For each voxel: [sdf_weight, weight] (dynamic channel)
            - Access: block_data[pool_idx, local_idx, 0/1]
            - local_idx = lz * 64 + ly * 8 + lx

        static_block_data: (max_blocks, 512) float16
            - For each voxel: sdf value (static channel, primitives)
            - Access: static_block_data[pool_idx, local_idx]
            - Initialized to +inf (no obstacle)

        block_rgb: (max_blocks, 4) float32
            - Per-block RGBW: [R×w, G×w, B×w, weight_sum]
            - One color per block (not per voxel) - 512× memory savings
            - Divide by channel 3 at read time for averaged color
            - Access: block_rgb[pool_idx, 0/1/2/3]

        block_coords: (max_blocks * 3,) int32
            - Block world coordinates: [bx, by, bz] for each block
            - Access: bx = block_coords[pool_idx * 3 + 0]

    Note: 3D array is used for block_data to avoid exceeding Warp's
    INT32_MAX limit on array dimensions when using large max_blocks values.
    """

    # Hash table (packed key+value in single int64)
    hash_table: wp.array(dtype=wp.int64)  # (capacity,)

    # Block pool - dynamic channel (depth integration)
    block_data: wp.array3d(dtype=wp.float16)  # (max_blocks, 512, 2) or (1, 1, 2) dummy
    block_rgb: wp.array2d(dtype=wp.float32)  # (max_blocks, 4) per-block [R×w, G×w, B×w, W]

    # Block pool - static channel (primitive SDF)
    static_block_data: wp.array2d(dtype=wp.float16)  # (max_blocks, 512) or (1, 1) dummy

    # Feature flags
    has_dynamic: wp.bool  # True if dynamic channel is enabled
    has_static: wp.bool  # True if static channel is enabled

    # Block metadata
    block_coords: wp.array(dtype=wp.int32)  # (max_blocks * 3,) block world coords
    block_to_hash_slot: wp.array(dtype=wp.int32)  # (max_blocks,) reverse mapping

    # Free list (stack)
    free_list: wp.array(dtype=wp.int32)  # (max_blocks,) stack of free pool indices
    free_count: wp.array(dtype=wp.int32)  # (1,) current stack size

    # Counters
    num_allocated: wp.array(dtype=wp.int32)  # (1,) high-water mark
    allocation_failures: wp.array(dtype=wp.int32)  # (1,) failure counter

    # Block sums for decay (warp-reduced)
    block_sums: wp.array(dtype=wp.float32)  # (max_blocks,) - dynamic channel weights
    static_block_sums: wp.array(dtype=wp.int32)  # (max_blocks,) - static channel voxel count

    # Per-frame new block tracking (for clearing)
    new_blocks: wp.array(dtype=wp.int32)  # (max_blocks,)
    new_block_count: wp.array(dtype=wp.int32)  # (1,)

    # Recycle counter (pre-allocated for CUDA graph safety)
    recycle_count: wp.array(dtype=wp.int32)  # (1,)

    # Grid parameters (scalars)
    origin: wp.vec3  # World coordinate of grid origin
    voxel_size: float  # Size of each voxel in meters
    hash_capacity: int  # Hash table capacity
    max_blocks: int  # Maximum allocatable blocks
    truncation_distance: float  # TSDF truncation distance
    block_size: int  # Voxels per block edge (typically 8)
    # Grid shape for center-origin convention (0 = corner-origin)
    grid_W: int  # Grid width (nx)
    grid_H: int  # Grid height (ny)
    grid_D: int  # Grid depth (nz)


# =============================================================================
# Block Size Constant - Single source of truth
# =============================================================================

# Block size (voxels per edge). Must be 8 - kernels use hardcoded bit operations.
# Changing this requires updating all kernels that use BLOCK_SIZE, BLOCK_SIZE_SQ, etc.
BLOCK_SIZE: int = 8
BLOCK_SIZE_SQ: int = BLOCK_SIZE * BLOCK_SIZE  # 64
BLOCK_SIZE_CUBED: int = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE  # 512


# =============================================================================
# Hash Table Constants - Python values (source of truth)
# =============================================================================

# Entry values
PY_HASH_EMPTY: int = -1
PY_HASH_TOMBSTONE: int = -2

# Spatial hash primes (from Teschner et al.)
PY_HASH_PRIME_X: int = 73856093
PY_HASH_PRIME_Y: int = 19349663
PY_HASH_PRIME_Z: int = 83492791

# Value mask for pool_idx extraction (22 bits)
PY_VALUE_MASK: int = 0x3FFFFF

# Sign bit mask for ensuring positive hash (matches GPU behavior)
PY_POSITIVE_MASK: int = 0x7FFFFFFFFFFFFFFF

# Coordinate encoding
PY_COORD_BITS: int = 14
PY_COORD_OFFSET: int = 8192  # ±8192 range (14-bit signed)
PY_COORD_MASK: int = 0x3FFF  # 14 bits


# =============================================================================
# Hash Table Constants - Warp constants (derived from Python values)
# =============================================================================

# Entry values
HASH_EMPTY = wp.constant(wp.int64(PY_HASH_EMPTY))
HASH_TOMBSTONE = wp.constant(wp.int64(PY_HASH_TOMBSTONE))

# Spatial hash primes
HASH_PRIME_X = wp.constant(wp.int64(PY_HASH_PRIME_X))
HASH_PRIME_Y = wp.constant(wp.int64(PY_HASH_PRIME_Y))
HASH_PRIME_Z = wp.constant(wp.int64(PY_HASH_PRIME_Z))

# Coordinate encoding
BLOCK_KEY_BITS = wp.constant(PY_COORD_BITS)
BLOCK_KEY_OFFSET = wp.constant(wp.int64(PY_COORD_OFFSET))
BLOCK_KEY_MASK = wp.constant(wp.int64(PY_COORD_MASK))
