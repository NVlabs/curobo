# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp struct definition for block-sparse TSDF kernel access.

This module holds the :class:`BlockSparseTSDFWarp` dataclass that is
passed to Warp kernels. Shared constants (``BLOCK_SIZE``, hash-table
bit layout, pool-index sentinels) live in
:mod:`curobo._src.perception.mapper.constants`.

Memory layout and key ranges of the block-sparse TSDF are documented
on :class:`BlockSparseTSDFWarp` below.
"""

import warp as wp


@wp.struct
class BlockSparseTSDFWarp:
    """Warp struct for block-sparse TSDF kernel access.

    This struct is created from ``BlockSparseTSDFData.to_warp()`` and
    should be cached for CUDA graph compatibility.

    Memory Layout:
        hash_table: (capacity,) int64
            Packed entries: key (39 bits) + pool_idx (25 bits).
            See :mod:`curobo._src.perception.mapper.constants` for the
            bit layout. -1 = empty slot, -2 = tombstone.

        block_data: (max_blocks, BLOCK_SIZE**3, 2) float16
            For each voxel: [sdf_weight, weight] (dynamic channel).
            Access: ``block_data[pool_idx, local_idx, 0/1]``.
            ``local_idx = lz * BLOCK_SIZE**2 + ly * BLOCK_SIZE + lx``.

        static_block_data: (max_blocks, BLOCK_SIZE**3) float16
            For each voxel: sdf value (static channel, primitives).
            Access: ``static_block_data[pool_idx, local_idx]``.
            Initialized to +inf (no obstacle).

        block_rgb: (max_blocks, 4) float16
            Per-block RGBW: [R*w, G*w, B*w, weight_sum]. One color per
            block (not per voxel) - ``BLOCK_SIZE**3`` memory savings.
            Integration pre-normalizes RGB to ``[0, 1]``; divide by
            channel 3 at read time and multiply by 255 to get uint8 RGB.
            A post-frame rescale kernel caps the weight at
            ``config.accumulator_w_max`` so the fp16 accumulator stays
            inside finite range and per-atomic ulp loss stays bounded.
            Access: ``block_rgb[pool_idx, 0/1/2/3]``.

        block_coords: (max_blocks * 3,) int32
            Signed centered hash block keys: [bx, by, bz] for each block.
            Access: ``bx = block_coords[pool_idx * 3 + 0]``.

    Note: a 3D array is used for ``block_data`` to avoid exceeding
    Warp's ``INT32_MAX`` limit on array dimensions when using large
    ``max_blocks`` values.

    Default key ranges:
        - Block coords: [-4096, 4095] per axis (13-bit signed range)
        - Pool index: 0 to 33,554,430 (25 bits minus PENDING sentinel)
    """

    # Hash table (packed key+value in single int64)
    hash_table: wp.array(dtype=wp.int64)  # (capacity,)

    # Block pool - dynamic channel (depth integration)
    block_data: wp.array3d(dtype=wp.float16)  # (max_blocks, BLOCK_SIZE**3, 2) or (1, 1, 2) dummy
    block_rgb: wp.array2d(dtype=wp.float16)  # (max_blocks, 4) per-block [R*w, G*w, B*w, W]

    # Per-block feature channel (fp16 weighted sums + dedicated weight)
    block_features: wp.array2d(dtype=wp.float16)  # (max_blocks, feature_dim) or (1, 1) dummy
    block_feature_weight: wp.array(dtype=wp.float16)  # (max_blocks,) or (1,) dummy

    # Block pool - static channel (primitive SDF)
    static_block_data: wp.array2d(dtype=wp.float16)  # (max_blocks, BLOCK_SIZE**3) or (1, 1) dummy

    # Feature flags
    has_dynamic: wp.bool  # True if dynamic channel is enabled
    has_static: wp.bool  # True if static channel is enabled
    has_features: wp.bool  # True if per-block feature channel is enabled
    feature_dim: int  # 0 when the feature channel is disabled

    # Block metadata
    block_coords: wp.array(dtype=wp.int32)  # (max_blocks * 3,) signed block keys
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
    # Bounded grid shape for center-origin convention.
    grid_W: int  # Grid width (nx)
    grid_H: int  # Grid height (ny)
    grid_D: int  # Grid depth (nz)
