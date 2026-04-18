# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unified SDF sampling functions for block-sparse TSDF.

This module provides helper functions for sampling the combined SDF from
dynamic (depth) and static (primitive) channels. All TSDF consumers should
use these functions to ensure consistent behavior.

SDF Composition:
    final_sdf = min(dynamic_sdf, static_sdf)

This produces the union of obstacles (conservative for collision avoidance).
"""

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp

# Large value representing "no obstacle" or "unobserved"
SDF_INFINITY = wp.constant(wp.float32(1e10))


@wp.func
def sample_dynamic_sdf(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
    min_weight: wp.float32,
) -> wp.float32:
    """Sample SDF from dynamic (depth) channel only.

    Args:
        tsdf: Block-sparse TSDF struct.
        pool_idx: Block pool index.
        local_idx: Local voxel index within block (0-511).
        min_weight: Minimum weight threshold for valid observation.

    Returns:
        SDF value in meters, or +inf if unobserved/disabled.
    """
    if not tsdf.has_dynamic:
        return SDF_INFINITY

    sdf_weight = wp.float32(tsdf.block_data[pool_idx, local_idx, 0])
    weight = wp.float32(tsdf.block_data[pool_idx, local_idx, 1])

    if weight > min_weight:
        return sdf_weight / weight

    return SDF_INFINITY


@wp.func
def sample_static_sdf(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.float32:
    """Sample SDF from static (primitive) channel only.

    Args:
        tsdf: Block-sparse TSDF struct.
        pool_idx: Block pool index.
        local_idx: Local voxel index within block (0-511).

    Returns:
        SDF value in meters, or +inf if no primitive.
    """
    if not tsdf.has_static:
        return SDF_INFINITY

    return wp.float32(tsdf.static_block_data[pool_idx, local_idx])


@wp.func
def sample_combined_sdf(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
    min_weight: wp.float32,
) -> wp.float32:
    """Sample combined SDF from dynamic and static channels.

    Uses min() to combine channels, producing the union of all obstacles.
    This is conservative for collision avoidance: any blocking surface wins.

    Args:
        tsdf: Block-sparse TSDF struct.
        pool_idx: Block pool index.
        local_idx: Local voxel index within block (0-511).
        min_weight: Minimum weight threshold for dynamic observations.

    Returns:
        Combined SDF value in meters (min of both channels).
    """
    dynamic_sdf = sample_dynamic_sdf(tsdf, pool_idx, local_idx, min_weight)
    static_sdf = sample_static_sdf(tsdf, pool_idx, local_idx)

    return wp.min(dynamic_sdf, static_sdf)


@wp.func
def has_valid_observation(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    local_idx: wp.int32,
    min_weight: wp.float32,
) -> wp.bool:
    """Check if voxel has any valid observation (dynamic or static).

    Args:
        tsdf: Block-sparse TSDF struct.
        pool_idx: Block pool index.
        local_idx: Local voxel index within block (0-511).
        min_weight: Minimum weight threshold for dynamic observations.

    Returns:
        True if either channel has valid data.
    """
    # Check dynamic
    if tsdf.has_dynamic:
        weight = wp.float32(tsdf.block_data[pool_idx, local_idx, 1])
        if weight > min_weight:
            return True

    # Check static
    if tsdf.has_static:
        static_sdf = wp.float32(tsdf.static_block_data[pool_idx, local_idx])
        if static_sdf < SDF_INFINITY:
            return True

    return False
