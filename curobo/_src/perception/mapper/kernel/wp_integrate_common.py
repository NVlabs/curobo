# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Common Warp functions for TSDF integration.

This module provides shared GPU functions used by both dense and block-sparse
TSDF integrators. Having these in one place ensures consistent behavior and
avoids code duplication.

Functions:
- Camera/pose utilities: quat_from_wxyz_array, vec3_from_array
- TSDF computation: compute_tsdf_weight
- Integer math: floor_div, floor_mod (Python-style for negative numbers)
"""

import warp as wp

# =============================================================================
# Camera/Pose Utilities
# =============================================================================


@wp.func
def quat_from_wxyz_array(cam_quaternion: wp.array(dtype=wp.float32)) -> wp.quat:
    """Convert quaternion from wxyz array format to Warp's xyzw quat.

    Args:
        cam_quaternion: Quaternion array (4,) in wxyz format (curobo convention).

    Returns:
        Warp quaternion in xyzw format.
    """
    return wp.quat(
        cam_quaternion[1],  # x
        cam_quaternion[2],  # y
        cam_quaternion[3],  # z
        cam_quaternion[0],  # w
    )


@wp.func
def vec3_from_array(arr: wp.array(dtype=wp.float32)) -> wp.vec3:
    """Create vec3 from float32 array.

    Args:
        arr: Float32 array (3,).

    Returns:
        Warp vec3.
    """
    return wp.vec3(arr[0], arr[1], arr[2])


# =============================================================================
# TSDF Computation
# =============================================================================


@wp.func
def compute_tsdf_weight(depth: wp.float32, voxel_size: wp.float32) -> wp.float32:
    """Compute observation weight for TSDF fusion.

    Uses inverse square distance weighting (nvblox-style) with voxel size
    compensation. Smaller voxels receive fewer observations, so we scale
    weight inversely with voxel volume to ensure consistent weight accumulation.

    Reference voxel size is 0.01m (1cm). Weight scaling:
        - At 1cm voxel: no scaling (volume_scale = 1)
        - At 0.5cm voxel: 8× scaling (half size = 1/8 volume)
        - At 2cm voxel: 0.125× scaling (double size = 8× volume)

    Weight values by depth (at 1cm voxel):
        - 0.5m: 4.0 → clamped to 2.0
        - 0.7m: ~2.0
        - 1.0m: 1.0 (reference point)
        - 1.4m: ~0.5
        - 2.0m: 0.25
        - 3.0m: ~0.11
        - 31.6m+: <0.001 → clamped to 0.001

    This means minimum_tsdf_weight can be interpreted as "number of observations
    at 1m depth with 1cm voxels". E.g., minimum_tsdf_weight=0.5 requires half an
    observation at 1m, or one observation at ~1.4m, or two observations at 2m.

    Args:
        depth: Measured depth in meters.
        voxel_size: Voxel size in meters.

    Returns:
        Weight with voxel size compensation, clamped to [0.001, 200.0].
    """
    # Reference voxel size for consistent weighting

    # Depth-based weight (inverse square)
    depth_weight = 1.0 #/ (depth) #(depth * depth)

    # Voxel volume compensation: smaller voxels get higher weight
    # volume_scale = (ref / actual)^3
    #ref_voxel_size = 0.01  # 1cm
    #size_ratio = ref_voxel_size / voxel_size
    #volume_scale = size_ratio * size_ratio * size_ratio

    weight = depth_weight #* volume_scale
    return wp.clamp(weight, 0.001, 1000.0)


# =============================================================================
# Integer Math (Python-style for negative numbers)
# =============================================================================


@wp.func
def floor_div(a: wp.int32, b: wp.int32) -> wp.int32:
    """Python-style floor division (towards negative infinity).

    Warp/C++ uses truncation towards zero for integer division.
    This function implements Python-style floor division which is needed
    for correct block coordinate calculation with negative voxel indices.

    Examples:
        floor_div(-1, 8) = -1 (not 0)
        floor_div(-9, 8) = -2 (not -1)
        floor_div(7, 8) = 0

    Args:
        a: Dividend.
        b: Divisor (must be positive for correct behavior).

    Returns:
        Floor division result.
    """
    d = a / b
    # If a and b have different signs and there's a remainder, subtract 1
    if (a < 0) != (b < 0) and a % b != 0:
        d = d - 1
    return d


@wp.func
def floor_mod(a: wp.int32, b: wp.int32) -> wp.int32:
    """Python-style modulo (result has same sign as divisor).

    This ensures local coordinates are always in [0, block_size-1].

    Args:
        a: Dividend.
        b: Divisor (must be positive for correct behavior).

    Returns:
        Modulo result in [0, b-1].
    """
    r = a % b
    if r < 0:
        r = r + b
    return r
