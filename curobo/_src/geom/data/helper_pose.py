# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Warp helper functions for pose manipulation in obstacle data.

These functions are shared across all obstacle types (cuboid, mesh, voxel)
for indexing and pose loading from obstacle data structs.
"""

import warp as wp


@wp.func
def get_obs_idx(env_idx: wp.int32, local_idx: wp.int32, max_n: wp.int32) -> wp.int32:
    """Compute flat obstacle index from environment and local indices.

    Args:
        env_idx: Environment index.
        local_idx: Local obstacle index within the environment.
        max_n: Maximum obstacles per environment.

    Returns:
        Flat index into obstacle arrays: env_idx * max_n + local_idx.
    """
    return env_idx * max_n + local_idx


@wp.func
def load_inv_position(inv_pose: wp.array2d(dtype=wp.float32), idx: wp.int32) -> wp.vec3:
    """Load inverse position from inverse pose array.

    Args:
        inv_pose: Inverse pose array with layout [x, y, z, qw, qx, qy, qz, pad].
        idx: Flat index into the pose array.

    Returns:
        Inverse position as vec3.
    """
    return wp.vec3(inv_pose[idx, 0], inv_pose[idx, 1], inv_pose[idx, 2])


@wp.func
def load_inv_quat(inv_pose: wp.array2d(dtype=wp.float32), idx: wp.int32) -> wp.quat:
    """Load inverse quaternion from inverse pose array.

    Args:
        inv_pose: Inverse pose array with layout [x, y, z, qw, qx, qy, qz, pad].
        idx: Flat index into the pose array.

    Returns:
        Inverse quaternion. Note: Warp quat constructor is (x, y, z, w).
    """
    return wp.quat(inv_pose[idx, 4], inv_pose[idx, 5], inv_pose[idx, 6], inv_pose[idx, 3])


@wp.func
def get_forward_quat(inv_quat: wp.quat) -> wp.quat:
    """Compute forward quaternion from inverse quaternion.

    Args:
        inv_quat: Inverse quaternion.

    Returns:
        Forward quaternion (conjugate of inverse).
    """
    return wp.quat_inverse(inv_quat)


@wp.func
def load_transform_from_inv_pose(
    inv_pose: wp.array2d(dtype=wp.float32),
    flat_idx: wp.int32,
) -> wp.transform:
    """Load world-to-local transform from inverse pose array.

    This function creates a wp.transform from the stored inverse pose data.
    The returned transform can be used with wp.transform_point() to convert
    world-frame points to local-frame, and wp.transform_inverse() followed
    by wp.transform_vector() to convert local-frame gradients to world-frame.

    Args:
        inv_pose: Inverse pose array with layout [x, y, z, qw, qx, qy, qz, pad].
        flat_idx: Flat index into the pose array.

    Returns:
        wp.transform representing the world-to-local transformation.
    """
    inv_pos = load_inv_position(inv_pose, flat_idx)
    inv_quat = load_inv_quat(inv_pose, flat_idx)
    return wp.transform(inv_pos, inv_quat)
