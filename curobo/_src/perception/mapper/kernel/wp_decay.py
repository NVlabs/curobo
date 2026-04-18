# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp kernels for block-sparse TSDF weight decay and block recycling.

This module provides GPU kernels for:
1. Decaying TSDF weights over time
2. Detecting and recycling empty blocks
3. Managing block sums for efficient empty detection

The decay mechanism enables tracking dynamic scenes by gradually forgetting
old observations while maintaining recent ones.
"""

import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.perception.mapper.kernel.warp_types import (
    HASH_TOMBSTONE,
)
from curobo._src.perception.mapper.kernel.wp_hash import (
    free_list_push,
)
from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    quat_from_wxyz_array,
    vec3_from_array,
)
from curobo._src.util.warp import get_warp_device_stream

# =============================================================================
# Decay Constants
# =============================================================================

# Threshold for considering a block empty (sum of weights across all voxels)
BLOCK_EMPTY_THRESHOLD = wp.constant(0.01)

# =============================================================================
# Block-Level Frustum Marking Kernel (Pass 1)
# =============================================================================


@wp.kernel
def mark_blocks_in_frustum_kernel(
    # Block data
    block_coords: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    num_allocated: wp.array(dtype=wp.int32),
    # Grid parameters
    origin: wp.array(dtype=wp.float32),
    voxel_size: float,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    # Batched camera parameters
    intrinsics: wp.array3d(dtype=wp.float32),
    cam_positions: wp.array2d(dtype=wp.float32),
    cam_quaternions: wp.array2d(dtype=wp.float32),
    n_cameras: wp.int32,
    img_H: wp.int32,
    img_W: wp.int32,
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    # Output
    block_in_frustum: wp.array(dtype=wp.int32),
    max_blocks: wp.int32,
):
    """Mark blocks visible in ANY camera's frustum.

    Parallelized across (block, camera) pairs.  Multiple threads may write
    ``1`` to the same ``block_in_frustum[block_idx]``. This is safe
    because the write is idempotent (only ``1`` is ever written; ``0`` is
    the pre-cleared default).

    Launch with ``dim = max_blocks * n_cameras``.  The flags array must
    be zeroed before launch.
    """
    tid = wp.tid()
    block_idx = tid // n_cameras
    cam_i = tid % n_cameras

    if block_idx >= max_blocks:
        return
    if block_idx >= num_allocated[0]:
        return
    if block_to_hash_slot[block_idx] < 0:
        return

    bx = block_coords[block_idx * 3 + 0]
    by = block_coords[block_idx * 3 + 1]
    bz = block_coords[block_idx * 3 + 2]

    half_block = wp.float32(block_size) * 0.5
    gx = wp.float32(bx * block_size) + half_block
    gy = wp.float32(by * block_size) + half_block
    gz = wp.float32(bz * block_size) + half_block

    if grid_W > 0:
        gx = gx - wp.float32(grid_W) * 0.5
        gy = gy - wp.float32(grid_H) * 0.5
        gz = gz - wp.float32(grid_D) * 0.5

    block_center_x = origin[0] + gx * voxel_size
    block_center_y = origin[1] + gy * voxel_size
    block_center_z = origin[2] + gz * voxel_size

    block_extent = wp.float32(block_size) * voxel_size
    sphere_radius = 0.866 * block_extent

    cam_pos = wp.vec3(
        cam_positions[cam_i, 0],
        cam_positions[cam_i, 1],
        cam_positions[cam_i, 2],
    )
    cam_quat = wp.quaternion(
        cam_quaternions[cam_i, 1],
        cam_quaternions[cam_i, 2],
        cam_quaternions[cam_i, 3],
        cam_quaternions[cam_i, 0],
    )
    block_world = wp.vec3(block_center_x, block_center_y, block_center_z)
    v_rel = block_world - cam_pos
    block_cam = wp.quat_rotate(wp.quat_inverse(cam_quat), v_rel)

    z_cam = block_cam[2]
    if z_cam + sphere_radius < depth_minimum_distance:
        return
    if z_cam - sphere_radius > depth_maximum_distance:
        return

    if z_cam < 0.01:
        block_in_frustum[block_idx] = 1
        return

    fx = intrinsics[cam_i, 0, 0]
    fy = intrinsics[cam_i, 1, 1]
    cx = intrinsics[cam_i, 0, 2]
    cy = intrinsics[cam_i, 1, 2]

    u = fx * block_cam[0] / z_cam + cx
    v = fy * block_cam[1] / z_cam + cy

    pixel_radius_x = fx * sphere_radius / z_cam
    pixel_radius_y = fy * sphere_radius / z_cam

    if u + pixel_radius_x < 0.0:
        return
    if u - pixel_radius_x > wp.float32(img_W):
        return
    if v + pixel_radius_y < 0.0:
        return
    if v - pixel_radius_y > wp.float32(img_H):
        return

    block_in_frustum[block_idx] = 1


# =============================================================================
# Block Recycling Kernel
# =============================================================================


@wp.kernel
def recycle_empty_blocks_kernel(
    block_sums: wp.array(dtype=wp.float32),
    static_block_sums: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    hash_table: wp.array(dtype=wp.int64),
    free_list: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    num_allocated: wp.array(dtype=wp.int32),
    max_blocks: wp.int32,
    recycle_count: wp.array(dtype=wp.int32),
):
    """Recycle blocks with no data in either channel.

    CUDA graph safe: Launch with fixed dim = max_blocks.
    Early-exits for threads beyond num_allocated.

    Blocks are recycled only if BOTH:
    - Dynamic weight sum < threshold (no depth observations)
    - Static voxel count == 0 (no primitive SDF)

    Args:
        block_sums: Dynamic channel weight sums from decay kernel.
        static_block_sums: Static channel voxel counts.
        block_to_hash_slot: Reverse mapping.
        hash_table: Packed hash table (key+value).
        free_list: Free list stack.
        free_count: Free list size.
        num_allocated: High-water mark.
        max_blocks: Maximum blocks (for bounds check, enables CUDA graph safety).
        recycle_count: Output counter for recycled blocks.
    """
    tid = wp.tid()

    # Early exit for fixed launch dim (CUDA graph safe)
    if tid >= max_blocks:
        return
    if tid >= num_allocated[0]:
        return

    # Skip already freed blocks
    hash_slot = block_to_hash_slot[tid]
    if hash_slot < 0:
        return

    # Check if block is empty in BOTH channels
    # Keep block if it has dynamic data OR static data
    if block_sums[tid] >= BLOCK_EMPTY_THRESHOLD:
        return  # Has dynamic data, keep it
    if static_block_sums[tid] > 0:
        return  # Has static data, keep it

    # Block is empty in both channels - recycle it

    # 1. Mark hash slot as tombstone
    hash_table[hash_slot] = HASH_TOMBSTONE

    # 2. Mark block as freed
    block_to_hash_slot[tid] = wp.int32(-1)

    # 3. Push to free list
    free_list_push(free_list, free_count, tid, max_blocks)

    # 4. Count recycled blocks
    wp.atomic_add(recycle_count, 0, wp.int32(1))


# =============================================================================
# Public API
# =============================================================================


def decay_and_recycle(
    tsdf,  # BlockSparseTSDF instance
    decay_factor: float = 0.95,
) -> int:
    """Decay weights and recycle empty blocks.

    NOT CUDA graph safe (returns count, requires sync).
    Call this periodically OUTSIDE of CUDA graphs to:
    1. Decay all voxel weights by decay_factor
    2. Recycle blocks whose total weight falls below threshold

    Args:
        tsdf: BlockSparseTSDF instance.
        decay_factor: Weight multiplier per call (0.95 = 5% decay).

    Returns:
        Number of blocks recycled.
    """
    max_blocks = tsdf.config.max_blocks

    # Decay weights via PyTorch (zero atomics)
    if decay_factor < 1.0:
        tsdf.data.block_data[:max_blocks].mul_(decay_factor)
        tsdf.data.block_rgb[:max_blocks].mul_(decay_factor)
    tsdf.data.block_sums[:max_blocks] = (
        tsdf.data.block_data[:max_blocks, :, 1].sum(dim=1, dtype=torch.float32)
    )

    # Reset recycle count
    tsdf.data.recycle_count.zero_()

    # Recycle empty blocks - FIXED launch dim
    data = tsdf.get_warp_data()
    device, stream = get_warp_device_stream(tsdf.data.block_data)
    wp.launch(
        recycle_empty_blocks_kernel,
        dim=max_blocks,
        inputs=[
            data.block_sums,
            data.static_block_sums,
            data.block_to_hash_slot,
            data.hash_table,
            data.free_list,
            data.free_count,
            data.num_allocated,
            max_blocks,
            wp.from_torch(tsdf.data.recycle_count, dtype=wp.int32),
        ],
        device=device,
        stream=stream,
    )

    # Sync to read count (NOT in CUDA graph)
    return tsdf.data.recycle_count


def recycle_graph_safe(tsdf, num_blocks: int = None):
    """Recycle empty blocks without returning count.

    CUDA graph safe when num_blocks is None (fixed launch dim).
    When num_blocks is provided, launches with data-dependent dim.

    Note: Must call decay first to populate block_sums.

    Args:
        tsdf: BlockSparseTSDF instance.
        num_blocks: If provided, use as launch dim instead of max_blocks.
    """
    data = tsdf.get_warp_data()
    launch_blocks = num_blocks if num_blocks is not None else tsdf.config.max_blocks

    # Reset recycle count (even though we won't read it in graph)
    tsdf.data.recycle_count.zero_()
    device, stream = get_warp_device_stream(tsdf.data.block_data)
    wp.launch(
        recycle_empty_blocks_kernel,
        dim=launch_blocks,
        inputs=[
            data.block_sums,
            data.static_block_sums,
            data.block_to_hash_slot,
            data.hash_table,
            data.free_list,
            data.free_count,
            data.num_allocated,
            tsdf.config.max_blocks,
            wp.from_torch(tsdf.data.recycle_count, dtype=wp.int32),
        ],
        device=device,
        stream=stream,
    )


# =============================================================================
# Frustum-Aware Decay API
# =============================================================================


def decay_frustum_aware_multi_camera(
    tsdf,
    intrinsics: torch.Tensor,
    cam_positions: torch.Tensor,
    cam_quaternions: torch.Tensor,
    img_shape: tuple,
    depth_minimum_distance: float = 0.1,
    depth_maximum_distance: float = 10.0,
    time_decay: float = 1.0,
    frustum_decay: float = 0.5,
    num_blocks: int = None,
):
    """Frustum-aware decay for multiple cameras + block recycling.

    A block is marked as "in frustum" if it is visible in ANY camera.
    Decay is applied once using the union frustum, then empty blocks are
    recycled.

    Args:
        tsdf: BlockSparseTSDF instance.
        intrinsics: Camera intrinsics ``(num_cameras, 3, 3)`` float32.
        cam_positions: Camera positions ``(num_cameras, 3)`` float32.
        cam_quaternions: Camera quaternions ``(num_cameras, 4)`` wxyz float32.
        img_shape: Image dimensions ``(H, W)`` (shared across cameras).
        depth_minimum_distance: Minimum observable depth [m].
        depth_maximum_distance: Maximum observable depth [m].
        time_decay: Decay for all voxels.
        frustum_decay: Extra decay for in-view blocks.
        num_blocks: If provided, use as slice size instead of ``max_blocks``.
    """
    max_blocks = tsdf.config.max_blocks
    n = num_blocks if num_blocks is not None else max_blocks
    n_cameras = intrinsics.shape[0]

    if frustum_decay >= 1.0:
        block_data = tsdf.data.block_data[:n]
        if time_decay < 1.0:
            block_data.mul_(time_decay)
            tsdf.data.block_rgb[:n].mul_(time_decay)
        tsdf.data.block_sums[:n] = block_data[:, :, 1].sum(
            dim=1, dtype=torch.float32
        )
        recycle_graph_safe(tsdf, num_blocks=num_blocks)
        return

    data = tsdf.get_warp_data()
    device, stream = get_warp_device_stream(tsdf.data.block_data)

    img_H, img_W = img_shape

    if tsdf.config.grid_shape is not None:
        grid_D, grid_H_dim, grid_W_dim = tsdf.config.grid_shape
    else:
        grid_W_dim, grid_H_dim, grid_D = 0, 0, 0

    frustum_flags = tsdf.data.frustum_flags
    frustum_flags.zero_()

    check_float32_tensors(
        intrinsics.device,
        intrinsics=intrinsics,
        cam_positions=cam_positions,
        cam_quaternions=cam_quaternions,
    )
    wp.launch(
        mark_blocks_in_frustum_kernel,
        dim=max_blocks * n_cameras,
        inputs=[
            data.block_coords,
            data.block_to_hash_slot,
            data.num_allocated,
            wp.from_torch(tsdf.config.origin, dtype=wp.float32),
            tsdf.config.voxel_size,
            tsdf.config.block_size,
            grid_W_dim,
            grid_H_dim,
            grid_D,
            wp.from_torch(intrinsics, dtype=wp.float32),
            wp.from_torch(cam_positions, dtype=wp.float32),
            wp.from_torch(cam_quaternions, dtype=wp.float32),
            n_cameras,
            img_H,
            img_W,
            depth_minimum_distance,
            depth_maximum_distance,
            wp.from_torch(frustum_flags, dtype=wp.int32),
            max_blocks,
        ],
        device=device,
        stream=stream,
    )

    factor = tsdf.data.decay_factor[:n]
    factor.fill_(time_decay)
    factor.masked_fill_(frustum_flags[:n] > 0, time_decay * frustum_decay)

    tsdf.data.block_data[:n].mul_(factor.view(n, 1, 1))
    tsdf.data.block_rgb[:n].mul_(factor.view(n, 1))

    tsdf.data.block_sums[:n] = tsdf.data.block_data[:n, :, 1].sum(
        dim=1, dtype=torch.float32
    )

    recycle_graph_safe(tsdf, num_blocks=num_blocks)



