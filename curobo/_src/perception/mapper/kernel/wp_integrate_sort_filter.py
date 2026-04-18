# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Sort & Filter TSDF Integration - CUDA Graph Safe, Zero Contention.

This module implements a 4-phase integration strategy that eliminates the
"Thundering Herd" problem in block allocation:

    Phase 1: Compute block keys (N threads, zero contention)
    Phase 2: Sort keys (torch.sort, graph-safe)
    Phase 3: Allocate unique blocks (N threads, ~K allocate, zero contention)
    Phase 4: Integrate samples (N threads, lookup only)

Key Insight:
    Instead of using torch.unique (which has variable output size and breaks
    CUDA graphs), we use torch.sort (fixed N→N output) and boundary detection.
    Only the first thread of each unique key performs allocation.

Performance:
    - Coalesced reads from sorted array
    - Minimal warp divergence (sorted data)
    - Zero contention on allocation
    - CUDA graph compatible (all shapes static)
"""

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.wp_coord import (
    voxel_to_world,
    world_to_continuous_voxel,
)
from curobo._src.perception.mapper.kernel.wp_hash import (
    ENTRY_EMPTY,
    ENTRY_TOMBSTONE,
    free_list_pop,
    get_key_part,
    get_pool_idx,
    hash_lookup,
    hash_table_insert_with_pool_idx,
    pack_key_only,
    spatial_hash,
    unpack_block_key,
)
from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    compute_tsdf_weight,
    floor_div,
    floor_mod,
    quat_from_wxyz_array,
    vec3_from_array,
)
from curobo._src.util.warp import get_warp_device_stream



# =============================================================================
# Phase 3: Allocate Unique Blocks (Zero Contention)
# =============================================================================


@wp.kernel
def allocate_unique_blocks_kernel(
    # Sorted keys
    sorted_keys: wp.array(dtype=wp.int64),
    n_keys: wp.int32,
    # Hash table
    hash_table: wp.array(dtype=wp.int64),
    hash_capacity: wp.int32,
    # Block metadata
    block_coords: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    # Free list for recycled blocks
    free_list: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    # Counters
    num_allocated: wp.array(dtype=wp.int32),
    max_blocks: wp.int32,
    # Per-frame tracking
    new_blocks: wp.array(dtype=wp.int32),
    new_block_count: wp.array(dtype=wp.int32),
):
    """Phase 3: Allocate unique blocks using CAS for collision handling.

    After sorting, only the first occurrence of each key performs allocation.
    Uses free list for recycled blocks before allocating new ones.
    Uses shared hash_table_insert_with_pool_idx for safe CAS-based insertion.
    """
    tid = wp.tid()

    if tid >= n_keys:
        return

    key = sorted_keys[tid]

    # Skip invalid keys
    if key == wp.int64(-1):
        return

    # Skip duplicates: only first occurrence proceeds
    if tid > 0:
        prev_key = sorted_keys[tid - 1]
        if key == prev_key:
            return

    # --- This thread is responsible for this unique block ---

    # Unpack coordinates
    coords = unpack_block_key(key)
    bx = coords[0]
    by = coords[1]
    bz = coords[2]

    # Check if block already exists in hash table
    existing_pool_idx = hash_lookup(hash_table, bx, by, bz, hash_capacity)
    if existing_pool_idx >= wp.int32(0):
        # Block already exists - nothing to do
        return

    # Try to get pool index from free list first (recycled blocks)
    pool_idx = free_list_pop(free_list, free_count)

    if pool_idx < wp.int32(0):
        # Free list empty - allocate new from pool
        pool_idx = wp.atomic_add(num_allocated, 0, wp.int32(1))

        if pool_idx >= max_blocks:
            # Pool exhausted - revert and fail
            wp.atomic_add(num_allocated, 0, wp.int32(-1))
            return

    # Insert into hash table (handles CAS loop internally)
    # Result may differ from pool_idx if key already existed
    hash_table_insert_with_pool_idx(
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


# =============================================================================
# Integration Kernels
# =============================================================================


@wp.kernel
def compute_block_keys_kernel(
    # Per-camera parameters (batched)
    intrinsics: wp.array3d(dtype=wp.float32),
    cam_positions: wp.array2d(dtype=wp.float32),
    cam_quaternions: wp.array2d(dtype=wp.float32),
    depth_images: wp.array3d(dtype=wp.float32),
    # Grid parameters
    origin: wp.array(dtype=wp.float32),
    voxel_size: float,
    truncation_dist: float,
    depth_min: float,
    depth_max: float,
    block_size: wp.int32,
    num_samples: wp.int32,
    img_H: wp.int32,
    img_W: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    n_cameras: wp.int32,
    # Output
    block_keys: wp.array(dtype=wp.int64),
    sample_data: wp.array2d(dtype=wp.float32),
):
    """Phase 1 for multiple cameras.

    Launch with dim = n_cameras * n_pixels * num_samples.
    cam_idx is recoverable from tid in Phase 4 via
    ``cam_idx = orig_idx // (n_pixels * num_samples)``.
    """
    tid = wp.tid()

    n_pixels = img_H * img_W
    samples_per_cam = n_pixels * num_samples
    cam_idx = tid // samples_per_cam
    remainder = tid % samples_per_cam
    pixel_idx = remainder // num_samples
    sample_idx = remainder % num_samples

    if cam_idx >= n_cameras or pixel_idx >= n_pixels:
        block_keys[tid] = wp.int64(-1)
        return

    px = pixel_idx % img_W
    py = pixel_idx // img_W

    fx = intrinsics[cam_idx, 0, 0]
    fy = intrinsics[cam_idx, 1, 1]
    cx = intrinsics[cam_idx, 0, 2]
    cy = intrinsics[cam_idx, 1, 2]

    depth = depth_images[cam_idx, py, px]

    if depth < depth_min or depth > depth_max:
        block_keys[tid] = wp.int64(-1)
        return

    u_norm = (wp.float32(px) + 0.5 - cx) / fx
    v_norm = (wp.float32(py) + 0.5 - cy) / fy
    ray_dir = wp.vec3(u_norm, v_norm, 1.0)

    z_start = wp.max(depth - truncation_dist, depth_min)
    z_sample = z_start + wp.float32(sample_idx) * voxel_size

    if z_sample > depth + truncation_dist:
        block_keys[tid] = wp.int64(-1)
        return

    point_cam = ray_dir * z_sample

    cam_pos = wp.vec3(
        cam_positions[cam_idx, 0],
        cam_positions[cam_idx, 1],
        cam_positions[cam_idx, 2],
    )
    cam_quat = wp.quaternion(
        cam_quaternions[cam_idx, 1],
        cam_quaternions[cam_idx, 2],
        cam_quaternions[cam_idx, 3],
        cam_quaternions[cam_idx, 0],
    )
    point_world = cam_pos + wp.quat_rotate(cam_quat, point_cam)

    grid_origin = vec3_from_array(origin)
    voxel_f = world_to_continuous_voxel(
        point_world, grid_origin, voxel_size, grid_W, grid_H, grid_D,
    )

    vx = wp.int32(wp.floor(voxel_f[0]))
    vy = wp.int32(wp.floor(voxel_f[1]))
    vz = wp.int32(wp.floor(voxel_f[2]))

    if grid_W > 0:
        if vx < 0 or vx >= grid_W or vy < 0 or vy >= grid_H or vz < 0 or vz >= grid_D:
            block_keys[tid] = wp.int64(-1)
            return

    bx = floor_div(vx, block_size)
    by = floor_div(vy, block_size)
    bz = floor_div(vz, block_size)

    voxel_center = voxel_to_world(
        wp.vec3i(vx, vy, vz), grid_origin, voxel_size, grid_W, grid_H, grid_D,
    )
    v_rel = voxel_center - cam_pos
    cam_quat_inv = wp.quat_inverse(cam_quat)
    voxel_cam = wp.quat_rotate(cam_quat_inv, v_rel)
    z_cam = voxel_cam[2]

    sdf = depth - z_cam
    sdf_clamped = wp.clamp(sdf, -truncation_dist, truncation_dist)

    block_keys[tid] = pack_key_only(bx, by, bz)
    sample_data[tid, 0] = wp.float32(vx)
    sample_data[tid, 1] = wp.float32(vy)
    sample_data[tid, 2] = wp.float32(vz)
    sample_data[tid, 3] = sdf_clamped


@wp.kernel
def integrate_samples_kernel(
    sorted_keys: wp.array(dtype=wp.int64),
    sort_indices: wp.array(dtype=wp.int64),
    sample_data: wp.array2d(dtype=wp.float32),
    n_keys: wp.int32,
    # Multi-camera images
    depth_images: wp.array3d(dtype=wp.float32),
    # RGB flattened: (n_cameras * H, W, 3)
    rgb_images_flat: wp.array3d(dtype=wp.uint8),
    img_H: wp.int32,
    img_W: wp.int32,
    num_samples: wp.int32,
    samples_per_cam: wp.int32,
    voxel_size: float,
    # Hash table
    hash_table: wp.array(dtype=wp.int64),
    hash_capacity: wp.int32,
    # Block data
    block_data: wp.array3d(dtype=wp.float16),
    block_rgb: wp.array2d(dtype=wp.float32),
    block_size: wp.int32,
):
    """Phase 4 for multiple cameras.

    Derives cam_idx and pixel coordinates from the original thread index.
    """
    tid = wp.tid()

    if tid >= n_keys:
        return

    key = sorted_keys[tid]
    if key == wp.int64(-1):
        return

    orig_idx = wp.int32(sort_indices[tid])

    vx = wp.int32(sample_data[orig_idx, 0])
    vy = wp.int32(sample_data[orig_idx, 1])
    vz = wp.int32(sample_data[orig_idx, 2])
    sdf = sample_data[orig_idx, 3]

    # Derive camera and pixel from original index layout
    cam_idx = orig_idx // samples_per_cam
    within_cam = orig_idx % samples_per_cam
    pixel_idx = within_cam // num_samples
    px = pixel_idx % img_W
    py = pixel_idx // img_W

    depth = depth_images[cam_idx, py, px]
    weight = compute_tsdf_weight(depth, voxel_size)

    bx = floor_div(vx, block_size)
    by = floor_div(vy, block_size)
    bz = floor_div(vz, block_size)

    lx = floor_mod(vx, block_size)
    ly = floor_mod(vy, block_size)
    lz = floor_mod(vz, block_size)

    target_key = pack_key_only(bx, by, bz)
    target_key_part = get_key_part(target_key)
    slot = spatial_hash(bx, by, bz, hash_capacity)

    pool_idx = wp.int32(-1)
    for _probe in range(hash_capacity):
        entry = hash_table[slot]
        if entry == ENTRY_EMPTY:
            break
        if entry != ENTRY_TOMBSTONE:
            if get_key_part(entry) == target_key_part:
                pool_idx = get_pool_idx(entry)
                break
        slot = (slot + 1) % hash_capacity

    if pool_idx < 0:
        return

    local_idx = lz * 64 + ly * 8 + lx

    wp.atomic_add(block_data, pool_idx, local_idx, 0, wp.float16(sdf * weight))
    wp.atomic_add(block_data, pool_idx, local_idx, 1, wp.float16(weight))

    rgb_row = cam_idx * img_H + py
    r_weighted = wp.float32(rgb_images_flat[rgb_row, px, 0]) * weight
    g_weighted = wp.float32(rgb_images_flat[rgb_row, px, 1]) * weight
    b_weighted = wp.float32(rgb_images_flat[rgb_row, px, 2]) * weight

    wp.atomic_add(block_rgb, pool_idx, 0, r_weighted)
    wp.atomic_add(block_rgb, pool_idx, 1, g_weighted)
    wp.atomic_add(block_rgb, pool_idx, 2, b_weighted)
    wp.atomic_add(block_rgb, pool_idx, 3, weight)


# =============================================================================
# Public API
# =============================================================================


class SortFilterIntegrator:
    """Sort & Filter TSDF Integrator - CUDA Graph Safe, Zero Contention.

    This class manages pre-allocated buffers for the 4-phase integration.
    All tensor shapes are static, enabling CUDA graph capture.
    """

    def __init__(
        self,
        device: str = "cuda:0",
    ):
        """Initialize with pre-allocated buffers.

        Args:
            max_samples: Maximum samples per frame (H * W * num_samples).
            device: CUDA device.
        """
        self.device = device
        self.max_samples = -1



    def setup_buffers(self, max_samples: int):
        if max_samples != self.max_samples:
            # print(f"Setting up buffers for {max_samples} samples")
            self.block_keys = torch.empty(max_samples, dtype=torch.int64, device=self.device)
            self.sample_data = torch.empty((max_samples, 4), dtype=torch.float32, device=self.device)
            self.sorted_keys = torch.empty(max_samples, dtype=torch.int64, device=self.device)
            self.sort_indices = torch.empty(max_samples, dtype=torch.int64, device=self.device)
            self.max_samples = max_samples

    def integrate(
        self,
        tsdf,
        depth_images: torch.Tensor,
        rgb_images: torch.Tensor,
        cam_positions: torch.Tensor,
        cam_quaternions: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_min: float = 0.1,
        depth_max: float = 5.0,
        grid_size: tuple = None,
    ):
        """Integrate depth from one or more cameras using batched kernels.

        Args:
            tsdf: BlockSparseTSDF instance.
            depth_images: (num_cameras, H, W) float32, meters.
            rgb_images: (num_cameras, H, W, 3) uint8.
            cam_positions: (num_cameras, 3) float32.
            cam_quaternions: (num_cameras, 4) float32, wxyz.
            intrinsics: (num_cameras, 3, 3) float32.
            depth_min: Minimum valid depth.
            depth_max: Maximum valid depth.
            grid_size: Optional (nz, ny, nx) for bounds.
        """
        tsdf.prepare_frame()

        n_cameras = depth_images.shape[0]
        img_H, img_W = depth_images.shape[1], depth_images.shape[2]
        n_pixels = img_H * img_W

        step_size = tsdf.config.voxel_size
        num_samples = int((2.0 * tsdf.config.truncation_distance / step_size) + 1)
        samples_per_cam = n_pixels * num_samples
        N = n_cameras * samples_per_cam
        self.setup_buffers(N)

        if grid_size is not None:
            grid_D, grid_H_dim, grid_W_dim = grid_size
        else:
            grid_W_dim, grid_H_dim, grid_D = 0, 0, 0

        data = tsdf.get_warp_data()
        device, stream = get_warp_device_stream(depth_images)

        # Phase 1: Block discovery across all cameras
        wp.launch(
            compute_block_keys_kernel,
            dim=N,
            inputs=[
                wp.from_torch(intrinsics, dtype=wp.float32),
                wp.from_torch(cam_positions, dtype=wp.float32),
                wp.from_torch(cam_quaternions, dtype=wp.float32),
                wp.from_torch(depth_images, dtype=wp.float32),
                wp.from_torch(tsdf.data.origin, dtype=wp.float32),
                tsdf.config.voxel_size,
                tsdf.config.truncation_distance,
                depth_min,
                depth_max,
                tsdf.config.block_size,
                num_samples,
                img_H,
                img_W,
                grid_W_dim,
                grid_H_dim,
                grid_D,
                n_cameras,
                wp.from_torch(self.block_keys[:N]),
                wp.from_torch(self.sample_data[:N]),
            ],
            device=device,
            stream=stream,
        )

        # Phase 2: Sort keys
        self.sorted_keys[:N], self.sort_indices[:N] = torch.sort(self.block_keys[:N])

        # Phase 3: Allocate unique blocks
        wp.launch(
            allocate_unique_blocks_kernel,
            dim=N,
            inputs=[
                wp.from_torch(self.sorted_keys[:N]),
                N,
                data.hash_table,
                tsdf.config.hash_capacity,
                data.block_coords,
                data.block_to_hash_slot,
                data.free_list,
                data.free_count,
                data.num_allocated,
                tsdf.config.max_blocks,
                data.new_blocks,
                data.new_block_count,
            ],
            device=device,
            stream=stream,
        )

        # Phase 4: Integrate samples
        rgb_flat = rgb_images.reshape(n_cameras * img_H, img_W, 3)

        wp.launch(
            integrate_samples_kernel,
            dim=N,
            inputs=[
                wp.from_torch(self.sorted_keys[:N]),
                wp.from_torch(self.sort_indices[:N]),
                wp.from_torch(self.sample_data[:N]),
                N,
                wp.from_torch(depth_images, dtype=wp.float32),
                wp.from_torch(rgb_flat, dtype=wp.uint8),
                img_H,
                img_W,
                num_samples,
                samples_per_cam,
                tsdf.config.voxel_size,
                data.hash_table,
                tsdf.config.hash_capacity,
                data.block_data,
                data.block_rgb,
                tsdf.config.block_size,
            ],
            device=device,
            stream=stream,
        )


def integrate_depth_sort_filter(
    tsdf,
    depth_image: torch.Tensor,
    rgb_image: torch.Tensor,
    cam_position: torch.Tensor,
    cam_quaternion: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_min: float = 0.1,
    depth_max: float = 5.0,
    grid_size: tuple = None,
):
    """One-shot Sort & Filter integration (allocates buffers each call).

    For repeated calls, use SortFilterIntegrator class instead.
    """
    img_H, img_W = depth_image.shape
    step_size = tsdf.config.voxel_size
    num_samples = int((2.0 * tsdf.config.truncation_distance / step_size) + 1)
    max_samples = img_H * img_W * num_samples

    integrator = SortFilterIntegrator(max_samples, device=str(depth_image.device))
    integrator.integrate(
        tsdf, depth_image, rgb_image, cam_position, cam_quaternion,
        intrinsics, depth_min, depth_max, grid_size,
    )

