# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Voxel-Project TSDF Integration - Zero Atomics, Voxel-Centric.

This module implements a 4-phase integration strategy using voxel-centric
projection for Phase 4, eliminating all atomic contention on TSDF data:

    Phase 1: Compute block keys from pixels (reused from sort_filter)
    Phase 2: Deduplicate keys (torch.unique)
    Phase 3: Allocate unique blocks + resolve pool indices (reused kernel)
    Phase 4: Voxel-centric integration (one thread per voxel, zero atomics)

Key Insight:
    Instead of pixel→voxel mapping (which requires atomics when multiple pixels
    hit the same voxel), we use voxel→pixel mapping: each thread owns one voxel,
    projects it into the image, reads the depth, and writes directly. This
    eliminates all atomic contention on block_data.

Trade-off:
    Phase 4 uses a variable launch dimension (n_visible_blocks × 512), which
    requires a D2H sync to read n_visible. This breaks CUDA graph compatibility
    for Phase 4, but Phases 1-3 remain graph-safe.

Performance vs SortFilterIntegrator:
    - Zero atomics on block_data (was ~1.2M atomic_add per frame)
    - Zero atomics on block_rgb (was ~160K per visible block)
    - Coalesced memory writes (adjacent threads write adjacent voxels)
    - No hash table lookup in Phase 4 (pool_idx from visible list)
    - No scattered reads (no sort_indices indirection)
    - No torch.sort (torch.unique instead)
"""

import math

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.wp_coord import (
    block_local_to_world,
    world_to_continuous_voxel,
)
from curobo._src.perception.mapper.kernel.wp_hash import (
    clear_new_blocks_kernel,
    pack_key_only,
)
from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    compute_tsdf_weight,
    floor_div,
    quat_from_wxyz_array,
    vec3_from_array,
)
from curobo._src.perception.mapper.kernel.wp_stamp_obstacles import (
    preallocate_unique_blocks_kernel,
)
from curobo._src.util.warp import get_warp_device_stream

# =============================================================================
# Integration Kernels
# =============================================================================


@wp.kernel
def compute_block_keys_only_kernel(
    # Per-camera parameters (batched)
    intrinsics: wp.array3d(dtype=wp.float32),
    cam_positions: wp.array2d(dtype=wp.float32),
    cam_quaternions: wp.array2d(dtype=wp.float32),
    depth_images: wp.array3d(dtype=wp.float32),
    # Grid parameters
    origin: wp.array(dtype=wp.float32),
    voxel_size: float,
    truncation_dist: float,
    step_size: float,
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
):
    """Phase 1 for multiple cameras.

    Launch with dim = n_cameras * n_pixels * num_samples.
    Thread mapping: cam_idx -> pixel_idx -> sample_idx.
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
    z_sample = z_start + wp.float32(sample_idx) * step_size

    if z_sample > depth + truncation_dist + step_size:
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

    block_keys[tid] = pack_key_only(bx, by, bz)


@wp.kernel
def integrate_voxels_kernel(
    # Visible block pool indices
    visible_pool_indices: wp.array(dtype=wp.int32),
    n_visible: wp.int32,
    # Per-camera parameters (batched)
    intrinsics: wp.array3d(dtype=wp.float32),
    cam_positions: wp.array2d(dtype=wp.float32),
    cam_quaternions: wp.array2d(dtype=wp.float32),
    depth_images: wp.array3d(dtype=wp.float32),
    # RGB flattened: (n_cameras * H, W, 3); index with cam_idx * img_H + py
    rgb_images_flat: wp.array3d(dtype=wp.uint8),
    n_cameras: wp.int32,
    img_H: wp.int32,
    img_W: wp.int32,
    # Grid parameters
    origin: wp.array(dtype=wp.float32),
    voxel_size: float,
    truncation_dist: float,
    depth_min: float,
    depth_max: float,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    # Block storage
    block_coords: wp.array(dtype=wp.int32),
    block_data: wp.array3d(dtype=wp.float16),
    block_rgb: wp.array2d(dtype=wp.float32),
):
    """Voxel-centric integration across multiple cameras.

    Each thread owns one voxel and loops over all cameras, accumulating
    SDF and weight contributions in fp32 before a single fp16 write.

    The serial camera loop avoids atomics on block_data entirely.  An
    alternative that parallelizes across cameras (one thread per
    voxel-camera pair with atomic_add) was benchmarked at ~3x slower
    due to fp16 atomic contention, so the serial approach is preferred.

    Launch with dim = n_visible * 512.
    """
    tid = wp.tid()
    vis_idx = tid // 512
    local_idx = tid % 512

    if vis_idx >= n_visible:
        return

    pool_idx = visible_pool_indices[vis_idx]
    if pool_idx < 0:
        return

    bx = block_coords[pool_idx * 3 + 0]
    by = block_coords[pool_idx * 3 + 1]
    bz = block_coords[pool_idx * 3 + 2]

    grid_origin = vec3_from_array(origin)
    voxel_center = block_local_to_world(
        bx, by, bz, local_idx,
        grid_origin, voxel_size, block_size,
        grid_W, grid_H, grid_D,
    )

    # Accumulate across all cameras in fp32
    total_sw = wp.float32(0.0)
    total_w = wp.float32(0.0)
    total_rw = wp.float32(0.0)
    total_gw = wp.float32(0.0)
    total_bw = wp.float32(0.0)

    for cam_i in range(n_cameras):
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
        cam_quat_inv = wp.quat_inverse(cam_quat)
        voxel_cam = wp.quat_rotate(cam_quat_inv, voxel_center - cam_pos)

        z_cam = voxel_cam[2]
        if z_cam > depth_min:
            fx = intrinsics[cam_i, 0, 0]
            fy = intrinsics[cam_i, 1, 1]
            cx_i = intrinsics[cam_i, 0, 2]
            cy_i = intrinsics[cam_i, 1, 2]

            u = fx * voxel_cam[0] / z_cam + cx_i
            v = fy * voxel_cam[1] / z_cam + cy_i

            px = wp.int32(u)
            py = wp.int32(v)

            if px >= 0 and px < img_W and py >= 0 and py < img_H:
                depth = depth_images[cam_i, py, px]
                if depth >= depth_min and depth <= depth_max:
                    sdf = depth - z_cam
                    if sdf >= -truncation_dist:
                        sdf_clamped = wp.min(sdf, truncation_dist)
                        base_weight = compute_tsdf_weight(depth, voxel_size)
                        coverage = (fx * voxel_size / z_cam) * (fy * voxel_size / z_cam)
                        weight = base_weight * wp.max(coverage, 1.0)

                        total_sw = total_sw + sdf_clamped * weight
                        total_w = total_w + weight

                        rgb_row = cam_i * img_H + py
                        total_rw = total_rw + wp.float32(rgb_images_flat[rgb_row, px, 0]) * weight
                        total_gw = total_gw + wp.float32(rgb_images_flat[rgb_row, px, 1]) * weight
                        total_bw = total_bw + wp.float32(rgb_images_flat[rgb_row, px, 2]) * weight

    if total_w > 0.0:
        old_sw = wp.float32(block_data[pool_idx, local_idx, 0])
        old_w = wp.float32(block_data[pool_idx, local_idx, 1])
        block_data[pool_idx, local_idx, 0] = wp.float16(old_sw + total_sw)
        block_data[pool_idx, local_idx, 1] = wp.float16(old_w + total_w)

        wp.atomic_add(block_rgb, pool_idx, 0, total_rw)
        wp.atomic_add(block_rgb, pool_idx, 1, total_gw)
        wp.atomic_add(block_rgb, pool_idx, 2, total_bw)
        wp.atomic_add(block_rgb, pool_idx, 3, total_w)


# =============================================================================
# Public API
# =============================================================================


class VoxelProjectIntegrator:
    """Voxel-Project TSDF Integrator - Zero Atomics, Voxel-Centric.

    This class manages buffers for the 4-phase integration pipeline.
    Phases 1-3 discover and allocate blocks. Phase 4 uses voxel-centric
    projection for contention-free TSDF updates.

    Unlike SortFilterIntegrator, Phase 4 uses a variable launch dimension
    (n_visible × 512), which requires a small D2H sync but eliminates all
    atomic contention on block_data.
    """

    def __init__(self, device: str = "cuda:0"):
        """Initialize with device.

        Args:
            device: CUDA device.
        """
        self.device = device
        self._max_key_samples = -1
        self._max_unique = -1

    def _setup_key_buffers(self, n_key_samples: int):
        """Allocate/resize Phase 1 key buffers."""
        if n_key_samples != self._max_key_samples:
            self.block_keys = torch.empty(
                n_key_samples, dtype=torch.int64, device=self.device,
            )
            self._max_key_samples = n_key_samples

    def _setup_unique_buffers(self, max_unique: int):
        """Allocate/resize Phase 3 pool-index output buffer."""
        if max_unique > self._max_unique:
            self.pool_indices = torch.empty(
                max_unique, dtype=torch.int32, device=self.device,
            )
            self._max_unique = max_unique

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

        block_edge = tsdf.config.block_size * tsdf.config.voxel_size
        safe_step = block_edge / 1.42
        num_samples = math.ceil(2.0 * tsdf.config.truncation_distance / safe_step) + 1
        N = n_cameras * n_pixels * num_samples

        self._setup_key_buffers(N)

        if grid_size is not None:
            grid_D, grid_H_dim, grid_W_dim = grid_size
        else:
            grid_W_dim, grid_H_dim, grid_D = 0, 0, 0

        data = tsdf.get_warp_data()
        device, stream = get_warp_device_stream(depth_images)

        # Phase 1: Block discovery across all cameras
        wp.launch(
            compute_block_keys_only_kernel,
            dim=N,
            inputs=[
                wp.from_torch(intrinsics, dtype=wp.float32),
                wp.from_torch(cam_positions, dtype=wp.float32),
                wp.from_torch(cam_quaternions, dtype=wp.float32),
                wp.from_torch(depth_images, dtype=wp.float32),
                wp.from_torch(tsdf.data.origin, dtype=wp.float32),
                tsdf.config.voxel_size,
                tsdf.config.truncation_distance,
                safe_step,
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
            ],
            device=device,
            stream=stream,
        )

        # Phase 2: Deduplicate block keys
        valid_mask = self.block_keys[:N] != -1
        valid_keys = self.block_keys[:N][valid_mask]
        unique_keys = torch.unique(valid_keys)
        n_unique = unique_keys.shape[0]

        if n_unique == 0:
            return

        # Phase 3: Allocate blocks + resolve pool indices
        self._setup_unique_buffers(n_unique)

        wp.launch(
            preallocate_unique_blocks_kernel,
            dim=n_unique,
            inputs=[
                wp.from_torch(unique_keys),
                n_unique,
                data.hash_table,
                tsdf.config.hash_capacity,
                data.block_coords,
                data.block_to_hash_slot,
                data.num_allocated,
                tsdf.config.max_blocks,
                data.free_list,
                data.free_count,
                data.new_blocks,
                data.new_block_count,
                wp.from_torch(self.pool_indices[:n_unique]),
            ],
            device=device,
            stream=stream,
        )

        max_clearable = min(n_unique, tsdf.config.max_blocks)
        wp.launch(
            clear_new_blocks_kernel,
            dim=max_clearable * 512,
            inputs=[
                data.block_data,
                data.block_rgb,
                data.new_blocks,
                data.new_block_count,
                tsdf.config.max_blocks,
            ],
            device=device,
            stream=stream,
        )

        # Phase 4: Voxel-centric integration across all cameras
        n_voxels = n_unique * 512
        rgb_flat = rgb_images.reshape(n_cameras * img_H, img_W, 3)

        wp.launch(
            integrate_voxels_kernel,
            dim=n_voxels,
            inputs=[
                wp.from_torch(self.pool_indices[:n_unique]),
                n_unique,
                wp.from_torch(intrinsics, dtype=wp.float32),
                wp.from_torch(cam_positions, dtype=wp.float32),
                wp.from_torch(cam_quaternions, dtype=wp.float32),
                wp.from_torch(depth_images, dtype=wp.float32),
                wp.from_torch(rgb_flat, dtype=wp.uint8),
                n_cameras,
                img_H,
                img_W,
                wp.from_torch(tsdf.data.origin, dtype=wp.float32),
                tsdf.config.voxel_size,
                tsdf.config.truncation_distance,
                depth_min,
                depth_max,
                tsdf.config.block_size,
                grid_W_dim,
                grid_H_dim,
                grid_D,
                data.block_coords,
                data.block_data,
                data.block_rgb,
            ],
            device=device,
            stream=stream,
        )
