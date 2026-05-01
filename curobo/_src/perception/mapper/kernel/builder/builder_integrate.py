# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Voxel-project TSDF/RGB/feature integration kernels for one block size.

The generated kernels close over map geometry, image shape, camera count,
sample count, and feature-grid layout. Runtime arguments are reserved for
map tensors, camera tensors, observation tensors, visible-block state, and
integration thresholds that are intentionally frame-configurable.
"""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    compute_tsdf_weight,
    floor_div,
)
from curobo._src.util.warp import warp_constant_suffix, warp_kernel


def make_integrate_kernels(
    block_size: int,
    *,
    feature_dim: int,
    num_cameras: int,
    image_height: int,
    image_width: int,
    num_samples: int,
    grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
    truncation_distance: float,
    feature_grid_shape: tuple[int, int] | None,
    feature_channels_per_thread: int,
    max_feature_tile_channels: int,
    max_support_pixels_per_block_camera: int,
    pack_key_only,
    unpack_block_key,
    find_or_insert_block,
    hash_lookup,
    voxel_to_world,
    voxel_to_world_corner,
    world_to_continuous_voxel,
    block_local_to_world,
    block_grid_to_key_coords,
    block_key_to_grid_coords,
) -> dict[str, object]:
    """Build voxel-project TSDF integration kernels."""
    BLOCK_SIZE = wp.constant(block_size)
    NUM_CAMERAS = wp.constant(wp.int32(num_cameras))
    IMAGE_HEIGHT = wp.constant(wp.int32(image_height))
    IMAGE_WIDTH = wp.constant(wp.int32(image_width))
    NUM_SAMPLES = wp.constant(wp.int32(num_samples))
    GRID_D = wp.constant(wp.int32(grid_shape[0]))
    GRID_H = wp.constant(wp.int32(grid_shape[1]))
    GRID_W = wp.constant(wp.int32(grid_shape[2]))
    ORIGIN_X = wp.constant(wp.float32(origin_xyz[0]))
    ORIGIN_Y = wp.constant(wp.float32(origin_xyz[1]))
    ORIGIN_Z = wp.constant(wp.float32(origin_xyz[2]))
    VOXEL_SIZE = wp.constant(wp.float32(voxel_size))
    TRUNCATION_DIST = wp.constant(wp.float32(truncation_distance))
    safe_step = (float(block_size) * float(voxel_size)) / 1.42
    STEP_SIZE = wp.constant(wp.float32(safe_step))
    FEATURE_DIM = wp.constant(wp.int32(feature_dim))
    if feature_grid_shape is None:
        feature_grid_height = 1
        feature_grid_width = 1
    else:
        feature_grid_height = int(feature_grid_shape[0])
        feature_grid_width = int(feature_grid_shape[1])
    FEATURE_GRID_HEIGHT = wp.constant(wp.int32(feature_grid_height))
    FEATURE_GRID_WIDTH = wp.constant(wp.int32(feature_grid_width))
    suffix = (
        f"bs{block_size}_cfg"
        f"{warp_constant_suffix(block_size, feature_dim, num_cameras, image_height, image_width, num_samples, grid_shape, origin_xyz, voxel_size, truncation_distance, feature_grid_shape, feature_channels_per_thread, max_feature_tile_channels, max_support_pixels_per_block_camera)}"
    )

    # Cross-domain helpers are explicit parameters so Warp sees them as
    # local closure bindings when compiling dependent kernels.
    FEATURE_CHANNELS_PER_THREAD = wp.constant(feature_channels_per_thread)
    feature_tile_channels = max(1, min(int(feature_dim), int(max_feature_tile_channels)))
    FEATURE_TILE_CHANNELS = wp.constant(feature_tile_channels)
    support_capacity = int(max_support_pixels_per_block_camera)
    SUPPORT_CAPACITY = wp.constant(support_capacity)

    @warp_kernel(f"compute_block_keys_only_kernel_{suffix}")
    def compute_block_keys_only_kernel(
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        depth_min: float,
        depth_max: float,
        block_keys: wp.array(dtype=wp.int64),
    ):
        """Phase 1 (voxel-project): emit only block keys, no sample data."""
        tid = wp.tid()
        n_pixels = IMAGE_HEIGHT * IMAGE_WIDTH
        samples_per_cam = n_pixels * NUM_SAMPLES
        cam_idx = tid // samples_per_cam
        remainder = tid % samples_per_cam
        pixel_idx = remainder // NUM_SAMPLES
        sample_idx = remainder % NUM_SAMPLES

        if cam_idx >= NUM_CAMERAS or pixel_idx >= n_pixels:
            block_keys[tid] = wp.int64(-1)
            return

        px = pixel_idx % IMAGE_WIDTH
        py = pixel_idx // IMAGE_WIDTH

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

        z_start = wp.max(depth - TRUNCATION_DIST, depth_min)
        z_sample = z_start + wp.float32(sample_idx) * STEP_SIZE

        if z_sample > depth + TRUNCATION_DIST + STEP_SIZE:
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

        voxel_f = world_to_continuous_voxel(point_world)

        vx = wp.int32(wp.floor(voxel_f[0]))
        vy = wp.int32(wp.floor(voxel_f[1]))
        vz = wp.int32(wp.floor(voxel_f[2]))

        if vx < 0 or vx >= GRID_W or vy < 0 or vy >= GRID_H or vz < 0 or vz >= GRID_D:
            block_keys[tid] = wp.int64(-1)
            return

        bx_grid = floor_div(vx, BLOCK_SIZE)
        by_grid = floor_div(vy, BLOCK_SIZE)
        bz_grid = floor_div(vz, BLOCK_SIZE)
        key = block_grid_to_key_coords(bx_grid, by_grid, bz_grid)

        block_keys[tid] = pack_key_only(key[0], key[1], key[2])

    @warp_kernel(f"allocate_visible_blocks_from_keys_kernel_bs{block_size}")
    def allocate_visible_blocks_from_keys_kernel(
        block_keys: wp.array(dtype=wp.int64),
        n_keys: wp.int32,
        hash_table: wp.array(dtype=wp.int64),
        hash_capacity: wp.int32,
        block_coords: wp.array(dtype=wp.int32),
        block_to_hash_slot: wp.array(dtype=wp.int32),
        num_allocated: wp.array(dtype=wp.int32),
        allocation_failures: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
        free_list: wp.array(dtype=wp.int32),
        free_count: wp.array(dtype=wp.int32),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        visible_epoch: wp.array(dtype=wp.int32),
        visible_count: wp.array(dtype=wp.int32),
        frame_epoch: wp.int32,
        pool_indices: wp.array(dtype=wp.int32),
        pool_to_visible_slot: wp.array(dtype=wp.int32),
        visible_capacity: wp.int32,
    ):
        """Allocate/lookup candidate keys and emit each visible pool once."""
        tid = wp.tid()
        if tid >= n_keys:
            return

        key = block_keys[tid]
        if key == wp.int64(-1):
            return

        coords = unpack_block_key(key)
        pool_idx = find_or_insert_block(
            hash_table,
            block_coords,
            block_to_hash_slot,
            free_list,
            free_count,
            num_allocated,
            allocation_failures,
            new_blocks,
            new_block_count,
            coords[0],
            coords[1],
            coords[2],
            hash_capacity,
            max_blocks,
        )
        if pool_idx < wp.int32(0):
            return

        old_epoch = visible_epoch[pool_idx]
        if old_epoch != frame_epoch:
            prev_epoch = wp.atomic_cas(visible_epoch, pool_idx, old_epoch, frame_epoch)
            if prev_epoch == old_epoch:
                out_idx = wp.atomic_add(visible_count, 0, wp.int32(1))
                if out_idx < visible_capacity:
                    pool_indices[out_idx] = pool_idx
                    pool_to_visible_slot[pool_idx] = out_idx

    @warp_kernel(
        f"build_support_pixels_from_keys_kernel_{suffix}_sc{support_capacity}"
    )
    def build_support_pixels_from_keys_kernel(
        block_keys: wp.array(dtype=wp.int64),
        n_keys: wp.int32,
        hash_table: wp.array(dtype=wp.int64),
        hash_capacity: wp.int32,
        max_blocks: wp.int32,
        visible_epoch: wp.array(dtype=wp.int32),
        frame_epoch: wp.int32,
        pool_to_visible_slot: wp.array(dtype=wp.int32),
        visible_capacity: wp.int32,
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        support_overflow_count: wp.array(dtype=wp.int32),
    ):
        """Append pixel support lists after visible slots are published."""
        tid = wp.tid()
        if tid >= n_keys:
            return

        key = block_keys[tid]
        if key == wp.int64(-1):
            return

        n_pixels = IMAGE_HEIGHT * IMAGE_WIDTH
        samples_per_cam = n_pixels * NUM_SAMPLES
        cam_idx = tid // samples_per_cam
        remainder = tid % samples_per_cam
        pixel_idx = remainder // NUM_SAMPLES
        sample_idx = remainder % NUM_SAMPLES

        if cam_idx >= NUM_CAMERAS or pixel_idx >= n_pixels:
            return

        if sample_idx > wp.int32(0):
            prev_key = block_keys[tid - wp.int32(1)]
            if prev_key == key:
                return

        coords = unpack_block_key(key)
        pool_idx = hash_lookup(
            hash_table,
            coords[0],
            coords[1],
            coords[2],
            hash_capacity,
        )
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return
        if visible_epoch[pool_idx] != frame_epoch:
            return

        vis_idx = pool_to_visible_slot[pool_idx]
        if vis_idx < wp.int32(0) or vis_idx >= visible_capacity:
            return

        slot = wp.atomic_add(support_counts, vis_idx, cam_idx, wp.int32(1))
        if slot < SUPPORT_CAPACITY:
            support_pixels[vis_idx, cam_idx, slot] = pixel_idx
        else:
            wp.atomic_add(support_overflow_count, 0, wp.int32(1))

    @warp_kernel(f"collect_blocks_in_aabb_kernel_{suffix}")
    def collect_blocks_in_aabb_kernel(
        hash_table: wp.array(dtype=wp.int64),
        hash_capacity: wp.int32,
        min_bx: wp.int32,
        min_by: wp.int32,
        min_bz: wp.int32,
        count_x: wp.int32,
        count_y: wp.int32,
        count_z: wp.int32,
        clear_pool_indices: wp.array(dtype=wp.int32),
        clear_count: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
    ):
        """Collect allocated blocks whose volume intersects a world AABB.

        Launch with ``dim = (count_x, count_y, count_z)``.
        """
        local_x, local_y, local_z = wp.tid()
        if local_x >= count_x or local_y >= count_y or local_z >= count_z:
            return

        bx = min_bx + local_x
        by = min_by + local_y
        bz = min_bz + local_z

        grid = block_key_to_grid_coords(bx, by, bz)
        max_bx = (GRID_W + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        max_by = (GRID_H + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        max_bz = (GRID_D + BLOCK_SIZE - wp.int32(1)) // BLOCK_SIZE
        if (
            grid[0] < 0
            or grid[0] >= max_bx
            or grid[1] < 0
            or grid[1] >= max_by
            or grid[2] < 0
            or grid[2] >= max_bz
        ):
            return

        pool_idx = hash_lookup(hash_table, bx, by, bz, hash_capacity)
        if pool_idx < wp.int32(0):
            return

        out_idx = wp.atomic_add(clear_count, 0, wp.int32(1))
        if out_idx < max_blocks:
            clear_pool_indices[out_idx] = pool_idx

    @warp_kernel(f"clear_blocks_by_pool_kernel_{suffix}")
    def clear_blocks_by_pool_kernel(
        clear_pool_indices: wp.array(dtype=wp.int32),
        clear_count: wp.array(dtype=wp.int32),
        block_data: wp.array3d(dtype=wp.float16),
        block_rgb: wp.array2d(dtype=wp.float16),
        block_sums: wp.array(dtype=wp.float32),
        max_blocks: wp.int32,
    ):
        """Zero dynamic TSDF/RGB data for already allocated blocks."""
        slot_idx, local_idx = wp.tid()

        count = clear_count[0]
        if count > max_blocks:
            count = max_blocks
        if slot_idx >= count:
            return

        pool_idx = clear_pool_indices[slot_idx]
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return

        block_data[pool_idx, local_idx, 0] = wp.float16(0.0)
        block_data[pool_idx, local_idx, 1] = wp.float16(0.0)

        if local_idx < wp.int32(4): # 0, 1, 2, 3
            block_rgb[pool_idx, local_idx] = wp.float16(0.0)
        if local_idx == wp.int32(0):
            block_sums[pool_idx] = wp.float32(0.0)

    @warp_kernel(f"clear_block_features_by_pool_kernel_{suffix}")
    def clear_block_features_by_pool_kernel(
        clear_pool_indices: wp.array(dtype=wp.int32),
        clear_count: wp.array(dtype=wp.int32),
        block_features: wp.array2d(dtype=wp.float16),
        block_feature_weight: wp.array(dtype=wp.float16),
        max_blocks: wp.int32,
    ):
        """Zero feature accumulators for already allocated blocks.

        Launch with ``dim = (n_clear, feature_dim)`` so one thread
        clears one ``(block, channel)`` cell; the thread with
        ``ch == 0`` also zeroes the per-block feature weight.
        """
        slot_idx, ch = wp.tid()

        count = clear_count[0]
        if count > max_blocks:
            count = max_blocks
        if slot_idx >= count:
            return

        pool_idx = clear_pool_indices[slot_idx]
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return

        block_features[pool_idx, ch] = wp.float16(0.0)
        if ch == wp.int32(0):
            block_feature_weight[pool_idx] = wp.float16(0.0)

    @warp_kernel(f"integrate_voxels_kernel_{suffix}")
    def integrate_voxels_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        depth_min: float,
        depth_max: float,
        block_coords: wp.array(dtype=wp.int32),
        block_data: wp.array3d(dtype=wp.float16),
    ):
        """Phase 4 (voxel-project): one thread per voxel, serial camera loop.

        Launch with ``dim = (n_visible, BLOCK_SIZE ** 3)``. ``BLOCK_SIZE`` is
        closure-captured so thread indexing stays consistent with
        the specialized block-voxel count.
        """
        vis_idx, local_idx = wp.tid()

        if vis_idx >= n_visible:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < 0:
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]

        voxel_center = block_local_to_world(
            bx,
            by,
            bz,
            local_idx,
        )

        total_sw = wp.float32(0.0)
        total_w = wp.float32(0.0)

        for cam_i in range(num_cameras):
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

                if px >= 0 and px < IMAGE_WIDTH and py >= 0 and py < IMAGE_HEIGHT:
                    depth = depth_images[cam_i, py, px]
                    if depth >= depth_min and depth <= depth_max:
                        sdf = depth - z_cam
                        if sdf >= -TRUNCATION_DIST:
                            sdf_clamped = wp.min(sdf, TRUNCATION_DIST)
                            base_weight = compute_tsdf_weight(depth, VOXEL_SIZE)
                            coverage = (fx * VOXEL_SIZE / z_cam) * (fy * VOXEL_SIZE / z_cam)
                            weight = base_weight * wp.max(coverage, 1.0)

                            total_sw = total_sw + sdf_clamped * weight
                            total_w = total_w + weight

        if total_w > 0.0:
            old_sw = wp.float32(block_data[pool_idx, local_idx, 0])
            old_w = wp.float32(block_data[pool_idx, local_idx, 1])
            block_data[pool_idx, local_idx, 0] = wp.float16(old_sw + total_sw)
            block_data[pool_idx, local_idx, 1] = wp.float16(old_w + total_w)

    @warp_kernel(
        f"integrate_block_rgb_from_support_kernel_{suffix}_sc{support_capacity}"
    )
    def integrate_block_rgb_from_support_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        rgb_images_flat: wp.array3d(dtype=wp.uint8),
        block_rgb: wp.array2d(dtype=wp.float16),
    ):
        """Per-block RGB aggregation from allocation-time support pixels."""
        vis_idx, cam_i = wp.tid()
        if vis_idx >= n_visible:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        count = support_counts[vis_idx, cam_i]
        if count > SUPPORT_CAPACITY:
            count = SUPPORT_CAPACITY
        if count <= wp.int32(0):
            return

        inv_255 = wp.float32(1.0 / 255.0)
        total_r = wp.float32(0.0)
        total_g = wp.float32(0.0)
        total_b = wp.float32(0.0)

        for k in range(support_capacity):
            if k < count:
                pixel_idx = support_pixels[vis_idx, cam_i, k]
                py = pixel_idx // IMAGE_WIDTH
                px = pixel_idx - py * IMAGE_WIDTH
                if (
                    py >= wp.int32(0)
                    and py < IMAGE_HEIGHT
                    and px >= wp.int32(0)
                    and px < IMAGE_WIDTH
                ):
                    rgb_row = cam_i * IMAGE_HEIGHT + py
                    total_r = total_r + wp.float32(rgb_images_flat[rgb_row, px, 0]) * inv_255
                    total_g = total_g + wp.float32(rgb_images_flat[rgb_row, px, 1]) * inv_255
                    total_b = total_b + wp.float32(rgb_images_flat[rgb_row, px, 2]) * inv_255

        weight = wp.float32(count)
        wp.atomic_add(block_rgb, pool_idx, 0, wp.float16(total_r))
        wp.atomic_add(block_rgb, pool_idx, 1, wp.float16(total_g))
        wp.atomic_add(block_rgb, pool_idx, 2, wp.float16(total_b))
        wp.atomic_add(block_rgb, pool_idx, 3, wp.float16(weight))

    @warp_kernel(
        f"integrate_features_from_support_grouped_kernel_{suffix}_fcpt"
        f"{feature_channels_per_thread}_sc{support_capacity}"
    )
    def integrate_features_from_support_grouped_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        feature_grid: wp.array4d(dtype=wp.float16),
        block_features: wp.array2d(dtype=wp.float16),
        block_feature_weight: wp.array(dtype=wp.float16),
    ):
        """Per-block feature aggregation from allocation-time support pixels."""
        n_channel_groups = (
            FEATURE_DIM + FEATURE_CHANNELS_PER_THREAD - wp.int32(1)
        ) // FEATURE_CHANNELS_PER_THREAD
        vis_idx, cam_i, feature_channel_group_idx = wp.tid()

        if vis_idx >= n_visible or feature_channel_group_idx >= n_channel_groups:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        count = support_counts[vis_idx, cam_i]
        if count > SUPPORT_CAPACITY:
            count = SUPPORT_CAPACITY
        if count <= wp.int32(0):
            return

        base_feature_channel = feature_channel_group_idx * FEATURE_CHANNELS_PER_THREAD
        feature_acc = wp.zeros(FEATURE_CHANNELS_PER_THREAD, dtype=wp.float32)

        for k in range(support_capacity):
            if k < count:
                pixel_idx = support_pixels[vis_idx, cam_i, k]
                py = pixel_idx // IMAGE_WIDTH
                px = pixel_idx - py * IMAGE_WIDTH
                if (
                    py >= wp.int32(0)
                    and py < IMAGE_HEIGHT
                    and px >= wp.int32(0)
                    and px < IMAGE_WIDTH
                ):
                    gy = (py * FEATURE_GRID_HEIGHT) // IMAGE_HEIGHT
                    gx = (px * FEATURE_GRID_WIDTH) // IMAGE_WIDTH
                    if gy < wp.int32(0):
                        gy = wp.int32(0)
                    if gx < wp.int32(0):
                        gx = wp.int32(0)
                    if gy >= FEATURE_GRID_HEIGHT:
                        gy = FEATURE_GRID_HEIGHT - wp.int32(1)
                    if gx >= FEATURE_GRID_WIDTH:
                        gx = FEATURE_GRID_WIDTH - wp.int32(1)
                    for feature_channel_offset in range(FEATURE_CHANNELS_PER_THREAD):
                        feature_channel = base_feature_channel + feature_channel_offset
                        if feature_channel < FEATURE_DIM:
                            feature_acc[feature_channel_offset] = (
                                feature_acc[feature_channel_offset]
                                + wp.float32(
                                    feature_grid[
                                        cam_i,
                                        gy,
                                        gx,
                                        feature_channel,
                                    ]
                                )
                            )

        for feature_channel_offset in range(FEATURE_CHANNELS_PER_THREAD):
            feature_channel = base_feature_channel + feature_channel_offset
            if feature_channel < FEATURE_DIM:
                wp.atomic_add(
                    block_features,
                    pool_idx,
                    feature_channel,
                    wp.float16(feature_acc[feature_channel_offset]),
                )
        if base_feature_channel == wp.int32(0):
            wp.atomic_add(block_feature_weight, pool_idx, wp.float16(wp.float32(count)))

    @warp_kernel(
        f"integrate_features_from_support_tiled_kernel_{suffix}_tile"
        f"{feature_tile_channels}_sc{support_capacity}"
    )
    def integrate_features_from_support_tiled_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        feature_grid: wp.array4d(dtype=wp.float16),
        block_features: wp.array2d(dtype=wp.float16),
        block_feature_weight: wp.array(dtype=wp.float16),
    ):
        """Tile prototype: one CTA accumulates a feature-channel slice."""
        n_channel_tiles = (
            FEATURE_DIM + FEATURE_TILE_CHANNELS - wp.int32(1)
        ) // FEATURE_TILE_CHANNELS
        vis_idx, cam_i, feature_tile_idx, lane = wp.tid()

        if vis_idx >= n_visible or feature_tile_idx >= n_channel_tiles:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        count = support_counts[vis_idx, cam_i]
        if count > SUPPORT_CAPACITY:
            count = SUPPORT_CAPACITY
        if count <= wp.int32(0):
            return

        base_feature_channel = feature_tile_idx * FEATURE_TILE_CHANNELS
        feature_acc = wp.tile_zeros(shape=feature_tile_channels, dtype=wp.float32)

        for k in range(support_capacity):
            if k < count:
                pixel_idx = support_pixels[vis_idx, cam_i, k]
                py = pixel_idx // IMAGE_WIDTH
                px = pixel_idx - py * IMAGE_WIDTH
                if (
                    py >= wp.int32(0)
                    and py < IMAGE_HEIGHT
                    and px >= wp.int32(0)
                    and px < IMAGE_WIDTH
                ):
                    gy = (py * FEATURE_GRID_HEIGHT) // IMAGE_HEIGHT
                    gx = (px * FEATURE_GRID_WIDTH) // IMAGE_WIDTH
                    if gy < wp.int32(0):
                        gy = wp.int32(0)
                    if gx < wp.int32(0):
                        gx = wp.int32(0)
                    if gy >= FEATURE_GRID_HEIGHT:
                        gy = FEATURE_GRID_HEIGHT - wp.int32(1)
                    if gx >= FEATURE_GRID_WIDTH:
                        gx = FEATURE_GRID_WIDTH - wp.int32(1)

                    feature_vals_h = wp.tile_load(
                        feature_grid[cam_i, gy, gx],
                        shape=feature_tile_channels,
                        offset=base_feature_channel,
                        bounds_check=True,
                    )
                    feature_acc = feature_acc + wp.tile_astype(
                        feature_vals_h,
                        dtype=wp.float32,
                    )

        feature_acc_h = wp.tile_astype(feature_acc, dtype=wp.float16)
        wp.tile_atomic_add(
            block_features[pool_idx],
            feature_acc_h,
            offset=base_feature_channel,
            bounds_check=True,
        )
        if feature_tile_idx == wp.int32(0) and lane == wp.int32(0):
            wp.atomic_add(block_feature_weight, pool_idx, wp.float16(wp.float32(count)))

    return {
        "compute_block_keys_only_kernel": compute_block_keys_only_kernel,
        "allocate_visible_blocks_from_keys_kernel": allocate_visible_blocks_from_keys_kernel,
        "build_support_pixels_from_keys_kernel": build_support_pixels_from_keys_kernel,
        "collect_blocks_in_aabb_kernel": collect_blocks_in_aabb_kernel,
        "clear_blocks_by_pool_kernel": clear_blocks_by_pool_kernel,
        "clear_block_features_by_pool_kernel": clear_block_features_by_pool_kernel,
        "integrate_voxels_kernel": integrate_voxels_kernel,
        "integrate_block_rgb_from_support_kernel": integrate_block_rgb_from_support_kernel,
        "integrate_features_from_support_grouped_kernel": (
            integrate_features_from_support_grouped_kernel
        ),
        "integrate_features_from_support_tiled_kernel": (
            integrate_features_from_support_tiled_kernel
        ),
        "integrate_block_rgb_kernel": integrate_block_rgb_from_support_kernel,
        "integrate_features_grouped_kernel": integrate_features_from_support_grouped_kernel,
    }
