# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Camera projective TSDF/RGB/feature integration kernels for one block size.

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
from curobo._src.util.warp import warp_constant_suffix, warp_func, warp_kernel


def make_camera_integrate_kernels(
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
    color_grid_size: int,
    feature_block_grid_size: int,
    instance_grid_size: int,
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
    block_key_to_voxel_base,
) -> dict[str, object]:
    """Build camera projective TSDF integration kernels."""
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
    COLOR_GRID_SIZE = wp.constant(wp.int32(color_grid_size))
    color_grid_voxels = int(color_grid_size) ** 3
    COLOR_GRID_VOXELS = wp.constant(wp.int32(color_grid_voxels))
    COLOR_GRID_CELLS = wp.constant(wp.int32(color_grid_voxels * 4))
    feature_grid_voxels = int(feature_block_grid_size) ** 3
    FEATURE_BLOCK_GRID_SIZE = wp.constant(wp.int32(feature_block_grid_size))
    FEATURE_GRID_VOXELS = wp.constant(wp.int32(feature_grid_voxels))
    instance_grid_voxels = max(1, int(instance_grid_size) ** 3)
    # Keep disabled builds compilable: the instance kernel is never launched
    # when ``instance_grid_size == 0``, but Warp still parses its arithmetic.
    INSTANCE_GRID_SIZE = wp.constant(wp.int32(max(1, instance_grid_size)))
    INSTANCE_GRID_VOXELS = wp.constant(wp.int32(instance_grid_voxels))
    if feature_grid_shape is None:
        feature_grid_height = 1
        feature_grid_width = 1
    else:
        feature_grid_height = int(feature_grid_shape[0])
        feature_grid_width = int(feature_grid_shape[1])
    FEATURE_GRID_HEIGHT = wp.constant(wp.int32(feature_grid_height))
    FEATURE_GRID_WIDTH = wp.constant(wp.int32(feature_grid_width))
    suffix_hash = warp_constant_suffix(
        block_size,
        feature_dim,
        num_cameras,
        image_height,
        image_width,
        num_samples,
        grid_shape,
        origin_xyz,
        voxel_size,
        truncation_distance,
        feature_grid_shape,
        feature_channels_per_thread,
        max_feature_tile_channels,
        max_support_pixels_per_block_camera,
        color_grid_size,
        feature_block_grid_size,
        instance_grid_size,
    )
    suffix = f"bs{block_size}_cfg{suffix_hash}"

    # Cross-domain helpers are explicit parameters so Warp sees them as
    # local closure bindings when compiling dependent kernels.
    FEATURE_CHANNELS_PER_THREAD = wp.constant(feature_channels_per_thread)
    feature_tile_channels = max(1, min(int(feature_dim), int(max_feature_tile_channels)))
    FEATURE_TILE_CHANNELS = wp.constant(feature_tile_channels)
    support_capacity = int(max_support_pixels_per_block_camera)
    SUPPORT_CAPACITY = wp.constant(support_capacity)
    INSTANCE_MATCH_INCREMENT = wp.constant(wp.int32(16))
    INSTANCE_MISMATCH_DECREMENT = wp.constant(wp.int32(32))
    INSTANCE_BACKGROUND_DECREMENT = wp.constant(wp.int32(2))
    INSTANCE_CONFIDENCE_MAX = wp.constant(wp.int32(255))

    @warp_func(f"sample_instance_slot_{suffix}")
    def sample_instance_slot(
        cam_idx: wp.int32,
        point_world: wp.vec3,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        instance_slots: wp.array3d(dtype=wp.uint8),
        depth_min: wp.float32,
        depth_max: wp.float32,
    ) -> wp.int32:
        """Project one map cell and return slot, background, or -1 if invalid."""
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
        point_cam = wp.quat_rotate(wp.quat_inverse(cam_quat), point_world - cam_pos)
        z_cam = point_cam[2]
        if z_cam <= depth_min or z_cam > depth_max:
            return wp.int32(-1)

        fx = intrinsics[cam_idx, 0, 0]
        fy = intrinsics[cam_idx, 1, 1]
        cx = intrinsics[cam_idx, 0, 2]
        cy = intrinsics[cam_idx, 1, 2]
        u = fx * point_cam[0] / z_cam + cx
        v = fy * point_cam[1] / z_cam + cy
        px = wp.int32(wp.floor(u + wp.float32(0.5)))
        py = wp.int32(wp.floor(v + wp.float32(0.5)))
        if px < wp.int32(0) or px >= IMAGE_WIDTH or py < wp.int32(0) or py >= IMAGE_HEIGHT:
            return wp.int32(-1)

        depth = depth_images[cam_idx, py, px]
        if depth < depth_min or depth > depth_max:
            return wp.int32(-1)
        if wp.abs(depth - z_cam) > TRUNCATION_DIST:
            return wp.int32(-1)
        return wp.int32(instance_slots[cam_idx, py, px])

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
        """Phase 1 (camera projective): emit only block keys, no sample data."""
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

    @warp_kernel(f"build_support_pixels_from_keys_kernel_{suffix}_sc{support_capacity}")
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

    @warp_kernel(f"clear_new_block_grid_rgb_kernel_{suffix}")
    def clear_new_block_grid_rgb_kernel(
        block_grid_rgb: wp.array3d(dtype=wp.float16),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
    ):
        """Zero RGB grid accumulators for blocks allocated during this frame."""
        slot_idx, cell_idx = wp.tid()

        count = new_block_count[0]
        if count > max_blocks:
            count = max_blocks
        if slot_idx >= count or cell_idx >= COLOR_GRID_CELLS:
            return

        pool_idx = new_blocks[slot_idx]
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return

        node_idx = cell_idx // wp.int32(4)
        ch = cell_idx - node_idx * wp.int32(4)
        block_grid_rgb[pool_idx, node_idx, ch] = wp.float16(0.0)

    @warp_kernel(f"clear_blocks_by_pool_kernel_{suffix}")
    def clear_blocks_by_pool_kernel(
        clear_pool_indices: wp.array(dtype=wp.int32),
        clear_count: wp.array(dtype=wp.int32),
        block_data: wp.array3d(dtype=wp.float16),
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

        if local_idx == wp.int32(0):
            block_sums[pool_idx] = wp.float32(0.0)

    @warp_kernel(f"clear_block_grid_rgb_by_pool_kernel_{suffix}")
    def clear_block_grid_rgb_by_pool_kernel(
        clear_pool_indices: wp.array(dtype=wp.int32),
        clear_count: wp.array(dtype=wp.int32),
        block_grid_rgb: wp.array3d(dtype=wp.float16),
        max_blocks: wp.int32,
    ):
        """Zero RGB grid accumulators for explicitly cleared block slots."""
        slot_idx, cell_idx = wp.tid()

        count = clear_count[0]
        if count > max_blocks:
            count = max_blocks
        if slot_idx >= count or cell_idx >= COLOR_GRID_CELLS:
            return

        pool_idx = clear_pool_indices[slot_idx]
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return

        node_idx = cell_idx // wp.int32(4)
        ch = cell_idx - node_idx * wp.int32(4)
        block_grid_rgb[pool_idx, node_idx, ch] = wp.float16(0.0)

    @warp_kernel(f"clear_block_features_by_pool_kernel_{suffix}")
    def clear_block_features_by_pool_kernel(
        clear_pool_indices: wp.array(dtype=wp.int32),
        clear_count: wp.array(dtype=wp.int32),
        block_features: wp.array3d(dtype=wp.float16),
        block_feature_weight: wp.array2d(dtype=wp.float16),
        max_blocks: wp.int32,
    ):
        """Zero feature-grid accumulators for already allocated blocks.

        Launch with ``dim = (n_clear, feature_grid_voxels, feature_dim)`` so
        one thread clears one ``(block, node, channel)`` cell; the thread
        with ``ch == 0`` also zeroes the per-node feature weight.
        """
        slot_idx, node_idx, ch = wp.tid()

        count = clear_count[0]
        if count > max_blocks:
            count = max_blocks
        if slot_idx >= count or node_idx >= FEATURE_GRID_VOXELS:
            return

        pool_idx = clear_pool_indices[slot_idx]
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return

        block_features[pool_idx, node_idx, ch] = wp.float16(0.0)
        if ch == wp.int32(0):
            block_feature_weight[pool_idx, node_idx] = wp.float16(0.0)

    @warp_kernel(f"clear_new_block_instances_kernel_{suffix}")
    def clear_new_block_instances_kernel(
        block_instance_state: wp.array2d(dtype=wp.int16),
        new_blocks: wp.array(dtype=wp.int32),
        new_block_count: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
    ):
        """Clear categorical state for blocks allocated during this frame."""
        slot_idx, node_idx = wp.tid()
        count = wp.min(new_block_count[0], max_blocks)
        if slot_idx >= count or node_idx >= INSTANCE_GRID_VOXELS:
            return
        pool_idx = new_blocks[slot_idx]
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return
        block_instance_state[pool_idx, node_idx] = wp.int16(0)

    @warp_kernel(f"clear_block_instances_by_pool_kernel_{suffix}")
    def clear_block_instances_by_pool_kernel(
        clear_pool_indices: wp.array(dtype=wp.int32),
        clear_count: wp.array(dtype=wp.int32),
        block_instance_state: wp.array2d(dtype=wp.int16),
        max_blocks: wp.int32,
    ):
        """Clear categorical state for explicit block-pool slots."""
        slot_idx, node_idx = wp.tid()
        count = wp.min(clear_count[0], max_blocks)
        if slot_idx >= count or node_idx >= INSTANCE_GRID_VOXELS:
            return
        pool_idx = clear_pool_indices[slot_idx]
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return
        block_instance_state[pool_idx, node_idx] = wp.int16(0)

    @warp_kernel(f"clear_instance_slot_kernel_{suffix}")
    def clear_instance_slot_kernel(
        block_instance_state: wp.array2d(dtype=wp.int16),
        num_allocated: wp.array(dtype=wp.int32),
        instance_slot: wp.int32,
        max_blocks: wp.int32,
    ):
        """Clear every categorical cell owned by one retired registry slot."""
        pool_idx, node_idx = wp.tid()
        active_count = wp.min(num_allocated[0], max_blocks)
        if pool_idx >= active_count or node_idx >= INSTANCE_GRID_VOXELS:
            return
        packed = wp.int32(block_instance_state[pool_idx, node_idx]) & wp.int32(0xFFFF)
        if (packed & wp.int32(0xFF)) == instance_slot:
            block_instance_state[pool_idx, node_idx] = wp.int16(0)

    @warp_kernel(f"mark_instance_removal_blocks_kernel_{suffix}")
    def mark_instance_removal_blocks_kernel(
        hash_table: wp.array(dtype=wp.int64),
        block_coords: wp.array(dtype=wp.int32),
        block_to_hash_slot: wp.array(dtype=wp.int32),
        block_instance_state: wp.array2d(dtype=wp.int16),
        num_allocated: wp.array(dtype=wp.int32),
        instance_slot: wp.int32,
        hash_capacity: wp.int32,
        max_blocks: wp.int32,
        block_remove_mask: wp.array(dtype=wp.uint8),
    ):
        """Mark active blocks adjacent to ownership by one instance."""
        pool_idx = wp.tid()
        active_count = wp.min(num_allocated[0], max_blocks)
        if pool_idx >= active_count:
            return
        if block_to_hash_slot[pool_idx] < wp.int32(0):
            block_remove_mask[pool_idx] = wp.uint8(0)
            return

        bx = block_coords[pool_idx * wp.int32(3)]
        by = block_coords[pool_idx * wp.int32(3) + wp.int32(1)]
        bz = block_coords[pool_idx * wp.int32(3) + wp.int32(2)]
        block_has_instance = bool(False)
        for neighbor_idx in range(27):
            dx = neighbor_idx % wp.int32(3) - wp.int32(1)
            dy = (neighbor_idx // wp.int32(3)) % wp.int32(3) - wp.int32(1)
            dz = neighbor_idx // wp.int32(9) - wp.int32(1)
            neighbor_pool_idx = hash_lookup(
                hash_table,
                bx + dx,
                by + dy,
                bz + dz,
                hash_capacity,
            )
            if neighbor_pool_idx >= wp.int32(0):
                for node_idx in range(instance_grid_voxels):
                    packed = wp.int32(
                        block_instance_state[neighbor_pool_idx, node_idx]
                    ) & wp.int32(0xFFFF)
                    if (packed & wp.int32(0xFF)) == instance_slot:
                        block_has_instance = True
        block_remove_mask[pool_idx] = wp.uint8(1) if block_has_instance else wp.uint8(0)

    @warp_kernel(f"remove_instance_geometry_kernel_{suffix}")
    def remove_instance_geometry_kernel(
        block_data: wp.array3d(dtype=wp.float16),
        block_sums: wp.array(dtype=wp.float32),
        block_grid_rgb: wp.array3d(dtype=wp.float16),
        block_features: wp.array3d(dtype=wp.float16),
        block_feature_weight: wp.array2d(dtype=wp.float16),
        block_instance_state: wp.array2d(dtype=wp.int16),
        block_remove_mask: wp.array(dtype=wp.uint8),
        num_allocated: wp.array(dtype=wp.int32),
        instance_slot: wp.int32,
        semantic_class_mask_low: wp.uint32,
        semantic_class_mask_high: wp.uint32,
        max_blocks: wp.int32,
    ):
        """Remove dynamic data selected by instance ownership and semantic class."""
        pool_idx, local_idx = wp.tid()
        active_count = wp.min(num_allocated[0], max_blocks)
        if pool_idx >= active_count or local_idx >= BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE:
            return

        zero_mask = wp.uint32(0)
        use_semantic_mask = (
            semantic_class_mask_low != zero_mask or semantic_class_mask_high != zero_mask
        )
        if use_semantic_mask:
            if block_remove_mask[pool_idx] == wp.uint8(0):
                return
        else:
            x = local_idx % BLOCK_SIZE
            y = (local_idx // BLOCK_SIZE) % BLOCK_SIZE
            z = local_idx // (BLOCK_SIZE * BLOCK_SIZE)
            instance_x = wp.min(
                INSTANCE_GRID_SIZE - wp.int32(1),
                x * INSTANCE_GRID_SIZE // BLOCK_SIZE,
            )
            instance_y = wp.min(
                INSTANCE_GRID_SIZE - wp.int32(1),
                y * INSTANCE_GRID_SIZE // BLOCK_SIZE,
            )
            instance_z = wp.min(
                INSTANCE_GRID_SIZE - wp.int32(1),
                z * INSTANCE_GRID_SIZE // BLOCK_SIZE,
            )
            instance_node = (
                instance_z * INSTANCE_GRID_SIZE * INSTANCE_GRID_SIZE
                + instance_y * INSTANCE_GRID_SIZE
                + instance_x
            )
            packed = wp.int32(block_instance_state[pool_idx, instance_node]) & wp.int32(0xFFFF)
            if (packed & wp.int32(0xFF)) != instance_slot:
                return

        if use_semantic_mask:
            semantic_class = wp.int32(0)
            semantic_score = wp.float32(block_features[pool_idx, local_idx, 0])
            for channel_idx in range(1, feature_dim):
                score = wp.float32(block_features[pool_idx, local_idx, channel_idx])
                if score > semantic_score:
                    semantic_score = score
                    semantic_class = channel_idx
            class_selected = bool(False)
            if semantic_class < wp.int32(32):
                class_selected = (
                    semantic_class_mask_low & (wp.uint32(1) << wp.uint32(semantic_class))
                ) != wp.uint32(0)
            else:
                class_selected = (
                    semantic_class_mask_high
                    & (wp.uint32(1) << wp.uint32(semantic_class - wp.int32(32)))
                ) != wp.uint32(0)
            if not class_selected:
                return

        old_weight = wp.float32(block_data[pool_idx, local_idx, 1])
        block_data[pool_idx, local_idx, 0] = wp.float16(0.0)
        block_data[pool_idx, local_idx, 1] = wp.float16(0.0)
        if old_weight != wp.float32(0.0):
            wp.atomic_add(block_sums, pool_idx, -old_weight)

        if COLOR_GRID_SIZE == BLOCK_SIZE:
            for channel_idx in range(4):
                block_grid_rgb[pool_idx, local_idx, channel_idx] = wp.float16(0.0)
        if FEATURE_BLOCK_GRID_SIZE == BLOCK_SIZE:
            for channel_idx in range(feature_dim):
                block_features[pool_idx, local_idx, channel_idx] = wp.float16(0.0)
            block_feature_weight[pool_idx, local_idx] = wp.float16(0.0)

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
        """Phase 4 (camera projective): one thread per voxel, serial camera loop.

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

    @warp_kernel(f"integrate_block_grid_rgb_kernel_{suffix}")
    def integrate_block_grid_rgb_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        rgb_images_flat: wp.array2d(dtype=wp.vec3ub),
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        depth_min: float,
        depth_max: float,
        block_coords: wp.array(dtype=wp.int32),
        block_grid_rgb: wp.array3d(dtype=wp.float16),
    ):
        """Project each RGB-grid node into cameras and integrate weighted RGBW."""
        vis_idx, node_idx = wp.tid()
        if vis_idx >= n_visible or node_idx >= COLOR_GRID_VOXELS:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < 0:
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]

        gx_node = node_idx % COLOR_GRID_SIZE
        gy_node = (node_idx // COLOR_GRID_SIZE) % COLOR_GRID_SIZE
        gz_node = node_idx // (COLOR_GRID_SIZE * COLOR_GRID_SIZE)

        local_x = wp.float32(0.0)
        local_y = wp.float32(0.0)
        local_z = wp.float32(0.0)
        if COLOR_GRID_SIZE > wp.int32(1):
            span = wp.float32(BLOCK_SIZE - wp.int32(1))
            denom = wp.float32(COLOR_GRID_SIZE - wp.int32(1))
            local_x = wp.float32(0.5) + wp.float32(gx_node) * span / denom
            local_y = wp.float32(0.5) + wp.float32(gy_node) * span / denom
            local_z = wp.float32(0.5) + wp.float32(gz_node) * span / denom
        else:
            center = wp.float32(BLOCK_SIZE) * wp.float32(0.5)
            local_x = center
            local_y = center
            local_z = center

        base = block_key_to_voxel_base(bx, by, bz)
        center_offset_x = wp.float32(GRID_W) * wp.float32(0.5)
        center_offset_y = wp.float32(GRID_H) * wp.float32(0.5)
        center_offset_z = wp.float32(GRID_D) * wp.float32(0.5)
        node_world = (
            wp.vec3(ORIGIN_X, ORIGIN_Y, ORIGIN_Z)
            + wp.vec3(
                wp.float32(base[0]) + local_x - center_offset_x,
                wp.float32(base[1]) + local_y - center_offset_y,
                wp.float32(base[2]) + local_z - center_offset_z,
            )
            * VOXEL_SIZE
        )

        inv_255 = wp.float32(1.0 / 255.0)
        total_r = wp.float32(0.0)
        total_g = wp.float32(0.0)
        total_b = wp.float32(0.0)
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
            node_cam = wp.quat_rotate(cam_quat_inv, node_world - cam_pos)

            z_cam = node_cam[2]
            if z_cam > depth_min and z_cam <= depth_max:
                fx = intrinsics[cam_i, 0, 0]
                fy = intrinsics[cam_i, 1, 1]
                cx_i = intrinsics[cam_i, 0, 2]
                cy_i = intrinsics[cam_i, 1, 2]

                u = fx * node_cam[0] / z_cam + cx_i
                v = fy * node_cam[1] / z_cam + cy_i

                coverage = (fx * VOXEL_SIZE / z_cam) * (fy * VOXEL_SIZE / z_cam)
                coverage_weight = wp.max(coverage, 1.0)

                if (
                    u >= wp.float32(0.0)
                    and u <= wp.float32(IMAGE_WIDTH - wp.int32(1))
                    and v >= wp.float32(0.0)
                    and v <= wp.float32(IMAGE_HEIGHT - wp.int32(1))
                ):
                    px0 = wp.int32(wp.floor(u))
                    py0 = wp.int32(wp.floor(v))
                    px1 = wp.min(px0 + wp.int32(1), IMAGE_WIDTH - wp.int32(1))
                    py1 = wp.min(py0 + wp.int32(1), IMAGE_HEIGHT - wp.int32(1))
                    tx = u - wp.float32(px0)
                    ty = v - wp.float32(py0)
                    wx0 = wp.float32(1.0) - tx
                    wy0 = wp.float32(1.0) - ty
                    w00 = wx0 * wy0
                    w10 = tx * wy0
                    w01 = wx0 * ty
                    w11 = tx * ty

                    depth00 = depth_images[cam_i, py0, px0]
                    if depth00 >= depth_min and depth00 <= depth_max:
                        sdf00 = depth00 - z_cam
                        if sdf00 >= -TRUNCATION_DIST and sdf00 <= TRUNCATION_DIST:
                            weight00 = (
                                w00 * compute_tsdf_weight(depth00, VOXEL_SIZE) * coverage_weight
                            )
                            row00 = cam_i * IMAGE_HEIGHT + py0
                            rgb00 = rgb_images_flat[row00, px0]
                            total_r = total_r + wp.float32(rgb00[0]) * inv_255 * weight00
                            total_g = total_g + wp.float32(rgb00[1]) * inv_255 * weight00
                            total_b = total_b + wp.float32(rgb00[2]) * inv_255 * weight00
                            total_w = total_w + weight00

                    depth10 = depth_images[cam_i, py0, px1]
                    if depth10 >= depth_min and depth10 <= depth_max:
                        sdf10 = depth10 - z_cam
                        if sdf10 >= -TRUNCATION_DIST and sdf10 <= TRUNCATION_DIST:
                            weight10 = (
                                w10 * compute_tsdf_weight(depth10, VOXEL_SIZE) * coverage_weight
                            )
                            row10 = cam_i * IMAGE_HEIGHT + py0
                            rgb10 = rgb_images_flat[row10, px1]
                            total_r = total_r + wp.float32(rgb10[0]) * inv_255 * weight10
                            total_g = total_g + wp.float32(rgb10[1]) * inv_255 * weight10
                            total_b = total_b + wp.float32(rgb10[2]) * inv_255 * weight10
                            total_w = total_w + weight10

                    depth01 = depth_images[cam_i, py1, px0]
                    if depth01 >= depth_min and depth01 <= depth_max:
                        sdf01 = depth01 - z_cam
                        if sdf01 >= -TRUNCATION_DIST and sdf01 <= TRUNCATION_DIST:
                            weight01 = (
                                w01 * compute_tsdf_weight(depth01, VOXEL_SIZE) * coverage_weight
                            )
                            row01 = cam_i * IMAGE_HEIGHT + py1
                            rgb01 = rgb_images_flat[row01, px0]
                            total_r = total_r + wp.float32(rgb01[0]) * inv_255 * weight01
                            total_g = total_g + wp.float32(rgb01[1]) * inv_255 * weight01
                            total_b = total_b + wp.float32(rgb01[2]) * inv_255 * weight01
                            total_w = total_w + weight01

                    depth11 = depth_images[cam_i, py1, px1]
                    if depth11 >= depth_min and depth11 <= depth_max:
                        sdf11 = depth11 - z_cam
                        if sdf11 >= -TRUNCATION_DIST and sdf11 <= TRUNCATION_DIST:
                            weight11 = (
                                w11 * compute_tsdf_weight(depth11, VOXEL_SIZE) * coverage_weight
                            )
                            row11 = cam_i * IMAGE_HEIGHT + py1
                            rgb11 = rgb_images_flat[row11, px1]
                            total_r = total_r + wp.float32(rgb11[0]) * inv_255 * weight11
                            total_g = total_g + wp.float32(rgb11[1]) * inv_255 * weight11
                            total_b = total_b + wp.float32(rgb11[2]) * inv_255 * weight11
                            total_w = total_w + weight11

        old_r = wp.float32(block_grid_rgb[pool_idx, node_idx, 0])
        old_g = wp.float32(block_grid_rgb[pool_idx, node_idx, 1])
        old_b = wp.float32(block_grid_rgb[pool_idx, node_idx, 2])
        old_w = wp.float32(block_grid_rgb[pool_idx, node_idx, 3])
        if total_w > wp.float32(0.0):
            block_grid_rgb[pool_idx, node_idx, 0] = wp.float16(old_r + total_r)
            block_grid_rgb[pool_idx, node_idx, 1] = wp.float16(old_g + total_g)
            block_grid_rgb[pool_idx, node_idx, 2] = wp.float16(old_b + total_b)
            block_grid_rgb[pool_idx, node_idx, 3] = wp.float16(old_w + total_w)
            return

        support_r = wp.float32(0.0)
        support_g = wp.float32(0.0)
        support_b = wp.float32(0.0)
        support_w = wp.float32(0.0)
        for cam_i in range(num_cameras):
            count = support_counts[vis_idx, cam_i]
            if count > wp.int32(0):
                pixel_idx = support_pixels[vis_idx, cam_i, 0]
                py = pixel_idx // IMAGE_WIDTH
                px = pixel_idx - py * IMAGE_WIDTH
                if (
                    py >= wp.int32(0)
                    and py < IMAGE_HEIGHT
                    and px >= wp.int32(0)
                    and px < IMAGE_WIDTH
                ):
                    row = cam_i * IMAGE_HEIGHT + py
                    rgb = rgb_images_flat[row, px]
                    support_r = support_r + wp.float32(rgb[0]) * inv_255
                    support_g = support_g + wp.float32(rgb[1]) * inv_255
                    support_b = support_b + wp.float32(rgb[2]) * inv_255
                    support_w = support_w + wp.float32(1.0)

        if support_w > wp.float32(0.0):
            block_grid_rgb[pool_idx, node_idx, 0] = wp.float16(old_r + support_r)
            block_grid_rgb[pool_idx, node_idx, 1] = wp.float16(old_g + support_g)
            block_grid_rgb[pool_idx, node_idx, 2] = wp.float16(old_b + support_b)
            block_grid_rgb[pool_idx, node_idx, 3] = wp.float16(old_w + support_w)

    @warp_kernel(f"integrate_instance_grid_kernel_{suffix}")
    def integrate_instance_grid_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        instance_slots: wp.array3d(dtype=wp.uint8),
        depth_min: wp.float32,
        depth_max: wp.float32,
        block_coords: wp.array(dtype=wp.int32),
        block_instance_state: wp.array2d(dtype=wp.int16),
    ):
        """Fuse multiview categorical votes and update temporal ownership."""
        vis_idx, node_idx = wp.tid()
        if vis_idx >= n_visible or node_idx >= INSTANCE_GRID_VOXELS:
            return
        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]
        gx = node_idx % INSTANCE_GRID_SIZE
        gy = (node_idx // INSTANCE_GRID_SIZE) % INSTANCE_GRID_SIZE
        gz = node_idx // (INSTANCE_GRID_SIZE * INSTANCE_GRID_SIZE)
        cell_width = wp.float32(BLOCK_SIZE) / wp.float32(INSTANCE_GRID_SIZE)
        local_x = (wp.float32(gx) + wp.float32(0.5)) * cell_width
        local_y = (wp.float32(gy) + wp.float32(0.5)) * cell_width
        local_z = (wp.float32(gz) + wp.float32(0.5)) * cell_width
        base = block_key_to_voxel_base(bx, by, bz)
        node_world = (
            wp.vec3(ORIGIN_X, ORIGIN_Y, ORIGIN_Z)
            + wp.vec3(
                wp.float32(base[0]) + local_x - wp.float32(GRID_W) * wp.float32(0.5),
                wp.float32(base[1]) + local_y - wp.float32(GRID_H) * wp.float32(0.5),
                wp.float32(base[2]) + local_z - wp.float32(GRID_D) * wp.float32(0.5),
            )
            * VOXEL_SIZE
        )

        valid_views = wp.int32(0)
        background_views = wp.int32(0)
        best_slot = wp.int32(0)
        best_votes = wp.int32(0)
        tied = False
        for candidate_cam in range(num_cameras):
            candidate_slot = sample_instance_slot(
                candidate_cam,
                node_world,
                intrinsics,
                cam_positions,
                cam_quaternions,
                depth_images,
                instance_slots,
                depth_min,
                depth_max,
            )
            if candidate_slot >= wp.int32(0):
                valid_views = valid_views + wp.int32(1)
                if candidate_slot == wp.int32(0):
                    background_views = background_views + wp.int32(1)
            if candidate_slot > wp.int32(0):
                candidate_votes = wp.int32(0)
                for vote_cam in range(num_cameras):
                    vote_slot = sample_instance_slot(
                        vote_cam,
                        node_world,
                        intrinsics,
                        cam_positions,
                        cam_quaternions,
                        depth_images,
                        instance_slots,
                        depth_min,
                        depth_max,
                    )
                    if vote_slot == candidate_slot:
                        candidate_votes = candidate_votes + wp.int32(1)
                if candidate_votes > best_votes:
                    best_slot = candidate_slot
                    best_votes = candidate_votes
                    tied = False
                elif candidate_slot != best_slot and candidate_votes == best_votes:
                    tied = True

        packed = wp.int32(block_instance_state[pool_idx, node_idx]) & wp.int32(0xFFFF)
        current_slot = packed & wp.int32(0xFF)
        confidence = (packed >> wp.int32(8)) & wp.int32(0xFF)
        if best_votes > wp.int32(0) and not tied:
            if current_slot == best_slot:
                confidence = wp.min(
                    INSTANCE_CONFIDENCE_MAX,
                    confidence + best_votes * INSTANCE_MATCH_INCREMENT,
                )
            elif current_slot == wp.int32(0):
                current_slot = best_slot
                confidence = wp.min(
                    INSTANCE_CONFIDENCE_MAX,
                    best_votes * INSTANCE_MATCH_INCREMENT,
                )
            else:
                contradiction = best_votes * INSTANCE_MISMATCH_DECREMENT
                if contradiction >= confidence:
                    current_slot = best_slot
                    confidence = wp.min(
                        INSTANCE_CONFIDENCE_MAX,
                        best_votes * INSTANCE_MATCH_INCREMENT,
                    )
                else:
                    confidence = confidence - contradiction
        elif valid_views > wp.int32(0) and background_views == valid_views:
            background_evidence = background_views * INSTANCE_BACKGROUND_DECREMENT
            if background_evidence >= confidence:
                current_slot = wp.int32(0)
                confidence = wp.int32(0)
            else:
                confidence = confidence - background_evidence
        else:
            return

        block_instance_state[pool_idx, node_idx] = wp.int16(
            (confidence << wp.int32(8)) | current_slot
        )

    @warp_kernel(
        f"integrate_features_from_support_grouped_kernel_{suffix}_fcpt"
        f"{feature_channels_per_thread}_sc{support_capacity}"
    )
    def integrate_features_from_support_grouped_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        feature_grid: wp.array4d(dtype=wp.float16),
        depth_min: float,
        depth_max: float,
        block_coords: wp.array(dtype=wp.int32),
        block_features: wp.array3d(dtype=wp.float16),
        block_feature_weight: wp.array2d(dtype=wp.float16),
    ):
        """Per-node feature-grid integration with support fallback for empty nodes."""
        n_channel_groups = (
            FEATURE_DIM + FEATURE_CHANNELS_PER_THREAD - wp.int32(1)
        ) // FEATURE_CHANNELS_PER_THREAD
        vis_idx, node_idx, feature_channel_group_idx = wp.tid()

        if (
            vis_idx >= n_visible
            or node_idx >= FEATURE_GRID_VOXELS
            or feature_channel_group_idx >= n_channel_groups
        ):
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]

        gx_node = node_idx % FEATURE_BLOCK_GRID_SIZE
        gy_node = (node_idx // FEATURE_BLOCK_GRID_SIZE) % FEATURE_BLOCK_GRID_SIZE
        gz_node = node_idx // (FEATURE_BLOCK_GRID_SIZE * FEATURE_BLOCK_GRID_SIZE)

        local_x = wp.float32(0.0)
        local_y = wp.float32(0.0)
        local_z = wp.float32(0.0)
        if FEATURE_BLOCK_GRID_SIZE > wp.int32(1):
            span = wp.float32(BLOCK_SIZE - wp.int32(1))
            denom = wp.float32(FEATURE_BLOCK_GRID_SIZE - wp.int32(1))
            local_x = wp.float32(0.5) + wp.float32(gx_node) * span / denom
            local_y = wp.float32(0.5) + wp.float32(gy_node) * span / denom
            local_z = wp.float32(0.5) + wp.float32(gz_node) * span / denom
        else:
            center = wp.float32(BLOCK_SIZE) * wp.float32(0.5)
            local_x = center
            local_y = center
            local_z = center

        base = block_key_to_voxel_base(bx, by, bz)
        center_offset_x = wp.float32(GRID_W) * wp.float32(0.5)
        center_offset_y = wp.float32(GRID_H) * wp.float32(0.5)
        center_offset_z = wp.float32(GRID_D) * wp.float32(0.5)
        node_world = (
            wp.vec3(ORIGIN_X, ORIGIN_Y, ORIGIN_Z)
            + wp.vec3(
                wp.float32(base[0]) + local_x - center_offset_x,
                wp.float32(base[1]) + local_y - center_offset_y,
                wp.float32(base[2]) + local_z - center_offset_z,
            )
            * VOXEL_SIZE
        )

        base_feature_channel = feature_channel_group_idx * FEATURE_CHANNELS_PER_THREAD
        feature_acc = wp.zeros(FEATURE_CHANNELS_PER_THREAD, dtype=wp.float32)
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
            node_cam = wp.quat_rotate(wp.quat_inverse(cam_quat), node_world - cam_pos)
            z_cam = node_cam[2]
            if z_cam > depth_min and z_cam <= depth_max:
                fx = intrinsics[cam_i, 0, 0]
                fy = intrinsics[cam_i, 1, 1]
                cx_i = intrinsics[cam_i, 0, 2]
                cy_i = intrinsics[cam_i, 1, 2]
                u = fx * node_cam[0] / z_cam + cx_i
                v = fy * node_cam[1] / z_cam + cy_i
                px = wp.int32(wp.floor(u + wp.float32(0.5)))
                py = wp.int32(wp.floor(v + wp.float32(0.5)))
                if px >= 0 and px < IMAGE_WIDTH and py >= 0 and py < IMAGE_HEIGHT:
                    depth = depth_images[cam_i, py, px]
                    sdf = depth - z_cam
                    if (
                        depth >= depth_min
                        and depth <= depth_max
                        and sdf >= -TRUNCATION_DIST
                        and sdf <= TRUNCATION_DIST
                    ):
                        weight = compute_tsdf_weight(depth, VOXEL_SIZE)
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
                                    * weight
                                )
                        total_w = total_w + weight

        if total_w <= wp.float32(0.0):
            for cam_i in range(num_cameras):
                count = support_counts[vis_idx, cam_i]
                if count > wp.int32(0):
                    pixel_idx = support_pixels[vis_idx, cam_i, 0]
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
                                feature_acc[feature_channel_offset] = feature_acc[
                                    feature_channel_offset
                                ] + wp.float32(
                                    feature_grid[
                                        cam_i,
                                        gy,
                                        gx,
                                        feature_channel,
                                    ]
                                )
                        total_w = total_w + wp.float32(1.0)

        if total_w <= wp.float32(0.0):
            return

        for feature_channel_offset in range(FEATURE_CHANNELS_PER_THREAD):
            feature_channel = base_feature_channel + feature_channel_offset
            if feature_channel < FEATURE_DIM:
                wp.atomic_add(
                    block_features,
                    pool_idx,
                    node_idx,
                    feature_channel,
                    wp.float16(feature_acc[feature_channel_offset]),
                )
        if base_feature_channel == wp.int32(0):
            wp.atomic_add(block_feature_weight, pool_idx, node_idx, wp.float16(total_w))

    @warp_kernel(
        f"integrate_features_from_support_tiled_kernel_{suffix}_tile"
        f"{feature_tile_channels}_sc{support_capacity}"
    )
    def integrate_features_from_support_tiled_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        feature_grid: wp.array4d(dtype=wp.float16),
        depth_min: float,
        depth_max: float,
        block_coords: wp.array(dtype=wp.int32),
        block_features: wp.array3d(dtype=wp.float16),
        block_feature_weight: wp.array2d(dtype=wp.float16),
    ):
        """Tiled per-node feature-grid integration."""
        n_channel_tiles = (
            FEATURE_DIM + FEATURE_TILE_CHANNELS - wp.int32(1)
        ) // FEATURE_TILE_CHANNELS
        vis_idx, node_idx, feature_tile_idx, lane = wp.tid()

        if (
            vis_idx >= n_visible
            or node_idx >= FEATURE_GRID_VOXELS
            or feature_tile_idx >= n_channel_tiles
        ):
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]

        gx_node = node_idx % FEATURE_BLOCK_GRID_SIZE
        gy_node = (node_idx // FEATURE_BLOCK_GRID_SIZE) % FEATURE_BLOCK_GRID_SIZE
        gz_node = node_idx // (FEATURE_BLOCK_GRID_SIZE * FEATURE_BLOCK_GRID_SIZE)

        local_x = wp.float32(0.0)
        local_y = wp.float32(0.0)
        local_z = wp.float32(0.0)
        if FEATURE_BLOCK_GRID_SIZE > wp.int32(1):
            span = wp.float32(BLOCK_SIZE - wp.int32(1))
            denom = wp.float32(FEATURE_BLOCK_GRID_SIZE - wp.int32(1))
            local_x = wp.float32(0.5) + wp.float32(gx_node) * span / denom
            local_y = wp.float32(0.5) + wp.float32(gy_node) * span / denom
            local_z = wp.float32(0.5) + wp.float32(gz_node) * span / denom
        else:
            center = wp.float32(BLOCK_SIZE) * wp.float32(0.5)
            local_x = center
            local_y = center
            local_z = center

        base = block_key_to_voxel_base(bx, by, bz)
        center_offset_x = wp.float32(GRID_W) * wp.float32(0.5)
        center_offset_y = wp.float32(GRID_H) * wp.float32(0.5)
        center_offset_z = wp.float32(GRID_D) * wp.float32(0.5)
        node_world = (
            wp.vec3(ORIGIN_X, ORIGIN_Y, ORIGIN_Z)
            + wp.vec3(
                wp.float32(base[0]) + local_x - center_offset_x,
                wp.float32(base[1]) + local_y - center_offset_y,
                wp.float32(base[2]) + local_z - center_offset_z,
            )
            * VOXEL_SIZE
        )

        base_feature_channel = feature_tile_idx * FEATURE_TILE_CHANNELS
        feature_acc = wp.tile_zeros(shape=feature_tile_channels, dtype=wp.float32)
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
            node_cam = wp.quat_rotate(wp.quat_inverse(cam_quat), node_world - cam_pos)
            z_cam = node_cam[2]
            if z_cam > depth_min and z_cam <= depth_max:
                fx = intrinsics[cam_i, 0, 0]
                fy = intrinsics[cam_i, 1, 1]
                cx_i = intrinsics[cam_i, 0, 2]
                cy_i = intrinsics[cam_i, 1, 2]
                u = fx * node_cam[0] / z_cam + cx_i
                v = fy * node_cam[1] / z_cam + cy_i
                px = wp.int32(wp.floor(u + wp.float32(0.5)))
                py = wp.int32(wp.floor(v + wp.float32(0.5)))
                if px >= 0 and px < IMAGE_WIDTH and py >= 0 and py < IMAGE_HEIGHT:
                    depth = depth_images[cam_i, py, px]
                    sdf = depth - z_cam
                    if (
                        depth >= depth_min
                        and depth <= depth_max
                        and sdf >= -TRUNCATION_DIST
                        and sdf <= TRUNCATION_DIST
                    ):
                        weight = compute_tsdf_weight(depth, VOXEL_SIZE)
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
                        feature_acc = (
                            feature_acc
                            + wp.tile_astype(
                                feature_vals_h,
                                dtype=wp.float32,
                            )
                            * weight
                        )
                        total_w = total_w + weight

        if total_w <= wp.float32(0.0):
            for cam_i in range(num_cameras):
                count = support_counts[vis_idx, cam_i]
                if count > wp.int32(0):
                    pixel_idx = support_pixels[vis_idx, cam_i, 0]
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
                        total_w = total_w + wp.float32(1.0)

        if total_w <= wp.float32(0.0):
            return

        feature_acc_h = wp.tile_astype(feature_acc, dtype=wp.float16)
        wp.tile_atomic_add(
            block_features[pool_idx, node_idx],
            feature_acc_h,
            offset=base_feature_channel,
            bounds_check=True,
        )
        if feature_tile_idx == wp.int32(0) and lane == wp.int32(0):
            wp.atomic_add(block_feature_weight, pool_idx, node_idx, wp.float16(total_w))

    return {
        "compute_block_keys_only_kernel": compute_block_keys_only_kernel,
        "allocate_visible_blocks_from_keys_kernel": allocate_visible_blocks_from_keys_kernel,
        "build_support_pixels_from_keys_kernel": build_support_pixels_from_keys_kernel,
        "collect_blocks_in_aabb_kernel": collect_blocks_in_aabb_kernel,
        "clear_new_block_grid_rgb_kernel": clear_new_block_grid_rgb_kernel,
        "clear_new_block_instances_kernel": clear_new_block_instances_kernel,
        "clear_blocks_by_pool_kernel": clear_blocks_by_pool_kernel,
        "clear_block_grid_rgb_by_pool_kernel": clear_block_grid_rgb_by_pool_kernel,
        "clear_block_features_by_pool_kernel": clear_block_features_by_pool_kernel,
        "clear_block_instances_by_pool_kernel": clear_block_instances_by_pool_kernel,
        "clear_instance_slot_kernel": clear_instance_slot_kernel,
        "mark_instance_removal_blocks_kernel": mark_instance_removal_blocks_kernel,
        "remove_instance_geometry_kernel": remove_instance_geometry_kernel,
        "integrate_voxels_kernel": integrate_voxels_kernel,
        "integrate_block_grid_rgb_kernel": integrate_block_grid_rgb_kernel,
        "integrate_instance_grid_kernel": integrate_instance_grid_kernel,
        "integrate_features_from_support_grouped_kernel": (
            integrate_features_from_support_grouped_kernel
        ),
        "integrate_features_from_support_tiled_kernel": (
            integrate_features_from_support_tiled_kernel
        ),
        "integrate_features_grouped_kernel": integrate_features_from_support_grouped_kernel,
    }
