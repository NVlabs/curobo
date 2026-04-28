# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF weight decay + recycling kernels, per-``block_size`` builder.

Moved from :mod:`curobo._src.perception.mapper.kernel.wp_decay` in the
block-size builder refactor.

The BS-sensitive piece is ``BLOCK_EMPTY_THRESHOLD``, a ``wp.constant``
baked into :func:`recycle_empty_blocks_kernel`. It scales with
``block_size ** 3`` so that the time-to-recycle for actively observed
surface blocks is invariant across block sizes. See the long rationale
below.

``mark_blocks_in_frustum_kernel`` takes ``block_size`` as a runtime
parameter (not a closure-captured constant) and therefore doesn't
strictly require per-BS codegen. It lives here anyway so that both
decay kernels dispatch through the same :class:`BlockSparseKernels`
instance, keeping the launch-wrapper side of ``wp_decay.py`` simple.
"""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.constants import (
    HASH_TOMBSTONE,
    REFERENCE_BLOCK_SIZE,
)
from curobo._src.util.warp import warp_kernel

# Block recycle threshold at the anchor BS=``REFERENCE_BLOCK_SIZE``. A block
# is recycled when its sum-of-weights falls below this value, scaled by
# (block_size / REFERENCE_BLOCK_SIZE) ** 3.
#
# Scaling rationale:
#
#   ``block_sums[i]`` is the sum of per-voxel weights over all observed
#   voxels in a block. The dominant case for surface blocks depends on
#   how the surface band (2 × truncation_distance thick) compares to the
#   block edge length:
#
#     - block_edge < truncation_dist: the entire block sits inside the
#       surface band, so ALL B^3 voxels accumulate weight and
#       ``block_sums`` scales as O(block_size^3).
#     - block_edge >> truncation_dist: only a thin slice (~B^2 voxels)
#       accumulates weight and ``block_sums`` scales as O(block_size^2).
#
#   With the typical default config (voxel_size=0.005m,
#   truncation_distance=0.04m), block_edge equals truncation_dist at
#   block_size=8 and is strictly smaller for block_size in {2, 4}, so the
#   B^3 regime dominates across all supported block sizes.
#
#   B^3 scaling makes the time-to-recycle invariant across block_size for
#   actively observed surface blocks: starting weight K ∝ B^3, threshold
#   ∝ B^3, so ``log(threshold / K) / log(decay_factor)`` is independent
#   of block_size. This avoids two failure modes:
#
#     - Constant threshold (B^0) → smaller blocks recycle much faster
#       than larger ones (~6 frames sooner at block_size=2 vs 8), which
#       matches the previously reported "vanishing voxels" symptom.
#     - Surface scaling (B^2) → recycling is slightly more aggressive at
#       smaller block sizes (~1 frame), which still creates extra hash
#       table churn. With the 64-probe linear-probing limit, that churn
#       can produce localized allocation failures ("no new voxels in
#       some regions" reported by the user).
#
# Calibration: at the historical block_size=8 this evaluates to 0.01,
# preserving legacy behavior.
_REFERENCE_BLOCK_EMPTY_THRESHOLD: float = 0.01


def make_decay_kernels(block_size: int, *, free_list_push) -> dict[str, object]:
    """Build TSDF weight-decay and block-recycling kernels."""
    # BS^3-scaled empty-weight threshold, closure-captured per
    # specialization.
    block_empty_threshold = wp.constant(
        _REFERENCE_BLOCK_EMPTY_THRESHOLD * float(block_size**3) / float(REFERENCE_BLOCK_SIZE**3)
    )

    # =====================================================================
    # Block-Level Frustum Marking Kernel (Pass 1)
    # =====================================================================

    @warp_kernel(f"mark_blocks_in_frustum_kernel_bs{block_size}")
    def mark_blocks_in_frustum_kernel(
        block_coords: wp.array(dtype=wp.int32),
        block_to_hash_slot: wp.array(dtype=wp.int32),
        num_allocated: wp.array(dtype=wp.int32),
        origin: wp.array(dtype=wp.float32),
        voxel_size: float,
        block_size: wp.int32,
        grid_W: wp.int32,
        grid_H: wp.int32,
        grid_D: wp.int32,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        n_cameras: wp.int32,
        img_H: wp.int32,
        img_W: wp.int32,
        depth_minimum_distance: float,
        depth_maximum_distance: float,
        block_in_frustum: wp.array(dtype=wp.int32),
        max_blocks: wp.int32,
    ):
        """Mark blocks visible in ANY camera's frustum.

        Parallelized across (block, camera) pairs. Multiple threads
        may write ``1`` to the same ``block_in_frustum[block_idx]``;
        the write is idempotent so no atomics are needed.

        Launch with ``dim = (max_blocks, n_cameras)``. The flags
        array must be zeroed before launch.
        """
        block_idx, cam_i = wp.tid()

        if block_idx >= max_blocks:
            return
        if block_idx >= num_allocated[0]:
            return
        if block_to_hash_slot[block_idx] < 0:
            return

        bx_key = block_coords[block_idx * 3 + 0]
        by_key = block_coords[block_idx * 3 + 1]
        bz_key = block_coords[block_idx * 3 + 2]
        blocks_W = (grid_W + block_size - wp.int32(1)) // block_size
        blocks_H = (grid_H + block_size - wp.int32(1)) // block_size
        blocks_D = (grid_D + block_size - wp.int32(1)) // block_size
        bx = bx_key + blocks_W // wp.int32(2)
        by = by_key + blocks_H // wp.int32(2)
        bz = bz_key + blocks_D // wp.int32(2)

        half_block = wp.float32(block_size) * 0.5
        gx = wp.float32(bx * block_size) + half_block
        gy = wp.float32(by * block_size) + half_block
        gz = wp.float32(bz * block_size) + half_block

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

    # =====================================================================
    # Block Recycling Kernel
    # =====================================================================

    # ``free_list_push`` is passed explicitly so Warp sees it as a local
    # closure binding when compiling the recycle kernel below.

    @warp_kernel(f"recycle_empty_blocks_kernel_bs{block_size}")
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

        Launch over the active block-pool high-water mark. ``max_blocks``
        is still passed as a storage bound and free-list capacity.

        Blocks are recycled only if BOTH:
            - Dynamic weight sum < ``block_empty_threshold`` (no
              depth observations)
            - Static voxel count == 0 (no primitive SDF)
        """
        tid = wp.tid()

        if tid >= max_blocks:
            return
        if tid >= num_allocated[0]:
            return

        hash_slot = block_to_hash_slot[tid]
        if hash_slot < 0:
            return

        if block_sums[tid] >= block_empty_threshold:
            return
        if static_block_sums[tid] > 0:
            return

        hash_table[hash_slot] = HASH_TOMBSTONE
        block_to_hash_slot[tid] = wp.int32(-1)
        free_list_push(free_list, free_count, tid, max_blocks)
        wp.atomic_add(recycle_count, 0, wp.int32(1))

    return {
        "mark_blocks_in_frustum_kernel": mark_blocks_in_frustum_kernel,
        "recycle_empty_blocks_kernel": recycle_empty_blocks_kernel,
        "block_empty_threshold": block_empty_threshold,
    }
