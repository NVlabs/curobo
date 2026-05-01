# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Raycast + voxel-extraction kernels, per-``block_size`` builder.

Moved from :mod:`curobo._src.perception.mapper.kernel.wp_raycast_common`,
:mod:`curobo._src.perception.mapper.kernel.wp_raycast`, and
:mod:`curobo._src.perception.mapper.kernel.wp_voxel_extraction` in the
block-size builder refactor.

BS-sensitivity:
    The only helper that closure-captures ``BS`` is
    :func:`_sample_voxel_at_block_local`, used by
    :func:`sample_tsdf_trilinear` to compute ``local_idx`` after a
    block-boundary crossing. All other kernels and helpers consume
    ``block_size`` as a runtime struct field or runtime argument and
    are BS-invariant in codegen.
"""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    quat_from_wxyz_array,
    vec3_from_array,
)
from curobo._src.util.warp import warp_func, warp_kernel

_SDF_INFINITY = wp.constant(wp.float32(1e10))


def make_raycast_kernels(
    block_size: int,
    *,
    hash_lookup,
    compute_avg_rgb_from_block,
    compute_avg_rgb_uint8_from_block,
    voxel_to_world,
    voxel_to_world_corner,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
    world_to_block_coords,
    world_to_block_and_local,
    world_to_continuous_voxel,
) -> dict[str, object]:
    """Build raycast and voxel-extraction kernels."""
    BS = wp.constant(block_size)

    # Cross-domain helpers are explicit parameters so Warp sees them as
    # local closure bindings when compiling dependent functions.

    # Raymarching tuning constants (used inside raycast kernels).
    MIN_STEP_SCALE = wp.constant(wp.float32(0.5))
    HIT_REFINE_ITERATIONS = wp.constant(10)

    # =====================================================================
    # Low-level voxel sampling
    # =====================================================================

    @warp_func(f"sample_voxel_bs{block_size}")
    def sample_voxel(
        tsdf: BlockSparseTSDFWarp,
        pool_idx: wp.int32,
        local_idx: wp.int32,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Sample combined SDF at a specific voxel (lowest-level)."""
        dynamic_sdf = _SDF_INFINITY
        static_sdf = _SDF_INFINITY
        has_valid = False

        if tsdf.has_dynamic:
            sdf_w = wp.float32(tsdf.block_data[pool_idx, local_idx, 0])
            w = wp.float32(tsdf.block_data[pool_idx, local_idx, 1])
            if w >= minimum_tsdf_weight:
                dynamic_sdf = sdf_w / w
                has_valid = True

        if tsdf.has_static:
            static_sdf = wp.float32(tsdf.static_block_data[pool_idx, local_idx])
            if static_sdf < 1e9:
                has_valid = True

        if not has_valid:
            return wp.vec2(_SDF_INFINITY, 0.0)

        return wp.vec2(wp.min(dynamic_sdf, static_sdf), 1.0)

    @warp_func(f"sample_tsdf_bs{block_size}")
    def sample_tsdf(
        tsdf: BlockSparseTSDFWarp,
        world_pos: wp.vec3,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Sample combined SDF from TSDF struct (nearest-neighbor)."""
        coords = world_to_block_and_local(world_pos)
        bx = coords[0]
        by = coords[1]
        bz = coords[2]
        local_idx = coords[3]

        pool_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
        if pool_idx < 0:
            return wp.vec2(_SDF_INFINITY, 0.0)

        return sample_voxel(tsdf, pool_idx, local_idx, minimum_tsdf_weight)

    @warp_func(f"_sample_voxel_at_block_local_bs{block_size}")
    def _sample_voxel_at_block_local(
        tsdf: BlockSparseTSDFWarp,
        bx: int,
        by: int,
        bz: int,
        lx: int,
        ly: int,
        lz: int,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Sample combined SDF at specific block/local coords.

        Includes block-boundary handling.

        BS-sensitive: the ``local_idx = lz * BS^2 + ly * BS + lx``
        packing depends on the block-size specialization.
        """
        actual_bx = bx
        actual_by = by
        actual_bz = bz
        actual_lx = lx
        actual_ly = ly
        actual_lz = lz

        if lx < 0:
            actual_bx = bx - 1
            actual_lx = lx + tsdf.block_size
        elif lx >= tsdf.block_size:
            actual_bx = bx + 1
            actual_lx = 0
        if ly < 0:
            actual_by = by - 1
            actual_ly = ly + tsdf.block_size
        elif ly >= tsdf.block_size:
            actual_by = by + 1
            actual_ly = 0
        if lz < 0:
            actual_bz = bz - 1
            actual_lz = lz + tsdf.block_size
        elif lz >= tsdf.block_size:
            actual_bz = bz + 1
            actual_lz = 0

        pool_idx = hash_lookup(
            tsdf.hash_table,
            actual_bx,
            actual_by,
            actual_bz,
            tsdf.hash_capacity,
        )
        if pool_idx < 0:
            return wp.vec2(_SDF_INFINITY, 0.0)

        local_idx = actual_lz * BS * BS + actual_ly * BS + actual_lx
        return sample_voxel(tsdf, pool_idx, local_idx, minimum_tsdf_weight)

    @warp_func(f"sample_tsdf_trilinear_bs{block_size}")
    def sample_tsdf_trilinear(
        tsdf: BlockSparseTSDFWarp,
        world_pos: wp.vec3,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Sample combined SDF with trilinear interpolation."""
        block_size_f = wp.float32(tsdf.block_size)

        voxel_f = world_to_continuous_voxel(world_pos)
        vx = voxel_f[0]
        vy = voxel_f[1]
        vz = voxel_f[2]

        bx_grid = wp.int32(wp.floor(vx / block_size_f))
        by_grid = wp.int32(wp.floor(vy / block_size_f))
        bz_grid = wp.int32(wp.floor(vz / block_size_f))

        lxf = vx - wp.float32(bx_grid) * block_size_f - 0.5
        lyf = vy - wp.float32(by_grid) * block_size_f - 0.5
        lzf = vz - wp.float32(bz_grid) * block_size_f - 0.5

        lx0 = wp.int32(wp.floor(lxf))
        ly0 = wp.int32(wp.floor(lyf))
        lz0 = wp.int32(wp.floor(lzf))

        tx = lxf - wp.float32(lx0)
        ty = lyf - wp.float32(ly0)
        tz = lzf - wp.float32(lz0)

        key = block_grid_to_key_coords(bx_grid, by_grid, bz_grid)

        c000 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0, ly0, lz0, minimum_tsdf_weight
        )
        c001 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0 + 1, ly0, lz0, minimum_tsdf_weight
        )
        c010 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0, ly0 + 1, lz0, minimum_tsdf_weight
        )
        c011 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0 + 1, ly0 + 1, lz0, minimum_tsdf_weight
        )
        c100 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0, ly0, lz0 + 1, minimum_tsdf_weight
        )
        c101 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0 + 1, ly0, lz0 + 1, minimum_tsdf_weight
        )
        c110 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0, ly0 + 1, lz0 + 1, minimum_tsdf_weight
        )
        c111 = _sample_voxel_at_block_local(
            tsdf, key[0], key[1], key[2], lx0 + 1, ly0 + 1, lz0 + 1, minimum_tsdf_weight
        )

        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w001 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w011 = tx * ty * (1.0 - tz)
        w100 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w110 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz

        n_valid = c000[1] + c001[1] + c010[1] + c011[1] + c100[1] + c101[1] + c110[1] + c111[1]
        if n_valid < 0.5:
            return wp.vec2(_SDF_INFINITY, 0.0)

        fallback = tsdf.truncation_distance
        s000 = wp.where(c000[1] > 0.5, c000[0], fallback)
        s001 = wp.where(c001[1] > 0.5, c001[0], fallback)
        s010 = wp.where(c010[1] > 0.5, c010[0], fallback)
        s011 = wp.where(c011[1] > 0.5, c011[0], fallback)
        s100 = wp.where(c100[1] > 0.5, c100[0], fallback)
        s101 = wp.where(c101[1] > 0.5, c101[0], fallback)
        s110 = wp.where(c110[1] > 0.5, c110[0], fallback)
        s111 = wp.where(c111[1] > 0.5, c111[0], fallback)

        sdf = (
            w000 * s000
            + w001 * s001
            + w010 * s010
            + w011 * s011
            + w100 * s100
            + w101 * s101
            + w110 * s110
            + w111 * s111
        )
        return wp.vec2(sdf, 1.0)

    @warp_func(f"sample_rgb_bs{block_size}")
    def sample_rgb(
        tsdf: BlockSparseTSDFWarp,
        world_pos: wp.vec3,
    ) -> wp.vec3:
        """Sample RGB color from TSDF struct (per-block average)."""
        coords = world_to_block_and_local(world_pos)
        bx = coords[0]
        by = coords[1]
        bz = coords[2]

        pool_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
        if pool_idx < 0:
            return wp.vec3(0.0, 0.0, 0.0)

        return compute_avg_rgb_from_block(tsdf.block_rgb, pool_idx)

    @warp_func(f"compute_gradient_bs{block_size}")
    def compute_gradient(
        tsdf: BlockSparseTSDFWarp,
        world_pos: wp.vec3,
        minimum_tsdf_weight: float,
    ) -> wp.vec3:
        """Compute surface normal from combined SDF gradient.

        Uses trilinear samples.
        """
        eps = tsdf.voxel_size
        sdf_xp = sample_tsdf_trilinear(
            tsdf, wp.vec3(world_pos[0] + eps, world_pos[1], world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_xm = sample_tsdf_trilinear(
            tsdf, wp.vec3(world_pos[0] - eps, world_pos[1], world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_yp = sample_tsdf_trilinear(
            tsdf, wp.vec3(world_pos[0], world_pos[1] + eps, world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_ym = sample_tsdf_trilinear(
            tsdf, wp.vec3(world_pos[0], world_pos[1] - eps, world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_zp = sample_tsdf_trilinear(
            tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] + eps), minimum_tsdf_weight
        )[0]
        sdf_zm = sample_tsdf_trilinear(
            tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] - eps), minimum_tsdf_weight
        )[0]

        if (
            sdf_xp > 1e9
            or sdf_xm > 1e9
            or sdf_yp > 1e9
            or sdf_ym > 1e9
            or sdf_zp > 1e9
            or sdf_zm > 1e9
        ):
            return wp.vec3(0.0, 0.0, 1.0)

        grad_x = (sdf_xp - sdf_xm) / (2.0 * eps)
        grad_y = (sdf_yp - sdf_ym) / (2.0 * eps)
        grad_z = (sdf_zp - sdf_zm) / (2.0 * eps)

        grad_mag = wp.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
        if grad_mag < 1e-6:
            return wp.vec3(0.0, 0.0, 1.0)

        return wp.vec3(grad_x / grad_mag, grad_y / grad_mag, grad_z / grad_mag)

    @warp_func(f"compute_gradient_nearest_bs{block_size}")
    def compute_gradient_nearest(
        tsdf: BlockSparseTSDFWarp,
        world_pos: wp.vec3,
        minimum_tsdf_weight: float,
    ) -> wp.vec3:
        """Compute surface normal using nearest-neighbor sampling.

        This is faster and coarser than trilinear sampling.
        """
        eps = tsdf.voxel_size
        sdf_xp = sample_tsdf(
            tsdf, wp.vec3(world_pos[0] + eps, world_pos[1], world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_xm = sample_tsdf(
            tsdf, wp.vec3(world_pos[0] - eps, world_pos[1], world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_yp = sample_tsdf(
            tsdf, wp.vec3(world_pos[0], world_pos[1] + eps, world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_ym = sample_tsdf(
            tsdf, wp.vec3(world_pos[0], world_pos[1] - eps, world_pos[2]), minimum_tsdf_weight
        )[0]
        sdf_zp = sample_tsdf(
            tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] + eps), minimum_tsdf_weight
        )[0]
        sdf_zm = sample_tsdf(
            tsdf, wp.vec3(world_pos[0], world_pos[1], world_pos[2] - eps), minimum_tsdf_weight
        )[0]

        if (
            sdf_xp > 1e9
            or sdf_xm > 1e9
            or sdf_yp > 1e9
            or sdf_ym > 1e9
            or sdf_zp > 1e9
            or sdf_zm > 1e9
        ):
            return wp.vec3(0.0, 0.0, 1.0)

        grad_x = (sdf_xp - sdf_xm) / (2.0 * eps)
        grad_y = (sdf_yp - sdf_ym) / (2.0 * eps)
        grad_z = (sdf_zp - sdf_zm) / (2.0 * eps)

        grad_mag = wp.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
        if grad_mag < 1e-6:
            return wp.vec3(0.0, 0.0, 1.0)

        return wp.vec3(grad_x / grad_mag, grad_y / grad_mag, grad_z / grad_mag)

    # =====================================================================
    # Raycast helpers
    # =====================================================================

    @warp_func(f"refine_hit_bisection_bs{block_size}")
    def refine_hit_bisection(
        tsdf: BlockSparseTSDFWarp,
        cam_pos: wp.vec3,
        ray_world: wp.vec3,
        t_lo: wp.float32,
        t_hi: wp.float32,
        sdf_lo: wp.float32,
        sdf_hi: wp.float32,
        minimum_tsdf_weight: wp.float32,
    ) -> wp.float32:
        """Refine hit position using bisection for smooth depth values."""
        denom = sdf_lo - sdf_hi
        if wp.abs(denom) < 1e-6:
            return (t_lo + t_hi) * 0.5

        t_mid = t_lo + sdf_lo * (t_hi - t_lo) / denom

        for _ in range(HIT_REFINE_ITERATIONS):
            pos_mid = cam_pos + ray_world * t_mid
            result = sample_tsdf_trilinear(tsdf, pos_mid, minimum_tsdf_weight)
            sdf_mid = result[0]

            if result[1] < 0.5:
                break
            if wp.abs(sdf_mid) < 1e-6:
                break

            if sdf_mid > 0.0:
                t_lo = t_mid
                sdf_lo = sdf_mid
            else:
                t_hi = t_mid
                sdf_hi = sdf_mid

            denom = sdf_lo - sdf_hi
            if wp.abs(denom) < 1e-6:
                t_mid = (t_lo + t_hi) * 0.5
            else:
                t_mid = t_lo + sdf_lo * (t_hi - t_lo) / denom

        return t_mid

    @warp_func(f"ray_block_exit_t_bs{block_size}")
    def ray_block_exit_t(
        ray_origin: wp.vec3,
        ray_dir: wp.vec3,
        bx: int,
        by: int,
        bz: int,
    ) -> float:
        """Compute t parameter where ray exits a block (slab method)."""
        voxel_min = block_key_to_voxel_base(bx, by, bz)
        voxel_max = wp.vec3i(voxel_min[0] + BS, voxel_min[1] + BS, voxel_min[2] + BS)
        block_min = voxel_to_world_corner(voxel_min)
        block_max = voxel_to_world_corner(voxel_max)

        block_min_x = block_min[0]
        block_min_y = block_min[1]
        block_min_z = block_min[2]

        block_max_x = block_max[0]
        block_max_y = block_max[1]
        block_max_z = block_max[2]

        inv_dir_x = 1.0 / ray_dir[0] if wp.abs(ray_dir[0]) > 1e-8 else 1e10 * wp.sign(ray_dir[0])
        inv_dir_y = 1.0 / ray_dir[1] if wp.abs(ray_dir[1]) > 1e-8 else 1e10 * wp.sign(ray_dir[1])
        inv_dir_z = 1.0 / ray_dir[2] if wp.abs(ray_dir[2]) > 1e-8 else 1e10 * wp.sign(ray_dir[2])

        t1_x = (block_min_x - ray_origin[0]) * inv_dir_x
        t2_x = (block_max_x - ray_origin[0]) * inv_dir_x
        t1_y = (block_min_y - ray_origin[1]) * inv_dir_y
        t2_y = (block_max_y - ray_origin[1]) * inv_dir_y
        t1_z = (block_min_z - ray_origin[2]) * inv_dir_z
        t2_z = (block_max_z - ray_origin[2]) * inv_dir_z

        t_max_x = wp.max(t1_x, t2_x)
        t_max_y = wp.max(t1_y, t2_y)
        t_max_z = wp.max(t1_z, t2_z)

        return wp.min(wp.min(t_max_x, t_max_y), t_max_z)

    # =====================================================================
    # Raycast kernels (4)
    # =====================================================================

    @warp_kernel(f"raycast_block_sparse_kernel_bs{block_size}")
    def raycast_block_sparse_kernel(
        intrinsics: wp.array2d(dtype=wp.float32),
        cam_position: wp.array(dtype=wp.float32),
        cam_quaternion: wp.array(dtype=wp.float32),
        tsdf: BlockSparseTSDFWarp,
        depth_minimum_distance: float,
        depth_maximum_distance: float,
        minimum_tsdf_weight: float,
        hit_points: wp.array2d(dtype=wp.float32),
        hit_normals: wp.array2d(dtype=wp.float32),
        hit_depths: wp.array(dtype=wp.float32),
        hit_mask: wp.array(dtype=wp.uint8),
        img_H: int,
        img_W: int,
    ):
        """Raycast block-sparse TSDF (depth + normals)."""
        tid = wp.tid()

        hit_points[tid, 0] = 0.0
        hit_points[tid, 1] = 0.0
        hit_points[tid, 2] = 0.0
        hit_normals[tid, 0] = 0.0
        hit_normals[tid, 1] = 0.0
        hit_normals[tid, 2] = 0.0
        hit_depths[tid] = 0.0
        hit_mask[tid] = wp.uint8(0)

        px = tid % img_W
        py = tid // img_W
        if py >= img_H:
            return

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        cam_pos = vec3_from_array(cam_position)
        cam_quat = quat_from_wxyz_array(cam_quaternion)

        ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
        ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
        ray_cam_z = 1.0

        ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
        ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)
        ray_world = wp.quat_rotate(cam_quat, ray_cam)

        step_size = tsdf.voxel_size * MIN_STEP_SCALE
        max_steps = wp.int32((depth_maximum_distance - depth_minimum_distance) / step_size) + 1
        max_steps = wp.min(max_steps, 10000)

        t = float(depth_minimum_distance)
        prev_sdf = float(1e10)
        prev_t = float(depth_minimum_distance)

        hit_found = bool(False)
        hit_t = float(0.0)

        for step in range(max_steps):
            if t > depth_maximum_distance:
                break

            pos = cam_pos + ray_world * t

            sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
            sdf = sdf_result[0]
            valid = sdf_result[1]

            if valid < 0.5:
                t += step_size
                continue

            if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
                hit_t = refine_hit_bisection(
                    tsdf,
                    cam_pos,
                    ray_world,
                    prev_t,
                    t,
                    prev_sdf,
                    sdf,
                    minimum_tsdf_weight,
                )
                hit_found = True
                break

            prev_sdf = sdf
            prev_t = t
            t += step_size

        if not hit_found:
            return

        hit_pos = cam_pos + ray_world * hit_t
        normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)
        z_depth = hit_t * ray_cam_z

        hit_points[tid, 0] = hit_pos[0]
        hit_points[tid, 1] = hit_pos[1]
        hit_points[tid, 2] = hit_pos[2]
        hit_normals[tid, 0] = normal[0]
        hit_normals[tid, 1] = normal[1]
        hit_normals[tid, 2] = normal[2]
        hit_depths[tid] = z_depth
        hit_mask[tid] = wp.uint8(1)

    @warp_kernel(f"raycast_block_sparse_color_kernel_bs{block_size}")
    def raycast_block_sparse_color_kernel(
        intrinsics: wp.array2d(dtype=wp.float32),
        cam_position: wp.array(dtype=wp.float32),
        cam_quaternion: wp.array(dtype=wp.float32),
        tsdf: BlockSparseTSDFWarp,
        depth_minimum_distance: float,
        depth_maximum_distance: float,
        minimum_tsdf_weight: float,
        hit_points: wp.array2d(dtype=wp.float32),
        hit_normals: wp.array2d(dtype=wp.float32),
        hit_colors: wp.array2d(dtype=wp.uint8),
        hit_depths: wp.array(dtype=wp.float32),
        hit_mask: wp.array(dtype=wp.uint8),
        img_H: int,
        img_W: int,
    ):
        """Raycast block-sparse TSDF with color output."""
        tid = wp.tid()

        hit_points[tid, 0] = 0.0
        hit_points[tid, 1] = 0.0
        hit_points[tid, 2] = 0.0
        hit_normals[tid, 0] = 0.0
        hit_normals[tid, 1] = 0.0
        hit_normals[tid, 2] = 0.0
        hit_colors[tid, 0] = wp.uint8(0)
        hit_colors[tid, 1] = wp.uint8(0)
        hit_colors[tid, 2] = wp.uint8(0)
        hit_depths[tid] = 0.0
        hit_mask[tid] = wp.uint8(0)

        px = tid % img_W
        py = tid // img_W
        if py >= img_H:
            return

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        cam_pos = vec3_from_array(cam_position)
        cam_quat = quat_from_wxyz_array(cam_quaternion)

        ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
        ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
        ray_cam_z = 1.0

        ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
        ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)
        ray_world = wp.quat_rotate(cam_quat, ray_cam)

        step_size = tsdf.voxel_size * MIN_STEP_SCALE
        max_steps = wp.int32((depth_maximum_distance - depth_minimum_distance) / step_size) + 1
        max_steps = wp.min(max_steps, 200000)

        t = float(depth_minimum_distance)
        prev_sdf = float(1e10)
        prev_t = float(depth_minimum_distance)

        hit_found = bool(False)
        hit_t = float(0.0)

        for step in range(max_steps):
            if t > depth_maximum_distance:
                break

            pos = cam_pos + ray_world * t
            sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
            sdf = sdf_result[0]
            valid = sdf_result[1]

            if valid < 0.5:
                t += step_size
                continue

            if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
                hit_t = refine_hit_bisection(
                    tsdf,
                    cam_pos,
                    ray_world,
                    prev_t,
                    t,
                    prev_sdf,
                    sdf,
                    minimum_tsdf_weight,
                )
                hit_found = True
                break

            prev_sdf = sdf
            prev_t = t
            t += step_size

        if not hit_found:
            return

        hit_pos = cam_pos + ray_world * hit_t
        normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)
        color = sample_rgb(tsdf, hit_pos)
        z_depth = hit_t * ray_cam_z

        hit_points[tid, 0] = hit_pos[0]
        hit_points[tid, 1] = hit_pos[1]
        hit_points[tid, 2] = hit_pos[2]
        hit_normals[tid, 0] = normal[0]
        hit_normals[tid, 1] = normal[1]
        hit_normals[tid, 2] = normal[2]
        hit_colors[tid, 0] = wp.uint8(color[0])
        hit_colors[tid, 1] = wp.uint8(color[1])
        hit_colors[tid, 2] = wp.uint8(color[2])
        hit_depths[tid] = z_depth
        hit_mask[tid] = wp.uint8(1)

    @warp_kernel(f"raycast_block_sparse_accelerated_kernel_bs{block_size}")
    def raycast_block_sparse_accelerated_kernel(
        intrinsics: wp.array2d(dtype=wp.float32),
        cam_position: wp.array(dtype=wp.float32),
        cam_quaternion: wp.array(dtype=wp.float32),
        tsdf: BlockSparseTSDFWarp,
        depth_minimum_distance: float,
        depth_maximum_distance: float,
        minimum_tsdf_weight: float,
        hit_points: wp.array2d(dtype=wp.float32),
        hit_normals: wp.array2d(dtype=wp.float32),
        hit_depths: wp.array(dtype=wp.float32),
        hit_mask: wp.array(dtype=wp.uint8),
        img_H: int,
        img_W: int,
    ):
        """Block-accelerated raycast (skip unallocated blocks)."""
        tid = wp.tid()

        hit_points[tid, 0] = 0.0
        hit_points[tid, 1] = 0.0
        hit_points[tid, 2] = 0.0
        hit_normals[tid, 0] = 0.0
        hit_normals[tid, 1] = 0.0
        hit_normals[tid, 2] = 0.0
        hit_depths[tid] = 0.0
        hit_mask[tid] = wp.uint8(0)

        px = tid % img_W
        py = tid // img_W
        if py >= img_H:
            return

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        cam_pos = vec3_from_array(cam_position)
        cam_quat = quat_from_wxyz_array(cam_quaternion)

        ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
        ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
        ray_cam_z = 1.0

        ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
        ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)
        ray_world = wp.quat_rotate(cam_quat, ray_cam)

        step_size = tsdf.voxel_size * MIN_STEP_SCALE

        t = float(depth_minimum_distance)
        prev_sdf = float(1e10)
        prev_t = float(depth_minimum_distance)

        hit_found = bool(False)
        hit_t = float(0.0)

        curr_bx = wp.int32(-999999)
        curr_by = wp.int32(-999999)
        curr_bz = wp.int32(-999999)
        curr_block_allocated = bool(False)
        curr_block_exit_t = float(0.0)

        max_iterations = 10000

        for iteration in range(max_iterations):
            if t > depth_maximum_distance:
                break

            pos = cam_pos + ray_world * t

            block_coords = world_to_block_coords(pos)
            bx = block_coords[0]
            by = block_coords[1]
            bz = block_coords[2]

            if bx != curr_bx or by != curr_by or bz != curr_bz:
                curr_bx = bx
                curr_by = by
                curr_bz = bz

                block_idx = hash_lookup(
                    tsdf.hash_table,
                    bx,
                    by,
                    bz,
                    tsdf.hash_capacity,
                )
                curr_block_allocated = block_idx >= 0

                curr_block_exit_t = ray_block_exit_t(
                    cam_pos,
                    ray_world,
                    bx,
                    by,
                    bz,
                )

            if not curr_block_allocated:
                t = curr_block_exit_t + step_size
                prev_sdf = 1e10
                continue

            sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
            sdf = sdf_result[0]
            valid = sdf_result[1]

            if valid < 0.5:
                t += step_size
                continue

            if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
                hit_t = refine_hit_bisection(
                    tsdf,
                    cam_pos,
                    ray_world,
                    prev_t,
                    t,
                    prev_sdf,
                    sdf,
                    minimum_tsdf_weight,
                )
                hit_found = True
                break

            prev_sdf = sdf
            prev_t = t
            t += step_size

        if not hit_found:
            return

        hit_pos = cam_pos + ray_world * hit_t
        normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)
        z_depth = hit_t * ray_cam_z

        hit_points[tid, 0] = hit_pos[0]
        hit_points[tid, 1] = hit_pos[1]
        hit_points[tid, 2] = hit_pos[2]
        hit_normals[tid, 0] = normal[0]
        hit_normals[tid, 1] = normal[1]
        hit_normals[tid, 2] = normal[2]
        hit_depths[tid] = z_depth
        hit_mask[tid] = wp.uint8(1)

    @warp_kernel(f"raycast_block_sparse_accelerated_color_kernel_bs{block_size}")
    def raycast_block_sparse_accelerated_color_kernel(
        intrinsics: wp.array2d(dtype=wp.float32),
        cam_position: wp.array(dtype=wp.float32),
        cam_quaternion: wp.array(dtype=wp.float32),
        tsdf: BlockSparseTSDFWarp,
        depth_minimum_distance: float,
        depth_maximum_distance: float,
        minimum_tsdf_weight: float,
        hit_points: wp.array2d(dtype=wp.float32),
        hit_normals: wp.array2d(dtype=wp.float32),
        hit_colors: wp.array2d(dtype=wp.uint8),
        hit_depths: wp.array(dtype=wp.float32),
        hit_mask: wp.array(dtype=wp.uint8),
        img_H: int,
        img_W: int,
    ):
        """Block-accelerated raycast with color output."""
        tid = wp.tid()

        hit_points[tid, 0] = 0.0
        hit_points[tid, 1] = 0.0
        hit_points[tid, 2] = 0.0
        hit_normals[tid, 0] = 0.0
        hit_normals[tid, 1] = 0.0
        hit_normals[tid, 2] = 0.0
        hit_colors[tid, 0] = wp.uint8(0)
        hit_colors[tid, 1] = wp.uint8(0)
        hit_colors[tid, 2] = wp.uint8(0)
        hit_depths[tid] = 0.0
        hit_mask[tid] = wp.uint8(0)

        px = tid % img_W
        py = tid // img_W
        if py >= img_H:
            return

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        cam_pos = vec3_from_array(cam_position)
        cam_quat = quat_from_wxyz_array(cam_quaternion)

        ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
        ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
        ray_cam_z = 1.0

        ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
        ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)
        ray_world = wp.quat_rotate(cam_quat, ray_cam)

        step_size = tsdf.voxel_size * MIN_STEP_SCALE

        t = float(depth_minimum_distance)
        prev_sdf = float(1e10)
        prev_t = float(depth_minimum_distance)

        hit_found = bool(False)
        hit_t = float(0.0)

        curr_bx = wp.int32(-999999)
        curr_by = wp.int32(-999999)
        curr_bz = wp.int32(-999999)
        curr_block_allocated = bool(False)
        curr_block_exit_t = float(0.0)

        max_iterations = 10000

        for iteration in range(max_iterations):
            if t > depth_maximum_distance:
                break

            pos = cam_pos + ray_world * t

            block_coords = world_to_block_coords(pos)
            bx = block_coords[0]
            by = block_coords[1]
            bz = block_coords[2]

            if bx != curr_bx or by != curr_by or bz != curr_bz:
                curr_bx = bx
                curr_by = by
                curr_bz = bz

                block_idx = hash_lookup(
                    tsdf.hash_table,
                    bx,
                    by,
                    bz,
                    tsdf.hash_capacity,
                )
                curr_block_allocated = block_idx >= 0

                curr_block_exit_t = ray_block_exit_t(
                    cam_pos,
                    ray_world,
                    bx,
                    by,
                    bz,
                )

            if not curr_block_allocated:
                t = curr_block_exit_t + step_size
                prev_sdf = 1e10
                continue

            sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
            sdf = sdf_result[0]
            valid = sdf_result[1]

            if valid < 0.5:
                t += step_size
                continue

            if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
                hit_t = refine_hit_bisection(
                    tsdf,
                    cam_pos,
                    ray_world,
                    prev_t,
                    t,
                    prev_sdf,
                    sdf,
                    minimum_tsdf_weight,
                )
                hit_found = True
                break

            prev_sdf = sdf
            prev_t = t
            t += step_size

        if not hit_found:
            return

        hit_pos = cam_pos + ray_world * hit_t
        normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)
        color = sample_rgb(tsdf, hit_pos)
        z_depth = hit_t * ray_cam_z

        hit_points[tid, 0] = hit_pos[0]
        hit_points[tid, 1] = hit_pos[1]
        hit_points[tid, 2] = hit_pos[2]
        hit_normals[tid, 0] = normal[0]
        hit_normals[tid, 1] = normal[1]
        hit_normals[tid, 2] = normal[2]
        hit_colors[tid, 0] = wp.uint8(color[0])
        hit_colors[tid, 1] = wp.uint8(color[1])
        hit_colors[tid, 2] = wp.uint8(color[2])
        hit_depths[tid] = z_depth
        hit_mask[tid] = wp.uint8(1)

    # =====================================================================
    # Voxel-extraction kernels (4, BS-invariant)
    # =====================================================================

    @warp_kernel(f"count_surface_voxels_kernel_bs{block_size}")
    def count_surface_voxels_kernel(
        tsdf: BlockSparseTSDFWarp,
        sdf_threshold: float,
        minimum_tsdf_weight: float,
        voxel_count: wp.array(dtype=wp.int32),
    ):
        block_idx, local_idx = wp.tid()
        n_alloc = tsdf.num_allocated[0]

        if block_idx >= n_alloc:
            return

        result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return
        sdf = result[0]
        if wp.abs(sdf) > sdf_threshold:
            return
        wp.atomic_add(voxel_count, 0, 1)

    @warp_kernel(f"count_occupied_voxels_kernel_bs{block_size}")
    def count_occupied_voxels_kernel(
        tsdf: BlockSparseTSDFWarp,
        minimum_tsdf_weight: float,
        surface_only: wp.int32,
        sdf_threshold: float,
        voxel_count: wp.array(dtype=wp.int32),
    ):
        block_idx, local_idx = wp.tid()
        n_alloc = tsdf.num_allocated[0]

        if block_idx >= n_alloc:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return
        sdf = result[0]

        if surface_only == wp.int32(1):
            if wp.abs(sdf) >= sdf_threshold:
                return
        else:
            if sdf > 0.0:
                return
        wp.atomic_add(voxel_count, 0, 1)

    @warp_kernel(f"extract_occupied_voxels_kernel_bs{block_size}")
    def extract_occupied_voxels_kernel(
        tsdf: BlockSparseTSDFWarp,
        minimum_tsdf_weight: float,
        surface_only: wp.int32,
        sdf_threshold: float,
        out_centers: wp.array2d(dtype=wp.float32),
        out_pool_idx: wp.array(dtype=wp.int32),
        out_count: wp.array(dtype=wp.int32),
        max_voxels: wp.int32,
    ):
        block_idx, local_idx = wp.tid()
        n_alloc = tsdf.num_allocated[0]

        if block_idx >= n_alloc:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return
        sdf = result[0]

        if surface_only == wp.int32(1):
            if wp.abs(sdf) >= sdf_threshold:
                return
        else:
            if sdf > 0.0:
                return

        slot = wp.atomic_add(out_count, 0, 1)
        if slot >= max_voxels:
            return

        bx = tsdf.block_coords[block_idx * 3]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        lx = local_idx % tsdf.block_size
        ly = (local_idx // tsdf.block_size) % tsdf.block_size
        lz = local_idx // (tsdf.block_size * tsdf.block_size)

        base = block_key_to_voxel_base(bx, by, bz)
        vx = base[0] + lx
        vy = base[1] + ly
        vz = base[2] + lz

        world_pos = voxel_to_world(wp.vec3i(vx, vy, vz))

        out_centers[slot, 0] = world_pos[0]
        out_centers[slot, 1] = world_pos[1]
        out_centers[slot, 2] = world_pos[2]

        out_pool_idx[slot] = block_idx

    @warp_kernel(f"extract_occupied_voxels_masked_kernel_bs{block_size}")
    def extract_occupied_voxels_masked_kernel(
        tsdf: BlockSparseTSDFWarp,
        minimum_tsdf_weight: float,
        surface_only: wp.int32,
        sdf_threshold: float,
        block_mask: wp.array(dtype=wp.uint8),
        out_centers: wp.array2d(dtype=wp.float32),
        out_pool_idx: wp.array(dtype=wp.int32),
        out_count: wp.array(dtype=wp.int32),
        max_voxels: wp.int32,
    ):
        """Masked variant of ``extract_occupied_voxels_kernel``.

        ``block_mask`` is a ``(max_blocks,)`` boolean (uint8) array;
        only blocks whose entry is non-zero contribute voxels. Used
        by ``extract_matching_feature_voxels`` to restrict extraction to
        the top-k most feature-similar blocks without a second pass.
        """
        block_idx, local_idx = wp.tid()
        n_alloc = tsdf.num_allocated[0]

        if block_idx >= n_alloc:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return
        if block_mask[block_idx] == wp.uint8(0):
            return

        result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return
        sdf = result[0]

        if surface_only == wp.int32(1):
            if wp.abs(sdf) >= sdf_threshold:
                return
        else:
            if sdf > 0.0:
                return

        slot = wp.atomic_add(out_count, 0, 1)
        if slot >= max_voxels:
            return

        bx = tsdf.block_coords[block_idx * 3]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        lx = local_idx % tsdf.block_size
        ly = (local_idx // tsdf.block_size) % tsdf.block_size
        lz = local_idx // (tsdf.block_size * tsdf.block_size)

        base = block_key_to_voxel_base(bx, by, bz)
        vx = base[0] + lx
        vy = base[1] + ly
        vz = base[2] + lz

        world_pos = voxel_to_world(wp.vec3i(vx, vy, vz))

        out_centers[slot, 0] = world_pos[0]
        out_centers[slot, 1] = world_pos[1]
        out_centers[slot, 2] = world_pos[2]

        out_pool_idx[slot] = block_idx

    @warp_kernel(f"count_occupied_voxels_masked_kernel_bs{block_size}")
    def count_occupied_voxels_masked_kernel(
        tsdf: BlockSparseTSDFWarp,
        minimum_tsdf_weight: float,
        surface_only: wp.int32,
        sdf_threshold: float,
        block_mask: wp.array(dtype=wp.uint8),
        voxel_count: wp.array(dtype=wp.int32),
    ):
        """Count-only variant of ``extract_occupied_voxels_masked_kernel``."""
        block_idx, local_idx = wp.tid()
        n_alloc = tsdf.num_allocated[0]

        if block_idx >= n_alloc:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return
        if block_mask[block_idx] == wp.uint8(0):
            return

        result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return
        sdf = result[0]

        if surface_only == wp.int32(1):
            if wp.abs(sdf) >= sdf_threshold:
                return
        else:
            if sdf > 0.0:
                return
        wp.atomic_add(voxel_count, 0, 1)

    @warp_kernel(f"extract_surface_voxels_kernel_bs{block_size}")
    def extract_surface_voxels_kernel(
        tsdf: BlockSparseTSDFWarp,
        sdf_threshold: float,
        minimum_tsdf_weight: float,
        out_centers: wp.array2d(dtype=wp.float32),
        out_colors: wp.array2d(dtype=wp.uint8),
        out_sdf: wp.array(dtype=wp.float32),
        out_count: wp.array(dtype=wp.int32),
        max_voxels: wp.int32,
    ):
        block_idx, local_idx = wp.tid()
        n_alloc = tsdf.num_allocated[0]

        if block_idx >= n_alloc:
            return

        result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return
        sdf = result[0]
        if wp.abs(sdf) > sdf_threshold:
            return

        slot = wp.atomic_add(out_count, 0, 1)
        if slot >= max_voxels:
            return

        bx = tsdf.block_coords[block_idx * 3]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        lx = local_idx % tsdf.block_size
        ly = (local_idx // tsdf.block_size) % tsdf.block_size
        lz = local_idx // (tsdf.block_size * tsdf.block_size)

        base = block_key_to_voxel_base(bx, by, bz)
        vx = base[0] + lx
        vy = base[1] + ly
        vz = base[2] + lz

        world_pos = voxel_to_world(wp.vec3i(vx, vy, vz))

        out_centers[slot, 0] = world_pos[0]
        out_centers[slot, 1] = world_pos[1]
        out_centers[slot, 2] = world_pos[2]

        out_sdf[slot] = sdf

        rgb = compute_avg_rgb_uint8_from_block(tsdf.block_rgb, block_idx)
        out_colors[slot, 0] = wp.uint8(rgb[0])
        out_colors[slot, 1] = wp.uint8(rgb[1])
        out_colors[slot, 2] = wp.uint8(rgb[2])

    # Expose everything on the instance.
    return {
        "sample_voxel": sample_voxel,
        "sample_tsdf": sample_tsdf,
        "_sample_voxel_at_block_local": _sample_voxel_at_block_local,
        "sample_tsdf_trilinear": sample_tsdf_trilinear,
        "sample_rgb": sample_rgb,
        "compute_gradient": compute_gradient,
        "compute_gradient_nearest": compute_gradient_nearest,
        "refine_hit_bisection": refine_hit_bisection,
        "ray_block_exit_t": ray_block_exit_t,
        "raycast_block_sparse_kernel": raycast_block_sparse_kernel,
        "raycast_block_sparse_color_kernel": raycast_block_sparse_color_kernel,
        "raycast_block_sparse_accelerated_kernel": raycast_block_sparse_accelerated_kernel,
        "raycast_block_sparse_accelerated_color_kernel": (
            raycast_block_sparse_accelerated_color_kernel
        ),
        "count_surface_voxels_kernel": count_surface_voxels_kernel,
        "count_occupied_voxels_kernel": count_occupied_voxels_kernel,
        "count_occupied_voxels_masked_kernel": count_occupied_voxels_masked_kernel,
        "extract_occupied_voxels_kernel": extract_occupied_voxels_kernel,
        "extract_occupied_voxels_masked_kernel": extract_occupied_voxels_masked_kernel,
        "extract_surface_voxels_kernel": extract_surface_voxels_kernel,
    }
