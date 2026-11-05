# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse marching-cubes mesh extraction, per-``block_size`` builder.

Moved from :mod:`curobo._src.perception.mapper.mesh_extractor` in the
block-size builder refactor.

The kernels and ``@wp.func`` helpers are BS-sensitive: cube indexing uses
``BS``-based packing (``local_idx = lz*BS^2 + ly*BS + lx``), and
block-boundary crossing logic tests against ``BS`` directly. Every kernel
closure-captures ``BS = wp.constant(block_size)``.
"""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import (
    BlockSparseTSDFWarp,
    PreparedSemanticFilterWarp,
)
from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
    get_edge_vertex,
)
from curobo._src.util.warp import warp_func, warp_kernel


def make_mesh_kernels(
    block_size: int,
    *,
    num_cameras: int = 1,
    image_height: int = 1,
    image_width: int = 1,
    hash_lookup,
    sample_rgb,
    sample_voxel,
    sample_tsdf_trilinear,
    compute_gradient,
    compute_gradient_nearest,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
    semantic_local_keep,
) -> dict[str, object]:
    """Build block-sparse marching-cubes kernels."""
    BS = wp.constant(block_size)
    IMAGE_HEIGHT = wp.constant(image_height)
    IMAGE_WIDTH = wp.constant(image_width)

    # Cross-domain helpers are explicit parameters so Warp sees them as
    # local closure bindings when compiling dependent functions.

    # =====================================================================
    # Vertex refinement
    # =====================================================================

    @warp_func(f"refine_vertex_mesh_bs{block_size}")
    def refine_vertex_mesh(
        tsdf: BlockSparseTSDFWarp,
        vertex: wp.vec3,
        level: wp.float32,
        iterations: wp.int32,
        minimum_tsdf_weight: wp.float32,
    ) -> wp.vec3:
        """Refine vertex to true SDF zero-crossing via Newton-Raphson."""
        pos = vertex
        for _ in range(iterations):
            result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
            if result[1] < 0.5:
                break
            sdf_val = result[0] - level
            if wp.abs(sdf_val) < 1e-6 or sdf_val > 100.0:
                break

            grad = compute_gradient(tsdf, pos, minimum_tsdf_weight)
            grad_mag = wp.sqrt(wp.dot(grad, grad))
            if grad_mag < 1e-4:
                break

            step_size = wp.clamp(
                sdf_val / grad_mag,
                -tsdf.voxel_size * 0.5,
                tsdf.voxel_size * 0.5,
            )
            pos = pos - step_size * (grad / grad_mag)
        return pos

    # =====================================================================
    # SDF access (BS-sensitive: local_idx packing)
    # =====================================================================

    @warp_func(f"get_block_sdf_bs{block_size}")
    def get_block_sdf(
        tsdf: BlockSparseTSDFWarp,
        pool_idx: wp.int32,
        lx: wp.int32,
        ly: wp.int32,
        lz: wp.int32,
        level: float,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Get combined SDF value at (pool_idx, lx, ly, lz)."""
        local_idx = lz * BS * BS + ly * BS + lx
        result = sample_voxel(tsdf, pool_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return wp.vec2(1e10, 0.0)
        return wp.vec2(result[0] - level, 1.0)

    @warp_func(f"sample_cube_corner_bs{block_size}")
    def sample_cube_corner(
        cx: wp.int32,
        cy: wp.int32,
        cz: wp.int32,
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        pool_idx: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Sample combined SDF at cube corner, handling block boundary crossing."""
        if cx < BS and cy < BS and cz < BS:
            return get_block_sdf(tsdf, pool_idx, cx, cy, cz, level, minimum_tsdf_weight)

        nbx = bx
        nby = by
        nbz = bz
        nlx = cx
        nly = cy
        nlz = cz

        if cx >= BS:
            nbx = bx + 1
            nlx = 0
        if cy >= BS:
            nby = by + 1
            nly = 0
        if cz >= BS:
            nbz = bz + 1
            nlz = 0

        neighbor_idx = hash_lookup(tsdf.hash_table, nbx, nby, nbz, tsdf.hash_capacity)
        if neighbor_idx < 0:
            return wp.vec2(1e10, 0.0)

        return get_block_sdf(tsdf, neighbor_idx, nlx, nly, nlz, level, minimum_tsdf_weight)

    # =====================================================================
    # Surface-cube predicate
    # =====================================================================

    @warp_func(f"is_surface_cube_combined_bs{block_size}")
    def is_surface_cube_combined(
        cx: wp.int32,
        cy: wp.int32,
        cz: wp.int32,
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        block_idx: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
    ) -> wp.bool:
        """Check if a cube contains a surface (sign change across corners)."""
        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        if s0[1] < 0.5 or s1[1] < 0.5 or s2[1] < 0.5 or s3[1] < 0.5:
            return False
        if s4[1] < 0.5 or s5[1] < 0.5 or s6[1] < 0.5 or s7[1] < 0.5:
            return False

        has_positive = (
            s0[0] > 0.0
            or s1[0] > 0.0
            or s2[0] > 0.0
            or s3[0] > 0.0
            or s4[0] > 0.0
            or s5[0] > 0.0
            or s6[0] > 0.0
            or s7[0] > 0.0
        )
        has_negative = (
            s0[0] < 0.0
            or s1[0] < 0.0
            or s2[0] < 0.0
            or s3[0] < 0.0
            or s4[0] < 0.0
            or s5[0] < 0.0
            or s6[0] < 0.0
            or s7[0] < 0.0
        )
        if not (has_positive and has_negative):
            return False

        if surface_band > 0.0:
            in_band = (
                wp.abs(s0[0]) < surface_band
                or wp.abs(s1[0]) < surface_band
                or wp.abs(s2[0]) < surface_band
                or wp.abs(s3[0]) < surface_band
                or wp.abs(s4[0]) < surface_band
                or wp.abs(s5[0]) < surface_band
                or wp.abs(s6[0]) < surface_band
                or wp.abs(s7[0]) < surface_band
            )
            if not in_band:
                return False

        return True

    # =====================================================================
    # Surface detection kernels
    # =====================================================================

    @warp_kernel(f"append_active_blocks_kernel_bs{block_size}", enable_backward=False)
    def append_active_blocks_kernel(
        tsdf: BlockSparseTSDFWarp,
        active_count: wp.array(dtype=wp.int32),
        active_block_idx: wp.array(dtype=wp.int32),
    ):
        """Compact active pool indices into a dense block list."""
        block_idx = wp.tid()

        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        out_idx = wp.atomic_add(active_count, 0, wp.int32(1))
        active_block_idx[out_idx] = block_idx

    @warp_kernel(
        f"append_active_blocks_semantic_kernel_bs{block_size}",
        enable_backward=False,
    )
    def append_active_blocks_semantic_kernel(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        active_count: wp.array(dtype=wp.int32),
        active_block_idx: wp.array(dtype=wp.int32),
    ):
        """Compact active blocks that contain retained semantic nodes."""
        block_idx = wp.tid()
        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < wp.int32(0):
            return
        if semantic_filter.block_has_keep[block_idx] == wp.uint8(0):
            return
        out_idx = wp.atomic_add(active_count, 0, wp.int32(1))
        active_block_idx[out_idx] = block_idx

    @warp_kernel(
        f"count_surface_cubes_from_blocks_kernel_bs{block_size}", enable_backward=False
    )
    def count_surface_cubes_from_blocks_kernel(
        tsdf: BlockSparseTSDFWarp,
        active_block_idx: wp.array(dtype=wp.int32),
        n_active_blocks: wp.int32,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
    ):
        """Count surface cubes from a compact active block list."""
        active_idx, cube_idx = wp.tid()

        if active_idx >= n_active_blocks:
            return

        block_idx = active_block_idx[active_idx]
        if block_idx < 0 or block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            wp.atomic_add(surface_count, 0, wp.int32(1))

    @warp_kernel(
        f"count_surface_cubes_from_blocks_semantic_kernel_bs{block_size}",
        enable_backward=False,
    )
    def count_surface_cubes_from_blocks_semantic_kernel(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        active_block_idx: wp.array(dtype=wp.int32),
        n_active_blocks: wp.int32,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
    ):
        """Count only surface cubes whose centers are semantically retained."""
        active_idx, cube_idx = wp.tid()
        if active_idx >= n_active_blocks:
            return
        block_idx = active_block_idx[active_idx]
        if block_idx < wp.int32(0) or block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < wp.int32(0):
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)
        if not semantic_local_keep(
            semantic_filter,
            block_idx,
            wp.float32(cx) + wp.float32(0.5),
            wp.float32(cy) + wp.float32(0.5),
            wp.float32(cz) + wp.float32(0.5),
        ):
            return

        bx = tsdf.block_coords[block_idx * wp.int32(3)]
        by = tsdf.block_coords[block_idx * wp.int32(3) + wp.int32(1)]
        bz = tsdf.block_coords[block_idx * wp.int32(3) + wp.int32(2)]
        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            wp.atomic_add(surface_count, 0, wp.int32(1))

    @warp_kernel(
        f"append_surface_cubes_from_blocks_kernel_bs{block_size}", enable_backward=False
    )
    def append_surface_cubes_from_blocks_kernel(
        tsdf: BlockSparseTSDFWarp,
        active_block_idx: wp.array(dtype=wp.int32),
        n_active_blocks: wp.int32,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
    ):
        """Append surface cubes from a compact active block list."""
        active_idx, cube_idx = wp.tid()

        if active_idx >= n_active_blocks:
            return

        block_idx = active_block_idx[active_idx]
        if block_idx < 0 or block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            out_idx = wp.atomic_add(surface_count, 0, wp.int32(1))
            surface_block_idx[out_idx] = block_idx
            surface_cube_idx[out_idx] = cube_idx

    @warp_kernel(
        f"append_surface_cubes_from_blocks_semantic_kernel_bs{block_size}",
        enable_backward=False,
    )
    def append_surface_cubes_from_blocks_semantic_kernel(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        active_block_idx: wp.array(dtype=wp.int32),
        n_active_blocks: wp.int32,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
    ):
        """Append only surface cubes whose centers are semantically retained."""
        active_idx, cube_idx = wp.tid()
        if active_idx >= n_active_blocks:
            return
        block_idx = active_block_idx[active_idx]
        if block_idx < wp.int32(0) or block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < wp.int32(0):
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)
        if not semantic_local_keep(
            semantic_filter,
            block_idx,
            wp.float32(cx) + wp.float32(0.5),
            wp.float32(cy) + wp.float32(0.5),
            wp.float32(cz) + wp.float32(0.5),
        ):
            return

        bx = tsdf.block_coords[block_idx * wp.int32(3)]
        by = tsdf.block_coords[block_idx * wp.int32(3) + wp.int32(1)]
        bz = tsdf.block_coords[block_idx * wp.int32(3) + wp.int32(2)]
        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            out_idx = wp.atomic_add(surface_count, 0, wp.int32(1))
            surface_block_idx[out_idx] = block_idx
            surface_cube_idx[out_idx] = cube_idx

    @warp_kernel(f"count_total_triangles_kernel_bs{block_size}", enable_backward=False)
    def count_total_triangles_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        num_tris_table: wp.array(dtype=wp.int32),
        triangle_count: wp.array(dtype=wp.int32),
    ):
        """Count triangle-soup mesh triangles with one device-side total."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        wp.atomic_add(triangle_count, 0, num_tris_table[cube_config])

    @warp_kernel(f"generate_mesh_kernel_bs{block_size}")
    def generate_mesh_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        refine_iterations: wp.int32,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        tri_table: wp.array(dtype=wp.int32),
        vertices: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3ub),
        triangle_count: wp.array(dtype=wp.int32),
        triangle_capacity: wp.int32,
    ):
        """Generate a triangle-soup mesh with atomic output allocation."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0] + cx
        gy = base[1] + cy
        gz = base[2] + cz

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        p0 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p1 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p2 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p3 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p4 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p5 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p6 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p7 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )

        rgb = sample_rgb(tsdf, (p0 + p6) * wp.float32(0.5))
        color = wp.vec3ub(wp.uint8(rgb[0]), wp.uint8(rgb[1]), wp.uint8(rgb[2]))

        table_offset = cube_config * 16

        for t in range(5):
            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break

            tri_idx = wp.atomic_add(triangle_count, 0, wp.int32(1))
            if tri_idx >= triangle_capacity:
                continue

            e1 = tri_table[table_offset + t * 3 + 1]
            e2 = tri_table[table_offset + t * 3 + 2]

            out_base = tri_idx * 3
            for v_idx in range(3):
                edge = e0
                if v_idx == 1:
                    edge = e2
                elif v_idx == 2:
                    edge = e1

                vertex_id = out_base + v_idx
                v = get_edge_vertex(
                    edge,
                    p0,
                    p1,
                    p2,
                    p3,
                    p4,
                    p5,
                    p6,
                    p7,
                    s0[0],
                    s1[0],
                    s2[0],
                    s3[0],
                    s4[0],
                    s5[0],
                    s6[0],
                    s7[0],
                )
                if refine_iterations > 0:
                    v = refine_vertex_mesh(
                        tsdf, v, level, refine_iterations, minimum_tsdf_weight
                    )
                n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
                vertices[vertex_id] = v
                normals[vertex_id] = wp.normalize(n)
                colors[vertex_id] = color

    # =====================================================================
    # Projective texture mapping for triangle-list meshes
    # =====================================================================

    @warp_func(f"texture_depth_z_visible_bs{block_size}")
    def texture_depth_z_visible(
        visibility_depth: wp.float32,
        projected_z: wp.float32,
        texture_depth_tolerance_m: wp.float32,
        voxel_size: wp.float32,
    ) -> bool:
        """Check camera-frame z against a sampled visibility depth."""
        if projected_z <= 0.0 or not wp.isfinite(projected_z):
            return False
        if visibility_depth <= 0.0 or not wp.isfinite(visibility_depth):
            return False

        tolerance = texture_depth_tolerance_m
        if tolerance < 0.0:
            voxel_tolerance = wp.float32(2.0) * voxel_size
            range_tolerance = wp.float32(0.01) * projected_z
            tolerance = voxel_tolerance
            if range_tolerance > tolerance:
                tolerance = range_tolerance

        return wp.abs(projected_z - visibility_depth) <= tolerance

    @warp_kernel(f"project_mesh_uvs_kernel_bs{block_size}", enable_backward=False)
    def project_mesh_uvs_kernel(
        vertices: wp.array(dtype=wp.vec3),
        vertex_count: wp.int32,
        visibility_depths: wp.array3d(dtype=wp.float32),
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        camera_min_distance: float,
        camera_max_distance: float,
        texture_depth_tolerance_m: float,
        camera_offset: wp.int32,
        atlas_camera_count: wp.int32,
        atlas_columns: wp.int32,
        vertex_uvs: wp.array(dtype=wp.vec2),
        projection_scores: wp.array(dtype=wp.float32),
        colors: wp.array(dtype=wp.vec3ub),
        fill_missing_colors: wp.int32,
        tsdf: BlockSparseTSDFWarp,
    ):
        """Project triangle-soup mesh vertices into a horizontal camera atlas."""
        tri_idx = wp.tid()
        base = tri_idx * wp.int32(3)
        if base + wp.int32(2) >= vertex_count:
            return

        invalid_uv = wp.vec2(-1.0, -1.0)
        v0 = vertices[base]
        v1 = vertices[base + wp.int32(1)]
        v2 = vertices[base + wp.int32(2)]

        center = (v0 + v1 + v2) / wp.float32(3.0)
        face_normal = wp.cross(v1 - v0, v2 - v0)

        uv0 = invalid_uv
        uv1 = invalid_uv
        uv2 = invalid_uv
        projected = bool(False)
        score_limit = wp.float32(1.0e20)

        for _attempt in range(num_cameras):
            best_cam = wp.int32(-1)
            best_score = wp.float32(-1.0e20)

            for cam_i in range(num_cameras):
                cam_pos_i = wp.vec3(
                    cam_positions[cam_i, 0],
                    cam_positions[cam_i, 1],
                    cam_positions[cam_i, 2],
                )
                cam_quat_i = wp.quaternion(
                    cam_quaternions[cam_i, 1],
                    cam_quaternions[cam_i, 2],
                    cam_quaternions[cam_i, 3],
                    cam_quaternions[cam_i, 0],
                )
                center_cam_i = wp.quat_rotate(wp.quat_inverse(cam_quat_i), center - cam_pos_i)
                z_center_i = center_cam_i[2]

                if z_center_i > camera_min_distance and z_center_i <= camera_max_distance:
                    fx_i = intrinsics[cam_i, 0, 0]
                    fy_i = intrinsics[cam_i, 1, 1]
                    cx_i = intrinsics[cam_i, 0, 2]
                    cy_i = intrinsics[cam_i, 1, 2]

                    u_center_i = fx_i * center_cam_i[0] / z_center_i + cx_i
                    v_center_i = fy_i * center_cam_i[1] / z_center_i + cy_i
                    if (
                        u_center_i >= 0.0
                        and u_center_i < wp.float32(IMAGE_WIDTH)
                        and v_center_i >= 0.0
                        and v_center_i < wp.float32(IMAGE_HEIGHT)
                    ):
                        px_center_i = wp.int32(wp.floor(u_center_i))
                        py_center_i = wp.int32(wp.floor(v_center_i))
                        center_depth_i = visibility_depths[cam_i, py_center_i, px_center_i]
                        if texture_depth_z_visible(
                            center_depth_i,
                            z_center_i,
                            texture_depth_tolerance_m,
                            tsdf.voxel_size,
                        ):
                            viewing_dir = cam_pos_i - center
                            viewing_len_sq = wp.dot(viewing_dir, viewing_dir)
                            if viewing_len_sq > 1.0e-20:
                                viewing_dir = viewing_dir / wp.sqrt(viewing_len_sq)
                                score = wp.dot(face_normal, viewing_dir)
                                if (
                                    score > 0.0
                                    and score < score_limit
                                    and score > best_score
                                ):
                                    best_score = score
                                    best_cam = wp.int32(cam_i)

            if best_cam < 0:
                break

            cam_pos = wp.vec3(
                cam_positions[best_cam, 0],
                cam_positions[best_cam, 1],
                cam_positions[best_cam, 2],
            )
            cam_quat = wp.quaternion(
                cam_quaternions[best_cam, 1],
                cam_quaternions[best_cam, 2],
                cam_quaternions[best_cam, 3],
                cam_quaternions[best_cam, 0],
            )
            cam_quat_inv = wp.quat_inverse(cam_quat)
            fx = intrinsics[best_cam, 0, 0]
            fy = intrinsics[best_cam, 1, 1]
            cx = intrinsics[best_cam, 0, 2]
            cy = intrinsics[best_cam, 1, 2]
            atlas_rows = (
                atlas_camera_count + atlas_columns - wp.int32(1)
            ) // atlas_columns
            atlas_camera_idx = camera_offset + best_cam
            atlas_col = atlas_camera_idx % atlas_columns
            atlas_row = atlas_camera_idx // atlas_columns
            atlas_width = wp.float32(atlas_columns * IMAGE_WIDTH)
            atlas_height = wp.float32(atlas_rows * IMAGE_HEIGHT)
            atlas_x_offset = wp.float32(atlas_col * IMAGE_WIDTH)
            atlas_y_offset = wp.float32(atlas_row * IMAGE_HEIGHT)

            all_valid = bool(True)

            p0_cam = wp.quat_rotate(cam_quat_inv, v0 - cam_pos)
            z0 = p0_cam[2]
            if z0 > camera_min_distance and z0 <= camera_max_distance:
                u0 = fx * p0_cam[0] / z0 + cx
                vv0 = fy * p0_cam[1] / z0 + cy
                if (
                    u0 >= 0.0
                    and u0 < wp.float32(IMAGE_WIDTH)
                    and vv0 >= 0.0
                    and vv0 < wp.float32(IMAGE_HEIGHT)
                ):
                    px0 = wp.int32(wp.floor(u0))
                    py0 = wp.int32(wp.floor(vv0))
                    depth0 = visibility_depths[best_cam, py0, px0]
                    if texture_depth_z_visible(
                        depth0,
                        z0,
                        texture_depth_tolerance_m,
                        tsdf.voxel_size,
                    ):
                        uv0 = wp.vec2(
                            (atlas_x_offset + u0) / atlas_width,
                            (atlas_y_offset + vv0) / atlas_height,
                        )
                    else:
                        all_valid = False
                else:
                    all_valid = False
            else:
                all_valid = False

            p1_cam = wp.quat_rotate(cam_quat_inv, v1 - cam_pos)
            z1 = p1_cam[2]
            if all_valid and z1 > camera_min_distance and z1 <= camera_max_distance:
                u1 = fx * p1_cam[0] / z1 + cx
                vv1 = fy * p1_cam[1] / z1 + cy
                if (
                    u1 >= 0.0
                    and u1 < wp.float32(IMAGE_WIDTH)
                    and vv1 >= 0.0
                    and vv1 < wp.float32(IMAGE_HEIGHT)
                ):
                    px1 = wp.int32(wp.floor(u1))
                    py1 = wp.int32(wp.floor(vv1))
                    depth1 = visibility_depths[best_cam, py1, px1]
                    if texture_depth_z_visible(
                        depth1,
                        z1,
                        texture_depth_tolerance_m,
                        tsdf.voxel_size,
                    ):
                        uv1 = wp.vec2(
                            (atlas_x_offset + u1) / atlas_width,
                            (atlas_y_offset + vv1) / atlas_height,
                        )
                    else:
                        all_valid = False
                else:
                    all_valid = False
            else:
                all_valid = False

            p2_cam = wp.quat_rotate(cam_quat_inv, v2 - cam_pos)
            z2 = p2_cam[2]
            if all_valid and z2 > camera_min_distance and z2 <= camera_max_distance:
                u2 = fx * p2_cam[0] / z2 + cx
                vv2 = fy * p2_cam[1] / z2 + cy
                if (
                    u2 >= 0.0
                    and u2 < wp.float32(IMAGE_WIDTH)
                    and vv2 >= 0.0
                    and vv2 < wp.float32(IMAGE_HEIGHT)
                ):
                    px2 = wp.int32(wp.floor(u2))
                    py2 = wp.int32(wp.floor(vv2))
                    depth2 = visibility_depths[best_cam, py2, px2]
                    if texture_depth_z_visible(
                        depth2,
                        z2,
                        texture_depth_tolerance_m,
                        tsdf.voxel_size,
                    ):
                        uv2 = wp.vec2(
                            (atlas_x_offset + u2) / atlas_width,
                            (atlas_y_offset + vv2) / atlas_height,
                        )
                    else:
                        all_valid = False
                else:
                    all_valid = False
            else:
                all_valid = False

            if all_valid:
                if best_score > projection_scores[tri_idx]:
                    vertex_uvs[base] = uv0
                    vertex_uvs[base + wp.int32(1)] = uv1
                    vertex_uvs[base + wp.int32(2)] = uv2
                    projection_scores[tri_idx] = best_score
                projected = True
                break

            score_limit = best_score - wp.float32(1.0e-6)

        if not projected:
            if projection_scores[tri_idx] < wp.float32(-1.0e19):
                vertex_uvs[base] = invalid_uv
                vertex_uvs[base + wp.int32(1)] = invalid_uv
                vertex_uvs[base + wp.int32(2)] = invalid_uv
            if fill_missing_colors != wp.int32(0):
                c0 = sample_rgb(tsdf, v0)
                c1 = sample_rgb(tsdf, v1)
                c2 = sample_rgb(tsdf, v2)
                colors[base] = wp.vec3ub(wp.uint8(c0[0]), wp.uint8(c0[1]), wp.uint8(c0[2]))
                colors[base + wp.int32(1)] = wp.vec3ub(
                    wp.uint8(c1[0]), wp.uint8(c1[1]), wp.uint8(c1[2])
                )
                colors[base + wp.int32(2)] = wp.vec3ub(
                    wp.uint8(c2[0]), wp.uint8(c2[1]), wp.uint8(c2[2])
                )

    # =====================================================================
    # Color sampling
    # =====================================================================

    @warp_kernel(f"sample_vertex_colors_kernel_bs{block_size}", enable_backward=False)
    def sample_vertex_colors_kernel(
        vertices: wp.array(dtype=wp.vec3),
        n_vertices: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        colors: wp.array(dtype=wp.vec3ub),
    ):
        """Sample colors for mesh vertices from weighted RGB sums."""
        tid = wp.tid()
        if tid >= n_vertices:
            return

        pos = vertices[tid]

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        vx = (pos[0] - tsdf.origin[0]) / tsdf.voxel_size + center_offset_x
        vy = (pos[1] - tsdf.origin[1]) / tsdf.voxel_size + center_offset_y
        vz = (pos[2] - tsdf.origin[2]) / tsdf.voxel_size + center_offset_z

        block_size_f = wp.float32(tsdf.block_size)
        bx = wp.int32(wp.floor(vx / block_size_f))
        by = wp.int32(wp.floor(vy / block_size_f))
        bz = wp.int32(wp.floor(vz / block_size_f))

        key = block_grid_to_key_coords(bx, by, bz)
        pool_idx = hash_lookup(tsdf.hash_table, key[0], key[1], key[2], tsdf.hash_capacity)

        if pool_idx < 0:
            colors[tid] = wp.vec3ub(wp.uint8(128), wp.uint8(128), wp.uint8(128))
            return

        rgb_grid = sample_rgb(tsdf, pos)
        colors[tid] = wp.vec3ub(
            wp.uint8(rgb_grid[0]), wp.uint8(rgb_grid[1]), wp.uint8(rgb_grid[2])
        )

    # Expose kernels on the instance.
    return {
        "append_active_blocks_kernel": append_active_blocks_kernel,
        "append_active_blocks_semantic_kernel": append_active_blocks_semantic_kernel,
        "count_surface_cubes_from_blocks_kernel": count_surface_cubes_from_blocks_kernel,
        "count_surface_cubes_from_blocks_semantic_kernel": (
            count_surface_cubes_from_blocks_semantic_kernel
        ),
        "append_surface_cubes_from_blocks_kernel": append_surface_cubes_from_blocks_kernel,
        "append_surface_cubes_from_blocks_semantic_kernel": (
            append_surface_cubes_from_blocks_semantic_kernel
        ),
        "count_total_triangles_kernel": count_total_triangles_kernel,
        "generate_mesh_kernel": generate_mesh_kernel,
        "project_mesh_uvs_kernel": project_mesh_uvs_kernel,
        "sample_vertex_colors_kernel": sample_vertex_colors_kernel,
    }
