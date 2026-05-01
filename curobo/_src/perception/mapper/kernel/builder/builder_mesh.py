# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse marching-cubes mesh extraction, per-``block_size`` builder.

Moved from :mod:`curobo._src.perception.mapper.mesh_extractor` in the
block-size builder refactor.

All 7 kernels and 5 ``@wp.func`` helpers are BS-sensitive: cube and
edge indexing uses ``BS``-based packing (``local_idx = lz*BS^2 +
ly*BS + lx``), and block-boundary crossing logic tests against ``BS``
directly. Every kernel closure-captures ``BS = wp.constant(block_size)``.
"""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
    binary_search_int64,
    interpolate_edge_vertex,
    local_edge_to_array_idx,
)
from curobo._src.util.warp import warp_func, warp_kernel


def make_mesh_kernels(
    block_size: int,
    *,
    hash_lookup,
    compute_avg_rgb_uint8_from_block,
    sample_voxel,
    sample_tsdf_trilinear,
    compute_gradient,
    compute_gradient_nearest,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
) -> dict[str, object]:
    """Build block-sparse marching-cubes kernels."""
    BS = wp.constant(block_size)

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
    # Edge ownership
    # =====================================================================

    @warp_func(f"get_edge_owner_combined_bs{block_size}")
    def get_edge_owner_combined(
        edge: wp.int32,
        block_idx: wp.int32,
        cx: wp.int32,
        cy: wp.int32,
        cz: wp.int32,
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        edge_owner: wp.array(dtype=wp.int32),
    ) -> wp.vec3i:
        """Find which cube owns an edge, handling block boundaries."""
        owner_dx = edge_owner[edge * 4 + 2]
        owner_dy = edge_owner[edge * 4 + 1]
        owner_dz = edge_owner[edge * 4 + 0]
        local_edge = edge_owner[edge * 4 + 3]

        new_cx = cx + owner_dx
        new_cy = cy + owner_dy
        new_cz = cz + owner_dz

        if new_cx < BS and new_cy < BS and new_cz < BS:
            owner_cube = new_cz * BS * BS + new_cy * BS + new_cx
            return wp.vec3i(block_idx, owner_cube, local_edge)

        new_bx = bx
        new_by = by
        new_bz = bz

        if new_cx >= BS:
            new_bx = bx + 1
            new_cx = 0
        if new_cy >= BS:
            new_by = by + 1
            new_cy = 0
        if new_cz >= BS:
            new_bz = bz + 1
            new_cz = 0

        neighbor_idx = hash_lookup(tsdf.hash_table, new_bx, new_by, new_bz, tsdf.hash_capacity)
        if neighbor_idx < 0:
            return wp.vec3i(-1, -1, local_edge)

        owner_cube = new_cz * BS * BS + new_cy * BS + new_cx
        return wp.vec3i(neighbor_idx, owner_cube, local_edge)

    # =====================================================================
    # Surface detection kernels
    # =====================================================================

    @warp_kernel(f"count_surface_cubes_kernel_bs{block_size}", enable_backward=False)
    def count_surface_cubes_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
    ):
        """Count cubes that contain a surface (pass 1).

        Launch with ``dim = (num_allocated, BS ** 3)``.
        """
        block_idx, cube_idx = wp.tid()

        if block_idx >= tsdf.num_allocated[0]:
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

    @warp_kernel(f"append_surface_cubes_kernel_bs{block_size}", enable_backward=False)
    def append_surface_cubes_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
    ):
        """Append surface cubes to output arrays (pass 2)."""
        block_idx, cube_idx = wp.tid()

        if block_idx >= tsdf.num_allocated[0]:
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

    # =====================================================================
    # Edge counting + vertex generation
    # =====================================================================

    @warp_kernel(f"count_edges_block_sparse_kernel_bs{block_size}", enable_backward=False)
    def count_edges_block_sparse_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        edge_counts: wp.array(dtype=wp.int32),
    ):
        """Count owned edges (0, 3, 8) for each surface cube."""
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
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        count = wp.int32(0)
        if s0[0] * s1[0] < 0.0:
            count += wp.int32(1)
        if s0[0] * s3[0] < 0.0:
            count += wp.int32(1)
        if s0[0] * s4[0] < 0.0:
            count += wp.int32(1)

        edge_counts[tid] = count

    @warp_kernel(f"generate_vertices_block_sparse_kernel_bs{block_size}", enable_backward=False)
    def generate_vertices_block_sparse_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        edge_offsets: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        refine_iterations: wp.int32,
        vertices: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        edge_vertex_indices: wp.array(dtype=wp.int32),
    ):
        """Generate vertices + normals for owned edges."""
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
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

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

        vertex_idx = edge_offsets[tid]

        edge_vertex_indices[tid * 3 + 0] = wp.int32(-1)
        edge_vertex_indices[tid * 3 + 1] = wp.int32(-1)
        edge_vertex_indices[tid * 3 + 2] = wp.int32(-1)

        if s0[0] * s1[0] < 0.0:
            v = interpolate_edge_vertex(p0, p1, s0[0], s1[0])
            if refine_iterations > 0:
                v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
            n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
            vertices[vertex_idx] = v
            normals[vertex_idx] = wp.normalize(n)
            edge_vertex_indices[tid * 3 + 0] = vertex_idx
            vertex_idx += wp.int32(1)

        if s0[0] * s3[0] < 0.0:
            v = interpolate_edge_vertex(p0, p3, s0[0], s3[0])
            if refine_iterations > 0:
                v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
            n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
            vertices[vertex_idx] = v
            normals[vertex_idx] = wp.normalize(n)
            edge_vertex_indices[tid * 3 + 1] = vertex_idx
            vertex_idx += wp.int32(1)

        if s0[0] * s4[0] < 0.0:
            v = interpolate_edge_vertex(p0, p4, s0[0], s4[0])
            if refine_iterations > 0:
                v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
            n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
            vertices[vertex_idx] = v
            normals[vertex_idx] = wp.normalize(n)
            edge_vertex_indices[tid * 3 + 2] = vertex_idx

    # =====================================================================
    # Triangle generation / counting
    # =====================================================================

    @warp_kernel(f"generate_triangles_shared_kernel_bs{block_size}")
    def generate_triangles_shared_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        tri_offsets: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        tri_table: wp.array(dtype=wp.int32),
        edge_owner: wp.array(dtype=wp.int32),
        sorted_global_ids: wp.array(dtype=wp.int64),
        sparse_indices_sorted: wp.array(dtype=wp.int32),
        edge_vertex_indices: wp.array(dtype=wp.int32),
        triangles: wp.array(dtype=wp.int32),
    ):
        """Generate triangles with shared vertex lookup."""
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

        tri_base = tri_offsets[tid]
        table_offset = cube_config * 16

        for t in range(5):
            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break

            e1 = tri_table[table_offset + t * 3 + 1]
            e2 = tri_table[table_offset + t * 3 + 2]

            for v_idx in range(3):
                edge = e0
                if v_idx == 1:
                    edge = e1
                elif v_idx == 2:
                    edge = e2

                owner = get_edge_owner_combined(
                    edge,
                    block_idx,
                    cx,
                    cy,
                    cz,
                    bx,
                    by,
                    bz,
                    tsdf,
                    edge_owner,
                )

                vertex_id = wp.int32(-1)

                if owner[0] >= 0:
                    owner_global_id = wp.int64(owner[0]) * wp.int64(BS * BS * BS) + wp.int64(
                        owner[1]
                    )
                    search_idx = binary_search_int64(
                        sorted_global_ids,
                        n_surfaces,
                        owner_global_id,
                    )
                    if search_idx >= 0:
                        owner_sparse = sparse_indices_sorted[search_idx]
                        edge_array_idx = local_edge_to_array_idx(owner[2])
                        vertex_id = edge_vertex_indices[owner_sparse * 3 + edge_array_idx]

                triangles[(tri_base + t) * 3 + v_idx] = vertex_id

    @warp_kernel(f"count_triangles_kernel_bs{block_size}", enable_backward=False)
    def count_triangles_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        num_tris_table: wp.array(dtype=wp.int32),
        tri_counts: wp.array(dtype=wp.int32),
    ):
        """Count triangles for each surface cube (combined SDF)."""
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

        tri_counts[tid] = num_tris_table[cube_config]

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

        rgb = compute_avg_rgb_uint8_from_block(tsdf.block_rgb, pool_idx)
        colors[tid] = wp.vec3ub(wp.uint8(rgb[0]), wp.uint8(rgb[1]), wp.uint8(rgb[2]))

    # Expose kernels on the instance.
    return {
        "count_surface_cubes_kernel": count_surface_cubes_kernel,
        "append_surface_cubes_kernel": append_surface_cubes_kernel,
        "count_edges_block_sparse_kernel": count_edges_block_sparse_kernel,
        "generate_vertices_block_sparse_kernel": generate_vertices_block_sparse_kernel,
        "generate_triangles_shared_kernel": generate_triangles_shared_kernel,
        "count_triangles_kernel": count_triangles_kernel,
        "sample_vertex_colors_kernel": sample_vertex_colors_kernel,
    }
