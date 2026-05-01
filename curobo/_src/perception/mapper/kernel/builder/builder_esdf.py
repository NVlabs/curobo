# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""ESDF seeding and distance kernels, per-``block_size`` builder.

Moved from :mod:`curobo._src.perception.mapper.esdf.kernel.wp_esdf_seed` and
:mod:`curobo._src.perception.mapper.esdf.kernel.wp_esdf_distance` in the
block-size builder refactor.

Two kinds of kernels live in this builder:

1. **Seeding kernels**, both built so Python launchers can choose:
   * ``scatter`` — :func:`seed_esdf_sites_from_block_sparse_kernel`
     iterates allocated TSDF voxels. Not CUDA-graph-safe.
   * ``gather`` — :func:`seed_esdf_sites_gather_kernel` iterates ESDF
     voxels; uses ``_check_seed_at_world_pos`` helper. Graph-safe.

2. **Distance kernels**, always built:
   * :func:`compute_esdf_from_min_tsdf_kernel`

BS-sensitive points are the ``gx // BS`` / ``gx % BS`` block/local-coord
splits inside :func:`_check_seed_at_world_pos` and
:func:`_hash_lookup_at_global_coords`, plus the ``block_voxels = BS**3``
thread-dispatch inside
:func:`seed_esdf_sites_from_block_sparse_kernel`.
"""

from __future__ import annotations

import warp as wp

from curobo._src.util.warp import warp_constant_suffix, warp_func, warp_kernel
from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_tsdf_sample import (
    sample_combined_sdf,
    sample_static_sdf,
)
from curobo._src.perception.mapper.util.utils_quantization import (
    pack_site_coords,
)


def make_esdf_kernels(
    block_size: int,
    *,
    grid_shape: tuple[int, int, int],
    esdf_grid_shape: tuple[int, int, int],
    origin_xyz: tuple[float, float, float],
    voxel_size: float,
    truncation_distance: float,
    hash_lookup,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
) -> dict[str, object]:
    """Build ESDF seeding and distance kernels."""
    suffix = (
        f"bs{block_size}_cfg"
        f"{warp_constant_suffix(block_size, grid_shape, esdf_grid_shape, origin_xyz, voxel_size, truncation_distance)}"
    )
    BS = wp.constant(block_size)
    GRID_D = wp.constant(wp.int32(grid_shape[0]))
    GRID_H = wp.constant(wp.int32(grid_shape[1]))
    GRID_W = wp.constant(wp.int32(grid_shape[2]))
    ESDF_D = wp.constant(wp.int32(esdf_grid_shape[0]))
    ESDF_H = wp.constant(wp.int32(esdf_grid_shape[1]))
    ESDF_W = wp.constant(wp.int32(esdf_grid_shape[2]))
    TSDF_ORIGIN_X = wp.constant(wp.float32(origin_xyz[0]))
    TSDF_ORIGIN_Y = wp.constant(wp.float32(origin_xyz[1]))
    TSDF_ORIGIN_Z = wp.constant(wp.float32(origin_xyz[2]))
    TSDF_VOXEL_SIZE = wp.constant(wp.float32(voxel_size))
    TRUNCATION_DIST = wp.constant(wp.float32(truncation_distance))

    # ``hash_lookup`` is passed explicitly so Warp sees it as a local
    # closure binding when compiling dependent functions.

    # =====================================================================
    # Shared helpers used by both seeding variants and distance computation.
    # =====================================================================

    @warp_func(f"_hash_lookup_at_global_coords_{suffix}")
    def _hash_lookup_at_global_coords(
        tsdf: BlockSparseTSDFWarp,
        gx: wp.int32,
        gy: wp.int32,
        gz: wp.int32,
    ) -> wp.vec2i:
        """Hash lookup returning ``(pool_idx, local_idx)``.

        ``pool_idx = -1`` if not found.
        """
        bx_grid = gx // BS
        by_grid = gy // BS
        bz_grid = gz // BS
        key = block_grid_to_key_coords(bx_grid, by_grid, bz_grid)

        pool_idx = hash_lookup(tsdf.hash_table, key[0], key[1], key[2], tsdf.hash_capacity)
        if pool_idx < 0:
            return wp.vec2i(-1, 0)

        lx = gx % BS
        ly = gy % BS
        lz = gz % BS
        local_idx = lz * BS * BS + ly * BS + lx
        return wp.vec2i(pool_idx, local_idx)

    @warp_func(f"_esdf_to_tsdf_voxel_coords_{suffix}")
    def _esdf_to_tsdf_voxel_coords(
        esdf_ix: wp.int32,
        esdf_iy: wp.int32,
        esdf_iz: wp.int32,
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_vs: wp.float32,
    ) -> wp.vec4i:
        """Map ESDF grid coords to TSDF global voxel coords.

        Returns ``vec4i(gx, gy, gz, valid)``.
        """
        world_x = esdf_origin[0] + (wp.float32(esdf_ix) + 0.5 - wp.float32(ESDF_D) * 0.5) * esdf_vs
        world_y = esdf_origin[1] + (wp.float32(esdf_iy) + 0.5 - wp.float32(ESDF_H) * 0.5) * esdf_vs
        world_z = esdf_origin[2] + (wp.float32(esdf_iz) + 0.5 - wp.float32(ESDF_W) * 0.5) * esdf_vs

        gx = wp.int32(0)
        gy = wp.int32(0)
        gz = wp.int32(0)

        gx = wp.int32((world_x - TSDF_ORIGIN_X) / TSDF_VOXEL_SIZE + wp.float32(GRID_W) * 0.5)
        gy = wp.int32((world_y - TSDF_ORIGIN_Y) / TSDF_VOXEL_SIZE + wp.float32(GRID_H) * 0.5)
        gz = wp.int32((world_z - TSDF_ORIGIN_Z) / TSDF_VOXEL_SIZE + wp.float32(GRID_D) * 0.5)
        if gx < 0 or gx >= GRID_W or gy < 0 or gy >= GRID_H or gz < 0 or gz >= GRID_D:
            return wp.vec4i(0, 0, 0, 0)

        return wp.vec4i(gx, gy, gz, 1)

    @warp_func(f"lookup_combined_sdf_at_esdf_coords_{suffix}")
    def lookup_combined_sdf_at_esdf_coords(
        tsdf: BlockSparseTSDFWarp,
        esdf_ix: wp.int32,
        esdf_iy: wp.int32,
        esdf_iz: wp.int32,
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_vs: wp.float32,
        min_weight: wp.float32,
    ) -> wp.float32:
        """Sample combined (dynamic+static) SDF at an ESDF voxel position."""
        coords = _esdf_to_tsdf_voxel_coords(
            esdf_ix,
            esdf_iy,
            esdf_iz,
            esdf_origin,
            esdf_vs,
        )
        if coords[3] == 0:
            return wp.float32(1e10)

        result = _hash_lookup_at_global_coords(tsdf, coords[0], coords[1], coords[2])
        if result[0] < 0:
            return wp.float32(1e10)

        return sample_combined_sdf(tsdf, result[0], result[1], min_weight)

    @warp_func(f"lookup_static_sdf_at_esdf_coords_{suffix}")
    def lookup_static_sdf_at_esdf_coords(
        tsdf: BlockSparseTSDFWarp,
        esdf_ix: wp.int32,
        esdf_iy: wp.int32,
        esdf_iz: wp.int32,
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_vs: wp.float32,
    ) -> wp.float32:
        """Sample static-only SDF at an ESDF voxel position."""
        coords = _esdf_to_tsdf_voxel_coords(
            esdf_ix,
            esdf_iy,
            esdf_iz,
            esdf_origin,
            esdf_vs,
        )
        if coords[3] == 0:
            return wp.float32(1e10)

        result = _hash_lookup_at_global_coords(tsdf, coords[0], coords[1], coords[2])
        if result[0] < 0:
            return wp.float32(1e10)

        return sample_static_sdf(tsdf, result[0], result[1])

    # =====================================================================
    # Seeding kernels.
    # =====================================================================

    @warp_kernel(f"seed_esdf_sites_from_block_sparse_kernel_{suffix}")
    def seed_esdf_sites_from_block_sparse_kernel(
        tsdf: BlockSparseTSDFWarp,
        esdf_site_index: wp.array(dtype=wp.int32),
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        minimum_tsdf_weight: wp.float32,
    ):
        """Scatter-based ESDF surface seeding from block-sparse TSDF.

        Each thread processes one voxel in an allocated block.
        Launch with ``dim = (num_allocated, BS ** 3)``.
        """
        block_idx, local_idx = wp.tid()

        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        lx = local_idx % BS
        ly = (local_idx // BS) % BS
        lz = local_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0] + lx
        gy = base[1] + ly
        gz = base[2] + lz

        if gx < 0 or gx >= GRID_W:
            return
        if gy < 0 or gy >= GRID_H:
            return
        if gz < 0 or gz >= GRID_D:
            return

        sdf = sample_combined_sdf(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if sdf > 1e9:
            return

        world_x = (
            TSDF_ORIGIN_X + (wp.float32(gx) + 0.5 - wp.float32(GRID_W) * 0.5) * TSDF_VOXEL_SIZE
        )
        world_y = (
            TSDF_ORIGIN_Y + (wp.float32(gy) + 0.5 - wp.float32(GRID_H) * 0.5) * TSDF_VOXEL_SIZE
        )
        world_z = (
            TSDF_ORIGIN_Z + (wp.float32(gz) + 0.5 - wp.float32(GRID_D) * 0.5) * TSDF_VOXEL_SIZE
        )

        esdf_vs = esdf_voxel_size[0]
        esdf_x = wp.int32((world_x - esdf_origin[0]) / esdf_vs + wp.float32(ESDF_D) * 0.5)
        esdf_y = wp.int32((world_y - esdf_origin[1]) / esdf_vs + wp.float32(ESDF_H) * 0.5)
        esdf_z = wp.int32((world_z - esdf_origin[2]) / esdf_vs + wp.float32(ESDF_W) * 0.5)

        if esdf_x < 0 or esdf_x >= ESDF_D:
            return
        if esdf_y < 0 or esdf_y >= ESDF_H:
            return
        if esdf_z < 0 or esdf_z >= ESDF_W:
            return

        surface_threshold = TSDF_VOXEL_SIZE * 0.9
        is_surface = wp.abs(sdf) <= surface_threshold
        is_trunc_boundary = sdf < -(TRUNCATION_DIST - TSDF_VOXEL_SIZE * 1.1)
        if is_surface or is_trunc_boundary:
            esdf_tid = esdf_x * ESDF_H * ESDF_W + esdf_y * ESDF_W + esdf_z
            esdf_site_index[esdf_tid] = pack_site_coords(
                wp.int32(esdf_x), wp.int32(esdf_y), wp.int32(esdf_z)
            )

    @warp_func(f"_check_seed_at_world_pos_{suffix}")
    def _check_seed_at_world_pos(
        tsdf: BlockSparseTSDFWarp,
        world_x: wp.float32,
        world_y: wp.float32,
        world_z: wp.float32,
        min_weight: wp.float32,
    ) -> wp.bool:
        """Check if a world position maps to a seed voxel.

        Seeds are surface or truncation-boundary voxels.
        """
        gx = wp.int32((world_x - TSDF_ORIGIN_X) / TSDF_VOXEL_SIZE + wp.float32(GRID_W) * 0.5)
        gy = wp.int32((world_y - TSDF_ORIGIN_Y) / TSDF_VOXEL_SIZE + wp.float32(GRID_H) * 0.5)
        gz = wp.int32((world_z - TSDF_ORIGIN_Z) / TSDF_VOXEL_SIZE + wp.float32(GRID_D) * 0.5)
        if gx < 0 or gx >= GRID_W or gy < 0 or gy >= GRID_H or gz < 0 or gz >= GRID_D:
            return False

        bx_grid = gx // BS
        by_grid = gy // BS
        bz_grid = gz // BS
        key = block_grid_to_key_coords(bx_grid, by_grid, bz_grid)
        pool_idx = hash_lookup(tsdf.hash_table, key[0], key[1], key[2], tsdf.hash_capacity)
        if pool_idx < 0:
            return False

        lx = gx % BS
        ly = gy % BS
        lz = gz % BS
        local_idx = lz * BS * BS + ly * BS + lx

        sdf = sample_combined_sdf(tsdf, pool_idx, local_idx, min_weight)
        if sdf > wp.float32(1e9):
            return False

        if wp.abs(sdf) <= TSDF_VOXEL_SIZE * 0.9:
            return True
        if sdf < -(TRUNCATION_DIST - TSDF_VOXEL_SIZE * 1.1):
            return True
        return False

    @warp_kernel(f"seed_esdf_sites_gather_kernel_{suffix}")
    def seed_esdf_sites_gather_kernel(
        tsdf: BlockSparseTSDFWarp,
        esdf_site_index: wp.array(dtype=wp.int32),
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        minimum_tsdf_weight: wp.float32,
    ):
        """Gather-based ESDF surface seeding.

        Fixed launch dim = ``esdf_D * esdf_H * esdf_W``. CUDA
        graph safe. ``esdf_site_index`` must be pre-cleared to
        ``-1`` before launch.
        """
        tid = wp.tid()

        esdf_z = tid % ESDF_W
        esdf_y = (tid // ESDF_W) % ESDF_H
        esdf_x = tid // (ESDF_W * ESDF_H)

        if esdf_x >= ESDF_D:
            return

        esdf_vs = esdf_voxel_size[0]
        half_step = esdf_vs * 0.5

        cx = esdf_origin[0] + (wp.float32(esdf_x) + 0.5 - wp.float32(ESDF_D) * 0.5) * esdf_vs
        cy = esdf_origin[1] + (wp.float32(esdf_y) + 0.5 - wp.float32(ESDF_H) * 0.5) * esdf_vs
        cz = esdf_origin[2] + (wp.float32(esdf_z) + 0.5 - wp.float32(ESDF_W) * 0.5) * esdf_vs

        if _check_seed_at_world_pos(
            tsdf,
            cx,
            cy,
            cz,
            minimum_tsdf_weight,
        ):
            esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
            return

        if _check_seed_at_world_pos(
            tsdf,
            cx + half_step,
            cy,
            cz,
            minimum_tsdf_weight,
        ):
            esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
            return

        if _check_seed_at_world_pos(
            tsdf,
            cx - half_step,
            cy,
            cz,
            minimum_tsdf_weight,
        ):
            esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
            return

        if _check_seed_at_world_pos(
            tsdf,
            cx,
            cy + half_step,
            cz,
            minimum_tsdf_weight,
        ):
            esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
            return

        if _check_seed_at_world_pos(
            tsdf,
            cx,
            cy - half_step,
            cz,
            minimum_tsdf_weight,
        ):
            esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
            return

        if _check_seed_at_world_pos(
            tsdf,
            cx,
            cy,
            cz + half_step,
            minimum_tsdf_weight,
        ):
            esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
            return

        if _check_seed_at_world_pos(
            tsdf,
            cx,
            cy,
            cz - half_step,
            minimum_tsdf_weight,
        ):
            esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
            return

    # =====================================================================
    # Distance kernels.
    # =====================================================================

    @warp_kernel(f"compute_esdf_from_min_tsdf_kernel_{suffix}")
    def compute_esdf_from_min_tsdf_kernel(
        esdf_site_index: wp.array(dtype=wp.int32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        esdf_dist_field: wp.array(dtype=wp.float16),
        tsdf: BlockSparseTSDFWarp,
        minimum_tsdf_weight: wp.float32,
        esdf_origin: wp.array(dtype=wp.float32),
        skip_steps: wp.float32,
    ):
        """Compute ESDF distances with sign from direct TSDF hash lookup."""
        tid = wp.tid()

        esdf_z = tid % ESDF_W
        esdf_y = (tid // ESDF_W) % ESDF_H
        esdf_x = tid // (ESDF_W * ESDF_H)

        if esdf_x >= ESDF_D:
            return

        site_packed = esdf_site_index[tid]
        if site_packed < wp.int32(0):
            esdf_dist_field[tid] = wp.float16(1e4)
            return

        sx = site_packed & 0x3FF
        sy = (site_packed >> 10) & 0x3FF
        sz = (site_packed >> 20) & 0x3FF

        esdf_vs = esdf_voxel_size[0]
        dx = wp.float32(esdf_x - sx)
        dy = wp.float32(esdf_y - sy)
        dz = wp.float32(esdf_z - sz)
        dist_voxels = wp.sqrt(dx * dx + dy * dy + dz * dz)
        edt_dist = dist_voxels * esdf_vs

        tsdf_sdf = wp.float32(1e10)

        if dist_voxels > 1.0 and skip_steps > 0.0:
            inv_dist = 1.0 / dist_voxels
            adj_x = sx + wp.int32(wp.round(dx * inv_dist * skip_steps))
            adj_y = sy + wp.int32(wp.round(dy * inv_dist * skip_steps))
            adj_z = sz + wp.int32(wp.round(dz * inv_dist * skip_steps))

            if (
                adj_x >= 0
                and adj_x < ESDF_D
                and adj_y >= 0
                and adj_y < ESDF_H
                and adj_z >= 0
                and adj_z < ESDF_W
            ):
                tsdf_sdf = lookup_static_sdf_at_esdf_coords(
                    tsdf,
                    adj_x,
                    adj_y,
                    adj_z,
                    esdf_origin,
                    esdf_vs,
                )

        if tsdf_sdf > 1e9:
            tsdf_sdf = lookup_combined_sdf_at_esdf_coords(
                tsdf,
                esdf_x,
                esdf_y,
                esdf_z,
                esdf_origin,
                esdf_vs,
                minimum_tsdf_weight,
            )

        if tsdf_sdf > 1e9:
            esdf_dist_field[tid] = wp.float16(edt_dist)
            return

        if tsdf_sdf < 0.0:
            edt_dist = -edt_dist

        esdf_dist_field[tid] = wp.float16(edt_dist)

    return {
        "seed_esdf_sites_from_block_sparse_kernel": seed_esdf_sites_from_block_sparse_kernel,
        "seed_esdf_sites_gather_kernel": seed_esdf_sites_gather_kernel,
        "compute_esdf_from_min_tsdf_kernel": compute_esdf_from_min_tsdf_kernel,
        "lookup_combined_sdf_at_esdf_coords": lookup_combined_sdf_at_esdf_coords,
        "lookup_static_sdf_at_esdf_coords": lookup_static_sdf_at_esdf_coords,
    }
