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

from curobo._src.perception.mapper.kernel.warp_types import (
    BlockSparseTSDFWarp,
    PreparedSemanticFilterWarp,
)
from curobo._src.perception.mapper.kernel.wp_tsdf_sample import (
    sample_combined_sdf,
    sample_dynamic_sdf,
    sample_static_sdf,
)
from curobo._src.perception.mapper.util.utils_quantization import (
    pack_site_coords,
)
from curobo._src.util.warp import warp_constant_suffix, warp_func, warp_kernel


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
    semantic_local_idx_keep,
) -> dict[str, object]:
    """Build ESDF seeding and distance kernels."""
    suffix = (
        f"bs{block_size}_cfg"
        f"{warp_constant_suffix(block_size, grid_shape, esdf_grid_shape, origin_xyz, voxel_size, truncation_distance)}"  # noqa: E501
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

    @warp_func(f"lookup_combined_sdf_semantic_at_esdf_coords_{suffix}")
    def lookup_combined_sdf_semantic_at_esdf_coords(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        esdf_ix: wp.int32,
        esdf_iy: wp.int32,
        esdf_iz: wp.int32,
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_vs: wp.float32,
        min_weight: wp.float32,
    ) -> wp.float32:
        coords = _esdf_to_tsdf_voxel_coords(
            esdf_ix,
            esdf_iy,
            esdf_iz,
            esdf_origin,
            esdf_vs,
        )
        if coords[3] == wp.int32(0):
            return wp.float32(1e10)
        result = _hash_lookup_at_global_coords(tsdf, coords[0], coords[1], coords[2])
        if result[0] < wp.int32(0):
            return wp.float32(1e10)
        if not semantic_local_idx_keep(
            semantic_filter,
            result[0],
            result[1],
        ):
            return wp.float32(1e10)
        return sample_combined_sdf(tsdf, result[0], result[1], min_weight)

    @warp_func(f"lookup_static_sdf_semantic_at_esdf_coords_{suffix}")
    def lookup_static_sdf_semantic_at_esdf_coords(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        esdf_ix: wp.int32,
        esdf_iy: wp.int32,
        esdf_iz: wp.int32,
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_vs: wp.float32,
    ) -> wp.float32:
        coords = _esdf_to_tsdf_voxel_coords(
            esdf_ix,
            esdf_iy,
            esdf_iz,
            esdf_origin,
            esdf_vs,
        )
        if coords[3] == wp.int32(0):
            return wp.float32(1e10)
        result = _hash_lookup_at_global_coords(tsdf, coords[0], coords[1], coords[2])
        if result[0] < wp.int32(0):
            return wp.float32(1e10)
        if not semantic_local_idx_keep(
            semantic_filter,
            result[0],
            result[1],
        ):
            return wp.float32(1e10)
        return sample_static_sdf(tsdf, result[0], result[1])

    @warp_func(f"_seed_source_at_voxel_{suffix}")
    def _seed_source_at_voxel(
        tsdf: BlockSparseTSDFWarp,
        pool_idx: wp.int32,
        local_idx: wp.int32,
        min_weight: wp.float32,
    ) -> wp.int32:
        """Return 2 for a static surface, 1 for a sensor site, or 0 for neither."""
        static_sdf = sample_static_sdf(tsdf, pool_idx, local_idx)
        if wp.abs(static_sdf) <= TSDF_VOXEL_SIZE * wp.float32(0.9):
            return wp.int32(2)
        dynamic_sdf = sample_dynamic_sdf(tsdf, pool_idx, local_idx, min_weight)
        if wp.abs(dynamic_sdf) <= TSDF_VOXEL_SIZE * wp.float32(0.9):
            return wp.int32(1)
        if dynamic_sdf < -(TRUNCATION_DIST - TSDF_VOXEL_SIZE * wp.float32(1.1)):
            return wp.int32(1)
        return wp.int32(0)

    @warp_func(f"_write_esdf_site_{suffix}")
    def _write_esdf_site(
        sites: wp.array(dtype=wp.int32),
        static_site_mask: wp.array(dtype=wp.int32),
        tid: wp.int32,
        x: wp.int32,
        y: wp.int32,
        z: wp.int32,
        source: wp.int32,
    ):
        """Write a site and atomically retain static ownership on coincidences."""
        sites[tid] = pack_site_coords(x, y, z)
        if source == wp.int32(2):
            bit = wp.int32(wp.uint32(1) << wp.uint32(tid & wp.int32(31)))
            wp.atomic_or(static_site_mask, tid >> wp.int32(5), bit)

    # =====================================================================
    # Seeding kernels.
    # =====================================================================

    @warp_kernel(f"seed_esdf_sites_from_block_sparse_kernel_{suffix}")
    def seed_esdf_sites_from_block_sparse_kernel(
        tsdf: BlockSparseTSDFWarp,
        esdf_site_index: wp.array(dtype=wp.int32),
        static_site_mask: wp.array(dtype=wp.int32),
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        minimum_tsdf_weight: wp.float32,
    ):
        """Scatter ESDF sites from static surfaces and sensor truncation bands.

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

        source = _seed_source_at_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if source == wp.int32(0):
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

        esdf_tid = esdf_x * ESDF_H * ESDF_W + esdf_y * ESDF_W + esdf_z
        _write_esdf_site(
            esdf_site_index, static_site_mask, esdf_tid, esdf_x, esdf_y, esdf_z, source
        )

    @warp_kernel(f"seed_esdf_sites_from_block_sparse_semantic_kernel_{suffix}")
    def seed_esdf_sites_from_block_sparse_semantic_kernel(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        esdf_site_index: wp.array(dtype=wp.int32),
        static_site_mask: wp.array(dtype=wp.int32),
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        minimum_tsdf_weight: wp.float32,
    ):
        """Scatter ESDF sites only from semantically retained TSDF voxels."""
        block_idx, local_idx = wp.tid()
        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < wp.int32(0):
            return
        if not semantic_local_idx_keep(semantic_filter, block_idx, local_idx):
            return

        lx = local_idx % BS
        ly = (local_idx // BS) % BS
        lz = local_idx // (BS * BS)
        bx = tsdf.block_coords[block_idx * wp.int32(3)]
        by = tsdf.block_coords[block_idx * wp.int32(3) + wp.int32(1)]
        bz = tsdf.block_coords[block_idx * wp.int32(3) + wp.int32(2)]
        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0] + lx
        gy = base[1] + ly
        gz = base[2] + lz
        if gx < wp.int32(0) or gx >= GRID_W:
            return
        if gy < wp.int32(0) or gy >= GRID_H:
            return
        if gz < wp.int32(0) or gz >= GRID_D:
            return

        source = _seed_source_at_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)
        if source == wp.int32(0):
            return
        world_x = (
            TSDF_ORIGIN_X
            + (wp.float32(gx) + wp.float32(0.5) - wp.float32(GRID_W) * wp.float32(0.5))
            * TSDF_VOXEL_SIZE
        )
        world_y = (
            TSDF_ORIGIN_Y
            + (wp.float32(gy) + wp.float32(0.5) - wp.float32(GRID_H) * wp.float32(0.5))
            * TSDF_VOXEL_SIZE
        )
        world_z = (
            TSDF_ORIGIN_Z
            + (wp.float32(gz) + wp.float32(0.5) - wp.float32(GRID_D) * wp.float32(0.5))
            * TSDF_VOXEL_SIZE
        )
        esdf_vs = esdf_voxel_size[0]
        esdf_x = wp.int32(
            (world_x - esdf_origin[0]) / esdf_vs + wp.float32(ESDF_D) * wp.float32(0.5)
        )
        esdf_y = wp.int32(
            (world_y - esdf_origin[1]) / esdf_vs + wp.float32(ESDF_H) * wp.float32(0.5)
        )
        esdf_z = wp.int32(
            (world_z - esdf_origin[2]) / esdf_vs + wp.float32(ESDF_W) * wp.float32(0.5)
        )
        if esdf_x < wp.int32(0) or esdf_x >= ESDF_D:
            return
        if esdf_y < wp.int32(0) or esdf_y >= ESDF_H:
            return
        if esdf_z < wp.int32(0) or esdf_z >= ESDF_W:
            return

        esdf_tid = esdf_x * ESDF_H * ESDF_W + esdf_y * ESDF_W + esdf_z
        _write_esdf_site(
            esdf_site_index, static_site_mask, esdf_tid, esdf_x, esdf_y, esdf_z, source
        )

    @warp_func(f"_check_seed_at_world_pos_{suffix}")
    def _check_seed_at_world_pos(
        tsdf: BlockSparseTSDFWarp,
        world_x: wp.float32,
        world_y: wp.float32,
        world_z: wp.float32,
        min_weight: wp.float32,
    ) -> wp.int32:
        """Return the seed source at a world position: static 2, sensor 1, none 0."""
        gx = wp.int32((world_x - TSDF_ORIGIN_X) / TSDF_VOXEL_SIZE + wp.float32(GRID_W) * 0.5)
        gy = wp.int32((world_y - TSDF_ORIGIN_Y) / TSDF_VOXEL_SIZE + wp.float32(GRID_H) * 0.5)
        gz = wp.int32((world_z - TSDF_ORIGIN_Z) / TSDF_VOXEL_SIZE + wp.float32(GRID_D) * 0.5)
        if gx < 0 or gx >= GRID_W or gy < 0 or gy >= GRID_H or gz < 0 or gz >= GRID_D:
            return wp.int32(0)

        bx_grid = gx // BS
        by_grid = gy // BS
        bz_grid = gz // BS
        key = block_grid_to_key_coords(bx_grid, by_grid, bz_grid)
        pool_idx = hash_lookup(tsdf.hash_table, key[0], key[1], key[2], tsdf.hash_capacity)
        if pool_idx < 0:
            return wp.int32(0)

        lx = gx % BS
        ly = gy % BS
        lz = gz % BS
        local_idx = lz * BS * BS + ly * BS + lx

        return _seed_source_at_voxel(tsdf, pool_idx, local_idx, min_weight)

    @warp_func(f"_check_seed_at_world_pos_semantic_{suffix}")
    def _check_seed_at_world_pos_semantic(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        world_x: wp.float32,
        world_y: wp.float32,
        world_z: wp.float32,
        min_weight: wp.float32,
    ) -> wp.int32:
        """Check retained semantics and seed geometry with one hash lookup."""
        gx = wp.int32(
            (world_x - TSDF_ORIGIN_X) / TSDF_VOXEL_SIZE + wp.float32(GRID_W) * wp.float32(0.5)
        )
        gy = wp.int32(
            (world_y - TSDF_ORIGIN_Y) / TSDF_VOXEL_SIZE + wp.float32(GRID_H) * wp.float32(0.5)
        )
        gz = wp.int32(
            (world_z - TSDF_ORIGIN_Z) / TSDF_VOXEL_SIZE + wp.float32(GRID_D) * wp.float32(0.5)
        )
        if gx < wp.int32(0) or gx >= GRID_W:
            return wp.int32(0)
        if gy < wp.int32(0) or gy >= GRID_H:
            return wp.int32(0)
        if gz < wp.int32(0) or gz >= GRID_D:
            return wp.int32(0)

        lookup = _hash_lookup_at_global_coords(tsdf, gx, gy, gz)
        if lookup[0] < wp.int32(0):
            return wp.int32(0)
        if not semantic_local_idx_keep(
            semantic_filter,
            lookup[0],
            lookup[1],
        ):
            return wp.int32(0)

        return _seed_source_at_voxel(tsdf, lookup[0], lookup[1], min_weight)

    @warp_kernel(f"seed_esdf_sites_gather_kernel_{suffix}")
    def seed_esdf_sites_gather_kernel(
        tsdf: BlockSparseTSDFWarp,
        esdf_site_index: wp.array(dtype=wp.int32),
        static_site_mask: wp.array(dtype=wp.int32),
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        minimum_tsdf_weight: wp.float32,
    ):
        """Gather source-aware sites with static priority; fixed launch dimensions."""
        tid = wp.tid()
        esdf_z = tid % ESDF_W
        esdf_y = (tid // ESDF_W) % ESDF_H
        esdf_x = tid // (ESDF_W * ESDF_H)
        if esdf_x >= ESDF_D:
            return

        esdf_vs = esdf_voxel_size[0]
        half_step = esdf_vs * wp.float32(0.5)
        center = wp.vec3(
            esdf_origin[0] + (wp.float32(esdf_x) + 0.5 - wp.float32(ESDF_D) * 0.5) * esdf_vs,
            esdf_origin[1] + (wp.float32(esdf_y) + 0.5 - wp.float32(ESDF_H) * 0.5) * esdf_vs,
            esdf_origin[2] + (wp.float32(esdf_z) + 0.5 - wp.float32(ESDF_W) * 0.5) * esdf_vs,
        )
        source = wp.int32(0)
        for sample in range(7):
            position = center
            if sample > 0:
                axis = (sample - 1) // 2
                direction = wp.float32(1.0)
                if sample % 2 == 0:
                    direction = wp.float32(-1.0)
                position[axis] = position[axis] + direction * half_step
            source = wp.max(
                source,
                _check_seed_at_world_pos(
                    tsdf, position[0], position[1], position[2], minimum_tsdf_weight
                ),
            )
            if source == wp.int32(2) or (source == wp.int32(1) and not tsdf.has_static):
                break
        if source != wp.int32(0):
            _write_esdf_site(
                esdf_site_index, static_site_mask, tid, esdf_x, esdf_y, esdf_z, source
            )

    @warp_kernel(f"seed_esdf_sites_gather_semantic_kernel_{suffix}")
    def seed_esdf_sites_gather_semantic_kernel(
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        esdf_site_index: wp.array(dtype=wp.int32),
        static_site_mask: wp.array(dtype=wp.int32),
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        minimum_tsdf_weight: wp.float32,
    ):
        """Gather source-aware sites with static priority; fixed launch dimensions."""
        tid = wp.tid()
        esdf_z = tid % ESDF_W
        esdf_y = (tid // ESDF_W) % ESDF_H
        esdf_x = tid // (ESDF_W * ESDF_H)
        if esdf_x >= ESDF_D:
            return

        esdf_vs = esdf_voxel_size[0]
        half_step = esdf_vs * wp.float32(0.5)
        center = wp.vec3(
            esdf_origin[0] + (wp.float32(esdf_x) + 0.5 - wp.float32(ESDF_D) * 0.5) * esdf_vs,
            esdf_origin[1] + (wp.float32(esdf_y) + 0.5 - wp.float32(ESDF_H) * 0.5) * esdf_vs,
            esdf_origin[2] + (wp.float32(esdf_z) + 0.5 - wp.float32(ESDF_W) * 0.5) * esdf_vs,
        )
        source = wp.int32(0)
        for sample in range(7):
            position = center
            if sample > 0:
                axis = (sample - 1) // 2
                direction = wp.float32(1.0)
                if sample % 2 == 0:
                    direction = wp.float32(-1.0)
                position[axis] = position[axis] + direction * half_step
            source = wp.max(
                source,
                _check_seed_at_world_pos_semantic(
                    tsdf,
                    semantic_filter,
                    position[0],
                    position[1],
                    position[2],
                    minimum_tsdf_weight,
                ),
            )
            if source == wp.int32(2) or (source == wp.int32(1) and not tsdf.has_static):
                break
        if source != wp.int32(0):
            _write_esdf_site(
                esdf_site_index, static_site_mask, tid, esdf_x, esdf_y, esdf_z, source
            )

    # =====================================================================
    # Distance kernels.
    # =====================================================================

    @warp_func(f"_sign_search_bounds_{suffix}")
    def _sign_search_bounds(
        query_distance: wp.float32,
        esdf_vs: wp.float32,
        skip_steps: wp.float32,
    ) -> wp.vec3:
        """Return band probe, local search start, and seed uncertainty in meters."""
        uncertainty = wp.sqrt(wp.float32(3.0)) * (esdf_vs + TSDF_VOXEL_SIZE)
        local_start = wp.min(query_distance, wp.max(skip_steps * esdf_vs, uncertainty))
        # A half-band probe leaves room on either side for seed displacement.
        # It is a fast candidate, not a guarantee for incomplete observations.
        # Keep the fallback local: do not march across the truncation band.
        band_probe = wp.min(query_distance, TRUNCATION_DIST * wp.float32(0.5))
        if (
            band_probe >= query_distance
            or band_probe <= local_start + TSDF_VOXEL_SIZE * wp.float32(0.5)
        ):
            band_probe = local_start
        return wp.vec3(band_probe, local_start, uncertainty)

    @warp_func(f"_line_sign_sample_coords_{suffix}")
    def _line_sign_sample_coords(
        sx: wp.int32,
        sy: wp.int32,
        sz: wp.int32,
        direction: wp.vec3,
        offset: wp.float32,
        esdf_origin: wp.array(dtype=wp.float32),
        esdf_vs: wp.float32,
    ) -> wp.vec4i:
        """Map a continuous seed-to-query line sample to a TSDF voxel."""
        seed_world = wp.vec3(
            esdf_origin[0] + (wp.float32(sx) + 0.5 - wp.float32(ESDF_D) * 0.5) * esdf_vs,
            esdf_origin[1] + (wp.float32(sy) + 0.5 - wp.float32(ESDF_H) * 0.5) * esdf_vs,
            esdf_origin[2] + (wp.float32(sz) + 0.5 - wp.float32(ESDF_W) * 0.5) * esdf_vs,
        )
        position = seed_world + offset * direction
        gx = wp.int32(
            wp.floor((position[0] - TSDF_ORIGIN_X) / TSDF_VOXEL_SIZE + wp.float32(GRID_W) * 0.5)
        )
        gy = wp.int32(
            wp.floor((position[1] - TSDF_ORIGIN_Y) / TSDF_VOXEL_SIZE + wp.float32(GRID_H) * 0.5)
        )
        gz = wp.int32(
            wp.floor((position[2] - TSDF_ORIGIN_Z) / TSDF_VOXEL_SIZE + wp.float32(GRID_D) * 0.5)
        )
        if gx < 0 or gx >= GRID_W or gy < 0 or gy >= GRID_H or gz < 0 or gz >= GRID_D:
            return wp.vec4i(0, 0, 0, 0)
        return wp.vec4i(gx, gy, gz, 1)

    @warp_kernel(f"compute_esdf_from_min_tsdf_kernel_{suffix}")
    def compute_esdf_from_min_tsdf_kernel(
        esdf_site_index: wp.array(dtype=wp.int32),
        static_site_mask: wp.array(dtype=wp.int32),
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

        # Sensor signs remain query-local; only static sites permit sign recovery.
        tsdf_sdf = lookup_combined_sdf_at_esdf_coords(
            tsdf,
            esdf_x,
            esdf_y,
            esdf_z,
            esdf_origin,
            esdf_vs,
            minimum_tsdf_weight,
        )
        # A clipped static band cannot classify points outside TSDF coverage.
        if (
            tsdf.has_static
            and tsdf_sdf >= 0.0
            and dist_voxels > 0.0
            and skip_steps > 0.0
            and _esdf_to_tsdf_voxel_coords(esdf_x, esdf_y, esdf_z, esdf_origin, esdf_vs)[3] != 0
        ):
            seed_tid = sx * ESDF_H * ESDF_W + sy * ESDF_W + sz
            is_static = (static_site_mask[seed_tid >> 5] >> (seed_tid & 31)) & 1
            if is_static != 0:
                static_sdf = lookup_static_sdf_at_esdf_coords(
                    tsdf, esdf_x, esdf_y, esdf_z, esdf_origin, esdf_vs
                )
                if static_sdf > wp.float32(1e9):
                    direction = wp.vec3(dx, dy, dz) / dist_voxels
                    # Gather and voxel quantization displace seeds from the zero crossing.
                    # Search from the query side back across the seed's uncertainty
                    # interval. A coarse scatter seed's center can itself be outside
                    # the stored band, so the interval extends slightly behind it.
                    bounds = _sign_search_bounds(edt_dist, esdf_vs, skip_steps)
                    band_probe = bounds[0]
                    search_distance = bounds[1]
                    seed_uncertainty = bounds[2]
                    step_size = TSDF_VOXEL_SIZE * wp.float32(0.5)
                    extra_probe = wp.int32(0)
                    if band_probe > search_distance:
                        extra_probe = wp.int32(1)
                    num_steps = (
                        wp.int32(wp.ceil((search_distance + seed_uncertainty) / step_size)) + 1
                    )
                    for step in range(num_steps + extra_probe):
                        offset = wp.max(
                            search_distance - wp.float32(step - extra_probe) * step_size,
                            -seed_uncertainty,
                        )
                        if step == 0:
                            offset = band_probe
                        coords = _line_sign_sample_coords(
                            sx, sy, sz, direction, offset, esdf_origin, esdf_vs
                        )
                        if coords[3] == 0:
                            continue
                        lookup = _hash_lookup_at_global_coords(
                            tsdf, coords[0], coords[1], coords[2]
                        )
                        if lookup[0] < 0:
                            continue
                        static_sdf = sample_static_sdf(tsdf, lookup[0], lookup[1])
                        if static_sdf < wp.float32(1e9):
                            break
                tsdf_sdf = wp.min(tsdf_sdf, static_sdf)
        if tsdf_sdf > 1e9:
            esdf_dist_field[tid] = wp.float16(edt_dist)
            return

        if tsdf_sdf < 0.0:
            edt_dist = -edt_dist

        esdf_dist_field[tid] = wp.float16(edt_dist)

    @warp_kernel(f"compute_esdf_from_min_tsdf_semantic_kernel_{suffix}")
    def compute_esdf_from_min_tsdf_semantic_kernel(
        esdf_site_index: wp.array(dtype=wp.int32),
        static_site_mask: wp.array(dtype=wp.int32),
        esdf_voxel_size: wp.array(dtype=wp.float32),
        esdf_dist_field: wp.array(dtype=wp.float16),
        tsdf: BlockSparseTSDFWarp,
        semantic_filter: PreparedSemanticFilterWarp,
        minimum_tsdf_weight: wp.float32,
        esdf_origin: wp.array(dtype=wp.float32),
        skip_steps: wp.float32,
    ):
        """Compute ESDF distance and sign from retained TSDF geometry."""
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

        sx = site_packed & wp.int32(0x3FF)
        sy = (site_packed >> wp.int32(10)) & wp.int32(0x3FF)
        sz = (site_packed >> wp.int32(20)) & wp.int32(0x3FF)
        esdf_vs = esdf_voxel_size[0]
        dx = wp.float32(esdf_x - sx)
        dy = wp.float32(esdf_y - sy)
        dz = wp.float32(esdf_z - sz)
        dist_voxels = wp.sqrt(dx * dx + dy * dy + dz * dz)
        edt_dist = dist_voxels * esdf_vs
        # Sensor signs remain query-local; only static sites permit sign recovery.
        tsdf_sdf = lookup_combined_sdf_semantic_at_esdf_coords(
            tsdf,
            semantic_filter,
            esdf_x,
            esdf_y,
            esdf_z,
            esdf_origin,
            esdf_vs,
            minimum_tsdf_weight,
        )
        # A clipped static band cannot classify points outside TSDF coverage.
        if (
            tsdf.has_static
            and tsdf_sdf >= 0.0
            and dist_voxels > 0.0
            and skip_steps > 0.0
            and _esdf_to_tsdf_voxel_coords(esdf_x, esdf_y, esdf_z, esdf_origin, esdf_vs)[3] != 0
        ):
            seed_tid = sx * ESDF_H * ESDF_W + sy * ESDF_W + sz
            is_static = (static_site_mask[seed_tid >> 5] >> (seed_tid & 31)) & 1
            if is_static != 0:
                static_sdf = lookup_static_sdf_semantic_at_esdf_coords(
                    tsdf, semantic_filter, esdf_x, esdf_y, esdf_z, esdf_origin, esdf_vs
                )
                if static_sdf > wp.float32(1e9):
                    direction = wp.vec3(dx, dy, dz) / dist_voxels
                    # Gather and voxel quantization displace seeds from the zero crossing.
                    # Search from the query side back across the seed's uncertainty
                    # interval. A coarse scatter seed's center can itself be outside
                    # the stored band, so the interval extends slightly behind it.
                    bounds = _sign_search_bounds(edt_dist, esdf_vs, skip_steps)
                    band_probe = bounds[0]
                    search_distance = bounds[1]
                    seed_uncertainty = bounds[2]
                    step_size = TSDF_VOXEL_SIZE * wp.float32(0.5)
                    extra_probe = wp.int32(0)
                    if band_probe > search_distance:
                        extra_probe = wp.int32(1)
                    num_steps = (
                        wp.int32(wp.ceil((search_distance + seed_uncertainty) / step_size)) + 1
                    )
                    for step in range(num_steps + extra_probe):
                        offset = wp.max(
                            search_distance - wp.float32(step - extra_probe) * step_size,
                            -seed_uncertainty,
                        )
                        if step == 0:
                            offset = band_probe
                        coords = _line_sign_sample_coords(
                            sx, sy, sz, direction, offset, esdf_origin, esdf_vs
                        )
                        if coords[3] == 0:
                            continue
                        lookup = _hash_lookup_at_global_coords(
                            tsdf, coords[0], coords[1], coords[2]
                        )
                        if lookup[0] < 0:
                            continue
                        if not semantic_local_idx_keep(semantic_filter, lookup[0], lookup[1]):
                            continue
                        static_sdf = sample_static_sdf(tsdf, lookup[0], lookup[1])
                        if static_sdf < wp.float32(1e9):
                            break
                tsdf_sdf = wp.min(tsdf_sdf, static_sdf)
        if tsdf_sdf < wp.float32(0.0):
            edt_dist = -edt_dist
        esdf_dist_field[tid] = wp.float16(edt_dist)

    return {
        "seed_esdf_sites_from_block_sparse_kernel": seed_esdf_sites_from_block_sparse_kernel,
        "seed_esdf_sites_from_block_sparse_semantic_kernel": (
            seed_esdf_sites_from_block_sparse_semantic_kernel
        ),
        "seed_esdf_sites_gather_kernel": seed_esdf_sites_gather_kernel,
        "seed_esdf_sites_gather_semantic_kernel": seed_esdf_sites_gather_semantic_kernel,
        "compute_esdf_from_min_tsdf_kernel": compute_esdf_from_min_tsdf_kernel,
        "compute_esdf_from_min_tsdf_semantic_kernel": compute_esdf_from_min_tsdf_semantic_kernel,
        "lookup_combined_sdf_at_esdf_coords": lookup_combined_sdf_at_esdf_coords,
        "lookup_static_sdf_at_esdf_coords": lookup_static_sdf_at_esdf_coords,
    }
