# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Functional builder for block-sparse Warp kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from curobo._src.perception.mapper.constants import (
    DEFAULT_HASH_LAYOUT,
    HashLayout,
    _validate_feature_channels_per_thread,
    _validate_block_size,
)
from curobo._src.perception.mapper.kernel.builder.builder_coord import make_coord_kernels
from curobo._src.perception.mapper.kernel.builder.builder_decay import make_decay_kernels
from curobo._src.perception.mapper.kernel.builder.builder_esdf import make_esdf_kernels
from curobo._src.perception.mapper.kernel.builder.builder_hash import make_hash_kernels
from curobo._src.perception.mapper.kernel.builder.builder_integrate import (
    make_integrate_kernels,
)
from curobo._src.perception.mapper.kernel.builder.builder_mesh import make_mesh_kernels
from curobo._src.perception.mapper.kernel.builder.builder_raycast import (
    make_raycast_kernels,
)
from curobo._src.perception.mapper.kernel.builder.builder_rescale import (
    make_rescale_kernels,
)
from curobo._src.perception.mapper.kernel.builder.builder_stamp import make_stamp_kernels
from curobo._src.util.logging import log_and_raise

WarpKernel: TypeAlias = Any
WarpFunction: TypeAlias = Any


@dataclass(frozen=True)
class BlockSparseKernels:
    """Flat bundle of block-size-specialized Warp handles."""

    block_size: int
    feature_dim: int
    feature_channels_per_thread: int
    max_feature_tile_channels: int
    max_support_pixels_per_block_camera: int
    hash_layout: HashLayout

    pack_entry: WarpFunction
    pack_key_only: WarpFunction
    unpack_entry: WarpFunction
    get_pool_idx: WarpFunction
    get_key_part: WarpFunction
    is_valid_block_key: WarpFunction
    spatial_hash: WarpFunction
    pack_rgb: WarpFunction
    compute_avg_rgb_from_block: WarpFunction
    compute_avg_rgb_uint8_from_block: WarpFunction
    hash_lookup: WarpFunction
    free_list_pop: WarpFunction
    free_list_push: WarpFunction
    spin_until_ready: WarpFunction
    find_or_insert_block: WarpFunction
    hash_table_insert_with_pool_idx: WarpFunction
    read_tsdf_voxel: WarpFunction
    write_tsdf_voxel: WarpFunction
    pack_block_key: WarpFunction
    unpack_block_key: WarpFunction
    clear_new_blocks_kernel: WarpKernel
    clear_new_block_features_kernel: WarpKernel

    world_to_continuous_voxel: WarpFunction
    voxel_to_world: WarpFunction
    voxel_to_world_corner: WarpFunction
    block_offsets: WarpFunction
    block_grid_to_key_coords: WarpFunction
    block_key_to_grid_coords: WarpFunction
    block_key_to_voxel_base: WarpFunction
    world_to_block_coords: WarpFunction
    world_to_block_and_local: WarpFunction
    block_local_to_world: WarpFunction
    local_to_linear_index: WarpFunction
    linear_to_local_coords: WarpFunction

    mark_blocks_in_frustum_kernel: WarpKernel
    recycle_empty_blocks_kernel: WarpKernel
    block_empty_threshold: Any

    preallocate_unique_blocks_kernel: WarpKernel
    enumerate_blocks_from_aabb_kernel: WarpKernel
    filter_blocks_by_sdf_kernel: WarpKernel
    stamp_sdf_kernel: WarpKernel
    update_block_rgb_kernel: WarpKernel

    sample_voxel: WarpFunction
    sample_tsdf: WarpFunction
    _sample_voxel_at_block_local: WarpFunction
    sample_tsdf_trilinear: WarpFunction
    sample_rgb: WarpFunction
    compute_gradient: WarpFunction
    compute_gradient_nearest: WarpFunction
    refine_hit_bisection: WarpFunction
    ray_block_exit_t: WarpFunction
    raycast_block_sparse_kernel: WarpKernel
    raycast_block_sparse_color_kernel: WarpKernel
    raycast_block_sparse_accelerated_kernel: WarpKernel
    raycast_block_sparse_accelerated_color_kernel: WarpKernel
    count_surface_voxels_kernel: WarpKernel
    count_occupied_voxels_kernel: WarpKernel
    count_occupied_voxels_masked_kernel: WarpKernel
    extract_occupied_voxels_kernel: WarpKernel
    extract_occupied_voxels_masked_kernel: WarpKernel
    extract_surface_voxels_kernel: WarpKernel

    count_surface_cubes_kernel: WarpKernel
    append_surface_cubes_kernel: WarpKernel
    count_edges_block_sparse_kernel: WarpKernel
    generate_vertices_block_sparse_kernel: WarpKernel
    generate_triangles_shared_kernel: WarpKernel
    count_triangles_kernel: WarpKernel
    sample_vertex_colors_kernel: WarpKernel

    rescale_block_accumulators_kernel: WarpKernel

    compute_block_keys_only_kernel: WarpKernel
    allocate_visible_blocks_from_keys_kernel: WarpKernel
    build_support_pixels_from_keys_kernel: WarpKernel
    collect_blocks_in_aabb_kernel: WarpKernel
    clear_blocks_by_pool_kernel: WarpKernel
    clear_block_features_by_pool_kernel: WarpKernel
    integrate_voxels_kernel: WarpKernel
    integrate_block_rgb_from_support_kernel: WarpKernel
    integrate_features_from_support_grouped_kernel: WarpKernel
    integrate_features_from_support_tiled_kernel: WarpKernel
    # Backward-compatible aliases for callers/tests that still look up the
    # pre-refactor field names. Both point at the support-list consumers.
    integrate_block_rgb_kernel: WarpKernel
    integrate_features_grouped_kernel: WarpKernel

    seed_esdf_sites_from_block_sparse_kernel: WarpKernel
    seed_esdf_sites_gather_kernel: WarpKernel
    compute_esdf_from_min_tsdf_kernel: WarpKernel
    lookup_combined_sdf_at_esdf_coords: WarpFunction
    lookup_static_sdf_at_esdf_coords: WarpFunction


def _resolve_block_size(cfg: Any | None, block_size: int | None) -> int:
    if block_size is not None:
        return block_size
    if cfg is None:
        return 8
    if isinstance(cfg, int):
        return cfg
    return int(getattr(cfg, "block_size"))


def _resolve_feature_dim(cfg: Any | None) -> int:
    if cfg is None or isinstance(cfg, int):
        return 0
    return int(getattr(cfg, "feature_dim", 0))


def _resolve_max_feature_tile_channels(cfg: Any | None) -> int:
    if cfg is None or isinstance(cfg, int):
        return 4096
    return int(getattr(cfg, "max_feature_tile_channels", 4096))


def _resolve_max_support_pixels_per_block_camera(cfg: Any | None) -> int:
    if cfg is None or isinstance(cfg, int):
        return 32
    return int(getattr(cfg, "max_support_pixels_per_block_camera", 32))


def _validate_seeding_method(cfg: Any | None, seeding_method: str | None) -> None:
    selected = seeding_method
    if selected is None and cfg is not None and not isinstance(cfg, int):
        selected = getattr(cfg, "seeding_method", None)
    if selected is not None and selected not in ("scatter", "gather"):
        log_and_raise(f"Unknown seeding_method {selected!r}; expected 'scatter' or 'gather'.")


def make_block_sparse_kernels(
    cfg: Any | None = None,
    *,
    block_size: int | None = None,
    seeding_method: str | None = None,
    feature_channels_per_thread: int | None = None,
) -> BlockSparseKernels:
    """Build a fresh flat bundle of block-sparse Warp handles."""
    resolved_block_size = _resolve_block_size(cfg, block_size)
    resolved_feature_dim = _resolve_feature_dim(cfg)
    resolved_max_feature_tile_channels = _resolve_max_feature_tile_channels(cfg)
    resolved_max_support_pixels_per_block_camera = (
        _resolve_max_support_pixels_per_block_camera(cfg)
    )
    if feature_channels_per_thread is None:
        feature_channels_per_thread = getattr(cfg, "feature_channels_per_thread", 4)
    _validate_block_size(resolved_block_size)
    _validate_feature_channels_per_thread(feature_channels_per_thread)
    if resolved_feature_dim < 0:
        log_and_raise(f"feature_dim must be >= 0, got {resolved_feature_dim}.")
    if resolved_max_feature_tile_channels <= 0:
        log_and_raise(
            "max_feature_tile_channels must be > 0, "
            f"got {resolved_max_feature_tile_channels}."
        )
    if resolved_max_support_pixels_per_block_camera <= 0:
        log_and_raise(
            "max_support_pixels_per_block_camera must be > 0, "
            f"got {resolved_max_support_pixels_per_block_camera}."
        )
    _validate_seeding_method(cfg, seeding_method)

    hash_layout = DEFAULT_HASH_LAYOUT
    hash_exports = make_hash_kernels(resolved_block_size, hash_layout)
    coord_exports = make_coord_kernels(resolved_block_size)
    decay_exports = make_decay_kernels(
        resolved_block_size,
        free_list_push=hash_exports["free_list_push"],
    )
    stamp_exports = make_stamp_kernels(
        resolved_block_size,
        pack_key_only=hash_exports["pack_key_only"],
        unpack_block_key=hash_exports["unpack_block_key"],
        block_local_to_world=coord_exports["block_local_to_world"],
        hash_lookup=hash_exports["hash_lookup"],
        hash_table_insert_with_pool_idx=hash_exports["hash_table_insert_with_pool_idx"],
        free_list_pop=hash_exports["free_list_pop"],
    )
    raycast_exports = make_raycast_kernels(
        resolved_block_size,
        hash_lookup=hash_exports["hash_lookup"],
        compute_avg_rgb_from_block=hash_exports["compute_avg_rgb_from_block"],
        compute_avg_rgb_uint8_from_block=hash_exports["compute_avg_rgb_uint8_from_block"],
        voxel_to_world=coord_exports["voxel_to_world"],
        voxel_to_world_corner=coord_exports["voxel_to_world_corner"],
        block_grid_to_key_coords=coord_exports["block_grid_to_key_coords"],
        block_key_to_voxel_base=coord_exports["block_key_to_voxel_base"],
        world_to_block_coords=coord_exports["world_to_block_coords"],
        world_to_block_and_local=coord_exports["world_to_block_and_local"],
        world_to_continuous_voxel=coord_exports["world_to_continuous_voxel"],
    )
    mesh_exports = make_mesh_kernels(
        resolved_block_size,
        hash_lookup=hash_exports["hash_lookup"],
        compute_avg_rgb_uint8_from_block=hash_exports["compute_avg_rgb_uint8_from_block"],
        sample_voxel=raycast_exports["sample_voxel"],
        sample_tsdf_trilinear=raycast_exports["sample_tsdf_trilinear"],
        compute_gradient=raycast_exports["compute_gradient"],
        compute_gradient_nearest=raycast_exports["compute_gradient_nearest"],
        block_grid_to_key_coords=coord_exports["block_grid_to_key_coords"],
        block_key_to_voxel_base=coord_exports["block_key_to_voxel_base"],
    )
    rescale_exports = make_rescale_kernels(resolved_block_size)
    integrate_exports = make_integrate_kernels(
        resolved_block_size,
        feature_dim=resolved_feature_dim,
        feature_channels_per_thread=feature_channels_per_thread,
        max_feature_tile_channels=resolved_max_feature_tile_channels,
        max_support_pixels_per_block_camera=resolved_max_support_pixels_per_block_camera,
        pack_key_only=hash_exports["pack_key_only"],
        unpack_block_key=hash_exports["unpack_block_key"],
        find_or_insert_block=hash_exports["find_or_insert_block"],
        hash_lookup=hash_exports["hash_lookup"],
        voxel_to_world=coord_exports["voxel_to_world"],
        voxel_to_world_corner=coord_exports["voxel_to_world_corner"],
        world_to_continuous_voxel=coord_exports["world_to_continuous_voxel"],
        block_local_to_world=coord_exports["block_local_to_world"],
        block_grid_to_key_coords=coord_exports["block_grid_to_key_coords"],
        block_key_to_grid_coords=coord_exports["block_key_to_grid_coords"],
    )
    esdf_exports = make_esdf_kernels(
        resolved_block_size,
        hash_lookup=hash_exports["hash_lookup"],
        block_grid_to_key_coords=coord_exports["block_grid_to_key_coords"],
        block_key_to_voxel_base=coord_exports["block_key_to_voxel_base"],
    )

    return BlockSparseKernels(
        block_size=resolved_block_size,
        feature_dim=resolved_feature_dim,
        feature_channels_per_thread=feature_channels_per_thread,
        max_feature_tile_channels=resolved_max_feature_tile_channels,
        max_support_pixels_per_block_camera=resolved_max_support_pixels_per_block_camera,
        hash_layout=hash_layout,
        **hash_exports,
        **coord_exports,
        **decay_exports,
        **stamp_exports,
        **raycast_exports,
        **mesh_exports,
        **rescale_exports,
        **integrate_exports,
        **esdf_exports,
    )
