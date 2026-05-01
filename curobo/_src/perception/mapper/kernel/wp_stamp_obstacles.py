# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Obstacle-stamping helpers, overload registration, and launchers.

The five stamp kernels are built by
:func:`curobo._src.perception.mapper.kernel.builder.builder_stamp.make_stamp_kernels`
and are reached through ``tsdf.kernels`` at launch time. What remains
here:

- The obstacle-SDF overload registration
  (:data:`is_obs_enabled`, :data:`load_obstacle_transform`,
  :data:`compute_local_sdf`). These are ``@wp.func`` objects whose
  overload set is computed by iterating
  :data:`curobo._src.geom.data.registry.OBSTACLE_SDF_MODULES` at import time.
  The stamp kernels call them via cross-module ``@wp.func`` resolution.
- The public Python API (:func:`stamp_scene_obstacles`,
  :func:`stamp_obstacles`, :func:`clear_static_channel`,
  :func:`compute_aabb_block_bounds`) used by the integrator.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Tuple, Union

import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_int32_tensors
from curobo._src.geom.data.registry import OBSTACLE_SDF_MODULES
from curobo._src.types.pose import Pose
from curobo._src.util.warp import get_warp_device_stream

# NOTE: ``BlockSparseKernels`` is NOT imported at module scope here.
# ``kernel/__init__.py`` imports the builder, which transitively imports
# ``builder.builder_stamp``. That builder imports the overloaded ``@wp.func`` names
# from this module, so importing ``curobo._src.perception.mapper.kernel`` here
# would close an import cycle. Callers reach the bundle through
# ``tsdf.kernels`` after this module has finished importing.

if TYPE_CHECKING:
    from curobo._src.geom.data.data_cuboid import CuboidData
    from curobo._src.geom.data.data_mesh import MeshData
    from curobo._src.geom.data.data_scene import SceneData
    from curobo._src.geom.data.data_voxel import VoxelData
    from curobo._src.perception.mapper.storage import (
        BlockSparseTSDF,
        BlockSparseTSDFData,
    )


# =============================================================================
# Obstacle-SDF Function Overload Registration
# =============================================================================
#
# Registers SDF function overloads from all obstacle data modules.
# Module list is centralized in curobo._src.geom.data.registry.OBSTACLE_SDF_MODULES.
#
# How it works:
# - wp.func uses scope_locals.get(func.func.__name__) to find existing Functions
# - By keeping variables named "is_obs_enabled" and "compute_local_sdf" in
#   scope, subsequent wp.func calls detect the existing Function and add
#   overloads.

is_obs_enabled = None
load_obstacle_transform = None
compute_local_sdf = None

for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    _obs_fn = getattr(_data_module, "is_obs_enabled")
    _transform_fn = getattr(_data_module, "load_obstacle_transform")
    _sdf_fn = getattr(_data_module, "compute_local_sdf")

    is_obs_enabled = wp.func(_obs_fn, module=__name__)
    load_obstacle_transform = wp.func(_transform_fn, module=__name__)
    compute_local_sdf = wp.func(_sdf_fn, module=__name__)

del _module_path, _data_module, _obs_fn, _transform_fn, _sdf_fn


# =============================================================================
# Python API
# =============================================================================


def compute_aabb_block_bounds(
    dims: torch.Tensor,
    inv_pose: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    truncation: float,
    grid_dims: Tuple[int, int, int],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute AABB block bounds for obstacles.

    Generic function that works with any obstacle type having dims and inv_pose.

    Args:
        dims: Full obstacle dimensions (N, 3) or (N, 4).
        inv_pose: Inverse pose [x, y, z, qw, qx, qy, qz, pad] (N, 8).
        origin: Grid origin (3,).
        voxel_size: Voxel size in meters.
        truncation: Truncation distance in meters.
        grid_dims: Grid dimensions (W, H, D).
        block_size: Voxels per block edge.

    Returns:
        Tuple of (bmin, bmax) where each is (N, 3) int32 tensor of
        block coordinates.
    """
    device = dims.device
    dtype = dims.dtype

    half_extents = dims[:, :3] * 0.5
    inv_pos = inv_pose[:, :3].contiguous()
    inv_quat = inv_pose[:, 3:7].contiguous()

    inv_pose_obj = Pose(position=inv_pos, quaternion=inv_quat)
    fwd_pose = inv_pose_obj.inverse()
    centers = fwd_pose.position

    R = fwd_pose.get_rotation_matrix()
    R_abs = R.abs()
    aabb_half = torch.bmm(R_abs, half_extents.unsqueeze(-1)).squeeze(-1)

    aabb_min_world = centers - aabb_half - truncation
    aabb_max_world = centers + aabb_half + truncation

    grid_W, grid_H, grid_D = grid_dims
    center_offset = torch.tensor(
        [grid_W * 0.5, grid_H * 0.5, grid_D * 0.5], device=device, dtype=dtype
    )

    vmin = ((aabb_min_world - origin) / voxel_size) + center_offset
    vmax = ((aabb_max_world - origin) / voxel_size) + center_offset

    block_size_f = float(block_size)
    bmin_grid = (vmin / block_size_f).floor().to(torch.int32)
    bmax_grid = (vmax / block_size_f).floor().to(torch.int32)

    blocks = torch.tensor(
        [
            (grid_W + block_size - 1) // block_size,
            (grid_H + block_size - 1) // block_size,
            (grid_D + block_size - 1) // block_size,
        ],
        device=device,
        dtype=torch.int32,
    )
    empty = (bmax_grid < 0).any(dim=1) | (bmin_grid >= blocks).any(dim=1)
    bmin_grid = torch.maximum(bmin_grid, torch.zeros_like(bmin_grid))
    bmax_grid = torch.minimum(bmax_grid, blocks - 1)

    offsets = blocks // 2
    bmin = bmin_grid - offsets
    bmax = bmax_grid - offsets

    if empty.any():
        bmin[empty] = 0
        bmax[empty] = -1

    return bmin, bmax


def stamp_scene_obstacles(
    tsdf: "BlockSparseTSDF",
    scene: "SceneData",
    env_idx: int = 0,
    static_color: Tuple[float, float, float] = (1.0, 0.5, 0.5),
    debug: bool = True,
) -> None:
    """Stamp all enabled obstacles from a scene into the static SDF channel.

    Iterates over all obstacle types using :meth:`get_valid_data`. Each
    type is stamped using the generic :func:`stamp_obstacles` function.
    """
    for data in scene.get_valid_data():
        stamp_obstacles(tsdf, data, env_idx, static_color, debug=debug)


def stamp_obstacles(
    tsdf: "BlockSparseTSDF",
    obs_data: Union["CuboidData", "MeshData", "VoxelData"],
    env_idx: int,
    static_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    debug: bool = False,
) -> None:
    """Stamp all enabled obstacles of a single type into the static SDF channel.

    Entry point for generic obstacle stamping. Uses generic kernels
    that leverage Warp's function overloading for ``compute_local_sdf``
    and ``is_obs_enabled``.

    Algorithm:
        1. Get dims and inv_pose from obs_data (duck-typed).
        2. Compute AABB block bounds.
        3. Enumerate candidate blocks.
        4. Filter by SDF at block center (generic kernel).
        5. Deduplicate and allocate.
        6. Stamp SDF values (generic kernel with read-min-write).
        7. Update block colors.
    """
    if not tsdf.data.has_static:
        if debug:
            print("[stamp_obstacles] SKIP: has_static=False")
        return

    n_obs = int(obs_data.count[env_idx].item())
    if n_obs == 0:
        if debug:
            print("[stamp_obstacles] SKIP: n_obs=0")
        return

    if debug:
        num_alloc = tsdf.data.num_allocated.item()
        max_blocks = tsdf.config.max_blocks
        free_count = tsdf.data.free_count.item()
        print(f"[stamp_obstacles] n_obs={n_obs}, type={type(obs_data).__name__}")
        print(
            f"[stamp_obstacles] num_allocated={num_alloc}/{max_blocks} "
            f"({100 * num_alloc / max_blocks:.1f}% full), "
            f"free_list={free_count}"
        )

    device = tsdf.device
    truncation = tsdf.config.truncation_distance
    voxel_size = tsdf.config.voxel_size
    block_size = tsdf.block_size
    warp_tsdf = tsdf.get_warp_data()
    kernels = tsdf.kernels

    _, stream = get_warp_device_stream(tsdf.data.static_block_data)

    dims = obs_data.dims[env_idx, :n_obs]
    inv_pose = obs_data.inv_pose[env_idx, :n_obs]

    origin = torch.tensor(
        [warp_tsdf.origin[0], warp_tsdf.origin[1], warp_tsdf.origin[2]],
        device=device,
        dtype=dims.dtype,
    )
    grid_dims = (warp_tsdf.grid_W, warp_tsdf.grid_H, warp_tsdf.grid_D)

    bmin, bmax = compute_aabb_block_bounds(
        dims=dims,
        inv_pose=inv_pose,
        origin=origin,
        voxel_size=voxel_size,
        truncation=truncation,
        grid_dims=grid_dims,
        block_size=block_size,
    )

    if debug:
        print(f"[stamp_obstacles] grid_dims={grid_dims}, origin={origin.tolist()}")
        print(f"[stamp_obstacles] bmin={bmin.tolist()}, bmax={bmax.tolist()}")

    block_counts = (bmax - bmin + 1).prod(dim=1)
    total_blocks = int(block_counts.sum().item())
    if total_blocks == 0:
        if debug:
            print("[stamp_obstacles] SKIP: total_blocks=0")
        return

    if debug:
        print(f"[stamp_obstacles] Phase 2: total_blocks={total_blocks}")

    offsets = torch.zeros(n_obs, dtype=torch.int32, device=device)
    if n_obs > 1:
        offsets[1:] = block_counts[:-1].cumsum(dim=0).to(torch.int32)

    candidate_blocks = torch.empty(total_blocks, dtype=torch.int64, device=device)
    check_int32_tensors(bmin.device, bmin=bmin, bmax=bmax)
    wp.launch(
        kernel=kernels.enumerate_blocks_from_aabb_kernel,
        dim=n_obs,
        inputs=[
            wp.from_torch(bmin, dtype=wp.int32),
            wp.from_torch(bmax, dtype=wp.int32),
            n_obs,
            wp.from_torch(offsets, dtype=wp.int32),
            wp.from_torch(candidate_blocks, dtype=wp.int64),
        ],
        stream=stream,
    )

    block_extent = block_size * voxel_size
    half_block_diagonal = (3.0**0.5) * block_extent * 0.5
    max_sdf_threshold = truncation + half_block_diagonal

    filtered_blocks = torch.empty(total_blocks, dtype=torch.int64, device=device)
    filtered_count = torch.zeros(1, dtype=torch.int32, device=device)

    obs_wp = obs_data.to_warp()

    wp.launch(
        kernel=kernels.filter_blocks_by_sdf_kernel,
        dim=total_blocks,
        inputs=[
            wp.from_torch(candidate_blocks, dtype=wp.int64),
            total_blocks,
            obs_wp,
            env_idx,
            wp.from_torch(filtered_blocks, dtype=wp.int64),
            wp.from_torch(filtered_count, dtype=wp.int32),
        ],
        stream=stream,
    )

    n_filtered = int(filtered_count.item())
    if debug:
        print(
            f"[stamp_obstacles] Phase 3: n_filtered={n_filtered} "
            f"(max_sdf_threshold={max_sdf_threshold:.4f})"
        )
    if n_filtered == 0:
        if debug:
            print("[stamp_obstacles] SKIP: n_filtered=0 - no blocks passed SDF filter!")
        return
    filtered_blocks = filtered_blocks[:n_filtered]
    unique_blocks = torch.unique(filtered_blocks)
    n_unique = unique_blocks.shape[0]
    if n_unique == 0:
        if debug:
            print("[stamp_obstacles] SKIP: n_unique=0")
        return

    if debug:
        print(f"[stamp_obstacles] Phase 4: n_unique={n_unique}")

    pool_indices = torch.empty(n_unique, dtype=torch.int32, device=device)
    wp.launch(
        kernel=kernels.preallocate_unique_blocks_kernel,
        dim=n_unique,
        inputs=[
            wp.from_torch(unique_blocks, dtype=wp.int64),
            n_unique,
            warp_tsdf.hash_table,
            tsdf.config.hash_capacity,
            warp_tsdf.block_coords,
            warp_tsdf.block_to_hash_slot,
            warp_tsdf.num_allocated,
            warp_tsdf.allocation_failures,
            tsdf.config.max_blocks,
            warp_tsdf.free_list,
            warp_tsdf.free_count,
            warp_tsdf.new_blocks,
            warp_tsdf.new_block_count,
            wp.from_torch(pool_indices, dtype=wp.int32),
        ],
        stream=stream,
    )

    wp.launch(
        kernel=kernels.stamp_sdf_kernel,
        dim=(n_unique, block_size**3),
        inputs=[
            wp.from_torch(unique_blocks, dtype=wp.int64),
            wp.from_torch(pool_indices, dtype=wp.int32),
            n_unique,
            obs_wp,
            env_idx,
            wp.from_torch(tsdf.data.static_block_data, dtype=wp.float16),
            wp.from_torch(tsdf.data.static_block_sums, dtype=wp.int32),
        ],
        stream=stream,
    )

    # ``block_rgb`` stores uint8-normalized weighted sums (RGB in [0, 1]).
    # Callers pass uint8-style colors (e.g. (20, 20, 20)); divide by 255
    # so the stored accumulator matches the dynamic-channel convention
    # and ``compute_avg_rgb_*_from_block`` recover the intended color.
    inv_255 = 1.0 / 255.0
    static_rgb = wp.vec3(
        float(static_color[0]) * inv_255,
        float(static_color[1]) * inv_255,
        float(static_color[2]) * inv_255,
    )
    wp.launch(
        kernel=kernels.update_block_rgb_kernel,
        dim=n_unique,
        inputs=[
            warp_tsdf.block_rgb,
            wp.from_torch(pool_indices, dtype=wp.int32),
            n_unique,
            static_rgb,
        ],
        stream=stream,
    )

    if debug:
        valid_pool_indices = (pool_indices >= 0).sum().item()
        static_finite = (
            (tsdf.data.static_block_data[: tsdf.data.num_allocated.item()].float() < 1e9)
            .sum()
            .item()
        )
        print(f"[stamp_obstacles] Phase 5-7: valid_pool_indices={valid_pool_indices}/{n_unique}")
        print(f"[stamp_obstacles] Static voxels with finite SDF: {static_finite}")


def clear_static_channel(tsdf_data: "BlockSparseTSDFData") -> None:
    """Clear static SDF channel to +inf for all allocated blocks.

    Args:
        tsdf_data: Block-sparse TSDF data container.
    """
    if not tsdf_data.has_static:
        return

    tsdf_data.static_block_data.fill_(float("inf"))
    tsdf_data.static_block_sums.zero_()
