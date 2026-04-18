# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Generic obstacle stamping for block-sparse TSDF using Warp function overloading.

This module provides a generic implementation for stamping any obstacle type
(CuboidData, MeshData, VoxelData) into the static SDF channel. It leverages
Warp's function overloading to dispatch to type-specific SDF computations.

Key Design:
    - Single `stamp_obstacles()` function handles all obstacle types
    - Generic Warp kernels use `compute_sdf_value()` and `is_obs_enabled()` overloads
    - Kernels use `obs_set: Any` allowing Warp to compile specialized versions
    - Zero code change required to support new obstacle types (OCP)
    - Follows the pattern established in `wp_autograd.py`

Algorithm:
    Phase 1: Compute AABB block bounds from dims/inv_pose (PyTorch)
    Phase 2: Enumerate candidate blocks from AABBs
    Phase 3: Filter blocks by SDF at block center (generic kernel)
    Phase 4: Deduplicate filtered blocks (torch.unique)
    Phase 5: Pre-allocate unique blocks in hash table
    Phase 6: Stamp SDF values per voxel (generic kernel with read-min-write)
    Phase 7: Update block colors
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Union

import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_int32_tensors

if TYPE_CHECKING:
    from curobo._src.geom.data.data_cuboid import CuboidData
    from curobo._src.geom.data.data_mesh import MeshData
    from curobo._src.geom.data.data_scene import SceneData
    from curobo._src.geom.data.data_voxel import VoxelData
    from curobo._src.perception.mapper.integrator_tsdf import BlockSparseTSDF
    from curobo._src.perception.mapper.storage import BlockSparseTSDFData

# =============================================================================
# Automatic SDF Function Overload Registration
# =============================================================================
#
# Registers SDF function overloads from all obstacle data modules.
# Module list is centralized in curobo._src.geom.data.OBSTACLE_SDF_MODULES.
#
# How it works:
# - wp.func uses scope_locals.get(func.func.__name__) to find existing Functions
# - By keeping variables named "is_obs_enabled" and "compute_sdf_value" in scope,
#   subsequent wp.func calls detect the existing Function and add overloads
from importlib import import_module

from curobo._src.geom.data import OBSTACLE_SDF_MODULES
from curobo._src.perception.mapper.kernel.warp_types import (
    BLOCK_SIZE,
    BLOCK_SIZE_CUBED,
)
from curobo._src.perception.mapper.kernel.wp_coord import (
    block_local_to_world,
)
from curobo._src.perception.mapper.kernel.wp_hash import (
    free_list_pop,
    hash_lookup,
    hash_table_insert_with_pool_idx,
    pack_key_only,
    unpack_block_key,
)
from curobo._src.types.pose import Pose
from curobo._src.util.warp import get_warp_device_stream

# Initialize to None - will be set by first iteration
is_obs_enabled = None
load_obstacle_transform = None
compute_local_sdf = None

for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    _obs_fn = getattr(_data_module, "is_obs_enabled")
    _transform_fn = getattr(_data_module, "load_obstacle_transform")
    _sdf_fn = getattr(_data_module, "compute_local_sdf")

    # Register with wp.func - adds overloads on subsequent iterations
    is_obs_enabled = wp.func(_obs_fn, module=__name__)
    load_obstacle_transform = wp.func(_transform_fn, module=__name__)
    compute_local_sdf = wp.func(_sdf_fn, module=__name__)

del _module_path, _data_module, _obs_fn, _transform_fn, _sdf_fn


# =============================================================================
# Block Pre-allocation Kernel
# =============================================================================


@wp.kernel
def preallocate_unique_blocks_kernel(
    # Unique block keys (from torch.unique)
    unique_blocks: wp.array(dtype=wp.int64),
    n_unique: wp.int32,
    # Hash table
    hash_table: wp.array(dtype=wp.int64),
    hash_capacity: wp.int32,
    # Block metadata
    block_coords: wp.array(dtype=wp.int32),
    block_to_hash_slot: wp.array(dtype=wp.int32),
    # Counters
    num_allocated: wp.array(dtype=wp.int32),
    max_blocks: wp.int32,
    # Free list for recycled blocks
    free_list: wp.array(dtype=wp.int32),
    free_count: wp.array(dtype=wp.int32),
    # Per-frame tracking
    new_blocks: wp.array(dtype=wp.int32),
    new_block_count: wp.array(dtype=wp.int32),
    # Output: pool index for each unique block
    pool_indices: wp.array(dtype=wp.int32),
):
    """Pre-allocate unique blocks using CAS for collision handling.

    Each thread handles exactly one unique block key. First checks if block
    already exists, then allocates from free list or pool if needed.
    """
    tid = wp.tid()

    if tid >= n_unique:
        return

    key = unique_blocks[tid]

    # Unpack coordinates
    coords = unpack_block_key(key)
    bx = coords[0]
    by = coords[1]
    bz = coords[2]

    # First, check if block already exists in hash table
    existing_pool_idx = hash_lookup(hash_table, bx, by, bz, hash_capacity)
    if existing_pool_idx >= 0:
        # Block already exists - use existing pool index
        pool_indices[tid] = existing_pool_idx
        return

    # Block doesn't exist - need to allocate
    # Try to get pool index from free list first (recycled blocks)
    pool_idx = free_list_pop(free_list, free_count)

    if pool_idx < 0:
        # Free list empty - allocate new from pool
        pool_idx = wp.atomic_add(num_allocated, 0, wp.int32(1))

        if pool_idx >= max_blocks:
            # Pool exhausted - revert and fail
            wp.atomic_add(num_allocated, 0, wp.int32(-1))
            pool_indices[tid] = wp.int32(-1)
            return

    # Insert into hash table (handles CAS loop internally)
    result_pool_idx = hash_table_insert_with_pool_idx(
        hash_table,
        block_coords,
        block_to_hash_slot,
        new_blocks,
        new_block_count,
        bx,
        by,
        bz,
        pool_idx,
        hash_capacity,
        max_blocks,
    )

    pool_indices[tid] = result_pool_idx



# =============================================================================
# Helper Functions
# =============================================================================


@wp.func
def block_center_to_world(
    bx: wp.int32,
    by: wp.int32,
    bz: wp.int32,
    origin: wp.vec3,
    voxel_size: wp.float32,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
) -> wp.vec3:
    """Compute world position of block center."""
    block_size_f = wp.float32(block_size)
    half_block = block_size_f * 0.5

    vx = wp.float32(bx) * block_size_f + half_block
    vy = wp.float32(by) * block_size_f + half_block
    vz = wp.float32(bz) * block_size_f + half_block

    wx = (vx - wp.float32(grid_W) * 0.5) * voxel_size + origin[0]
    wy = (vy - wp.float32(grid_H) * 0.5) * voxel_size + origin[1]
    wz = (vz - wp.float32(grid_D) * 0.5) * voxel_size + origin[2]

    return wp.vec3(wx, wy, wz)


@wp.func
def is_block_in_bounds(
    bx: wp.int32,
    by: wp.int32,
    bz: wp.int32,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
) -> wp.bool:
    """Check if block coordinates are within valid grid bounds.

    When grid_W = 0 (unbounded mode), always returns True.
    """
    # Unbounded mode: grid_W = 0 means no bounds checking
    if grid_W == 0:
        return True

    blocks_W = grid_W // block_size
    blocks_H = grid_H // block_size
    blocks_D = grid_D // block_size

    if bx < 0 or bx >= blocks_W:
        return False
    if by < 0 or by >= blocks_H:
        return False
    if bz < 0 or bz >= blocks_D:
        return False
    return True


# =============================================================================
# Phase 2: Block Enumeration Kernel
# =============================================================================


@wp.kernel
def enumerate_blocks_from_aabb_kernel(
    aabb_bmin: wp.array2d(dtype=wp.int32),
    aabb_bmax: wp.array2d(dtype=wp.int32),
    n_obs: wp.int32,
    offsets: wp.array(dtype=wp.int32),
    block_keys: wp.array(dtype=wp.int64),
):
    """Enumerate blocks from pre-computed AABB bounds."""
    obs_idx = wp.tid()

    if obs_idx >= n_obs:
        return

    bmin_x = aabb_bmin[obs_idx, 0]
    bmin_y = aabb_bmin[obs_idx, 1]
    bmin_z = aabb_bmin[obs_idx, 2]
    bmax_x = aabb_bmax[obs_idx, 0]
    bmax_y = aabb_bmax[obs_idx, 1]
    bmax_z = aabb_bmax[obs_idx, 2]

    write_base = offsets[obs_idx]
    key_idx = wp.int32(0)

    for bz in range(bmin_z, bmax_z + 1):
        for by in range(bmin_y, bmax_y + 1):
            for bx in range(bmin_x, bmax_x + 1):
                key = pack_key_only(bx, by, bz)
                block_keys[write_base + key_idx] = key
                key_idx = key_idx + 1


# =============================================================================
# Phase 3: Generic Filter Kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def filter_blocks_by_sdf_kernel(
    candidate_blocks: wp.array(dtype=wp.int64),
    n_candidates: wp.int32,
    obs_set: Any,  # Generic: CuboidDataWarp, MeshDataWarp, or VoxelDataWarp
    env_idx: wp.int32,
    origin: wp.vec3,
    voxel_size: wp.float32,
    block_size: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    max_sdf_threshold: wp.float32,
    filtered_blocks: wp.array(dtype=wp.int64),
    filtered_count: wp.array(dtype=wp.int32),
):
    """Filter blocks by SDF at block center using generic compute_sdf_value.

    Uses Warp function overloading: compute_sdf_value and is_obs_enabled are
    dispatched based on the concrete type of obs_set at kernel compile time.
    """
    tid = wp.tid()

    if tid >= n_candidates:
        return

    key = candidate_blocks[tid]
    coords = unpack_block_key(key)
    bx = coords[0]
    by = coords[1]
    bz = coords[2]

    if not is_block_in_bounds(bx, by, bz, block_size, grid_W, grid_H, grid_D):
        return

    block_center = block_center_to_world(
        bx, by, bz, origin, voxel_size, block_size, grid_W, grid_H, grid_D
    )

    # Compute min SDF across all enabled obstacles
    min_sdf = wp.float32(1e10)
    for i in range(obs_set.max_n):
        if is_obs_enabled(obs_set, env_idx, i):
            inv_t = load_obstacle_transform(obs_set, env_idx, i)
            local_pt = wp.transform_point(inv_t, block_center)
            sdf = compute_local_sdf(obs_set, env_idx, i, local_pt)
            min_sdf = wp.min(min_sdf, sdf)

    if wp.abs(min_sdf) <= max_sdf_threshold:
        out_idx = wp.atomic_add(filtered_count, 0, wp.int32(1))
        filtered_blocks[out_idx] = key


# =============================================================================
# Phase 6: Generic Stamp SDF Kernel
# =============================================================================


@wp.kernel
def stamp_sdf_kernel(
    unique_blocks: wp.array(dtype=wp.int64),
    pool_indices: wp.array(dtype=wp.int32),
    n_unique: wp.int32,
    obs_set: Any,  # Generic: CuboidDataWarp, MeshDataWarp, or VoxelDataWarp
    env_idx: wp.int32,
    static_block_data: wp.array2d(dtype=wp.float16),
    static_block_sums: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: wp.float32,
    block_size: wp.int32,
    truncation: wp.float32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
):
    """Stamp SDF values using generic compute_sdf_value with read-min-write.

    Uses Warp function overloading: compute_sdf_value and is_obs_enabled are
    dispatched based on the concrete type of obs_set at kernel compile time.

    Read-min-write pattern: reads existing SDF, computes min with this type's
    contribution, writes back. This allows multiple obstacle types to accumulate.
    """
    tid = wp.tid()

    if tid >= n_unique * BLOCK_SIZE_CUBED:
        return

    block_list_idx = tid // BLOCK_SIZE_CUBED
    local_idx = tid % BLOCK_SIZE_CUBED

    if block_list_idx >= n_unique:
        return

    pool_idx = pool_indices[block_list_idx]
    if pool_idx < 0:
        return

    key = unique_blocks[block_list_idx]
    coords = unpack_block_key(key)
    bx = coords[0]
    by = coords[1]
    bz = coords[2]

    voxel_pos = block_local_to_world(
        bx, by, bz, local_idx,
        origin, voxel_size, block_size,
        grid_W, grid_H, grid_D,
    )

    # Compute min SDF across all enabled obstacles of this type
    min_sdf = wp.float32(1e10)
    for i in range(obs_set.max_n):
        if is_obs_enabled(obs_set, env_idx, i):
            inv_t = load_obstacle_transform(obs_set, env_idx, i)
            local_pt = wp.transform_point(inv_t, voxel_pos)
            sdf = compute_local_sdf(obs_set, env_idx, i, local_pt)
            min_sdf = wp.min(min_sdf, sdf)

    # Read-min-write for cross-type accumulation
    if wp.abs(min_sdf) <= truncation:
        existing = wp.float32(static_block_data[pool_idx, local_idx])
        final_sdf = wp.min(existing, min_sdf)
        clamped_sdf = wp.clamp(final_sdf, -truncation, truncation)
        was_infinite = existing > 1e9
        static_block_data[pool_idx, local_idx] = wp.float16(clamped_sdf)
        if was_infinite:
            wp.atomic_add(static_block_sums, pool_idx, wp.int32(1))


# =============================================================================
# Phase 7: Update Block Colors
# =============================================================================


@wp.kernel
def update_block_rgb_kernel(
    block_rgb: wp.array2d(dtype=wp.float32),
    pool_indices: wp.array(dtype=wp.int32),
    n_unique: wp.int32,
    static_color: wp.vec3,
):
    """Update block RGB with constant static color for allocated blocks."""
    tid = wp.tid()

    if tid >= n_unique:
        return

    pool_idx = pool_indices[tid]
    if pool_idx < 0:
        return

    block_rgb[pool_idx, 0] = static_color[0]
    block_rgb[pool_idx, 1] = static_color[1]
    block_rgb[pool_idx, 2] = static_color[2]
    block_rgb[pool_idx, 3] = 1.0


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

    Returns:
        Tuple of (bmin, bmax) where each is (N, 3) int32 tensor of block coordinates.
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

    block_size_f = float(BLOCK_SIZE)
    bmin = (vmin / block_size_f).floor().to(torch.int32)
    bmax = (vmax / block_size_f).floor().to(torch.int32)

    return bmin, bmax


def stamp_scene_obstacles(
    tsdf: "BlockSparseTSDF",
    scene: "SceneData",
    env_idx: int = 0,
    static_color: Tuple[float, float, float] = (1.0, 0.5, 0.5),
    debug: bool = True,
) -> None:
    """Stamp all enabled obstacles from a scene into the static SDF channel.

    Iterates over all obstacle types using get_valid_data(). Each type is
    stamped using the generic stamp_obstacles() function.

    Args:
        tsdf: BlockSparseTSDF instance.
        scene: SceneData containing obstacle data.
        env_idx: Environment index.
        static_color: RGB color for static obstacles.
        debug: If True, print diagnostic information.
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
    """Stamp all enabled obstacles of a single type into static SDF channel.

    This is the main entry point for generic obstacle stamping. It uses
    generic kernels that leverage Warp's function overloading for
    compute_sdf_value and is_obs_enabled.

    Algorithm:
        1. Get dims and inv_pose from obs_data (duck-typed)
        2. Compute AABB block bounds
        3. Enumerate candidate blocks
        4. Filter by SDF at block center (generic kernel)
        5. Deduplicate and allocate
        6. Stamp SDF values (generic kernel with read-min-write)
        7. Update block colors

    Args:
        tsdf: BlockSparseTSDF instance.
        obs_data: Obstacle data (CuboidData, MeshData, or VoxelData).
        env_idx: Environment index.
        static_color: RGB color for static obstacles.
        debug: If True, print diagnostic information.
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
        print(f"[stamp_obstacles] num_allocated={num_alloc}/{max_blocks} ({100*num_alloc/max_blocks:.1f}% full), free_list={free_count}")

    device = tsdf.device
    truncation = tsdf.config.truncation_distance
    voxel_size = tsdf.config.voxel_size
    block_size = tsdf.config.block_size
    warp_tsdf = tsdf.get_warp_data()

    # Get stream from torch for proper synchronization
    _, stream = get_warp_device_stream(tsdf.data.static_block_data)

    # Get dims and inv_pose
    dims = obs_data.dims[env_idx, :n_obs]
    inv_pose = obs_data.inv_pose[env_idx, :n_obs]

    # Phase 1: Compute AABB block bounds
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
    )

    if debug:
        print(f"[stamp_obstacles] grid_dims={grid_dims}, origin={origin.tolist()}")
        print(f"[stamp_obstacles] bmin={bmin.tolist()}, bmax={bmax.tolist()}")

    # Compute block counts and total
    block_counts = (bmax - bmin + 1).prod(dim=1)
    total_blocks = int(block_counts.sum().item())
    if total_blocks == 0:
        if debug:
            print("[stamp_obstacles] SKIP: total_blocks=0")
        return

    if debug:
        print(f"[stamp_obstacles] Phase 2: total_blocks={total_blocks}")

    # Phase 2: Enumerate candidate blocks
    offsets = torch.zeros(n_obs, dtype=torch.int32, device=device)
    if n_obs > 1:
        offsets[1:] = block_counts[:-1].cumsum(dim=0).to(torch.int32)

    candidate_blocks = torch.empty(total_blocks, dtype=torch.int64, device=device)
    check_int32_tensors(bmin.device, bmin=bmin, bmax=bmax)
    wp.launch(
        kernel=enumerate_blocks_from_aabb_kernel,
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

    # Phase 3: Filter blocks by SDF
    block_extent = block_size * voxel_size
    half_block_diagonal = (3.0 ** 0.5) * block_extent * 0.5
    max_sdf_threshold = truncation + half_block_diagonal

    filtered_blocks = torch.empty(total_blocks, dtype=torch.int64, device=device)
    filtered_count = torch.zeros(1, dtype=torch.int32, device=device)

    obs_wp = obs_data.to_warp()

    # Launch generic filter kernel - Warp compiles specialized version based on obs_wp type
    wp.launch(
        kernel=filter_blocks_by_sdf_kernel,
        dim=total_blocks,
        inputs=[
            wp.from_torch(candidate_blocks, dtype=wp.int64),
            total_blocks,
            obs_wp,
            env_idx,
            warp_tsdf.origin,
            voxel_size,
            block_size,
            warp_tsdf.grid_W,
            warp_tsdf.grid_H,
            warp_tsdf.grid_D,
            max_sdf_threshold,
            wp.from_torch(filtered_blocks, dtype=wp.int64),
            wp.from_torch(filtered_count, dtype=wp.int32),
        ],
        stream=stream,
    )

    # Phase 4: Deduplicate
    n_filtered = int(filtered_count.item())
    if debug:
        print(f"[stamp_obstacles] Phase 3: n_filtered={n_filtered} (max_sdf_threshold={max_sdf_threshold:.4f})")
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

    # Phase 5: Allocate blocks
    pool_indices = torch.empty(n_unique, dtype=torch.int32, device=device)

    wp.launch(
        kernel=preallocate_unique_blocks_kernel,
        dim=n_unique,
        inputs=[
            wp.from_torch(unique_blocks, dtype=wp.int64),
            n_unique,
            warp_tsdf.hash_table,
            tsdf.config.hash_capacity,
            warp_tsdf.block_coords,
            warp_tsdf.block_to_hash_slot,
            warp_tsdf.num_allocated,
            tsdf.config.max_blocks,
            warp_tsdf.free_list,
            warp_tsdf.free_count,
            warp_tsdf.new_blocks,
            warp_tsdf.new_block_count,
            wp.from_torch(pool_indices, dtype=wp.int32),
        ],
        stream=stream,
    )

    # Phase 6: Stamp SDF values - generic kernel, Warp compiles specialized version
    wp.launch(
        kernel=stamp_sdf_kernel,
        dim=n_unique * BLOCK_SIZE_CUBED,
        inputs=[
            wp.from_torch(unique_blocks, dtype=wp.int64),
            wp.from_torch(pool_indices, dtype=wp.int32),
            n_unique,
            obs_wp,
            env_idx,
            wp.from_torch(tsdf.data.static_block_data, dtype=wp.float16),
            wp.from_torch(tsdf.data.static_block_sums, dtype=wp.int32),
            warp_tsdf.origin,
            warp_tsdf.voxel_size,
            tsdf.config.block_size,
            truncation,
            warp_tsdf.grid_W,
            warp_tsdf.grid_H,
            warp_tsdf.grid_D,
        ],
        stream=stream,
    )

    # Phase 7: Update block colors
    wp.launch(
        kernel=update_block_rgb_kernel,
        dim=n_unique,
        inputs=[
            warp_tsdf.block_rgb,
            wp.from_torch(pool_indices, dtype=wp.int32),
            n_unique,
            wp.vec3(static_color[0], static_color[1], static_color[2]),
        ],
        stream=stream,
    )

    if debug:
        # PyTorch .item() will synchronize the stream automatically
        valid_pool_indices = (pool_indices >= 0).sum().item()
        # Check how many static voxels were written
        static_finite = (tsdf.data.static_block_data[:tsdf.data.num_allocated.item()].float() < 1e9).sum().item()
        print(f"[stamp_obstacles] Phase 5-7: valid_pool_indices={valid_pool_indices}/{n_unique}")
        print(f"[stamp_obstacles] Static voxels with finite SDF: {static_finite}")


# =============================================================================
# Clear Static Channel
# =============================================================================


def clear_static_channel(tsdf_data: "BlockSparseTSDFData") -> None:
    """Clear static SDF channel to +inf for all allocated blocks.

    Args:
        tsdf_data: Block-sparse TSDF data container.
    """
    if not tsdf_data.has_static:
        return

    # Simple and efficient: just fill tensors directly
    tsdf_data.static_block_data.fill_(float("inf"))
    tsdf_data.static_block_sums.zero_()
