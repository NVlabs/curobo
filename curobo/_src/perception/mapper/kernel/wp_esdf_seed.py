# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp kernels for ESDF seeding from block-sparse TSDF.

This module provides two seeding strategies:

1. **Scatter** (``seed_esdf_sites_from_block_sparse_warp``):
   One thread per TSDF voxel. Iterates allocated TSDF voxels and maps
   near-surface TSDF voxels into ESDF site cells. This is the most direct
   and faithful seeding rule. Variable launch dim - not CUDA graph safe.

2. **Gather** (``seed_esdf_sites_gather_warp``):
   One thread per ESDF voxel. Samples 7 TSDF positions via hash lookup per
   cell (center + 6 face centers) and marks a site if any sample is
   near-surface. Fixed launch dim - CUDA graph safe, but approximate and not
   numerically identical to scatter.

Because the seeding operators differ, small downstream differences in ESDF and
collision recall between scatter and gather are expected.
"""

from typing import Optional, Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_hash import (
    hash_lookup,
)
from curobo._src.perception.mapper.kernel.wp_tsdf_sample import (
    sample_combined_sdf,
)
from curobo._src.perception.mapper.storage import (
    BlockSparseTSDF,
)
from curobo._src.perception.mapper.util.utils_quantization import (
    pack_site_coords,
)
from curobo._src.util.warp import get_warp_device_stream

# =============================================================================
# ESDF Seeding Kernel (Scatter)
# =============================================================================


@wp.kernel
def seed_esdf_sites_from_block_sparse_kernel(
    # Block-sparse TSDF struct with combined SDF support
    tsdf: BlockSparseTSDFWarp,
    tsdf_origin: wp.array(dtype=wp.float32),  # (3,) tensor for CUDA graph safety
    tsdf_voxel_size: wp.float32,
    # ESDF target
    esdf_site_index: wp.array(dtype=wp.int32),
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_voxel_size: wp.array(dtype=wp.float32),
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    # Parameters
    minimum_tsdf_weight: wp.float32,
    truncation_dist: wp.float32,
    # Optional grid bounds (for center-origin convention)
    use_grid_bounds: wp.int32,  # 0 = no bounds, 1 = use bounds
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
):
    """Scatter-based ESDF surface seeding from block-sparse TSDF.

    Each thread processes one voxel in an allocated block.
    Launch with dim = num_allocated * 512 (block_size^3).

    Seeds two bands:
    1. Surface voxels (|sdf| < threshold): the zero-crossing.
    2. Truncation boundary voxels (sdf < -(trunc - threshold)): the
       deepest observed inside voxels. This prevents a sign discontinuity
       at the truncation boundary where TSDF data ends.

    Not CUDA-graph safe (launch dim depends on allocated blocks).
    """
    tid = wp.tid()

    block_idx = tid // 512
    local_idx = tid % 512

    if block_idx >= tsdf.num_allocated[0]:
        return

    if tsdf.block_to_hash_slot[block_idx] < 0:
        return

    lx = local_idx % 8
    ly = (local_idx // 8) % 8
    lz = local_idx // 64

    bx = tsdf.block_coords[block_idx * 3 + 0]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    gx = bx * 8 + lx
    gy = by * 8 + ly
    gz = bz * 8 + lz

    if use_grid_bounds != 0:
        if gx < 0 or gx >= grid_W:
            return
        if gy < 0 or gy >= grid_H:
            return
        if gz < 0 or gz >= grid_D:
            return

    sdf = sample_combined_sdf(tsdf, block_idx, local_idx, minimum_tsdf_weight)
    if sdf > 1e9:
        return

    if use_grid_bounds != 0:
        world_x = tsdf_origin[0] + (wp.float32(gx) + 0.5 - wp.float32(grid_W) * 0.5) * tsdf_voxel_size
        world_y = tsdf_origin[1] + (wp.float32(gy) + 0.5 - wp.float32(grid_H) * 0.5) * tsdf_voxel_size
        world_z = tsdf_origin[2] + (wp.float32(gz) + 0.5 - wp.float32(grid_D) * 0.5) * tsdf_voxel_size
    else:
        world_x = tsdf_origin[0] + wp.float32(gx) * tsdf_voxel_size
        world_y = tsdf_origin[1] + wp.float32(gy) * tsdf_voxel_size
        world_z = tsdf_origin[2] + wp.float32(gz) * tsdf_voxel_size

    esdf_vs = esdf_voxel_size[0]
    esdf_x = wp.int32((world_x - esdf_origin[0]) / esdf_vs + wp.float32(esdf_D) * 0.5)
    esdf_y = wp.int32((world_y - esdf_origin[1]) / esdf_vs + wp.float32(esdf_H) * 0.5)
    esdf_z = wp.int32((world_z - esdf_origin[2]) / esdf_vs + wp.float32(esdf_W) * 0.5)

    if esdf_x < 0 or esdf_x >= esdf_D:
        return
    if esdf_y < 0 or esdf_y >= esdf_H:
        return
    if esdf_z < 0 or esdf_z >= esdf_W:
        return

    surface_threshold = tsdf_voxel_size * 0.9
    is_surface = wp.abs(sdf) <= surface_threshold
    is_trunc_boundary = sdf < -(truncation_dist - tsdf_voxel_size * 1.1)
    if is_surface or is_trunc_boundary:
        esdf_tid = esdf_x * esdf_H * esdf_W + esdf_y * esdf_W + esdf_z
        esdf_site_index[esdf_tid] = pack_site_coords(
            wp.int32(esdf_x), wp.int32(esdf_y), wp.int32(esdf_z)
        )


# =============================================================================
# Gather-Based ESDF Seeding Kernel (CUDA graph safe, fixed launch dim)
# =============================================================================


@wp.func
def _check_seed_at_world_pos(
    tsdf: BlockSparseTSDFWarp,
    world_x: wp.float32,
    world_y: wp.float32,
    world_z: wp.float32,
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    min_weight: wp.float32,
    surface_threshold: wp.float32,
    truncation_dist: wp.float32,
    use_grid_bounds: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
) -> wp.bool:
    """Check if a world position maps to a seed voxel (surface or truncation boundary)."""
    if use_grid_bounds != 0:
        gx = wp.int32((world_x - tsdf_origin[0]) / tsdf_voxel_size + wp.float32(grid_W) * 0.5)
        gy = wp.int32((world_y - tsdf_origin[1]) / tsdf_voxel_size + wp.float32(grid_H) * 0.5)
        gz = wp.int32((world_z - tsdf_origin[2]) / tsdf_voxel_size + wp.float32(grid_D) * 0.5)
        if gx < 0 or gx >= grid_W or gy < 0 or gy >= grid_H or gz < 0 or gz >= grid_D:
            return False
    else:
        gx = wp.int32((world_x - tsdf_origin[0]) / tsdf_voxel_size)
        gy = wp.int32((world_y - tsdf_origin[1]) / tsdf_voxel_size)
        gz = wp.int32((world_z - tsdf_origin[2]) / tsdf_voxel_size)

    bx = gx // 8
    by = gy // 8
    bz = gz // 8
    pool_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
    if pool_idx < 0:
        return False

    lx = gx % 8
    ly = gy % 8
    lz = gz % 8
    local_idx = lz * 64 + ly * 8 + lx

    sdf = sample_combined_sdf(tsdf, pool_idx, local_idx, min_weight)
    if sdf > wp.float32(1e9):
        return False

    if wp.abs(sdf) <= surface_threshold:
        return True
    if sdf < -(truncation_dist - tsdf_voxel_size * 1.1):
        return True
    return False


@wp.kernel
def seed_esdf_sites_gather_kernel(
    tsdf: BlockSparseTSDFWarp,
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    esdf_site_index: wp.array(dtype=wp.int32),
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_voxel_size: wp.array(dtype=wp.float32),
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    minimum_tsdf_weight: wp.float32,
    truncation_dist: wp.float32,
    use_grid_bounds: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
):
    """Gather-based ESDF surface seeding. One thread per ESDF voxel.

    Samples 7 positions (center + 6 face centers) in the TSDF via hash
    lookup. If any sample is a seed voxel (surface or truncation boundary),
    marks the ESDF cell as a site for PBA propagation.

    Fixed launch dim = esdf_D * esdf_H * esdf_W. CUDA graph safe.

    Note:
        ``esdf_site_index`` must be pre-cleared to ``-1`` before launch.
    """
    tid = wp.tid()

    esdf_z = tid % esdf_W
    esdf_y = (tid // esdf_W) % esdf_H
    esdf_x = tid // (esdf_W * esdf_H)

    if esdf_x >= esdf_D:
        return

    esdf_vs = esdf_voxel_size[0]
    surface_threshold = tsdf_voxel_size * 0.9
    half_step = esdf_vs * 0.5

    # ESDF cell center in world coords
    cx = esdf_origin[0] + (wp.float32(esdf_x) + 0.5 - wp.float32(esdf_D) * 0.5) * esdf_vs
    cy = esdf_origin[1] + (wp.float32(esdf_y) + 0.5 - wp.float32(esdf_H) * 0.5) * esdf_vs
    cz = esdf_origin[2] + (wp.float32(esdf_z) + 0.5 - wp.float32(esdf_W) * 0.5) * esdf_vs

    # Sample center
    if _check_seed_at_world_pos(
        tsdf, cx, cy, cz,
        tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight, surface_threshold,
        truncation_dist, use_grid_bounds, grid_W, grid_H, grid_D,
    ):
        esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
        return

    # Sample 6 face centers, early-exit on first hit
    if _check_seed_at_world_pos(
        tsdf, cx + half_step, cy, cz,
        tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight, surface_threshold,
        truncation_dist, use_grid_bounds, grid_W, grid_H, grid_D,
    ):
        esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
        return

    if _check_seed_at_world_pos(
        tsdf, cx - half_step, cy, cz,
        tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight, surface_threshold,
        truncation_dist, use_grid_bounds, grid_W, grid_H, grid_D,
    ):
        esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
        return

    if _check_seed_at_world_pos(
        tsdf, cx, cy + half_step, cz,
        tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight, surface_threshold,
        truncation_dist, use_grid_bounds, grid_W, grid_H, grid_D,
    ):
        esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
        return

    if _check_seed_at_world_pos(
        tsdf, cx, cy - half_step, cz,
        tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight, surface_threshold,
        truncation_dist, use_grid_bounds, grid_W, grid_H, grid_D,
    ):
        esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
        return

    if _check_seed_at_world_pos(
        tsdf, cx, cy, cz + half_step,
        tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight, surface_threshold,
        truncation_dist, use_grid_bounds, grid_W, grid_H, grid_D,
    ):
        esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
        return

    if _check_seed_at_world_pos(
        tsdf, cx, cy, cz - half_step,
        tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight, surface_threshold,
        truncation_dist, use_grid_bounds, grid_W, grid_H, grid_D,
    ):
        esdf_site_index[tid] = pack_site_coords(esdf_x, esdf_y, esdf_z)
        return


def seed_esdf_sites_from_block_sparse_warp(
    tsdf: BlockSparseTSDF,
    esdf_site_index: torch.Tensor,
    esdf_origin: torch.Tensor,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    minimum_tsdf_weight: float = 0.1,
    truncation_distance: float = 0.04,
    grid_shape: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Seed ESDF sites from block-sparse TSDF (surface + truncation boundary).

    Args:
        tsdf: Block-sparse TSDF storage.
        esdf_site_index: Dense ESDF site buffer (nx, ny, nz) int32, modified in-place.
        esdf_origin: ESDF grid origin (3,) float32 tensor.
        esdf_voxel_size_tensor: ESDF voxel size (1,) float32 tensor.
        esdf_grid_shape: ESDF dimensions (nx, ny, nz) - X slowest, Z fastest.
        minimum_tsdf_weight: Minimum weight to consider voxel observed.
        truncation_distance: TSDF truncation distance in meters.
        grid_shape: Optional TSDF grid dimensions (nz, ny, nx) for bounds checking.
    """
    esdf_site_index.fill_(-1)
    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    warp_data = tsdf.get_warp_data()
    data = tsdf.data

    use_grid_bounds = 1 if grid_shape is not None else 0
    grid_D = grid_shape[0] if grid_shape else 0
    grid_H = grid_shape[1] if grid_shape else 0
    grid_W = grid_shape[2] if grid_shape else 0

    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    num_allocated = int(data.num_allocated.item())
    n_threads = num_allocated * 512
    if n_threads == 0:
        return

    wp.launch(
        kernel=seed_esdf_sites_from_block_sparse_kernel,
        dim=n_threads,
        inputs=[
            warp_data,
            wp.from_torch(data.origin.view(-1), dtype=wp.float32),
            wp.float32(data.voxel_size),
            wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32),
            wp.from_torch(esdf_origin, dtype=wp.float32),
            wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32),
            wp.int32(esdf_D),
            wp.int32(esdf_H),
            wp.int32(esdf_W),
            wp.float32(minimum_tsdf_weight),
            wp.float32(truncation_distance),
            wp.int32(use_grid_bounds),
            wp.int32(grid_W),
            wp.int32(grid_H),
            wp.int32(grid_D),
        ],
        device=wp.device_from_torch(device),
        stream=stream,
    )


def seed_esdf_sites_gather_warp(
    tsdf: BlockSparseTSDF,
    esdf_site_index: torch.Tensor,
    esdf_origin: torch.Tensor,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    minimum_tsdf_weight: float = 0.1,
    truncation_distance: float = 0.04,
    grid_shape: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Gather-based ESDF site seeding (surface + truncation boundary). CUDA graph safe.

    Args:
        tsdf: Block-sparse TSDF storage.
        esdf_site_index: Dense ESDF site buffer (nx, ny, nz) int32, modified in-place.
        esdf_origin: ESDF grid origin (3,) float32 tensor.
        esdf_voxel_size_tensor: ESDF voxel size (1,) float32 tensor.
        esdf_grid_shape: ESDF dimensions (nx, ny, nz) - X slowest, Z fastest.
        minimum_tsdf_weight: Minimum weight to consider voxel observed.
        truncation_distance: TSDF truncation distance in meters.
        grid_shape: Optional TSDF grid dimensions (nz, ny, nx) for bounds checking.
    """
    esdf_site_index.fill_(-1)

    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    n_esdf_voxels = esdf_D * esdf_H * esdf_W

    warp_data = tsdf.get_warp_data()
    data = tsdf.data

    use_grid_bounds = 1 if grid_shape is not None else 0
    grid_D = grid_shape[0] if grid_shape else 0
    grid_H = grid_shape[1] if grid_shape else 0
    grid_W = grid_shape[2] if grid_shape else 0

    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    wp.launch(
        kernel=seed_esdf_sites_gather_kernel,
        dim=n_esdf_voxels,
        inputs=[
            warp_data,
            wp.from_torch(data.origin.view(-1), dtype=wp.float32),
            wp.float32(data.voxel_size),
            wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32),
            wp.from_torch(esdf_origin, dtype=wp.float32),
            wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32),
            wp.int32(esdf_D),
            wp.int32(esdf_H),
            wp.int32(esdf_W),
            wp.float32(minimum_tsdf_weight),
            wp.float32(truncation_distance),
            wp.int32(use_grid_bounds),
            wp.int32(grid_W),
            wp.int32(grid_H),
            wp.int32(grid_D),
        ],
        device=wp.device_from_torch(device),
        stream=stream,
        block_dim=256,
    )
