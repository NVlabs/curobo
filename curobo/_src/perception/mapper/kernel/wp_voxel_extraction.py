# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse voxel extraction for debugging and visualization.

This module extracts raw voxel data from block-sparse TSDF for debugging
purposes, bypassing marching cubes mesh extraction.
"""

from typing import Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_coord import voxel_to_world
from curobo._src.perception.mapper.kernel.wp_hash import compute_avg_rgb_uint8_from_block
from curobo._src.perception.mapper.kernel.wp_raycast_common import sample_voxel
from curobo._src.util.warp import get_warp_device_stream, init_warp


@wp.kernel
def count_surface_voxels_kernel(
    tsdf: BlockSparseTSDFWarp,
    sdf_threshold: float,
    minimum_tsdf_weight: float,
    # Output
    voxel_count: wp.array(dtype=wp.int32),
):
    """Count surface voxels across all allocated blocks (combined SDF)."""
    tid = wp.tid()
    n_alloc = tsdf.num_allocated[0]
    n_voxels_per_block = tsdf.block_size * tsdf.block_size * tsdf.block_size

    block_idx = tid // n_voxels_per_block
    local_idx = tid % n_voxels_per_block

    if block_idx >= n_alloc:
        return

    # Sample combined SDF
    result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)

    if result[1] < 0.5:  # Not valid
        return

    sdf = result[0]

    if wp.abs(sdf) > sdf_threshold:
        return

    # Count this voxel
    wp.atomic_add(voxel_count, 0, 1)


@wp.kernel
def count_occupied_voxels_kernel(
    tsdf: BlockSparseTSDFWarp,
    minimum_tsdf_weight: float,
    surface_only: wp.int32,  # 1 = surface only (|sdf| < threshold), 0 = surface+inside (sdf <= 0)
    sdf_threshold: float,  # Threshold for surface_only=1 (typically voxel_size)
    # Output
    voxel_count: wp.array(dtype=wp.int32),
):
    """Count occupied voxels across all allocated blocks (combined SDF).

    Args:
        surface_only: If 1, only count voxels near zero-crossing (|SDF| < sdf_threshold).
                      If 0, count surface + inside voxels (SDF <= 0).
    """
    tid = wp.tid()
    n_alloc = tsdf.num_allocated[0]
    n_voxels_per_block = tsdf.block_size * tsdf.block_size * tsdf.block_size

    block_idx = tid // n_voxels_per_block
    local_idx = tid % n_voxels_per_block

    if block_idx >= n_alloc:
        return

    # Skip freed blocks
    if tsdf.block_to_hash_slot[block_idx] < 0:
        return

    # Sample combined SDF
    result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)

    if result[1] < 0.5:  # Not valid
        return

    sdf = result[0]

    # Check surface condition (matches dense TSDFIntegrator behavior)
    if surface_only == wp.int32(1):
        # Surface only: voxels near zero-crossing (|sdf| < threshold)
        if wp.abs(sdf) >= sdf_threshold:
            return
    else:
        # Include surface and inside voxels (sdf <= 0)
        if sdf > 0.0:
            return

    wp.atomic_add(voxel_count, 0, 1)


@wp.kernel
def extract_occupied_voxels_kernel(
    tsdf: BlockSparseTSDFWarp,
    minimum_tsdf_weight: float,
    surface_only: wp.int32,  # 1 = surface only (|sdf| < threshold), 0 = surface+inside (sdf <= 0)
    sdf_threshold: float,  # Threshold for surface_only=1 (typically voxel_size)
    # Output
    out_centers: wp.array2d(dtype=wp.float32),  # (max_voxels, 3)
    out_colors: wp.array2d(dtype=wp.uint8),  # (max_voxels, 3)
    out_count: wp.array(dtype=wp.int32),
    max_voxels: wp.int32,
):
    """Extract occupied voxel centers and colors (combined SDF)."""
    tid = wp.tid()
    n_alloc = tsdf.num_allocated[0]
    n_voxels_per_block = tsdf.block_size * tsdf.block_size * tsdf.block_size

    block_idx = tid // n_voxels_per_block
    local_idx = tid % n_voxels_per_block

    if block_idx >= n_alloc:
        return

    # Skip freed blocks
    if tsdf.block_to_hash_slot[block_idx] < 0:
        return

    # Sample combined SDF
    result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)

    if result[1] < 0.5:  # Not valid
        return

    sdf = result[0]

    # Check surface condition (matches dense TSDFIntegrator behavior)
    if surface_only == wp.int32(1):
        # Surface only: voxels near zero-crossing (|sdf| < threshold)
        if wp.abs(sdf) >= sdf_threshold:
            return
    else:
        # Include surface and inside voxels (sdf <= 0)
        if sdf > 0.0:
            return

    # Allocate output slot
    slot = wp.atomic_add(out_count, 0, 1)
    if slot >= max_voxels:
        return

    # Get block coordinates
    bx = tsdf.block_coords[block_idx * 3]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    # Local coordinates within block
    lx = local_idx % tsdf.block_size
    ly = (local_idx // tsdf.block_size) % tsdf.block_size
    lz = local_idx // (tsdf.block_size * tsdf.block_size)

    # Global voxel coordinates
    vx = bx * tsdf.block_size + lx
    vy = by * tsdf.block_size + ly
    vz = bz * tsdf.block_size + lz

    # World position of voxel center using common function
    world_pos = voxel_to_world(
        wp.vec3i(vx, vy, vz), tsdf.origin, tsdf.voxel_size,
        tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
    )

    out_centers[slot, 0] = world_pos[0]
    out_centers[slot, 1] = world_pos[1]
    out_centers[slot, 2] = world_pos[2]

    # Read per-block RGB and compute average
    rgb = compute_avg_rgb_uint8_from_block(tsdf.block_rgb, block_idx)
    out_colors[slot, 0] = wp.uint8(rgb[0])
    out_colors[slot, 1] = wp.uint8(rgb[1])
    out_colors[slot, 2] = wp.uint8(rgb[2])


@wp.kernel
def extract_surface_voxels_kernel(
    tsdf: BlockSparseTSDFWarp,
    sdf_threshold: float,
    minimum_tsdf_weight: float,
    # Output
    out_centers: wp.array2d(dtype=wp.float32),  # (max_voxels, 3)
    out_colors: wp.array2d(dtype=wp.uint8),  # (max_voxels, 3)
    out_sdf: wp.array(dtype=wp.float32),  # (max_voxels,)
    out_count: wp.array(dtype=wp.int32),
    max_voxels: wp.int32,
):
    """Extract surface voxel centers, colors, and SDF values (combined SDF)."""
    tid = wp.tid()
    n_alloc = tsdf.num_allocated[0]
    n_voxels_per_block = tsdf.block_size * tsdf.block_size * tsdf.block_size

    block_idx = tid // n_voxels_per_block
    local_idx = tid % n_voxels_per_block

    if block_idx >= n_alloc:
        return

    # Sample combined SDF
    result = sample_voxel(tsdf, block_idx, local_idx, minimum_tsdf_weight)

    if result[1] < 0.5:  # Not valid
        return

    sdf = result[0]

    if wp.abs(sdf) > sdf_threshold:
        return

    # Allocate output slot
    slot = wp.atomic_add(out_count, 0, 1)
    if slot >= max_voxels:
        return

    # Get block coordinates
    bx = tsdf.block_coords[block_idx * 3]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    # Local coordinates within block
    lx = local_idx % tsdf.block_size
    ly = (local_idx // tsdf.block_size) % tsdf.block_size
    lz = local_idx // (tsdf.block_size * tsdf.block_size)

    # Global voxel coordinates
    vx = bx * tsdf.block_size + lx
    vy = by * tsdf.block_size + ly
    vz = bz * tsdf.block_size + lz

    # World position of voxel center using common function
    world_pos = voxel_to_world(
        wp.vec3i(vx, vy, vz), tsdf.origin, tsdf.voxel_size,
        tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
    )

    out_centers[slot, 0] = world_pos[0]
    out_centers[slot, 1] = world_pos[1]
    out_centers[slot, 2] = world_pos[2]

    out_sdf[slot] = sdf

    # Read per-block RGB and compute average
    rgb = compute_avg_rgb_uint8_from_block(tsdf.block_rgb, block_idx)
    out_colors[slot, 0] = wp.uint8(rgb[0])
    out_colors[slot, 1] = wp.uint8(rgb[1])
    out_colors[slot, 2] = wp.uint8(rgb[2])


def extract_surface_voxels_block_sparse(
    tsdf,  # BlockSparseTSDF instance
    sdf_threshold: float = 0.04,
    minimum_tsdf_weight: float = 0.1,
    grid_shape: Tuple[int, int, int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract surface voxels from block-sparse TSDF (combined SDF).

    Args:
        tsdf: BlockSparseTSDF instance.
        sdf_threshold: Maximum |SDF| for surface voxels.
        minimum_tsdf_weight: Minimum weight for valid voxels.
        grid_shape: Optional (nz, ny, nx) for center-origin convention (ignored, uses struct).

    Returns:
        Tuple of (centers, colors, sdf_values):
            - centers: (N, 3) float32 voxel world positions
            - colors: (N, 3) uint8 RGB colors
            - sdf_values: (N,) float32 SDF values
    """
    init_warp()

    device = tsdf.device
    warp_data = tsdf.get_warp_data()
    num_alloc = tsdf.data.num_allocated.item()

    if num_alloc == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.uint8, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    block_size = tsdf.config.block_size
    n_voxels_per_block = block_size ** 3
    n_threads = num_alloc * n_voxels_per_block

    # Get stream from torch for proper synchronization
    _, stream = get_warp_device_stream(tsdf.data.static_block_data)

    # First pass: count surface voxels
    voxel_count = torch.zeros(1, dtype=torch.int32, device=device)

    wp.launch(
        count_surface_voxels_kernel,
        dim=n_threads,
        inputs=[
            warp_data,
            sdf_threshold,
            minimum_tsdf_weight,
            wp.from_torch(voxel_count, dtype=wp.int32),
        ],
        stream=stream,
    )

    n_surface = voxel_count.item()
    if n_surface == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.uint8, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    # Allocate output buffers
    max_voxels = n_surface + 1000  # Small buffer for race conditions
    out_centers = torch.zeros((max_voxels, 3), dtype=torch.float32, device=device)
    out_colors = torch.zeros((max_voxels, 3), dtype=torch.uint8, device=device)
    out_sdf = torch.zeros(max_voxels, dtype=torch.float32, device=device)
    out_count = torch.zeros(1, dtype=torch.int32, device=device)

    # Second pass: extract voxels
    wp.launch(
        extract_surface_voxels_kernel,
        dim=n_threads,
        inputs=[
            warp_data,
            sdf_threshold,
            minimum_tsdf_weight,
            wp.from_torch(out_centers, dtype=wp.float32),
            wp.from_torch(out_colors, dtype=wp.uint8),
            wp.from_torch(out_sdf, dtype=wp.float32),
            wp.from_torch(out_count, dtype=wp.int32),
            max_voxels,
        ],
        stream=stream,
    )

    n_extracted = min(out_count.item(), max_voxels)

    return (
        out_centers[:n_extracted],
        out_colors[:n_extracted],
        out_sdf[:n_extracted],
    )


def extract_occupied_voxels_block_sparse(
    tsdf,  # BlockSparseTSDF instance
    surface_only: bool = False,
    sdf_threshold: float = None,
    minimum_tsdf_weight: float = 0.1,
    grid_shape: Tuple[int, int, int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract occupied voxels from block-sparse TSDF (combined SDF).

    Matches dense TSDFIntegrator.extract_occupied_voxels behavior:
    - surface_only=True: extract voxels near zero-crossing (|SDF| < sdf_threshold)
    - surface_only=False: extract surface + inside voxels (SDF <= 0)

    Args:
        tsdf: BlockSparseTSDF instance.
        surface_only: If True, extract only surface voxels (|SDF| < sdf_threshold).
                      If False (default), include inside voxels too (SDF <= 0).
        sdf_threshold: Threshold for surface_only=True. Defaults to voxel_size.
        minimum_tsdf_weight: Minimum weight for valid voxels.
        grid_shape: Optional (nz, ny, nx) for center-origin convention (ignored, uses struct).

    Returns:
        Tuple of (centers, colors):
            - centers: (N, 3) float32 voxel world positions
            - colors: (N, 3) uint8 RGB colors
    """
    # Default threshold to voxel_size (matches dense behavior)
    if sdf_threshold is None:
        sdf_threshold = tsdf.config.voxel_size
    init_warp()

    device = tsdf.device
    warp_data = tsdf.get_warp_data()
    num_alloc = tsdf.data.num_allocated.item()

    if num_alloc == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.uint8, device=device),
        )

    block_size = tsdf.config.block_size
    n_voxels_per_block = block_size ** 3
    n_threads = num_alloc * n_voxels_per_block

    surface_only_int = 1 if surface_only else 0

    # Get stream from torch for proper synchronization
    _, stream = get_warp_device_stream(tsdf.data.static_block_data)

    # First pass: count voxels
    voxel_count = torch.zeros(1, dtype=torch.int32, device=device)

    wp.launch(
        count_occupied_voxels_kernel,
        dim=n_threads,
        inputs=[
            warp_data,
            minimum_tsdf_weight,
            surface_only_int,
            sdf_threshold,
            wp.from_torch(voxel_count, dtype=wp.int32),
        ],
        stream=stream,
    )

    n_occupied = voxel_count.item()
    if n_occupied == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.uint8, device=device),
        )

    # Allocate output buffers
    max_voxels = n_occupied + 1000  # Small buffer for race conditions
    out_centers = torch.zeros((max_voxels, 3), dtype=torch.float32, device=device)
    out_colors = torch.zeros((max_voxels, 3), dtype=torch.uint8, device=device)
    out_count = torch.zeros(1, dtype=torch.int32, device=device)

    # Second pass: extract voxels
    wp.launch(
        extract_occupied_voxels_kernel,
        dim=n_threads,
        inputs=[
            warp_data,
            minimum_tsdf_weight,
            surface_only_int,
            sdf_threshold,
            wp.from_torch(out_centers, dtype=wp.float32),
            wp.from_torch(out_colors, dtype=wp.uint8),
            wp.from_torch(out_count, dtype=wp.int32),
            max_voxels,
        ],
        stream=stream,
    )

    n_extracted = min(out_count.item(), max_voxels)

    return (
        out_centers[:n_extracted],
        out_colors[:n_extracted],
    )

