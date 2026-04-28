# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse voxel extraction launch wrappers.

The raw extraction kernels are built by
:func:`curobo._src.perception.mapper.kernel.builder.builder_raycast.make_raycast_kernels`
and are reached through ``tsdf.kernels`` at launch time. This module
only hosts the public Python wrappers
(:func:`extract_surface_voxels_block_sparse`,
:func:`extract_occupied_voxels_block_sparse`,
:func:`extract_matching_voxels_block_sparse`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.storage import BlockDataView, OccupiedVoxels
from curobo._src.util.warp import get_warp_device_stream, init_warp


def _build_block_data_view(tsdf, num_alloc: int) -> BlockDataView:
    """Zero-copy view over per-block storage tensors."""
    return BlockDataView(
        rgb=tsdf.data.block_rgb,
        coords=tsdf.data.block_coords,
        num_allocated=num_alloc,
        voxel_size=tsdf.config.voxel_size,
        block_size=tsdf.block_size,
        features=tsdf.data.block_features,
        feature_weight=tsdf.data.block_feature_weight,
        feature_dim=tsdf.data.feature_dim,
    )


def extract_surface_voxels_block_sparse(
    tsdf,
    sdf_threshold: float = 0.04,
    minimum_tsdf_weight: float = 0.1,
    grid_shape: Tuple[int, int, int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract surface voxels from block-sparse TSDF (combined SDF).

    Args:
        tsdf: BlockSparseTSDF instance.
        sdf_threshold: Maximum |SDF| for surface voxels.
        minimum_tsdf_weight: Minimum weight for valid voxels.
        grid_shape: Ignored; present for API compatibility.

    Returns:
        Tuple of (centers, colors, sdf_values) as torch tensors.
    """
    init_warp()

    device = tsdf.device
    warp_data = tsdf.get_warp_data()
    num_alloc = tsdf.data.num_allocated.item()
    kernels = tsdf.kernels

    if num_alloc == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.uint8, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    block_size = tsdf.block_size
    n_voxels_per_block = block_size**3

    _, stream = get_warp_device_stream(tsdf.data.static_block_data)

    voxel_count = torch.zeros(1, dtype=torch.int32, device=device)
    wp.launch(
        kernels.count_surface_voxels_kernel,
        dim=(num_alloc, n_voxels_per_block),
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

    max_voxels = n_surface + 1000
    out_centers = torch.zeros((max_voxels, 3), dtype=torch.float32, device=device)
    out_colors = torch.zeros((max_voxels, 3), dtype=torch.uint8, device=device)
    out_sdf = torch.zeros(max_voxels, dtype=torch.float32, device=device)
    out_count = torch.zeros(1, dtype=torch.int32, device=device)

    wp.launch(
        kernels.extract_surface_voxels_kernel,
        dim=(num_alloc, n_voxels_per_block),
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
    tsdf,
    surface_only: bool = False,
    sdf_threshold: float = None,
    minimum_tsdf_weight: float = 0.1,
    grid_shape: Tuple[int, int, int] = None,
) -> OccupiedVoxels:
    """Extract occupied voxels from block-sparse TSDF (combined SDF).

    Matches dense TSDFIntegrator.extract_occupied_voxels behavior.

    Args:
        tsdf: BlockSparseTSDF instance.
        surface_only: If True, extract only surface voxels.
        sdf_threshold: Threshold for surface_only=True. Defaults to voxel_size.
        minimum_tsdf_weight: Minimum weight for valid voxels.
        grid_shape: Ignored; present for API compatibility.

    Returns:
        :class:`OccupiedVoxels` with ``centers``, ``block_idx_per_voxel``,
        and a ``block_data`` view over per-block storage.
    """
    if sdf_threshold is None:
        sdf_threshold = tsdf.config.voxel_size
    init_warp()

    device = tsdf.device
    warp_data = tsdf.get_warp_data()
    num_alloc = tsdf.data.num_allocated.item()
    kernels = tsdf.kernels

    view = _build_block_data_view(tsdf, num_alloc)

    if num_alloc == 0:
        return OccupiedVoxels(
            centers=torch.empty((0, 3), dtype=torch.float32, device=device),
            block_idx_per_voxel=torch.empty((0,), dtype=torch.int32, device=device),
            block_data=view,
        )

    block_size = tsdf.block_size
    n_voxels_per_block = block_size**3

    surface_only_int = 1 if surface_only else 0

    _, stream = get_warp_device_stream(tsdf.data.static_block_data)

    voxel_count = torch.zeros(1, dtype=torch.int32, device=device)
    wp.launch(
        kernels.count_occupied_voxels_kernel,
        dim=(num_alloc, n_voxels_per_block),
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
        return OccupiedVoxels(
            centers=torch.empty((0, 3), dtype=torch.float32, device=device),
            block_idx_per_voxel=torch.empty((0,), dtype=torch.int32, device=device),
            block_data=view,
        )

    max_voxels = n_occupied + 1000
    out_centers = torch.zeros((max_voxels, 3), dtype=torch.float32, device=device)
    out_pool_idx = torch.zeros(max_voxels, dtype=torch.int32, device=device)
    out_count = torch.zeros(1, dtype=torch.int32, device=device)

    wp.launch(
        kernels.extract_occupied_voxels_kernel,
        dim=(num_alloc, n_voxels_per_block),
        inputs=[
            warp_data,
            minimum_tsdf_weight,
            surface_only_int,
            sdf_threshold,
            wp.from_torch(out_centers, dtype=wp.float32),
            wp.from_torch(out_pool_idx, dtype=wp.int32),
            wp.from_torch(out_count, dtype=wp.int32),
            max_voxels,
        ],
        stream=stream,
    )

    n_extracted = min(out_count.item(), max_voxels)
    return OccupiedVoxels(
        centers=out_centers[:n_extracted],
        block_idx_per_voxel=out_pool_idx[:n_extracted],
        block_data=view,
    )


def extract_matching_voxels_block_sparse(
    tsdf,
    block_mask: torch.Tensor,
    surface_only: bool = False,
    sdf_threshold: Optional[float] = None,
    minimum_tsdf_weight: float = 0.1,
) -> OccupiedVoxels:
    """Extract occupied voxels only from blocks flagged in ``block_mask``.

    Counterpart to :func:`extract_occupied_voxels_block_sparse` that
    restricts extraction to a caller-supplied subset of blocks. Used by
    the feature-matching query API to materialize voxels only from the
    top-k most feature-similar blocks without a second scan.

    Args:
        tsdf: BlockSparseTSDF instance.
        block_mask: ``(max_blocks,)`` uint8/bool tensor. Non-zero entries
            select blocks to extract from.
        surface_only: If True, keep only surface voxels (|SDF| < threshold).
        sdf_threshold: Threshold for surface_only. Defaults to voxel_size.
        minimum_tsdf_weight: Minimum weight for valid voxels.

    Returns:
        :class:`OccupiedVoxels` restricted to masked blocks.
    """
    if sdf_threshold is None:
        sdf_threshold = tsdf.config.voxel_size
    init_warp()

    if block_mask.dtype != torch.uint8:
        block_mask = block_mask.to(torch.uint8)
    if block_mask.device != tsdf.data.block_rgb.device:
        block_mask = block_mask.to(tsdf.data.block_rgb.device)
    if block_mask.shape[0] < tsdf.config.max_blocks:
        raise ValueError(
            f"block_mask must cover max_blocks={tsdf.config.max_blocks} entries, "
            f"got shape {tuple(block_mask.shape)}"
        )

    device = tsdf.device
    warp_data = tsdf.get_warp_data()
    num_alloc = tsdf.data.num_allocated.item()
    kernels = tsdf.kernels

    view = _build_block_data_view(tsdf, num_alloc)

    if num_alloc == 0:
        return OccupiedVoxels(
            centers=torch.empty((0, 3), dtype=torch.float32, device=device),
            block_idx_per_voxel=torch.empty((0,), dtype=torch.int32, device=device),
            block_data=view,
        )

    block_size = tsdf.block_size
    n_voxels_per_block = block_size**3

    surface_only_int = 1 if surface_only else 0

    _, stream = get_warp_device_stream(tsdf.data.static_block_data)
    mask_wp = wp.from_torch(block_mask, dtype=wp.uint8)

    voxel_count = torch.zeros(1, dtype=torch.int32, device=device)
    wp.launch(
        kernels.count_occupied_voxels_masked_kernel,
        dim=(num_alloc, n_voxels_per_block),
        inputs=[
            warp_data,
            minimum_tsdf_weight,
            surface_only_int,
            sdf_threshold,
            mask_wp,
            wp.from_torch(voxel_count, dtype=wp.int32),
        ],
        stream=stream,
    )

    n_occupied = voxel_count.item()
    if n_occupied == 0:
        return OccupiedVoxels(
            centers=torch.empty((0, 3), dtype=torch.float32, device=device),
            block_idx_per_voxel=torch.empty((0,), dtype=torch.int32, device=device),
            block_data=view,
        )

    max_voxels = n_occupied + 1000
    out_centers = torch.zeros((max_voxels, 3), dtype=torch.float32, device=device)
    out_pool_idx = torch.zeros(max_voxels, dtype=torch.int32, device=device)
    out_count = torch.zeros(1, dtype=torch.int32, device=device)

    wp.launch(
        kernels.extract_occupied_voxels_masked_kernel,
        dim=(num_alloc, n_voxels_per_block),
        inputs=[
            warp_data,
            minimum_tsdf_weight,
            surface_only_int,
            sdf_threshold,
            mask_wp,
            wp.from_torch(out_centers, dtype=wp.float32),
            wp.from_torch(out_pool_idx, dtype=wp.int32),
            wp.from_torch(out_count, dtype=wp.int32),
            max_voxels,
        ],
        stream=stream,
    )

    n_extracted = min(out_count.item(), max_voxels)
    return OccupiedVoxels(
        centers=out_centers[:n_extracted],
        block_idx_per_voxel=out_pool_idx[:n_extracted],
        block_data=view,
    )
