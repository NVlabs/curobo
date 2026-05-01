# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""ESDF seeding launcher wrappers.

The seeding Warp kernels (scatter + gather variants) live in
:func:`curobo._src.perception.mapper.kernel.builder.builder_esdf.make_esdf_kernels`,
which builds both variants. The ESDF integrator chooses which launcher
to call from ``cfg.seeding_method``. This module only hosts the Python
launchers used by
:class:`BlockSparseESDFIntegrator`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.storage import BlockSparseTSDF
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise


def _validate_esdf_launch_shapes(
    tsdf: BlockSparseTSDF,
    esdf_grid_shape: Tuple[int, int, int],
) -> None:
    esdf_shape = tuple(int(v) for v in esdf_grid_shape)
    if esdf_shape != tsdf.kernels.esdf_grid_shape:
        log_and_raise(
            f"esdf_grid_shape={esdf_shape} does not match compiled "
            f"kernel esdf_grid_shape={tsdf.kernels.esdf_grid_shape}."
        )
    grid_shape = tsdf.config.grid_shape
    grid_shape = tuple(int(v) for v in grid_shape)
    if grid_shape != tsdf.kernels.grid_shape:
        log_and_raise(
            f"grid_shape={grid_shape} does not match compiled "
            f"kernel grid_shape={tsdf.kernels.grid_shape}."
        )


def seed_esdf_sites_from_block_sparse_warp(
    tsdf: BlockSparseTSDF,
    esdf_site_index: torch.Tensor,
    esdf_origin: torch.Tensor,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    minimum_tsdf_weight: float = 0.1,
) -> None:
    """Seed ESDF sites from block-sparse TSDF (surface + truncation boundary).

    Uses the ``scatter`` specialization: variable launch dim,
    NOT CUDA-graph safe.
    """
    esdf_site_index.fill_(-1)
    warp_data = tsdf.get_warp_data()
    data = tsdf.data

    _validate_esdf_launch_shapes(tsdf, esdf_grid_shape)

    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    num_allocated = int(data.num_allocated.item())
    block_voxels = tsdf.block_size**3
    if num_allocated == 0:
        return

    kernels = tsdf.kernels

    wp.launch(
        kernel=kernels.seed_esdf_sites_from_block_sparse_kernel,
        dim=(num_allocated, block_voxels),
        inputs=[
            warp_data,
            wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32),
            wp.from_torch(esdf_origin, dtype=wp.float32),
            wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32),
            wp.float32(minimum_tsdf_weight),
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
) -> None:
    """Gather-based ESDF site seeding. CUDA-graph safe (fixed launch dim).

    Uses the ``gather`` specialization.
    """
    esdf_site_index.fill_(-1)

    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    n_esdf_voxels = esdf_D * esdf_H * esdf_W

    warp_data = tsdf.get_warp_data()

    _validate_esdf_launch_shapes(tsdf, esdf_grid_shape)

    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    # Launches against the gather seed kernel in the owning bundle.
    kernels = tsdf.kernels

    wp.launch(
        kernel=kernels.seed_esdf_sites_gather_kernel,
        dim=n_esdf_voxels,
        inputs=[
            warp_data,
            wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32),
            wp.from_torch(esdf_origin, dtype=wp.float32),
            wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32),
            wp.float32(minimum_tsdf_weight),
        ],
        device=wp.device_from_torch(device),
        stream=stream,
        block_dim=256,
    )
