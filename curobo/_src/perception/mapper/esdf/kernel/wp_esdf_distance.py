# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""ESDF distance launcher wrappers.

The ESDF distance Warp kernels are built by
:func:`curobo._src.perception.mapper.kernel.builder.builder_esdf.make_esdf_kernels`
and are reached through ``tsdf.kernels`` at launch time. This module only hosts
the Python launchers used by :class:`BlockSparseESDFIntegrator`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.storage import BlockSparseTSDF
from curobo._src.util.warp import get_warp_device_stream
from curobo.logging import log_and_raise


def compute_esdf_from_min_tsdf_warp(
    esdf_site_index: torch.Tensor,
    tsdf: BlockSparseTSDF,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    esdf_dist_field: torch.Tensor,
    esdf_origin: torch.Tensor,
    adjacent_skip_steps: float = 0.0,
    minimum_tsdf_weight: float = 0.1,
) -> None:
    """Compute ESDF distances with sign from direct TSDF hash lookup."""
    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    n_esdf_voxels = esdf_D * esdf_H * esdf_W
    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    warp_data = tsdf.get_warp_data()

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

    kernels = tsdf.kernels

    wp.launch(
        kernel=kernels.compute_esdf_from_min_tsdf_kernel,
        dim=n_esdf_voxels,
        inputs=[
            wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32),
            wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32),
            wp.from_torch(esdf_dist_field.view(-1), dtype=wp.float16),
            warp_data,
            wp.float32(minimum_tsdf_weight),
            wp.from_torch(esdf_origin, dtype=wp.float32),
            wp.float32(adjacent_skip_steps),
        ],
        device=wp.device_from_torch(device),
        stream=stream,
    )
