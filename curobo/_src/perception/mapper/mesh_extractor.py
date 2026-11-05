# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse marching-cubes mesh extraction launcher.

The Warp kernels and helpers are built by
:func:`curobo._src.perception.mapper.kernel.builder.builder_mesh.make_mesh_kernels`
and are reached through ``tsdf.kernels`` at launch time. This module hosts the
single Python mesh extraction API for block-sparse TSDF storage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import PreparedSemanticFilterWarp
from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
    MCLookupTables,
)
from curobo._src.util.warp import get_warp_device_stream


def _empty_mesh(
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.uint8, device=device),
    )


def extract_mesh_block_sparse(
    tsdf,
    level: float = 0.0,
    surface_only: bool = False,
    refine_iterations: int = 0,
    minimum_tsdf_weight: float = 0.1,
    semantic_filter: Optional[PreparedSemanticFilterWarp] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract a triangle-soup mesh from a block-sparse TSDF.

    Each triangle owns its three vertices. Faces are identity indices and can
    be created by callers that need a face tensor.

    Args:
        tsdf: BlockSparseTSDF instance.
        level: Isosurface level, typically ``0.0``.
        surface_only: If True, only extract mesh near the surface.
        refine_iterations: Number of Newton-Raphson iterations for vertex
            refinement. ``0`` keeps linear edge interpolation.
        minimum_tsdf_weight: Minimum weight to consider a voxel observed.
        semantic_filter: Optional prepared feature-node filter.

    Returns:
        Tuple of ``(vertices, normals, colors)``.
    """
    device = tsdf.device
    mc_tables = MCLookupTables.get(device)
    warp_data = tsdf.get_warp_data()
    num_alloc = int(tsdf.data.num_allocated.item())
    kernels = tsdf.kernels

    if num_alloc == 0:
        return _empty_mesh(device)

    surface_band = tsdf.config.truncation_distance if surface_only else 0.0
    block_voxels = tsdf.block_size**3

    active_count = torch.zeros(1, dtype=torch.int32, device=device)
    active_block_idx = torch.empty(num_alloc, dtype=torch.int32, device=device)
    wp_device, stream = get_warp_device_stream(active_count)

    append_active_inputs = [
        warp_data,
        wp.from_torch(active_count, dtype=wp.int32),
        wp.from_torch(active_block_idx, dtype=wp.int32),
    ]
    append_active_kernel = kernels.append_active_blocks_kernel
    if semantic_filter is not None:
        append_active_kernel = kernels.append_active_blocks_semantic_kernel
        append_active_inputs.insert(1, semantic_filter)
    wp.launch(
        append_active_kernel,
        dim=num_alloc,
        inputs=append_active_inputs,
        device=wp_device,
        stream=stream,
        adjoint=False,
    )

    n_active = int(active_count.item())
    if n_active == 0:
        return _empty_mesh(device)

    active_block_idx = active_block_idx[:n_active]
    surface_count = torch.zeros(1, dtype=torch.int32, device=device)

    count_surface_inputs = [
        warp_data,
        wp.from_torch(active_block_idx, dtype=wp.int32),
        n_active,
        level,
        surface_band,
        minimum_tsdf_weight,
        wp.from_torch(surface_count, dtype=wp.int32),
    ]
    count_surface_kernel = kernels.count_surface_cubes_from_blocks_kernel
    if semantic_filter is not None:
        count_surface_kernel = kernels.count_surface_cubes_from_blocks_semantic_kernel
        count_surface_inputs.insert(1, semantic_filter)
    wp.launch(
        count_surface_kernel,
        dim=(n_active, block_voxels),
        inputs=count_surface_inputs,
        device=wp_device,
        stream=stream,
        adjoint=False,
    )

    n_surfaces = int(surface_count.item())
    if n_surfaces == 0:
        return _empty_mesh(device)

    surface_block_idx = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    surface_cube_idx = torch.zeros(n_surfaces, dtype=torch.int32, device=device)

    surface_count.zero_()

    append_surface_inputs = [
        warp_data,
        wp.from_torch(active_block_idx, dtype=wp.int32),
        n_active,
        level,
        surface_band,
        minimum_tsdf_weight,
        wp.from_torch(surface_count, dtype=wp.int32),
        wp.from_torch(surface_block_idx, dtype=wp.int32),
        wp.from_torch(surface_cube_idx, dtype=wp.int32),
    ]
    append_surface_kernel = kernels.append_surface_cubes_from_blocks_kernel
    if semantic_filter is not None:
        append_surface_kernel = kernels.append_surface_cubes_from_blocks_semantic_kernel
        append_surface_inputs.insert(1, semantic_filter)
    wp.launch(
        append_surface_kernel,
        dim=(n_active, block_voxels),
        inputs=append_surface_inputs,
        device=wp_device,
        stream=stream,
    )

    triangle_count = torch.zeros(1, dtype=torch.int32, device=device)
    wp.launch(
        kernels.count_total_triangles_kernel,
        dim=n_surfaces,
        inputs=[
            warp_data,
            level,
            minimum_tsdf_weight,
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
            n_surfaces,
            mc_tables.num_tris_table,
            wp.from_torch(triangle_count, dtype=wp.int32),
        ],
        device=wp_device,
        stream=stream,
    )

    total_tris = int(triangle_count.item())
    if total_tris == 0:
        return _empty_mesh(device)

    total_vertices = total_tris * 3
    vertices = torch.zeros((total_vertices, 3), dtype=torch.float32, device=device)
    normals = torch.zeros((total_vertices, 3), dtype=torch.float32, device=device)
    colors = torch.zeros((total_vertices, 3), dtype=torch.uint8, device=device)

    triangle_count.zero_()
    wp.launch(
        kernels.generate_mesh_kernel,
        dim=n_surfaces,
        inputs=[
            warp_data,
            level,
            minimum_tsdf_weight,
            refine_iterations,
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
            n_surfaces,
            mc_tables.tri_table,
            wp.from_torch(vertices, dtype=wp.vec3),
            wp.from_torch(normals, dtype=wp.vec3),
            wp.from_torch(colors, dtype=wp.vec3ub),
            wp.from_torch(triangle_count, dtype=wp.int32),
            total_tris,
        ],
        device=wp_device,
        stream=stream,
    )

    return vertices, normals, colors
