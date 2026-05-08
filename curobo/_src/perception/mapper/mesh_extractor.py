# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse marching-cubes mesh extraction launcher.

The Warp kernels and helpers are built by
:func:`curobo._src.perception.mapper.kernel.builder.builder_mesh.make_mesh_kernels`
and are reached through ``tsdf.kernels`` at launch time. This module
only hosts the public :func:`extract_mesh_block_sparse` Python API.
"""

from __future__ import annotations

from typing import Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
    MCLookupTables,
)
from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_filter import (
    filter_triangles,
)
from curobo._src.util.warp import get_warp_device_stream


def extract_mesh_block_sparse(
    tsdf,
    level: float = 0.0,
    surface_only: bool = False,
    refine_iterations: int = 0,
    minimum_tsdf_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract mesh from block-sparse TSDF using marching cubes with edge sharing.

    This version uses shared vertices to reduce vertex count by ~3x
    compared to the non-shared version.

    Args:
        tsdf: BlockSparseTSDF instance.
        level: Isosurface level (typically 0.0).
        surface_only: If True, only extract mesh near the surface.
        refine_iterations: Number of Newton-Raphson iterations for vertex
            refinement. 0 = no refinement.
        minimum_tsdf_weight: Minimum weight to consider a voxel as observed.

    Returns:
        Tuple of (vertices, triangles, normals, colors).
    """
    device = tsdf.device
    mc_tables = MCLookupTables.get(device)
    warp_data = tsdf.get_warp_data()
    num_alloc = tsdf.data.num_allocated.item()
    kernels = tsdf.kernels

    empty_result = (
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.int32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.uint8, device=device),
    )

    if num_alloc == 0:
        return empty_result

    surface_band = tsdf.config.truncation_distance if surface_only else 0.0

    block_size = tsdf.block_size
    block_voxels = block_size**3

    # Step 1a: Count surface cubes.
    surface_count = torch.zeros(1, dtype=torch.int32, device=device)
    wp_device, stream = get_warp_device_stream(surface_count)

    wp.launch(
        kernels.count_surface_cubes_kernel,
        dim=(num_alloc, block_voxels),
        inputs=[
            warp_data,
            level,
            surface_band,
            minimum_tsdf_weight,
            wp.from_torch(surface_count, dtype=wp.int32),
        ],
        device=wp_device,
        stream=stream,
        adjoint=False,
    )

    n_surfaces = int(surface_count.item())
    if n_surfaces == 0:
        return empty_result

    # Step 1b: Allocate + append.
    surface_block_idx = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    surface_cube_idx = torch.zeros(n_surfaces, dtype=torch.int32, device=device)

    surface_count.zero_()

    wp.launch(
        kernels.append_surface_cubes_kernel,
        dim=(num_alloc, block_voxels),
        inputs=[
            warp_data,
            level,
            surface_band,
            minimum_tsdf_weight,
            wp.from_torch(surface_count, dtype=wp.int32),
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
        ],
        device=wp_device,
        stream=stream,
    )

    # Step 2: Count owned edges.
    edge_counts = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    wp.launch(
        kernels.count_edges_block_sparse_kernel,
        dim=n_surfaces,
        inputs=[
            warp_data,
            level,
            minimum_tsdf_weight,
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
            n_surfaces,
            wp.from_torch(edge_counts, dtype=wp.int32),
        ],
        device=wp_device,
        stream=stream,
    )

    # Step 3: Prefix sum → vertex offsets.
    edge_offsets = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    if n_surfaces > 0:
        edge_offsets[1:] = torch.cumsum(edge_counts[:-1], dim=0)
    total_vertices = int(edge_counts.sum().item())

    if total_vertices == 0:
        return empty_result

    # Step 4: Generate vertices for owned edges.
    vertices = torch.zeros((total_vertices, 3), dtype=torch.float32, device=device)
    normals = torch.zeros((total_vertices, 3), dtype=torch.float32, device=device)
    edge_vertex_indices = torch.full((n_surfaces * 3,), -1, dtype=torch.int32, device=device)

    wp.launch(
        kernels.generate_vertices_block_sparse_kernel,
        dim=n_surfaces,
        inputs=[
            warp_data,
            level,
            minimum_tsdf_weight,
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
            wp.from_torch(edge_offsets, dtype=wp.int32),
            n_surfaces,
            refine_iterations,
            wp.from_torch(vertices, dtype=wp.vec3),
            wp.from_torch(normals, dtype=wp.vec3),
            wp.from_torch(edge_vertex_indices, dtype=wp.int32),
        ],
        device=wp_device,
        stream=stream,
    )

    # Step 5: Sort surface cubes for binary-search lookup.
    global_ids = surface_block_idx.to(torch.int64) * int(block_voxels) + surface_cube_idx.to(
        torch.int64
    )
    sorted_global_ids, sort_order = torch.sort(global_ids)
    sparse_indices_sorted = sort_order.to(torch.int32)

    # Step 6: Count triangles per surface cube.
    tri_counts = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    wp.launch(
        kernels.count_triangles_kernel,
        dim=n_surfaces,
        inputs=[
            warp_data,
            level,
            minimum_tsdf_weight,
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
            n_surfaces,
            mc_tables.num_tris_table,
            wp.from_torch(tri_counts, dtype=wp.int32),
        ],
        device=wp_device,
        stream=stream,
    )

    # Step 7: Prefix sum → triangle offsets.
    tri_offsets = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    if n_surfaces > 0:
        tri_offsets[1:] = torch.cumsum(tri_counts[:-1], dim=0)
    total_tris = int(tri_counts.sum().item())

    if total_tris == 0:
        return empty_result

    # Step 8: Generate triangles with shared-vertex lookup. Allocate as
    # a flat tensor because the kernel indexes it as ``triangles[i * 3 + j]``;
    # return it reshaped to (M, 3) for the caller.
    triangles_flat = torch.full(
        (total_tris * 3,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    wp.launch(
        kernels.generate_triangles_shared_kernel,
        dim=n_surfaces,
        inputs=[
            warp_data,
            level,
            minimum_tsdf_weight,
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
            wp.from_torch(tri_offsets, dtype=wp.int32),
            n_surfaces,
            mc_tables.tri_table,
            mc_tables.edge_owner,
            wp.from_torch(sorted_global_ids, dtype=wp.int64),
            wp.from_torch(sparse_indices_sorted, dtype=wp.int32),
            wp.from_torch(edge_vertex_indices, dtype=wp.int32),
            wp.from_torch(triangles_flat, dtype=wp.int32),
        ],
        device=wp_device,
        stream=stream,
    )

    triangles = triangles_flat.view(total_tris, 3)

    # Step 9: Filter out invalid (unresolved -1 indices) and degenerate
    # triangles. Triangles on the boundary of the observed region can
    # reference edges owned by cubes that are not in the surface list
    # (owner block not allocated); those entries stay as -1 and are
    # compacted out here. Also flips winding so normals point outward.
    triangles = filter_triangles(
        triangles,
        vertices,
        tsdf.config.voxel_size,
        flip_winding=True,
    )

    # Step 10: Sample vertex colors.
    colors = torch.zeros(
        (total_vertices, 3),
        dtype=torch.uint8,
        device=device,
    )
    wp.launch(
        kernels.sample_vertex_colors_kernel,
        dim=total_vertices,
        inputs=[
            wp.from_torch(vertices, dtype=wp.vec3),
            total_vertices,
            warp_data,
            wp.from_torch(colors, dtype=wp.vec3ub),
        ],
        device=wp_device,
        stream=stream,
    )

    return vertices, triangles, normals, colors
