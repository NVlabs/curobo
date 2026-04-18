# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared triangle filtering utilities for marching cubes implementations.

This module provides memory-efficient triangle filtering that avoids large
intermediate allocations by processing triangles in a single kernel pass.
"""


import torch
import warp as wp

from curobo._src.curobolib.cuda_ops.tensor_checks import check_int32_tensors
from curobo._src.util.warp import get_warp_device_stream


@wp.kernel
def count_valid_triangles_kernel(
    triangles: wp.array(dtype=wp.int32),  # [n_triangles * 3] flat
    vertices: wp.array(dtype=wp.vec3),
    n_triangles: wp.int32,
    min_area_sq: wp.float32,
    valid_count: wp.array(dtype=wp.int32),
):
    """Count valid triangles (first pass for exact allocation).

    A triangle is valid if:
    - All vertex indices >= 0 (not missing)
    - All vertex indices are different (not degenerate indices)
    - Triangle has non-zero area (vertices not at same position)
    """
    tid = wp.tid()
    if tid >= n_triangles:
        return

    base = tid * 3
    v0_idx = triangles[base + 0]
    v1_idx = triangles[base + 1]
    v2_idx = triangles[base + 2]

    # Check for valid indices (no -1)
    if v0_idx < 0 or v1_idx < 0 or v2_idx < 0:
        return

    # Check for non-degenerate indices
    if v0_idx == v1_idx or v1_idx == v2_idx or v0_idx == v2_idx:
        return

    # Check for non-zero area
    p0 = vertices[v0_idx]
    p1 = vertices[v1_idx]
    p2 = vertices[v2_idx]

    edge1 = p1 - p0
    edge2 = p2 - p0
    cross = wp.cross(edge1, edge2)
    area_sq = wp.dot(cross, cross)

    if area_sq <= min_area_sq:
        return

    wp.atomic_add(valid_count, 0, 1)


@wp.kernel
def compact_valid_triangles_kernel(
    triangles_in: wp.array(dtype=wp.int32),  # [n_triangles * 3] flat
    vertices: wp.array(dtype=wp.vec3),
    n_triangles: wp.int32,
    min_area_sq: wp.float32,
    flip_winding: wp.bool,
    output_counter: wp.array(dtype=wp.int32),
    triangles_out: wp.array(dtype=wp.int32),  # [n_valid * 3] flat
):
    """Compact valid triangles to output (second pass).

    Optionally flips winding order (swaps v1 and v2) for correct normals.
    """
    tid = wp.tid()
    if tid >= n_triangles:
        return

    base = tid * 3
    v0_idx = triangles_in[base + 0]
    v1_idx = triangles_in[base + 1]
    v2_idx = triangles_in[base + 2]

    # Check for valid indices (no -1)
    if v0_idx < 0 or v1_idx < 0 or v2_idx < 0:
        return

    # Check for non-degenerate indices
    if v0_idx == v1_idx or v1_idx == v2_idx or v0_idx == v2_idx:
        return

    # Check for non-zero area
    p0 = vertices[v0_idx]
    p1 = vertices[v1_idx]
    p2 = vertices[v2_idx]

    edge1 = p1 - p0
    edge2 = p2 - p0
    cross = wp.cross(edge1, edge2)
    area_sq = wp.dot(cross, cross)

    if area_sq <= min_area_sq:
        return

    # Atomically get output slot
    out_idx = wp.atomic_add(output_counter, 0, 1)
    out_base = out_idx * 3

    # Write with optional winding flip
    if flip_winding:
        triangles_out[out_base + 0] = v0_idx
        triangles_out[out_base + 1] = v2_idx  # swapped
        triangles_out[out_base + 2] = v1_idx  # swapped
    else:
        triangles_out[out_base + 0] = v0_idx
        triangles_out[out_base + 1] = v1_idx
        triangles_out[out_base + 2] = v2_idx


def filter_triangles(
    triangles: torch.Tensor,
    vertices: torch.Tensor,
    voxel_size: float,
    flip_winding: bool = True,
) -> torch.Tensor:
    """Filter invalid and degenerate triangles efficiently.

    Uses a two-pass kernel approach to avoid large intermediate allocations:
    1. Count valid triangles
    2. Allocate exact-sized output and compact

    This replaces the Python-based filtering that required ~1.3GB for 17M triangles.

    Args:
        triangles: (N, 3) int32 tensor of vertex indices.
        vertices: (M, 3) float32 tensor of vertex positions.
        voxel_size: Voxel size for area threshold calculation.
        flip_winding: If True, swap v1 and v2 to flip triangle winding order.

    Returns:
        Filtered triangles tensor (K, 3) where K <= N.
    """
    device = triangles.device
    n_triangles = triangles.shape[0]

    if n_triangles == 0:
        return triangles

    # Minimum area threshold
    min_area_sq = (voxel_size * 1e-6) ** 2

    # Get stream from torch for proper synchronization
    _, stream = get_warp_device_stream(triangles)

    triangles_flat = triangles.view(-1)
    check_int32_tensors(triangles_flat.device, triangles_flat=triangles_flat)
    vertices_wp = wp.from_torch(vertices, dtype=wp.vec3)
    triangles_wp = wp.from_torch(triangles_flat, dtype=wp.int32)

    # Pass 1: Count valid triangles
    valid_count = torch.zeros(1, dtype=torch.int32, device=device)
    valid_count_wp = wp.from_torch(valid_count, dtype=wp.int32)

    wp.launch(
        count_valid_triangles_kernel,
        dim=n_triangles,
        inputs=[
            triangles_wp,
            vertices_wp,
            n_triangles,
            min_area_sq,
            valid_count_wp,
        ],
        stream=stream,
    )

    n_valid = int(valid_count.item())
    if n_valid == 0:
        return torch.empty((0, 3), dtype=torch.int32, device=device)

    # Pass 2: Compact valid triangles
    triangles_out = torch.zeros((n_valid, 3), dtype=torch.int32, device=device)
    triangles_out_wp = wp.from_torch(triangles_out.view(-1), dtype=wp.int32)

    # Reset counter for second pass
    valid_count.zero_()
    valid_count_wp = wp.from_torch(valid_count, dtype=wp.int32)

    wp.launch(
        compact_valid_triangles_kernel,
        dim=n_triangles,
        inputs=[
            triangles_wp,
            vertices_wp,
            n_triangles,
            min_area_sq,
            flip_winding,
            valid_count_wp,
            triangles_out_wp,
        ],
        stream=stream,
    )

    return triangles_out

