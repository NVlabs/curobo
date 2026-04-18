# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse Marching Cubes for mesh extraction.

This module provides GPU kernels for extracting meshes from block-sparse TSDF.
The implementation handles block boundaries by looking up neighbor blocks
via the hash table.

Key differences from dense marching cubes:
- Iterates over allocated blocks (not all voxels)
- Uses hash table lookup for neighbor block access
- Handles missing neighbors gracefully (unobserved = skip)
"""

from typing import Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_hash import (
    compute_avg_rgb_uint8_from_block,
    hash_lookup,
)
from curobo._src.perception.mapper.kernel.wp_raycast_common import (
    compute_gradient,
    compute_gradient_nearest,
    sample_tsdf_trilinear,
    sample_voxel,
)
from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
    MCLookupTables,
    binary_search_int32,
    interpolate_edge_vertex,
    local_edge_to_array_idx,
)
from curobo._src.util.warp import get_warp_device_stream

# =============================================================================
# Vertex Refinement for Mesh Extraction
# =============================================================================


@wp.func
def refine_vertex_mesh(
    tsdf: BlockSparseTSDFWarp,
    vertex: wp.vec3,
    level: wp.float32,
    iterations: wp.int32,
    minimum_tsdf_weight: wp.float32,
) -> wp.vec3:
    """Refine vertex position to true SDF zero-crossing using Newton-Raphson.

    Args:
        tsdf: BlockSparseTSDFWarp struct.
        vertex: Initial vertex position.
        level: Isosurface level.
        iterations: Number of refinement iterations.
        minimum_tsdf_weight: Minimum weight for valid voxel.

    Returns:
        Refined vertex position.
    """
    pos = vertex

    for _ in range(iterations):
        result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
        if result[1] < 0.5:
            break
        sdf_val = result[0] - level

        if wp.abs(sdf_val) < 1e-6 or sdf_val > 100.0:
            break

        # Use trilinear gradient for refinement (smooth convergence)
        grad = compute_gradient(tsdf, pos, minimum_tsdf_weight)
        grad_mag = wp.sqrt(wp.dot(grad, grad))

        if grad_mag < 1e-4:
            break

        step_size = wp.clamp(sdf_val / grad_mag, -tsdf.voxel_size * 0.5, tsdf.voxel_size * 0.5)
        pos = pos - step_size * (grad / grad_mag)

    return pos


# =============================================================================
# SDF Access Functions
# =============================================================================


@wp.func
def get_block_sdf(
    tsdf: BlockSparseTSDFWarp,
    pool_idx: wp.int32,
    lx: wp.int32,
    ly: wp.int32,
    lz: wp.int32,
    level: float,
    minimum_tsdf_weight: float,
) -> wp.vec2:
    """Get combined SDF value from a block (dynamic + static).

    Args:
        tsdf: BlockSparseTSDFWarp struct.
        pool_idx: Block pool index.
        lx, ly, lz: Local voxel coordinates within block.
        level: Isosurface level to subtract.
        minimum_tsdf_weight: Minimum weight for valid voxel.

    Returns:
        vec2 with (sdf - level, valid). If unobserved, sdf = 1e10.
    """
    local_idx = lz * 64 + ly * 8 + lx
    result = sample_voxel(tsdf, pool_idx, local_idx, minimum_tsdf_weight)

    if result[1] < 0.5:
        return wp.vec2(1e10, 0.0)

    return wp.vec2(result[0] - level, 1.0)


@wp.func
def sample_cube_corner(
    cx: wp.int32,
    cy: wp.int32,
    cz: wp.int32,  # Corner position within block (0-8)
    bx: wp.int32,
    by: wp.int32,
    bz: wp.int32,  # Block world coords
    pool_idx: wp.int32,  # This block's pool index
    tsdf: BlockSparseTSDFWarp,
    level: float,
    minimum_tsdf_weight: float,
) -> wp.vec2:
    """Sample combined SDF at cube corner, handling block boundary crossing.

    For corners within this block [0,7], read directly.
    For corners at block edge [8 in any dim], look up neighbor block.

    Returns:
        vec2 with (sdf - level, valid).
    """
    # Check if within this block
    if cx < 8 and cy < 8 and cz < 8:
        return get_block_sdf(tsdf, pool_idx, cx, cy, cz, level, minimum_tsdf_weight)

    # Need neighbor block
    nbx = bx
    nby = by
    nbz = bz
    nlx = cx
    nly = cy
    nlz = cz

    if cx >= 8:
        nbx = bx + 1
        nlx = 0
    if cy >= 8:
        nby = by + 1
        nly = 0
    if cz >= 8:
        nbz = bz + 1
        nlz = 0

    # Look up neighbor block
    neighbor_idx = hash_lookup(tsdf.hash_table, nbx, nby, nbz, tsdf.hash_capacity)

    if neighbor_idx < 0:
        return wp.vec2(1e10, 0.0)

    return get_block_sdf(tsdf, neighbor_idx, nlx, nly, nlz, level, minimum_tsdf_weight)


# Lookup tables are now managed by MCLookupTables in wp_mc_common


# =============================================================================
# Surface Detection Kernels
# =============================================================================


@wp.func
def is_surface_cube_combined(
    cx: wp.int32,
    cy: wp.int32,
    cz: wp.int32,
    bx: wp.int32,
    by: wp.int32,
    bz: wp.int32,
    block_idx: wp.int32,
    tsdf: BlockSparseTSDFWarp,
    level: float,
    surface_band: float,
    minimum_tsdf_weight: float,
) -> wp.bool:
    """Check if a cube contains a surface (sign change across corners).

    Uses combined (dynamic + static) SDF sampling.
    Returns True if the cube should be included in marching cubes processing.
    """
    # Sample 8 cube corners
    s0 = sample_cube_corner(cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s1 = sample_cube_corner(cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s2 = sample_cube_corner(cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s3 = sample_cube_corner(cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s4 = sample_cube_corner(cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s5 = sample_cube_corner(cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s6 = sample_cube_corner(cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s7 = sample_cube_corner(cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)

    # Skip if any corner is unobserved (valid < 0.5)
    if s0[1] < 0.5 or s1[1] < 0.5 or s2[1] < 0.5 or s3[1] < 0.5:
        return False
    if s4[1] < 0.5 or s5[1] < 0.5 or s6[1] < 0.5 or s7[1] < 0.5:
        return False

    # Check for sign change (surface)
    has_positive = (
        s0[0] > 0.0 or s1[0] > 0.0 or s2[0] > 0.0 or s3[0] > 0.0 or
        s4[0] > 0.0 or s5[0] > 0.0 or s6[0] > 0.0 or s7[0] > 0.0
    )
    has_negative = (
        s0[0] < 0.0 or s1[0] < 0.0 or s2[0] < 0.0 or s3[0] < 0.0 or
        s4[0] < 0.0 or s5[0] < 0.0 or s6[0] < 0.0 or s7[0] < 0.0
    )

    if not (has_positive and has_negative):
        return False

    # Surface band filtering: skip cubes where no corner is within the band
    if surface_band > 0.0:
        in_band = (
            wp.abs(s0[0]) < surface_band or wp.abs(s1[0]) < surface_band or
            wp.abs(s2[0]) < surface_band or wp.abs(s3[0]) < surface_band or
            wp.abs(s4[0]) < surface_band or wp.abs(s5[0]) < surface_band or
            wp.abs(s6[0]) < surface_band or wp.abs(s7[0]) < surface_band
        )
        if not in_band:
            return False

    return True


@wp.kernel(enable_backward=False)
def count_surface_cubes_kernel(
    tsdf: BlockSparseTSDFWarp,
    level: float,
    surface_band: float,
    minimum_tsdf_weight: float,
    # Output
    surface_count: wp.array(dtype=wp.int32),
):
    """Count cubes that contain a surface (first pass - count only).

    Each block owns 8×8×8 = 512 cubes. Each thread processes one cube.
    Launch with dim = num_allocated * 512.
    """
    tid = wp.tid()
    block_idx = tid // 512
    cube_idx = tid % 512

    if block_idx >= tsdf.num_allocated[0]:
        return


    # Skip freed blocks
    if tsdf.block_to_hash_slot[block_idx] < 0:
        return

    # Cube local coords (0-7 in each dimension)
    cx = cube_idx % 8
    cy = (cube_idx // 8) % 8
    cz = cube_idx // 64

    # Block world coords
    bx = tsdf.block_coords[block_idx * 3 + 0]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    is_surface =  is_surface_cube_combined(cx, cy, cz, bx, by, bz, block_idx, tsdf, level, surface_band, minimum_tsdf_weight)
    if is_surface:
         wp.atomic_add(surface_count, 0, wp.int32(1))


@wp.kernel(enable_backward=False)
def append_surface_cubes_kernel(
    tsdf: BlockSparseTSDFWarp,
    level: float,
    surface_band: float,
    minimum_tsdf_weight: float,
    # Output
    surface_count: wp.array(dtype=wp.int32),
    surface_block_idx: wp.array(dtype=wp.int32),
    surface_cube_idx: wp.array(dtype=wp.int32),
):
    """Append surface cubes to output arrays (second pass - store).

    Launch with dim = num_allocated * 512.
    """
    tid = wp.tid()
    block_idx = tid // 512
    cube_idx = tid % 512

    if block_idx >= tsdf.num_allocated[0]:
        return

    # Skip freed blocks
    if tsdf.block_to_hash_slot[block_idx] < 0:
        return

    # Cube local coords (0-7 in each dimension)
    cx = cube_idx % 8
    cy = (cube_idx // 8) % 8
    cz = cube_idx // 64

    # Block world coords
    bx = tsdf.block_coords[block_idx * 3 + 0]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    if is_surface_cube_combined(cx, cy, cz, bx, by, bz, block_idx, tsdf, level, surface_band, minimum_tsdf_weight):
        out_idx = wp.atomic_add(surface_count, 0, wp.int32(1))
        surface_block_idx[out_idx] = block_idx
        surface_cube_idx[out_idx] = cube_idx


# =============================================================================
# Edge Sharing Kernels
# =============================================================================


@wp.func
def get_edge_owner_combined(
    edge: wp.int32,
    block_idx: wp.int32,
    cx: wp.int32,
    cy: wp.int32,
    cz: wp.int32,
    bx: wp.int32,
    by: wp.int32,
    bz: wp.int32,
    tsdf: BlockSparseTSDFWarp,
    edge_owner: wp.array(dtype=wp.int32),
) -> wp.vec3i:
    """Find which cube owns an edge, handling block boundaries.

    Args:
        edge: Edge index (0-11).
        block_idx: Current block's pool index.
        cx, cy, cz: Local cube coords (0-7) within current block.
            Note: (cx, cy, cz) = (x, y, z) in world coordinates.
        bx, by, bz: Block world coordinates.
        hash_table: Block hash table.
        capacity: Hash table capacity.
        edge_owner: Edge ownership lookup table in (di, dj, dk, local_edge) format.
            Note: (di, dj, dk) = (dz, dy, dx) since grid uses (i,j,k) = (z,y,x).

    Returns:
        vec3i with (owner_block_idx, owner_cube_idx, local_edge).
        If owner block not found, owner_block_idx = -1.
    """
    # Get edge owner offset from table
    # Table is in (di, dj, dk) = (dz, dy, dx) format, so swap to (dx, dy, dz)
    owner_dx = edge_owner[edge * 4 + 2]  # dk = dx
    owner_dy = edge_owner[edge * 4 + 1]  # dj = dy
    owner_dz = edge_owner[edge * 4 + 0]  # di = dz
    local_edge = edge_owner[edge * 4 + 3]

    # Compute owner cube position (in global voxel space)
    new_cx = cx + owner_dx
    new_cy = cy + owner_dy
    new_cz = cz + owner_dz

    # Check if within same block
    if new_cx < 8 and new_cy < 8 and new_cz < 8:
        owner_cube = new_cz * 64 + new_cy * 8 + new_cx
        return wp.vec3i(block_idx, owner_cube, local_edge)

    # Need neighbor block lookup
    new_bx = bx
    new_by = by
    new_bz = bz

    if new_cx >= 8:
        new_bx = bx + 1
        new_cx = 0
    if new_cy >= 8:
        new_by = by + 1
        new_cy = 0
    if new_cz >= 8:
        new_bz = bz + 1
        new_cz = 0

    neighbor_idx = hash_lookup(tsdf.hash_table, new_bx, new_by, new_bz, tsdf.hash_capacity)
    if neighbor_idx < 0:
        return wp.vec3i(-1, -1, local_edge)  # Neighbor block not allocated

    owner_cube = new_cz * 64 + new_cy * 8 + new_cx
    return wp.vec3i(neighbor_idx, owner_cube, local_edge)


@wp.kernel(enable_backward=False)
def count_edges_block_sparse_kernel(
    tsdf: BlockSparseTSDFWarp,
    level: float,
    minimum_tsdf_weight: float,
    surface_block_idx: wp.array(dtype=wp.int32),
    surface_cube_idx: wp.array(dtype=wp.int32),
    n_surfaces: wp.int32,
    # Output
    edge_counts: wp.array(dtype=wp.int32),
):
    """Count owned edges (0, 3, 8) for each surface cube.

    Each cube owns 3 edges that meet at corner 0:
    - Edge 0: along +x (corners 0-1)
    - Edge 3: along +y (corners 0-3)
    - Edge 8: along +z (corners 0-4)
    """
    tid = wp.tid()
    if tid >= n_surfaces:
        return

    block_idx = surface_block_idx[tid]
    cube_idx = surface_cube_idx[tid]

    cx = cube_idx % 8
    cy = (cube_idx // 8) % 8
    cz = cube_idx // 64

    bx = tsdf.block_coords[block_idx * 3 + 0]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    # Get SDF at corners 0, 1, 3, 4 (the corners for owned edges)
    s0 = sample_cube_corner(cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s1 = sample_cube_corner(cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s3 = sample_cube_corner(cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s4 = sample_cube_corner(cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)

    count = wp.int32(0)

    # Edge 0: corners 0-1 (along +x)
    if s0[0] * s1[0] < 0.0:
        count += wp.int32(1)

    # Edge 3: corners 0-3 (along +y)
    if s0[0] * s3[0] < 0.0:
        count += wp.int32(1)

    # Edge 8: corners 0-4 (along +z)
    if s0[0] * s4[0] < 0.0:
        count += wp.int32(1)

    edge_counts[tid] = count


@wp.kernel(enable_backward=False)
def generate_vertices_block_sparse_kernel(
    tsdf: BlockSparseTSDFWarp,
    level: float,
    minimum_tsdf_weight: float,
    surface_block_idx: wp.array(dtype=wp.int32),
    surface_cube_idx: wp.array(dtype=wp.int32),
    edge_offsets: wp.array(dtype=wp.int32),
    n_surfaces: wp.int32,
    refine_iterations: wp.int32,
    # Output
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    edge_vertex_indices: wp.array(dtype=wp.int32),  # [n_surfaces * 3]
):
    """Generate vertices and normals for owned edges only.

    Each surface cube owns up to 3 edges (0, 3, 8).
    Stores mapping from (sparse_idx, edge_array_idx) → vertex_id.
    Uses combined (dynamic + static) SDF sampling.

    Args:
        refine_iterations: Number of Newton-Raphson iterations for vertex refinement.
            0 = no refinement (linear interpolation only).
    """
    tid = wp.tid()
    if tid >= n_surfaces:
        return

    block_idx = surface_block_idx[tid]
    cube_idx = surface_cube_idx[tid]

    cx = cube_idx % 8
    cy = (cube_idx // 8) % 8
    cz = cube_idx // 64

    bx = tsdf.block_coords[block_idx * 3 + 0]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    # Get SDF at corners
    s0 = sample_cube_corner(cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s1 = sample_cube_corner(cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s3 = sample_cube_corner(cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s4 = sample_cube_corner(cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)

    # Global voxel position
    gx = bx * tsdf.block_size + cx
    gy = by * tsdf.block_size + cy
    gz = bz * tsdf.block_size + cz

    # Center offset for center-origin convention
    center_offset_x = wp.float32(tsdf.grid_W) * 0.5
    center_offset_y = wp.float32(tsdf.grid_H) * 0.5
    center_offset_z = wp.float32(tsdf.grid_D) * 0.5

    # Corner world positions
    p0 = tsdf.origin + wp.vec3(
        wp.float32(gx) - center_offset_x,
        wp.float32(gy) - center_offset_y,
        wp.float32(gz) - center_offset_z
    ) * tsdf.voxel_size
    p1 = tsdf.origin + wp.vec3(
        wp.float32(gx + 1) - center_offset_x,
        wp.float32(gy) - center_offset_y,
        wp.float32(gz) - center_offset_z
    ) * tsdf.voxel_size
    p3 = tsdf.origin + wp.vec3(
        wp.float32(gx) - center_offset_x,
        wp.float32(gy + 1) - center_offset_y,
        wp.float32(gz) - center_offset_z
    ) * tsdf.voxel_size
    p4 = tsdf.origin + wp.vec3(
        wp.float32(gx) - center_offset_x,
        wp.float32(gy) - center_offset_y,
        wp.float32(gz + 1) - center_offset_z
    ) * tsdf.voxel_size

    vertex_idx = edge_offsets[tid]

    # Initialize edge vertex indices to -1 (not created)
    edge_vertex_indices[tid * 3 + 0] = wp.int32(-1)
    edge_vertex_indices[tid * 3 + 1] = wp.int32(-1)
    edge_vertex_indices[tid * 3 + 2] = wp.int32(-1)

    # Edge 0: corners 0-1 (along +x)
    if s0[0] * s1[0] < 0.0:
        v = interpolate_edge_vertex(p0, p1, s0[0], s1[0])
        if refine_iterations > 0:
            v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
        n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
        vertices[vertex_idx] = v
        normals[vertex_idx] = wp.normalize(n)
        edge_vertex_indices[tid * 3 + 0] = vertex_idx
        vertex_idx += wp.int32(1)

    # Edge 3: corners 0-3 (along +y)
    if s0[0] * s3[0] < 0.0:
        v = interpolate_edge_vertex(p0, p3, s0[0], s3[0])
        if refine_iterations > 0:
            v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
        n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
        vertices[vertex_idx] = v
        normals[vertex_idx] = wp.normalize(n)
        edge_vertex_indices[tid * 3 + 1] = vertex_idx
        vertex_idx += wp.int32(1)

    # Edge 8: corners 0-4 (along +z)
    if s0[0] * s4[0] < 0.0:
        v = interpolate_edge_vertex(p0, p4, s0[0], s4[0])
        if refine_iterations > 0:
            v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
        n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
        vertices[vertex_idx] = v
        normals[vertex_idx] = wp.normalize(n)
        edge_vertex_indices[tid * 3 + 2] = vertex_idx


@wp.kernel
def generate_triangles_shared_kernel(
    tsdf: BlockSparseTSDFWarp,
    level: float,
    minimum_tsdf_weight: float,
    surface_block_idx: wp.array(dtype=wp.int32),
    surface_cube_idx: wp.array(dtype=wp.int32),
    tri_offsets: wp.array(dtype=wp.int32),
    n_surfaces: wp.int32,
    tri_table: wp.array(dtype=wp.int32),
    edge_owner: wp.array(dtype=wp.int32),
    sorted_global_ids: wp.array(dtype=wp.int32),
    sparse_indices_sorted: wp.array(dtype=wp.int32),
    edge_vertex_indices: wp.array(dtype=wp.int32),
    # Output
    triangles: wp.array(dtype=wp.int32),
):
    """Generate triangles with shared vertex lookup.

    For each triangle edge:
    1. Look up edge owner (may be in neighbor block)
    2. Binary search to find owner's sparse index
    3. Look up vertex index from owner's edge_vertex_indices
    """
    tid = wp.tid()
    if tid >= n_surfaces:
        return

    block_idx = surface_block_idx[tid]
    cube_idx = surface_cube_idx[tid]

    cx = cube_idx % 8
    cy = (cube_idx // 8) % 8
    cz = cube_idx // 64

    bx = tsdf.block_coords[block_idx * 3 + 0]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    # Sample 8 corners for cube config
    s0 = sample_cube_corner(cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s1 = sample_cube_corner(cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s2 = sample_cube_corner(cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s3 = sample_cube_corner(cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s4 = sample_cube_corner(cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s5 = sample_cube_corner(cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s6 = sample_cube_corner(cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s7 = sample_cube_corner(cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)

    # Compute cube config
    cube_config = wp.int32(0)
    if s0[0] < 0.0:
        cube_config = cube_config | wp.int32(1)
    if s1[0] < 0.0:
        cube_config = cube_config | wp.int32(2)
    if s2[0] < 0.0:
        cube_config = cube_config | wp.int32(4)
    if s3[0] < 0.0:
        cube_config = cube_config | wp.int32(8)
    if s4[0] < 0.0:
        cube_config = cube_config | wp.int32(16)
    if s5[0] < 0.0:
        cube_config = cube_config | wp.int32(32)
    if s6[0] < 0.0:
        cube_config = cube_config | wp.int32(64)
    if s7[0] < 0.0:
        cube_config = cube_config | wp.int32(128)

    tri_base = tri_offsets[tid]
    table_offset = cube_config * 16

    for t in range(5):  # Max 5 triangles per cube
        e0 = tri_table[table_offset + t * 3]
        if e0 < 0:
            break

        e1 = tri_table[table_offset + t * 3 + 1]
        e2 = tri_table[table_offset + t * 3 + 2]

        # Look up vertex for each edge
        for v_idx in range(3):
            edge = e0
            if v_idx == 1:
                edge = e1
            elif v_idx == 2:
                edge = e2

            # Get edge owner
            owner = get_edge_owner_combined(
                edge, block_idx, cx, cy, cz, bx, by, bz,
                tsdf, edge_owner
            )

            vertex_id = wp.int32(-1)

            if owner[0] >= 0:
                # Compute global_id for owner cube
                owner_global_id = owner[0] * 512 + owner[1]

                # Binary search for owner's sparse index
                search_idx = binary_search_int32(sorted_global_ids, n_surfaces, owner_global_id)

                if search_idx >= 0:
                    owner_sparse = sparse_indices_sorted[search_idx]
                    edge_array_idx = local_edge_to_array_idx(owner[2])
                    vertex_id = edge_vertex_indices[owner_sparse * 3 + edge_array_idx]

            triangles[(tri_base + t) * 3 + v_idx] = vertex_id


# =============================================================================
# Triangle Counting Kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def count_triangles_kernel(
    tsdf: BlockSparseTSDFWarp,
    level: float,
    minimum_tsdf_weight: float,
    surface_block_idx: wp.array(dtype=wp.int32),
    surface_cube_idx: wp.array(dtype=wp.int32),
    n_surfaces: wp.int32,
    num_tris_table: wp.array(dtype=wp.int32),
    # Output
    tri_counts: wp.array(dtype=wp.int32),
):
    """Count triangles for each surface cube (combined SDF)."""
    tid = wp.tid()
    if tid >= n_surfaces:
        return

    block_idx = surface_block_idx[tid]
    cube_idx = surface_cube_idx[tid]

    # Cube local coords (0-7 in each dimension)
    cx = cube_idx % 8
    cy = (cube_idx // 8) % 8
    cz = cube_idx // 64

    # Block world coords
    bx = tsdf.block_coords[block_idx * 3 + 0]
    by = tsdf.block_coords[block_idx * 3 + 1]
    bz = tsdf.block_coords[block_idx * 3 + 2]

    # Sample 8 corners
    s0 = sample_cube_corner(cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s1 = sample_cube_corner(cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s2 = sample_cube_corner(cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s3 = sample_cube_corner(cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s4 = sample_cube_corner(cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s5 = sample_cube_corner(cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s6 = sample_cube_corner(cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)
    s7 = sample_cube_corner(cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight)

    # Compute cube index (8-bit configuration)
    cube_config = wp.int32(0)
    if s0[0] < 0.0:
        cube_config = cube_config | wp.int32(1)
    if s1[0] < 0.0:
        cube_config = cube_config | wp.int32(2)
    if s2[0] < 0.0:
        cube_config = cube_config | wp.int32(4)
    if s3[0] < 0.0:
        cube_config = cube_config | wp.int32(8)
    if s4[0] < 0.0:
        cube_config = cube_config | wp.int32(16)
    if s5[0] < 0.0:
        cube_config = cube_config | wp.int32(32)
    if s6[0] < 0.0:
        cube_config = cube_config | wp.int32(64)
    if s7[0] < 0.0:
        cube_config = cube_config | wp.int32(128)

    tri_counts[tid] = num_tris_table[cube_config]


# =============================================================================
# Color Sampling Kernel
# =============================================================================


@wp.kernel(enable_backward=False)
def sample_vertex_colors_kernel(
    vertices: wp.array(dtype=wp.vec3),
    n_vertices: wp.int32,
    tsdf: BlockSparseTSDFWarp,
    # Output
    colors: wp.array(dtype=wp.vec3ub),
):
    """Sample colors for mesh vertices from weighted RGB sums."""
    tid = wp.tid()
    if tid >= n_vertices:
        return

    pos = vertices[tid]

    # Center offset for center-origin convention
    center_offset_x = wp.float32(tsdf.grid_W) * 0.5
    center_offset_y = wp.float32(tsdf.grid_H) * 0.5
    center_offset_z = wp.float32(tsdf.grid_D) * 0.5

    # Compute voxel coordinates (reverse of center-origin world-to-voxel)
    vx = (pos[0] - tsdf.origin[0]) / tsdf.voxel_size + center_offset_x
    vy = (pos[1] - tsdf.origin[1]) / tsdf.voxel_size + center_offset_y
    vz = (pos[2] - tsdf.origin[2]) / tsdf.voxel_size + center_offset_z

    # Block coordinates
    block_size_f = wp.float32(tsdf.block_size)
    bx = wp.int32(wp.floor(vx / block_size_f))
    by = wp.int32(wp.floor(vy / block_size_f))
    bz = wp.int32(wp.floor(vz / block_size_f))

    # Look up block
    pool_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)

    if pool_idx < 0:
        # Block not found - use gray
        colors[tid] = wp.vec3ub(wp.uint8(128), wp.uint8(128), wp.uint8(128))
        return

    # Read per-block RGB and compute average
    rgb = compute_avg_rgb_uint8_from_block(tsdf.block_rgb, pool_idx)

    colors[tid] = wp.vec3ub(wp.uint8(rgb[0]), wp.uint8(rgb[1]), wp.uint8(rgb[2]))


# =============================================================================
# Public API
# =============================================================================


def extract_mesh_block_sparse(
    tsdf,  # BlockSparseTSDF instance
    level: float = 0.0,
    surface_only: bool = False,
    refine_iterations: int = 0,
    minimum_tsdf_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract mesh from block-sparse TSDF using marching cubes with edge sharing.

    This version uses shared vertices (like dense-sparse MC) to reduce
    vertex count by ~3x compared to the non-shared version.

    Args:
        tsdf: BlockSparseTSDF instance.
        level: Isosurface level (typically 0.0).
        surface_only: If True, only extract mesh near the surface (|sdf| < truncation).
            Excludes triangles from regions deep inside the object where TSDF
            is clamped to -truncation_distance.
        refine_iterations: Number of Newton-Raphson iterations for vertex refinement.
            0 = no refinement (linear interpolation only). Higher values (2-5)
            produce smoother meshes at the cost of more computation.
        minimum_tsdf_weight: Minimum weight to consider a voxel as observed.

    Returns:
        Tuple of (vertices, triangles, normals, colors):
            - vertices: (N, 3) float32 tensor of vertex positions
            - triangles: (M, 3) int32 tensor of vertex indices
            - normals: (N, 3) float32 tensor of vertex normals
            - colors: (N, 3) uint8 tensor of vertex colors
    """
    device = tsdf.device
    mc_tables = MCLookupTables.get(device)
    warp_data = tsdf.get_warp_data()
    num_alloc = tsdf.data.num_allocated.item()

    empty_result = (
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.int32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.uint8, device=device),
    )

    if num_alloc == 0:
        return empty_result

    # Surface band for filtering (0.0 = no filtering)
    surface_band = tsdf.config.truncation_distance if surface_only else 0.0

    # =========================================================================
    # Step 1a: Count surface cubes (no allocation, just count)
    # =========================================================================
    surface_count = torch.zeros(1, dtype=torch.int32, device=device)
    n_threads = num_alloc * 512  # block size
    _, stream = get_warp_device_stream(surface_count)

    wp.launch(
        count_surface_cubes_kernel,
        dim=n_threads,
        inputs=[
            warp_data,
            level,
            surface_band,
            minimum_tsdf_weight,
            wp.from_torch(surface_count, dtype=wp.int32),
        ],
        device=wp.torch.device_from_torch(device),
        stream=stream,
        adjoint=False,
    )

    n_surfaces = int(surface_count.item())
    if n_surfaces == 0:
        return empty_result

    # =========================================================================
    # Step 1b: Allocate exact size and append surface cubes
    # =========================================================================
    surface_block_idx = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    surface_cube_idx = torch.zeros(n_surfaces, dtype=torch.int32, device=device)

    # Reset counter for second pass
    surface_count.zero_()

    wp.launch(
        append_surface_cubes_kernel,
        dim=n_threads,
        inputs=[
            warp_data,
            level,
            surface_band,
            minimum_tsdf_weight,
            wp.from_torch(surface_count, dtype=wp.int32),
            wp.from_torch(surface_block_idx, dtype=wp.int32),
            wp.from_torch(surface_cube_idx, dtype=wp.int32),
        ],
        device=wp.torch.device_from_torch(device),
        stream=stream,
    )

    # =========================================================================
    # Step 2: Count owned edges per surface cube
    # =========================================================================
    edge_counts = torch.zeros(n_surfaces, dtype=torch.int32, device=device)

    wp.launch(
        count_edges_block_sparse_kernel,
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
        device=wp.torch.device_from_torch(device),
        stream=stream,
    )

    # =========================================================================
    # Step 3: Prefix sum for vertex offsets
    # =========================================================================
    edge_offsets = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    if n_surfaces > 0:
        edge_offsets[1:] = torch.cumsum(edge_counts[:-1], dim=0)
    total_vertices = int(edge_counts.sum().item())

    if total_vertices == 0:
        return empty_result

    # =========================================================================
    # Step 4: Generate vertices for owned edges
    # =========================================================================
    vertices = torch.zeros((total_vertices, 3), dtype=torch.float32, device=device)
    normals = torch.zeros((total_vertices, 3), dtype=torch.float32, device=device)
    edge_vertex_indices = torch.full(
        (n_surfaces * 3,), -1, dtype=torch.int32, device=device
    )

    wp.launch(
        generate_vertices_block_sparse_kernel,
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
        device=wp.torch.device_from_torch(device),
        stream=stream,
    )

    # =========================================================================
    # Step 5: Sort surface cubes for binary search lookup
    # =========================================================================
    global_ids = surface_block_idx * 512 + surface_cube_idx
    sorted_global_ids, sort_order = torch.sort(global_ids)
    sparse_indices_sorted = sort_order.to(torch.int32)
    sorted_global_ids = sorted_global_ids.to(torch.int32)

    # =========================================================================
    # Step 6: Count triangles per surface cube
    # =========================================================================
    tri_counts = torch.zeros(n_surfaces, dtype=torch.int32, device=device)

    wp.launch(
        count_triangles_kernel,
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
        device=wp.torch.device_from_torch(device),
        stream=stream,
    )

    # =========================================================================
    # Step 7: Prefix sum for triangle offsets
    # =========================================================================
    tri_offsets = torch.zeros(n_surfaces, dtype=torch.int32, device=device)
    if n_surfaces > 0:
        tri_offsets[1:] = torch.cumsum(tri_counts[:-1], dim=0)
    total_tris = int(tri_counts.sum().item())

    if total_tris == 0:
        return empty_result

    # =========================================================================
    # Step 8: Generate triangles with shared vertex lookup
    # =========================================================================
    triangles = torch.zeros((total_tris, 3), dtype=torch.int32, device=device)

    wp.launch(
        generate_triangles_shared_kernel,
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
            wp.from_torch(sorted_global_ids, dtype=wp.int32),
            wp.from_torch(sparse_indices_sorted, dtype=wp.int32),
            wp.from_torch(edge_vertex_indices, dtype=wp.int32),
            wp.from_torch(triangles.view(-1), dtype=wp.int32),
        ],
        device=wp.torch.device_from_torch(device),
        stream=stream,
    )

    # =========================================================================
    # Step 9: Filter invalid triangles (memory-efficient kernel-based)
    # =========================================================================
    from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_filter import filter_triangles

    triangles = filter_triangles(
        triangles, vertices, tsdf.config.voxel_size, flip_winding=True
    )

    if triangles.shape[0] == 0:
        return empty_result

    # =========================================================================
    # Step 10: Sample vertex colors
    # =========================================================================
    colors = torch.zeros((total_vertices, 3), dtype=torch.uint8, device=device)

    wp.launch(
        sample_vertex_colors_kernel,
        dim=total_vertices,
        inputs=[
            wp.from_torch(vertices, dtype=wp.vec3),
            total_vertices,
            warp_data,
            wp.from_torch(colors, dtype=wp.vec3ub),
        ],
        device=wp.torch.device_from_torch(device),
        stream=stream,
    )

    return vertices, triangles, normals, colors

