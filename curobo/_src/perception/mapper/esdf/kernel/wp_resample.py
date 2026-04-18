# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp kernels for decoupled ESDF generation.

This module provides kernels for generating ESDF at a different resolution or
origin than the source TSDF. It enables sliding-window ESDF generation for
motion planning without re-integrating depth observations.

Key Features:
- Seed ESDF sites from fine TSDF at arbitrary resolution/origin
- Compute signed distances with sign lookup from fine TSDF
- CUDA graph safe with tensor-based dynamic parameters

Usage:
    # Seed sites from TSDF to ESDF (different resolution/origin)
    seed_esdf_sites_from_tsdf_warp(
        tsdf_grid, tsdf_origin, tsdf_voxel_size, tsdf_grid_shape,
        esdf_site_index, esdf_origin, esdf_voxel_size, esdf_grid_shape,
        minimum_tsdf_weight,
    )
"""

from typing import Tuple

import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_hash import hash_lookup
from curobo._src.perception.mapper.kernel.wp_tsdf_sample import (
    sample_combined_sdf,
    sample_static_sdf,
)
from curobo._src.perception.mapper.storage import BlockSparseTSDF
from curobo._src.perception.mapper.util.utils_quantization import (
    get_sdf_from_float16_grids,
    get_weight_from_float16,
    pack_site_coords,
)
from curobo._src.util.warp import get_warp_device_stream


@wp.kernel
def seed_esdf_sites_from_tsdf_kernel(
    # TSDF source (fine resolution)
    sdf_weight_grid: wp.array(dtype=wp.float16),  # float16: accumulated sdf*weight
    weight_grid: wp.array(dtype=wp.float16),      # float16: accumulated weight
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    tsdf_D: wp.int32,
    tsdf_H: wp.int32,
    tsdf_W: wp.int32,
    # ESDF target (coarse resolution, different origin)
    esdf_site_index: wp.array(dtype=wp.int32),
    esdf_min_tsdf: wp.array(dtype=wp.float32),  # float32 for atomic_min
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_voxel_size: wp.array(dtype=wp.float32),  # Tensor for CUDA graph safety
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    # Parameters
    minimum_tsdf_weight: wp.float32,
):
    """Seed ESDF sites by mapping TSDF surface voxels to ESDF grid.

    Each thread processes one TSDF voxel. If the voxel is observed,
    writes atomic_min to esdf_min_tsdf for conservative sign determination.
    If on the surface, also writes site coordinates for PBA seeding.

    This kernel iterates over the FINE TSDF voxels to ensure all surfaces
    are captured, even when ESDF resolution is coarser.

    Args:
        sdf_weight_grid: Float16 grid with accumulated sdf*weight.
        weight_grid: Float16 grid with accumulated weight.
        tsdf_origin: Fine TSDF origin (3,).
        tsdf_voxel_size: Fine TSDF voxel size.
        tsdf_D, tsdf_H, tsdf_W: Fine TSDF dimensions.
        esdf_site_index: Coarse ESDF site buffer (to be filled).
        esdf_min_tsdf: Min TSDF buffer (float32) for conservative sign.
        esdf_origin: Coarse ESDF origin (3,).
        esdf_voxel_size: Coarse ESDF voxel size (tensor for CUDA graph safety).
        esdf_D, esdf_H, esdf_W: Coarse ESDF dimensions.
        minimum_tsdf_weight: Minimum weight to consider voxel observed.
    """
    tid = wp.tid()

    # Compute 3D TSDF voxel index
    tsdf_x = tid % tsdf_W
    tsdf_y = (tid // tsdf_W) % tsdf_H
    tsdf_z = tid // (tsdf_W * tsdf_H)

    if tsdf_z >= tsdf_D:
        return

    # Read TSDF values from float16 grids
    sdf = get_sdf_from_float16_grids(sdf_weight_grid[tid], weight_grid[tid])
    weight = get_weight_from_float16(weight_grid[tid])

    # Skip unobserved voxels
    is_observed = weight >= minimum_tsdf_weight
    if not is_observed:
        return

    # Compute world position of TSDF voxel center (origin is at center/half-extents)
    world_x = tsdf_origin[0] + (wp.float32(tsdf_x) + 0.5 - wp.float32(tsdf_W) * 0.5) * tsdf_voxel_size
    world_y = tsdf_origin[1] + (wp.float32(tsdf_y) + 0.5 - wp.float32(tsdf_H) * 0.5) * tsdf_voxel_size
    world_z = tsdf_origin[2] + (wp.float32(tsdf_z) + 0.5 - wp.float32(tsdf_D) * 0.5) * tsdf_voxel_size

    # Map to ESDF grid indices (origin is at center/half-extents)
    # ESDF layout: (nx, ny, nz) - D=nx, H=ny, W=nz (X slowest, Z fastest)
    esdf_vs = esdf_voxel_size[0]  # Read from tensor for CUDA graph safety
    esdf_x = wp.int32((world_x - esdf_origin[0]) / esdf_vs + wp.float32(esdf_D) * 0.5)
    esdf_y = wp.int32((world_y - esdf_origin[1]) / esdf_vs + wp.float32(esdf_H) * 0.5)
    esdf_z = wp.int32((world_z - esdf_origin[2]) / esdf_vs + wp.float32(esdf_W) * 0.5)

    # Bounds check for ESDF grid (X against D, Y against H, Z against W)
    if esdf_x < 0 or esdf_x >= esdf_D:
        return
    if esdf_y < 0 or esdf_y >= esdf_H:
        return
    if esdf_z < 0 or esdf_z >= esdf_W:
        return

    # Compute flat ESDF index (X slowest, Z fastest)
    esdf_tid = esdf_x * esdf_H * esdf_W + esdf_y * esdf_W + esdf_z

    # Vote on sign: +1 for outside, -1 for inside
    # Majority vote determines final sign (more robust than atomic_min)
    # Initialize esdf_min_tsdf to 0.0, then accumulate votes
    if sdf >= 0.0:
        wp.atomic_add(esdf_min_tsdf, esdf_tid, wp.float32(1.0))
    else:
        wp.atomic_add(esdf_min_tsdf, esdf_tid, wp.float32(-1.0))

    # Only write site coordinates for surface voxels (for PBA seeding)
    is_surface = (sdf <= 0.0) and (sdf >= -tsdf_voxel_size)
    if is_surface:
        esdf_site_index[esdf_tid] = pack_site_coords(
            wp.int32(esdf_x), wp.int32(esdf_y), wp.int32(esdf_z)
        )


def seed_esdf_sites_from_tsdf_warp(
    # TSDF source
    sdf_weight_grid: torch.Tensor,
    weight_grid: torch.Tensor,
    tsdf_origin: torch.Tensor,
    tsdf_voxel_size: float,
    tsdf_grid_shape: Tuple[int, int, int],
    # ESDF target
    esdf_site_index: torch.Tensor,
    esdf_min_tsdf: torch.Tensor,
    esdf_origin: torch.Tensor,
    esdf_voxel_size_tensor: torch.Tensor,  # (1,) tensor for CUDA graph safety
    esdf_grid_shape: Tuple[int, int, int],
    # Parameters
    minimum_tsdf_weight: float,
) -> None:
    """Seed ESDF sites from TSDF with conservative min-TSDF tracking.

    This function maps surface voxels from the fine TSDF grid to the coarse
    ESDF grid. It iterates over all TSDF voxels to ensure all surfaces are
    captured. Also writes atomic_min to esdf_min_tsdf for conservative sign.

    Args:
        sdf_weight_grid: Float16 grid (D, H, W) with accumulated sdf*weight.
        weight_grid: Float16 grid (D, H, W) with accumulated weight.
        tsdf_origin: Fine TSDF origin (3,) float32.
        tsdf_voxel_size: Fine TSDF voxel size in meters.
        tsdf_grid_shape: Fine TSDF dimensions (D, H, W).
        esdf_site_index: Coarse ESDF site buffer (D, H, W) int32, modified in-place.
        esdf_min_tsdf: Min TSDF buffer (D, H, W) float32, modified in-place.
            Should be initialized to 1e10 before calling.
        esdf_origin: Coarse ESDF origin (3,) float32 tensor.
        esdf_voxel_size_tensor: Coarse ESDF voxel size (1,) float32 tensor.
        esdf_grid_shape: Coarse ESDF dimensions (D, H, W).
        minimum_tsdf_weight: Minimum weight to consider voxel observed.

    Note:
        esdf_origin and esdf_voxel_size_tensor are tensors (not scalars) for
        CUDA graph safety. Update their contents to change the ESDF region
        without re-capturing the graph.
    """
    tsdf_D, tsdf_H, tsdf_W = tsdf_grid_shape
    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    n_tsdf_voxels = tsdf_D * tsdf_H * tsdf_W

    device = sdf_weight_grid.device
    _, stream = get_warp_device_stream(sdf_weight_grid)

    # Convert to Warp arrays
    sdf_weight_wp = wp.from_torch(sdf_weight_grid.view(-1), dtype=wp.float16)
    weight_wp = wp.from_torch(weight_grid.view(-1), dtype=wp.float16)
    tsdf_origin_wp = wp.from_torch(tsdf_origin, dtype=wp.float32)
    esdf_site_wp = wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32)
    esdf_min_tsdf_wp = wp.from_torch(esdf_min_tsdf.view(-1), dtype=wp.float32)
    esdf_origin_wp = wp.from_torch(esdf_origin, dtype=wp.float32)
    esdf_vs_wp = wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32)

    wp.launch(
        kernel=seed_esdf_sites_from_tsdf_kernel,
        dim=n_tsdf_voxels,
        inputs=[
            sdf_weight_wp,
            weight_wp,
            tsdf_origin_wp,
            wp.float32(tsdf_voxel_size),
            wp.int32(tsdf_D),
            wp.int32(tsdf_H),
            wp.int32(tsdf_W),
            esdf_site_wp,
            esdf_min_tsdf_wp,
            esdf_origin_wp,
            esdf_vs_wp,
            wp.int32(esdf_D),
            wp.int32(esdf_H),
            wp.int32(esdf_W),
            wp.float32(minimum_tsdf_weight),
        ],
        device=wp.device_from_torch(device),
        stream=stream,
    )


@wp.kernel
def compute_decoupled_esdf_sign_kernel(
    # ESDF (from PBA)
    esdf_site_index: wp.array(dtype=wp.int32),
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_voxel_size: wp.array(dtype=wp.float32),  # Tensor for CUDA graph safety
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    # TSDF source (for sign lookup)
    sdf_weight_grid: wp.array(dtype=wp.float16),  # float16: accumulated sdf*weight
    weight_grid: wp.array(dtype=wp.float16),      # float16: accumulated weight
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    tsdf_D: wp.int32,
    tsdf_H: wp.int32,
    tsdf_W: wp.int32,
    # Output
    esdf_dist_field: wp.array(dtype=wp.float16),
):
    """Compute signed ESDF distances with sign from fine TSDF.

    After PBA propagation, this kernel computes the Euclidean distance to
    the nearest site and determines the sign by sampling the fine TSDF.

    Args:
        esdf_site_index: ESDF sites after PBA propagation.
        esdf_origin: ESDF grid origin.
        esdf_voxel_size: ESDF voxel size (tensor).
        esdf_D, esdf_H, esdf_W: ESDF dimensions.
        sdf_weight_grid: Float16 grid with accumulated sdf*weight.
        weight_grid: Float16 grid with accumulated weight.
        tsdf_origin: Fine TSDF origin.
        tsdf_voxel_size: Fine TSDF voxel size.
        tsdf_D, tsdf_H, tsdf_W: Fine TSDF dimensions.
        esdf_dist_field: Output distance field.
    ESDF layout: (nx, ny, nz) - D=nx, H=ny, W=nz (X slowest, Z fastest).
    TSDF layout: (nz, ny, nx) - D=nz, H=ny, W=nx (Z slowest, X fastest).
    """
    tid = wp.tid()

    # Compute 3D ESDF voxel index
    # ESDF layout: (nx, ny, nz) - X slowest (D), Z fastest (W)
    esdf_z = tid % esdf_W                  # Z varies fastest
    esdf_y = (tid // esdf_W) % esdf_H      # Y varies middle
    esdf_x = tid // (esdf_W * esdf_H)      # X varies slowest

    if esdf_x >= esdf_D:
        return

    # Read site
    site_packed = esdf_site_index[tid]
    if site_packed < wp.int32(0):
        # No site found
        esdf_dist_field[tid] = wp.float16(1e10)
        return

    # Unpack site coordinates (in ESDF grid space)
    sx = site_packed & 0x3FF
    sy = (site_packed >> 10) & 0x3FF
    sz = (site_packed >> 20) & 0x3FF

    # Compute Euclidean distance in voxel units
    dx = wp.float32(esdf_x - sx)
    dy = wp.float32(esdf_y - sy)
    dz = wp.float32(esdf_z - sz)
    dist_voxels = wp.sqrt(dx * dx + dy * dy + dz * dz)

    # Convert to meters
    esdf_vs = esdf_voxel_size[0]
    dist_meters = dist_voxels * esdf_vs

    # Compute world position of ESDF voxel center (origin is at center/half-extents)
    # ESDF layout: D=nx, H=ny, W=nz
    world_x = esdf_origin[0] + (wp.float32(esdf_x) + 0.5 - wp.float32(esdf_D) * 0.5) * esdf_vs
    world_y = esdf_origin[1] + (wp.float32(esdf_y) + 0.5 - wp.float32(esdf_H) * 0.5) * esdf_vs
    world_z = esdf_origin[2] + (wp.float32(esdf_z) + 0.5 - wp.float32(esdf_W) * 0.5) * esdf_vs

    # Map to TSDF grid indices (origin is at center/half-extents)
    # TSDF layout: D=nz, H=ny, W=nx (Z slowest, X fastest)
    tsdf_x = wp.int32((world_x - tsdf_origin[0]) / tsdf_voxel_size + wp.float32(tsdf_W) * 0.5)
    tsdf_y = wp.int32((world_y - tsdf_origin[1]) / tsdf_voxel_size + wp.float32(tsdf_H) * 0.5)
    tsdf_z = wp.int32((world_z - tsdf_origin[2]) / tsdf_voxel_size + wp.float32(tsdf_D) * 0.5)

    # Clamp to TSDF bounds
    tsdf_x = wp.clamp(tsdf_x, 0, tsdf_W - 1)
    tsdf_y = wp.clamp(tsdf_y, 0, tsdf_H - 1)
    tsdf_z = wp.clamp(tsdf_z, 0, tsdf_D - 1)

    # Read TSDF value for sign from float16 grids (TSDF: Z slowest)
    tsdf_tid = tsdf_z * tsdf_H * tsdf_W + tsdf_y * tsdf_W + tsdf_x
    tsdf_sdf = get_sdf_from_float16_grids(sdf_weight_grid[tsdf_tid], weight_grid[tsdf_tid])

    # Apply sign: negative inside, positive outside
    if tsdf_sdf < 0.0:
        dist_meters = -dist_meters

    esdf_dist_field[tid] = wp.float16(dist_meters)


@wp.func
def smoothstep(t: wp.float32) -> wp.float32:
    """Hermite smoothstep interpolation: 0 at t=0, 1 at t=1, smooth derivatives."""
    t_clamped = wp.clamp(t, 0.0, 1.0)
    return t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)


@wp.kernel
def compute_decoupled_esdf_blended_kernel(
    # ESDF (from PBA)
    esdf_site_index: wp.array(dtype=wp.int32),
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_voxel_size: wp.array(dtype=wp.float32),  # Tensor for CUDA graph safety
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    # TSDF source (for sign and blending)
    sdf_weight_grid: wp.array(dtype=wp.float16),  # float16: accumulated sdf*weight
    weight_grid: wp.array(dtype=wp.float16),      # float16: accumulated weight
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    tsdf_D: wp.int32,
    tsdf_H: wp.int32,
    tsdf_W: wp.int32,
    # Blending parameters
    truncation_distance: wp.float32,  # meters
    # Output
    esdf_dist_field: wp.array(dtype=wp.float16),
):
    """Compute TSDF-blended ESDF distances for decoupled grids.

    Blends TSDF (accurate near surface) with EDT (correct topology far from surface)
    using smooth interpolation. This version handles different ESDF/TSDF resolutions.

    Blending behavior (when truncation_distance > 0):
    - |tsdf| < 0.5 * truncation: Use TSDF directly (sub-voxel precision).
    - |tsdf| >= truncation: Use EDT distance (correct topology).
    - In between: Smooth Hermite blend.

    Args:
        esdf_site_index: ESDF sites after PBA propagation.
        esdf_origin: ESDF grid origin.
        esdf_voxel_size: ESDF voxel size (tensor for CUDA graph safety).
        esdf_D, esdf_H, esdf_W: ESDF dimensions.
        sdf_weight_grid: Float16 grid with accumulated sdf*weight.
        weight_grid: Float16 grid with accumulated weight.
        tsdf_origin: Fine TSDF origin.
        tsdf_voxel_size: Fine TSDF voxel size.
        tsdf_D, tsdf_H, tsdf_W: Fine TSDF dimensions.
        truncation_distance: TSDF truncation distance (0 = no blending).
        esdf_dist_field: Output distance field.

    ESDF layout: (nx, ny, nz) - D=nx, H=ny, W=nz (X slowest, Z fastest).
    TSDF layout: (nz, ny, nx) - D=nz, H=ny, W=nx (Z slowest, X fastest).
    """
    tid = wp.tid()

    # Compute 3D ESDF voxel index
    # ESDF layout: (nx, ny, nz) - X slowest (D), Z fastest (W)
    esdf_z = tid % esdf_W                  # Z varies fastest
    esdf_y = (tid // esdf_W) % esdf_H      # Y varies middle
    esdf_x = tid // (esdf_W * esdf_H)      # X varies slowest

    if esdf_x >= esdf_D:
        return

    # Read ESDF voxel size
    esdf_vs = esdf_voxel_size[0]

    # Compute world position of ESDF voxel center (origin is at center/half-extents)
    # ESDF layout: D=nx, H=ny, W=nz
    world_x = esdf_origin[0] + (wp.float32(esdf_x) + 0.5 - wp.float32(esdf_D) * 0.5) * esdf_vs
    world_y = esdf_origin[1] + (wp.float32(esdf_y) + 0.5 - wp.float32(esdf_H) * 0.5) * esdf_vs
    world_z = esdf_origin[2] + (wp.float32(esdf_z) + 0.5 - wp.float32(esdf_W) * 0.5) * esdf_vs

    # Map to TSDF grid indices (origin is at center/half-extents)
    # TSDF layout: D=nz, H=ny, W=nx (Z slowest, X fastest)
    tsdf_x = wp.int32((world_x - tsdf_origin[0]) / tsdf_voxel_size + wp.float32(tsdf_W) * 0.5)
    tsdf_y = wp.int32((world_y - tsdf_origin[1]) / tsdf_voxel_size + wp.float32(tsdf_H) * 0.5)
    tsdf_z = wp.int32((world_z - tsdf_origin[2]) / tsdf_voxel_size + wp.float32(tsdf_D) * 0.5)

    # Check if within TSDF bounds
    in_tsdf_bounds = (
        tsdf_x >= 0 and tsdf_x < tsdf_W and
        tsdf_y >= 0 and tsdf_y < tsdf_H and
        tsdf_z >= 0 and tsdf_z < tsdf_D
    )

    # Read site
    site_packed = esdf_site_index[tid]
    if site_packed < wp.int32(0):
        # No site found - output large distance
        esdf_dist_field[tid] = wp.float16(1e10)
        return

    # Unpack site coordinates (in ESDF grid space)
    sx = site_packed & 0x3FF
    sy = (site_packed >> 10) & 0x3FF
    sz = (site_packed >> 20) & 0x3FF

    # Compute EDT distance in voxel units
    dx = wp.float32(esdf_x - sx)
    dy = wp.float32(esdf_y - sy)
    dz = wp.float32(esdf_z - sz)
    dist_voxels = wp.sqrt(dx * dx + dy * dy + dz * dz)
    edt_dist = dist_voxels * esdf_vs

    # Get TSDF value for sign and blending
    tsdf_sdf = 0.0
    if in_tsdf_bounds:
        # Clamp for safety
        tsdf_x_clamped = wp.clamp(tsdf_x, 0, tsdf_W - 1)
        tsdf_y_clamped = wp.clamp(tsdf_y, 0, tsdf_H - 1)
        tsdf_z_clamped = wp.clamp(tsdf_z, 0, tsdf_D - 1)
        # TSDF: Z slowest, X fastest
        tsdf_tid = tsdf_z_clamped * tsdf_H * tsdf_W + tsdf_y_clamped * tsdf_W + tsdf_x_clamped
        tsdf_sdf = get_sdf_from_float16_grids(sdf_weight_grid[tsdf_tid], weight_grid[tsdf_tid])

    # Apply sign from TSDF: negative inside
    if tsdf_sdf < 0.0:
        edt_dist = -edt_dist

    # Skip blending if outside TSDF bounds or blend disabled
    if not in_tsdf_bounds or truncation_distance <= 0.0:
        esdf_dist_field[tid] = wp.float16(edt_dist)
        return

    # Blend TSDF with EDT based on truncation distance
    # TSDF is only accurate within truncation; beyond that it's clamped
    abs_tsdf = wp.abs(tsdf_sdf)
    fade_start = 0.5 * truncation_distance  # 50% of truncation: use pure TSDF
    fade_end = truncation_distance  # 100% of truncation: use pure EDT

    if abs_tsdf <= fade_start:
        # Near surface: use TSDF directly (sub-voxel precision)
        esdf_dist_field[tid] = wp.float16(tsdf_sdf)
    elif abs_tsdf >= fade_end:
        # Far from surface: use EDT (correct topology)
        esdf_dist_field[tid] = wp.float16(edt_dist)
    else:
        # Transition zone: smooth Hermite blend
        t = (abs_tsdf - fade_start) / (fade_end - fade_start)
        alpha = smoothstep(t)
        blended = (1.0 - alpha) * tsdf_sdf + alpha * edt_dist
        esdf_dist_field[tid] = wp.float16(blended)


def compute_decoupled_esdf_warp(
    # ESDF (from PBA)
    esdf_site_index: torch.Tensor,
    esdf_origin: torch.Tensor,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    # TSDF source (for sign lookup)
    sdf_weight_grid: torch.Tensor,
    weight_grid: torch.Tensor,
    tsdf_origin: torch.Tensor,
    tsdf_voxel_size: float,
    tsdf_grid_shape: Tuple[int, int, int],
    # Output
    esdf_dist_field: torch.Tensor,
    # Blending (optional)
    truncation_distance: float = 0.0,
) -> None:
    """Compute signed ESDF distances with sign from fine TSDF.

    Optionally blends TSDF with EDT for smoother output near surfaces.

    When truncation_distance > 0, blends TSDF with EDT:
    - |tsdf| < 0.5 * truncation: Use TSDF (sub-voxel precision).
    - |tsdf| >= truncation: Use EDT (TSDF is clamped).
    - In between: Smooth Hermite interpolation.

    Args:
        esdf_site_index: ESDF sites after PBA propagation.
        esdf_origin: ESDF grid origin (3,) float32 tensor.
        esdf_voxel_size_tensor: ESDF voxel size (1,) float32 tensor.
        esdf_grid_shape: ESDF dimensions (D, H, W).
        sdf_weight_grid: Float16 grid with accumulated sdf*weight.
        weight_grid: Float16 grid with accumulated weight.
        tsdf_origin: Fine TSDF origin.
        tsdf_voxel_size: Fine TSDF voxel size.
        tsdf_grid_shape: Fine TSDF dimensions.
        esdf_dist_field: Output distance field (D, H, W) float16.
        truncation_distance: TSDF truncation distance. 0 = no blending, use EDT only.
    """
    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    tsdf_D, tsdf_H, tsdf_W = tsdf_grid_shape
    n_esdf_voxels = esdf_D * esdf_H * esdf_W

    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    # Convert to Warp arrays
    esdf_site_wp = wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32)
    esdf_origin_wp = wp.from_torch(esdf_origin, dtype=wp.float32)
    esdf_vs_wp = wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32)
    sdf_weight_wp = wp.from_torch(sdf_weight_grid.view(-1), dtype=wp.float16)
    weight_wp = wp.from_torch(weight_grid.view(-1), dtype=wp.float16)
    tsdf_origin_wp = wp.from_torch(tsdf_origin, dtype=wp.float32)
    dist_wp = wp.from_torch(esdf_dist_field.view(-1), dtype=wp.float16)

    if truncation_distance > 0.0:
        # Use blended kernel
        wp.launch(
            kernel=compute_decoupled_esdf_blended_kernel,
            dim=n_esdf_voxels,
            inputs=[
                esdf_site_wp,
                esdf_origin_wp,
                esdf_vs_wp,
                wp.int32(esdf_D),
                wp.int32(esdf_H),
                wp.int32(esdf_W),
                sdf_weight_wp,
                weight_wp,
                tsdf_origin_wp,
                wp.float32(tsdf_voxel_size),
                wp.int32(tsdf_D),
                wp.int32(tsdf_H),
                wp.int32(tsdf_W),
                wp.float32(truncation_distance),
                dist_wp,
            ],
            device=wp.device_from_torch(device),
            stream=stream,
        )
    else:
        # Use simple sign-only kernel
        wp.launch(
            kernel=compute_decoupled_esdf_sign_kernel,
            dim=n_esdf_voxels,
            inputs=[
                esdf_site_wp,
                esdf_origin_wp,
                esdf_vs_wp,
                wp.int32(esdf_D),
                wp.int32(esdf_H),
                wp.int32(esdf_W),
                sdf_weight_wp,
                weight_wp,
                tsdf_origin_wp,
                wp.float32(tsdf_voxel_size),
                wp.int32(tsdf_D),
                wp.int32(tsdf_H),
                wp.int32(tsdf_W),
                dist_wp,
            ],
            device=wp.device_from_torch(device),
            stream=stream,
        )


# =============================================================================
# Unified ESDF Distance Kernel (using pre-aggregated min_tsdf)
# =============================================================================


@wp.kernel
def infer_inside_ray_parity_kernel(
    site_index: wp.array(dtype=wp.int32),
    esdf_min_tsdf: wp.array(dtype=wp.float32),
    esdf_dist_field: wp.array(dtype=wp.float16),
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    max_ray_distance_voxels: wp.int32,  # Only check voxels within this EDT distance
):
    """Infer inside/outside using ray parity test for voxels with no TSDF observation.

    Grid layout: (nx, ny, nz) - X slowest (D), Z fastest (W).

    Only runs on voxels that:
    1. Have no TSDF observation (min_tsdf > 1e9)
    2. Are within max_ray_distance_voxels of a surface (already have small positive ESDF)

    Shoots 6 axis-aligned rays and counts surface intersections.
    Odd intersections = inside, Even = outside. Uses majority vote.
    """
    tid = wp.tid()

    # Grid layout: (nx, ny, nz) - X slowest (D), Z fastest (W)
    z = tid % esdf_W                  # Z varies fastest
    y = (tid // esdf_W) % esdf_H      # Y varies middle
    x = tid // (esdf_W * esdf_H)      # X varies slowest

    if x >= esdf_D:
        return

    # Only process voxels with no TSDF observation
    if esdf_min_tsdf[tid] < 1e9:
        return

    # Only process voxels close to a surface (small positive ESDF)
    current_dist = esdf_dist_field[tid]
    if current_dist < wp.float16(0.0):
        return  # Already marked inside
    if current_dist > wp.float16(wp.float32(max_ray_distance_voxels)):
        return  # Too far from surface, skip expensive ray test

    # Count inside votes from 6 axis-aligned rays
    inside_votes = wp.int32(0)

    # Ray +X (along slowest dimension)
    crossings = wp.int32(0)
    cx = x + 1
    while cx < esdf_D:
        idx = cx * esdf_H * esdf_W + y * esdf_W + z
        if site_index[idx] >= wp.int32(0):
            crossings = crossings + 1
        cx = cx + 1
    if crossings % 2 == 1:
        inside_votes = inside_votes + 1

    # Ray -X
    crossings = wp.int32(0)
    cx = x - 1
    while cx >= 0:
        idx = cx * esdf_H * esdf_W + y * esdf_W + z
        if site_index[idx] >= wp.int32(0):
            crossings = crossings + 1
        cx = cx - 1
    if crossings % 2 == 1:
        inside_votes = inside_votes + 1

    # Ray +Y
    crossings = wp.int32(0)
    cy = y + 1
    while cy < esdf_H:
        idx = x * esdf_H * esdf_W + cy * esdf_W + z
        if site_index[idx] >= wp.int32(0):
            crossings = crossings + 1
        cy = cy + 1
    if crossings % 2 == 1:
        inside_votes = inside_votes + 1

    # Ray -Y
    crossings = wp.int32(0)
    cy = y - 1
    while cy >= 0:
        idx = x * esdf_H * esdf_W + cy * esdf_W + z
        if site_index[idx] >= wp.int32(0):
            crossings = crossings + 1
        cy = cy - 1
    if crossings % 2 == 1:
        inside_votes = inside_votes + 1

    # Ray +Z (along fastest dimension)
    crossings = wp.int32(0)
    cz = z + 1
    while cz < esdf_W:
        idx = x * esdf_H * esdf_W + y * esdf_W + cz
        if site_index[idx] >= wp.int32(0):
            crossings = crossings + 1
        cz = cz + 1
    if crossings % 2 == 1:
        inside_votes = inside_votes + 1

    # Ray -Z
    crossings = wp.int32(0)
    cz = z - 1
    while cz >= 0:
        idx = x * esdf_H * esdf_W + y * esdf_W + cz
        if site_index[idx] >= wp.int32(0):
            crossings = crossings + 1
        cz = cz - 1
    if crossings % 2 == 1:
        inside_votes = inside_votes + 1

    # Majority vote: at least 4 of 6 rays agree we're inside
    if inside_votes >= wp.int32(4):
        # Negate the distance to mark as inside
        esdf_dist_field[tid] = -wp.abs(esdf_dist_field[tid])


@wp.func
def _esdf_to_tsdf_voxel_coords(
    esdf_ix: wp.int32,
    esdf_iy: wp.int32,
    esdf_iz: wp.int32,
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_vs: wp.float32,
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    use_grid_bounds: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
) -> wp.vec4i:
    """Map ESDF grid coords to TSDF global voxel coords.

    Returns vec4i(gx, gy, gz, valid) where valid=1 if in bounds, 0 otherwise.
    """
    world_x = esdf_origin[0] + (wp.float32(esdf_ix) + 0.5 - wp.float32(esdf_D) * 0.5) * esdf_vs
    world_y = esdf_origin[1] + (wp.float32(esdf_iy) + 0.5 - wp.float32(esdf_H) * 0.5) * esdf_vs
    world_z = esdf_origin[2] + (wp.float32(esdf_iz) + 0.5 - wp.float32(esdf_W) * 0.5) * esdf_vs

    gx = wp.int32(0)
    gy = wp.int32(0)
    gz = wp.int32(0)

    if use_grid_bounds != 0:
        gx = wp.int32((world_x - tsdf_origin[0]) / tsdf_voxel_size + wp.float32(grid_W) * 0.5)
        gy = wp.int32((world_y - tsdf_origin[1]) / tsdf_voxel_size + wp.float32(grid_H) * 0.5)
        gz = wp.int32((world_z - tsdf_origin[2]) / tsdf_voxel_size + wp.float32(grid_D) * 0.5)
        if gx < 0 or gx >= grid_W or gy < 0 or gy >= grid_H or gz < 0 or gz >= grid_D:
            return wp.vec4i(0, 0, 0, 0)
    else:
        gx = wp.int32((world_x - tsdf_origin[0]) / tsdf_voxel_size)
        gy = wp.int32((world_y - tsdf_origin[1]) / tsdf_voxel_size)
        gz = wp.int32((world_z - tsdf_origin[2]) / tsdf_voxel_size)

    return wp.vec4i(gx, gy, gz, 1)


@wp.func
def _hash_lookup_at_global_coords(
    tsdf: BlockSparseTSDFWarp,
    gx: wp.int32,
    gy: wp.int32,
    gz: wp.int32,
) -> wp.vec2i:
    """Hash lookup returning (pool_idx, local_idx). pool_idx=-1 if not found."""
    bx = gx // 8
    by = gy // 8
    bz = gz // 8

    pool_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
    if pool_idx < 0:
        return wp.vec2i(-1, 0)

    lx = gx % 8
    ly = gy % 8
    lz = gz % 8
    local_idx = lz * 64 + ly * 8 + lx
    return wp.vec2i(pool_idx, local_idx)


@wp.func
def lookup_combined_sdf_at_esdf_coords(
    tsdf: BlockSparseTSDFWarp,
    esdf_ix: wp.int32,
    esdf_iy: wp.int32,
    esdf_iz: wp.int32,
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_vs: wp.float32,
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    min_weight: wp.float32,
    use_grid_bounds: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
) -> wp.float32:
    """Sample combined (dynamic+static) SDF at an ESDF voxel position."""
    coords = _esdf_to_tsdf_voxel_coords(
        esdf_ix, esdf_iy, esdf_iz,
        esdf_origin, esdf_vs, esdf_D, esdf_H, esdf_W,
        tsdf_origin, tsdf_voxel_size, use_grid_bounds, grid_W, grid_H, grid_D,
    )
    if coords[3] == 0:
        return wp.float32(1e10)

    result = _hash_lookup_at_global_coords(tsdf, coords[0], coords[1], coords[2])
    if result[0] < 0:
        return wp.float32(1e10)

    return sample_combined_sdf(tsdf, result[0], result[1], min_weight)


@wp.func
def lookup_static_sdf_at_esdf_coords(
    tsdf: BlockSparseTSDFWarp,
    esdf_ix: wp.int32,
    esdf_iy: wp.int32,
    esdf_iz: wp.int32,
    esdf_origin: wp.array(dtype=wp.float32),
    esdf_vs: wp.float32,
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    use_grid_bounds: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
) -> wp.float32:
    """Sample static-only SDF at an ESDF voxel position.

    Only static (primitive) geometry is watertight, so only static SDF
    is reliable for sign determination via the adjacent skip step method.
    """
    coords = _esdf_to_tsdf_voxel_coords(
        esdf_ix, esdf_iy, esdf_iz,
        esdf_origin, esdf_vs, esdf_D, esdf_H, esdf_W,
        tsdf_origin, tsdf_voxel_size, use_grid_bounds, grid_W, grid_H, grid_D,
    )
    if coords[3] == 0:
        return wp.float32(1e10)

    result = _hash_lookup_at_global_coords(tsdf, coords[0], coords[1], coords[2])
    if result[0] < 0:
        return wp.float32(1e10)

    return sample_static_sdf(tsdf, result[0], result[1])


@wp.kernel
def compute_esdf_from_min_tsdf_kernel(
    # ESDF buffers
    esdf_site_index: wp.array(dtype=wp.int32),
    esdf_voxel_size: wp.array(dtype=wp.float32),
    esdf_D: wp.int32,
    esdf_H: wp.int32,
    esdf_W: wp.int32,
    esdf_dist_field: wp.array(dtype=wp.float16),
    # Block-sparse TSDF for sign lookup
    tsdf: BlockSparseTSDFWarp,
    tsdf_origin: wp.array(dtype=wp.float32),
    tsdf_voxel_size: wp.float32,
    minimum_tsdf_weight: wp.float32,
    esdf_origin: wp.array(dtype=wp.float32),
    # Grid bounds for TSDF coord mapping
    use_grid_bounds: wp.int32,
    grid_W: wp.int32,
    grid_H: wp.int32,
    grid_D: wp.int32,
    # Parameters
    skip_steps: wp.float32,
):
    """Compute ESDF distances with sign from direct TSDF hash lookup.

    Grid layout: (nx, ny, nz) - X slowest, Z fastest. D=nx, H=ny, W=nz.

    Sign Determination (Adjacent Voxel Method):
    - The site is ON the surface (TSDF ~ 0, ambiguous sign)
    - Step from site towards query, sample TSDF via hash lookup
    - Falls back to query voxel's own TSDF if adjacent lookup misses
    """
    tid = wp.tid()

    esdf_z = tid % esdf_W
    esdf_y = (tid // esdf_W) % esdf_H
    esdf_x = tid // (esdf_W * esdf_H)

    if esdf_x >= esdf_D:
        return

    site_packed = esdf_site_index[tid]
    if site_packed < wp.int32(0):
        esdf_dist_field[tid] = wp.float16(1e4)
        return

    sx = site_packed & 0x3FF
    sy = (site_packed >> 10) & 0x3FF
    sz = (site_packed >> 20) & 0x3FF

    esdf_vs = esdf_voxel_size[0]
    dx = wp.float32(esdf_x - sx)
    dy = wp.float32(esdf_y - sy)
    dz = wp.float32(esdf_z - sz)
    dist_voxels = wp.sqrt(dx * dx + dy * dy + dz * dz)
    edt_dist = dist_voxels * esdf_vs

    # Sign via hash lookup.
    # Adjacent step uses STATIC-ONLY SDF because only watertight (primitive)
    # geometry has reliable sign when stepping away from the surface.
    # Fallback uses combined SDF at the query voxel itself.
    tsdf_sdf = wp.float32(1e10)

    if dist_voxels > 1.0 and skip_steps > 0.0:
        inv_dist = 1.0 / dist_voxels
        adj_x = sx + wp.int32(wp.round(dx * inv_dist * skip_steps))
        adj_y = sy + wp.int32(wp.round(dy * inv_dist * skip_steps))
        adj_z = sz + wp.int32(wp.round(dz * inv_dist * skip_steps))

        if adj_x >= 0 and adj_x < esdf_D and adj_y >= 0 and adj_y < esdf_H and adj_z >= 0 and adj_z < esdf_W:
            tsdf_sdf = lookup_static_sdf_at_esdf_coords(
                tsdf, adj_x, adj_y, adj_z,
                esdf_origin, esdf_vs, esdf_D, esdf_H, esdf_W,
                tsdf_origin, tsdf_voxel_size,
                use_grid_bounds, grid_W, grid_H, grid_D,
            )

    # Fallback: combined SDF at query voxel (both dynamic + static)
    if tsdf_sdf > 1e9:
        tsdf_sdf = lookup_combined_sdf_at_esdf_coords(
            tsdf, esdf_x, esdf_y, esdf_z,
            esdf_origin, esdf_vs, esdf_D, esdf_H, esdf_W,
            tsdf_origin, tsdf_voxel_size, minimum_tsdf_weight,
            use_grid_bounds, grid_W, grid_H, grid_D,
        )

    # No observation: assume outside
    if tsdf_sdf > 1e9:
        esdf_dist_field[tid] = wp.float16(edt_dist)
        return

    if tsdf_sdf < 0.0:
        edt_dist = -edt_dist

    esdf_dist_field[tid] = wp.float16(edt_dist)


def compute_esdf_from_min_tsdf_warp(
    esdf_site_index: torch.Tensor,
    tsdf: BlockSparseTSDF,
    esdf_voxel_size_tensor: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    esdf_dist_field: torch.Tensor,
    esdf_origin: torch.Tensor,
    adjacent_skip_steps: float = 0.0,
    minimum_tsdf_weight: float = 0.1,
    grid_shape=None,
) -> None:
    """Compute ESDF distances with sign from direct TSDF hash lookup.

    No pre-aggregated buffers needed. For each ESDF voxel, determines sign
    by doing a hash lookup into the block-sparse TSDF at the adjacent voxel
    position (one step from nearest site towards query).

    Args:
        esdf_site_index: Sites after PBA propagation (D, H, W) int32.
        tsdf: Block-sparse TSDF storage for sign lookups.
        esdf_voxel_size_tensor: ESDF voxel size (1,) float32 tensor.
        esdf_grid_shape: ESDF dimensions (D, H, W).
        esdf_dist_field: Output distance field (D, H, W) float16.
        esdf_origin: ESDF grid origin (3,) float32 tensor.
        adjacent_skip_steps: Steps from site towards query for sign lookup.
        minimum_tsdf_weight: Minimum weight for valid TSDF observation.
        grid_shape: Optional TSDF grid dimensions (nz, ny, nx) for bounds.
    """
    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    n_esdf_voxels = esdf_D * esdf_H * esdf_W
    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    warp_data = tsdf.get_warp_data()
    data = tsdf.data

    use_grid_bounds = 1 if grid_shape is not None else 0
    grid_D = grid_shape[0] if grid_shape else 0
    grid_H = grid_shape[1] if grid_shape else 0
    grid_W = grid_shape[2] if grid_shape else 0

    wp.launch(
        kernel=compute_esdf_from_min_tsdf_kernel,
        dim=n_esdf_voxels,
        inputs=[
            wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32),
            wp.from_torch(esdf_voxel_size_tensor, dtype=wp.float32),
            wp.int32(esdf_D),
            wp.int32(esdf_H),
            wp.int32(esdf_W),
            wp.from_torch(esdf_dist_field.view(-1), dtype=wp.float16),
            warp_data,
            wp.from_torch(data.origin.view(-1), dtype=wp.float32),
            wp.float32(data.voxel_size),
            wp.float32(minimum_tsdf_weight),
            wp.from_torch(esdf_origin, dtype=wp.float32),
            wp.int32(use_grid_bounds),
            wp.int32(grid_W),
            wp.int32(grid_H),
            wp.int32(grid_D),
            wp.float32(adjacent_skip_steps),
        ],
        device=wp.device_from_torch(device),
        stream=stream,
    )


def infer_inside_ray_parity_warp(
    esdf_site_index: torch.Tensor,
    esdf_min_tsdf: torch.Tensor,
    esdf_dist_field: torch.Tensor,
    esdf_grid_shape: Tuple[int, int, int],
    max_ray_distance_voxels: int = 5,
) -> None:
    """Infer inside/outside using ray parity test for voxels with no TSDF observation.

    Post-processing step after compute_esdf_from_min_tsdf_warp. Only processes
    voxels that have no TSDF observation and are close to a surface.

    Uses 6 axis-aligned rays and counts surface intersections. Odd = inside.
    Requires majority vote (4/6 rays) to flip sign, handles noise/artifacts.

    Args:
        esdf_site_index: Sites after PBA propagation (D, H, W) int32.
        esdf_min_tsdf: Min TSDF per ESDF voxel (D, H, W) float32.
        esdf_dist_field: Distance field to modify in-place (D, H, W) float16.
        esdf_grid_shape: ESDF dimensions (D, H, W).
        max_ray_distance_voxels: Only test voxels within this many voxels of surface.
    """
    esdf_D, esdf_H, esdf_W = esdf_grid_shape
    n_esdf_voxels = esdf_D * esdf_H * esdf_W

    device = esdf_site_index.device
    _, stream = get_warp_device_stream(esdf_site_index)

    wp.launch(
        kernel=infer_inside_ray_parity_kernel,
        dim=n_esdf_voxels,
        inputs=[
            wp.from_torch(esdf_site_index.view(-1), dtype=wp.int32),
            wp.from_torch(esdf_min_tsdf.view(-1), dtype=wp.float32),
            wp.from_torch(esdf_dist_field.view(-1), dtype=wp.float16),
            wp.int32(esdf_D),
            wp.int32(esdf_H),
            wp.int32(esdf_W),
            wp.int32(max_ray_distance_voxels),
        ],
        device=wp.device_from_torch(device),
        stream=stream,
    )

