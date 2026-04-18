# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Canonical coordinate transforms for volumetric mapping.

This module provides the authoritative definitions for converting between
voxel indices and world coordinates. All code (Python, CUDA, Warp) should
use these definitions for consistency.

Coordinate Convention:
    - Grid shape: (nz, ny, nx) - Z slowest, X fastest (row-major)
    - Voxel index: (iz, iy, ix) - same order as grid shape
    - World coords: (x, y, z) - standard Cartesian
    - grid_center: world coordinate at center of grid

Example:
    >>> from curobo._src.perception.mapper.util.utils_coords import (
    ...     voxel_to_world, world_to_voxel, get_grid_bounds
    ... )
    >>> grid_shape = (100, 100, 100)  # (nz, ny, nx)
    >>> voxel_size = 0.01
    >>> grid_center = (0.0, 0.0, 0.5)  # (cx, cy, cz)
    >>>
    >>> # Convert voxel (50, 50, 50) to world coordinates
    >>> x, y, z = voxel_to_world(50, 50, 50, grid_center, grid_shape, voxel_size)
    >>> print(f"World: ({x:.3f}, {y:.3f}, {z:.3f})")
    World: (0.000, 0.000, 0.500)
"""

from typing import Tuple, Union

import torch


def voxel_to_world(
    iz: Union[int, float, torch.Tensor],
    iy: Union[int, float, torch.Tensor],
    ix: Union[int, float, torch.Tensor],
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[float, float, float]:
    """Convert voxel index to world coordinate at voxel center.

    This is the canonical definition used by all mapping code.

    Args:
        iz: Z index (0 to nz-1) - slowest varying
        iy: Y index (0 to ny-1)
        ix: X index (0 to nx-1) - fastest varying
        grid_center: (cx, cy, cz) world coordinate at grid center
        grid_shape: (nz, ny, nx) voxel counts
        voxel_size: voxel edge length in meters

    Returns:
        (world_x, world_y, world_z) coordinate at voxel center

    Example:
        >>> # Voxel (0, 0, 0) is at the corner, not the center
        >>> x, y, z = voxel_to_world(0, 0, 0, (0, 0, 0), (100, 100, 100), 0.01)
        >>> print(f"({x:.3f}, {y:.3f}, {z:.3f})")
        (-0.495, -0.495, -0.495)
    """
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center

    # Center voxel index would be ((nz-1)/2, (ny-1)/2, (nx-1)/2)
    # Offset from center: (ix - (nx-1)/2), etc.
    world_x = cx + (float(ix) - (nx - 1) / 2.0) * s
    world_y = cy + (float(iy) - (ny - 1) / 2.0) * s
    world_z = cz + (float(iz) - (nz - 1) / 2.0) * s

    return (world_x, world_y, world_z)


def world_to_voxel(
    world_x: float,
    world_y: float,
    world_z: float,
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[int, int, int]:
    """Convert world coordinate to nearest voxel index.

    This is the canonical definition used by all mapping code.
    Returns (-1, -1, -1) if the point is outside the grid bounds.

    Args:
        world_x: World X coordinate
        world_y: World Y coordinate
        world_z: World Z coordinate
        grid_center: (cx, cy, cz) world coordinate at grid center
        grid_shape: (nz, ny, nx) voxel counts
        voxel_size: voxel edge length in meters

    Returns:
        (iz, iy, ix) voxel indices, or (-1, -1, -1) if out of bounds

    Example:
        >>> iz, iy, ix = world_to_voxel(0.0, 0.0, 0.5, (0, 0, 0.5), (100, 100, 100), 0.01)
        >>> print(f"Voxel: ({iz}, {iy}, {ix})")
        Voxel: (50, 50, 50)
    """
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center

    # Inverse of voxel_to_world: ix = (x - cx) / s + (nx - 1) / 2
    fx = (world_x - cx) / s + (nx - 1) / 2.0
    fy = (world_y - cy) / s + (ny - 1) / 2.0
    fz = (world_z - cz) / s + (nz - 1) / 2.0

    # Round to nearest voxel
    ix = int(round(fx))
    iy = int(round(fy))
    iz = int(round(fz))

    # Bounds check
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        return (iz, iy, ix)
    return (-1, -1, -1)


def world_to_voxel_continuous(
    world_x: float,
    world_y: float,
    world_z: float,
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[float, float, float]:
    """Convert world coordinate to continuous voxel index (no rounding).

    Useful for trilinear interpolation or sub-voxel sampling.

    Args:
        world_x: World X coordinate
        world_y: World Y coordinate
        world_z: World Z coordinate
        grid_center: (cx, cy, cz) world coordinate at grid center
        grid_shape: (nz, ny, nx) voxel counts
        voxel_size: voxel edge length in meters

    Returns:
        (fz, fy, fx) continuous voxel indices
    """
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center

    fx = (world_x - cx) / s + (nx - 1) / 2.0
    fy = (world_y - cy) / s + (ny - 1) / 2.0
    fz = (world_z - cz) / s + (nz - 1) / 2.0

    return (fz, fy, fx)


def get_grid_bounds(
    grid_center: Tuple[float, float, float],
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Get world coordinates of grid bounds (outer edges of corner voxels).

    The bounds represent the physical extent of the grid, including the
    half-voxel padding around the outermost voxel centers.

    Args:
        grid_center: (cx, cy, cz) world coordinate at grid center
        grid_shape: (nz, ny, nx) voxel counts
        voxel_size: voxel edge length in meters

    Returns:
        (min_corner, max_corner) as ((x, y, z), (x, y, z))

    Example:
        >>> min_c, max_c = get_grid_bounds((0, 0, 0), (100, 100, 100), 0.01)
        >>> print(f"Min: {min_c}, Max: {max_c}")
        Min: (-0.5, -0.5, -0.5), Max: (0.5, 0.5, 0.5)
    """
    nz, ny, nx = grid_shape
    s = voxel_size
    cx, cy, cz = grid_center

    # Half-extents include the half-voxel padding around corner voxel centers
    half_extent_x = (nx - 1) / 2.0 * s + s / 2.0
    half_extent_y = (ny - 1) / 2.0 * s + s / 2.0
    half_extent_z = (nz - 1) / 2.0 * s + s / 2.0

    min_corner = (cx - half_extent_x, cy - half_extent_y, cz - half_extent_z)
    max_corner = (cx + half_extent_x, cy + half_extent_y, cz + half_extent_z)

    return (min_corner, max_corner)


def get_grid_extent(
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
) -> Tuple[float, float, float]:
    """Get physical extent of grid in meters.

    This is the total size of the grid including all voxels.

    Args:
        grid_shape: (nz, ny, nx) voxel counts
        voxel_size: voxel edge length in meters

    Returns:
        (extent_x, extent_y, extent_z) in meters
    """
    nz, ny, nx = grid_shape
    return (nx * voxel_size, ny * voxel_size, nz * voxel_size)

