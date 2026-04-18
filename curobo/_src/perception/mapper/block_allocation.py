# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block allocation utilities for block-sparse TSDF.

This module provides heuristics for calculating the number of blocks
needed for a block-sparse TSDF based on grid dimensions and truncation settings.
"""

import math
from typing import Tuple


def calculate_tsdf_max_blocks(
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
    block_dim: int,
    truncation_dist: float,
    roughness: float = 3.0,
) -> int:
    """Calculate max blocks for block-sparse TSDF using surface area heuristic.

    TSDF only allocates blocks near surfaces (not throughout the volume).
    The surface band has thickness = 2 × truncation_distance.

    Formula:
        max_blocks = surface_area_blocks × thickness_factor × roughness

    This scales O(N²) with grid size, not O(N³), matching actual TSDF memory usage.

    Args:
        grid_shape: (nz, ny, nx) dense voxel dimensions.
        voxel_size: Meters per voxel (e.g., 0.005).
        block_dim: Voxels per block edge (e.g., 8).
        truncation_dist: TSDF truncation distance in meters.
        roughness: Complexity multiplier (1.0 = simple walls, 3.0 = cluttered room).

    Returns:
        Estimated maximum number of blocks needed.

    Example:
        >>> grid = (400, 400, 200)  # nz, ny, nx
        >>> max_blocks = calculate_tsdf_max_blocks(grid, 0.005, 8, 0.04, 3.0)
        >>> print(f"{max_blocks:,} blocks")
        90,000 blocks
    """
    # 1. Physical size of one block in meters
    block_size_meters = voxel_size * block_dim

    # 2. Convert grid dimensions to "Block Units"
    nz, ny, nx = grid_shape
    bx = nx / block_dim
    by = ny / block_dim
    bz = nz / block_dim

    # 3. Surface Area of Bounding Box (in Blocks)
    #    Base shell size (Floor + Ceiling + Walls)
    surface_area_blocks = 2.0 * (bx * by + bx * bz + by * bz)

    # 4. Calculate "Shell Thickness"
    #    How many blocks deep is the TSDF band?
    #    (+1 is for alignment safety, ensuring we capture straddling blocks)
    thickness_factor = math.ceil((2.0 * truncation_dist) / block_size_meters) + 1

    # 5. Combine: Area × Thickness × Complexity
    max_blocks = int(surface_area_blocks * thickness_factor * roughness)

    # Ensure minimum reasonable size
    return max(10000, max_blocks)

