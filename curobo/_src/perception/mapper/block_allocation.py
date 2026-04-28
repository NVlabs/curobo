# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block allocation utilities for block-sparse TSDF.

This module provides heuristics for calculating the number of blocks
needed for a block-sparse TSDF based on grid dimensions and truncation settings.
"""

import math
from typing import Tuple

from curobo._src.perception.mapper.constants import (
    MAX_POOL_IDX as _MAX_POOL_IDX,
    REFERENCE_BLOCK_SIZE,
)
from curobo._src.util.logging import log_warn

# Minimum allocation floor expressed in voxels (not blocks), so the memory
# footprint of the floor is invariant to ``block_size``. Historically the
# floor was 10,000 blocks at block_size=REFERENCE_BLOCK_SIZE (=8); we
# preserve that total voxel budget and derive the block floor from it.
_MIN_VOXEL_FLOOR: int = 10_000 * REFERENCE_BLOCK_SIZE**3

# Fallback block budget used when neither ``max_blocks`` nor ``grid_shape``
# is available. Expressed in voxels so the allocated scratch pool is
# invariant to ``block_size``: historically this default was 100,000
# blocks at block_size=8. Without this scaling, dropping block_size
# (e.g. to 4) keeps the block count at 100K and silently shrinks the
# scratch pool by 8x, which saturates mid-frame and causes entire regions
# of the scene to fail to allocate.
_DEFAULT_VOXEL_BUDGET: int = 100_000 * REFERENCE_BLOCK_SIZE**3


def default_max_blocks(block_size: int) -> int:
    """Fallback ``max_blocks`` value scaled to preserve a BLOCK_SIZE-independent
    voxel budget.

    Use this helper in place of a hardcoded ``100_000`` when neither
    ``max_blocks`` nor ``grid_shape`` is known. The returned count grows as
    ``block_size`` shrinks so that the total voxel scratch space stays
    constant across block sizes, but is clamped to ``_MAX_POOL_IDX`` to
    avoid overflowing the packed hash entry's ``pool_idx`` field.
    table.

    Args:
        block_size: Voxels per block edge (e.g., 8).

    Returns:
        Default block count targeting ``_DEFAULT_VOXEL_BUDGET`` voxels total,
        capped at the active hash layout's pool_idx limit.
    """
    voxels_per_block = block_size**3
    scaled = max(1, _DEFAULT_VOXEL_BUDGET // voxels_per_block)
    return min(scaled, _MAX_POOL_IDX)


def calculate_tsdf_max_blocks(
    grid_shape: Tuple[int, int, int],
    voxel_size: float,
    block_size: int,
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
        block_size: Voxels per block edge (e.g., 8).
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
    block_size_meters = voxel_size * block_size

    # 2. Convert grid dimensions to "Block Units"
    nz, ny, nx = grid_shape
    bx = nx / block_size
    by = ny / block_size
    bz = nz / block_size

    # 3. Surface Area of Bounding Box (in Blocks)
    #    Base shell size (Floor + Ceiling + Walls)
    surface_area_blocks = 2.0 * (bx * by + bx * bz + by * bz)

    # 4. Calculate "Shell Thickness"
    #    How many blocks deep is the TSDF band?
    #    (+1 is for alignment safety, ensuring we capture straddling blocks)
    thickness_factor = math.ceil((2.0 * truncation_dist) / block_size_meters) + 1

    # 5. Combine: Area × Thickness × Complexity
    max_blocks = int(surface_area_blocks * thickness_factor * roughness)

    # Ensure a minimum reasonable size. The floor is defined in voxels so
    # total scratch memory stays roughly constant across BLOCK_SIZE choices.
    voxels_per_block = block_size**3
    min_blocks = max(1, _MIN_VOXEL_FLOOR // voxels_per_block)
    max_blocks = max(min_blocks, max_blocks)

    # Clamp to the active hash layout's pool_idx range. The all-ones
    # value is reserved as the PENDING sentinel during allocation.
    # Exceeding ``_MAX_POOL_IDX`` causes ``pack_entry`` to
    # alias real blocks onto the sentinel (making them permanently
    # invisible to ``hash_lookup``) or onto lower pool indices (silent
    # data corruption via shared slots). This is reached in practice at
    # block_size=2 on very large scenes where the surface-area heuristic
    # exceeds the active layout's finite pool.
    if max_blocks > _MAX_POOL_IDX:
        log_warn(
            f"calculate_tsdf_max_blocks: surface-area heuristic produced "
            f"{max_blocks:,} blocks, which exceeds the hash table's "
            f"pool_idx limit ({_MAX_POOL_IDX:,}). Clamping to "
            f"{_MAX_POOL_IDX:,}. This scene is under-provisioned at "
            f"block_size={block_size}; increase block_size (e.g. to 2, "
            f"4, or 8), reduce scene extent, reduce roughness, or "
            f"reduce truncation_distance to fit within the pool."
        )
        max_blocks = _MAX_POOL_IDX

    return max_blocks
