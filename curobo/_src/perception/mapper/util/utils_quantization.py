# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantization and packing utilities for ESDF integrator.

Provides Warp functions and PyTorch utilities for:
1. Site index packing (3 × 10-bit coordinates in int32)
2. Float16 grid SDF/weight extraction

Site index packing (for PBA):
- 3 × 10-bit coordinates packed into int32
- Supports grids up to 1024³
- Bit layout: | 2 unused | 10 bits Z | 10 bits Y | 10 bits X |
- Faster decode than flat index (bit ops vs division)

Usage in Warp kernels::

    from curobo._src.perception.mapper.util.utils_quantization import (
        pack_site_coords,
        unpack_site_x,
        unpack_site_y,
        unpack_site_z,
    )
"""

import torch
import warp as wp

# =============================================================================
# Constants for Site Index Packing (3 × 10-bit coordinates)
# =============================================================================

# Bit layout: | 2 unused | 10 bits Z | 10 bits Y | 10 bits X |
SITE_COORD_BITS = 10
SITE_COORD_MASK = (1 << SITE_COORD_BITS) - 1  # 0x3FF = 1023


# =============================================================================
# Warp Functions: Site Index Packing (3 × 10-bit coordinates)
# =============================================================================


@wp.func
def pack_site_coords(x: wp.int32, y: wp.int32, z: wp.int32) -> wp.int32:
    """Pack 3D voxel coordinates into a single int32.

    Bit layout: | 2 unused | 10 bits Z | 10 bits Y | 10 bits X |

    Args:
        x, y, z: Voxel coordinates (0-1023 each).

    Returns:
        Packed int32 with all three coordinates.
    """
    return (z << 20) | (y << 10) | x


@wp.func
def unpack_site_x(packed: wp.int32) -> wp.int32:
    """Extract X coordinate from packed site index."""
    return packed & wp.int32(0x3FF)


@wp.func
def unpack_site_y(packed: wp.int32) -> wp.int32:
    """Extract Y coordinate from packed site index."""
    return (packed >> 10) & wp.int32(0x3FF)


@wp.func
def unpack_site_z(packed: wp.int32) -> wp.int32:
    """Extract Z coordinate from packed site index."""
    return (packed >> 20) & wp.int32(0x3FF)


# =============================================================================
# Float16 Grid Utilities (for separate sdf_weight/weight format)
# =============================================================================


@wp.func
def get_sdf_from_float16_grids(
    sdf_weight: wp.float16,
    weight: wp.float16,
) -> wp.float32:
    """Compute SDF from separate sdf_weight and weight values.

    Args:
        sdf_weight: Accumulated sum of (sdf * weight).
        weight: Accumulated sum of weights.

    Returns:
        Fused SDF value, or large value if unobserved.
    """
    if weight <= wp.float16(0.0):
        return 1e10
    return wp.float32(sdf_weight) / wp.float32(weight)


@wp.func
def get_sdf_from_float16_grids(
    sdf_weight: wp.float16,
    weight: wp.float32,
) -> wp.float32:
    """Compute SDF from separate sdf_weight and weight values.

    Args:
        sdf_weight: Accumulated sum of (sdf * weight).
        weight: Accumulated sum of weights.

    Returns:
        Fused SDF value, or large value if unobserved.
    """
    if weight <= 0.0:
        return 1e10
    return wp.float32(sdf_weight) / weight


@wp.func
def get_weight_from_float16(weight: wp.float16) -> wp.float32:
    """Convert float16 weight to float32.

    Args:
        weight: Float16 weight value.

    Returns:
        Float32 weight value.
    """
    return wp.float32(weight)


# =============================================================================
# PyTorch Functions: Site Index Unpacking
# =============================================================================


def unpack_site_coords_torch(packed: torch.Tensor) -> tuple:
    """Unpack coordinates from packed site index tensor.

    Args:
        packed: Packed int32 tensor.

    Returns:
        Tuple of (x, y, z) coordinate tensors.
    """
    x = packed & 0x3FF
    y = (packed >> 10) & 0x3FF
    z = (packed >> 20) & 0x3FF
    return x, y, z
