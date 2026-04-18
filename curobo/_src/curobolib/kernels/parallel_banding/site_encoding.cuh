/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Packed site encoding for 3D voxel grids.
 *
 * Packs three 10-bit coordinates and two 1-bit flags into a single
 * 32-bit integer, used by the PBA distance-transform kernels.
 *
 * Layout:
 *     bit  31      : empty flag   (1 = no site at this voxel)
 *     bit  30      : stack flag   (1 = predecessor exists in Maurer stack)
 *     bits 29-20   : x coordinate (10 bits, range 0-1023)
 *     bits 19-10   : y coordinate (10 bits; doubles as a row-link in
 *                    Maurer stack entries)
 *     bits  9- 0   : z coordinate (10 bits, range 0-1023)
 */

#pragma once

namespace curobo {
namespace parallel_banding {

// ============================================================================
// Constants
// ============================================================================

constexpr int EMPTY_VOXEL    = -2147483647 - 1;  // 0x80000000 sentinel
constexpr int MAX_COORD_DIST = 2147483647;        // guarantees empty loses
constexpr int COORD_BITS     = 10;
constexpr int COORD_MASK     = 0x3ff;             // (1 << 10) - 1

// ============================================================================
// Decoded site coordinates
// ============================================================================

/// Decoded (x, y, z) triple from a packed 32-bit voxel.
/// For Maurer stack entries the y field stores a row-link to the
/// predecessor, not the site's own Y coordinate.
struct SiteCoords {
    int x;
    int y;
    int z;
};

// ============================================================================
// Encoding / decoding
// ============================================================================

/// Pack three 10-bit coordinates and two 1-bit flags into a 32-bit int.
__device__ __forceinline__
int encode_site(int x, int y, int z, int empty_flag, int stack_flag) {
    return (x << 20) | (y << 10) | z
         | (empty_flag << 31) | (stack_flag << 30);
}

/// Unpack the three coordinate fields from a packed voxel.
__device__ __forceinline__
SiteCoords decode_site(int packed) {
    return {
        (packed >> 20) & COORD_MASK,
        (packed >> 10) & COORD_MASK,
         packed        & COORD_MASK,
    };
}

/// True when the packed value represents an empty (non-site) voxel.
__device__ __forceinline__ bool is_empty_voxel(int packed) {
    return (packed >> 31) & 1;
}

/// True when a Maurer stack entry has a predecessor below it.
__device__ __forceinline__ bool has_stack_below(int packed) {
    return (packed >> 30) & 1;
}

/// Z-coordinate of a site, returning MAX_COORD_DIST for empty voxels
/// so that empty entries always lose the distance comparison in the
/// Z-flood pass.
__device__ __forceinline__ int safe_z(int packed) {
    return is_empty_voxel(packed) ? MAX_COORD_DIST : (packed & COORD_MASK);
}

// ============================================================================
// Grid indexing
// ============================================================================

/// Row-major linear index: z-slowest, x-fastest.
__device__ __forceinline__
int voxel_index(int x, int y, int z, int stride_x, int stride_y) {
    return (z * stride_y + y) * stride_x + x;
}

}  // namespace parallel_banding
}  // namespace curobo
