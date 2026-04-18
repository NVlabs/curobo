/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * 3D Euclidean Distance Transform -- Parallel Banding Algorithm
 *
 * Computes an exact 3D Voronoi diagram on the GPU in three separable
 * phases (Z-flood, Y-Maurer, X-Maurer via transpose), requiring only
 * 5 kernel launches total (3 unique kernels).
 *
 * Phase 1 -- Z-axis flood:
 *     Bidirectional sweep (forward + backward) along each Z-column to
 *     find every voxel's nearest occupied site in the Z dimension.
 *
 * Phase 2 -- Y-axis Voronoi via Maurer stacks:
 *   (a) Build a compressed stack of Voronoi-dominant sites along each
 *       Y-column using Maurer's parabola intersection test.
 *   (b) Walk the stack to assign the nearest site to every voxel,
 *       writing results through shared-memory tiles with an X<->Y
 *       coordinate swap so Phase 3 can reuse the same kernels for X.
 *
 * Phase 3 -- X-axis Voronoi (reuses Phase 2 kernels on transposed data).
 *
 * Grid convention:
 *     CuRobo uses X-slowest / Z-fastest ordering.
 *     Kernels receive (sx, sy, sz) = (nz, ny, nx) so the internal
 *     Z-slowest convention maps directly onto memory layout without
 *     any data transposition.
 *
 * Algorithm reference:
 *     Cao Thanh Tung & Zheng Jiaqi, "Parallel Banding Algorithm to
 *     compute exact distance transform with the GPU", ACM SIGGRAPH
 *     2010 / PBA+ 2019.
 */

#pragma once

#include "site_encoding.cuh"

namespace curobo {
namespace parallel_banding {

// ============================================================================
// Kernel launch constants
// ============================================================================

constexpr int FLOOD_THREADS_X = 32;
constexpr int FLOOD_THREADS_Y = 4;
constexpr int TILE_DIM        = 32;

// ============================================================================
// Phase 1 -- Bidirectional Z-flood
//
// Each thread owns one (x, y) column and sweeps forward then backward
// along the Z dimension, propagating the nearest occupied site in 1D.
//
// After this phase every voxel stores its nearest site *along Z only*.
// Phases 2-3 extend the result to full 3D Voronoi via Maurer stacks.
//
// Launch:  grid (ceil(sx/32), ceil(sy/4)),  block (32, 4)
// ============================================================================

__global__ void kernel_flood_z(int *input, int *output,
                               int sx, int sy, int sz) {
    const int col_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int col_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_x >= sx || col_y >= sy) return;

    const int z_stride = sx * sy;
    int idx = voxel_index(col_x, col_y, 0, sx, sy);

    // ------- Forward sweep (z = 0 -> sz-1) -------
    // Carry the most recently seen site down the column.
    int propagated = encode_site(0, 0, 0, 1, 0);  // starts empty

    for (int z = 0; z < sz; z++, idx += z_stride) {
        const int current = input[idx];
        if (!is_empty_voxel(current))
            propagated = current;
        output[idx] = propagated;
    }

    // ------- Backward sweep (z = sz-2 -> 0) -------
    // Compare the forward result at each voxel with the backward-
    // propagated site and keep whichever is closer in the Z dimension.
    idx -= z_stride + z_stride;  // step back to z = sz-2

    for (int z = sz - 2; z >= 0; z--, idx -= z_stride) {
        const int back_z    = safe_z(propagated);
        const int dist_back = abs(back_z - z);

        const int fwd_entry = output[idx];
        const int fwd_z     = safe_z(fwd_entry);
        const int dist_fwd  = abs(fwd_z - z);

        if (dist_fwd < dist_back)
            propagated = fwd_entry;

        output[idx] = propagated;
    }
}

// ============================================================================
// Voronoi dominance test (Maurer's parabola intersection criterion)
//
// Three sites A, B, C lie along the sweep axis at rows ay < by < cy.
// Returns true when site B is *dominated*: the intersection of the
// parabolas d^2(A, .) and d^2(C, .) falls before row by, so B's
// Voronoi cell has zero extent and can be discarded from the stack.
//
// All arithmetic is in long long to avoid 32-bit overflow when grid
// coordinates approach 1023.
// ============================================================================

__device__ bool is_voronoi_dominated(
    long long ax, long long ay, long long az,   // site A (lower in stack)
    long long bx, long long by, long long bz,   // site B (candidate to pop)
    long long cx, long long cy, long long cz,   // site C (newly arriving)
    long long qx, long long qz) {               // query column (x, z)

    const long long dy_ab = by - ay;
    const long long dy_bc = cy - by;

    const long long lhs =
        ((ay + by) * dy_ab
         + (bx - ax) * (ax + bx - (qx << 1))
         + (bz - az) * (az + bz - (qz << 1))) * dy_bc;

    const long long rhs =
        ((by + cy) * dy_bc
         + (cx - bx) * (bx + cx - (qx << 1))
         + (cz - bz) * (bz + cz - (qz << 1))) * dy_ab;

    return lhs > rhs;
}

// ============================================================================
// Phase 2a / 3a -- Maurer stack construction along the Y axis
//
// Each thread handles one (x, z) column, sweeping along Y to build a
// compressed stack of Voronoi-dominant sites.  At each occupied voxel
// the new site is tested against the stack top: dominated entries are
// popped via is_voronoi_dominated(), then the new site is pushed.
//
// Stack storage is in-place: at row y the output encodes the site's
// (x, z) coordinates plus a y-link pointing to the previous stack
// entry's row.
//
// Launch:  grid (ceil(sx/32), ceil(sz/4)),  block (32, 4)
// ============================================================================

__global__ void kernel_maurer_axis(int *input, int *stack,
                                   int sx, int sy, int sz) {
    const int col_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int col_z = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_x >= sx || col_z >= sz) return;

    int idx = voxel_index(col_x, 0, col_z, sx, sy);

    // Stack state: top_entry is the current stack top, below_entry is
    // the entry immediately below it.  top_row tracks the y-row where
    // top_entry was pushed.
    int top_row     = 0;
    int top_entry   = encode_site(0, 0, 0, 1, 0);  // empty sentinel
    int below_entry = encode_site(0, 0, 0, 1, 0);
    int has_pred    = 0;  // becomes 1 after the first real site is pushed

    int voxel = encode_site(0, 0, 0, 1, 0);

    for (int y = 0; y < sy; ++y, idx += sx) {
        voxel = input[idx];

        if (!is_empty_voxel(voxel)) {
            // Pop dominated entries from the stack
            while (has_stack_below(top_entry)) {
                const SiteCoords below = decode_site(below_entry);
                const SiteCoords top   = decode_site(top_entry);
                const SiteCoords site  = decode_site(voxel);

                // below lives at row = top.y  (the y-link in top_entry)
                // top   lives at row = top_row
                // site  lives at row = y
                if (!is_voronoi_dominated(
                        below.x, top.y,   below.z,
                        top.x,   top_row, top.z,
                        site.x,  y,       site.z,
                        col_x,   col_z))
                    break;

                // Pop: promote below_entry to stack top
                top_row   = top.y;
                top_entry = below_entry;

                if (has_stack_below(top_entry))
                    below_entry = stack[voxel_index(
                        col_x, below.y, col_z, sx, sy)];
            }

            // Push the new site onto the stack
            const SiteCoords site = decode_site(voxel);
            below_entry = top_entry;
            top_entry   = encode_site(
                site.x, top_row, site.z, 0, has_pred);
            top_row     = y;

            stack[idx] = top_entry;
            has_pred   = 1;
        }
    }

    // If the last voxel in this column was empty, write a sentinel at
    // row sy-1 so the color pass can locate the stack top.
    if (is_empty_voxel(voxel))
        stack[voxel_index(col_x, sy - 1, col_z, sx, sy)] =
            encode_site(0, top_row, 0, 1, has_pred);
}

// ============================================================================
// Phase 2b / 3b -- Color axis with X<->Y transpose
//
// Each thread walks backward through the Maurer stack to assign the
// nearest site to every voxel along Y.  Results are written through
// shared-memory tiles with an X<->Y coordinate swap, so that Phase 3
// can reuse the same Maurer/Color kernels to process the X axis.
//
// Launch:  grid (ceil(sx/TILE_DIM), sz),  block (TILE_DIM, m3)
// ============================================================================

__global__ void kernel_color_axis(int *input, int *output,
                                  int sx, int sy, int sz) {
    __shared__ int tile[TILE_DIM][TILE_DIM];

    const int col   = threadIdx.x;
    const int tid   = threadIdx.y;
    const int col_x = blockIdx.x * blockDim.x + col;
    const int col_z = blockIdx.y;

    // Threads with col_x >= sx are out of bounds but must still
    // participate in __syncthreads(); guard with in_bounds.
    const bool in_bounds = (col_x < sx);

    // Stack traversal state: navigate from the top of the Maurer
    // stack downward, finding the nearest site for each row.
    int top_entry   = encode_site(0, 0, 0, 1, 0);
    int below_entry = encode_site(0, 0, 0, 1, 0);
    int top_row     = 0;

    // Decoded coordinates of the current stack top and the entry
    // below it.  top_site.y = row-link to the entry below;
    // below_site.y = link further down.
    SiteCoords top_site   = {0, 0, 0};
    SiteCoords below_site = {0, 0, 0};

    if (in_bounds) {
        // Start at the last row in the column
        top_row   = sy - 1;
        top_entry = input[voxel_index(col_x, top_row, col_z, sx, sy)];
        top_site  = decode_site(top_entry);

        // If the sentinel at sy-1 is empty, follow its y-link to
        // the actual stack top.
        if (is_empty_voxel(top_entry)) {
            top_row = top_site.y;
            if (has_stack_below(top_entry)) {
                top_entry = input[voxel_index(
                    col_x, top_row, col_z, sx, sy)];
                top_site = decode_site(top_entry);
            }
        }

        // Load the entry below the stack top (if one exists)
        if (has_stack_below(top_entry)) {
            below_entry = input[voxel_index(
                col_x, top_site.y, col_z, sx, sy)];
            below_site = decode_site(below_entry);
        }
    }

    // Process Y rows in TILE_DIM-wide chunks, from high to low
    const int n_tiles = (sy + TILE_DIM - 1) / TILE_DIM;

    for (int step = 0; step < n_tiles; ++step) {
        const int y_hi = sy - step * TILE_DIM - 1;
        const int y_lo = max(0, sy - (step + 1) * TILE_DIM);
        const int tile_height = y_hi - y_lo + 1;

        if (in_bounds) {
            // Each sub-thread handles rows strided by blockDim.y
            for (int y = y_hi - tid; y >= y_lo; y -= blockDim.y) {
                // Distance^2 to current stack-top site
                long long dx = top_site.x - col_x;
                long long dy = top_row - y;
                long long dz = top_site.z - col_z;
                long long min_dist_sq = dx * dx + dy * dy + dz * dz;

                // Walk the stack downward while the entry below
                // is closer than the current top
                while (has_stack_below(top_entry)) {
                    dx = below_site.x - col_x;
                    dy = top_site.y   - y;   // top_site.y = row of below
                    dz = below_site.z - col_z;
                    long long cand_sq = dx * dx + dy * dy + dz * dz;

                    if (cand_sq > min_dist_sq) break;

                    // Pop: below becomes new top
                    min_dist_sq = cand_sq;
                    top_row     = top_site.y;
                    top_entry   = below_entry;
                    top_site    = decode_site(top_entry);

                    if (has_stack_below(top_entry)) {
                        below_entry = input[voxel_index(
                            col_x, top_site.y, col_z, sx, sy)];
                        below_site = decode_site(below_entry);
                    }
                }

                // Write the X<->Y-transposed encoding into the tile.
                // Swap: output x-field <- top_row (y-position),
                //        output y-field <- top_site.x (original x)
                tile[col][y - y_lo] = encode_site(
                    top_row, top_site.x, top_site.z,
                    is_empty_voxel(top_entry), 0);
            }
        }

        __syncthreads();

        // Transposed write: swap X<->Y in the output buffer.
        // The output grid has dimensions (sy, sx, sz) instead
        // of (sx, sy, sz).  Inactive threads still hold valid
        // tile data and must participate in the write.
        if (!tid && col < tile_height) {
            int out_x   = y_lo + col;
            int out_idx = voxel_index(
                out_x, blockIdx.x * TILE_DIM, col_z, sy, sx);
            const int n_cols = min(
                TILE_DIM, sx - (int)(blockIdx.x * TILE_DIM));
            for (int i = 0; i < n_cols; i++, out_idx += sy) {
                output[out_idx] = tile[i][col];
            }
        }

        __syncthreads();
    }
}

}  // namespace parallel_banding
}  // namespace curobo
