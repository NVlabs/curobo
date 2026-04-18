/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Reuse joint type constants and enum from kinematics
#include "../kinematics/kinematics_constants.h"

namespace curobo {
namespace dynamics {

    // Spatial vector dimension (angular 3 + linear 3)
    constexpr int SPATIAL_DIM = 6;

    // Maximum batches per block for RNEA forward kernel (1 thread per batch)
    constexpr int MAX_RNEA_FW_BATCHES_PER_BLOCK = 128;

    // Maximum batches per block for RNEA backward kernel
    constexpr int MAX_RNEA_BW_BATCHES_PER_BLOCK = 128;

    // Shared memory floats per link per batch: forward
    // Layout: v(6) + a/f(6) = 12
    constexpr int SMEM_FLOATS_PER_LINK = 2 * SPATIAL_DIM;  // 12

    // Shared memory floats per link per batch: backward
    // Layout: v(6) + a(6) + f_bar(6) + a_bar(6) + v_bar(6) = 30
    // Note: sincos caching is NOT used in the backward kernel; the extra smem
    // pointer causes register spill, making it slower despite saving sincosf calls.
    constexpr int SMEM_FLOATS_PER_LINK_BWD = 5 * SPATIAL_DIM;  // 30

    // Forward cache: floats per link stored in global memory for backward reuse.
    // Layout (stride must be multiple of 4 for float4 alignment):
    //   [0:12]  v+a packed (3×float4, no padding)
    //   [12:20] f   padded (2×float4, 6 useful + 2 pad)
    // Total: 20 floats per link.
    //
    // R and p are NOT cached; they are recomputed from fixed_transforms + q
    // in the backward kernel via compute_local_Rp (~48 inst/call).  This avoids
    // 3×float4 global stores per link in the forward kernel's sequential Loop 1.
    constexpr int CACHE_FLOATS_PER_LINK = 20;
    constexpr int CACHE_VA_OFFSET = 0;
    constexpr int CACHE_F_OFFSET  = 12;

} // namespace dynamics
} // namespace curobo
