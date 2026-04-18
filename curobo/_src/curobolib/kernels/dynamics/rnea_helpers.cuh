/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file rnea_helpers.cuh
 * @brief RNEA helper device functions: local transform computation, joint helpers.
 *
 * Memory layout for fixed_transforms (same as FK kernel):
 *   - Per-link: 12 floats in row-major 3×4: [R | p]
 *   - ft[row * 4 + col]: R[row][col] for col < 3, p[row] for col == 3
 */

#pragma once

#include "dynamics_constants.h"

namespace curobo {
namespace dynamics {

/**
 * @brief Map joint type to S-vector index (which element of the 6D S is non-zero).
 *
 * @param j_type Joint type enum value.
 * @return Index into the 6D spatial vector (0-5), or -1 for FIXED.
 */
__device__ __forceinline__ int get_s_index(int8_t j_type) {
    if (j_type >= X_ROT) {
        return j_type - X_ROT;   // revolute: 0, 1, 2 (angular)
    }
    return 3 + j_type;           // prismatic: 3, 4, 5 (linear)
}

/**
 * @brief Compute local rotation R and translation p from fixed_transform
 *        and effective joint angle.
 *
 * For FIXED joints: R = R_fixed, p = p_fixed.
 * For revolute joints: R = R_fixed · R_axis(θ), p = p_fixed.
 * For prismatic joints: R = R_fixed, p = p_fixed + R_fixed · (axis · θ).
 *
 * Uses float4 vectorized loads for the 3×4 fixed_transform matrix.
 *
 * Note: The axis dispatch uses if/else instead of branchless index arithmetic
 * (R[a], r0[b]) because runtime indexing into R[9] and local arrays forces the
 * compiler to spill them to local memory. The kinematics xyz_rot_fn pattern
 * works because it reads from pointer args and writes to fixed-index outputs.
 *
 * @param ft     Pointer to this link's fixed_transform (12 floats, row-major 3×4).
 * @param j_type Joint type enum.
 * @param angle  Effective joint angle/displacement (after mimic transform).
 * @param R      Output 9 floats (row-major 3×3).
 * @param p      Output 3 floats.
 */
__device__ __forceinline__ void compute_local_Rp(
    const float *ft,
    int8_t j_type,
    float angle,
    float *R,
    float *p
) {
    // Use float4 vectorized loads for the 3×4 fixed_transform
    // Layout: [R00,R01,R02,p0, R10,R11,R12,p1, R20,R21,R22,p2]
    const float4 row0 = *reinterpret_cast<const float4*>(&ft[0]);
    const float4 row1 = *reinterpret_cast<const float4*>(&ft[4]);
    const float4 row2 = *reinterpret_cast<const float4*>(&ft[8]);

    // Extract p_fixed (translation column = .w components)
    p[0] = row0.w;
    p[1] = row1.w;
    p[2] = row2.w;

    if (j_type == FIXED) {
        // R = R_fixed
        R[0] = row0.x; R[1] = row0.y; R[2] = row0.z;
        R[3] = row1.x; R[4] = row1.y; R[5] = row1.z;
        R[6] = row2.x; R[7] = row2.y; R[8] = row2.z;
        return;
    }

    if (j_type >= X_ROT) {
        // Revolute joint: R = R_fixed · R_axis(θ)
        float s, c;
        sincosf(angle, &s, &c);
        const int axis = j_type - X_ROT;

        if (axis == 0) {
            // R_joint(X) = [[1,0,0],[0,c,-s],[0,s,c]]
            R[0] = row0.x; R[1] = c * row0.y + s * row0.z; R[2] = -s * row0.y + c * row0.z;
            R[3] = row1.x; R[4] = c * row1.y + s * row1.z; R[5] = -s * row1.y + c * row1.z;
            R[6] = row2.x; R[7] = c * row2.y + s * row2.z; R[8] = -s * row2.y + c * row2.z;
        } else if (axis == 1) {
            // R_joint(Y) = [[c,0,s],[0,1,0],[-s,0,c]]
            R[0] = c * row0.x - s * row0.z; R[1] = row0.y; R[2] = s * row0.x + c * row0.z;
            R[3] = c * row1.x - s * row1.z; R[4] = row1.y; R[5] = s * row1.x + c * row1.z;
            R[6] = c * row2.x - s * row2.z; R[7] = row2.y; R[8] = s * row2.x + c * row2.z;
        } else {
            // R_joint(Z) = [[c,-s,0],[s,c,0],[0,0,1]]
            R[0] = c * row0.x + s * row0.y; R[1] = -s * row0.x + c * row0.y; R[2] = row0.z;
            R[3] = c * row1.x + s * row1.y; R[4] = -s * row1.x + c * row1.y; R[5] = row1.z;
            R[6] = c * row2.x + s * row2.y; R[7] = -s * row2.x + c * row2.y; R[8] = row2.z;
        }
    } else {
        // Prismatic joint: R = R_fixed, p = p_fixed + R_fixed · (axis · θ)
        R[0] = row0.x; R[1] = row0.y; R[2] = row0.z;
        R[3] = row1.x; R[4] = row1.y; R[5] = row1.z;
        R[6] = row2.x; R[7] = row2.y; R[8] = row2.z;

        // p += R_fixed[:, axis] * angle
        const int axis = j_type;  // 0=X, 1=Y, 2=Z
        if (axis == 0) {
            p[0] += row0.x * angle;
            p[1] += row1.x * angle;
            p[2] += row2.x * angle;
        } else if (axis == 1) {
            p[0] += row0.y * angle;
            p[1] += row1.y * angle;
            p[2] += row2.y * angle;
        } else {
            p[0] += row0.z * angle;
            p[1] += row1.z * angle;
            p[2] += row2.z * angle;
        }
    }
}

// ========================================================================
// Forward cache save/load helpers
//
// Cache layout per link (CACHE_FLOATS_PER_LINK = 20):
//   [0:12]  v+a packed  (3×float4, no padding)
//   [12:20] f padded    (2×float4, 6 useful + 2 pad)
//
// R and p are NOT cached; recomputed via compute_local_Rp in backward.
// This avoids 3×float4 global stores per link in the forward sequential loop.
//
// Stride of 20 is a multiple of 4, so all float4 accesses are aligned
// when the base pointer is 16-byte aligned.
// ========================================================================

/**
 * @brief Save v[6]+a[6] packed as 3×float4 (12 floats, no padding).
 *
 * Layout: {v0,v1,v2,v3}, {v4,v5,a0,a1}, {a2,a3,a4,a5}
 *
 * @param v   Pointer to 6-float velocity in shared memory.
 * @param a   Pointer to 6-float acceleration in shared memory.
 * @param dst Pointer to 12-float slot in global memory (16-byte aligned).
 */
__device__ __forceinline__ void cache_save_va(
    const float *v, const float *a, float *dst
) {
    *reinterpret_cast<float4*>(&dst[0]) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(&dst[4]) = make_float4(v[4], v[5], a[0], a[1]);
    *reinterpret_cast<float4*>(&dst[8]) = make_float4(a[2], a[3], a[4], a[5]);
}

/**
 * @brief Load v[6]+a[6] from cache into shared memory via 3×float4.
 *
 * @param src Pointer to 12-float slot in global memory (16-byte aligned).
 * @param v   Pointer to 6-float velocity in shared memory.
 * @param a   Pointer to 6-float acceleration in shared memory.
 */
__device__ __forceinline__ void cache_load_va(
    const float *src, float *v, float *a
) {
    const float4 r0 = *reinterpret_cast<const float4*>(&src[0]);
    const float4 r1 = *reinterpret_cast<const float4*>(&src[4]);
    const float4 r2 = *reinterpret_cast<const float4*>(&src[8]);
    v[0] = r0.x; v[1] = r0.y; v[2] = r0.z; v[3] = r0.w;
    v[4] = r1.x; v[5] = r1.y;
    a[0] = r1.z; a[1] = r1.w;
    a[2] = r2.x; a[3] = r2.y; a[4] = r2.z; a[5] = r2.w;
}

/**
 * @brief Save f[6] to cache as 2×float4 (padded to 8, last 2 zeros).
 *
 * @param src Pointer to 6-float force in shared memory.
 * @param dst Pointer to 8-float slot in global memory (16-byte aligned).
 */
__device__ __forceinline__ void cache_save_f(
    const float *src, float *dst
) {
    *reinterpret_cast<float4*>(&dst[0]) = make_float4(src[0], src[1], src[2], src[3]);
    *reinterpret_cast<float4*>(&dst[4]) = make_float4(src[4], src[5], 0.0f, 0.0f);
}

/**
 * @brief Load f[6] from cache into shared memory via 2×float4.
 *
 * @param src Pointer to 8-float slot in global memory (16-byte aligned).
 * @param dst Pointer to 6-float force in shared memory.
 */
__device__ __forceinline__ void cache_load_f(
    const float *src, float *dst
) {
    const float4 r0 = *reinterpret_cast<const float4*>(&src[0]);
    const float4 r1 = *reinterpret_cast<const float4*>(&src[4]);
    dst[0] = r0.x; dst[1] = r0.y; dst[2] = r0.z; dst[3] = r0.w;
    dst[4] = r1.x; dst[5] = r1.y;
}

} // namespace dynamics
} // namespace curobo
