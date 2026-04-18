/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file spatial_algebra.cuh
 * @brief 6D spatial algebra device functions for RNEA.
 *
 * Conventions (Featherstone):
 *   - Spatial vectors: [angular(3); linear(3)]
 *   - Spatial transform X = [E, 0; -E·skew(p), E] where E = R^T
 *   - Efficient multiply: X·[ω;v] = [E·ω; E·(v + ω×p)]
 *   - Force transpose:   X^T·f = [R·f_ang + p×(R·f_lin); R·f_lin]
 *   - Spatial inertia:   I = [Ic + m·cx^T·cx, m·cx; m·cx^T, m·I₃]
 */

#pragma once

#include "dynamics_constants.h"

namespace curobo {
namespace dynamics {

// ========================================================================
// Spatial transform operations
// ========================================================================

/**
 * @brief Spatial motion transform: v_out = X · v_in.
 *
 * X = [E, 0; -E·skew(p), E] where E = R^T.
 * Result: [E·ω; E·(v_lin + ω × p)]
 *
 * @param R Row-major 3×3 rotation (child axes in parent frame). E = R^T.
 * @param p Child origin in parent frame (3 floats).
 * @param v_in Input spatial motion vector [angular(3); linear(3)].
 * @param v_out Output spatial motion vector.
 */
__device__ __forceinline__ void spatial_Xv(
    const float *R, const float *p,
    const float *v_in,
    float *v_out
) {
    const float w0 = v_in[0], w1 = v_in[1], w2 = v_in[2];
    const float vl0 = v_in[3], vl1 = v_in[4], vl2 = v_in[5];

    // ω × p
    const float wxp0 = w1 * p[2] - w2 * p[1];
    const float wxp1 = w2 * p[0] - w0 * p[2];
    const float wxp2 = w0 * p[1] - w1 * p[0];

    // u = v_lin + ω × p
    const float u0 = vl0 + wxp0;
    const float u1 = vl1 + wxp1;
    const float u2 = vl2 + wxp2;

    // Top 3: R^T · ω (using column access pattern for transpose)
    v_out[0] = R[0] * w0 + R[3] * w1 + R[6] * w2;
    v_out[1] = R[1] * w0 + R[4] * w1 + R[7] * w2;
    v_out[2] = R[2] * w0 + R[5] * w1 + R[8] * w2;

    // Bottom 3: R^T · u
    v_out[3] = R[0] * u0 + R[3] * u1 + R[6] * u2;
    v_out[4] = R[1] * u0 + R[4] * u1 + R[7] * u2;
    v_out[5] = R[2] * u0 + R[5] * u1 + R[8] * u2;
}

/**
 * @brief Spatial force transpose transform: f_out = X^T · f_in.
 *
 * X^T = [R, skew(p)·R; 0, R]
 * Result: [R·f_ang + p × (R·f_lin); R·f_lin]
 *
 * @param R Row-major 3×3 rotation.
 * @param p Child origin in parent frame (3 floats).
 * @param f_in Input spatial force [torque(3); force(3)].
 * @param f_out Output spatial force.
 */
__device__ __forceinline__ void spatial_XTf(
    const float *R, const float *p,
    const float *f_in,
    float *f_out
) {
    const float n0 = f_in[0], n1 = f_in[1], n2 = f_in[2];
    const float fl0 = f_in[3], fl1 = f_in[4], fl2 = f_in[5];

    // R · f_lin
    const float Rfl0 = R[0] * fl0 + R[1] * fl1 + R[2] * fl2;
    const float Rfl1 = R[3] * fl0 + R[4] * fl1 + R[5] * fl2;
    const float Rfl2 = R[6] * fl0 + R[7] * fl1 + R[8] * fl2;

    // R · f_ang
    const float Rn0 = R[0] * n0 + R[1] * n1 + R[2] * n2;
    const float Rn1 = R[3] * n0 + R[4] * n1 + R[5] * n2;
    const float Rn2 = R[6] * n0 + R[7] * n1 + R[8] * n2;

    // p × (R · f_lin)
    const float pxRfl0 = p[1] * Rfl2 - p[2] * Rfl1;
    const float pxRfl1 = p[2] * Rfl0 - p[0] * Rfl2;
    const float pxRfl2 = p[0] * Rfl1 - p[1] * Rfl0;

    // Top: R·n + p × (R·f_lin)
    f_out[0] = Rn0 + pxRfl0;
    f_out[1] = Rn1 + pxRfl1;
    f_out[2] = Rn2 + pxRfl2;

    // Bottom: R · f_lin
    f_out[3] = Rfl0;
    f_out[4] = Rfl1;
    f_out[5] = Rfl2;
}

// ========================================================================
// Spatial cross product operations
//
// Note: switch/case is intentionally used here instead of branchless index
// arithmetic (result[i1] with runtime i1). On GPU, runtime indexing into
// local arrays forces them to stack memory (ld.local/st.local), which is
// much slower than the register-based access the compiler generates for
// compile-time constant indices in switch cases. The kinematics pattern
// (xyz_rot_fn) avoids this because it reads from global memory pointers
// and writes to fixed-index outputs.
// ========================================================================

/**
 * @brief Optimized motion cross with unit S vector: result = v ×_m (alpha · e_s).
 *
 * Since S has exactly one non-zero element, only 4 out of 6 result components
 * are non-zero. Uses switch/case for register-friendly compile-time indexing.
 *
 * @param v Spatial velocity [ω(3); v_lin(3)].
 * @param s_idx Index of the non-zero S element (0-5).
 * @param alpha Scalar multiplier (qd value).
 * @param result Output 6D vector (zeroed first).
 */
__device__ __forceinline__ void motion_cross_S(
    const float *v,
    int s_idx,
    float alpha,
    float *result
) {
    for (int i = 0; i < 6; i++) result[i] = 0.0f;

    const float w0 = v[0], w1 = v[1], w2 = v[2];
    const float v0 = v[3], v1 = v[4], v2 = v[5];

    switch (s_idx) {
        case 0: // revolute X
            result[1] =  w2 * alpha;
            result[2] = -w1 * alpha;
            result[4] =  v2 * alpha;
            result[5] = -v1 * alpha;
            break;
        case 1: // revolute Y
            result[0] = -w2 * alpha;
            result[2] =  w0 * alpha;
            result[3] = -v2 * alpha;
            result[5] =  v0 * alpha;
            break;
        case 2: // revolute Z
            result[0] =  w1 * alpha;
            result[1] = -w0 * alpha;
            result[3] =  v1 * alpha;
            result[4] = -v0 * alpha;
            break;
        case 3: // prismatic X
            result[1] =  w2 * alpha;
            result[2] = -w1 * alpha;
            break;
        case 4: // prismatic Y
            result[0] = -w2 * alpha;
            result[2] =  w0 * alpha;
            break;
        case 5: // prismatic Z
            result[0] =  w1 * alpha;
            result[1] = -w0 * alpha;
            break;
    }
}

/**
 * @brief Spatial force cross product: result = v ×* f = crf(v) · f.
 *
 * crf(v) · f = [ω × f_ang + v_lin × f_lin; ω × f_lin]
 *
 * @param v Spatial motion vector [ω(3); v_lin(3)].
 * @param f Spatial force vector [torque(3); force(3)].
 * @param result Output spatial force vector.
 */
__device__ __forceinline__ void force_cross(
    const float *v,
    const float *f,
    float *result
) {
    const float w0 = v[0], w1 = v[1], w2 = v[2];
    const float v0 = v[3], v1 = v[4], v2 = v[5];
    const float n0 = f[0], n1 = f[1], n2 = f[2];
    const float f0 = f[3], f1 = f[4], f2 = f[5];

    // Top: ω × n + v_lin × f_lin
    result[0] = -w2 * n1 + w1 * n2 - v2 * f1 + v1 * f2;
    result[1] =  w2 * n0 - w0 * n2 + v2 * f0 - v0 * f2;
    result[2] = -w1 * n0 + w0 * n1 - v1 * f0 + v0 * f1;

    // Bottom: ω × f_lin
    result[3] = -w2 * f1 + w1 * f2;
    result[4] =  w2 * f0 - w0 * f2;
    result[5] = -w1 * f0 + w0 * f1;
}

// ========================================================================
// Spatial inertia operations
// ========================================================================

/**
 * @brief Compute I · u on-the-fly from mass, CoM, and inertia.
 *
 * Spatial inertia:
 *   I = [Ic + m·cx^T·cx,  m·cx  ]
 *       [m·cx^T,           m·I₃  ]
 *
 * Efficient formula:
 *   h = v_lin + ω × c
 *   result_lin = m · h
 *   result_ang = Ic · ω + m · (c × h)
 *
 * @param mc  [4 floats]: cx, cy, cz, mass.
 * @param inertia [8 floats]: ixx, iyy, izz, ixy, ixz, iyz, pad0, pad1 (padded for float4).
 * @param u   [6 floats]: spatial vector [ω; v_lin].
 * @param result [6 floats]: I · u.
 */
__device__ __forceinline__ void spatial_inertia_times_vec(
    const float *mc,
    const float *inertia,
    const float *u,
    float *result
) {
    // Use float4 vectorized load for mc (4 floats = 16 bytes, aligned)
    // link_masses_com tensor is [n_links, 4], contiguous, so each row is 16-byte aligned
    const float4 mc_vec = *reinterpret_cast<const float4*>(mc);
    const float cx = mc_vec.x, cy = mc_vec.y, cz = mc_vec.z, m = mc_vec.w;
    // Use 2×float4 vectorized loads for inertia (8 floats = 32 bytes, aligned)
    // link_inertias tensor is [n_links, 8], contiguous, so each row is 32-byte aligned
    const float4 inertia_0 = *reinterpret_cast<const float4*>(&inertia[0]);  // ixx, iyy, izz, ixy
    const float4 inertia_1 = *reinterpret_cast<const float4*>(&inertia[4]);  // ixz, iyz, pad, pad
    const float ixx = inertia_0.x, iyy = inertia_0.y, izz = inertia_0.z, ixy = inertia_0.w;
    const float ixz = inertia_1.x, iyz = inertia_1.y;
    const float w0 = u[0], w1 = u[1], w2 = u[2];
    const float vl0 = u[3], vl1 = u[4], vl2 = u[5];

    // h = v_lin + ω × c
    const float h0 = vl0 + w1 * cz - w2 * cy;
    const float h1 = vl1 + w2 * cx - w0 * cz;
    const float h2 = vl2 + w0 * cy - w1 * cx;

    // Bottom 3: m · h
    result[3] = m * h0;
    result[4] = m * h1;
    result[5] = m * h2;

    // Ic · ω (symmetric 3×3)
    const float Iw0 = ixx * w0 + ixy * w1 + ixz * w2;
    const float Iw1 = ixy * w0 + iyy * w1 + iyz * w2;
    const float Iw2 = ixz * w0 + iyz * w1 + izz * w2;

    // c × h
    const float cxh0 = cy * h2 - cz * h1;
    const float cxh1 = cz * h0 - cx * h2;
    const float cxh2 = cx * h1 - cy * h0;

    // Top 3: Ic·ω + m·(c × h)
    result[0] = Iw0 + m * cxh0;
    result[1] = Iw1 + m * cxh1;
    result[2] = Iw2 + m * cxh2;
}

// ========================================================================
// Additional operations for RNEA backward (VJP)
// ========================================================================

/**
 * @brief Full spatial motion cross product: result = crm(a) · b.
 *
 * crm(a)·b = [a_ω × b_ω; a_v × b_ω + a_ω × b_v]
 *
 * @param a Spatial motion vector [ω(3); v_lin(3)].
 * @param b Spatial motion/force vector [3+3].
 * @param result Output 6D vector.
 */
__device__ __forceinline__ void motion_cross(
    const float *a, const float *b, float *result
) {
    const float aw0 = a[0], aw1 = a[1], aw2 = a[2];
    const float av0 = a[3], av1 = a[4], av2 = a[5];
    const float bw0 = b[0], bw1 = b[1], bw2 = b[2];
    const float bv0 = b[3], bv1 = b[4], bv2 = b[5];

    result[0] =  aw1 * bw2 - aw2 * bw1;
    result[1] =  aw2 * bw0 - aw0 * bw2;
    result[2] =  aw0 * bw1 - aw1 * bw0;
    result[3] =  av1 * bw2 - av2 * bw1 + aw1 * bv2 - aw2 * bv1;
    result[4] =  av2 * bw0 - av0 * bw2 + aw2 * bv0 - aw0 * bv2;
    result[5] =  av0 * bw1 - av1 * bw0 + aw0 * bv1 - aw1 * bv0;
}

/**
 * @brief Scalar: a^T · crf(e_s) · b, where e_s is the s_idx-th unit spatial vector.
 *
 * Uses switch/case: even though the pattern is axis-parameterizable, the pointer
 * arguments often point to caller's local arrays (e.g. X_fbar_p[6], f_k[6]),
 * and runtime indexing like a[i1] forces those arrays to local memory.
 *
 * Used in Pass 1 for: grad_q += mult · dot(X·f̄_parent, crf(S)·f[k])
 */
__device__ __forceinline__ float dot_crf_S(
    const float *a, const float *b, int s_idx
) {
    switch (s_idx) {
        case 0: return -a[1]*b[2] + a[2]*b[1] - a[4]*b[5] + a[5]*b[4];
        case 1: return  a[0]*b[2] - a[2]*b[0] + a[3]*b[5] - a[5]*b[3];
        case 2: return -a[0]*b[1] + a[1]*b[0] - a[3]*b[4] + a[4]*b[3];
        case 3: return -a[1]*b[5] + a[2]*b[4];
        case 4: return  a[0]*b[5] - a[2]*b[3];
        case 5: return -a[0]*b[4] + a[1]*b[3];
        default: return 0.0f;
    }
}

/**
 * @brief Scalar: a^T · crm(e_s) · b, where e_s is the s_idx-th unit spatial vector.
 *
 * Used in Pass 2 for: grad_q -= mult · dot(ā, crm(S)·X_a_parent)
 * and                  grad_q -= mult · dot(v̄, crm(S)·X_v_parent)
 */
__device__ __forceinline__ float dot_crm_S(
    const float *a, const float *b, int s_idx
) {
    switch (s_idx) {
        case 0: return -a[1]*b[2] + a[2]*b[1] - a[4]*b[5] + a[5]*b[4];
        case 1: return  a[0]*b[2] - a[2]*b[0] + a[3]*b[5] - a[5]*b[3];
        case 2: return -a[0]*b[1] + a[1]*b[0] - a[3]*b[4] + a[4]*b[3];
        case 3: return -a[4]*b[2] + a[5]*b[1];
        case 4: return  a[3]*b[2] - a[5]*b[0];
        case 5: return -a[3]*b[1] + a[4]*b[0];
        default: return 0.0f;
    }
}

/**
 * @brief Accumulate: result += crf(alpha · e_s) · b.
 *
 * Uses switch/case to keep compile-time constant indices for result[] writes,
 * avoiding local memory spill from runtime array indexing.
 *
 * Used for: v̄ += crf(S·qd) · ā
 */
__device__ __forceinline__ void force_cross_S_add(
    float *result, int s_idx, float alpha, const float *b
) {
    switch (s_idx) {
        case 0:
            result[1] -= alpha * b[2]; result[2] += alpha * b[1];
            result[4] -= alpha * b[5]; result[5] += alpha * b[4];
            break;
        case 1:
            result[0] += alpha * b[2]; result[2] -= alpha * b[0];
            result[3] += alpha * b[5]; result[5] -= alpha * b[3];
            break;
        case 2:
            result[0] -= alpha * b[1]; result[1] += alpha * b[0];
            result[3] -= alpha * b[4]; result[4] += alpha * b[3];
            break;
        case 3:
            result[1] -= alpha * b[5]; result[2] += alpha * b[4];
            break;
        case 4:
            result[0] += alpha * b[5]; result[2] -= alpha * b[3];
            break;
        case 5:
            result[0] -= alpha * b[4]; result[1] += alpha * b[3];
            break;
    }
}

/**
 * @brief Extract element s_idx of crf(v)·f, i.e. (v ×* f)[s_idx].
 *
 * Used for: grad_qd += multiplier · (−crf(v)·ā)[s_idx]
 */
__device__ __forceinline__ float force_cross_extract_s(
    const float *v, const float *f, int s_idx
) {
    const float w0 = v[0], w1 = v[1], w2 = v[2];
    const float v0 = v[3], v1 = v[4], v2 = v[5];
    const float n0 = f[0], n1 = f[1], n2 = f[2];
    const float f0 = f[3], f1 = f[4], f2 = f[5];

    switch (s_idx) {
        case 0: return -w2*n1 + w1*n2 - v2*f1 + v1*f2;
        case 1: return  w2*n0 - w0*n2 + v2*f0 - v0*f2;
        case 2: return -w1*n0 + w0*n1 - v1*f0 + v0*f1;
        case 3: return -w2*f1 + w1*f2;
        case 4: return  w2*f0 - w0*f2;
        case 5: return -w1*f0 + w0*f1;
        default: return 0.0f;
    }
}

} // namespace dynamics
} // namespace curobo
