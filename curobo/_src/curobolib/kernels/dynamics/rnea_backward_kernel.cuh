/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file rnea_backward_kernel.cuh
 * @brief RNEA backward pass (VJP) CUDA kernel with tree-level parallelism.
 *
 * Given upstream gradient dL/dτ ("tau_bar"), computes dL/dq, dL/dqd, dL/dqdd.
 * When HAS_EXTERNAL_FORCES=true, also computes dL/df_ext.
 *
 * Algorithm (reverse-mode AD of RNEA, O(n) per batch):
 *   Inertia load: Load mc, inertia into block-shared memory (one copy per block)
 *   Cache load: Load v, a, f from forward_cache into shared memory (replaces
 *               forward recomputation; data saved by forward kernel).
 *   Phase 2: Pass 1 (root→leaf), adjoint of RNEA backward sweep → f_bar
 *   Phase 2.5: (if HAS_EXTERNAL_FORCES) Write grad_f_ext = -f_bar
 *   Phase 3: Zero a_bar, v_bar accumulators
 *   Phase 4: Pass 2 (leaf→root), adjoint of RNEA forward sweep → a_bar, v_bar
 *
 * Threading: TPB threads per batch element. When TPB=1, degenerates to serial.
 * Links at the same tree depth are processed in parallel by different workers.
 *
 * Shared memory layout:
 *   Block-shared (one copy per block): N_LINKS × 12 floats = mc(4) + inertia(8)
 *   Per-batch: N_LINKS × 30 floats = v(6) + a(6) + f_bar(6) + a_bar(6) + v_bar(6)
 *
 * The block-shared inertia caching eliminates 80% of global loads for inertia
 * params (loaded once per block instead of 5× per link per batch).
 *
 * R and p transforms are recomputed from fixed_transforms + q via
 * compute_local_Rp (~48 inst/call). This avoids 3×float4 global stores
 * per link in the forward kernel's sequential Loop 1.
 *
 * Forward cache layout: [batch_size, N_LINKS, 20] floats.
 *   Per link: v+a packed (12) + f padded (8).
 *
 * Template params:
 *   - N_LINKS, N_DOF: Compile-time smem layout, loop unrolling
 *   - TPB: Threads per batch (1=serial, >1=tree-parallel)
 *   - HAS_EXTERNAL_FORCES: When true, computes grad_f_ext = -f_bar
 */

#pragma once

#include "spatial_algebra.cuh"
#include "rnea_helpers.cuh"

namespace curobo {
namespace dynamics {

/**
 * @brief RNEA backward (VJP) kernel: computes grad_q, grad_qd, grad_qdd.
 *
 * @tparam N_LINKS Compile-time link count (must be > 0).
 * @tparam N_DOF   Compile-time DOF count (must be > 0).
 * @tparam TPB     Threads per batch element (must be power of 2, 1..32).
 * @tparam HAS_EXTERNAL_FORCES When true, computes grad_f_ext = -f_bar.
 *
 * @param grad_f_ext [batch_size, N_LINKS, 6] Gradient w.r.t. external forces.
 *                   Only written when HAS_EXTERNAL_FORCES=true.
 *                   Set to nullptr when HAS_EXTERNAL_FORCES=false.
 */
template<int N_LINKS, int N_DOF, int TPB = 1, bool HAS_EXTERNAL_FORCES = false>
__global__ void rnea_backward_kernel(
    float * __restrict__ grad_q,
    float * __restrict__ grad_qd,
    float * __restrict__ grad_qdd,
    float * __restrict__ grad_f_ext,  // Gradient w.r.t. external forces (or nullptr)
    const float * __restrict__ grad_tau,
    const float * __restrict__ q,
    const float * __restrict__ qd,
    const float * __restrict__ fixed_transforms,
    const float * __restrict__ link_masses_com,
    const float * __restrict__ link_inertias,
    const int8_t * __restrict__ joint_map_type,
    const int16_t * __restrict__ joint_map,
    const int16_t * __restrict__ link_map,
    const float * __restrict__ joint_offset_map,
    const float * __restrict__ gravity,
    const int16_t * __restrict__ level_starts,
    const int16_t * __restrict__ level_links,
    const float * __restrict__ forward_cache,
    const int batch_size,
    const int n_levels
) {
    static_assert(N_LINKS > 0, "N_LINKS must be > 0");
    static_assert(N_DOF > 0, "N_DOF must be > 0");

    const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int batch = global_tid / TPB;
    const int worker = (TPB > 1) ? (global_tid % TPB) : 0;

    if (TPB == 1 && batch >= batch_size) return;

    const bool active = (batch < batch_size);

    // ------------------------------------------------------------------
    // Shared memory layout (compile-time offsets via N_LINKS):
    //
    // Block-shared (robot params, same for all batches):
    //   s_mc:      N_LINKS × 4  (mass + com xyz)
    //   s_inertia: N_LINKS × 8  (inertia tensor, packed)
    //
    // Per-batch:
    //   sv:     N_LINKS × 6  (velocity)
    //   sa:     N_LINKS × 6  (acceleration)
    //   sf:     N_LINKS × 6  (force → f_bar)
    //   sa_bar: N_LINKS × 6  (acceleration adjoint)
    //   sv_bar: N_LINKS × 6  (velocity adjoint)
    // ------------------------------------------------------------------
    constexpr int SMEM_BLOCK_SHARED = N_LINKS * 12;  // mc(4) + inertia(8) per link
    constexpr int SMEM_PER_BATCH = N_LINKS * SMEM_FLOATS_PER_LINK_BWD;
    extern __shared__ __align__(16) float smem[];

    // Block-shared inertia params (one copy for entire block)
    float *s_mc      = &smem[0];
    float *s_inertia = &smem[N_LINKS * 4];

    // Per-batch arrays (offset past block-shared region)
    const int local_batch = threadIdx.x / TPB;
    float *sv     = &smem[SMEM_BLOCK_SHARED + local_batch * SMEM_PER_BATCH];
    float *sa     = &smem[SMEM_BLOCK_SHARED + local_batch * SMEM_PER_BATCH + N_LINKS * 6];
    float *sf     = &smem[SMEM_BLOCK_SHARED + local_batch * SMEM_PER_BATCH + N_LINKS * 12];
    float *sa_bar = &smem[SMEM_BLOCK_SHARED + local_batch * SMEM_PER_BATCH + N_LINKS * 18];
    float *sv_bar = &smem[SMEM_BLOCK_SHARED + local_batch * SMEM_PER_BATCH + N_LINKS * 24];

    const int q_base = batch * N_DOF;

    // Warp mask for intra-batch synchronization
    unsigned batch_mask = 0;
    if (TPB > 1) {
        constexpr unsigned tpb_mask = (TPB >= 32) ? 0xFFFFFFFFu : ((1u << TPB) - 1u);
        const int lane = threadIdx.x & 31;
        batch_mask = tpb_mask << ((lane / TPB) * TPB);
    }

    // NOTE: grad_q/grad_qd/grad_qdd are pre-zeroed by the caller via
    // cudaMemsetAsync (tensor.zero_()), which is faster than per-thread stores.

    // ==================================================================
    // Load robot inertia params (mc, inertia) into block-shared memory.
    // Done once per block; all batches share the same robot parameters.
    // Uses float4 vectorized loads for bandwidth efficiency.
    // ==================================================================
    {
        const int batches_per_block = blockDim.x / TPB;
        const int total_threads = batches_per_block * TPB;
        for (int k = threadIdx.x; k < N_LINKS; k += total_threads) {
            // Load mc (4 floats: mass, com_x, com_y, com_z)
            const float4 mc4 = *reinterpret_cast<const float4*>(&link_masses_com[k * 4]);
            *reinterpret_cast<float4*>(&s_mc[k * 4]) = mc4;

            // Load inertia (8 floats: packed symmetric tensor)
            const float4 in0 = *reinterpret_cast<const float4*>(&link_inertias[k * 8]);
            const float4 in1 = *reinterpret_cast<const float4*>(&link_inertias[k * 8 + 4]);
            *reinterpret_cast<float4*>(&s_inertia[k * 8]) = in0;
            *reinterpret_cast<float4*>(&s_inertia[k * 8 + 4]) = in1;
        }
    }
    __syncthreads();  // Ensure all threads see the loaded inertia params

    // ==================================================================
    // Load forward intermediates (v, a, f) from cache into shared memory.
    // Replaces Phase 1 (forward recomputation); data saved by forward kernel.
    // Uses float4 vectorized global loads for bandwidth efficiency.
    // ==================================================================
    if (active) {
        for (int k = worker; k < N_LINKS; k += TPB) {
            const float *cache_k = &forward_cache[
                batch * N_LINKS * CACHE_FLOATS_PER_LINK
                + k * CACHE_FLOATS_PER_LINK];
            cache_load_va(&cache_k[CACHE_VA_OFFSET], &sv[k * 6], &sa[k * 6]);
            cache_load_f(&cache_k[CACHE_F_OFFSET], &sf[k * 6]);
        }
    }
    if (TPB > 1) __syncwarp(batch_mask);

    // ==================================================================
    // Phase 2: Pass 1 (root→leaf), adjoint of RNEA backward sweep
    //
    //   f_bar[k] += S · τ_bar[k]
    //   f_bar[k] += X · f_bar[parent]
    //   grad_q   += mult · dot(X·f_bar_parent, crf(S)·f[k])
    //
    //   Overwrites sf (which held f) with f_bar, link by link.
    // ==================================================================
    // #praagma unroll 1
    for (int level = 0; level < n_levels; level++) {
        if (active) {
            const int start = level_starts[level];
            const int end = level_starts[level + 1];
            for (int idx = start + worker; idx < end; idx += TPB) {
                const int k = level_links[idx];
                const int8_t j_type = joint_map_type[k];
                const int16_t j_idx = joint_map[k];
                const int16_t parent = link_map[k];
                const bool is_root = (parent < 0) || (parent == k);

                // Copy f[k] to register before overwriting with f_bar
                float f_k[6];
                for (int i = 0; i < 6; i++) f_k[i] = sf[k * 6 + i];

                // Initialize f_bar[k]
                float fbar_k[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                // adj of τ = S^T · f → f_bar += S · τ_bar
                float multiplier = 1.0f;
                if (j_type != FIXED && j_idx >= 0) {
                    const float2 j_off = *(const float2 *)&joint_offset_map[k * 2];
                    multiplier = j_off.x;
                    const int s_idx = get_s_index(j_type);
                    fbar_k[s_idx] += multiplier * grad_tau[q_base + j_idx];
                }

                // adj of f[parent] += X^T · f → f_bar[k] += X · f_bar[parent]
                if (!is_root) {
                    // Recompute R, p from fixed_transforms + q
                    float q_eff = 0.0f;
                    if (j_type != FIXED && j_idx >= 0) {
                        const float2 j_off = *(const float2 *)&joint_offset_map[k * 2];
                        q_eff = j_off.x * q[q_base + j_idx] + j_off.y;
                    }
                    float R[9], p_loc[3];
                    compute_local_Rp(&fixed_transforms[k * 12], j_type, q_eff, R, p_loc);

                    // X · f_bar[parent] (parent's slot already holds f_bar)
                    float X_fbar_p[6];
                    spatial_Xv(R, p_loc, &sf[parent * 6], X_fbar_p);
                    for (int i = 0; i < 6; i++) fbar_k[i] += X_fbar_p[i];

                    // dX^T/dq: grad_q += mult · (X·f_bar_parent)^T · crf(S) · f[k]
                    if (j_type != FIXED && j_idx >= 0) {
                        const int s_idx = get_s_index(j_type);
                        const float gq_contrib = multiplier * dot_crf_S(
                            X_fbar_p, f_k, s_idx
                        );
                        if (TPB > 1) {
                            atomicAdd(&grad_q[q_base + j_idx], gq_contrib);
                        } else {
                            grad_q[q_base + j_idx] += gq_contrib;
                        }
                    }
                }

                // Write f_bar to shared memory (overwriting f)
                for (int i = 0; i < 6; i++) sf[k * 6 + i] = fbar_k[i];
            }
        }
        if (TPB > 1) __syncwarp(batch_mask);
    }

    // ==================================================================
    // Phase 2.5: Write grad_f_ext = -f_bar (only when HAS_EXTERNAL_FORCES)
    //
    // Since f[k] = I·a + v×*(I·v) - f_ext[k], the gradient is:
    //   grad_f_ext[k] = -f_bar[k]
    //
    // This runs after f_bar is fully computed (Phase 2 complete).
    // ==================================================================
    if constexpr (HAS_EXTERNAL_FORCES) {
        if (active) {
            for (int k = worker; k < N_LINKS; k += TPB) {
                float *gfext_k = &grad_f_ext[batch * N_LINKS * 6 + k * 6];
                const float *fbar_k = &sf[k * 6];
                for (int i = 0; i < 6; i++) {
                    gfext_k[i] = -fbar_k[i];
                }
            }
        }
        if (TPB > 1) __syncwarp(batch_mask);
    }

    // ==================================================================
    // Phase 3: Zero a_bar and v_bar accumulators (independent, distributed)
    // ==================================================================
    if (active) {
        for (int k = worker; k < N_LINKS; k += TPB) {
            for (int i = 0; i < 6; i++) {
                sa_bar[k * 6 + i] = 0.0f;
                sv_bar[k * 6 + i] = 0.0f;
            }
        }
    }
    if (TPB > 1) __syncwarp(batch_mask);

    // ==================================================================
    // Phase 4: Pass 2 (leaf→root), adjoint of RNEA forward sweep
    //
    //   [3]': adj of f = I·a + crf(v)·I·v  →  a_bar, v_bar
    //   [2]': adj of a = X·a_parent + S·qdd + crm(v)·S·qd
    //                  →  grad_qdd, grad_qd, v_bar, a_bar[parent], grad_q
    //   [1]': adj of v = X·v_parent + S·qd
    //                  →  grad_qd, v_bar[parent], grad_q
    // ==================================================================
    // #praagma unroll 1
    for (int level = n_levels - 1; level >= 0; level--) {
        if (active) {
            const int start = level_starts[level];
            const int end = level_starts[level + 1];
            for (int idx = start + worker; idx < end; idx += TPB) {
                const int k = level_links[idx];
                const int8_t j_type = joint_map_type[k];
                const int16_t j_idx = joint_map[k];
                const int16_t parent = link_map[k];
                const bool is_root = (parent < 0) || (parent == k);

                // Use block-shared inertia params from shared memory
                const float *mc = &s_mc[k * 4];
                const float *inertia = &s_inertia[k * 8];
                const float *v_k = &sv[k * 6];

                // Read accumulated a_bar, v_bar from children (or zero if leaf)
                float a_bar_k[6], v_bar_k[6];
                for (int i = 0; i < 6; i++) {
                    a_bar_k[i] = sa_bar[k * 6 + i];
                    v_bar_k[i] = sv_bar[k * 6 + i];
                }

                // Read f_bar[k]
                float fbar_k[6];
                for (int i = 0; i < 6; i++) fbar_k[i] = sf[k * 6 + i];

                // [3]' adj of f = I·a + crf(v)·I·v
                // Reuse 2 temp arrays (temp1, temp2) instead of 5 separate ones
                // to reduce peak register demand by 18 floats (72 bytes).
                float temp1[6], temp2[6];

                //   a_bar += I · f_bar
                spatial_inertia_times_vec(mc, inertia, fbar_k, temp1);
                for (int i = 0; i < 6; i++) a_bar_k[i] += temp1[i];

                //   v_bar -= crf(f_bar) · I·v
                spatial_inertia_times_vec(mc, inertia, v_k, temp1);  // temp1 = Iv
                force_cross(fbar_k, temp1, temp2);                   // temp2 = crf(fbar)·Iv
                for (int i = 0; i < 6; i++) v_bar_k[i] -= temp2[i];

                //   v_bar -= I · crm(v) · f_bar
                motion_cross(v_k, fbar_k, temp1);                    // temp1 = crm(v)·fbar
                spatial_inertia_times_vec(mc, inertia, temp1, temp2); // temp2 = I·crm(v)·fbar
                for (int i = 0; i < 6; i++) v_bar_k[i] -= temp2[i];

                // [2]' adj of a = X·a_parent + S·qdd + crm(v)·S·qd
                float link_multiplier = 1.0f;
                if (j_type != FIXED && j_idx >= 0) {
                    const float2 j_off = *(const float2 *)&joint_offset_map[k * 2];
                    link_multiplier = j_off.x;
                    const int s_idx = get_s_index(j_type);
                    const float qd_k = link_multiplier * qd[q_base + j_idx];

                    // grad_qdd += mult · S^T · a_bar
                    const float gqdd = link_multiplier * a_bar_k[s_idx];
                    if (TPB > 1) {
                        atomicAdd(&grad_qdd[q_base + j_idx], gqdd);
                    } else {
                        grad_qdd[q_base + j_idx] += gqdd;
                    }

                    // grad_qd -= mult · (crf(v)·a_bar)[s]
                    const float gqd_crm = link_multiplier * force_cross_extract_s(
                        v_k, a_bar_k, s_idx
                    );
                    if (TPB > 1) {
                        atomicAdd(&grad_qd[q_base + j_idx], -gqd_crm);
                    } else {
                        grad_qd[q_base + j_idx] -= gqd_crm;
                    }

                    // v_bar += crf(S·qd) · a_bar
                    force_cross_S_add(v_bar_k, s_idx, qd_k, a_bar_k);
                }

                // Recompute R, p from fixed_transforms + q
                float q_eff_rp = 0.0f;
                if (j_type != FIXED && j_idx >= 0) {
                    const float2 j_off = *(const float2 *)&joint_offset_map[k * 2];
                    q_eff_rp = j_off.x * q[q_base + j_idx] + j_off.y;
                }
                float R[9], p_loc[3];
                compute_local_Rp(&fixed_transforms[k * 12], j_type, q_eff_rp, R, p_loc);

                // Propagate a_bar to parent and dX/dq contribution
                if (!is_root) {
                    float XT_abar[6];
                    spatial_XTf(R, p_loc, a_bar_k, XT_abar);
                    if (TPB > 1) {
                        for (int i = 0; i < 6; i++)
                            atomicAdd(&sa_bar[parent * 6 + i], XT_abar[i]);
                    } else {
                        for (int i = 0; i < 6; i++)
                            sa_bar[parent * 6 + i] += XT_abar[i];
                    }

                    if (j_type != FIXED && j_idx >= 0) {
                        const int s_idx = get_s_index(j_type);
                        float Xa_parent[6];
                        spatial_Xv(R, p_loc, &sa[parent * 6], Xa_parent);
                        const float gq_a = link_multiplier * dot_crm_S(
                            a_bar_k, Xa_parent, s_idx
                        );
                        if (TPB > 1) {
                            atomicAdd(&grad_q[q_base + j_idx], -gq_a);
                        } else {
                            grad_q[q_base + j_idx] -= gq_a;
                        }
                    }
                } else {
                    if (j_type != FIXED && j_idx >= 0) {
                        const int s_idx = get_s_index(j_type);
                        float Xa_grav[6];
                        spatial_Xv(R, p_loc, gravity, Xa_grav);
                        const float gq_g = link_multiplier * dot_crm_S(
                            a_bar_k, Xa_grav, s_idx
                        );
                        if (TPB > 1) {
                            atomicAdd(&grad_q[q_base + j_idx], -gq_g);
                        } else {
                            grad_q[q_base + j_idx] -= gq_g;
                        }
                    }
                }

                // [1]' adj of v = X·v_parent + S·qd
                if (j_type != FIXED && j_idx >= 0) {
                    const int s_idx = get_s_index(j_type);
                    const float gqd_v = link_multiplier * v_bar_k[s_idx];
                    if (TPB > 1) {
                        atomicAdd(&grad_qd[q_base + j_idx], gqd_v);
                    } else {
                        grad_qd[q_base + j_idx] += gqd_v;
                    }
                }

                // Propagate v_bar to parent and dX/dq contribution
                if (!is_root) {
                    float XT_vbar[6];
                    spatial_XTf(R, p_loc, v_bar_k, XT_vbar);
                    if (TPB > 1) {
                        for (int i = 0; i < 6; i++)
                            atomicAdd(&sv_bar[parent * 6 + i], XT_vbar[i]);
                    } else {
                        for (int i = 0; i < 6; i++)
                            sv_bar[parent * 6 + i] += XT_vbar[i];
                    }

                    if (j_type != FIXED && j_idx >= 0) {
                        const int s_idx = get_s_index(j_type);
                        float Xv_parent[6];
                        spatial_Xv(R, p_loc, &sv[parent * 6], Xv_parent);
                        const float gq_v = link_multiplier * dot_crm_S(
                            v_bar_k, Xv_parent, s_idx
                        );
                        if (TPB > 1) {
                            atomicAdd(&grad_q[q_base + j_idx], -gq_v);
                        } else {
                            grad_q[q_base + j_idx] -= gq_v;
                        }
                    }
                }
                // Root: v_parent = 0, so X·v_parent = 0, no contribution
            }
        }
        if (TPB > 1) __syncwarp(batch_mask);
    }
}

} // namespace dynamics
} // namespace curobo
