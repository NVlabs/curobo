/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file crba_backward_kernel.cuh
 * @brief First-order position VJP for joint-space mass matrices computed by CRBA.
 */

#pragma once

#include "spatial_algebra.cuh"
#include "rnea_helpers.cuh"

namespace curobo {
namespace dynamics {

/**
 * @brief Return the upstream weight for one physical CRBA matrix contribution.
 *
 * CRBA writes each off-diagonal physical contribution to both symmetric
 * generalized entries. Mimic links can map both ends to the same generalized
 * coordinate, in which case the contribution is accumulated twice on the
 * diagonal.
 */
template<int N_DOF, bool ALL_MOVABLE, bool DIRECT_JOINTS>
__device__ __forceinline__ float crba_mass_weight(
    const float *grad_mass_matrix,
    const int matrix_base,
    const int row_link,
    const int current_link,
    const int8_t *joint_map_type,
    const int16_t *joint_map
) {
    const int8_t current_joint_type = joint_map_type[current_link];
    const int16_t current_joint_index =
        DIRECT_JOINTS ? current_link : joint_map[current_link];
    if (
        !ALL_MOVABLE
        && (current_joint_type == FIXED || current_joint_index < 0)
    ) {
        return 0.0f;
    }

    const int row_joint_index = DIRECT_JOINTS ? row_link : joint_map[row_link];
    if (current_link == row_link) {
        return grad_mass_matrix[
            matrix_base + row_joint_index * N_DOF + row_joint_index
        ];
    }
    if (current_joint_index == row_joint_index) {
        return 2.0f * grad_mass_matrix[
            matrix_base + row_joint_index * N_DOF + row_joint_index
        ];
    }
    return grad_mass_matrix[
               matrix_base + row_joint_index * N_DOF + current_joint_index
           ]
        + grad_mass_matrix[
               matrix_base + current_joint_index * N_DOF + row_joint_index
          ];
}

/**
 * @brief Reverse ``parent += X^T * child * X`` for one tree edge.
 *
 * Accumulates the child-inertia adjoint and returns the derivative with respect
 * to the edge's effective joint coordinate. The two transform derivatives use
 * the same spatial cross-product identities as the analytical RNEA backward.
 */
__device__ __forceinline__ float spatial_transform_inertia_vjp(
    const float *rotation,
    const float *translation,
    const float *child_inertia,
    const float *parent_adjoint,
    float *child_adjoint,
    const int motion_index
) {
    float position_gradient = 0.0f;
    #pragma unroll
    for (int column = 0; column < 6; column++) {
        float basis[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        basis[column] = 1.0f;

        float child_motion[6];
        spatial_Xv(rotation, translation, basis, child_motion);
        float child_force[6];
        spatial_matrix_times_vec(child_inertia, child_motion, child_force);

        float parent_force_adjoint[6];
        #pragma unroll
        for (int row = 0; row < 6; row++) {
            parent_force_adjoint[row] = parent_adjoint[row * 6 + column];
        }
        float child_force_adjoint[6];
        spatial_Xv(
            rotation,
            translation,
            parent_force_adjoint,
            child_force_adjoint
        );
        position_gradient +=
            dot_crf_S(child_force_adjoint, child_force, motion_index);

        float child_motion_adjoint[6];
        #pragma unroll
        for (int child_column = 0; child_column < 6; child_column++) {
            float value = 0.0f;
            #pragma unroll
            for (int row = 0; row < 6; row++) {
                value += child_inertia[row * 6 + child_column]
                    * child_force_adjoint[row];
            }
            child_motion_adjoint[child_column] = value;
        }
        position_gradient -=
            dot_crm_S(child_motion_adjoint, child_motion, motion_index);

        #pragma unroll
        for (int row = 0; row < 6; row++) {
            #pragma unroll
            for (int child_column = 0; child_column < 6; child_column++) {
                child_adjoint[row * 6 + child_column] +=
                    child_force_adjoint[row] * child_motion[child_column];
            }
        }
    }
    return position_gradient;
}

/**
 * @brief Compute ``dL/dq`` from ``dL/dM`` with an analytical reverse CRBA.
 *
 * The kernel recomputes transforms and composite inertias, reverses the mass
 * matrix extraction for all physical joint rows, then reverses the composite
 * inertia recursion from roots to leaves. It supports arbitrary dense upstream
 * matrices, including non-symmetric upstream gradients.
 */
template<
    int N_LINKS,
    int N_DOF,
    int TPB = 1,
    bool ALL_MOVABLE = false,
    bool DIRECT_JOINTS = false>
__global__ void crba_backward_kernel(
    float * __restrict__ grad_q,
    const float * __restrict__ grad_mass_matrix,
    const float * __restrict__ q,
    const float * __restrict__ fixed_transforms,
    const float * __restrict__ link_masses_com,
    const float * __restrict__ link_inertias,
    const int8_t * __restrict__ joint_map_type,
    const int16_t * __restrict__ joint_map,
    const int16_t * __restrict__ link_map,
    const int16_t * __restrict__ chain_map,
    const float * __restrict__ joint_offset_map,
    const int16_t * __restrict__ level_starts,
    const int16_t * __restrict__ level_links,
    const int batch_size,
    const int n_levels
) {
    static_assert(N_LINKS > 0, "N_LINKS must be > 0");
    static_assert(N_DOF > 0, "N_DOF must be > 0");
    static_assert(TPB >= 1 && TPB <= 32, "TPB must be in [1, 32]");
    static_assert((TPB & (TPB - 1)) == 0, "TPB must be a power of two");

    constexpr int INERTIA_FLOATS_PER_LINK = 36;
    constexpr int TRANSFORM_FLOATS_PER_LINK = 12;
    constexpr int SHARED_FLOATS_PER_LINK =
        2 * INERTIA_FLOATS_PER_LINK + TRANSFORM_FLOATS_PER_LINK;
    constexpr int SHARED_FLOATS_PER_BATCH = N_LINKS * SHARED_FLOATS_PER_LINK;

    const int global_thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int batch = global_thread / TPB;
    const int worker = global_thread % TPB;
    if (batch >= batch_size) return;

    extern __shared__ __align__(16) float shared_memory[];
    const int local_batch = threadIdx.x / TPB;
    float *composite_inertias =
        &shared_memory[local_batch * SHARED_FLOATS_PER_BATCH];
    float *composite_adjoint =
        &composite_inertias[N_LINKS * INERTIA_FLOATS_PER_LINK];
    float *local_transforms =
        &composite_adjoint[N_LINKS * INERTIA_FLOATS_PER_LINK];

    const int q_base = batch * N_DOF;
    const int matrix_base = batch * N_DOF * N_DOF;
    unsigned batch_mask = 0;
    if constexpr (TPB > 1) {
        constexpr unsigned TPB_MASK =
            (TPB == 32) ? 0xFFFFFFFFu : ((1u << TPB) - 1u);
        const int lane = threadIdx.x & 31;
        batch_mask = TPB_MASK << ((lane / TPB) * TPB);
    }

    for (int joint_index = worker; joint_index < N_DOF; joint_index += TPB) {
        grad_q[q_base + joint_index] = 0.0f;
    }
    for (int index = worker;
         index < N_LINKS * INERTIA_FLOATS_PER_LINK;
         index += TPB) {
        composite_adjoint[index] = 0.0f;
    }
    for (int link_index = worker; link_index < N_LINKS; link_index += TPB) {
        const int8_t joint_type = joint_map_type[link_index];
        const int16_t joint_index =
            DIRECT_JOINTS ? link_index : joint_map[link_index];
        float effective_position = 0.0f;
        if (ALL_MOVABLE || (joint_type != FIXED && joint_index >= 0)) {
            const float2 joint_offset = DIRECT_JOINTS
                ? make_float2(1.0f, 0.0f)
                : *reinterpret_cast<const float2*>(
                      &joint_offset_map[link_index * 2]
                  );
            effective_position =
                joint_offset.x * q[q_base + joint_index] + joint_offset.y;
        }

        float rotation[9];
        float translation[3];
        compute_local_Rp(
            &fixed_transforms[link_index * TRANSFORM_FLOATS_PER_LINK],
            joint_type,
            effective_position,
            rotation,
            translation
        );
        float *transform =
            &local_transforms[link_index * TRANSFORM_FLOATS_PER_LINK];
        *reinterpret_cast<float4*>(&transform[0]) = make_float4(
            rotation[0], rotation[1], rotation[2], rotation[3]
        );
        *reinterpret_cast<float4*>(&transform[4]) = make_float4(
            rotation[4], rotation[5], rotation[6], rotation[7]
        );
        *reinterpret_cast<float4*>(&transform[8]) = make_float4(
            rotation[8], translation[0], translation[1], translation[2]
        );
        spatial_inertia_to_matrix(
            &link_masses_com[link_index * 4],
            &link_inertias[link_index * 8],
            &composite_inertias[link_index * INERTIA_FLOATS_PER_LINK]
        );
    }
    if constexpr (TPB > 1) __syncwarp(batch_mask);

    // Recompute the forward composite inertias from leaves to roots.
    for (int level = n_levels - 1; level >= 0; level--) {
        const int level_start = level_starts[level];
        const int level_end = level_starts[level + 1];
        for (int schedule_index = level_start + worker;
             schedule_index < level_end;
             schedule_index += TPB) {
            const int segment_root = level_links[schedule_index * 2];
            int link_index = level_links[schedule_index * 2 + 1];
            while (true) {
                const int parent = link_map[link_index];
                if (parent >= 0 && parent != link_index) {
                    const float *transform =
                        &local_transforms[link_index * TRANSFORM_FLOATS_PER_LINK];
                    float *parent_composite = &composite_inertias[
                        parent * INERTIA_FLOATS_PER_LINK
                    ];
                    const float *child_composite = &composite_inertias[
                        link_index * INERTIA_FLOATS_PER_LINK
                    ];
                    if constexpr (TPB > 1) {
                        if (link_index == segment_root) {
                            spatial_transform_inertia_add<true>(
                                transform,
                                &transform[9],
                                child_composite,
                                parent_composite
                            );
                        } else {
                            spatial_transform_inertia_add<false>(
                                transform,
                                &transform[9],
                                child_composite,
                                parent_composite
                            );
                        }
                    } else {
                        spatial_transform_inertia_add<false>(
                            transform,
                            &transform[9],
                            child_composite,
                            parent_composite
                        );
                    }
                }
                if (link_index == segment_root) break;
                link_index = parent;
            }
        }
        if constexpr (TPB > 1) __syncwarp(batch_mask);
    }

    // Reverse the mass-matrix extraction. A worker retains the force path for
    // one physical joint row so the path can be traversed in reverse.
    for (int row_link = worker; row_link < N_LINKS; row_link += TPB) {
        const int8_t row_joint_type = joint_map_type[row_link];
        const int16_t row_joint_index =
            DIRECT_JOINTS ? row_link : joint_map[row_link];
        const bool row_movable =
            ALL_MOVABLE
            || (row_joint_type != FIXED && row_joint_index >= 0);
        if (!row_movable) continue;

        const float2 row_joint_offset = DIRECT_JOINTS
            ? make_float2(1.0f, 0.0f)
            : *reinterpret_cast<const float2*>(&joint_offset_map[row_link * 2]);
        const float row_multiplier = row_joint_offset.x;
        const int row_motion_index = get_s_index(row_joint_type);

        int path_links[N_LINKS];
        float path_forces[N_LINKS * 6];
        int path_size = 1;
        path_links[0] = row_link;
        const float *row_composite =
            &composite_inertias[row_link * INERTIA_FLOATS_PER_LINK];
        #pragma unroll
        for (int spatial_index = 0; spatial_index < 6; spatial_index++) {
            path_forces[spatial_index] =
                row_multiplier
                * row_composite[spatial_index * 6 + row_motion_index];
        }

        int force_link = row_link;
        while (true) {
            const int parent = link_map[force_link];
            if (parent < 0 || parent == force_link) break;
            const float *transform =
                &local_transforms[force_link * TRANSFORM_FLOATS_PER_LINK];
            spatial_XTf(
                transform,
                &transform[9],
                &path_forces[(path_size - 1) * 6],
                &path_forces[path_size * 6]
            );
            path_links[path_size] = parent;
            path_size++;
            force_link = parent;
        }

        const int root_link = path_links[path_size - 1];
        float force_adjoint[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const float root_weight = crba_mass_weight<
            N_DOF,
            ALL_MOVABLE,
            DIRECT_JOINTS
        >(
            grad_mass_matrix,
            matrix_base,
            row_link,
            root_link,
            joint_map_type,
            joint_map
        );
        const int8_t root_joint_type = joint_map_type[root_link];
        const int16_t root_joint_index =
            DIRECT_JOINTS ? root_link : joint_map[root_link];
        if (
            ALL_MOVABLE
            || (root_joint_type != FIXED && root_joint_index >= 0)
        ) {
            const float2 root_joint_offset = DIRECT_JOINTS
                ? make_float2(1.0f, 0.0f)
                : *reinterpret_cast<const float2*>(
                      &joint_offset_map[root_link * 2]
                  );
            force_adjoint[get_s_index(root_joint_type)] =
                root_weight * root_joint_offset.x;
        }

        for (int path_index = path_size - 2; path_index >= 0; path_index--) {
            const int child_link = path_links[path_index];
            const float *transform =
                &local_transforms[child_link * TRANSFORM_FLOATS_PER_LINK];
            float child_force_adjoint[6];
            spatial_Xv(
                transform,
                &transform[9],
                force_adjoint,
                child_force_adjoint
            );

            const int8_t child_joint_type = joint_map_type[child_link];
            const int16_t child_joint_index =
                DIRECT_JOINTS ? child_link : joint_map[child_link];
            const bool child_movable =
                ALL_MOVABLE
                || (child_joint_type != FIXED && child_joint_index >= 0);
            if (child_movable) {
                const float2 child_joint_offset = DIRECT_JOINTS
                    ? make_float2(1.0f, 0.0f)
                    : *reinterpret_cast<const float2*>(
                          &joint_offset_map[child_link * 2]
                      );
                const float child_multiplier = child_joint_offset.x;
                const int child_motion_index = get_s_index(child_joint_type);
                const float transform_gradient = child_multiplier * dot_crf_S(
                    child_force_adjoint,
                    &path_forces[path_index * 6],
                    child_motion_index
                );
                atomicAdd(
                    &grad_q[q_base + child_joint_index],
                    transform_gradient
                );

                const float child_weight = crba_mass_weight<
                    N_DOF,
                    ALL_MOVABLE,
                    DIRECT_JOINTS
                >(
                    grad_mass_matrix,
                    matrix_base,
                    row_link,
                    child_link,
                    joint_map_type,
                    joint_map
                );
                child_force_adjoint[child_motion_index] +=
                    child_weight * child_multiplier;
            }
            #pragma unroll
            for (int spatial_index = 0; spatial_index < 6; spatial_index++) {
                force_adjoint[spatial_index] = child_force_adjoint[spatial_index];
            }
        }

        float *row_composite_adjoint =
            &composite_adjoint[row_link * INERTIA_FLOATS_PER_LINK];
        #pragma unroll
        for (int spatial_index = 0; spatial_index < 6; spatial_index++) {
            row_composite_adjoint[spatial_index * 6 + row_motion_index] +=
                row_multiplier * force_adjoint[spatial_index];
        }
    }
    if constexpr (TPB > 1) __syncwarp(batch_mask);

    // Reverse the composite recursion from roots to leaves. At each level the
    // parent adjoints are complete before their independent child chains run.
    for (int level = 0; level < n_levels; level++) {
        const int level_start = level_starts[level];
        const int level_end = level_starts[level + 1];
        for (int schedule_index = level_start + worker;
             schedule_index < level_end;
             schedule_index += TPB) {
            int link_index = level_links[schedule_index * 2];
            const int segment_tail = level_links[schedule_index * 2 + 1];
            while (true) {
                const int parent = link_map[link_index];
                if (parent >= 0 && parent != link_index) {
                    const float *transform =
                        &local_transforms[link_index * TRANSFORM_FLOATS_PER_LINK];
                    const float *child_composite = &composite_inertias[
                        link_index * INERTIA_FLOATS_PER_LINK
                    ];
                    const float *parent_composite_adjoint = &composite_adjoint[
                        parent * INERTIA_FLOATS_PER_LINK
                    ];
                    float *child_composite_adjoint = &composite_adjoint[
                        link_index * INERTIA_FLOATS_PER_LINK
                    ];
                    const int8_t joint_type = joint_map_type[link_index];
                    const int16_t joint_index =
                        DIRECT_JOINTS ? link_index : joint_map[link_index];
                    const bool movable =
                        ALL_MOVABLE
                        || (joint_type != FIXED && joint_index >= 0);
                    const int motion_index = movable ? get_s_index(joint_type) : -1;
                    const float effective_position_gradient =
                        spatial_transform_inertia_vjp(
                            transform,
                            &transform[9],
                            child_composite,
                            parent_composite_adjoint,
                            child_composite_adjoint,
                            motion_index
                        );
                    if (movable) {
                        const float2 joint_offset = DIRECT_JOINTS
                            ? make_float2(1.0f, 0.0f)
                            : *reinterpret_cast<const float2*>(
                                  &joint_offset_map[link_index * 2]
                              );
                        atomicAdd(
                            &grad_q[q_base + joint_index],
                            joint_offset.x * effective_position_gradient
                        );
                    }
                }
                if (link_index == segment_tail) break;
                link_index = chain_map[link_index];
            }
        }
        if constexpr (TPB > 1) __syncwarp(batch_mask);
    }
}

}  // namespace dynamics
}  // namespace curobo
