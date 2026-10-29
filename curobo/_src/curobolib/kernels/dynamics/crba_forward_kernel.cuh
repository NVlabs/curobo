/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file crba_forward_kernel.cuh
 * @brief Composite rigid body algorithm for batched joint-space mass matrices.
 */

#pragma once

#include "spatial_algebra.cuh"
#include "rnea_helpers.cuh"

namespace curobo {
namespace dynamics {

/**
 * @brief Compute one dense joint-space mass matrix per batch element.
 *
 * One power-of-two thread group owns one robot state. Composite spatial
 * inertias and local transforms live in that group's shared-memory partition.
 * The existing maximal-chain schedule supplies a topology-safe traversal.
 *
 * @tparam N_LINKS Number of retained rigid bodies.
 * @tparam N_DOF Number of generalized coordinates.
 * @tparam TPB Number of cooperative threads per robot state.
 * @tparam ALL_MOVABLE Whether every retained link has a generalized coordinate.
 * @tparam DIRECT_JOINTS Whether link and joint indices have an identity mapping.
 */
template<
    int N_LINKS,
    int N_DOF,
    int TPB = 1,
    bool ALL_MOVABLE = false,
    bool DIRECT_JOINTS = false>
__global__ void crba_forward_kernel(
    float * __restrict__ mass_matrix,
    const float * __restrict__ q,
    const float * __restrict__ fixed_transforms,
    const float * __restrict__ link_masses_com,
    const float * __restrict__ link_inertias,
    const int8_t * __restrict__ joint_map_type,
    const int16_t * __restrict__ joint_map,
    const int16_t * __restrict__ link_map,
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
        INERTIA_FLOATS_PER_LINK + TRANSFORM_FLOATS_PER_LINK;
    constexpr int SHARED_FLOATS_PER_BATCH = N_LINKS * SHARED_FLOATS_PER_LINK;

    const int global_thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int batch = global_thread / TPB;
    const int worker = global_thread % TPB;
    if (batch >= batch_size) return;

    extern __shared__ __align__(16) float shared_memory[];
    const int local_batch = threadIdx.x / TPB;
    float *composite_inertias =
        &shared_memory[local_batch * SHARED_FLOATS_PER_BATCH];
    float *local_transforms =
        &composite_inertias[N_LINKS * INERTIA_FLOATS_PER_LINK];

    const int q_base = batch * N_DOF;
    const int matrix_base = batch * N_DOF * N_DOF;
    unsigned batch_mask = 0;
    if constexpr (TPB > 1) {
        constexpr unsigned tpb_mask =
            (TPB == 32) ? 0xFFFFFFFFu : ((1u << TPB) - 1u);
        const int lane = threadIdx.x & 31;
        batch_mask = tpb_mask << ((lane / TPB) * TPB);
    }

    for (int index = worker; index < N_DOF * N_DOF; index += TPB) {
        mass_matrix[matrix_base + index] = 0.0f;
    }

    for (int link_index = worker; link_index < N_LINKS; link_index += TPB) {
        const int8_t joint_type = joint_map_type[link_index];
        const int16_t joint_index = DIRECT_JOINTS ? link_index : joint_map[link_index];
        float effective_position = 0.0f;
        if (ALL_MOVABLE || (joint_type != FIXED && joint_index >= 0)) {
            const float2 joint_offset = DIRECT_JOINTS
                ? make_float2(1.0f, 0.0f)
                : *reinterpret_cast<const float2*>(&joint_offset_map[link_index * 2]);
            effective_position =
                joint_offset.x * q[q_base + joint_index] + joint_offset.y;
        }

        float rotation[9];
        float translation[3];
        compute_local_Rp(
            &fixed_transforms[link_index * 12],
            joint_type,
            effective_position,
            rotation,
            translation
        );
        float *transform = &local_transforms[link_index * TRANSFORM_FLOATS_PER_LINK];
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

    // Build composite inertias from leaves to roots. Separate workers own
    // independent maximal chains at each dependency level.
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
                    float *parent_composite =
                        &composite_inertias[parent * INERTIA_FLOATS_PER_LINK];
                    const float *child_composite =
                        &composite_inertias[link_index * INERTIA_FLOATS_PER_LINK];
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

    // Every composite inertia is complete. Joint rows are independent for a
    // one-link-per-coordinate model and can now be distributed across workers.
    for (int link_index = worker; link_index < N_LINKS; link_index += TPB) {
        const int8_t joint_type = joint_map_type[link_index];
        const int16_t joint_index = DIRECT_JOINTS ? link_index : joint_map[link_index];
        const bool movable =
            ALL_MOVABLE || (joint_type != FIXED && joint_index >= 0);
        if (!movable) continue;

        const float2 joint_offset = DIRECT_JOINTS
            ? make_float2(1.0f, 0.0f)
            : *reinterpret_cast<const float2*>(&joint_offset_map[link_index * 2]);
        const float multiplier = joint_offset.x;
        const int motion_index = get_s_index(joint_type);
        const float *composite =
            &composite_inertias[link_index * INERTIA_FLOATS_PER_LINK];
        float propagated_force[6];
        #pragma unroll
        for (int row = 0; row < 6; row++) {
            propagated_force[row] =
                multiplier * composite[row * 6 + motion_index];
        }

        const int diagonal_index =
            matrix_base + joint_index * N_DOF + joint_index;
        const float diagonal = multiplier * propagated_force[motion_index];
        if constexpr (DIRECT_JOINTS) {
            mass_matrix[diagonal_index] += diagonal;
        } else {
            atomicAdd(&mass_matrix[diagonal_index], diagonal);
        }

        int force_link = link_index;
        while (true) {
            const int parent = link_map[force_link];
            if (parent < 0 || parent == force_link) break;

            const float *transform =
                &local_transforms[force_link * TRANSFORM_FLOATS_PER_LINK];
            float parent_force[6];
            spatial_XTf(
                transform,
                &transform[9],
                propagated_force,
                parent_force
            );
            #pragma unroll
            for (int row = 0; row < 6; row++) {
                propagated_force[row] = parent_force[row];
            }

            force_link = parent;
            const int8_t parent_joint_type = joint_map_type[parent];
            const int16_t parent_joint_index =
                DIRECT_JOINTS ? parent : joint_map[parent];
            if (
                ALL_MOVABLE
                || (parent_joint_type != FIXED && parent_joint_index >= 0)
            ) {
                const float2 parent_joint_offset = DIRECT_JOINTS
                    ? make_float2(1.0f, 0.0f)
                    : *reinterpret_cast<const float2*>(&joint_offset_map[parent * 2]);
                const int parent_motion_index = get_s_index(parent_joint_type);
                const float contribution =
                    parent_joint_offset.x * propagated_force[parent_motion_index];
                const int upper_index =
                    matrix_base + joint_index * N_DOF + parent_joint_index;
                const int lower_index =
                    matrix_base + parent_joint_index * N_DOF + joint_index;
                if constexpr (DIRECT_JOINTS) {
                    mass_matrix[upper_index] += contribution;
                    mass_matrix[lower_index] += contribution;
                } else if (joint_index == parent_joint_index) {
                    atomicAdd(&mass_matrix[upper_index], 2.0f * contribution);
                } else {
                    atomicAdd(&mass_matrix[upper_index], contribution);
                    atomicAdd(&mass_matrix[lower_index], contribution);
                }
            }
        }
    }
}

}  // namespace dynamics
}  // namespace curobo
