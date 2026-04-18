/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kinematics_util.cuh"
#include "common/block_warp_reductions.cuh"

namespace curobo{
namespace kinematics{


    /**
     * Optimized helper function to compute the derivative of jacobian elements with respect to joint angles.
     * This computes dJ/dq where J is the jacobian matrix and q are joint angles.
     * Used for backpropagating gradients through jacobian-based cost functions.
     *
     * The gradient is computed as: dL/dq = sum over all jacobian elements of (dL/dJ * dJ/dq)
     *
     * This version uses precomputed joint-link mapping and distributes joints across threads
     * to eliminate atomicAdd operations and improve performance.
     */
     template<typename AccumulatorType>
     __forceinline__ __device__ void compute_jacobian_derivatives(
       AccumulatorType *grad_joint,
       const float *grad_jacobian,  // [batch_size * n_tool_frames * 6 * njoints]
       const float *cumul_mat,
       const int8_t *jointMapType,
       const int16_t *jointMap,
       const int16_t *linkMap,
       const int16_t *linkChainData,    // Packed array of actual link indices
       const int16_t *linkChainOffsets, // Offset array for each link's chain
       const int16_t *jointLinksData,   // NEW: Precomputed packed array of links per joint
       const int16_t *jointLinksOffsets, // NEW: Offset array for each joint's links
       const bool *joint_affects_endeffector,  // NEW: Cache for optimization
       const float *jointOffset,
       const int16_t *toolFrameMap,
       const int batch_index,
       const int njoints,
       const int nlinks,
       const int n_tool_frames,
       const int matAddrBase,
       const int thread_index,
       const int threads_per_batch
     ) {
       // Distribute joint processing across threads for better parallelization
       // This eliminates the need for atomicAdd since each thread has exclusive access to its joints
       const int16_t joints_per_thread = curobo::common::ceil_div(njoints, threads_per_batch);

       for (int16_t i = 0; i < joints_per_thread; i++) {
         // Use strided indexing for joints for better memory coalescing
         int16_t joint_idx = i * threads_per_batch + thread_index;

         if (joint_idx >= njoints) {
           break;
         }

         AccumulatorType joint_grad_sum = 0.0f;

         // Get precomputed links connected to this joint - MUCH FASTER!
         const int16_t joint_links_start = jointLinksOffsets[joint_idx];
         const int16_t joint_links_end = jointLinksOffsets[joint_idx + 1];

         // For each stored link (end-effector), process jacobian gradients
         for (int16_t ee_idx = 0; ee_idx < n_tool_frames; ee_idx++) {

           // 🚀 EARLY EXIT: Check cache first - O(1) lookup
           if (!joint_affects_endeffector[joint_idx * n_tool_frames + ee_idx]) {
             continue; // Skip entire computation for this joint-endeffector pair!
           }

           const int16_t tool_frame_idx = toolFrameMap[ee_idx];

           // Use precomputed chain data to get links that affect this end-effector
           const int16_t chain_start = linkChainOffsets[tool_frame_idx];
           const int16_t chain_end = linkChainOffsets[tool_frame_idx + 1];

           // Get the jacobian gradient for this end-effector: [6 * njoints]
           const int jac_offset = batch_index * n_tool_frames * 6 * njoints + ee_idx * 6 * njoints;

           // For each joint column k in the jacobian
           for (int16_t k = 0; k < njoints; k++) {
             // Find the link associated with joint k using the same chain data approach
             int16_t joint_k_link_idx = -1;

             // Look through the end-effector's chain to find joint k
             for (int16_t chain_search_idx = chain_start; chain_search_idx < chain_end; chain_search_idx++) {
               const int16_t candidate_link = linkChainData[chain_search_idx];
               if (candidate_link > 0 && jointMap[candidate_link] == k) {
                 joint_k_link_idx = candidate_link;
                 break;
               }
             }

             if (joint_k_link_idx == -1) continue;

             // Get jacobian gradient for joint k: grad_jacobian[batch_idx, ee_idx, :, k]
             // Shape is [batch_size, n_tool_frames, 6, n_joints]
             const float3 grad_jac_linear = make_float3(
               grad_jacobian[jac_offset + 0 * njoints + k],  // row 0 (linear x)
               grad_jacobian[jac_offset + 1 * njoints + k],  // row 1 (linear y)
               grad_jacobian[jac_offset + 2 * njoints + k]   // row 2 (linear z)
             );
             const float3 grad_jac_angular = make_float3(
               grad_jacobian[jac_offset + 3 * njoints + k],  // row 3 (angular x)
               grad_jacobian[jac_offset + 4 * njoints + k],  // row 4 (angular y)
               grad_jacobian[jac_offset + 5 * njoints + k]   // row 5 (angular z)
             );

             // Skip if gradients are zero
             if ((grad_jac_linear.x == 0.0f && grad_jac_linear.y == 0.0f && grad_jac_linear.z == 0.0f) &&
                 (grad_jac_angular.x == 0.0f && grad_jac_angular.y == 0.0f && grad_jac_angular.z == 0.0f)) {
               continue;
             }

             // Now compute dJ_k/dq_joint_idx using precomputed joint-link mapping
             // This is non-zero only if joint_idx affects joint k (i.e., joint_idx comes before joint k in the chain)

             // Check if joint_idx affects joint k by comparing their positions in the chain
             bool joint_idx_affects_k = false;
             bool joint_idx_affects_ee_direct = false;

             // Find the link for joint_idx in the precomputed mapping
             int16_t joint_idx_link_idx = -1;
             for (int16_t jl_idx = joint_links_start; jl_idx < joint_links_end; jl_idx++) {
               const int16_t candidate_link = jointLinksData[jl_idx];

               // Check if this link is in the kinematic chain
               for (int16_t chain_idx = chain_start; chain_idx < chain_end; chain_idx++) {
                 if (linkChainData[chain_idx] == candidate_link) {
                   joint_idx_link_idx = candidate_link;
                   joint_idx_affects_ee_direct = true;

                   // Check if joint_idx comes before joint k in the chain
                   for (int16_t search_idx = chain_start; search_idx < chain_end; search_idx++) {
                     if (linkChainData[search_idx] == joint_k_link_idx) {
                       // Found joint k, check if joint_idx comes before it
                       for (int16_t prior_idx = chain_start; prior_idx < search_idx; prior_idx++) {
                         if (linkChainData[prior_idx] == candidate_link) {
                           joint_idx_affects_k = true;
                           break;
                         }
                       }
                       break;
                     }
                   }
                   break;
                 }
               }
               if (joint_idx_link_idx != -1) break;
             }

             if (joint_idx_link_idx == -1) continue; // Joint not in this chain

             // Compute the derivative based on the relationship between joints
             AccumulatorType derivative_contribution = 0.0f;

             if (joint_idx_affects_k || joint_idx_affects_ee_direct) {
               // Get joint properties and positions
               const int joint_idx_addr = matAddrBase + joint_idx_link_idx * 12;
               const int joint_k_addr = matAddrBase + joint_k_link_idx * 12;
               const int ee_addr = matAddrBase + tool_frame_idx * 12;

               const int8_t joint_idx_type = jointMapType[joint_idx_link_idx];
               const int8_t joint_k_type = jointMapType[joint_k_link_idx];

               const float joint_idx_axis_sign = jointOffset[joint_idx_link_idx * 2];
               const float joint_k_axis_sign = jointOffset[joint_k_link_idx * 2];

               const float3 joint_idx_pos = make_float3(
                 cumul_mat[joint_idx_addr + 3],
                 cumul_mat[joint_idx_addr + 7],
                 cumul_mat[joint_idx_addr + 11]
               );
               const float3 joint_k_pos = make_float3(
                 cumul_mat[joint_k_addr + 3],
                 cumul_mat[joint_k_addr + 7],
                 cumul_mat[joint_k_addr + 11]
               );
               const float3 ee_pos = make_float3(
                 cumul_mat[ee_addr + 3],
                 cumul_mat[ee_addr + 7],
                 cumul_mat[ee_addr + 11]
               );

               if (joint_k_type >= X_ROT && joint_k_type <= Z_ROT) {
                 // Joint k is revolute: J_k = [omega_k × (p_ee - p_k); omega_k]
                 const int k_axis_idx = joint_k_type - X_ROT;
                 const float3 joint_k_axis = make_float3(
                   cumul_mat[joint_k_addr + k_axis_idx],
                   cumul_mat[joint_k_addr + 4 + k_axis_idx],
                   cumul_mat[joint_k_addr + 8 + k_axis_idx]
                 );
                 const float3 omega_k = joint_k_axis_sign * joint_k_axis;

                 if (joint_idx_type >= X_ROT && joint_idx_type <= Z_ROT) {
                   // Joint idx is also revolute
                   const int idx_axis_idx = joint_idx_type - X_ROT;
                   const float3 joint_idx_axis = make_float3(
                     cumul_mat[joint_idx_addr + idx_axis_idx],
                     cumul_mat[joint_idx_addr + 4 + idx_axis_idx],
                     cumul_mat[joint_idx_addr + 8 + idx_axis_idx]
                   );
                   const float3 omega_idx = joint_idx_axis_sign * joint_idx_axis;

                   float3 d_linear = make_float3(0.0f, 0.0f, 0.0f);
                   float3 d_angular = make_float3(0.0f, 0.0f, 0.0f);

                   if (joint_idx_affects_k) {
                     // omega_k changes due to joint_idx rotation
                     d_angular = cross(omega_idx, omega_k);

                     // Linear part when omega_k changes
                     const float3 pos_diff_ee_k = ee_pos - joint_k_pos;
                     d_linear = cross(d_angular, pos_diff_ee_k);
                   }

                   if (joint_idx_affects_ee_direct) {
                     // p_ee changes due to joint_idx rotation
                     const float3 pos_diff_ee_idx = ee_pos - joint_idx_pos;
                     d_linear = d_linear + cross(omega_k, cross(omega_idx, pos_diff_ee_idx));
                   }

                   if (joint_idx_affects_k) {
                     // p_k changes due to joint_idx rotation
                     const float3 pos_diff_k_idx = joint_k_pos - joint_idx_pos;
                     d_linear = d_linear - cross(omega_k, cross(omega_idx, pos_diff_k_idx));
                   }

                   derivative_contribution += dot(grad_jac_linear, d_linear);
                   derivative_contribution += dot(grad_jac_angular, d_angular);

                 } else if (joint_idx_type >= X_PRISM && joint_idx_type <= Z_PRISM) {
                   // Joint idx is prismatic
                   const int idx_axis_idx = joint_idx_type - X_PRISM;
                   const float3 joint_idx_axis = make_float3(
                     cumul_mat[joint_idx_addr + idx_axis_idx],
                     cumul_mat[joint_idx_addr + 4 + idx_axis_idx],
                     cumul_mat[joint_idx_addr + 8 + idx_axis_idx]
                   );
                   const float3 d_idx = joint_idx_axis_sign * joint_idx_axis;

                   float3 d_linear = make_float3(0.0f, 0.0f, 0.0f);

                   if (joint_idx_affects_ee_direct && !joint_idx_affects_k) {
                     // Only p_ee changes
                     d_linear = cross(omega_k, d_idx);
                   } else if (!joint_idx_affects_ee_direct && joint_idx_affects_k) {
                     // Only p_k changes
                     d_linear = cross(omega_k, -1.0f * d_idx);
                   }
                   // If both or neither are affected, derivative is 0

                   // Angular part is always 0 for prismatic joint_idx
                   derivative_contribution += dot(grad_jac_linear, d_linear);
                 }

               } else if (joint_k_type >= X_PRISM && joint_k_type <= Z_PRISM) {
                 // Joint k is prismatic: J_k = [d_k; 0]
                 const int k_axis_idx = joint_k_type - X_PRISM;
                 const float3 joint_k_axis = make_float3(
                   cumul_mat[joint_k_addr + k_axis_idx],
                   cumul_mat[joint_k_addr + 4 + k_axis_idx],
                   cumul_mat[joint_k_addr + 8 + k_axis_idx]
                 );
                 const float3 d_k = joint_k_axis_sign * joint_k_axis;

                 if (joint_idx_type >= X_ROT && joint_idx_type <= Z_ROT && joint_idx_affects_k) {
                   // Joint idx is revolute and affects joint k
                   const int idx_axis_idx = joint_idx_type - X_ROT;
                   const float3 joint_idx_axis = make_float3(
                     cumul_mat[joint_idx_addr + idx_axis_idx],
                     cumul_mat[joint_idx_addr + 4 + idx_axis_idx],
                     cumul_mat[joint_idx_addr + 8 + idx_axis_idx]
                   );
                   const float3 omega_idx = joint_idx_axis_sign * joint_idx_axis;

                   // d_k changes due to rotation by joint_idx
                   const float3 d_linear = cross(omega_idx, d_k);

                   derivative_contribution += dot(grad_jac_linear, d_linear);
                   // Angular part is 0 for prismatic joints
                 }
                 // If joint_idx is prismatic or doesn't affect k, derivative is 0
               }
             }

             joint_grad_sum += derivative_contribution;
           }
         }

         // Store result directly without atomicAdd since each thread has exclusive access to its joints
         grad_joint[joint_idx] = joint_grad_sum;
       }
     }

    }
}