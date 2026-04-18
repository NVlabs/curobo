/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kinematics_constants.h"
#include "third_party/helper_math.h"
#include "common/math.cuh"
#include "kinematics_joint_util.cuh"
#include "kinematics_util.cuh"
#include "common/block_warp_reductions.cuh"
#include "kinematics_jacobian_backward_helper.cuh"

namespace curobo{
namespace kinematics{

    /**
     * CUDA kernel for jacobian backward pass computation.
     * This kernel computes gradients w.r.t. joint angles from gradients w.r.t. jacobian elements.
     * Uses the chain rule: dL/dq = (dL/dJ) * (dJ/dq)
     *
     * Template parameters:
     * @tparam psum_t Type for partial sum accumulation
     * @tparam MAX_JOINTS Maximum number of joints (for static array sizing)
     * @tparam USE_WARP_REDUCE Whether to use warp-level or block-level reduction
     */
    template<typename AccumulatorType, int16_t MaxJoints, bool UseWarpReduce>
    __global__ void kinematics_jacobian_gradient_backward_kernel(
      float *grad_joint,                    // batchSize * njoints
      const float *grad_jacobian,           // batchSize * n_tool_frames * 6 * njoints
      const float *global_cumul_mat,        // batchSize * nlinks * 12
      const int8_t *jointMapType,           // nlinks
      const int16_t *jointMap,              // nlinks
      const int16_t *linkMap,               // nlinks
      const int16_t *linkChainData,         // Packed array of actual link indices
      const int16_t *linkChainOffsets,      // Offset array for each link's chain
      const int16_t *jointLinksData,        // NEW: Precomputed packed array of links per joint
      const int16_t *jointLinksOffsets,     // NEW: Offset array for each joint's links
      const bool *joint_affects_endeffector, // NEW: Cache for optimization
      const int16_t *toolFrameMap,          // n_tool_frames
      const float *jointOffset,             // nlinks * 2
      const int batch_size,
      const int njoints,
      const int nlinks,
      const int n_tool_frames,
      const int threads_per_batch
    ) {
      extern __shared__ __align__(16) float cumul_mat[];

      // Shared memory for block reduction (only used when !USE_WARP_REDUCE)
      __shared__ AccumulatorType block_reduce_shared_data[32];
      AccumulatorType reduced_sum;

      // Common indexing logic
      int t = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / threads_per_batch;

      if (batch >= batch_size)
        return;

      const int elem_idx = threadIdx.x % threads_per_batch;
      const int local_batch = threadIdx.x / threads_per_batch;
      const int matAddrBase = local_batch * nlinks * 12;

      // Calculate start batch index for this block
      const int start_batch_index = blockIdx.x * (blockDim.x / threads_per_batch);
      const int batches_per_block = (blockDim.x + threads_per_batch - 1) / threads_per_batch;

      // Load cumulative matrices to shared memory
      read_cumul_to_shared_mem(cumul_mat, global_cumul_mat, start_batch_index,
                               threadIdx.x, nlinks,
                               batch_size, batches_per_block, threads_per_batch);

      // Thread-local partial sum accumulators
      AccumulatorType psum_grad[MaxJoints];

      #pragma unroll
      for (int i = 0; i < njoints; i++) {
        psum_grad[i] = 0.0;
      }

      // Compute jacobian derivative backward pass
      compute_jacobian_derivatives<AccumulatorType>(
        psum_grad, grad_jacobian, cumul_mat, jointMapType, jointMap, linkMap,
        linkChainData, linkChainOffsets, jointLinksData, jointLinksOffsets, joint_affects_endeffector, jointOffset, toolFrameMap, batch, njoints, nlinks, n_tool_frames,
        matAddrBase, elem_idx, threads_per_batch);

      __syncthreads();

      // Reduction step - choose between warp and block reduction
      if constexpr (UseWarpReduce) {
        // Warp-level reduction
        for (int16_t j = 0; j < njoints; j++) {
          psum_grad[j] = curobo::common::warp_reduce_sum(psum_grad[j], threads_per_batch);
        }

        // Write results using thread 0 of each warp
        if (threadIdx.x % 32 == 0) {
          for (int16_t j = 0; j < njoints; j++) {
            grad_joint[batch * njoints + j] = psum_grad[j];
          }
        }
      } else {
        // Block-level reduction
        for (int16_t j = 0; j < njoints; j++) {
          AccumulatorType sum = psum_grad[j];
          curobo::common::block_reduce_sum(sum, threads_per_batch, &block_reduce_shared_data[0], &reduced_sum);
          psum_grad[j] = reduced_sum; // only thread 0 will have the correct value
        }

        // Write results using only thread 0
        if (threadIdx.x == 0) {
          for (int16_t j = 0; j < njoints; j++) {
            grad_joint[batch * njoints + j] = psum_grad[j];
          }
        }
      }
    }

}
}