/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <cooperative_groups.h>

#include "common/math.cuh"
#include "common/block_warp_reductions.cuh"
#include "common/pose_util.cuh"
#include "common/quaternion_util.cuh"

#include "kinematics_constants.h"
#include "third_party/helper_math.h"
#include "kinematics_joint_util.cuh"
#include "kinematics_util.cuh"
#include "kinematics_forward_helper.cuh"
#include "kinematics_backward_helper.cuh"
#include "kinematics_jacobian_backward_helper.cuh"




namespace curobo{
namespace kinematics{




    // Unified kernel that branches only at the reduction step
    template<
      typename scalar_t,
      typename psum_t,
      int16_t MAX_JOINTS,
      bool USE_WARP_REDUCE,
      bool COMPUTE_COM=false,
      bool RECOMPUTE_CUMUL=false,
      int N_LINKS=-1,
      bool RECOMPUTE_PRECOMPUTE=false>
    __global__ void kinematics_fused_backward_unified_kernel(
      float *grad_out_link_q,       // batchSize * njoints
      const float *grad_nlinks_pos, // batchSize * n_tool_frames * 16
      const float *grad_nlinks_quat,
      const scalar_t *grad_spheres,    // batchSize * nspheres * 4
      const float *grad_center_of_mass, // [batch_size * 4] - xyz=pos grad, w=mass grad (ignored)
      const float *batch_center_of_mass, // [batch_size * 4] - xyz=global CoM, w=total mass (INPUT)
      const float *global_cumul_mat,
      const float *q,               // batchSize * njoints
      const float *fixedTransform,  // nlinks * 16
      const float *robotSpheres,    // batchSize * nspheres * 4
      const float *link_masses_com,   // [n_links * 4] - xyz=local CoM, w=mass
      const int8_t *jointMapType,      // nlinks
      const int16_t *jointMap,         // nlinks
      const int16_t *linkMap,          // nlinks
      const int16_t *toolFrameMap,     // n_tool_frames
      const int16_t *linkSphereMap,    // nspheres
      const int32_t *env_query_idx,    // batchSize - env index per batch
      const int16_t *linkChainData,    // NEW: Packed actual link indices
      const int16_t *linkChainOffsets, // NEW: Offset for each link's chain
      const float *jointOffset,       // nlinks * 2
      const int batch_size, const int horizon,
      const int nspheres,
      const int nlinks, const int njoints, const int n_tool_frames,
      const int num_envs,
      const int threads_per_batch)
    {
      extern __shared__ __align__(16) float cumul_mat[];

      // Shared memory for block reduction (only used when !USE_WARP_REDUCE)
      __shared__ psum_t block_reduce_shared_data[32];
      psum_t reduced_sum;

      // Common indexing logic (same for both kernels)
      int t = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / threads_per_batch;
      const bool active_batch = batch < batch_size;

      if (!active_batch && !(RECOMPUTE_CUMUL && RECOMPUTE_PRECOMPUTE))
          return;

      const int elem_idx = threadIdx.x % threads_per_batch;
      const int local_batch = threadIdx.x / threads_per_batch;
      const int matAddrBase = local_batch * nlinks * 12;

      // Calculate start batch index for this block
      const int start_batch_index = blockIdx.x * (blockDim.x / threads_per_batch);
      const int batches_per_block = (blockDim.x + threads_per_batch - 1) / threads_per_batch;

      if constexpr (RECOMPUTE_CUMUL && RECOMPUTE_PRECOMPUTE) {
        float *local_mat = &cumul_mat[batches_per_block * nlinks * 12];

        compute_local_link_transforms<N_LINKS>(
          local_mat,
          q,
          fixedTransform,
          jointMapType,
          jointMap,
          jointOffset,
          start_batch_index,
          batch_size,
          nlinks,
          njoints,
          batches_per_block
        );

        const int actual_batches_per_block = min(batches_per_block, batch_size - start_batch_index);
        const int active_phase2_threads = actual_batches_per_block * 16;

        if (threadIdx.x < active_phase2_threads) {
          const int phase2_local_batch = threadIdx.x >> 4;
          const int phase2_lane_idx = threadIdx.x & 15;
          const int phase2_matAddrBase = phase2_local_batch * nlinks * 12;
          const unsigned int phase2_mask = (threadIdx.x & 16) == 0 ? 0x0000ffffu : 0xffff0000u;

          load_base_transform_halfwarp(
            cumul_mat, fixedTransform, nlinks, phase2_matAddrBase, phase2_lane_idx, phase2_mask);

          compute_cumulative_transforms_halfwarp<N_LINKS>(
            cumul_mat,
            local_mat,
            linkMap,
            nlinks,
            phase2_matAddrBase,
            phase2_matAddrBase,
            phase2_lane_idx,
            phase2_mask
          );
        }
        __syncthreads();
        if (!active_batch) {
          return;
        }
      } else if constexpr (RECOMPUTE_CUMUL) {
        auto block = cooperative_groups::this_thread_block();
        auto tile4 = cooperative_groups::tiled_partition<4>(block);

        if (elem_idx < 4) {
          const int col_idx = tile4.thread_rank();
          load_base_transform_tile(cumul_mat, fixedTransform, matAddrBase, tile4, col_idx, 4);
          compute_cumulative_transforms_tile<N_LINKS>(
              cumul_mat, q, fixedTransform, jointMapType, jointMap, linkMap, jointOffset,
              batch, njoints, nlinks, matAddrBase, tile4, col_idx);
        }
        __syncthreads();
      } else {
        // Load cumulative matrices to shared memory using unified function
        read_cumul_to_shared_mem(cumul_mat, global_cumul_mat, start_batch_index,
                                 threadIdx.x, nlinks,
                                 batch_size, batches_per_block, threads_per_batch);
      }

      // Thread-local partial sum accumulators
      psum_t psum_grad[MAX_JOINTS];

      #pragma unroll
      for (int i = 0; i < njoints; i++)
      {
          psum_grad[i] = 0.0;
      }

      // Compute sphere gradients using extracted device function
      compute_sphere_gradients<scalar_t, psum_t>(
          psum_grad, grad_spheres, cumul_mat, robotSpheres,
          linkSphereMap, env_query_idx, linkChainData, linkChainOffsets, jointMapType, jointMap, jointOffset,
          batch, nspheres, num_envs, horizon, nlinks, njoints,
          elem_idx, threads_per_batch, matAddrBase);

      // Compute link gradients using unified device function
      compute_link_gradients<psum_t>(
          psum_grad, grad_nlinks_pos, grad_nlinks_quat, cumul_mat,
          toolFrameMap, linkChainData, linkChainOffsets, jointMapType, jointMap, jointOffset,
          batch, n_tool_frames, nlinks, njoints,
          elem_idx, threads_per_batch, matAddrBase);


      if constexpr (COMPUTE_COM)
      {
      // Read total mass from pre-computed center of mass (more efficient than recomputation)
      // Total mass is stored as the 4th element (w component) of each batch entry
      float total_mass = batch_center_of_mass[batch * 4 + 3];

      // Compute center of mass gradients using extracted device function
      if (total_mass > 0.0f) {  // Only compute if robot has mass
          compute_center_of_mass_gradients<psum_t>(
              psum_grad, grad_center_of_mass, cumul_mat, link_masses_com, total_mass,
              linkChainData, linkChainOffsets, jointMapType, jointMap, jointOffset,
              batch, nlinks, njoints,
              elem_idx, threads_per_batch, matAddrBase);
      }
     }

      __syncthreads();

      // ONLY BRANCHING POINT: Choose reduction method
      if constexpr (USE_WARP_REDUCE) {
          // Warp-level reduction (original multi-block approach)
          for (int16_t j = 0; j < njoints; j++)
          {
              psum_grad[j] = curobo::common::warp_reduce_sum(psum_grad[j], threads_per_batch);
          }

          // Write results using thread 0 of each warp
          if (threadIdx.x % 32 == 0)
          {
              for (int16_t j = 0; j < njoints; j++)
              {
                  grad_out_link_q[batch * njoints + j] = psum_grad[j];
              }
          }
      } else {
          // Block-level reduction (original single-block approach)
          for (int16_t j = 0; j < njoints; j++)
          {
              psum_t sum = psum_grad[j];
              curobo::common::block_reduce_sum(sum, threads_per_batch, &block_reduce_shared_data[0], &reduced_sum);
              psum_grad[j] = reduced_sum; // only thread 0 will have the correct value
          }

          // Write results using only thread 0
          if (threadIdx.x == 0)
          {
              for (int16_t j = 0; j < njoints; j++)
              {
                  grad_out_link_q[batch * njoints + j] = psum_grad[j];
              }
          }
      }
    }

    // Experimental saved-cumul backward kernel that can also consume the
    // Jacobian-output gradient. This keeps batch_cumul_mat as the forward/
    // backward contract and only fuses the two backward reductions.
    template<
      typename scalar_t,
      typename psum_t,
      int16_t MAX_JOINTS,
      bool USE_WARP_REDUCE,
      bool COMPUTE_COM=false,
      bool COMPUTE_JACOBIAN_GRAD=false>
    __global__ void kinematics_fused_backward_saved_cumul_optional_jacobian_kernel(
      float *grad_out_link_q,       // batchSize * njoints
      const float *grad_nlinks_pos, // batchSize * n_tool_frames * 16
      const float *grad_nlinks_quat,
      const scalar_t *grad_spheres,    // batchSize * nspheres * 4
      const float *grad_center_of_mass, // [batch_size * 4] - xyz=pos grad, w=mass grad (ignored)
      const float *batch_center_of_mass, // [batch_size * 4] - xyz=global CoM, w=total mass (INPUT)
      const float *grad_jacobian,       // batchSize * n_tool_frames * 6 * njoints
      const float *global_cumul_mat,
      const float *robotSpheres,    // batchSize * nspheres * 4
      const float *link_masses_com,   // [n_links * 4] - xyz=local CoM, w=mass
      const int8_t *jointMapType,      // nlinks
      const int16_t *jointMap,         // nlinks
      const int16_t *linkMap,          // nlinks
      const int16_t *toolFrameMap,     // n_tool_frames
      const int16_t *linkSphereMap,    // nspheres
      const int32_t *env_query_idx,    // batchSize - env index per batch
      const int16_t *linkChainData,    // Packed actual link indices
      const int16_t *linkChainOffsets, // Offset for each link's chain
      const int16_t *jointLinksData,   // Packed links per joint
      const int16_t *jointLinksOffsets, // Offset for each joint's links
      const bool *joint_affects_endeffector,
      const float *jointOffset,       // nlinks * 2
      const int batch_size, const int horizon,
      const int nspheres,
      const int nlinks, const int njoints, const int n_tool_frames,
      const int num_envs,
      const int threads_per_batch)
    {
      extern __shared__ __align__(16) float cumul_mat[];

      __shared__ psum_t block_reduce_shared_data[32];
      psum_t reduced_sum;

      int t = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / threads_per_batch;
      if (batch >= batch_size)
          return;

      const int elem_idx = threadIdx.x % threads_per_batch;
      const int local_batch = threadIdx.x / threads_per_batch;
      const int matAddrBase = local_batch * nlinks * 12;

      const int start_batch_index = blockIdx.x * (blockDim.x / threads_per_batch);
      const int batches_per_block = (blockDim.x + threads_per_batch - 1) / threads_per_batch;

      read_cumul_to_shared_mem(cumul_mat, global_cumul_mat, start_batch_index,
                               threadIdx.x, nlinks,
                               batch_size, batches_per_block, threads_per_batch);

      psum_t psum_grad[MAX_JOINTS];

      #pragma unroll
      for (int i = 0; i < njoints; i++)
      {
          psum_grad[i] = 0.0;
      }

      compute_sphere_gradients<scalar_t, psum_t>(
          psum_grad, grad_spheres, cumul_mat, robotSpheres,
          linkSphereMap, env_query_idx, linkChainData, linkChainOffsets, jointMapType, jointMap, jointOffset,
          batch, nspheres, num_envs, horizon, nlinks, njoints,
          elem_idx, threads_per_batch, matAddrBase);

      compute_link_gradients<psum_t>(
          psum_grad, grad_nlinks_pos, grad_nlinks_quat, cumul_mat,
          toolFrameMap, linkChainData, linkChainOffsets, jointMapType, jointMap, jointOffset,
          batch, n_tool_frames, nlinks, njoints,
          elem_idx, threads_per_batch, matAddrBase);

      if constexpr (COMPUTE_COM)
      {
        float total_mass = batch_center_of_mass[batch * 4 + 3];
        if (total_mass > 0.0f) {
          compute_center_of_mass_gradients<psum_t>(
              psum_grad, grad_center_of_mass, cumul_mat, link_masses_com, total_mass,
              linkChainData, linkChainOffsets, jointMapType, jointMap, jointOffset,
              batch, nlinks, njoints,
              elem_idx, threads_per_batch, matAddrBase);
        }
      }

      if constexpr (COMPUTE_JACOBIAN_GRAD)
      {
        compute_jacobian_derivatives<psum_t, true>(
          psum_grad, grad_jacobian, cumul_mat, jointMapType, jointMap, linkMap,
          linkChainData, linkChainOffsets, jointLinksData, jointLinksOffsets,
          joint_affects_endeffector, jointOffset, toolFrameMap, batch, njoints,
          nlinks, n_tool_frames, matAddrBase, elem_idx, threads_per_batch);
      }

      __syncthreads();

      if constexpr (USE_WARP_REDUCE) {
          for (int16_t j = 0; j < njoints; j++)
          {
              psum_grad[j] = curobo::common::warp_reduce_sum(psum_grad[j], threads_per_batch);
          }

          if (threadIdx.x % 32 == 0)
          {
              for (int16_t j = 0; j < njoints; j++)
              {
                  grad_out_link_q[batch * njoints + j] = psum_grad[j];
              }
          }
      } else {
          for (int16_t j = 0; j < njoints; j++)
          {
              psum_t sum = psum_grad[j];
              curobo::common::block_reduce_sum(sum, threads_per_batch, &block_reduce_shared_data[0], &reduced_sum);
              psum_grad[j] = reduced_sum;
          }

          if (threadIdx.x == 0)
          {
              for (int16_t j = 0; j < njoints; j++)
              {
                  grad_out_link_q[batch * njoints + j] = psum_grad[j];
              }
          }
      }
    }

}
}
