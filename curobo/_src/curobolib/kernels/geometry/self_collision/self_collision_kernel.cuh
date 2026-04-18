/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "collision_pair.cuh"

#include "third_party/helper_math.h"
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "common/block_warp_reductions.cuh"
#include "self_collision_helper.cuh"

namespace curobo{
  namespace geometry{
    namespace self_collision {



    template<bool STORE_PAIR_DISTANCE>
    __global__ void self_collision_max_distance_kernel(
      float *out_distance,        // batch x 1
      float *out_gradient,             // batch x nspheres x 4
      float *pair_distance,       // batch x num_collision_pairs * 2
      uint8_t *sparse_index, // batch x nspheres
      const float *robot_spheres, // batch x horizon x nspheres x 4
      const float *offsets,       // nspheres
      const float *weight,
      int16_t *pair_locations,
      const int batch_size,
      const int horizon,
      const int num_spheres,
      const int num_collision_pairs,
      const bool write_grad
      )
    {
      extern __shared__ float4 spheres_shared[];

      __shared__ CollisionPair block_reduce_shared_data[32];
      __shared__ CollisionPair reduced_max_d;

      // Initialize shared memory if needed


      uint32_t batch_horizon_idx = blockIdx.x;
      uint32_t batch_idx = batch_horizon_idx / horizon ;
      if (batch_idx >= batch_size || batch_horizon_idx >= batch_size * horizon)
      {
        return;
      }

      const float local_weight = weight[0];


      // uint32_t horizon_idx = batch_horizon_idx % horizon;
      // get how many threads were launched:
      const uint32_t num_threads_per_block = blockDim.x;


      // Load spheres into shared memory and initialize gradients
      load_spheres_and_zero_gradients(
        spheres_shared,
        out_gradient,
        sparse_index,
        robot_spheres,
        offsets,
        batch_horizon_idx,
        num_spheres,
        num_threads_per_block);


      __syncthreads();

      // Call the new device function to compute max distance
      CollisionPair max_d = compute_max_collision_distance<STORE_PAIR_DISTANCE>(
        spheres_shared,
        pair_distance,
        pair_locations,
        batch_horizon_idx,
        num_collision_pairs,
        num_collision_pairs,
        num_threads_per_block,
        &block_reduce_shared_data[0],
        &reduced_max_d,
        0);

      if (threadIdx.x > 0)
      {
        return;
      }


      // Process final collision results
      finalize_collision_results<true>(
        max_d,
        spheres_shared,
        offsets,
        out_distance,
        out_gradient,
        sparse_index,
        batch_horizon_idx,
        batch_idx,
        num_spheres,
        local_weight,
        write_grad
        );





    } // end of kernel

 template<bool STORE_PAIR_DISTANCE>
    __global__ void self_collision_max_block_kernel(
      float *out_gradient,             // batch x nspheres x 4
      float *pair_distance,       // batch x num_collision_pairs * 2
      uint8_t *sparse_index, // batch x nspheres
      const float *robot_spheres, // batch x horizon x nspheres x 4
      const float *offsets,       // nspheres
      int16_t *pair_locations, // This is to be mapped to n_spheres index in robot_spheres.
      float *block_batch_max_value,
      int16_t *block_batch_max_index,
      const int num_blocks_per_batch,
      const int batch_size,
      const int horizon,
      const int num_spheres,
      const int num_collision_pairs
      )
    {
      extern __shared__ float4 spheres_shared[];

      __shared__ CollisionPair block_reduce_shared_data[32];
      __shared__ CollisionPair reduced_max_d;

      // Initialize shared memory if needed
      if (threadIdx.x == 0) {
          reduced_max_d = {0.0f, 0, 0};
      }

      uint32_t batch_horizon_idx = blockIdx.x / num_blocks_per_batch;
      uint32_t local_block_idx = blockIdx.x % num_blocks_per_batch;
      uint32_t batch_idx = batch_horizon_idx / horizon ;
      if (batch_idx >= batch_size || batch_horizon_idx >= batch_size * horizon)
      {
        return;
      }



      // uint32_t horizon_idx = batch_horizon_idx % horizon;
      // get how many threads were launched:
      const uint32_t num_threads_per_block = blockDim.x;
      const uint32_t num_collision_pairs_per_block = curobo::common::ceil_div(num_collision_pairs, num_blocks_per_batch);

      load_spheres(
        spheres_shared,
        robot_spheres,
        offsets,
        batch_horizon_idx,
        num_spheres,
        num_threads_per_block);

      __syncthreads();

      const int start_index = local_block_idx * num_collision_pairs_per_block;
      // Call the new device function to compute max distance
      CollisionPair max_d = compute_max_collision_distance<STORE_PAIR_DISTANCE>(
        spheres_shared,
        pair_distance,
        pair_locations,
        batch_horizon_idx,
        num_collision_pairs,
        num_collision_pairs_per_block,
        num_threads_per_block,
        &block_reduce_shared_data[0],
        &reduced_max_d,
        start_index);

      if (threadIdx.x > 0)
      {
        return;
      }


      // write out max collision pair:
      block_batch_max_value[num_blocks_per_batch * batch_horizon_idx + local_block_idx] = max_d.d;
      // write as short2:
      int16_t __align__(8) sphere_pair_idx[2] = {max_d.i, max_d.j};
      write_indices(sphere_pair_idx, block_batch_max_index, num_blocks_per_batch * batch_horizon_idx + local_block_idx);





    } // end of kernel

    __global__ void self_collision_max_reduce_kernel(
      float *out_distance,        // batch x 1
      float *out_gradient,             // batch x nspheres x 4
      float *pair_distance,       // batch x num_collision_pairs * 2
      uint8_t *sparse_index, // batch x nspheres
      const float *robot_spheres, // batch x horizon x nspheres x 4
      const float *offsets,       // nspheres
      const float *weight,
      int16_t *pair_locations,
      float *block_batch_max_value,
      int16_t *block_batch_max_index,
      const int num_blocks_per_batch,
      const int batch_size,
      const int horizon,
      const int num_spheres,
      const int num_collision_pairs,
      const bool write_grad
      )
      {

        __shared__ CollisionPair block_reduce_shared_data[32];
        __shared__ CollisionPair reduced_max_d;

        // Initialize shared memory if needed
        if (threadIdx.x == 0) {
            reduced_max_d = {0.0f, 0, 0};
        }

        // Launches num_blocks_per_batch threads
        const int batch_horizon_idx = blockIdx.x;
        const int batch_idx = batch_horizon_idx / horizon;
        const float local_weight = weight[0];

        const int thread_idx = threadIdx.x;
        const int num_threads_per_block = blockDim.x;

        if (thread_idx >= num_blocks_per_batch)
        {
          return;
        }
        // zero gradients:
        zero_gradients(out_gradient, sparse_index, batch_horizon_idx, num_spheres, num_threads_per_block);
        CollisionPair max_d = {0.0f, 0, 0};

        // if num_blocks_per_batch > num_threads_per_block, use a for loop to reduce across blocks
        int16_t sphere_pair_idx[2];
        const int start_batch_idx = batch_horizon_idx * num_blocks_per_batch;
        const int blocks_per_thread = curobo::common::ceil_div(num_blocks_per_batch, num_threads_per_block);
        // each thread will read blocks_per_thread = num_blocks_per_batch / num_threads_per_block blocks
        // first thread will read 0, blocks_per_thread, 2*blocks_per_thread, ...
        // second thread will read 1, blocks_per_thread+1, 2*blocks_per_thread+1, ...
        for (int i = 0; i < blocks_per_thread; i++)
        {
          // read collision pair

          int block_idx = i * num_threads_per_block + thread_idx;

          if (block_idx >= num_blocks_per_batch)
          {
            break;
          }
          block_idx = start_batch_idx + block_idx;

          // read float
          float d = block_batch_max_value[block_idx];
          // read as short:
          read_indices(sphere_pair_idx, block_batch_max_index, block_idx);

          CollisionPair current_pair = {d, sphere_pair_idx[0], sphere_pair_idx[1]};

          max_d = max(max_d, current_pair);


        }


        // reduce across block:
        curobo::common::block_reduce_max(max_d, num_threads_per_block, &block_reduce_shared_data[0], &reduced_max_d);

        if (thread_idx > 0)
        {
          return;
        }

        max_d = reduced_max_d;
        // Process final collision results
        finalize_collision_results<false>(
          max_d,
          reinterpret_cast<const float4*>(robot_spheres), // float*
          offsets,
          out_distance,
          out_gradient,
          sparse_index,
          batch_horizon_idx,
          batch_idx,
          num_spheres,
          local_weight,
          write_grad
          );

      }



    } // namespace collision
  } // namespace geometry
}   // namespace curobo
