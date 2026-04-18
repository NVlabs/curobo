/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "collision_pair.cuh"

#include "third_party/helper_math.h"
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "common/block_warp_reductions.cuh"

namespace curobo{
  namespace geometry{
    namespace self_collision {

        template<bool UseShort2=true>
        __forceinline__ __device__ void read_indices(
         int16_t* sphere_pair_idx, // output size 2
        int16_t* global_data,  // shape: [num_collision_pairs, 2]
        uint32_t global_index // shape: [1]
        )
        {
         // This function reads two int16_t values from global_data and stores them in sphere_pair_idx
         // uses short2 to read two int16_t values from global_data as 1 instruction.

         // Standard slow version:
         if (!UseShort2) {
           sphere_pair_idx[0] = global_data[2*global_index];
           sphere_pair_idx[1] = global_data[2*global_index + 1];
         }

         // Fast version:
         if (UseShort2) {
         short2* sphere_pair_idx_short2 = reinterpret_cast<short2*>(sphere_pair_idx);
         short2* pair_locations_short2 = reinterpret_cast<short2*>(&global_data[2*global_index]);
         *sphere_pair_idx_short2 = *pair_locations_short2;
         }

        }

        template<bool UseShort2=true>
        __forceinline__ __device__ void write_indices(
         int16_t* sphere_pair_idx, // output size 2
         int16_t* global_data,  // shape: [num_collision_pairs, 2]
         uint32_t global_index // shape: [1]
        )
        {
         if (UseShort2) {
         short2* sphere_pair_idx_short2 = reinterpret_cast<short2*>(sphere_pair_idx);
         short2* pair_locations_short2 = reinterpret_cast<short2*>(&global_data[2*global_index]);
         *pair_locations_short2 = *sphere_pair_idx_short2;
         }
         if (!UseShort2) {
           global_data[2*global_index] = sphere_pair_idx[0];
           global_data[2*global_index + 1] = sphere_pair_idx[1];
         }
        }

      __forceinline__ __device__ float sphere_squared_distance_fused(float4 sph1, float4 sph2)
         {
           float r_diff = sph1.w + sph2.w; // sum of two radii, radii include respective offsets

           sph1.w = 0.0;
           sph2.w = 0.0;
           float4 position_diff = sph1 - sph2;
           float d_squared = dot(position_diff, position_diff);
           float f_diff = (r_diff * r_diff) - d_squared;
           return f_diff;
         }

        // another kernel that gets the maximum across blocks and writes out the final results
        // Use block max to get max distance
        // if max distance is zero, return (also zero out distance and gradients if needed)




         __forceinline__ __device__ void zero_gradients_simple(
           float* out_gradient,
           uint8_t* sparse_index,
           uint32_t batch_horizon_idx,
           uint32_t num_spheres,
           uint32_t num_threads_per_block)
         {
           const float4 zero_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

           // calculate number of spheres to read per thread:
           uint32_t num_spheres_per_thread = curobo::common::ceil_div(num_spheres, num_threads_per_block);

           for (uint32_t i = 0; i < num_spheres_per_thread; i++) {
             uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;
             //uint8_t sparse_val = (sphere_idx < num_spheres) ?
             //  sparse_index[batch_horizon_idx * num_spheres + sphere_idx] : 0;

             //uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, sparse_val == 1);

             if (sphere_idx < num_spheres) {
               //if (warp_mask != 0) {
                 *(float4 *)&out_gradient[(batch_horizon_idx * num_spheres + sphere_idx) * 4] = zero_vec;
                 sparse_index[batch_horizon_idx * num_spheres + sphere_idx] = 0;
               //}
             }
           }
         }


         __forceinline__ __device__ void zero_gradients_with_ballot(
           float* out_gradient,
           uint8_t* sparse_index,
           uint32_t batch_horizon_idx,
           uint32_t num_spheres,
           uint32_t num_threads_per_block)
         {
           const float4 zero_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

           // calculate number of spheres to read per thread:
           uint32_t num_spheres_per_thread = curobo::common::ceil_div(num_spheres, num_threads_per_block);

           for (uint32_t i = 0; i < num_spheres_per_thread; i++) {
             uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;
             uint8_t sparse_val = (sphere_idx < num_spheres) ?
               sparse_index[batch_horizon_idx * num_spheres + sphere_idx] : 0;

             uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, sparse_val == 1);

             if (sphere_idx < num_spheres) {
               if (warp_mask != 0) {
                 *(float4 *)&out_gradient[(batch_horizon_idx * num_spheres + sphere_idx) * 4] = zero_vec;
                 sparse_index[batch_horizon_idx * num_spheres + sphere_idx] = 0;
               }
             }
           }
         }
         template<bool USE_BALLOT=true>
         __forceinline__ __device__ void zero_gradients(
           float* out_gradient,
           uint8_t* sparse_index,
           uint32_t batch_horizon_idx,
           uint32_t num_spheres,
           uint32_t num_threads_per_block)
         {
           if (USE_BALLOT) {
             zero_gradients_with_ballot(out_gradient, sparse_index, batch_horizon_idx, num_spheres, num_threads_per_block);
           } else {
             zero_gradients_simple(out_gradient, sparse_index, batch_horizon_idx, num_spheres, num_threads_per_block);
           }
         }

         __forceinline__ __device__ void load_spheres_and_zero_gradients(
           float4* spheres_shared,
           float* out_gradient,
           uint8_t* sparse_index,
           const float* robot_spheres,
           const float* offsets,
           uint32_t batch_horizon_idx,
           uint32_t num_spheres,
           uint32_t num_threads_per_block)
         {
           const float4 zero_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

           // calculate number of spheres to read per thread:
           uint32_t num_spheres_per_thread = curobo::common::ceil_div(num_spheres, num_threads_per_block);
           float4 temp_sphere;
           float temp_offset;
           // Combined loop: read spheres into shared memory AND zero out gradients/sparsity indices
           for (uint32_t i = 0; i < num_spheres_per_thread; i++) {

             // index major
             uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;

             // All threads participate in ballot sync (use safe values for out-of-bounds threads)
             uint8_t sparse_val = (sphere_idx < num_spheres) ?
               sparse_index[batch_horizon_idx * num_spheres + sphere_idx] : 0;
             uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, sparse_val == 1);

             if (sphere_idx < num_spheres) {
               // Read and store sphere data
               temp_sphere = *(float4 *)&robot_spheres[4 * ((batch_horizon_idx * num_spheres) + sphere_idx)];
               temp_offset = offsets[sphere_idx];
               temp_sphere.w += temp_offset;
               spheres_shared[sphere_idx] = temp_sphere;

               // Only execute if at least one thread in warp needs to do work
               if (warp_mask != 0) {
                   *(float4 *)&out_gradient[(batch_horizon_idx * num_spheres + sphere_idx) * 4] = zero_vec;
                   sparse_index[batch_horizon_idx * num_spheres + sphere_idx] = 0;
               }
             }
           }
         }

         __forceinline__ __device__ void load_spheres(
           float4* spheres_shared,
           const float* robot_spheres,
           const float* offsets,
           uint32_t batch_horizon_idx,
           uint32_t num_spheres,
           uint32_t num_threads_per_block)
         {

           // calculate number of spheres to read per thread:
           uint32_t num_spheres_per_thread = curobo::common::ceil_div(num_spheres, num_threads_per_block);
           float4 temp_sphere;
           float temp_offset;
           // Combined loop: read spheres into shared memory AND zero out gradients/sparsity indices
           for (uint32_t i = 0; i < num_spheres_per_thread; i++) {

             // index major
             uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;

             // All threads participate in ballot sync (use safe values for out-of-bounds threads)


             if (sphere_idx < num_spheres) {
               // Read and store sphere data
               temp_sphere = *(float4 *)&robot_spheres[4 * ((batch_horizon_idx * num_spheres) + sphere_idx)];
               temp_offset = offsets[sphere_idx];
               temp_sphere.w += temp_offset;
               spheres_shared[sphere_idx] = temp_sphere;
             }
           }
         }

         template<bool STORE_PAIR_DISTANCE>
         __forceinline__ __device__ CollisionPair compute_max_collision_distance(
           float4* spheres_shared,
           float* pair_distance,
           int16_t* pair_locations,
           uint32_t batch_horizon_idx,
           uint32_t num_collision_pairs,
           uint32_t num_collision_pairs_per_block,
           uint32_t num_threads_per_block,
           CollisionPair* block_reduce_shared_data,
           CollisionPair* reduced_max_d,
           const int start_index)
         {
           CollisionPair max_d = {0.0, 0, 0};
           int16_t sphere_pair_idx[2];


             uint32_t num_checks_per_thread = curobo::common::ceil_div(num_collision_pairs_per_block, num_threads_per_block);
           for (uint32_t i = 0; i < num_checks_per_thread; i++) {
             uint32_t check_idx = start_index + threadIdx.x + i * num_threads_per_block;
             if (check_idx < num_collision_pairs) {

               // pair_locations: int16_t:
               // read two int16_t values by reinterpretcast to short2:

               read_indices(sphere_pair_idx, pair_locations, check_idx);


               // compute distance
               // Read two spheres from shared memory:
               float4 sph1 = spheres_shared[sphere_pair_idx[0]];
               float4 sph2 = spheres_shared[sphere_pair_idx[1]];

               float valid_mask = (sph1.w >= 0.0f && sph2.w >= 0.0f) ? 1.0f : 0.0f;
               float f_diff = sphere_squared_distance_fused(sph1, sph2) * valid_mask;

               CollisionPair new_distance = {f_diff, sphere_pair_idx[0], sphere_pair_idx[1]};

               max_d = max(max_d, new_distance);
               if (STORE_PAIR_DISTANCE)
               {
                 pair_distance[batch_horizon_idx * num_collision_pairs + check_idx] = new_distance.d; // should we write 0 and negative values?
               }
             }
           }


           // blockwide max to get max distance
           curobo::common::block_reduce_max(max_d, num_threads_per_block, &block_reduce_shared_data[0], reduced_max_d);
           max_d = *reduced_max_d;
           return max_d;
         }

         template<bool LOAD_SPHERES_FROM_SHARED>
         __forceinline__ __device__ void finalize_collision_results(
           CollisionPair max_d,
           const float4* spheres_shared,
           const float* offsets,
           float* out_distance,
           float* out_gradient,
           uint8_t* sparse_index,
           uint32_t batch_horizon_idx,
           uint32_t batch_idx,
           uint32_t num_spheres,
           float local_weight,
           bool write_grad
           )
         {


           if(max_d.d <= 0.0f) {
           out_distance[batch_horizon_idx] = 0.0f;
           return;
           }


           //////////////////////////////////////////////////////
           // Write out the final results
           //////////////////////////////////////////////////////
           float4 dist_vec = make_float4(0,0,0,0);

           max_d.d = 0.5 * local_weight * max_d.d;

           if (write_grad)
           {
             float4 sph1_collision = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
             float4 sph2_collision = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
             if (LOAD_SPHERES_FROM_SHARED)
             {
             sph1_collision = spheres_shared[max_d.i];
             sph2_collision = spheres_shared[max_d.j];
             }
             else {
               sph1_collision = spheres_shared[(batch_horizon_idx) * num_spheres + max_d.i];
               sph2_collision = spheres_shared[(batch_horizon_idx) * num_spheres + max_d.j];
               sph1_collision.w += offsets[max_d.i];
               sph2_collision.w += offsets[max_d.j];
             }
             // calculate signed distance using api:

             dist_vec = local_weight * (sph2_collision - sph1_collision);
             dist_vec.w = local_weight * -1.0f;


           }


           out_distance[batch_horizon_idx] = max_d.d;

           if (write_grad)
           {
             // NOTE: spheres can also be read from rs_shared

             *(float4 *)&out_gradient[(batch_horizon_idx) * num_spheres * 4 + max_d.i * 4] =
               dist_vec;

             dist_vec = -1.0f * dist_vec;
             dist_vec.w = local_weight * -1.0f;
             *(float4 *)&out_gradient[(batch_horizon_idx) * num_spheres * 4 + max_d.j * 4] =
                dist_vec;
             sparse_index[(batch_horizon_idx) * num_spheres + max_d.i] = 1;
             sparse_index[(batch_horizon_idx) * num_spheres + max_d.j] = 1;
           }
         }

    }
}
}