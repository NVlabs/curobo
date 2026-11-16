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

      constexpr float LINK_AABB_EPSILON = 1.0e-5f;

      __forceinline__ __device__ float sphere_squared_distance_fused(
        const float4 sph1,
        const float4 sph2)
         {
           const float radius_sum = sph1.w + sph2.w;
           const float dx = sph1.x - sph2.x;
           const float dy = sph1.y - sph2.y;
           const float dz = sph1.z - sph2.z;
           const float distance_squared = dx * dx + dy * dy + dz * dz;
           return radius_sum * radius_sum - distance_squared;
         }

      __forceinline__ __device__ bool link_aabbs_overlap(
        const float4 lower_a,
        const float4 upper_a,
        const float4 lower_b,
        const float4 upper_b)
      {
        if (lower_a.w < 0.0f || lower_b.w < 0.0f)
        {
          return false;
        }
        return lower_a.x <= upper_b.x && lower_b.x <= upper_a.x &&
          lower_a.y <= upper_b.y && lower_b.y <= upper_a.y &&
          lower_a.z <= upper_b.z && lower_b.z <= upper_a.z;
      }

      __forceinline__ __device__ void compute_link_aabbs(
        const float4* spheres_shared,
        float4* link_aabbs_shared,
        const int16_t* link_sphere_indices,
        const int32_t* link_sphere_offsets,
        const int num_collision_links)
      {
        for (int link_idx = threadIdx.x;
             link_idx < num_collision_links;
             link_idx += blockDim.x)
        {
          float lower_x = CUDART_INF_F;
          float lower_y = CUDART_INF_F;
          float lower_z = CUDART_INF_F;
          float upper_x = -CUDART_INF_F;
          float upper_y = -CUDART_INF_F;
          float upper_z = -CUDART_INF_F;
          bool has_active_sphere = false;
          const int sphere_start = link_sphere_offsets[link_idx];
          const int sphere_end = link_sphere_offsets[link_idx + 1];
          for (int index = sphere_start; index < sphere_end; ++index)
          {
            const float4 sphere = spheres_shared[link_sphere_indices[index]];
            if (sphere.w >= 0.0f)
            {
              lower_x = fminf(lower_x, sphere.x - sphere.w);
              lower_y = fminf(lower_y, sphere.y - sphere.w);
              lower_z = fminf(lower_z, sphere.z - sphere.w);
              upper_x = fmaxf(upper_x, sphere.x + sphere.w);
              upper_y = fmaxf(upper_y, sphere.y + sphere.w);
              upper_z = fmaxf(upper_z, sphere.z + sphere.w);
              has_active_sphere = true;
            }
          }

          float4 lower = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
          float4 upper = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
          if (has_active_sphere)
          {
            lower = make_float4(
              lower_x - LINK_AABB_EPSILON,
              lower_y - LINK_AABB_EPSILON,
              lower_z - LINK_AABB_EPSILON,
              1.0f);
            upper = make_float4(
              upper_x + LINK_AABB_EPSILON,
              upper_y + LINK_AABB_EPSILON,
              upper_z + LINK_AABB_EPSILON,
              1.0f);
          }
          link_aabbs_shared[2 * link_idx] = lower;
          link_aabbs_shared[2 * link_idx + 1] = upper;
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

             if (sphere_idx < num_spheres) {
               const uint32_t output_idx =
                 batch_horizon_idx * num_spheres + sphere_idx;
               // Read and store sphere data
               temp_sphere = *(float4 *)&robot_spheres[4 * output_idx];
               temp_offset = offsets[sphere_idx];
               temp_sphere.w += temp_offset;
               spheres_shared[sphere_idx] = temp_sphere;

               if (sparse_index[output_idx] != 0) {
                   *(float4 *)&out_gradient[output_idx * 4] = zero_vec;
                   sparse_index[output_idx] = 0;
               }
             }
           }
         }

         template<bool StorePairDistance>
         __forceinline__ __device__ CollisionPair compute_max_collision_distance(
           float4* spheres_shared,
           float* pair_distance,
           const int16_t* pair_locations,
           uint32_t batch_horizon_idx,
           uint32_t num_collision_pairs,
           uint32_t num_threads_per_block,
           CollisionPair* block_reduce_shared_data)
         {
           CollisionPair max_d = {0.0f, 0, 0};
           const uint32_t num_checks_per_thread =
             curobo::common::ceil_div(num_collision_pairs, num_threads_per_block);
           for (uint32_t i = 0; i < num_checks_per_thread; i++) {
             const uint32_t check_idx = threadIdx.x + i * num_threads_per_block;
             if (check_idx < num_collision_pairs) {
               const short2 sphere_pair =
                 reinterpret_cast<const short2*>(pair_locations)[check_idx];


               // compute distance
               // Read two spheres from shared memory:
               const float4 sph1 = spheres_shared[sphere_pair.x];
               const float4 sph2 = spheres_shared[sphere_pair.y];

               const float valid_mask =
                 (sph1.w >= 0.0f && sph2.w >= 0.0f) ? 1.0f : 0.0f;
               const float f_diff = sphere_squared_distance_fused(sph1, sph2) * valid_mask;

               const CollisionPair new_distance = {f_diff, sphere_pair.x, sphere_pair.y};

               max_d = max(max_d, new_distance);
               if constexpr (StorePairDistance)
               {
                 pair_distance[batch_horizon_idx * num_collision_pairs + check_idx] = new_distance.d; // should we write 0 and negative values?
               }
             }
           }

           // Most warps see no collision. Skip their shuffle reduction, publish
           // one maximum per warp, then let thread 0 scan the small warp-result
           // array. Only thread 0 consumes the returned block maximum.
           const unsigned nonzero_mask =
             __ballot_sync(curobo::common::fullMask, max_d.d > 0.0f);
           if (nonzero_mask != 0) {
             #pragma unroll
             for (int offset = curobo::common::warpSize / 2;
                  offset > 0;
                  offset /= 2) {
               const CollisionPair other =
                 __shfl_down_sync(curobo::common::fullMask, max_d, offset);
               if ((threadIdx.x & 31) + offset < curobo::common::warpSize &&
                   other.d > max_d.d) {
                 max_d = other;
               }
             }
           }
           if ((threadIdx.x & 31) == 0) {
             block_reduce_shared_data[threadIdx.x / curobo::common::warpSize] = max_d;
           }
           __syncthreads();
           if (threadIdx.x == 0) {
             const uint32_t num_warps =
               num_threads_per_block / curobo::common::warpSize;
             #pragma unroll
             for (uint32_t warp_idx = 1; warp_idx < num_warps; ++warp_idx) {
               const CollisionPair other = block_reduce_shared_data[warp_idx];
               if (other.d > max_d.d) {
                 max_d = other;
               }
             }
           }
           return max_d;
         }

         __forceinline__ __device__ void zero_shared_gradients(
           float4* gradients_shared,
           const uint32_t num_spheres,
           const uint32_t num_threads_per_block)
         {
           const float4 zero_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
           const uint32_t num_spheres_per_thread =
             curobo::common::ceil_div(num_spheres, num_threads_per_block);
           for (uint32_t i = 0; i < num_spheres_per_thread; ++i) {
             const uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;
             if (sphere_idx < num_spheres) {
               gradients_shared[sphere_idx] = zero_vec;
             }
           }
         }

         // Reduce contributions for an equal sphere index within the active warp
         // before updating shared memory. Every active lane must call this helper.
         __forceinline__ __device__ void warp_aggregate_collision_gradient(
           float4* gradients_shared,
           const int sphere_idx,
           const float4 gradient,
           const float local_weight)
         {
           const unsigned active_mask = __activemask();
           const unsigned peers = __match_any_sync(active_mask, sphere_idx);
           float gradient_x = gradient.x;
           float gradient_y = gradient.y;
           float gradient_z = gradient.z;
           if (__popc(peers) > 1) {
             gradient_x = 0.0f;
             gradient_y = 0.0f;
             gradient_z = 0.0f;
             unsigned remaining = peers;
             while (remaining != 0) {
               const int source_lane = __ffs(remaining) - 1;
               gradient_x += __shfl_sync(peers, gradient.x, source_lane);
               gradient_y += __shfl_sync(peers, gradient.y, source_lane);
               gradient_z += __shfl_sync(peers, gradient.z, source_lane);
               remaining &= remaining - 1;
             }
           }
           if ((threadIdx.x & 31) == __ffs(peers) - 1) {
             atomicAdd(&gradients_shared[sphere_idx].x, gradient_x);
             atomicAdd(&gradients_shared[sphere_idx].y, gradient_y);
             atomicAdd(&gradients_shared[sphere_idx].z, gradient_z);
             // w is a sparse-output marker, not a radius derivative. Each positive
             // pair touches its endpoints, but the marker must not grow with the
             // number of pairs.
             atomicExch(&gradients_shared[sphere_idx].w, -local_weight);
           }
         }

         __forceinline__ __device__ void atomic_add_collision_gradient(
           float4* gradients_shared,
           const float4 sphere_a,
           const float4 sphere_b,
           const int sphere_a_idx,
           const int sphere_b_idx,
           const float local_weight)
         {
           const float4 gradient_a = make_float4(
             local_weight * (sphere_b.x - sphere_a.x),
             local_weight * (sphere_b.y - sphere_a.y),
             local_weight * (sphere_b.z - sphere_a.z),
             0.0f);
           warp_aggregate_collision_gradient(
             gradients_shared, sphere_a_idx, gradient_a, local_weight);
           warp_aggregate_collision_gradient(
             gradients_shared,
             sphere_b_idx,
             make_float4(-gradient_a.x, -gradient_a.y, -gradient_a.z, 0.0f),
             local_weight);
         }

         __forceinline__ __device__ void zero_shared_per_sphere_max(
           unsigned long long* per_sphere_max,
           const uint32_t num_spheres,
           const uint32_t num_threads_per_block)
         {
           const uint32_t num_spheres_per_thread =
             curobo::common::ceil_div(num_spheres, num_threads_per_block);
           for (uint32_t i = 0; i < num_spheres_per_thread; ++i) {
             const uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;
             if (sphere_idx < num_spheres) {
               per_sphere_max[sphere_idx] = 0ULL;
             }
           }
         }

         __forceinline__ __device__ unsigned long long encode_per_sphere_max(
           const float distance,
           const int partner_idx)
         {
           return (static_cast<unsigned long long>(__float_as_uint(distance)) << 32) |
             static_cast<unsigned int>(partner_idx);
         }

         __forceinline__ __device__ float decode_per_sphere_max_distance(
           const unsigned long long encoded_max)
         {
           return __uint_as_float(static_cast<unsigned int>(encoded_max >> 32));
         }

         __forceinline__ __device__ int decode_per_sphere_max_partner(
           const unsigned long long encoded_max)
         {
           return static_cast<int>(static_cast<unsigned int>(encoded_max));
         }

         __forceinline__ __device__ float finalize_per_sphere_max_collision_distance(
           const float4* spheres_shared,
           float4* gradients_shared,
           const unsigned long long* per_sphere_max,
           const uint32_t num_spheres,
           const uint32_t num_threads_per_block,
           const float local_weight,
           float* block_reduce_shared_data,
           float* reduced_sum)
         {
           float local_sum = 0.0f;
           const uint32_t num_spheres_per_thread =
             curobo::common::ceil_div(num_spheres, num_threads_per_block);
           for (uint32_t i = 0; i < num_spheres_per_thread; ++i) {
             const uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;
             if (sphere_idx >= num_spheres) {
               continue;
             }
             const unsigned long long encoded_max = per_sphere_max[sphere_idx];
             if (encoded_max == 0ULL) {
               continue;
             }
             const int partner_idx = decode_per_sphere_max_partner(encoded_max);
             const float distance = decode_per_sphere_max_distance(encoded_max);
             local_sum += distance;
             atomic_add_collision_gradient(
               gradients_shared,
               spheres_shared[sphere_idx],
               spheres_shared[partner_idx],
               sphere_idx,
               partner_idx,
               local_weight);
           }
           curobo::common::block_reduce_sum(
             local_sum, num_threads_per_block, block_reduce_shared_data, reduced_sum);
           return *reduced_sum;
         }

         __forceinline__ __device__ void compute_per_sphere_max_collision_distance(
           const float4* spheres_shared,
           unsigned long long* per_sphere_max,
           const int16_t* pair_locations,
           const uint32_t num_collision_pairs,
           const uint32_t num_threads_per_block)
         {
           const uint32_t num_checks_per_thread =
             curobo::common::ceil_div(num_collision_pairs, num_threads_per_block);
           for (uint32_t i = 0; i < num_checks_per_thread; ++i) {
             const uint32_t check_idx = threadIdx.x + i * num_threads_per_block;
             if (check_idx >= num_collision_pairs) {
               continue;
             }
             const short2 sphere_pair = reinterpret_cast<const short2*>(pair_locations)[check_idx];
             const float4 sphere_a = spheres_shared[sphere_pair.x];
             const float4 sphere_b = spheres_shared[sphere_pair.y];
             const float valid_mask =
               (sphere_a.w >= 0.0f && sphere_b.w >= 0.0f) ? 1.0f : 0.0f;
             const float distance = sphere_squared_distance_fused(sphere_a, sphere_b) * valid_mask;
             if (distance > 0.0f) {
               atomicMax(
                 &per_sphere_max[sphere_pair.x],
                 encode_per_sphere_max(distance, sphere_pair.y));
               atomicMax(
                 &per_sphere_max[sphere_pair.y],
                 encode_per_sphere_max(distance, sphere_pair.x));
             }
           }
         }

         __forceinline__ __device__ float compute_sum_collision_distance(
           const float4* spheres_shared,
           float4* gradients_shared,
           const int16_t* pair_locations,
           const uint32_t num_collision_pairs,
           const uint32_t num_threads_per_block,
           const float local_weight,
           float* block_reduce_shared_data,
           float* reduced_sum)
         {
           float local_sum = 0.0f;
           const uint32_t num_checks_per_thread =
             curobo::common::ceil_div(num_collision_pairs, num_threads_per_block);
           for (uint32_t i = 0; i < num_checks_per_thread; ++i) {
             const uint32_t check_idx = threadIdx.x + i * num_threads_per_block;
             if (check_idx >= num_collision_pairs) {
               continue;
             }
             const short2 sphere_pair = reinterpret_cast<const short2*>(pair_locations)[check_idx];
             const float4 sphere_a = spheres_shared[sphere_pair.x];
             const float4 sphere_b = spheres_shared[sphere_pair.y];
             const float valid_mask =
               (sphere_a.w >= 0.0f && sphere_b.w >= 0.0f) ? 1.0f : 0.0f;
             const float distance = sphere_squared_distance_fused(sphere_a, sphere_b) * valid_mask;
             if (distance > 0.0f) {
               local_sum += distance;
               atomic_add_collision_gradient(
                 gradients_shared, sphere_a, sphere_b, sphere_pair.x, sphere_pair.y, local_weight);
             }
           }
           curobo::common::block_reduce_sum(
             local_sum, num_threads_per_block, block_reduce_shared_data, reduced_sum);
           return *reduced_sum;
         }

         __forceinline__ __device__ CollisionPair compute_link_aabb_max_collision_distance(
           const float4* spheres_shared,
           const float4* link_aabbs,
           const int16_t* pair_locations,
           const int16_t* link_pairs,
           const int32_t* link_pair_offsets,
           const uint32_t num_link_pairs,
           const uint32_t num_threads_per_block,
           CollisionPair* block_reduce_shared_data,
           CollisionPair* reduced_max_d)
         {
           CollisionPair max_d = {0.0f, 0, 0};
           const uint32_t lane_idx = threadIdx.x & 31;
           const uint32_t warp_idx = threadIdx.x / 32;
           const uint32_t num_warps = num_threads_per_block / 32;

           for (uint32_t link_pair_idx = warp_idx;
                link_pair_idx < num_link_pairs;
                link_pair_idx += num_warps)
           {
             int pair_start = 0;
             int pair_end = 0;
             int overlaps = 0;
             if (lane_idx == 0)
             {
               const short2 link_pair =
                 reinterpret_cast<const short2*>(link_pairs)[link_pair_idx];
               overlaps = link_aabbs_overlap(
                 link_aabbs[2 * link_pair.x],
                 link_aabbs[2 * link_pair.x + 1],
                 link_aabbs[2 * link_pair.y],
                 link_aabbs[2 * link_pair.y + 1]);
               pair_start = link_pair_offsets[link_pair_idx];
               pair_end = link_pair_offsets[link_pair_idx + 1];
             }
             overlaps = ::__shfl_sync(curobo::common::fullMask, overlaps, 0);
             pair_start = ::__shfl_sync(curobo::common::fullMask, pair_start, 0);
             pair_end = ::__shfl_sync(curobo::common::fullMask, pair_end, 0);
             if (overlaps == 0)
             {
               continue;
             }

             for (int pair_idx = pair_start + lane_idx;
                  pair_idx < pair_end;
                  pair_idx += 32)
             {
               const short2 sphere_pair =
                 reinterpret_cast<const short2*>(pair_locations)[pair_idx];
               const float4 sphere_a = spheres_shared[sphere_pair.x];
               const float4 sphere_b = spheres_shared[sphere_pair.y];
               const float valid_mask =
                 (sphere_a.w >= 0.0f && sphere_b.w >= 0.0f) ? 1.0f : 0.0f;
               const float distance =
                 sphere_squared_distance_fused(sphere_a, sphere_b) * valid_mask;
               max_d = max(
                 max_d,
                 CollisionPair(distance, sphere_pair.x, sphere_pair.y));
             }
           }

           curobo::common::block_reduce_max(
             max_d,
             num_threads_per_block,
             block_reduce_shared_data,
             reduced_max_d);
           return *reduced_max_d;
         }

         __forceinline__ __device__ float compute_link_aabb_sum_collision_distance(
           const float4* spheres_shared,
           const float4* link_aabbs,
           float4* gradients_shared,
           const int16_t* pair_locations,
           const int16_t* link_pairs,
           const int32_t* link_pair_offsets,
           const uint32_t num_link_pairs,
           const uint32_t num_threads_per_block,
           const float local_weight,
           float* block_reduce_shared_data,
           float* reduced_sum)
         {
           float local_sum = 0.0f;
           const uint32_t lane_idx = threadIdx.x & 31;
           const uint32_t warp_idx = threadIdx.x / 32;
           const uint32_t num_warps = num_threads_per_block / 32;
           for (uint32_t link_pair_idx = warp_idx;
                link_pair_idx < num_link_pairs;
                link_pair_idx += num_warps)
           {
             int pair_start = 0;
             int pair_end = 0;
             int overlaps = 0;
             if (lane_idx == 0) {
               const short2 link_pair = reinterpret_cast<const short2*>(link_pairs)[link_pair_idx];
               overlaps = link_aabbs_overlap(
                 link_aabbs[2 * link_pair.x],
                 link_aabbs[2 * link_pair.x + 1],
                 link_aabbs[2 * link_pair.y],
                 link_aabbs[2 * link_pair.y + 1]);
               pair_start = link_pair_offsets[link_pair_idx];
               pair_end = link_pair_offsets[link_pair_idx + 1];
             }
             overlaps = ::__shfl_sync(curobo::common::fullMask, overlaps, 0);
             pair_start = ::__shfl_sync(curobo::common::fullMask, pair_start, 0);
             pair_end = ::__shfl_sync(curobo::common::fullMask, pair_end, 0);
             if (overlaps == 0) {
               continue;
             }
             for (int pair_idx = pair_start + lane_idx;
                  pair_idx < pair_end;
                  pair_idx += 32)
             {
               const short2 sphere_pair = reinterpret_cast<const short2*>(pair_locations)[pair_idx];
               const float4 sphere_a = spheres_shared[sphere_pair.x];
               const float4 sphere_b = spheres_shared[sphere_pair.y];
               const float valid_mask =
                 (sphere_a.w >= 0.0f && sphere_b.w >= 0.0f) ? 1.0f : 0.0f;
               const float distance = sphere_squared_distance_fused(sphere_a, sphere_b) * valid_mask;
               if (distance > 0.0f) {
                 local_sum += distance;
                 atomic_add_collision_gradient(
                   gradients_shared, sphere_a, sphere_b, sphere_pair.x, sphere_pair.y, local_weight);
               }
             }
           }
           curobo::common::block_reduce_sum(
             local_sum, num_threads_per_block, block_reduce_shared_data, reduced_sum);
           return *reduced_sum;
         }

         __forceinline__ __device__ void compute_link_aabb_per_sphere_max_collision_distance(
           const float4* spheres_shared,
           const float4* link_aabbs,
           unsigned long long* per_sphere_max,
           const int16_t* pair_locations,
           const int16_t* link_pairs,
           const int32_t* link_pair_offsets,
           const uint32_t num_link_pairs,
           const uint32_t num_threads_per_block)
         {
           const uint32_t lane_idx = threadIdx.x & 31;
           const uint32_t warp_idx = threadIdx.x / 32;
           const uint32_t num_warps = num_threads_per_block / 32;
           for (uint32_t link_pair_idx = warp_idx;
                link_pair_idx < num_link_pairs;
                link_pair_idx += num_warps)
           {
             int pair_start = 0;
             int pair_end = 0;
             int overlaps = 0;
             if (lane_idx == 0) {
               const short2 link_pair = reinterpret_cast<const short2*>(link_pairs)[link_pair_idx];
               overlaps = link_aabbs_overlap(
                 link_aabbs[2 * link_pair.x],
                 link_aabbs[2 * link_pair.x + 1],
                 link_aabbs[2 * link_pair.y],
                 link_aabbs[2 * link_pair.y + 1]);
               pair_start = link_pair_offsets[link_pair_idx];
               pair_end = link_pair_offsets[link_pair_idx + 1];
             }
             overlaps = ::__shfl_sync(curobo::common::fullMask, overlaps, 0);
             pair_start = ::__shfl_sync(curobo::common::fullMask, pair_start, 0);
             pair_end = ::__shfl_sync(curobo::common::fullMask, pair_end, 0);
             if (overlaps == 0) {
               continue;
             }
             for (int pair_idx = pair_start + lane_idx;
                  pair_idx < pair_end;
                  pair_idx += 32)
             {
               const short2 sphere_pair = reinterpret_cast<const short2*>(pair_locations)[pair_idx];
               const float4 sphere_a = spheres_shared[sphere_pair.x];
               const float4 sphere_b = spheres_shared[sphere_pair.y];
               const float valid_mask =
                 (sphere_a.w >= 0.0f && sphere_b.w >= 0.0f) ? 1.0f : 0.0f;
               const float distance = sphere_squared_distance_fused(sphere_a, sphere_b) * valid_mask;
               if (distance > 0.0f) {
                 atomicMax(
                   &per_sphere_max[sphere_pair.x],
                   encode_per_sphere_max(distance, sphere_pair.y));
                 atomicMax(
                   &per_sphere_max[sphere_pair.y],
                   encode_per_sphere_max(distance, sphere_pair.x));
               }
             }
           }
         }

         __forceinline__ __device__ void finalize_sum_collision_results(
           const float sum_distance,
           const float4* gradients_shared,
           float* out_distance,
           float* out_gradient,
           uint8_t* sparse_index,
           const uint32_t batch_horizon_idx,
           const uint32_t num_spheres,
           const uint32_t num_threads_per_block,
           const float local_weight,
           const bool write_grad)
         {
           if (threadIdx.x == 0) {
             out_distance[batch_horizon_idx] = 0.5f * local_weight * sum_distance;
           }
           if (!write_grad) {
             return;
           }
           const uint32_t num_spheres_per_thread =
             curobo::common::ceil_div(num_spheres, num_threads_per_block);
           for (uint32_t i = 0; i < num_spheres_per_thread; ++i) {
             const uint32_t sphere_idx = i * num_threads_per_block + threadIdx.x;
             if (sphere_idx >= num_spheres) {
               continue;
             }
             const float4 gradient = gradients_shared[sphere_idx];
             if (gradient.w != 0.0f) {
               *(float4 *)&out_gradient[(batch_horizon_idx * num_spheres + sphere_idx) * 4] = gradient;
               sparse_index[batch_horizon_idx * num_spheres + sphere_idx] = 1;
             }
           }
         }

         __forceinline__ __device__ void finalize_collision_results(
           CollisionPair max_d,
           const float4* spheres_shared,
           float* out_distance,
           float* out_gradient,
           uint8_t* sparse_index,
           uint32_t batch_horizon_idx,
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
             const float4 sph1_collision = spheres_shared[max_d.i];
             const float4 sph2_collision = spheres_shared[max_d.j];
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
