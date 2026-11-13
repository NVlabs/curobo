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

    constexpr int SELF_COLLISION_MAX = 0;
    constexpr int SELF_COLLISION_SUM = 1;
    constexpr int SELF_COLLISION_PER_SPHERE_MAX = 2;
    static_assert(
      sizeof(CollisionPair) == sizeof(unsigned long long),
      "Self-collision reduction storage assumes an eight-byte CollisionPair");

    template<
      int ReductionMode,
      bool WriteGrad,
      int StaticNumSpheres,
      int StaticNumLinkPairs,
      int StaticNumCollisionLinks,
      int StaticBlockThreads>
    __global__ void self_collision_link_aabb_distance_kernel(
      float* out_distance,
      float* out_gradient,
      uint8_t* sparse_index,
      const float* robot_spheres,
      const float* sphere_padding,
      const float* weight,
      const int16_t* pair_locations,
      const int16_t* link_sphere_indices,
      const int32_t* link_sphere_offsets,
      const int16_t* link_pairs,
      const int32_t* link_pair_offsets,
      const int batch_size,
      const int horizon,
      const int runtime_num_spheres,
      const int runtime_num_link_pairs,
      const int runtime_num_collision_links)
    {
      static_assert(
        ReductionMode >= SELF_COLLISION_MAX &&
        ReductionMode <= SELF_COLLISION_PER_SPHERE_MAX,
        "Unsupported AABB self-collision reduction mode");
      constexpr bool has_static_shape =
        StaticNumSpheres > 0 &&
        StaticNumLinkPairs > 0 &&
        StaticNumCollisionLinks > 0 &&
        StaticBlockThreads > 0;
      const int num_spheres =
        has_static_shape ? StaticNumSpheres : runtime_num_spheres;
      const int num_link_pairs =
        has_static_shape ? StaticNumLinkPairs : runtime_num_link_pairs;
      const int num_collision_links =
        has_static_shape ? StaticNumCollisionLinks : runtime_num_collision_links;
      const int num_threads_per_block =
        has_static_shape ? StaticBlockThreads : blockDim.x;

      extern __shared__ float4 shared_data[];
      __shared__ unsigned long long block_reduce_storage[32];
      __shared__ unsigned long long reduced_storage;
      float4* spheres_shared = shared_data;
      float4* link_aabbs_shared = spheres_shared + num_spheres;
      const uint32_t batch_horizon_idx = blockIdx.x;
      const uint32_t batch_idx = batch_horizon_idx / horizon;
      if (batch_idx >= batch_size)
      {
        return;
      }

      load_spheres_and_zero_gradients(
        spheres_shared,
        out_gradient,
        sparse_index,
        robot_spheres,
        sphere_padding,
        batch_horizon_idx,
        num_spheres,
        num_threads_per_block);
      if constexpr (ReductionMode != SELF_COLLISION_MAX)
      {
        float4* gradients_shared = link_aabbs_shared + 2 * num_collision_links;
        zero_shared_gradients(
          gradients_shared, num_spheres, num_threads_per_block);
        if constexpr (ReductionMode == SELF_COLLISION_PER_SPHERE_MAX)
        {
          unsigned long long* per_sphere_max =
            reinterpret_cast<unsigned long long*>(gradients_shared + num_spheres);
          zero_shared_per_sphere_max(
            per_sphere_max, num_spheres, num_threads_per_block);
        }
      }
      __syncthreads();
      compute_link_aabbs(
        spheres_shared,
        link_aabbs_shared,
        link_sphere_indices,
        link_sphere_offsets,
        num_collision_links);
      __syncthreads();

      if constexpr (ReductionMode == SELF_COLLISION_MAX)
      {
        const CollisionPair max_d = compute_link_aabb_max_collision_distance(
          spheres_shared,
          link_aabbs_shared,
          pair_locations,
          link_pairs,
          link_pair_offsets,
          num_link_pairs,
          num_threads_per_block,
          reinterpret_cast<CollisionPair*>(block_reduce_storage),
          reinterpret_cast<CollisionPair*>(&reduced_storage));
        if (threadIdx.x == 0)
        {
          finalize_collision_results(
            max_d,
            spheres_shared,
            out_distance,
            out_gradient,
            sparse_index,
            batch_horizon_idx,
            num_spheres,
            weight[0],
            WriteGrad);
        }
      }
      else
      {
        float4* gradients_shared = link_aabbs_shared + 2 * num_collision_links;
        float sum_distance = 0.0f;
        if constexpr (ReductionMode == SELF_COLLISION_SUM)
        {
          sum_distance = compute_link_aabb_sum_collision_distance(
            spheres_shared,
            link_aabbs_shared,
            gradients_shared,
            pair_locations,
            link_pairs,
            link_pair_offsets,
            num_link_pairs,
            num_threads_per_block,
            weight[0],
            reinterpret_cast<float*>(block_reduce_storage),
            reinterpret_cast<float*>(&reduced_storage));
        }
        else
        {
          unsigned long long* per_sphere_max =
            reinterpret_cast<unsigned long long*>(gradients_shared + num_spheres);
          compute_link_aabb_per_sphere_max_collision_distance(
            spheres_shared,
            link_aabbs_shared,
            per_sphere_max,
            pair_locations,
            link_pairs,
            link_pair_offsets,
            num_link_pairs,
            num_threads_per_block);
          __syncthreads();
          sum_distance = finalize_per_sphere_max_collision_distance(
            spheres_shared,
            gradients_shared,
            per_sphere_max,
            num_spheres,
            num_threads_per_block,
            weight[0],
            reinterpret_cast<float*>(block_reduce_storage),
            reinterpret_cast<float*>(&reduced_storage));
        }
        __syncthreads();
        finalize_sum_collision_results(
          sum_distance,
          gradients_shared,
          out_distance,
          out_gradient,
          sparse_index,
          batch_horizon_idx,
          num_spheres,
          num_threads_per_block,
          weight[0],
          WriteGrad);
      }
    }

    template<
      int ReductionMode,
      bool StorePairDistance,
      bool WriteGrad,
      int StaticNumSpheres,
      int StaticNumCollisionPairs,
      int StaticBlockThreads>
    __global__ void self_collision_exact_distance_kernel(
      float* out_distance,
      float* out_gradient,
      float* pair_distance,
      uint8_t* sparse_index,
      const float* robot_spheres,
      const float* offsets,
      const float* weight,
      const int16_t* pair_locations,
      const int batch_size,
      const int horizon,
      const int runtime_num_spheres,
      const int runtime_num_collision_pairs)
    {
      static_assert(
        ReductionMode >= SELF_COLLISION_MAX &&
        ReductionMode <= SELF_COLLISION_PER_SPHERE_MAX,
        "Unsupported exact self-collision reduction mode");
      static_assert(
        ReductionMode == SELF_COLLISION_MAX || !StorePairDistance,
        "Pair-distance storage is supported only for global-max self collision");

      constexpr bool has_static_shape =
        StaticNumSpheres > 0 &&
        StaticNumCollisionPairs > 0 &&
        StaticBlockThreads > 0;
      const int num_spheres =
        has_static_shape ? StaticNumSpheres : runtime_num_spheres;
      const int num_collision_pairs =
        has_static_shape ? StaticNumCollisionPairs : runtime_num_collision_pairs;
      const int num_threads_per_block =
        has_static_shape ? StaticBlockThreads : blockDim.x;

      extern __shared__ float4 shared_data[];
      __shared__ unsigned long long block_reduce_storage[32];
      __shared__ unsigned long long reduced_storage;
      float4* spheres_shared = shared_data;
      const uint32_t batch_horizon_idx = blockIdx.x;
      const uint32_t batch_idx = batch_horizon_idx / horizon;
      if (batch_idx >= batch_size || batch_horizon_idx >= batch_size * horizon)
      {
        return;
      }

      load_spheres_and_zero_gradients(
        spheres_shared,
        out_gradient,
        sparse_index,
        robot_spheres,
        offsets,
        batch_horizon_idx,
        num_spheres,
        num_threads_per_block);

      if constexpr (ReductionMode == SELF_COLLISION_MAX)
      {
        __syncthreads();
        const CollisionPair max_d = compute_max_collision_distance<StorePairDistance>(
          spheres_shared,
          pair_distance,
          pair_locations,
          batch_horizon_idx,
          num_collision_pairs,
          num_threads_per_block,
          reinterpret_cast<CollisionPair*>(block_reduce_storage));
        if (threadIdx.x == 0)
        {
          finalize_collision_results(
            max_d,
            spheres_shared,
            out_distance,
            out_gradient,
            sparse_index,
            batch_horizon_idx,
            num_spheres,
            weight[0],
            WriteGrad);
        }
      }
      else
      {
        float4* gradients_shared = spheres_shared + num_spheres;
        zero_shared_gradients(
          gradients_shared, num_spheres, num_threads_per_block);

        if constexpr (ReductionMode == SELF_COLLISION_SUM)
        {
          __syncthreads();
          const float sum_distance = compute_sum_collision_distance(
            spheres_shared,
            gradients_shared,
            pair_locations,
            num_collision_pairs,
            num_threads_per_block,
            weight[0],
            reinterpret_cast<float*>(block_reduce_storage),
            reinterpret_cast<float*>(&reduced_storage));
          __syncthreads();
          finalize_sum_collision_results(
            sum_distance,
            gradients_shared,
            out_distance,
            out_gradient,
            sparse_index,
            batch_horizon_idx,
            num_spheres,
            num_threads_per_block,
            weight[0],
            WriteGrad);
        }
        else
        {
          unsigned long long* per_sphere_max =
            reinterpret_cast<unsigned long long*>(gradients_shared + num_spheres);
          zero_shared_per_sphere_max(
            per_sphere_max, num_spheres, num_threads_per_block);
          __syncthreads();
          compute_per_sphere_max_collision_distance(
            spheres_shared,
            per_sphere_max,
            pair_locations,
            num_collision_pairs,
            num_threads_per_block);
          __syncthreads();
          const float sum_distance = finalize_per_sphere_max_collision_distance(
            spheres_shared,
            gradients_shared,
            per_sphere_max,
            num_spheres,
            num_threads_per_block,
            weight[0],
            reinterpret_cast<float*>(block_reduce_storage),
            reinterpret_cast<float*>(&reduced_storage));
          __syncthreads();
          finalize_sum_collision_results(
            sum_distance,
            gradients_shared,
            out_distance,
            out_gradient,
            sparse_index,
            batch_horizon_idx,
            num_spheres,
            num_threads_per_block,
            weight[0],
            WriteGrad);
        }
      }
    }



    } // namespace collision
  } // namespace geometry
}   // namespace curobo
