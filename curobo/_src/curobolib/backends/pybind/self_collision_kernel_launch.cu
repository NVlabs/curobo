/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "common/torch_cuda_utils.h"
#include "self_collision_kernel.cuh"



namespace curobo{
    namespace geometry{
      namespace self_collision {


// This is the best version so far.
// It precomputes the start addresses per thread on the cpu.
// The rest is similar to the version above.
void self_collision_distance(
    torch::Tensor out_distance,
    torch::Tensor out_vec,
    torch::Tensor pair_distance, // batch_size x horizon x num_collision_pairs
    torch::Tensor sparse_index,
    const torch::Tensor robot_spheres,    // batch_size x horizon x n_spheres x 4
    const torch::Tensor sphere_padding, // n_spheres
    const torch::Tensor weight,
    const torch::Tensor pair_locations,
    torch::Tensor block_batch_max_value, // batch_size x horizon x num_blocks_per_batch
    torch::Tensor block_batch_max_index, // batch_size x horizon x num_blocks_per_batch x 2
    const int num_blocks_per_batch,
    const int max_threads_per_block,
    const int batch_size,
    const int horizon,
    const int nspheres,
    const int num_collision_pairs,
    const bool store_pair_distance,
    const bool compute_grad
  )
  {
    curobo::common::validate_cuda_input(out_distance, "out_distance");
    curobo::common::validate_cuda_input(out_vec, "out_vec");
    curobo::common::validate_cuda_input(pair_distance, "pair_distance");
    curobo::common::validate_cuda_input(sparse_index, "sparse_index");
    curobo::common::validate_cuda_input(robot_spheres, "robot_spheres");
    curobo::common::validate_cuda_input(sphere_padding, "sphere_padding");
    curobo::common::validate_cuda_input(weight, "weight");
    curobo::common::validate_cuda_input(pair_locations, "pair_locations");



    if (num_blocks_per_batch == 0)
    {
      throw std::invalid_argument("num_blocks_per_batch must be greater than 0");
    }


    if (store_pair_distance)
    {
      if(pair_distance.size(0) != batch_size || pair_distance.size(1) != horizon || pair_distance.size(2) != num_collision_pairs)
      {
        throw std::invalid_argument("pair_distance must have the size (batch_size, horizon, num_collision_pairs), "
        "batch_size: " + std::to_string(batch_size) + ", horizon: " + std::to_string(horizon) + ", num_collision_pairs: " + std::to_string(num_collision_pairs)
        + ", pair_distance.size(0): " + std::to_string(pair_distance.size(0)) + ", pair_distance.size(1): " + std::to_string(pair_distance.size(1)) + ", pair_distance.size(2): " + std::to_string(pair_distance.size(2)));
      }
    }

    const int smemsize = sizeof(float4) * nspheres;
    constexpr int static_smem_overhead = sizeof(CollisionPair) * 33; // block_reduce_shared_data[32] + reduced_max_d
    const int total_smem = smemsize + static_smem_overhead;

    int device_id = 0;
    cudaGetDevice(&device_id);
    int device_max_smem = 0;
    cudaDeviceGetAttribute(&device_max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

    if (total_smem > device_max_smem) {
      const int max_nspheres = (device_max_smem - static_smem_overhead) / static_cast<int>(sizeof(float4));
      throw std::runtime_error(
          "Self-collision kernel requires " + std::to_string(total_smem) +
          " bytes of shared memory (" + std::to_string(smemsize) + " dynamic + " +
          std::to_string(static_smem_overhead) + " static) for " +
          std::to_string(nspheres) + " spheres, but device supports at most " +
          std::to_string(device_max_smem) + " bytes. Reduce nspheres to at most " +
          std::to_string(max_nspheres));
    }

    if (smemsize > 48000) {
      auto set_smem = [&](auto kernel_fn) {
        cudaError_t result = cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smemsize);
        if (result != cudaSuccess) {
          throw std::runtime_error(
              "cudaFuncSetAttribute failed for self-collision kernel: " +
              std::string(cudaGetErrorString(result)));
        }
      };
      set_smem(store_pair_distance ? self_collision_max_distance_kernel<true> : self_collision_max_distance_kernel<false>);
      set_smem(store_pair_distance ? self_collision_max_block_kernel<true> : self_collision_max_block_kernel<false>);
    }

    cudaStream_t stream = curobo::common::get_current_cuda_stream();



    if (num_blocks_per_batch == 1)
    {
      // call only fused kernel:




    int num_blocks = batch_size * horizon;


    // calculate number of threads to launch:
    int num_checks_per_thread = curobo::common::ceil_div(num_collision_pairs, max_threads_per_block);

    int num_threads_per_block = std::min(max_threads_per_block, num_collision_pairs);

    // Ensure num_threads_per_block is a multiple of 32 (warp size) for proper ballot sync
    const int warp_size = 32;
    num_threads_per_block = curobo::common::ceil_div(num_threads_per_block, warp_size) * warp_size;
    num_threads_per_block = std::min(num_threads_per_block, max_threads_per_block);



    auto selected_kernel = store_pair_distance ? self_collision_max_distance_kernel<true> : self_collision_max_distance_kernel<false>;




    selected_kernel<<<num_blocks, num_threads_per_block, smemsize, stream>>>(
            out_distance.data_ptr<float>(),
            out_vec.data_ptr<float>(),
            pair_distance.data_ptr<float>(),
            sparse_index.data_ptr<uint8_t>(),
            robot_spheres.data_ptr<float>(),
            sphere_padding.data_ptr<float>(),
            weight.data_ptr<float>(),
            pair_locations.data_ptr<int16_t>(),
            batch_size,
            horizon,
            nspheres,
            num_collision_pairs,
            compute_grad);
    }
    // if num_blocks_per_batch > 1, call two seperate kernels:
    else
    {
      // divide the work across blocks:


      auto selected_block_kernel = store_pair_distance ? self_collision_max_block_kernel<true> : self_collision_max_block_kernel<false>;

      // launch the first kernel with num_blocks_per_batch * num_threads_per_block threads
      int num_blocks_first_kernel = batch_size * horizon * num_blocks_per_batch;

      int num_threads_per_block_first_kernel = max_threads_per_block;
      selected_block_kernel<<<num_blocks_first_kernel, num_threads_per_block_first_kernel, smemsize, stream>>>(
        out_vec.data_ptr<float>(),
        pair_distance.data_ptr<float>(),
        sparse_index.data_ptr<uint8_t>(),
        robot_spheres.data_ptr<float>(),
        sphere_padding.data_ptr<float>(),
        pair_locations.data_ptr<int16_t>(),
        block_batch_max_value.data_ptr<float>(),
        block_batch_max_index.data_ptr<int16_t>(),
        num_blocks_per_batch,
        batch_size,
        horizon,
        nspheres,
        num_collision_pairs);

      C10_CUDA_KERNEL_LAUNCH_CHECK();

      // launch the second kernel with num_blocks_per_batch threads
      int num_blocks_second_kernel = batch_size * horizon;
      int num_threads_per_block_second_kernel = std::min(512, num_blocks_per_batch);
      self_collision_max_reduce_kernel<<<num_blocks_second_kernel, num_threads_per_block_second_kernel, 0, stream>>>(
        out_distance.data_ptr<float>(),
        out_vec.data_ptr<float>(),
        pair_distance.data_ptr<float>(),
        sparse_index.data_ptr<uint8_t>(),
        robot_spheres.data_ptr<float>(),
        sphere_padding.data_ptr<float>(),
        weight.data_ptr<float>(),
        pair_locations.data_ptr<int16_t>(),
        block_batch_max_value.data_ptr<float>(),
        block_batch_max_index.data_ptr<int16_t>(),
        num_blocks_per_batch,
        batch_size,
        horizon,
        nspheres,
        num_collision_pairs,
        compute_grad);



    }


    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return;

  }

      }
    }
}