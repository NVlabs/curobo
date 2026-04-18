/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <stdexcept>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "common/torch_cuda_utils.h"


#include "kinematics_jacobian_backward_kernel.cuh"

namespace curobo{
namespace kinematics{





// Helper struct to hold launch configuration for jacobian backward
struct KinJacobianBwdLaunchConfig {
    int threadsPerBlock;
    int blocksPerGrid;
    int sharedMemSize;
    int threads_per_batch;
    bool use_warp_reduce;
  };

  // Helper function to calculate launch configuration for jacobian backward
  KinJacobianBwdLaunchConfig calculate_kinematics_jacobian_backward_launch_config(
    const int64_t batch_size, const int n_links, const int n_tool_frames,
    const int max_threads_per_block = 128, const int max_shared_memory = 48 * 1024) {

    const bool use_warp_reduce = (n_tool_frames < 5000);

    KinJacobianBwdLaunchConfig config;
    config.use_warp_reduce = use_warp_reduce;

    if (use_warp_reduce) {
        // Warp reduction configuration
        int max_batches_per_block = MAX_BW_BATCH_PER_BLOCK;
        const int max_threads_per_batch = 32;

        // Check if even 1 batch per block would fit in shared memory
        const int shared_mem_per_batch = n_links * 12 * sizeof(float);
        if (shared_mem_per_batch > max_shared_memory) {
            throw std::runtime_error("Single batch shared memory requirement exceeds limit");
        }

        // Limit batches per block based on shared memory constraints
        const int max_batches_from_shared_mem = max_shared_memory / shared_mem_per_batch;
        max_batches_per_block = std::min(max_batches_per_block, max_batches_from_shared_mem);

        // Limit threads per block based on input parameter for occupancy
        if (max_batches_per_block * max_threads_per_batch > max_threads_per_block) {
            max_batches_per_block = max_threads_per_block / max_threads_per_batch;
        }

        // Ensure at least 1 batch per block
        max_batches_per_block = std::max(1, max_batches_per_block);

        // Actual batches per block (limited by batch_size)
        int batches_per_block = std::min((int)batch_size, max_batches_per_block);

        config.threads_per_batch = max_threads_per_batch;
        config.threadsPerBlock = batches_per_block * max_threads_per_batch;
        config.blocksPerGrid = (batch_size * max_threads_per_batch + config.threadsPerBlock - 1) / config.threadsPerBlock;
        config.sharedMemSize = batches_per_block * n_links * 12 * sizeof(float);
    } else {
        // Block reduction configuration
        const int required_shared_mem = n_links * 12 * sizeof(float);
        if (required_shared_mem > max_shared_memory) {
            throw std::runtime_error("Single batch shared memory requirement exceeds limit");
        }

        config.threadsPerBlock = std::min(max_threads_per_block, std::max(32, n_tool_frames));
        config.blocksPerGrid = batch_size;
        config.sharedMemSize = required_shared_mem;
        config.threads_per_batch = config.threadsPerBlock;
    }

    return config;
  }

  // Helper function to select jacobian backward kernel
  typedef void (*JacobianKernelFunction)(
    float*, const float*, const float*, const int8_t*, const int16_t*, const int16_t*,
    const int16_t*, const int16_t*, const int16_t*, const int16_t*, const bool*, const int16_t*, const float*, const int, const int, const int, const int, const int);

  JacobianKernelFunction select_jacobian_kernel_function(bool use_warp_reduce, int64_t n_joints) {
    if (use_warp_reduce) {
        if (n_joints < 16) {
            return kinematics_jacobian_gradient_backward_kernel<float, 16, true>;
        } else if (n_joints < 64) {
            return kinematics_jacobian_gradient_backward_kernel<float, 64, true>;
        } else {
            return kinematics_jacobian_gradient_backward_kernel<float, 128, true>;
        }
    } else {
        if (n_joints < 16) {
            return kinematics_jacobian_gradient_backward_kernel<float, 16, false>;
        } else if (n_joints < 64) {
            return kinematics_jacobian_gradient_backward_kernel<float, 64, false>;
        } else {
            return kinematics_jacobian_gradient_backward_kernel<float, 128, false>;
        }
    }
  }

  void launch_kinematics_jacobian_backward(
  torch::Tensor grad_joint, // [batch_size, n_joints]
  const torch::Tensor grad_jacobian, // [batch_size, n_tool_frames, 6, n_joints]
  const torch::Tensor global_cumul_mat, // [batch_size, n_links, 12]
  const torch::Tensor joint_map_type, // [n_joints]
  const torch::Tensor joint_map, // [n_joints]
  const torch::Tensor link_map, // [n_links]
  const torch::Tensor link_chain_data, // [chain_data_size]
  const torch::Tensor link_chain_offsets, // [n_links + 1]
  const torch::Tensor joint_links_data, // [joint_links_data_size]
  const torch::Tensor joint_links_offsets, // [n_joints + 1]
  const torch::Tensor joint_affects_endeffector, // [n_joints * n_tool_frames] - NEW
  const torch::Tensor tool_frame_map, // [n_tool_frames]
  const torch::Tensor joint_offset_map, // [n_joints]
  const int64_t batch_size,
  const int64_t n_joints,
  const int64_t n_tool_frames)
  {

  // Input validation
  curobo::common::validate_cuda_input(grad_joint, "grad_joint");
  curobo::common::validate_cuda_input(grad_jacobian, "grad_jacobian");
  curobo::common::validate_cuda_input(global_cumul_mat, "global_cumul_mat");
  curobo::common::validate_cuda_input(joint_map_type, "joint_map_type");
  curobo::common::validate_cuda_input(joint_map, "joint_map");
  curobo::common::validate_cuda_input(link_map, "link_map");
  curobo::common::validate_cuda_input(link_chain_data, "link_chain_data");
  curobo::common::validate_cuda_input(link_chain_offsets, "link_chain_offsets");
  curobo::common::validate_cuda_input(joint_links_data, "joint_links_data");
  curobo::common::validate_cuda_input(joint_links_offsets, "joint_links_offsets");
  curobo::common::validate_cuda_input(joint_affects_endeffector, "joint_affects_endeffector");
  curobo::common::validate_cuda_input(tool_frame_map, "tool_frame_map");
  curobo::common::validate_cuda_input(joint_offset_map, "joint_offset_map");

  const int n_links = link_map.size(0);
  assert(n_joints < 128); // for larger num. of joints, change kernel template value.

  cudaStream_t stream = curobo::common::get_cuda_stream();

  // Calculate launch configuration
  const int max_threads_per_block = 128;
  KinJacobianBwdLaunchConfig config = calculate_kinematics_jacobian_backward_launch_config(
    batch_size, n_links, n_tool_frames, max_threads_per_block);

  // Select kernel
  JacobianKernelFunction selected_kernel = select_jacobian_kernel_function(config.use_warp_reduce, n_joints);

  // Launch kernel
  selected_kernel
    << < config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream >> > (
    grad_joint.data_ptr<float>(),
    grad_jacobian.data_ptr<float>(),
    global_cumul_mat.data_ptr<float>(),
    joint_map_type.data_ptr<int8_t>(),
    joint_map.data_ptr<int16_t>(),
    link_map.data_ptr<int16_t>(),
    link_chain_data.data_ptr<int16_t>(), // linkChainData
    link_chain_offsets.data_ptr<int16_t>(), // linkChainOffsets
    joint_links_data.data_ptr<int16_t>(), // NEW: jointLinksData
    joint_links_offsets.data_ptr<int16_t>(), // NEW: jointLinksOffsets
    joint_affects_endeffector.data_ptr<bool>(), // NEW: precomputed cache
    tool_frame_map.data_ptr<int16_t>(),
    joint_offset_map.data_ptr<float>(),
    batch_size, n_joints, n_links, n_tool_frames, config.threads_per_batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
}
