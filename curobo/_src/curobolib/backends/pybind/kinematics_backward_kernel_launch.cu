/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/extension.h>
#include "common/torch_cuda_utils.h"

#include "kinematics/kinematics_backward_kernel.cuh"



namespace curobo{
namespace kinematics{
// Helper struct to hold launch configuration
struct KinBwdLaunchConfig {
    int threadsPerBlock;
    int blocksPerGrid;
    int sharedMemSize;
    int threads_per_batch;
    bool use_warp_reduce;
  };

  // Helper function to calculate launch configuration for both warp and block reduction
  KinBwdLaunchConfig calculate_kinematics_backward_launch_config(const int64_t batch_size, const int n_links, const int n_spheres, const int n_tool_frames,
                                     const int max_threads_per_block = 128, const int max_shared_memory = 48 * 1024) {
    const bool use_warp_reduce = (n_spheres < 5000);

    KinBwdLaunchConfig config;
    config.use_warp_reduce = use_warp_reduce;

        if (use_warp_reduce) {
        // Warp reduction configuration
        // Calculate batches per block based on different constraints
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
        // Check if shared memory requirement fits
        const int required_shared_mem = n_links * 12 * sizeof(float);
        if (required_shared_mem > max_shared_memory) {
            throw std::runtime_error("Single batch shared memory requirement exceeds limit");
        }

        config.threadsPerBlock = std::min(max_threads_per_block, std::max(32, std::max(n_spheres, n_tool_frames)));
        config.blocksPerGrid = batch_size;
        config.sharedMemSize = required_shared_mem;
        config.threads_per_batch = config.threadsPerBlock;
    }

    return config;
  }

  typedef void (*KinBwdFunction)(
    float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*,
    const int8_t*, const int16_t*, const int16_t*, const int16_t*, const int16_t*,
    const int32_t*, const int16_t*, const int16_t*, const int16_t*, const int16_t*,
    const bool*, const float*,
    const int, const int, const int, const int, const int, const int, const int, const int);

  template<bool USE_WARP_REDUCE, bool COMPUTE_COM, bool COMPUTE_JACOBIAN_GRAD>
  KinBwdFunction select_bwd_max_joints(int64_t n_joints) {
    if (n_joints < 16) {
      return kinematics_backward_kernel<
        float, float, 16, USE_WARP_REDUCE, COMPUTE_COM, COMPUTE_JACOBIAN_GRAD>;
    } else if (n_joints < 64) {
      return kinematics_backward_kernel<
        float, float, 64, USE_WARP_REDUCE, COMPUTE_COM, COMPUTE_JACOBIAN_GRAD>;
    }
    return kinematics_backward_kernel<
      float, float, 128, USE_WARP_REDUCE, COMPUTE_COM, COMPUTE_JACOBIAN_GRAD>;
  }

  KinBwdFunction select_bwd_kernel_function(
    bool use_warp_reduce,
    bool compute_com,
    bool compute_jacobian_grad,
    int64_t n_joints) {
    if (use_warp_reduce) {
      if (compute_com) {
        return compute_jacobian_grad
          ? select_bwd_max_joints<true, true, true>(n_joints)
          : select_bwd_max_joints<true, true, false>(n_joints);
      }
      return compute_jacobian_grad
        ? select_bwd_max_joints<true, false, true>(n_joints)
        : select_bwd_max_joints<true, false, false>(n_joints);
    }

    if (compute_com) {
      return compute_jacobian_grad
        ? select_bwd_max_joints<false, true, true>(n_joints)
        : select_bwd_max_joints<false, true, false>(n_joints);
    }
    return compute_jacobian_grad
      ? select_bwd_max_joints<false, false, true>(n_joints)
      : select_bwd_max_joints<false, false, false>(n_joints);
  }

  void launch_kinematics_backward(
  torch::Tensor grad_out,
  const torch::Tensor grad_nlinks_pos,
  const torch::Tensor grad_nlinks_quat,
  const torch::Tensor grad_spheres,
  const torch::Tensor grad_center_of_mass,
  const torch::Tensor batch_center_of_mass,
  const torch::Tensor grad_jacobian,
  const torch::Tensor global_cumul_mat,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_masses_com,
  const torch::Tensor link_map,
  const torch::Tensor joint_map,
  const torch::Tensor joint_map_type,
  const torch::Tensor tool_frame_map,
  const torch::Tensor link_sphere_map,
  const torch::Tensor link_chain_data,
  const torch::Tensor link_chain_offsets,
  const torch::Tensor joint_links_data,
  const torch::Tensor joint_links_offsets,
  const torch::Tensor joint_affects_endeffector,
  const torch::Tensor joint_offset_map,
  const torch::Tensor env_query_idx,
  const int64_t num_envs,
  const int64_t batch_size,
  const int64_t horizon,
  const int64_t n_joints,
  const int64_t n_spheres,
  const bool compute_com,
  const bool compute_jacobian_grad)
  {
  curobo::common::validate_cuda_input(grad_out, "grad_out");
  curobo::common::validate_cuda_input(grad_nlinks_pos, "grad_nlinks_pos");
  curobo::common::validate_cuda_input(grad_nlinks_quat, "grad_nlinks_quat");
  curobo::common::validate_cuda_input(grad_spheres, "grad_spheres");
  curobo::common::validate_cuda_input(grad_center_of_mass, "grad_center_of_mass");
  curobo::common::validate_cuda_input(batch_center_of_mass, "batch_center_of_mass");
  curobo::common::validate_cuda_input(grad_jacobian, "grad_jacobian");
  curobo::common::validate_cuda_input(global_cumul_mat, "global_cumul_mat");
  curobo::common::validate_cuda_input(robot_spheres, "robot_spheres");
  curobo::common::validate_cuda_input(link_masses_com, "link_masses_com");
  curobo::common::validate_cuda_input(link_map, "link_map");
  curobo::common::validate_cuda_input(joint_map, "joint_map");
  curobo::common::validate_cuda_input(joint_map_type, "joint_map_type");
  curobo::common::validate_cuda_input(tool_frame_map, "tool_frame_map");
  curobo::common::validate_cuda_input(link_sphere_map, "link_sphere_map");
  curobo::common::validate_cuda_input(link_chain_data, "link_chain_data");
  curobo::common::validate_cuda_input(link_chain_offsets, "link_chain_offsets");
  curobo::common::validate_cuda_input(joint_links_data, "joint_links_data");
  curobo::common::validate_cuda_input(joint_links_offsets, "joint_links_offsets");
  curobo::common::validate_cuda_input(joint_affects_endeffector, "joint_affects_endeffector");
  curobo::common::validate_cuda_input(joint_offset_map, "joint_offset_map");
  curobo::common::validate_cuda_input(env_query_idx, "env_query_idx");

  const int n_links       = link_map.size(0);
  const int n_tool_frames = tool_frame_map.size(0);

  assert(n_joints < 128);
  assert(joint_affects_endeffector.dtype() == torch::kBool);

  cudaStream_t stream = curobo::common::get_cuda_stream();

  const int max_threads_per_block = 128;
  KinBwdLaunchConfig config = calculate_kinematics_backward_launch_config(
    batch_size, n_links, n_spheres, n_tool_frames, max_threads_per_block);

  KinBwdFunction selected_kernel = select_bwd_kernel_function(
    config.use_warp_reduce, compute_com, compute_jacobian_grad, n_joints);

  selected_kernel
    << < config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream >> > (
    grad_out.data_ptr<float>(),
    grad_nlinks_pos.data_ptr<float>(),
    grad_nlinks_quat.data_ptr<float>(),
    grad_spheres.data_ptr<float>(),
    grad_center_of_mass.data_ptr<float>(),
    batch_center_of_mass.data_ptr<float>(),
    grad_jacobian.data_ptr<float>(),
    global_cumul_mat.data_ptr<float>(),
    robot_spheres.data_ptr<float>(),
    link_masses_com.data_ptr<float>(),
    joint_map_type.data_ptr<int8_t>(),
    joint_map.data_ptr<int16_t>(),
    link_map.data_ptr<int16_t>(),
    tool_frame_map.data_ptr<int16_t>(),
    link_sphere_map.data_ptr<int16_t>(),
    env_query_idx.data_ptr<int32_t>(),
    link_chain_data.data_ptr<int16_t>(),
    link_chain_offsets.data_ptr<int16_t>(),
    joint_links_data.data_ptr<int16_t>(),
    joint_links_offsets.data_ptr<int16_t>(),
    joint_affects_endeffector.data_ptr<bool>(),
    joint_offset_map.data_ptr<float>(),
    batch_size, horizon, n_spheres,
    n_links, n_joints, n_tool_frames, num_envs, config.threads_per_batch);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return;
  }
}
}
