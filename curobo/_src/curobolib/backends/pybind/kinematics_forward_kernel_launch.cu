/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



#include <torch/extension.h>

#include "common/torch_cuda_utils.h"

#include "kinematics_forward_kernel.cuh"

namespace curobo {
namespace kinematics {


// Helper struct to hold launch configuration for kinematics forward
struct KinFwdLaunchConfig {
    // Common configuration
    int threadsPerBlock;
    int blocksPerGrid;
    int sharedMemSize;
    int threads_per_batch;
    bool use_single_kernel;

    // Configuration for separate kernels (when use_single_kernel is false)
    int separate_threadsPerBlock;
    int separate_blocksPerGrid;
    int separate_sharedMemSize;
    int separate_threads_per_batch;
    int separate_batches_per_block;
    bool compute_jacobian;
};

// Helper function to calculate launch configuration for kinematics forward
KinFwdLaunchConfig calculate_kinematics_forward_launch_config(
    const bool compute_jacobian,
    const int64_t batch_size, const int n_links, const int n_spheres, const int n_tool_frames,
    const int max_threads_per_block = 128, const int max_shared_memory = 48 * 1024) {

    KinFwdLaunchConfig config;
    config.compute_jacobian = compute_jacobian;
    // Determine execution strategy based on workload size
    config.use_single_kernel = (n_spheres < 100);

        if (config.use_single_kernel) {
        // Single fused kernel configuration
        int max_batches_per_block = MAX_FW_BATCH_PER_BLOCK;

        config.threads_per_batch = 4;

        // Check if even 1 batch per block would fit in shared memory
        // Include space for cumul_mat + center of mass data
        const int shared_mem_per_batch = n_links * 12 * sizeof(float) + config.threads_per_batch * sizeof(float4);
        if (shared_mem_per_batch > max_shared_memory) {
            throw std::runtime_error("Single batch shared memory requirement exceeds limit");
        }

        // Limit batches per block based on shared memory constraints
        const int max_batches_from_shared_mem = max_shared_memory / shared_mem_per_batch;
        max_batches_per_block = std::min(max_batches_per_block, max_batches_from_shared_mem);

        // Limit threads per block based on input parameter for occupancy
        if (max_batches_per_block * config.threads_per_batch > max_threads_per_block) {
            max_batches_per_block = max_threads_per_block / config.threads_per_batch;
        }

        // Ensure at least 1 batch per block
        max_batches_per_block = std::max(1, max_batches_per_block);

        // Actual batches per block (limited by batch_size)
        int batches_per_block = std::min((int)batch_size, max_batches_per_block);

        config.threadsPerBlock = batches_per_block * config.threads_per_batch;
        config.blocksPerGrid = (batch_size * config.threads_per_batch + config.threadsPerBlock - 1) / config.threadsPerBlock;
        // Include space for cumul_mat + center of mass data
        config.sharedMemSize = batches_per_block * n_links * 12 * sizeof(float) + batches_per_block * config.threads_per_batch * sizeof(float4);

        // Initialize separate kernel config (not used but for completeness)
        config.separate_threadsPerBlock = 0;
        config.separate_blocksPerGrid = 0;
        config.separate_sharedMemSize = 0;
        config.separate_threads_per_batch = 0;
        config.separate_batches_per_block = 0;
            } else {
        // Separate kernels configuration
        // First kernel (cumulative transform) uses same config as single kernel
        int max_batches_per_block = MAX_FW_BATCH_PER_BLOCK;

        config.threads_per_batch = 4;

        // Check if even 1 batch per block would fit in shared memory
        // Include space for cumul_mat + center of mass data
        const int shared_mem_per_batch = n_links * 12 * sizeof(float) + config.threads_per_batch * sizeof(float4);
        if (shared_mem_per_batch > max_shared_memory) {
            throw std::runtime_error("Single batch shared memory requirement exceeds limit");
        }

        // Limit batches per block based on shared memory constraints
        const int max_batches_from_shared_mem = max_shared_memory / shared_mem_per_batch;
        max_batches_per_block = std::min(max_batches_per_block, max_batches_from_shared_mem);

        if (max_batches_per_block * config.threads_per_batch > max_threads_per_block) {
            max_batches_per_block = max_threads_per_block / config.threads_per_batch;
        }

        // Ensure at least 1 batch per block
        max_batches_per_block = std::max(1, max_batches_per_block);

        int batches_per_block = std::min((int)batch_size, max_batches_per_block);

        config.threadsPerBlock = batches_per_block * config.threads_per_batch;
        config.blocksPerGrid = (batch_size * config.threads_per_batch + config.threadsPerBlock - 1) / config.threadsPerBlock;
        // Include space for cumul_mat + center of mass data
        config.sharedMemSize = batches_per_block * n_links * 12 * sizeof(float) + batches_per_block * config.threads_per_batch * sizeof(float4);

        // Second kernel (spheres and links) configuration
        // Estimate threads_per_batch for shared memory constraint calculation
        const int estimated_threads_per_batch = std::min(128, std::max(n_spheres, n_tool_frames));
        const int separate_shared_mem_per_batch = n_links * 12 * sizeof(float) + estimated_threads_per_batch * sizeof(float4);
        // Limit batches per block based on shared memory constraints
        int separate_max_batches_per_block = max_shared_memory / separate_shared_mem_per_batch;
        separate_max_batches_per_block = std::min(separate_max_batches_per_block, 4); // Original default was 4
        separate_max_batches_per_block = std::max(1, separate_max_batches_per_block);

        config.separate_batches_per_block = separate_max_batches_per_block;
        config.separate_threads_per_batch = std::min(128, std::max(n_spheres, n_tool_frames));
        // Include space for shared_cumul_mat + center of mass data
        config.separate_sharedMemSize = config.separate_batches_per_block * n_links * 12 * sizeof(float) + config.separate_batches_per_block * config.separate_threads_per_batch * sizeof(float4);
        config.separate_threadsPerBlock = config.separate_batches_per_block * config.separate_threads_per_batch;
        config.separate_blocksPerGrid = (batch_size + config.separate_batches_per_block - 1) / config.separate_batches_per_block;
    }

    return config;
}

void launch_kinematics_forward(
  torch::Tensor link_pos, torch::Tensor link_quat,
  torch::Tensor batch_robot_spheres,
  torch::Tensor batch_center_of_mass, // [batch_size * 4] - xyz=global CoM, w=total mass
  torch::Tensor batch_jacobian,
  torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_masses_com,
  const torch::Tensor link_map,
  const torch::Tensor joint_map, const torch::Tensor joint_map_type,
  const torch::Tensor tool_frame_map, const torch::Tensor link_sphere_map,
  const torch::Tensor link_chain_data, const torch::Tensor link_chain_offsets,
  const torch::Tensor joint_links_data, const torch::Tensor joint_links_offsets,
  const torch::Tensor joint_affects_endeffector, // NEW: Precomputed cache from KinematicsTensorCfg
  const torch::Tensor joint_offset_map, // [n_links * 4] - xyz=local CoM, w=mass
  const torch::Tensor env_query_idx,
  const int64_t num_envs,
  const int64_t batch_size,
  const int64_t horizon,
  const int64_t n_joints,
  const int64_t n_spheres,
  const bool compute_jacobian,
  const bool compute_com)
{
  curobo::common::validate_cuda_input(joint_vec, "joint_vec");
  curobo::common::validate_cuda_input(link_pos, "link_pos");
  curobo::common::validate_cuda_input(link_quat, "link_quat");
  curobo::common::validate_cuda_input(global_cumul_mat, "global_cumul_mat");
  curobo::common::validate_cuda_input(batch_robot_spheres, "batch_robot_spheres");
  curobo::common::validate_cuda_input(fixed_transform, "fixed_transform");
  curobo::common::validate_cuda_input(robot_spheres, "robot_spheres");
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
  assert(env_query_idx.dtype() == torch::kInt32);
  curobo::common::validate_cuda_input(batch_jacobian, "batch_jacobian");
  const int n_links       = link_map.size(0);
  //const int n_tool_frames = link_pos.size(1);
  const int n_tool_frames = tool_frame_map.size(0);
  assert(joint_map.dtype() == torch::kInt16);
  assert(joint_map_type.dtype() == torch::kInt8);
  assert(tool_frame_map.dtype() == torch::kInt16);
  assert(link_sphere_map.dtype() == torch::kInt16);
  assert(link_map.dtype() == torch::kInt16);
  assert(joint_affects_endeffector.dtype() == torch::kBool);
  assert(batch_size > 0);

  // Calculate launch configuration using the new helper function
  const int max_threads_per_block = 128;
  KinFwdLaunchConfig config = calculate_kinematics_forward_launch_config(
    compute_jacobian,
    batch_size, n_links, n_spheres, n_tool_frames, max_threads_per_block);

  cudaStream_t stream = curobo::common::get_cuda_stream();


  if (config.use_single_kernel)
  {
    if (compute_jacobian) {
      kinematics_fused_jacobian_kernel
        << < config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream >> > (
        link_pos.data_ptr<float>(), link_quat.data_ptr<float>(),
        batch_robot_spheres.data_ptr<float>(),
        batch_center_of_mass.data_ptr<float>(),
        batch_jacobian.data_ptr<float>(),
        global_cumul_mat.data_ptr<float>(),
        joint_vec.data_ptr<float>(),
        fixed_transform.data_ptr<float>(),
        robot_spheres.data_ptr<float>(),
        link_masses_com.data_ptr<float>(),
        joint_map_type.data_ptr<int8_t>(),
        joint_map.data_ptr<int16_t>(),
        link_map.data_ptr<int16_t>(),
        tool_frame_map.data_ptr<int16_t>(),
        link_sphere_map.data_ptr<int16_t>(),
        link_chain_data.data_ptr<int16_t>(),
        link_chain_offsets.data_ptr<int16_t>(),
        joint_links_data.data_ptr<int16_t>(),
        joint_links_offsets.data_ptr<int16_t>(),
        joint_affects_endeffector.data_ptr<bool>(),
        joint_offset_map.data_ptr<float>(),
        env_query_idx.data_ptr<int32_t>(),
        batch_size, horizon, n_spheres,
        n_links, n_joints, n_tool_frames, num_envs);
    } else {
      kinematics_fused_kernel
        << < config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream >> > (
        link_pos.data_ptr<float>(), link_quat.data_ptr<float>(),
        batch_robot_spheres.data_ptr<float>(),
        batch_center_of_mass.data_ptr<float>(),
        global_cumul_mat.data_ptr<float>(),
        joint_vec.data_ptr<float>(),
        fixed_transform.data_ptr<float>(),
        robot_spheres.data_ptr<float>(),
        link_masses_com.data_ptr<float>(),
        joint_map_type.data_ptr<int8_t>(),
        joint_map.data_ptr<int16_t>(),
        link_map.data_ptr<int16_t>(),
        tool_frame_map.data_ptr<int16_t>(),
        link_sphere_map.data_ptr<int16_t>(),
        joint_links_data.data_ptr<int16_t>(),
        joint_links_offsets.data_ptr<int16_t>(),
        joint_offset_map.data_ptr<float>(),
        env_query_idx.data_ptr<int32_t>(),
        batch_size, horizon, n_spheres,
        n_links, n_joints, n_tool_frames, num_envs);
    }
  }
  else {
  // launch seperate kernels:
  // 1. kinematics_cumul_kernel computes the transform for each link and writes to global_cumul_mat
  kinematics_cumul_kernel<<<config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream>>>(
    global_cumul_mat.data_ptr<float>(),
    joint_vec.data_ptr<float>(),
    fixed_transform.data_ptr<float>(),
    joint_map_type.data_ptr<int8_t>(),
    joint_map.data_ptr<int16_t>(),
    link_map.data_ptr<int16_t>(),
    joint_offset_map.data_ptr<float>(),
    batch_size,
    n_links,
    n_joints
  );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (config.compute_jacobian) {
    //printf("compute_jacobian\n");

// 2. kernel that computes sphere locations and store link poses from global cumul mat
  kinematics_spheres_links_jacobian_kernel<<<config.separate_blocksPerGrid, config.separate_threadsPerBlock, config.separate_sharedMemSize, stream>>>(
    link_pos.data_ptr<float>(),
    link_quat.data_ptr<float>(),
    batch_robot_spheres.data_ptr<float>(),
    batch_center_of_mass.data_ptr<float>(),
    batch_jacobian.data_ptr<float>(),
    global_cumul_mat.data_ptr<float>(),
    robot_spheres.data_ptr<float>(),
    link_masses_com.data_ptr<float>(),
    tool_frame_map.data_ptr<int16_t>(),
    link_sphere_map.data_ptr<int16_t>(),
    env_query_idx.data_ptr<int32_t>(),
    joint_map_type.data_ptr<int8_t>(),
    joint_map.data_ptr<int16_t>(),
    link_map.data_ptr<int16_t>(),
    link_chain_data.data_ptr<int16_t>(),
    link_chain_offsets.data_ptr<int16_t>(),
    joint_links_data.data_ptr<int16_t>(),
    joint_links_offsets.data_ptr<int16_t>(),
    joint_affects_endeffector.data_ptr<bool>(),
    joint_offset_map.data_ptr<float>(),
    batch_size,
    horizon,
    n_spheres,
    num_envs,
    n_links,
    n_joints,
    n_tool_frames,
    config.separate_batches_per_block
  );



  }
  else {
  // 2. kernel that computes sphere locations and store link poses from global cumul mat
  kinematics_spheres_links_kernel<<<config.separate_blocksPerGrid, config.separate_threadsPerBlock, config.separate_sharedMemSize, stream>>>(
    link_pos.data_ptr<float>(),
    link_quat.data_ptr<float>(),
    batch_robot_spheres.data_ptr<float>(),
    batch_center_of_mass.data_ptr<float>(),
    global_cumul_mat.data_ptr<float>(),
    robot_spheres.data_ptr<float>(),
    link_masses_com.data_ptr<float>(),
    tool_frame_map.data_ptr<int16_t>(),
    link_sphere_map.data_ptr<int16_t>(),
    env_query_idx.data_ptr<int32_t>(),
    batch_size,
    horizon,
    n_spheres,
    num_envs,
    n_links,
    n_tool_frames,
    config.separate_batches_per_block
  );
  }


  }


  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return;

}



}
}