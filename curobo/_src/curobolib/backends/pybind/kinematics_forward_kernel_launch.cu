/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include <torch/extension.h>

#include "common/torch_cuda_utils.h"
#include "kinematics/kinematics_forward_kernel.cuh"

namespace curobo {
namespace kinematics {

namespace {

constexpr int kMaxForwardLinks = 500;
constexpr int kMaxForwardBatchesPerBlock = MAX_FW_BATCH_PER_BLOCK;
constexpr int kDefaultMaxSharedMemory = 48 * 1024;

struct KinFwdLaunchConfig {
  int threadsPerBlock;
  int blocksPerGrid;
  int sharedMemSize;
};

KinFwdLaunchConfig calculate_kinematics_forward_launch_config(
  const int64_t batch_size,
  const int n_links,
  const int threads_per_batch,
  const int max_threads_per_block = 256,
  const int max_shared_memory = kDefaultMaxSharedMemory)
{
  if (n_links > kMaxForwardLinks) {
    throw std::runtime_error("Forward kinematics kernel supports at most 500 links");
  }

  const int shared_mem_per_batch = n_links * 12 * static_cast<int>(sizeof(float)) * 2;
  if (shared_mem_per_batch > max_shared_memory) {
    throw std::runtime_error("Single batch shared memory requirement exceeds limit");
  }

  int max_batches_per_block = kMaxForwardBatchesPerBlock;
  const int max_batches_from_shared_mem = max_shared_memory / shared_mem_per_batch;
  max_batches_per_block = std::min(max_batches_per_block, max_batches_from_shared_mem);

  if (max_batches_per_block * threads_per_batch > max_threads_per_block) {
    max_batches_per_block = max_threads_per_block / threads_per_batch;
  }

  max_batches_per_block = std::max(1, max_batches_per_block);
  const int batches_per_block = std::min(static_cast<int>(batch_size), max_batches_per_block);

  KinFwdLaunchConfig config;
  config.threadsPerBlock = batches_per_block * threads_per_batch;
  config.blocksPerGrid =
    (static_cast<int>(batch_size) + batches_per_block - 1) / batches_per_block;
  config.sharedMemSize = batches_per_block * shared_mem_per_batch;
  return config;
}

void validate_forward_common(
  const torch::Tensor &link_pos,
  const torch::Tensor &link_quat,
  const torch::Tensor &batch_center_of_mass,
  const torch::Tensor &global_cumul_mat,
  const torch::Tensor &joint_vec,
  const torch::Tensor &fixed_transform,
  const torch::Tensor &link_masses_com,
  const torch::Tensor &joint_map_type,
  const torch::Tensor &joint_map,
  const torch::Tensor &link_map,
  const torch::Tensor &tool_frame_map,
  const torch::Tensor &joint_offset_map)
{
  curobo::common::validate_cuda_input(link_pos, "link_pos");
  curobo::common::validate_cuda_input(link_quat, "link_quat");
  curobo::common::validate_cuda_input(batch_center_of_mass, "batch_center_of_mass");
  curobo::common::validate_cuda_input(global_cumul_mat, "global_cumul_mat");
  curobo::common::validate_cuda_input(joint_vec, "joint_vec");
  curobo::common::validate_cuda_input(fixed_transform, "fixed_transform");
  curobo::common::validate_cuda_input(link_masses_com, "link_masses_com");
  curobo::common::validate_cuda_input(joint_map_type, "joint_map_type");
  curobo::common::validate_cuda_input(joint_map, "joint_map");
  curobo::common::validate_cuda_input(link_map, "link_map");
  curobo::common::validate_cuda_input(tool_frame_map, "tool_frame_map");
  curobo::common::validate_cuda_input(joint_offset_map, "joint_offset_map");

  assert(joint_map_type.dtype() == torch::kInt8);
  assert(joint_map.dtype() == torch::kInt16);
  assert(link_map.dtype() == torch::kInt16);
  assert(tool_frame_map.dtype() == torch::kInt16);
}

template<bool COMPUTE_COM>
void launch_forward_no_spheres_kernel(
  const KinFwdLaunchConfig &config,
  cudaStream_t stream,
  torch::Tensor link_pos,
  torch::Tensor link_quat,
  torch::Tensor batch_center_of_mass,
  torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor link_masses_com,
  const torch::Tensor joint_map_type,
  const torch::Tensor joint_map,
  const torch::Tensor link_map,
  const torch::Tensor tool_frame_map,
  const torch::Tensor joint_offset_map,
  const int64_t batch_size,
  const int64_t horizon,
  const int64_t n_joints)
{
  const int n_links = link_map.size(0);
  const int n_tool_frames = tool_frame_map.size(0);

  kinematics_forward_kernel<-1, COMPUTE_COM>
    <<<config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream>>>(
      link_pos.data_ptr<float>(),
      link_quat.data_ptr<float>(),
      batch_center_of_mass.data_ptr<float>(),
      global_cumul_mat.data_ptr<float>(),
      joint_vec.data_ptr<float>(),
      fixed_transform.data_ptr<float>(),
      link_masses_com.data_ptr<float>(),
      joint_map_type.data_ptr<int8_t>(),
      joint_map.data_ptr<int16_t>(),
      link_map.data_ptr<int16_t>(),
      tool_frame_map.data_ptr<int16_t>(),
      joint_offset_map.data_ptr<float>(),
      batch_size,
      horizon,
      n_links,
      n_joints,
      n_tool_frames);
}

template<int OUTPUT_THREADS_PER_BATCH, bool WRITE_GLOBAL_CUMUL, bool COMPUTE_COM>
void launch_forward_spheres_kernel(
  const KinFwdLaunchConfig &config,
  cudaStream_t stream,
  torch::Tensor link_pos,
  torch::Tensor link_quat,
  torch::Tensor batch_robot_spheres,
  torch::Tensor batch_center_of_mass,
  torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_masses_com,
  const torch::Tensor joint_map_type,
  const torch::Tensor joint_map,
  const torch::Tensor link_map,
  const torch::Tensor tool_frame_map,
  const torch::Tensor link_sphere_map,
  const torch::Tensor joint_offset_map,
  const torch::Tensor env_query_idx,
  const int64_t num_envs,
  const int64_t batch_size,
  const int64_t horizon,
  const int64_t n_joints,
  const int64_t num_spheres)
{
  const int n_links = link_map.size(0);
  const int n_tool_frames = tool_frame_map.size(0);

  kinematics_forward_spheres_kernel<-1, OUTPUT_THREADS_PER_BATCH, WRITE_GLOBAL_CUMUL, COMPUTE_COM>
    <<<config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream>>>(
      link_pos.data_ptr<float>(),
      link_quat.data_ptr<float>(),
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
      joint_offset_map.data_ptr<float>(),
      env_query_idx.data_ptr<int32_t>(),
      batch_size,
      horizon,
      num_spheres,
      num_envs,
      n_links,
      n_joints,
      n_tool_frames);
}

template<int OUTPUT_THREADS_PER_BATCH, bool WRITE_GLOBAL_CUMUL, bool COMPUTE_COM>
void launch_forward_spheres_jacobian_kernel(
  const KinFwdLaunchConfig &config,
  cudaStream_t stream,
  torch::Tensor link_pos,
  torch::Tensor link_quat,
  torch::Tensor batch_robot_spheres,
  torch::Tensor batch_center_of_mass,
  torch::Tensor batch_jacobian,
  torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_masses_com,
  const torch::Tensor joint_map_type,
  const torch::Tensor joint_map,
  const torch::Tensor link_map,
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
  const int64_t num_spheres)
{
  const int n_links = link_map.size(0);
  const int n_tool_frames = tool_frame_map.size(0);

  kinematics_forward_spheres_jacobian_kernel
    <-1, OUTPUT_THREADS_PER_BATCH, WRITE_GLOBAL_CUMUL, COMPUTE_COM>
    <<<config.blocksPerGrid, config.threadsPerBlock, config.sharedMemSize, stream>>>(
      link_pos.data_ptr<float>(),
      link_quat.data_ptr<float>(),
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
      batch_size,
      horizon,
      num_spheres,
      num_envs,
      n_links,
      n_joints,
      n_tool_frames);
}

}  // namespace

void launch_kinematics_forward(
  torch::Tensor link_pos,
  torch::Tensor link_quat,
  torch::Tensor batch_center_of_mass,
  torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor link_masses_com,
  const torch::Tensor joint_map_type,
  const torch::Tensor joint_map,
  const torch::Tensor link_map,
  const torch::Tensor tool_frame_map,
  const torch::Tensor joint_offset_map,
  const int64_t batch_size,
  const int64_t horizon,
  const int64_t n_joints,
  const bool compute_com)
{
  validate_forward_common(
    link_pos,
    link_quat,
    batch_center_of_mass,
    global_cumul_mat,
    joint_vec,
    fixed_transform,
    link_masses_com,
    joint_map_type,
    joint_map,
    link_map,
    tool_frame_map,
    joint_offset_map);

  assert(batch_size > 0);

  const int n_links = link_map.size(0);
  const KinFwdLaunchConfig config =
    calculate_kinematics_forward_launch_config(batch_size, n_links, 32);
  cudaStream_t stream = curobo::common::get_cuda_stream();

  if (compute_com) {
    launch_forward_no_spheres_kernel<true>(
      config,
      stream,
      link_pos,
      link_quat,
      batch_center_of_mass,
      global_cumul_mat,
      joint_vec,
      fixed_transform,
      link_masses_com,
      joint_map_type,
      joint_map,
      link_map,
      tool_frame_map,
      joint_offset_map,
      batch_size,
      horizon,
      n_joints);
  } else {
    launch_forward_no_spheres_kernel<false>(
      config,
      stream,
      link_pos,
      link_quat,
      batch_center_of_mass,
      global_cumul_mat,
      joint_vec,
      fixed_transform,
      link_masses_com,
      joint_map_type,
      joint_map,
      link_map,
      tool_frame_map,
      joint_offset_map,
      batch_size,
      horizon,
      n_joints);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_kinematics_forward_spheres(
  torch::Tensor link_pos,
  torch::Tensor link_quat,
  torch::Tensor batch_robot_spheres,
  torch::Tensor batch_center_of_mass,
  torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_masses_com,
  const torch::Tensor joint_map_type,
  const torch::Tensor joint_map,
  const torch::Tensor link_map,
  const torch::Tensor tool_frame_map,
  const torch::Tensor link_sphere_map,
  const torch::Tensor joint_offset_map,
  const torch::Tensor env_query_idx,
  const int64_t num_envs,
  const int64_t batch_size,
  const int64_t horizon,
  const int64_t n_joints,
  const int64_t num_spheres,
  const int64_t output_threads_per_batch,
  const bool write_global_cumul,
  const bool compute_com)
{
  validate_forward_common(
    link_pos,
    link_quat,
    batch_center_of_mass,
    global_cumul_mat,
    joint_vec,
    fixed_transform,
    link_masses_com,
    joint_map_type,
    joint_map,
    link_map,
    tool_frame_map,
    joint_offset_map);
  curobo::common::validate_cuda_input(batch_robot_spheres, "batch_robot_spheres");
  curobo::common::validate_cuda_input(robot_spheres, "robot_spheres");
  curobo::common::validate_cuda_input(link_sphere_map, "link_sphere_map");
  curobo::common::validate_cuda_input(env_query_idx, "env_query_idx");

  assert(link_sphere_map.dtype() == torch::kInt16);
  assert(env_query_idx.dtype() == torch::kInt32);
  assert(batch_size > 0);

  if (output_threads_per_batch != 32 && output_threads_per_batch != 64 &&
      output_threads_per_batch != 128) {
    throw std::runtime_error("output_threads_per_batch must be one of 32, 64, or 128");
  }

  const int n_links = link_map.size(0);
  const KinFwdLaunchConfig config = calculate_kinematics_forward_launch_config(
    batch_size, n_links, static_cast<int>(output_threads_per_batch));
  cudaStream_t stream = curobo::common::get_cuda_stream();

#define LAUNCH_SPHERES(THREADS, WRITE_CUMUL, COMPUTE_CENTER_OF_MASS)                  \
  launch_forward_spheres_kernel<THREADS, WRITE_CUMUL, COMPUTE_CENTER_OF_MASS>(        \
    config,                                                                           \
    stream,                                                                           \
    link_pos,                                                                         \
    link_quat,                                                                        \
    batch_robot_spheres,                                                              \
    batch_center_of_mass,                                                             \
    global_cumul_mat,                                                                 \
    joint_vec,                                                                        \
    fixed_transform,                                                            \
    robot_spheres,                                                                    \
    link_masses_com,                                                                  \
    joint_map_type,                                                                   \
    joint_map,                                                                        \
    link_map,                                                                         \
    tool_frame_map,                                                                   \
    link_sphere_map,                                                                  \
    joint_offset_map,                                                                 \
    env_query_idx,                                                                    \
    num_envs,                                                                         \
    batch_size,                                                                       \
    horizon,                                                                          \
    n_joints,                                                                         \
    num_spheres)

  if (output_threads_per_batch == 32) {
    if (write_global_cumul) {
      if (compute_com) {
        LAUNCH_SPHERES(32, true, true);
      } else {
        LAUNCH_SPHERES(32, true, false);
      }
    } else {
      if (compute_com) {
        LAUNCH_SPHERES(32, false, true);
      } else {
        LAUNCH_SPHERES(32, false, false);
      }
    }
  } else if (output_threads_per_batch == 64) {
    if (write_global_cumul) {
      if (compute_com) {
        LAUNCH_SPHERES(64, true, true);
      } else {
        LAUNCH_SPHERES(64, true, false);
      }
    } else {
      if (compute_com) {
        LAUNCH_SPHERES(64, false, true);
      } else {
        LAUNCH_SPHERES(64, false, false);
      }
    }
  } else {
    if (write_global_cumul) {
      if (compute_com) {
        LAUNCH_SPHERES(128, true, true);
      } else {
        LAUNCH_SPHERES(128, true, false);
      }
    } else {
      if (compute_com) {
        LAUNCH_SPHERES(128, false, true);
      } else {
        LAUNCH_SPHERES(128, false, false);
      }
    }
  }

#undef LAUNCH_SPHERES

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_kinematics_forward_spheres_jacobian(
  torch::Tensor link_pos,
  torch::Tensor link_quat,
  torch::Tensor batch_robot_spheres,
  torch::Tensor batch_center_of_mass,
  torch::Tensor batch_jacobian,
  torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_masses_com,
  const torch::Tensor joint_map_type,
  const torch::Tensor joint_map,
  const torch::Tensor link_map,
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
  const int64_t num_spheres,
  const int64_t output_threads_per_batch,
  const bool write_global_cumul,
  const bool compute_com)
{
  validate_forward_common(
    link_pos,
    link_quat,
    batch_center_of_mass,
    global_cumul_mat,
    joint_vec,
    fixed_transform,
    link_masses_com,
    joint_map_type,
    joint_map,
    link_map,
    tool_frame_map,
    joint_offset_map);
  curobo::common::validate_cuda_input(batch_robot_spheres, "batch_robot_spheres");
  curobo::common::validate_cuda_input(batch_jacobian, "batch_jacobian");
  curobo::common::validate_cuda_input(robot_spheres, "robot_spheres");
  curobo::common::validate_cuda_input(link_sphere_map, "link_sphere_map");
  curobo::common::validate_cuda_input(link_chain_data, "link_chain_data");
  curobo::common::validate_cuda_input(link_chain_offsets, "link_chain_offsets");
  curobo::common::validate_cuda_input(joint_links_data, "joint_links_data");
  curobo::common::validate_cuda_input(joint_links_offsets, "joint_links_offsets");
  curobo::common::validate_cuda_input(joint_affects_endeffector, "joint_affects_endeffector");
  curobo::common::validate_cuda_input(env_query_idx, "env_query_idx");

  assert(link_sphere_map.dtype() == torch::kInt16);
  assert(link_chain_data.dtype() == torch::kInt16);
  assert(link_chain_offsets.dtype() == torch::kInt16);
  assert(joint_links_data.dtype() == torch::kInt16);
  assert(joint_links_offsets.dtype() == torch::kInt16);
  assert(joint_affects_endeffector.dtype() == torch::kBool);
  assert(env_query_idx.dtype() == torch::kInt32);
  assert(batch_size > 0);

  if (output_threads_per_batch != 32 && output_threads_per_batch != 64 &&
      output_threads_per_batch != 128) {
    throw std::runtime_error("output_threads_per_batch must be one of 32, 64, or 128");
  }

  const int n_links = link_map.size(0);
  const KinFwdLaunchConfig config = calculate_kinematics_forward_launch_config(
    batch_size, n_links, static_cast<int>(output_threads_per_batch));
  cudaStream_t stream = curobo::common::get_cuda_stream();

#define LAUNCH_SPHERES_JACOBIAN(THREADS, WRITE_CUMUL, COMPUTE_CENTER_OF_MASS)          \
  launch_forward_spheres_jacobian_kernel<THREADS, WRITE_CUMUL, COMPUTE_CENTER_OF_MASS>(\
    config,                                                                            \
    stream,                                                                            \
    link_pos,                                                                          \
    link_quat,                                                                         \
    batch_robot_spheres,                                                               \
    batch_center_of_mass,                                                              \
    batch_jacobian,                                                                    \
    global_cumul_mat,                                                                  \
    joint_vec,                                                                         \
    fixed_transform,                                                             \
    robot_spheres,                                                                     \
    link_masses_com,                                                                   \
    joint_map_type,                                                                    \
    joint_map,                                                                         \
    link_map,                                                                          \
    tool_frame_map,                                                                    \
    link_sphere_map,                                                                   \
    link_chain_data,                                                                   \
    link_chain_offsets,                                                                \
    joint_links_data,                                                                  \
    joint_links_offsets,                                                               \
    joint_affects_endeffector,                                                         \
    joint_offset_map,                                                                  \
    env_query_idx,                                                                     \
    num_envs,                                                                          \
    batch_size,                                                                        \
    horizon,                                                                           \
    n_joints,                                                                          \
    num_spheres)

  if (output_threads_per_batch == 32) {
    if (write_global_cumul) {
      if (compute_com) {
        LAUNCH_SPHERES_JACOBIAN(32, true, true);
      } else {
        LAUNCH_SPHERES_JACOBIAN(32, true, false);
      }
    } else {
      if (compute_com) {
        LAUNCH_SPHERES_JACOBIAN(32, false, true);
      } else {
        LAUNCH_SPHERES_JACOBIAN(32, false, false);
      }
    }
  } else if (output_threads_per_batch == 64) {
    if (write_global_cumul) {
      if (compute_com) {
        LAUNCH_SPHERES_JACOBIAN(64, true, true);
      } else {
        LAUNCH_SPHERES_JACOBIAN(64, true, false);
      }
    } else {
      if (compute_com) {
        LAUNCH_SPHERES_JACOBIAN(64, false, true);
      } else {
        LAUNCH_SPHERES_JACOBIAN(64, false, false);
      }
    }
  } else {
    if (write_global_cumul) {
      if (compute_com) {
        LAUNCH_SPHERES_JACOBIAN(128, true, true);
      } else {
        LAUNCH_SPHERES_JACOBIAN(128, true, false);
      }
    } else {
      if (compute_com) {
        LAUNCH_SPHERES_JACOBIAN(128, false, true);
      } else {
        LAUNCH_SPHERES_JACOBIAN(128, false, false);
      }
    }
  }

#undef LAUNCH_SPHERES_JACOBIAN

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace kinematics
}  // namespace curobo
