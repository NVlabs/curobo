/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "kinematics_constants.h"


#include "third_party/helper_math.h"
#include "common/math.cuh"
#include "common/quaternion_util.cuh"
#include "common/pose_util.cuh"
#include "kinematics_joint_util.cuh"
#include "kinematics_util.cuh"
#include "kinematics_forward_helper.cuh"

namespace curobo {
namespace kinematics {

    template<bool COMPUTE_COM=false, int N_LINKS=-1>
    __global__ void
    kinematics_fused_kernel(float *link_pos,             // batchSize xz n_tool_frames x M x M
                           float *link_quat,            // batchSize x n_tool_frames x M x M
                           float *b_robot_spheres,      // batchSize x nspheres x M
                           float *batch_center_of_mass, // [batch_size * 4] - xyz=global CoM, w=total mass
                           float *global_cumul_mat, // batchSize x nlinks x 3 x 4
                           const float *q,              // batchSize x njoints
                           const float *fixedTransform, // nlinks x 3 x 4
                           const float *robot_spheres,  // nspheres x M
                           const float *link_masses_com, // [n_links * 4] - xyz=local CoM, w=mass
                           const int8_t *jointMapType,     // nlinks
                           const int16_t *jointMap,        // nlinks
                           const int16_t *linkMap,         // nlinks. Contains the index of fixed transform for each link.
                           const int16_t *toolFrameMap,    // n_tool_frames
                           const int16_t *linkSphereMap,   // nspheres
                           const int16_t *jointLinksData,   // NEW: Packed array of links per joint (unused in non-jacobian kernel)
                           const int16_t *jointLinksOffsets, // NEW: Offset array for each joint's links (unused in non-jacobian kernel)
                           const float *jointOffset, // nlinks
                           const int32_t *env_query_idx,   // batchSize - env index per batch
                           const int batchSize, const int horizon,
                           const int nspheres,
                           const int nlinks, const int njoints,
                           const int n_tool_frames, const int num_envs)
    {
      extern __shared__ __align__(16) float cumul_mat[];

      int t           = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / 4;

      if (batch >= batchSize)
        return;


      int col_idx           = threadIdx.x % 4;
      const int local_batch = threadIdx.x / 4;
      const int matAddrBase = local_batch * nlinks * 12;

      // read all fixed transforms to local cache:

      // copy base link transform:
      initialize_base_link_transform(cumul_mat, fixedTransform, matAddrBase, col_idx, 4);

      // Compute joint transformations for all links
      compute_joint_transformations<N_LINKS>(cumul_mat, q, fixedTransform, jointMapType,
                                  jointMap, linkMap, jointOffset, batch,
                                  njoints, nlinks, matAddrBase, col_idx);
      __syncthreads();

      write_cumul_to_global_mem<12>(global_cumul_mat,
      cumul_mat,
      blockIdx.x * (blockDim.x / 4),
      threadIdx.x,
      nlinks,
      batchSize,
      blockDim.x / 4,
      4);

      // write out link:

      // do robot_spheres

      process_robot_spheres(b_robot_spheres, cumul_mat, robot_spheres, linkSphereMap,
                           env_query_idx, batch, nspheres, num_envs, horizon, col_idx,
                            4, matAddrBase);

      // write position and rotation, we convert rotation matrix to a quaternion and
      // write it out

      process_stored_links(link_pos, link_quat, cumul_mat, toolFrameMap,
                          batch, n_tool_frames, col_idx, 4, matAddrBase);


      if constexpr (COMPUTE_COM) {
      // compute center of mass
      // Calculate offset for center of mass shared memory (after cumul_mat)
      const int com_offset = (blockDim.x / 4) * nlinks * 12; // cumul_mat size in floats
      float4* shared_com_data = (float4*)&cumul_mat[com_offset];
      process_center_of_mass(batch_center_of_mass, cumul_mat, link_masses_com, shared_com_data,
                            batch, nlinks, col_idx, 4, local_batch * 4, matAddrBase);
      }
    }



    template<bool COMPUTE_COM=false, int N_LINKS=-1>
    __global__ void
    kinematics_fused_jacobian_kernel(float *link_pos,             // batchSize xz n_tool_frames x M x M
                           float *link_quat,            // batchSize x n_tool_frames x M x M
                           float *b_robot_spheres,      // batchSize x nspheres x M
                           float *batch_center_of_mass, // batchSize - xyz=global CoM, w=total mass
                           float *jacobian,             // batchSize x n_tool_frames x 6 x njoints
                           float *global_cumul_mat, // batchSize x nlinks x 3 x 4
                           const float *q,              // batchSize x njoints
                           const float *fixedTransform, // nlinks x 3 x 4
                           const float *robot_spheres,  // nspheres x M
                           const float *link_masses_com, // nlinks - xyz=local CoM, w=mass
                           const int8_t *jointMapType,     // nlinks
                           const int16_t *jointMap,        // nlinks
                           const int16_t *linkMap,         // nlinks. Contains the index of fixed transform for each link.
                           const int16_t *toolFrameMap,    // n_tool_frames
                           const int16_t *linkSphereMap,   // nspheres
                           const int16_t *linkChainData,   // Packed array of actual link indices
                           const int16_t *linkChainOffsets, // Offset array for each link's chain
                           const int16_t *jointLinksData,   // NEW: Packed array of links per joint
                           const int16_t *jointLinksOffsets, // NEW: Offset array for each joint's links
                           const bool *joint_affects_endeffector, // NEW: Precomputed cache from KinematicsTensorCfg
                           const float *jointOffset, // nlinks
                           const int32_t *env_query_idx,   // batchSize - env index per batch
                           const int batchSize, const int horizon,
                           const int nspheres,
                           const int nlinks, const int njoints,
                           const int n_tool_frames, const int num_envs)
    {
      extern __shared__ __align__(16) float cumul_mat[];

      int t           = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / 4;

      if (batch >= batchSize)
        return;

      int col_idx           = threadIdx.x % 4;
      const int local_batch = threadIdx.x / 4;
      const int matAddrBase = local_batch * nlinks * 12;

      // read all fixed transforms to local cache:

      // copy base link transform:
      initialize_base_link_transform(cumul_mat, fixedTransform, matAddrBase, col_idx, 4);

      // Compute joint transformations for all links
      compute_joint_transformations<N_LINKS>(cumul_mat, q, fixedTransform, jointMapType,
                                  jointMap, linkMap, jointOffset, batch,
                                  njoints, nlinks, matAddrBase, col_idx);
      __syncthreads();

      write_cumul_to_global_mem<12>(global_cumul_mat,
      cumul_mat,
      blockIdx.x * (blockDim.x / 4),
      threadIdx.x,
      nlinks,
      batchSize,
      blockDim.x / 4,
      4);

      // write out link:

      // do robot_spheres

      process_robot_spheres(b_robot_spheres, cumul_mat, robot_spheres, linkSphereMap,
                           env_query_idx, batch, nspheres, num_envs, horizon, col_idx,
                            4, matAddrBase);

      // write position and rotation, we convert rotation matrix to a quaternion and
      // write it out

      process_stored_links(link_pos, link_quat, cumul_mat, toolFrameMap,
                          batch, n_tool_frames, col_idx, 4, matAddrBase);

      if constexpr (COMPUTE_COM) {
      // compute center of mass
      // Calculate offset for center of mass shared memory (after cumul_mat)
      const int com_offset = (blockDim.x / 4) * nlinks * 12; // cumul_mat size in floats
      float4* shared_com_data = (float4*)&cumul_mat[com_offset];
      process_center_of_mass(batch_center_of_mass, cumul_mat, link_masses_com, shared_com_data,
                            batch, nlinks, col_idx, 4, local_batch * 4, matAddrBase);
      }
      // Compute kinematic jacobian for each stored link
      for (int16_t i = 0; i < n_tool_frames; i++) {
        const int16_t tool_frame_idx = toolFrameMap[i];
        // Note: jacobian buffer has size [batch_size * n_tool_frames * 6 * njoints]
        // Each stored link gets its own jacobian matrix
        float* link_jacobian = &jacobian[batch * n_tool_frames * 6 * njoints + i * 6 * njoints];

        compute_kinematic_jacobian(link_jacobian, cumul_mat,
                                   jointMapType, jointMap, linkMap, linkChainData, linkChainOffsets,
                                   jointLinksData, jointLinksOffsets, joint_affects_endeffector, toolFrameMap, jointOffset, tool_frame_idx, i,
                                   0, njoints, nlinks, n_tool_frames, matAddrBase, col_idx, 4);
      }

    }


    // launched across 4 threads per batch
    template<int N_LINKS=-1>
    __global__ void
    kinematics_cumul_kernel(
                           float *global_cumul_mat, // batchSize x nlinks x 3 x 4
                           const float *q,              // batchSize x njoints
                           const float *fixedTransform, // nlinks x 3 x 4
                           const int8_t *jointMapType,     // nlinks
                           const int16_t *jointMap,        // nlinks
                           const int16_t *linkMap,         // nlinks
                           const float *jointOffset, // nlinks
                           const int batchSize,
                           const int nlinks,
                           const int njoints)
    {
      extern __shared__ __align__(16) float cumul_mat[]; // shape is batches_per_block x nlinks x 3 x 4

      int t           = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / 4;
      const int num_batches_per_block = blockDim.x / 4;
      const int start_batch_index = blockIdx.x * num_batches_per_block;


      if (batch >= batchSize)
        return;

      int col_idx           = threadIdx.x % 4; // range is 0, 1, 2, 3
      const int local_batch = threadIdx.x / 4;
      const int matAddrBase = local_batch * nlinks * 12;

      // read all fixed transforms to local cache:

      // copy base link transform:
      initialize_base_link_transform(cumul_mat, fixedTransform, matAddrBase, col_idx, 4);


      // Compute joint transformations for all links
      compute_joint_transformations<N_LINKS>(cumul_mat, q, fixedTransform, jointMapType,
                                  jointMap, linkMap, jointOffset, batch,
                                  njoints, nlinks, matAddrBase, col_idx);
      __syncthreads();
      write_cumul_to_global_mem<12>(global_cumul_mat,
      cumul_mat,
      start_batch_index,
      threadIdx.x,
      nlinks,
      batchSize,
      num_batches_per_block,
      4);
    }

    template<bool COMPUTE_COM=false>
    __global__ void
    kinematics_spheres_links_kernel(float *link_pos,             // batchSize xz n_tool_frames x M x M
                           float *link_quat,            // batchSize x n_tool_frames x M x M
                           float *b_robot_spheres,      // batchSize x nspheres x M
                           float *batch_center_of_mass, // [batch_size * 4] - xyz=global CoM, w=total mass
                           float *global_cumul_mat, // batchSize x nlinks x M x M
                           const float *robot_spheres,  // nspheres x M
                           const float *link_masses_com, // [n_links * 4] - xyz=local CoM, w=mass
                           const int16_t *toolFrameMap,    // n_tool_frames
                           const int16_t *linkSphereMap,   // nspheres
                           const int32_t *env_query_idx,   // batchSize - env index per batch
                           const int batchSize,
                           const int horizon,
                           const int nspheres,
                           const int num_envs,
                           const int nlinks,
                           const int n_tool_frames,
                           const int num_batches_per_block)
    {
      extern __shared__ __align__(16) float shared_cumul_mat[];

      const int start_batch_index = blockIdx.x * num_batches_per_block;

      const int num_threads = blockDim.x;

      const int threads_per_batch = num_threads / num_batches_per_block;
      read_cumul_to_shared_mem(shared_cumul_mat, global_cumul_mat, start_batch_index, threadIdx.x, nlinks, batchSize, num_batches_per_block, threads_per_batch);


      const int local_batch_index = threadIdx.x / threads_per_batch;
      const int batch_index = start_batch_index + local_batch_index;
      const int thread_in_batch = threadIdx.x % threads_per_batch;


      if (batch_index >= batchSize)
        return;

      const int batchAddrs = batch_index * nspheres * 4;



      const int local_batch_offset = local_batch_index * nlinks * 12;

      process_robot_spheres(b_robot_spheres, &shared_cumul_mat[local_batch_offset], robot_spheres, linkSphereMap,
                           env_query_idx, batch_index, nspheres, num_envs, horizon, thread_in_batch, threads_per_batch, 0);

      // write position and rotation, we convert rotation matrix to a quaternion and
      // write it out

      process_stored_links(link_pos, link_quat, &shared_cumul_mat[local_batch_offset], toolFrameMap,
                          batch_index, n_tool_frames, thread_in_batch, threads_per_batch, 0);

      if constexpr (COMPUTE_COM) {
      // compute center of mass
      const int local_batch_start = local_batch_index * threads_per_batch;
      // Calculate offset for center of mass shared memory (after shared_cumul_mat)
      const int com_offset = num_batches_per_block * nlinks * 12; // shared_cumul_mat size in floats
      float4* shared_com_data = (float4*)&shared_cumul_mat[com_offset];
      process_center_of_mass(batch_center_of_mass, &shared_cumul_mat[local_batch_offset], link_masses_com, shared_com_data,
                            batch_index, nlinks, thread_in_batch, threads_per_batch,
                            local_batch_start, local_batch_offset);
      }


    }


    template<bool COMPUTE_COM=false>
    __global__ void
    kinematics_spheres_links_jacobian_kernel(
                           float *link_pos,             // batchSize xz n_tool_frames x M x M
                           float *link_quat,            // batchSize x n_tool_frames x M x M
                           float *b_robot_spheres,      // batchSize x nspheres x M
                           float *batch_center_of_mass, // [batch_size * 4] - xyz=global CoM, w=total mass
                           float *jacobian, // batchSize x n_tool_frames x 6 x njoints
                           float *global_cumul_mat, // batchSize x nlinks x M x M
                           const float *robot_spheres,  // nspheres x M
                           const float *link_masses_com, // [n_links * 4] - xyz=local CoM, w=mass
                           const int16_t *toolFrameMap,    // n_tool_frames
                           const int16_t *linkSphereMap,   // nspheres
                           const int32_t *env_query_idx,   // batchSize - env index per batch
                           const int8_t *jointMapType,     // nlinks
                           const int16_t *jointMap,        // nlinks
                           const int16_t *linkMap,         // nlinks
                           const int16_t *linkChainData,   // Packed array of actual link indices
                           const int16_t *linkChainOffsets, // Offset array for each link's chain
                           const int16_t *jointLinksData,   // NEW: Packed array of links per joint
                           const int16_t *jointLinksOffsets, // NEW: Offset array for each joint's links
                           const bool *joint_affects_endeffector, // NEW: Precomputed cache from KinematicsTensorCfg
                           const float *jointOffset, // nlinks
                           const int batchSize,
                           const int horizon,
                           const int nspheres,
                           const int num_envs,
                           const int nlinks,
                           const int njoints,
                           const int n_tool_frames,
                           const int num_batches_per_block)
    {
      extern __shared__ __align__(16) float shared_cumul_mat[];

      const int start_batch_index = blockIdx.x * num_batches_per_block;

      const int num_threads = blockDim.x;

      const int threads_per_batch = num_threads / num_batches_per_block;
      read_cumul_to_shared_mem(shared_cumul_mat, global_cumul_mat, start_batch_index, threadIdx.x, nlinks, batchSize, num_batches_per_block, threads_per_batch);


      const int local_batch_index = threadIdx.x / threads_per_batch;
      const int batch_index = start_batch_index + local_batch_index;
      const int thread_in_batch = threadIdx.x % threads_per_batch;


      if (batch_index >= batchSize)
        return;

      const int batchAddrs = batch_index * nspheres * 4;



      const int local_batch_offset = local_batch_index * nlinks * 12;

      process_robot_spheres(b_robot_spheres, &shared_cumul_mat[local_batch_offset], robot_spheres, linkSphereMap,
                           env_query_idx, batch_index, nspheres, num_envs, horizon, thread_in_batch, threads_per_batch, 0);

      // write position and rotation, we convert rotation matrix to a quaternion and
      // write it out

      process_stored_links(link_pos, link_quat, &shared_cumul_mat[local_batch_offset], toolFrameMap,
                          batch_index, n_tool_frames, thread_in_batch, threads_per_batch, 0);

      if constexpr (COMPUTE_COM) {
      // compute center of mass
      const int local_batch_start = local_batch_index * threads_per_batch;
      // Calculate offset for center of mass shared memory (after shared_cumul_mat)
      const int com_offset = num_batches_per_block * nlinks * 12; // shared_cumul_mat size in floats
      float4* shared_com_data = (float4*)&shared_cumul_mat[com_offset];
      process_center_of_mass(batch_center_of_mass, &shared_cumul_mat[local_batch_offset], link_masses_com, shared_com_data,
                            batch_index, nlinks, thread_in_batch, threads_per_batch,
                            local_batch_start, local_batch_offset);
      }
      // Compute kinematic jacobian for each stored link
      for (int16_t i = 0; i < n_tool_frames; i++) {
        const int16_t tool_frame_idx = toolFrameMap[i];
        // Note: jacobian buffer has size [batch_size * n_tool_frames * 6 * njoints]
        // Each stored link gets its own jacobian matrix
        float* link_jacobian = &jacobian[batch_index * n_tool_frames * 6 * njoints + i * 6 * njoints];

        compute_kinematic_jacobian(link_jacobian, &shared_cumul_mat[local_batch_offset],
                                   jointMapType, jointMap, linkMap, linkChainData, linkChainOffsets, jointLinksData, jointLinksOffsets, joint_affects_endeffector, toolFrameMap, jointOffset, tool_frame_idx, i,
                                   0, njoints, nlinks, n_tool_frames, 0, thread_in_batch, threads_per_batch);
      }

    }


}
}