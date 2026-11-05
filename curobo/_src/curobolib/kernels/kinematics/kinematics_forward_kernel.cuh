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
#include "kinematics_util.cuh"
#include "kinematics_forward_helper.cuh"

namespace curobo {
namespace kinematics {

    template<
      int N_LINKS,
      bool COMPUTE_COM,
      int NumJoints,
      int NumToolFrames>
    __global__ void kinematics_forward_kernel(
      float *link_pos,
      float *link_quat,
      float *batch_center_of_mass,
      float *global_cumul_mat,
      const float *q,
      const float *fixedTransform,
      const float *link_masses_com,
      const int8_t *jointMapType,
      const int16_t *jointMap,
      const int16_t *linkMap,
      const int16_t *toolFrameMap,
      const float *jointOffset,
      const int batchSize,
      const int horizon,
      const int nlinks,
      const int njoints,
      const int n_tool_frames
    ) {
      (void)horizon;
      extern __shared__ __align__(16) float shared_mem[];

      constexpr int active_num_links = N_LINKS;
      constexpr int active_num_joints = NumJoints;
      constexpr int active_num_tool_frames = NumToolFrames;
      const int num_batches_per_block = blockDim.x / 32;
      const int start_batch_index = blockIdx.x * num_batches_per_block;

      float *cumul_mat = shared_mem;
      float *local_mat = &shared_mem[num_batches_per_block * active_num_links * 12];

      compute_local_link_transforms<N_LINKS, NumJoints>(
        local_mat,
        q,
        fixedTransform,
        jointMapType,
        jointMap,
        jointOffset,
        start_batch_index,
        batchSize,
        active_num_links,
        active_num_joints,
        num_batches_per_block
      );

      const int actual_batches_per_block =
        min(num_batches_per_block, batchSize - start_batch_index);
      const int active_phase2_threads = actual_batches_per_block * 16;
      if (threadIdx.x >= active_phase2_threads) {
        return;
      }

      const int phase2_local_batch = threadIdx.x >> 4;
      const int phase2_batch = start_batch_index + phase2_local_batch;
      const int phase2_lane_idx = threadIdx.x & 15;
      const int phase2_matAddrBase = phase2_local_batch * active_num_links * 12;
      const unsigned int phase2_mask =
        (threadIdx.x & 16) == 0 ? 0x0000ffffu : 0xffff0000u;

      load_base_transform_halfwarp(
        cumul_mat,
        fixedTransform,
        active_num_links,
        phase2_matAddrBase,
        phase2_lane_idx,
        phase2_mask);

      compute_cumulative_transforms_halfwarp<N_LINKS>(
        cumul_mat,
        local_mat,
        linkMap,
        active_num_links,
        phase2_matAddrBase,
        phase2_matAddrBase,
        phase2_lane_idx,
        phase2_mask
      );

      __syncthreads();

      write_cumulative_transforms_active(
        global_cumul_mat,
        cumul_mat,
        start_batch_index,
        active_num_links,
        actual_batches_per_block,
        active_phase2_threads
      );

      if constexpr (COMPUTE_COM) {
        write_center_of_mass_tile<16, N_LINKS>(
          batch_center_of_mass,
          cumul_mat,
          link_masses_com,
          phase2_batch,
          active_num_links,
          phase2_lane_idx,
          phase2_mask,
          phase2_matAddrBase);
      }

      write_link_poses<NumToolFrames>(
        link_pos,
        link_quat,
        cumul_mat,
        toolFrameMap,
        phase2_batch,
        active_num_tool_frames,
        phase2_lane_idx,
        16,
        phase2_matAddrBase);
    }

    template<
      int N_LINKS,
      int OUTPUT_THREADS_PER_BATCH,
      bool WRITE_GLOBAL_CUMUL,
      bool COMPUTE_COM,
      int NumJoints,
      int NumSpheres,
      int NumToolFrames,
      int NumEnvs>
    __global__ void kinematics_forward_spheres_kernel(
      float *link_pos,
      float *link_quat,
      float *b_robot_spheres,
      float *batch_center_of_mass,
      float *global_cumul_mat,
      const float *q,
      const float *fixedTransform,
      const float *robot_spheres,
      const float *link_masses_com,
      const int8_t *jointMapType,
      const int16_t *jointMap,
      const int16_t *linkMap,
      const int16_t *toolFrameMap,
      const int16_t *linkSphereMap,
      const float *jointOffset,
      const int32_t *env_query_idx,
      const int batchSize,
      const int horizon,
      const int nspheres,
      const int num_envs,
      const int nlinks,
      const int njoints,
      const int n_tool_frames
    ) {
      extern __shared__ __align__(16) float shared_mem[];

      constexpr int active_num_links = N_LINKS;
      constexpr int active_num_joints = NumJoints;
      constexpr int active_num_spheres = NumSpheres;
      constexpr int active_num_tool_frames = NumToolFrames;
      constexpr int active_num_envs = NumEnvs;
      const int num_batches_per_block = blockDim.x / OUTPUT_THREADS_PER_BATCH;
      const int start_batch_index = blockIdx.x * num_batches_per_block;
      const int local_batch = threadIdx.x / OUTPUT_THREADS_PER_BATCH;
      const int batch = start_batch_index + local_batch;
      const bool active_tile = batch < batchSize;
      const int lane_idx = threadIdx.x - local_batch * OUTPUT_THREADS_PER_BATCH;
      const int matAddrBase = local_batch * active_num_links * 12;

      float *cumul_mat = shared_mem;
      float *local_mat = &shared_mem[num_batches_per_block * active_num_links * 12];

      compute_local_link_transforms<N_LINKS, NumJoints>(
        local_mat,
        q,
        fixedTransform,
        jointMapType,
        jointMap,
        jointOffset,
        start_batch_index,
        batchSize,
        active_num_links,
        active_num_joints,
        num_batches_per_block
      );

      const int actual_batches_per_block =
        min(num_batches_per_block, batchSize - start_batch_index);
      const int phase2_local_batch = threadIdx.x >> 4;
      const bool active_phase2_tile = phase2_local_batch < actual_batches_per_block;
      const int phase2_lane_idx = threadIdx.x & 15;
      const int phase2_matAddrBase = phase2_local_batch * active_num_links * 12;
      const unsigned int phase2_mask =
        (threadIdx.x & 16) == 0 ? 0x0000ffffu : 0xffff0000u;

      if (active_phase2_tile) {
        load_base_transform_halfwarp(
          cumul_mat,
          fixedTransform,
          active_num_links,
          phase2_matAddrBase,
          phase2_lane_idx,
          phase2_mask);

        compute_cumulative_transforms_halfwarp<N_LINKS>(
          cumul_mat,
          local_mat,
          linkMap,
          active_num_links,
          phase2_matAddrBase,
          phase2_matAddrBase,
          phase2_lane_idx,
          phase2_mask
        );
      }

      __syncthreads();

      if (WRITE_GLOBAL_CUMUL) {
        write_cumul_to_global_mem<12>(
          global_cumul_mat,
          cumul_mat,
          start_batch_index,
          threadIdx.x,
          active_num_links,
          batchSize,
          num_batches_per_block,
          OUTPUT_THREADS_PER_BATCH
        );
      }

      if (active_tile) {
        transform_robot_spheres<NumSpheres, NumEnvs>(
          b_robot_spheres,
          cumul_mat,
          robot_spheres,
          linkSphereMap,
          env_query_idx,
          batch,
          active_num_spheres,
          active_num_envs,
          horizon,
          lane_idx,
          OUTPUT_THREADS_PER_BATCH,
          matAddrBase);

        if constexpr (COMPUTE_COM) {
          write_center_of_mass_tile<32, N_LINKS>(
            batch_center_of_mass,
            cumul_mat,
            link_masses_com,
            batch,
            active_num_links,
            lane_idx,
            0xffffffffu,
            matAddrBase);
        }

        write_link_poses<NumToolFrames>(
          link_pos,
          link_quat,
          cumul_mat,
          toolFrameMap,
          batch,
          active_num_tool_frames,
          lane_idx,
          OUTPUT_THREADS_PER_BATCH,
          matAddrBase);
      }
    }

    template<
      int N_LINKS,
      int OUTPUT_THREADS_PER_BATCH,
      bool WRITE_GLOBAL_CUMUL,
      bool COMPUTE_COM,
      int NumJoints,
      int NumSpheres,
      int NumToolFrames,
      int NumEnvs>
    __global__ void kinematics_forward_spheres_jacobian_kernel(
      float *link_pos,
      float *link_quat,
      float *b_robot_spheres,
      float *batch_center_of_mass,
      float *jacobian,
      float *global_cumul_mat,
      const float *q,
      const float *fixedTransform,
      const float *robot_spheres,
      const float *link_masses_com,
      const int8_t *jointMapType,
      const int16_t *jointMap,
      const int16_t *linkMap,
      const int16_t *toolFrameMap,
      const int16_t *linkSphereMap,
      const int16_t *toolJointLinksData,
      const int16_t *toolJointLinksOffsets,
      const float *jointOffset,
      const int32_t *env_query_idx,
      const int batchSize,
      const int horizon,
      const int nspheres,
      const int num_envs,
      const int nlinks,
      const int njoints,
      const int n_tool_frames
    ) {
      extern __shared__ __align__(16) float shared_mem[];

      constexpr int active_num_links = N_LINKS;
      constexpr int active_num_joints = NumJoints;
      constexpr int active_num_spheres = NumSpheres;
      constexpr int active_num_tool_frames = NumToolFrames;
      constexpr int active_num_envs = NumEnvs;
      const int num_batches_per_block = blockDim.x / OUTPUT_THREADS_PER_BATCH;
      const int start_batch_index = blockIdx.x * num_batches_per_block;
      const int local_batch = threadIdx.x / OUTPUT_THREADS_PER_BATCH;
      const int batch = start_batch_index + local_batch;
      const bool active_tile = batch < batchSize;
      const int lane_idx = threadIdx.x - local_batch * OUTPUT_THREADS_PER_BATCH;
      const int matAddrBase = local_batch * active_num_links * 12;

      float *cumul_mat = shared_mem;
      float *local_mat = &shared_mem[num_batches_per_block * active_num_links * 12];

      compute_local_link_transforms<N_LINKS, NumJoints>(
        local_mat,
        q,
        fixedTransform,
        jointMapType,
        jointMap,
        jointOffset,
        start_batch_index,
        batchSize,
        active_num_links,
        active_num_joints,
        num_batches_per_block
      );

      const int actual_batches_per_block =
        min(num_batches_per_block, batchSize - start_batch_index);
      const int phase2_local_batch = threadIdx.x >> 4;
      const bool active_phase2_tile = phase2_local_batch < actual_batches_per_block;
      const int phase2_lane_idx = threadIdx.x & 15;
      const int phase2_matAddrBase = phase2_local_batch * active_num_links * 12;
      const unsigned int phase2_mask =
        (threadIdx.x & 16) == 0 ? 0x0000ffffu : 0xffff0000u;

      if (active_phase2_tile) {
        load_base_transform_halfwarp(
          cumul_mat,
          fixedTransform,
          active_num_links,
          phase2_matAddrBase,
          phase2_lane_idx,
          phase2_mask);

        compute_cumulative_transforms_halfwarp<N_LINKS>(
          cumul_mat,
          local_mat,
          linkMap,
          active_num_links,
          phase2_matAddrBase,
          phase2_matAddrBase,
          phase2_lane_idx,
          phase2_mask
        );
      }

      __syncthreads();

      if (WRITE_GLOBAL_CUMUL) {
        write_cumul_to_global_mem<12>(
          global_cumul_mat,
          cumul_mat,
          start_batch_index,
          threadIdx.x,
          active_num_links,
          batchSize,
          num_batches_per_block,
          OUTPUT_THREADS_PER_BATCH
        );
      }

      if (active_tile) {
        transform_robot_spheres<NumSpheres, NumEnvs>(
          b_robot_spheres,
          cumul_mat,
          robot_spheres,
          linkSphereMap,
          env_query_idx,
          batch,
          active_num_spheres,
          active_num_envs,
          horizon,
          lane_idx,
          OUTPUT_THREADS_PER_BATCH,
          matAddrBase);

        if constexpr (COMPUTE_COM) {
          write_center_of_mass_tile<32, N_LINKS>(
            batch_center_of_mass,
            cumul_mat,
            link_masses_com,
            batch,
            active_num_links,
            lane_idx,
            0xffffffffu,
            matAddrBase);
        }

        write_link_poses<NumToolFrames>(
          link_pos,
          link_quat,
          cumul_mat,
          toolFrameMap,
          batch,
          active_num_tool_frames,
          lane_idx,
          OUTPUT_THREADS_PER_BATCH,
          matAddrBase);

        float *batch_jacobian =
          &jacobian[batch * active_num_tool_frames * 6 * active_num_joints];
        write_kinematic_jacobians<NumJoints, NumToolFrames, OUTPUT_THREADS_PER_BATCH>(
          batch_jacobian,
          cumul_mat,
          jointMapType,
          toolJointLinksData,
          toolJointLinksOffsets,
          toolFrameMap,
          jointOffset,
          matAddrBase,
          lane_idx);
      }
    }


}
}
