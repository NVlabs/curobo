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

namespace curobo{
    namespace kinematics{

        /**
         * Optimized helper function to compute kinematic jacobian using precomputed joint-link mapping.
         * This eliminates the inner loop search and directly accesses links connected to each joint.
         * The jacobian relates joint velocities to end-effector linear and angular velocities.
         * Output jacobian has shape [6 x num_joints] where:
         * - First 3 rows: linear velocity jacobian (∂position/∂q)
         * - Last 3 rows: angular velocity jacobian (∂orientation/∂q)
         *
         * @param jacobian Output jacobian matrix buffer [6 x num_joints]
         * @param cumul_mat Cumulative transformation matrix buffer
         * @param jointMapType Joint type mapping (FIXED, X_PRISM, Y_PRISM, Z_PRISM, X_ROT, Y_ROT, Z_ROT)
         * @param jointMap Joint index mapping
         * @param linkMap Link parent mapping
         * @param linkChainData Packed array of actual link indices for all kinematic chains
         * @param linkChainOffsets Offset array indicating where each link's chain starts in linkChainData
         * @param jointLinksData Precomputed packed array of links connected to each joint
         * @param jointLinksOffsets Offset array for each joint's connected links
         * @param jointOffset Joint offset values
         * @param tool_frame_idx End-effector link index
         * @param batch_index Current batch index
         * @param njoints Number of joints
         * @param nlinks Number of links
         * @param matAddrBase Base address in cumulative matrix
         * @param thread_index Current thread index within batch
         * @param threads_per_batch Number of threads per batch
         */
        __forceinline__ __device__ void write_kinematic_jacobian(
          float *jacobian,
          const float *cumul_mat,
          const int8_t *jointMapType,
          const int16_t *jointMap,
          const int16_t *linkMap,
          const int16_t *linkChainData,
          const int16_t *linkChainOffsets,
          const int16_t *jointLinksData,
          const int16_t *jointLinksOffsets,
          const bool *joint_affects_endeffector,  // NEW: Cache for optimization
          const int16_t *toolFrameMap,             // NEW: Added missing parameter
          const float *jointOffset,
          const int16_t tool_frame_idx,
          const int16_t ee_idx,                    // NEW: Index in toolFrameMap array
          const int batch_index,
          const int njoints,
          const int nlinks,
          const int n_tool_frames,
          const int matAddrBase,
          const int thread_index,
          const int threads_per_batch
        ) {
          // Get end-effector position
          const int ee_addr = matAddrBase + tool_frame_idx * 12;
          const float3 ee_pos = make_float3(
            cumul_mat[ee_addr + 3],
            cumul_mat[ee_addr + 7],
            cumul_mat[ee_addr + 11]
          );

          // Use precomputed chain data to get links that affect this end-effector
          const int16_t chain_start = linkChainOffsets[tool_frame_idx];
          const int16_t chain_end = linkChainOffsets[tool_frame_idx + 1];

          // Distribute joint processing across threads for better parallelization
          // This eliminates the need for atomicAdd since each thread has exclusive access to its joints
          const int16_t joints_per_thread = curobo::common::ceil_div(njoints, threads_per_batch);

          for (int16_t i = 0; i < joints_per_thread; i++) {
            // Use strided indexing for joints for better memory coalescing
            int16_t joint_idx = i * threads_per_batch + thread_index;

            if (joint_idx >= njoints) {
              break;
            }

            // 🚀 EARLY EXIT: Check cache first - O(1) lookup
            if (ee_idx >= 0 && !joint_affects_endeffector[joint_idx * n_tool_frames + ee_idx]) {
              // Write zeros directly - skip all expensive computation!
              jacobian[0 * njoints + joint_idx] = 0.0f;  // Linear X
              jacobian[1 * njoints + joint_idx] = 0.0f;  // Linear Y
              jacobian[2 * njoints + joint_idx] = 0.0f;  // Linear Z
              jacobian[3 * njoints + joint_idx] = 0.0f;  // Angular X
              jacobian[4 * njoints + joint_idx] = 0.0f;  // Angular Y
              jacobian[5 * njoints + joint_idx] = 0.0f;  // Angular Z
              continue;  // Skip expensive computation entirely!
            }

            // Initialize jacobian column to zero
            float jac_col[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

            // Get precomputed links connected to this joint - MUCH FASTER!
            const int16_t joint_links_start = jointLinksOffsets[joint_idx];
            const int16_t joint_links_end = jointLinksOffsets[joint_idx + 1];

            // Iterate through precomputed links connected to this joint
            for (int16_t jl_idx = joint_links_start; jl_idx < joint_links_end; jl_idx++) {
              const int16_t joint_link_idx = jointLinksData[jl_idx];

              // Skip base link (link 0) as it has no joint
              if (joint_link_idx == 0) {
                continue;
              }

              // Check if this link is in the kinematic chain affecting the end-effector
              bool link_in_chain = false;
              for (int16_t chain_idx = chain_start; chain_idx < chain_end; chain_idx++) {
                if (linkChainData[chain_idx] == joint_link_idx) {
                  link_in_chain = true;
                  break;
                }
              }

              if (!link_in_chain) {
                continue;
              }

              const int joint_addr = matAddrBase + joint_link_idx * 12;
              const int8_t joint_type = jointMapType[joint_link_idx];
              const float axis_sign = jointOffset[joint_link_idx * 2];

              // Get joint position
              const float3 joint_pos = make_float3(
                cumul_mat[joint_addr + 3],
                cumul_mat[joint_addr + 7],
                cumul_mat[joint_addr + 11]
              );

              if (joint_type >= int(JointType::XRevolute) && joint_type <= int(JointType::ZRevolute)) {
                // Revolute joint
                const int axis_idx = joint_type - int(JointType::XRevolute);
                const float3 joint_axis = make_float3(
                  cumul_mat[joint_addr + axis_idx],
                  cumul_mat[joint_addr + 4 + axis_idx],
                  cumul_mat[joint_addr + 8 + axis_idx]
                );

                const float3 scaled_axis = axis_sign * joint_axis;

                // Linear velocity: ω × (p_ee - p_joint)
                const float3 pos_diff = ee_pos - joint_pos;
                const float3 linear_vel = cross(scaled_axis, pos_diff);

                // Angular velocity: ω
                const float3 angular_vel = scaled_axis;

                // Accumulate contributions for mimic joints
                jac_col[0] += linear_vel.x;
                jac_col[1] += linear_vel.y;
                jac_col[2] += linear_vel.z;
                jac_col[3] += angular_vel.x;
                jac_col[4] += angular_vel.y;
                jac_col[5] += angular_vel.z;

              } else if (joint_type >= int(JointType::XPrismatic) && joint_type <= int(JointType::ZPrismatic)) {
                // Prismatic joint
                const int axis_idx = joint_type - int(JointType::XPrismatic);
                const float3 joint_axis = make_float3(
                  cumul_mat[joint_addr + axis_idx],
                  cumul_mat[joint_addr + 4 + axis_idx],
                  cumul_mat[joint_addr + 8 + axis_idx]
                );

                const float3 scaled_axis = axis_sign * joint_axis;

                // Linear velocity: translation axis direction
                jac_col[0] += scaled_axis.x;
                jac_col[1] += scaled_axis.y;
                jac_col[2] += scaled_axis.z;

                // Angular velocity: zero for prismatic joints (no change needed)
              }
            }

            // Write jacobian column to output without atomic operations
            // Each thread has exclusive access to its assigned joints
            // Layout: [6 x njoints] - 6 rows (linear + angular), njoints columns
            jacobian[0 * njoints + joint_idx] = jac_col[0];  // Linear X
            jacobian[1 * njoints + joint_idx] = jac_col[1];  // Linear Y
            jacobian[2 * njoints + joint_idx] = jac_col[2];  // Linear Z
            jacobian[3 * njoints + joint_idx] = jac_col[3];  // Angular X
            jacobian[4 * njoints + joint_idx] = jac_col[4];  // Angular Y
            jacobian[5 * njoints + joint_idx] = jac_col[5];  // Angular Z
          }
        }

        /**
         * Helper function to process robot spheres and write transformed spheres to output buffer.
         * This function handles the sphere transformation logic common to both fused kernels.
         *
         * @param b_robot_spheres Output buffer for transformed spheres
         * @param cumul_mat Cumulative transformation matrix buffer
         * @param robot_spheres Input robot spheres [num_envs x nspheres x 4]
         * @param linkSphereMap Mapping from sphere index to link index
         * @param env_query_idx Per-batch index into robot_spheres configs [batchSize]
         * @param batch_index Current batch index
         * @param nspheres Number of spheres
         * @param num_envs Number of sphere configurations (>1 enables per-batch indexing)
         * @param thread_index Current thread index within batch
         * @param threads_per_batch Number of threads per batch
         * @param matAddrBase Base address in cumulative matrix for current batch
         */
        __forceinline__ __device__ void transform_robot_spheres(
          float *b_robot_spheres,
          const float *cumul_mat,
          const float *robot_spheres,
          const int16_t *linkSphereMap,
          const int32_t *env_query_idx,
          const int batch_index,
          const int nspheres,
          const int num_envs,
          const int horizon,
          const int thread_index,
          const int threads_per_batch,
          const int matAddrBase
        ) {
          const int env_idx = (num_envs > 1) ? env_query_idx[batch_index / horizon] : 0;
          const int sphere_config_offset = env_idx * nspheres * 4;
          const int batchAddrs = batch_index * nspheres * 4;
          const int16_t spheres_perthread = (nspheres + threads_per_batch - 1) / threads_per_batch;

          for (int16_t i = 0; i < spheres_perthread; i++) {
            const int16_t sph_idx = i * threads_per_batch + thread_index;

            if (sph_idx >= nspheres) {
              break;
            }

            // read cumul idx:
            const int16_t read_cumul_idx = linkSphereMap[sph_idx];
            float4 spheres_mem = make_float4(0.0, 0.0, 0.0, 0.0);
            const int16_t sphAddrs = sph_idx * 4;

            transform_sphere_float4(&cumul_mat[matAddrBase + read_cumul_idx * 12],
                                   &robot_spheres[sphere_config_offset + sphAddrs], spheres_mem);

            *(float4 *)&b_robot_spheres[batchAddrs + sphAddrs] = spheres_mem;
          }
        }

      /**
       * Helper function to process stored links and write positions/quaternions to output buffers.
       * This function handles the link processing logic common to both fused kernels.
       *
       * @param link_pos Output buffer for link positions
       * @param link_quat Output buffer for link quaternions
       * @param cumul_mat Cumulative transformation matrix buffer
       * @param toolFrameMap Mapping from stored link index to link index
       * @param batch_index Current batch index
       * @param n_tool_frames Number of stored links
       * @param thread_index Current thread index within batch
       * @param threads_per_batch Number of threads per batch
       * @param matAddrBase Base address in cumulative matrix for current batch
       */
      __forceinline__ __device__ void write_link_poses(
        float *link_pos,
        float *link_quat,
        const float *cumul_mat,
        const int16_t *toolFrameMap,
        const int batch_index,
        const int n_tool_frames,
        const int thread_index,
        const int threads_per_batch,
        const int matAddrBase
      ) {
        const int16_t n_tool_frames_perthread = (n_tool_frames + threads_per_batch - 1) / threads_per_batch;

        for (int16_t i = 0; i < n_tool_frames_perthread; i++) {
          const int16_t tool_frame_idx = i * threads_per_batch + thread_index;

          if (tool_frame_idx >= n_tool_frames) {
            break;
          }

          const int16_t read_cumul_idx = toolFrameMap[tool_frame_idx];
          const int l_outAddrStart = batch_index * n_tool_frames;
          const int outAddrStart = matAddrBase + read_cumul_idx * 12;
          common::CuPose pose = common::CuPose::from_transform_matrix(&cumul_mat[outAddrStart]);

          // write quaternion out:
          *(float4 *)&link_quat[l_outAddrStart * 4 + tool_frame_idx * 4] = pose.get_quaternion_as_wxyz(); // need to write out as wxyz

          // write position out:
          *(float3 *)&link_pos[l_outAddrStart * 3 + tool_frame_idx * 3] = pose.position;
        }
      }

      __forceinline__ __device__ void store_local_transform_column(
        float *local_mat,
        const int col_idx,
        const float x,
        const float y,
        const float z
      ) {
        const int addr = col_idx * 3;
        local_mat[addr + 0] = x;
        local_mat[addr + 1] = y;
        local_mat[addr + 2] = z;
      }

      __forceinline__ __device__ void compute_local_link_transform(
        float *local_mat,
        const float *q,
        const float *fixedTransform,
        const int8_t *jointMapType,
        const int16_t *jointMap,
        const float *jointOffset,
        const int batch,
        const int link_idx,
        const int nlinks,
        const int njoints
      ) {
        const int ft_addr = link_idx * 12;
        const float f0 = fixedTransform[ft_addr + 0];
        const float f1 = fixedTransform[ft_addr + 1];
        const float f2 = fixedTransform[ft_addr + 2];
        const float f3 = fixedTransform[ft_addr + 3];
        const float f4 = fixedTransform[ft_addr + 4];
        const float f5 = fixedTransform[ft_addr + 5];
        const float f6 = fixedTransform[ft_addr + 6];
        const float f7 = fixedTransform[ft_addr + 7];
        const float f8 = fixedTransform[ft_addr + 8];
        const float f9 = fixedTransform[ft_addr + 9];
        const float f10 = fixedTransform[ft_addr + 10];
        const float f11 = fixedTransform[ft_addr + 11];

        const int j_type = jointMapType[link_idx];

        if (j_type == FIXED) {
          store_local_transform_column(local_mat, 0, f0, f4, f8);
          store_local_transform_column(local_mat, 1, f1, f5, f9);
          store_local_transform_column(local_mat, 2, f2, f6, f10);
          store_local_transform_column(local_mat, 3, f3, f7, f11);
          return;
        }

        float angle = q[batch * njoints + jointMap[link_idx]];
        const float2 angle_offset = *(const float2 *)&jointOffset[link_idx * 2];
        update_axis_direction(angle, j_type, angle_offset);

        if (j_type <= Z_PRISM) {
          store_local_transform_column(local_mat, 0, f0, f4, f8);
          store_local_transform_column(local_mat, 1, f1, f5, f9);
          store_local_transform_column(local_mat, 2, f2, f6, f10);
          store_local_transform_column(
            local_mat,
            3,
            f3 + (j_type == X_PRISM ? f0 : (j_type == Y_PRISM ? f1 : f2)) * angle,
            f7 + (j_type == X_PRISM ? f4 : (j_type == Y_PRISM ? f5 : f6)) * angle,
            f11 + (j_type == X_PRISM ? f8 : (j_type == Y_PRISM ? f9 : f10)) * angle
          );
          return;
        }

        float s = 0.0f;
        float c = 0.0f;
        sincosf(angle, &s, &c);

        const int xyz = j_type - X_ROT;
        const float is_x = xyz == 0 ? 1.0f : 0.0f;
        const float is_y = xyz == 1 ? 1.0f : 0.0f;
        const float is_z = xyz == 2 ? 1.0f : 0.0f;

        const float col0_scale = is_x + c * (is_y + is_z);
        const float col1_scale = is_y + c * (is_x + is_z);
        const float col2_scale = is_z + c * (is_x + is_y);

        local_mat[0] = f0 * col0_scale + s * (is_z * f1 - is_y * f2);
        local_mat[1] = f4 * col0_scale + s * (is_z * f5 - is_y * f6);
        local_mat[2] = f8 * col0_scale + s * (is_z * f9 - is_y * f10);
        local_mat[3] = f1 * col1_scale + s * (is_x * f2 - is_z * f0);
        local_mat[4] = f5 * col1_scale + s * (is_x * f6 - is_z * f4);
        local_mat[5] = f9 * col1_scale + s * (is_x * f10 - is_z * f8);
        local_mat[6] = f2 * col2_scale + s * (is_y * f0 - is_x * f1);
        local_mat[7] = f6 * col2_scale + s * (is_y * f4 - is_x * f5);
        local_mat[8] = f10 * col2_scale + s * (is_y * f8 - is_x * f9);
        store_local_transform_column(local_mat, 3, f3, f7, f11);
      }

      template<int N_LINKS>
      __forceinline__ __device__ void compute_local_link_transforms(
        float *local_mat,
        const float *q,
        const float *fixedTransform,
        const int8_t *jointMapType,
        const int16_t *jointMap,
        const float *jointOffset,
        const int start_batch_index,
        const int batchSize,
        const int nlinks,
        const int njoints,
        const int num_batches_per_block
      ) {
        const int remaining_batches = batchSize - start_batch_index;
        const int actual_batches_per_block = min(num_batches_per_block, remaining_batches);
        const int link_count = (N_LINKS < 0) ? nlinks : N_LINKS;
        const int total_links = actual_batches_per_block * link_count;

        for (int element_idx = threadIdx.x; element_idx < total_links; element_idx += blockDim.x) {
          const int local_batch = element_idx / link_count;
          const int link_idx = element_idx - local_batch * link_count;
          const int batch = start_batch_index + local_batch;
          const int local_offset = local_batch * nlinks * 12 + link_idx * 12;

          compute_local_link_transform(
            &local_mat[local_offset],
            q,
            fixedTransform,
            jointMapType,
            jointMap,
            jointOffset,
            batch,
            link_idx,
            nlinks,
            njoints
          );
        }

        __syncthreads();
      }

      __forceinline__ __device__ void compose_link_transform_halfwarp(
        float *cumul_mat,
        const float *local_mat,
        const int16_t *linkMap,
        const int link_idx,
        const int matAddrBase,
        const int localAddrBase,
        const int lane_idx,
        const int stride = 4
      ) {
        if (lane_idx >= 12) {
          return;
        }

        const int row_idx = lane_idx / 4;
        const int col_idx = lane_idx - row_idx * 4;
        const int inAddrStart = matAddrBase + linkMap[link_idx] * (stride * 3);
        const int outAddrStart = matAddrBase + link_idx * (stride * 3);
        const float *local_col = &local_mat[localAddrBase + link_idx * 12 + col_idx * 3];
        const float4 jm_vec = make_float4(
          local_col[0],
          local_col[1],
          local_col[2],
          col_idx == 3 ? 1.0f : 0.0f
        );

        cumul_mat[outAddrStart + row_idx * stride + col_idx] =
          dot(*(float4 *)&cumul_mat[inAddrStart + row_idx * stride], jm_vec);
      }

      __forceinline__ __device__ void load_base_transform_halfwarp(
        float *cumul_mat,
        const float *fixedTransform,
        const int nlinks,
        const int matAddrBase,
        const int lane_idx,
        const unsigned int warp_mask
      ) {
        (void)nlinks;
        if (lane_idx < 3) {
          const int fixed_row = lane_idx * 4;
          const int out_row = matAddrBase + lane_idx * 4;
          cumul_mat[out_row + 0] = fixedTransform[fixed_row + 0];
          cumul_mat[out_row + 1] = fixedTransform[fixed_row + 1];
          cumul_mat[out_row + 2] = fixedTransform[fixed_row + 2];
          cumul_mat[out_row + 3] = fixedTransform[fixed_row + 3];
        }
        __syncwarp(warp_mask);
      }

      template<int N_LINKS>
      __forceinline__ __device__ void compute_cumulative_transforms_halfwarp(
        float *cumul_mat,
        const float *local_mat,
        const int16_t *linkMap,
        const int nlinks,
        const int matAddrBase,
        const int localAddrBase,
        const int lane_idx,
        const unsigned int warp_mask
      ) {
        if constexpr (N_LINKS < 0) {
          for (int l = 1; l < nlinks; l++) {
            compose_link_transform_halfwarp(
              cumul_mat, local_mat, linkMap, l, matAddrBase, localAddrBase, lane_idx);
            __syncwarp(warp_mask);
          }
        } else {
        #pragma unroll 10
          for (int l = 1; l < N_LINKS; l++) {
            compose_link_transform_halfwarp(
              cumul_mat, local_mat, linkMap, l, matAddrBase, localAddrBase, lane_idx);
            __syncwarp(warp_mask);
          }
        }
      }

      __forceinline__ __device__ void write_cumulative_transforms_active(
        float *global_cumul_mat,
        const float *cumul_mat,
        const int start_batch_index,
        const int nlinks,
        const int actual_batches_per_block,
        const int active_threads
      ) {
        const int vec4_per_batch = nlinks * 3;
        const int total_vec4 = actual_batches_per_block * vec4_per_batch;

        for (int vec_idx = threadIdx.x; vec_idx < total_vec4; vec_idx += active_threads) {
          const int local_batch = vec_idx / vec4_per_batch;
          const int local_vec_idx = vec_idx - local_batch * vec4_per_batch;
          const int link_idx = local_vec_idx / 3;
          const int row_idx = local_vec_idx - link_idx * 3;
          const int shared_offset = local_batch * nlinks * 12 + link_idx * 12 + row_idx * 4;
          const int global_offset =
            (start_batch_index + local_batch) * nlinks * 12 + link_idx * 12 + row_idx * 4;

          *(float4 *)&global_cumul_mat[global_offset] = *(float4 *)&cumul_mat[shared_offset];
        }
      }

      template<int TILE_WIDTH>
      __forceinline__ __device__ void write_center_of_mass_tile(
        float *batch_center_of_mass,
        const float *cumul_mat,
        const float *link_masses_com,
        const int batch_index,
        const int n_links,
        const int thread_in_tile,
        const unsigned int tile_mask,
        const int matAddrBase
      ) {
        if (thread_in_tile < 0 || thread_in_tile >= TILE_WIDTH) {
          return;
        }

        float4 local_weighted_com = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int16_t link_idx = thread_in_tile; link_idx < n_links; link_idx += TILE_WIDTH) {
          const int link_addr = link_idx * 4;
          const float4 mass_com = make_float4(
            link_masses_com[link_addr + 0],
            link_masses_com[link_addr + 1],
            link_masses_com[link_addr + 2],
            link_masses_com[link_addr + 3]);
          const float mass = mass_com.w;

          if (mass > 0.0f) {
            float4 com_world;
            transform_sphere_float4(&cumul_mat[matAddrBase + link_idx * 12], mass_com, com_world);
            local_weighted_com.x += mass * com_world.x;
            local_weighted_com.y += mass * com_world.y;
            local_weighted_com.z += mass * com_world.z;
            local_weighted_com.w += mass;
          }
        }

        #pragma unroll
        for (int offset = TILE_WIDTH >> 1; offset > 0; offset >>= 1) {
          local_weighted_com.x +=
            __shfl_down_sync(tile_mask, local_weighted_com.x, offset, TILE_WIDTH);
          local_weighted_com.y +=
            __shfl_down_sync(tile_mask, local_weighted_com.y, offset, TILE_WIDTH);
          local_weighted_com.z +=
            __shfl_down_sync(tile_mask, local_weighted_com.z, offset, TILE_WIDTH);
          local_weighted_com.w +=
            __shfl_down_sync(tile_mask, local_weighted_com.w, offset, TILE_WIDTH);
        }

        if (thread_in_tile == 0) {
          const int batch_addr = batch_index * 4;
          if (local_weighted_com.w > 0.0f) {
            batch_center_of_mass[batch_addr + 0] = local_weighted_com.x / local_weighted_com.w;
            batch_center_of_mass[batch_addr + 1] = local_weighted_com.y / local_weighted_com.w;
            batch_center_of_mass[batch_addr + 2] = local_weighted_com.z / local_weighted_com.w;
            batch_center_of_mass[batch_addr + 3] = local_weighted_com.w;
          } else {
            batch_center_of_mass[batch_addr + 0] = 0.0f;
            batch_center_of_mass[batch_addr + 1] = 0.0f;
            batch_center_of_mass[batch_addr + 2] = 0.0f;
            batch_center_of_mass[batch_addr + 3] = 0.0f;
          }
        }
      }
    }
}
