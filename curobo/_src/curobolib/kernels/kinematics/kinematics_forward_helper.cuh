/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "kinematics_constants.h"

#include "third_party/helper_math.h"
#include "common/math.cuh"
#include "common/quaternion_util.cuh"
#include "common/pose_util.cuh"
#include "kinematics_joint_util.cuh"
#include "kinematics_util.cuh"

namespace curobo{
    namespace kinematics{

      // Helper function to initialize base link transform
      __forceinline__ __device__ void initialize_base_link_transform(
        float *cumul_mat,
        const float *fixedTransform,
        const int matAddrBase,
        const int col_idx,
        const int stride = 4  // Default to M (4) for compatibility
      ) {
        if (col_idx < 3) {
          *(float4 *)&cumul_mat[matAddrBase + col_idx * stride] =
            *(float4 *)&fixedTransform[col_idx * stride];
        }
        //__syncthreads();
        __syncwarp((0xF << ((threadIdx.x / 4) * 4)) & 0xFFFFFFFF);
      }

      // Helper function to apply joint transformation to cumulative matrix
      __forceinline__ __device__ void apply_joint_transform(
        float *cumul_mat,
        const float *JM,
        const int16_t *linkMap,
        const int link_idx,
        const int matAddrBase,
        const int col_idx,
        const int stride = 4  // Default stride for matrix rows
      ) {
        const int inAddrStart = matAddrBase + linkMap[link_idx] * (stride * 3);
        const int outAddrStart = matAddrBase + link_idx * (stride * 3);

        float4 jm_vec = *(float4 *)JM;

        #pragma unroll 3
        for (int i = 0; i < 3; i++) {
          cumul_mat[outAddrStart + (i * stride) + col_idx] =
            dot(*(float4 *)&cumul_mat[inAddrStart + (i * stride)], jm_vec);
        }
      }

      /**
       * Helper function to compute joint transformations for all links.
       * This function computes the cumulative transformation matrix for each link
       * by applying the appropriate joint transformation based on joint type.
       *
       * @param cumul_mat Cumulative transformation matrix buffer
       * @param q Joint angles array
       * @param fixedTransform Fixed transformation matrices for each link. Shape is nlinks x 3 x 4.
       * @param jointMapType Joint type mapping (FIXED, X_PRISM, Y_PRISM, Z_PRISM, X_ROT, Y_ROT, Z_ROT)
       * @param jointMap Joint index mapping
       * @param linkMap Link parent mapping
       * @param jointOffset Joint offset values
       * @param batch Current batch index
       * @param njoints Number of joints
       * @param nlinks Number of links
       * @param matAddrBase Base address in cumulative matrix
       * @param col_idx Column index (0-3) for 4x4 matrix operations
       */
       template<int N_LINKS=-1>
      __forceinline__ __device__ void compute_joint_transformations(
        float *cumul_mat,
        const float *q,
        const float *fixedTransform,
        const int8_t *jointMapType,
        const int16_t *jointMap,
        const int16_t *linkMap,
        const float *jointOffset,
        const int batch,
        const int njoints,
        const int nlinks,
        const int matAddrBase,
        const int col_idx
      ) {

        if constexpr (N_LINKS < 0) {
        for (int8_t l = 1; l < nlinks; l++) {
          // Get one row of fixedTransform
          int ftAddrStart  = l * 12;
          // int inAddrStart  = matAddrBase + linkMap[l] * 12;
          //int outAddrStart = matAddrBase + l * 12;

          // Check joint type and use one of the helper functions:
          float __align__(16) JM[4];
          const int   j_type = jointMapType[l];

          if (j_type == FIXED) {
            fixed_joint_fn(&fixedTransform[ftAddrStart + col_idx], col_idx, &JM[0]);
          } else {
            float angle = q[batch * njoints + jointMap[l]];
            float2 angle_offset = *(float2 *)&jointOffset[l*2];
            update_axis_direction(angle, j_type, angle_offset);

            if (j_type <= Z_PRISM) {
              prism_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0], j_type);
            } else {
              xyz_rot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0], j_type - X_ROT);
            }
          }

          // Apply the transformation
          apply_joint_transform(cumul_mat, JM, linkMap, l, matAddrBase, col_idx);
        }
        }
        else {
          #pragma unroll 10
          for (int l = 1; l < N_LINKS; l++) {
            // Get one row of fixedTransform
            const int ftAddrStart  = l * 12;
            // int inAddrStart  = matAddrBase + linkMap[l] * 12;
            //int outAddrStart = matAddrBase + l * 12;

            // Check joint type and use one of the helper functions:
            float __align__(16) JM[4];
            const int j_type = jointMapType[l];

            if (j_type == FIXED) {
              fixed_joint_fn(&fixedTransform[ftAddrStart + col_idx], col_idx, &JM[0]);
            } else {
              float angle = q[batch * njoints + jointMap[l]];
              float2 angle_offset = *(float2 *)&jointOffset[l*2];
              update_axis_direction(angle, j_type, angle_offset);

              if (j_type <= Z_PRISM) {
                prism_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0], j_type);
              } else {
                xyz_rot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0], j_type - X_ROT);
              }
            }

            // Apply the transformation
            apply_joint_transform(cumul_mat, JM, linkMap, l, matAddrBase, col_idx);
          }
        }
      }

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
        __forceinline__ __device__ void compute_kinematic_jacobian(
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
        __forceinline__ __device__ void process_robot_spheres(
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
      __forceinline__ __device__ void process_stored_links(
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

      /**
       * Helper function to compute center of mass in the forward pass.
       * Uses dedicated shared memory for reduction (no reuse of existing shared memory).
       * Supports multiple batches per thread block using general threading model.
       *
       * @param batch_center_of_mass Output buffer for center of mass [batch_size] - xyz=global CoM, w=total mass
       * @param cumul_mat Cumulative transformation matrix buffer
       * @param link_masses_com Input buffer [n_links] - xyz=local CoM, w=mass
       * @param batch_index Current batch index
       * @param n_links Number of links
       * @param thread_in_batch Thread index within this batch (0 to threads_per_batch-1)
       * @param threads_per_batch Number of threads processing this batch
       * @param local_batch_start Starting index in shared memory for this batch
       * @param matAddrBase Base address in cumulative matrix for current batch
       */
      __forceinline__ __device__ void process_center_of_mass(
        float *batch_center_of_mass,      // [batch_size * 4] - xyz=global CoM, w=total mass
        const float *cumul_mat,
        const float *link_masses_com,     // [n_links * 4] - xyz=local CoM, w=mass
        float4 *shared_com_data,           // Shared memory for reduction - passed from kernel
        const int batch_index,
        const int n_links,
        const int thread_in_batch,
        const int threads_per_batch,
        const int local_batch_start,
        const int matAddrBase
      ) {
        // Shared memory for reduction across threads within this batch
        // Note: Caller must ensure shared memory is large enough for all batches in block

        const int16_t links_per_thread = (n_links + threads_per_batch - 1) / threads_per_batch;

        // Thread-local accumulation
        float4 local_weighted_com = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Each thread processes a subset of links
        for (int16_t i = 0; i < links_per_thread; i++) {
          const int16_t link_idx = i * threads_per_batch + thread_in_batch;

          if (link_idx >= n_links) {
            break;
          }

          // Load link mass and local center of mass from float array (4 floats per link)
          const int link_addr = link_idx * 4;
          float4 mass_com = make_float4(
            link_masses_com[link_addr + 0],  // x - local CoM x
            link_masses_com[link_addr + 1],  // y - local CoM y
            link_masses_com[link_addr + 2],  // z - local CoM z
            link_masses_com[link_addr + 3]   // w - mass
          );
          float mass = mass_com.w;

          if (mass <= 0.0f) {
            continue;  // Skip massless links
          }

          // Transform local center of mass to world coordinates
          // Reuse existing transform_sphere_float4 function!
          float4 com_world;
          transform_sphere_float4(&cumul_mat[matAddrBase + link_idx * 12], &mass_com, com_world);

          // Accumulate mass-weighted position and total mass
          local_weighted_com.x += mass * com_world.x;
          local_weighted_com.y += mass * com_world.y;
          local_weighted_com.z += mass * com_world.z;
          local_weighted_com.w += mass;  // Total mass accumulation
        }

        // Store thread results in shared memory with proper offset
        shared_com_data[local_batch_start + thread_in_batch] = local_weighted_com;
        __syncthreads();

        // Reduction across threads (only thread 0 of this batch does the final computation)
        if (thread_in_batch == 0) {
          float4 total_weighted_com = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

          // Sum contributions from all threads in this batch
          for (int t = 0; t < threads_per_batch; t++) {
            float4 thread_contribution = shared_com_data[local_batch_start + t];
            total_weighted_com.x += thread_contribution.x;
            total_weighted_com.y += thread_contribution.y;
            total_weighted_com.z += thread_contribution.z;
            total_weighted_com.w += thread_contribution.w;
          }

          // Compute final center of mass and store result as 4 consecutive floats
          const int batch_addr = batch_index * 4;
          if (total_weighted_com.w > 0.0f) {
            batch_center_of_mass[batch_addr + 0] = total_weighted_com.x / total_weighted_com.w;  // CoM x
            batch_center_of_mass[batch_addr + 1] = total_weighted_com.y / total_weighted_com.w;  // CoM y
            batch_center_of_mass[batch_addr + 2] = total_weighted_com.z / total_weighted_com.w;  // CoM z
            batch_center_of_mass[batch_addr + 3] = total_weighted_com.w;                        // Total mass
          } else {
            // Handle edge case of massless robot
            batch_center_of_mass[batch_addr + 0] = 0.0f;
            batch_center_of_mass[batch_addr + 1] = 0.0f;
            batch_center_of_mass[batch_addr + 2] = 0.0f;
            batch_center_of_mass[batch_addr + 3] = 0.0f;
          }
        }
      }
    }
}