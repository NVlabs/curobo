/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kinematics_util.cuh"

namespace curobo {
namespace kinematics {

// Contract the Jacobian cotangent directly into joint gradients. The topology-only
// inputs are precomputed once in KinematicsParams:
//   toolJointLinks*: physical contributors for each (tool, output-joint) pair
//   jointLinkInfluenceWords: descendant-link bitset for each physical joint link
// This removes all runtime chain searches and sums every mimic-joint contributor.
template<typename AccumulatorType, bool Accumulate = false>
__forceinline__ __device__ void compute_jacobian_derivatives(
  AccumulatorType *grad_joint,
  const float *grad_jacobian,
  const float *cumul_mat,
  const int8_t *jointMapType,
  const int16_t *toolJointLinksData,
  const int16_t *toolJointLinksOffsets,
  const uint32_t *jointLinkInfluenceWords,
  const float *jointOffset,
  const int16_t *toolFrameMap,
  const int batch_index,
  const int njoints,
  const int nlinks,
  const int n_tool_frames,
  const int matAddrBase,
  const int thread_index,
  const int threads_per_batch)
{
  const int influence_word_count = (nlinks + 31) / 32;

  for (int joint_idx = thread_index; joint_idx < njoints;
       joint_idx += threads_per_batch) {
    AccumulatorType joint_grad_sum = static_cast<AccumulatorType>(0.0f);

    for (int tool_idx = 0; tool_idx < n_tool_frames; ++tool_idx) {
      const int joint_pair_idx = tool_idx * njoints + joint_idx;
      const int derivative_link_start = toolJointLinksOffsets[joint_pair_idx];
      const int derivative_link_end = toolJointLinksOffsets[joint_pair_idx + 1];
      if (derivative_link_start == derivative_link_end) {
        continue;
      }

      const int tool_link = toolFrameMap[tool_idx];
      const int tool_addr = matAddrBase + tool_link * 12;
      const float3 tool_position = make_float3(
        cumul_mat[tool_addr + 3],
        cumul_mat[tool_addr + 7],
        cumul_mat[tool_addr + 11]);
      const int jacobian_offset =
        (batch_index * n_tool_frames + tool_idx) * 6 * njoints;

      // A generalized coordinate can drive multiple physical links through mimic
      // joints. Each physical derivative link contributes independently.
      for (int derivative_link_idx = derivative_link_start;
           derivative_link_idx < derivative_link_end;
           ++derivative_link_idx) {
        const int derivative_link = toolJointLinksData[derivative_link_idx];
        const int derivative_addr = matAddrBase + derivative_link * 12;
        const int derivative_type = jointMapType[derivative_link];
        const float derivative_scale = jointOffset[derivative_link * 2];
        const bool derivative_is_revolute =
          derivative_type >= X_ROT && derivative_type <= Z_ROT;
        const bool derivative_is_prismatic =
          derivative_type >= X_PRISM && derivative_type <= Z_PRISM;
        if (!derivative_is_revolute && !derivative_is_prismatic) {
          continue;
        }

        const int derivative_axis_idx = derivative_is_revolute
          ? derivative_type - X_ROT
          : derivative_type - X_PRISM;
        const float3 derivative_axis = derivative_scale * make_float3(
          cumul_mat[derivative_addr + derivative_axis_idx],
          cumul_mat[derivative_addr + 4 + derivative_axis_idx],
          cumul_mat[derivative_addr + 8 + derivative_axis_idx]);
        const float3 derivative_position = make_float3(
          cumul_mat[derivative_addr + 3],
          cumul_mat[derivative_addr + 7],
          cumul_mat[derivative_addr + 11]);
        const int influence_offset = derivative_link * influence_word_count;

        for (int column_idx = 0; column_idx < njoints; ++column_idx) {
          const float3 grad_linear = make_float3(
            grad_jacobian[jacobian_offset + column_idx],
            grad_jacobian[jacobian_offset + njoints + column_idx],
            grad_jacobian[jacobian_offset + 2 * njoints + column_idx]);
          const float3 grad_angular = make_float3(
            grad_jacobian[jacobian_offset + 3 * njoints + column_idx],
            grad_jacobian[jacobian_offset + 4 * njoints + column_idx],
            grad_jacobian[jacobian_offset + 5 * njoints + column_idx]);
          if (grad_linear.x == 0.0f && grad_linear.y == 0.0f &&
              grad_linear.z == 0.0f && grad_angular.x == 0.0f &&
              grad_angular.y == 0.0f && grad_angular.z == 0.0f) {
            continue;
          }

          const int column_pair_idx = tool_idx * njoints + column_idx;
          const int column_link_start = toolJointLinksOffsets[column_pair_idx];
          const int column_link_end = toolJointLinksOffsets[column_pair_idx + 1];
          for (int column_link_idx = column_link_start;
               column_link_idx < column_link_end;
               ++column_link_idx) {
            const int column_link = toolJointLinksData[column_link_idx];
            const int column_addr = matAddrBase + column_link * 12;
            const int column_type = jointMapType[column_link];
            const float column_scale = jointOffset[column_link * 2];
            const bool derivative_strictly_affects_column =
              derivative_link != column_link &&
              (jointLinkInfluenceWords[
                 influence_offset + column_link / 32] &
               (uint32_t(1) << (column_link & 31))) != 0;

            if (column_type >= X_ROT && column_type <= Z_ROT) {
              const int column_axis_idx = column_type - X_ROT;
              const float3 column_axis = column_scale * make_float3(
                cumul_mat[column_addr + column_axis_idx],
                cumul_mat[column_addr + 4 + column_axis_idx],
                cumul_mat[column_addr + 8 + column_axis_idx]);
              const float3 column_position = make_float3(
                cumul_mat[column_addr + 3],
                cumul_mat[column_addr + 7],
                cumul_mat[column_addr + 11]);
              float3 derivative_linear = make_float3(0.0f, 0.0f, 0.0f);
              float3 derivative_angular = make_float3(0.0f, 0.0f, 0.0f);

              if (derivative_is_revolute) {
                if (derivative_strictly_affects_column) {
                  derivative_angular = cross(derivative_axis, column_axis);
                  derivative_linear =
                    cross(derivative_angular, tool_position - column_position);
                }
                derivative_linear += cross(
                  column_axis,
                  cross(derivative_axis, tool_position - derivative_position));
                if (derivative_strictly_affects_column) {
                  derivative_linear -= cross(
                    column_axis,
                    cross(derivative_axis, column_position - derivative_position));
                }
              } else if (!derivative_strictly_affects_column) {
                derivative_linear = cross(column_axis, derivative_axis);
              }

              joint_grad_sum += dot(grad_linear, derivative_linear);
              joint_grad_sum += dot(grad_angular, derivative_angular);
            } else if (
              column_type >= X_PRISM && column_type <= Z_PRISM &&
              derivative_is_revolute && derivative_strictly_affects_column) {
              const int column_axis_idx = column_type - X_PRISM;
              const float3 column_axis = column_scale * make_float3(
                cumul_mat[column_addr + column_axis_idx],
                cumul_mat[column_addr + 4 + column_axis_idx],
                cumul_mat[column_addr + 8 + column_axis_idx]);
              joint_grad_sum += dot(
                grad_linear, cross(derivative_axis, column_axis));
            }
          }
        }
      }
    }

    if constexpr (Accumulate) {
      grad_joint[joint_idx] += joint_grad_sum;
    } else {
      grad_joint[joint_idx] = joint_grad_sum;
    }
  }
}

}  // namespace kinematics
}  // namespace curobo
