// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace curobo::robot::detail {

// Device-resident pose inputs and outputs used between grasp planning phases.
struct GraspPosePreparation {
  const float* positions;
  const float* quaternions;
  const int32_t* goal_indices;
  int selected_seed;
  int num_tools;
  int num_grasps;
  int approach_axis;
  int lift_axis;
  float approach_offset;
  float lift_offset;
  bool approach_in_tool_frame;
  bool lift_in_tool_frame;
  float* selected_positions;
  float* selected_quaternions;
  float* approach_positions;
  float* approach_quaternions;
  float* lift_positions;
  float* lift_quaternions;
  int32_t* selected_index;
  uint8_t* valid;
};

cudaError_t launch_prepare_grasp_poses(const GraspPosePreparation& request,
                                     cudaStream_t stream);

cudaError_t launch_fit_grasp_attachment(const float* points, int count, const float* position,
                                        const float* quaternion, float voxel_radius,
                                        const int32_t* slots, float* spheres, uint8_t* valid,
                                        cudaStream_t stream);

cudaError_t launch_set_sphere_collision_enabled(
    float* spheres, const float* nominal_spheres, const int32_t* slots,
    int num_slots, bool enabled, cudaStream_t stream);

cudaError_t launch_set_sphere_collision_mask(uint32_t* masks, const int32_t* slots, int num_slots,
                                             uint32_t layers, cudaStream_t stream);

cudaError_t launch_motion_result_metadata(
    const int32_t* trajopt_selected_seed, const uint8_t* rrt_success,
    const int32_t* rrt_iterations, int num_seeds, bool rrt_stage,
    int32_t* selected_seed, uint8_t* used_rrt,
    int32_t* selected_rrt_iterations, cudaStream_t stream);

cudaError_t launch_pose_result_metadata(
    const int32_t* selected_seed, const float* ik_solutions,
    const float* ik_position_error, const float* ik_orientation_error,
    int num_seeds, int num_dof, int planning_attempts,
    float* selected_goal_position, float* position_error,
    float* orientation_error, int32_t* output_selected_seed,
    int32_t* output_planning_attempts, uint8_t* success,
    cudaStream_t stream);

}  // namespace curobo::robot::detail
