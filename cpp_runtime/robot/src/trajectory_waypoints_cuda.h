// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>
#include "curobo_robot_runtime/spline_layout.h"

namespace curobo::robot::detail {

// Stable device layout: mode (off/line/waypoints), orientation enabled,
// path position/angle, final position/angle, free axis xyz, axis in goal frame.
inline constexpr int TOOL_PATH_SETTINGS = 10;

enum class ToolPathMode { DISABLED = 0, LINE = 1, WAYPOINTS = 2 };
enum class OptimizationCondition { INITIAL_FAILED, INITIAL_SUCCEEDED, REFINEMENT_FAILED };

cudaError_t launch_tool_path_settings(float *settings, ToolPathMode mode, bool orientation,
                                      float path_position, float path_orientation,
                                      float final_position, float final_orientation,
                                      float direction_x, float direction_y, float direction_z,
                                      bool goal_frame, cudaStream_t stream);

// Inputs exclude the start, except count==0, which broadcasts a terminal pose.
cudaError_t launch_pose_waypoints(const float *positions, const float *quaternions, int count,
                                  const float *start_positions, const float *start_quaternions,
                                  int tools, SplineLayout layout, float *reference_positions,
                                  float *reference_quaternions, cudaStream_t stream);

cudaError_t launch_reference_progress(const int32_t *counts, int seeds, int samples,
                                      SplineLayout layout, float *progress, cudaStream_t stream);

cudaError_t launch_validate_tool_path(const float *positions, const float *quaternions,
                                      const float *reference_positions,
                                      const float *reference_quaternions, const float *progress,
                                      const int32_t *counts, const float *settings, int seeds,
                                      int samples, int tools, SplineLayout layout,
                                      int32_t *violation, cudaStream_t stream);

cudaError_t launch_moving_reference_margin(float *dt, const float *velocity,
                                           const float *acceleration, int seeds, int dof,
                                           float factor, float maximum_dt, cudaStream_t stream);

cudaError_t launch_initial_iteration_count(int32_t *count, int increment, cudaStream_t stream);
cudaError_t launch_optimization_condition(cudaGraphConditionalHandle handle,
                                          const int32_t *native_success,
                                          const uint8_t *result_success,
                                          OptimizationCondition condition, cudaStream_t stream);

}  // namespace curobo::robot::detail
