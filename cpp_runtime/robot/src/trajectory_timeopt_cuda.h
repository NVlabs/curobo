// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

#include "curobo_robot_runtime/trajectory_timeopt.h"

namespace curobo::robot::detail {

inline constexpr int TIMEOPT_CONTROL_COUNT = 32;
cudaError_t launch_timeopt_scale(TrajectoryTimeOptSamples samples,
    const float* scale, size_t values, float* output, cudaStream_t stream);

/** Fixed addresses captured by the timeopt; storage belongs to its runtime. */
struct TimeOptWorkspace {
  int batch{0}, dof{0}, knots{0}, points{0}, intervals{0}, tiles{0}, seed_tiles{0};
  int start_offset{0}, iterations{0}, samples{0};
  int *point_first{nullptr}, *meta{nullptr}, *offsets{nullptr};
  int *starts{nullptr}, *time_first{nullptr}, *segments{nullptr};
  float *basis{nullptr}, *time_basis{nullptr};
  float *time_weights{nullptr}, *local{nullptr}, *polynomial{nullptr}, *nodes{nullptr};
  float *vlim{nullptr}, *alim{nullptr}, *jlim{nullptr}, *lower{nullptr}, *upper{nullptr};
  float *input_knots{nullptr}, *input_start{nullptr}, *input_goal{nullptr}, *input_dt{nullptr};
  float *input_velocity{nullptr}, *input_acceleration{nullptr}, *input_jerk{nullptr};
  float *controls{nullptr}, *start{nullptr}, *goal{nullptr}, *reference_speed{nullptr};
  bool *input_valid{nullptr}, *stationary{nullptr};
  float *path{nullptr}, *q{nullptr}, *qd{nullptr}, *qdd{nullptr}, *tau{nullptr};
  float *coefficients{nullptr}, *seed_bounds{nullptr}, *output_bounds{nullptr};
  float *rows{nullptr}, *bounds{nullptr}, *jacobian{nullptr}, *row_state{nullptr};
  float *partials{nullptr}, *corrector_terms{nullptr}, *step_bounds{nullptr}, *state{nullptr};
  int *candidate_masks{nullptr};
  float *timing{nullptr}, *uniform{nullptr};
  bool *feasible{nullptr}, *numerical{nullptr};
  float *output_timing{nullptr}, *output_uniform{nullptr}, *output_joints{nullptr};
  float *output_torque{nullptr}, *output_time{nullptr}, *duration{nullptr}, *dt{nullptr};
  bool *output_numerical{nullptr}, *success{nullptr};
  int *status{nullptr};
  float *sample_times{nullptr}, *sample_joints{nullptr};
};

cudaError_t launch_timeopt_path(const TimeOptWorkspace &w, cudaStream_t stream);
cudaError_t launch_timeopt_solve(const TimeOptWorkspace &w, cudaStream_t stream);
cudaError_t launch_timeopt_sample(const TimeOptWorkspace &w, cudaStream_t stream);

cudaError_t launch_timeopt_boundaries(
    const float* position, const float* dt, const float* start_velocity,
    const float* start_acceleration, int batch, int dof, int steps,
    int interpolation_steps, float* boundaries, float* knot_dt, cudaStream_t stream);
cudaError_t launch_timeopt_spatial_output(
    const float* controls, const float* boundaries, const float* knot_dt,
    const int32_t* selected_seed, const uint8_t* success, int batch, int knots,
    int dof, float* output, cudaStream_t stream);
cudaError_t launch_timeopt_times(const float* duration, int batch, int capacity,
                                 float interpolation_dt, float* times, float* dt,
                                 int32_t* counts, cudaStream_t stream);
cudaError_t launch_timeopt_rank(
    const uint8_t* timeopt_success, const int32_t* violation,
    const int32_t* counts, const uint8_t* seed_valid, const uint8_t* pose_valid,
    const float* duration, int batch, uint8_t* success, float* rank,
    cudaStream_t stream);
cudaError_t launch_timeopt_output(
    TrajectoryTimeOptSamples samples, const float* timing_controls,
    const float* dt, const float* duration,
    const int32_t* counts, const int32_t* selected_seed, const uint8_t* success,
    int capacity, int dof, float* states, float* metadata, int32_t* count,
    float* selected_timing_controls, cudaStream_t stream);
}  // namespace curobo::robot::detail
