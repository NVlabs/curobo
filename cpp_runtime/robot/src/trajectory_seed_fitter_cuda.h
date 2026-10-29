// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

namespace curobo::robot::detail {

cudaError_t launch_seed_fitter_resample_dense(
    const float* target_path, const float* arc_progress, float* arc_target,
    int seeds, int steps, int dof, cudaStream_t stream);

cudaError_t launch_seed_fitter_initialize_fit(
    const float* arc_target, const float* boundary_position,
    const float* start_position, const float* start_velocity,
    const float* start_acceleration, const float* start_jerk,
    const float* goal_position, const float* fit_matrix,
    const float* start_position_coefficient,
    const float* start_velocity_coefficient,
    const float* start_acceleration_coefficient,
    const float* start_jerk_coefficient,
    const float* goal_position_coefficient, float* position_residual,
    float* knots, int seeds, int steps, int dof, int knot_count,
    cudaStream_t stream);

cudaError_t launch_seed_fitter_required_dt(
    const float* velocity, const float* acceleration, const float* jerk,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, int seeds, int steps, int dof,
    float trajectory_dt, float dt_epsilon, float* required_dt,
    cudaStream_t stream);

cudaError_t launch_seed_fitter_derivative_scale(
    const float* current_velocity, const float* current_acceleration,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, const float* basis_velocity,
    const float* basis_acceleration, const float* basis_jerk, int seeds,
    int steps, int dof, int knots, float path_normal_trace,
    float timing_regularization, float irls_epsilon,
    float* velocity_trace_workspace, float* acceleration_trace_workspace,
    float* jerk_trace_workspace, float* derivative_scale,
    cudaStream_t stream);

cudaError_t launch_seed_fitter_solve_irls(
    const float* position_residual, const float* boundary_velocity,
    const float* boundary_acceleration, const float* boundary_jerk,
    const float* current_velocity, const float* current_acceleration,
    const float* current_knots, const float* goal_position,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, const float* basis_position,
    const float* basis_velocity, const float* basis_acceleration,
    const float* basis_jerk, const float* path_normal,
    const float* derivative_scale, int seeds, int steps, int dof, int knots,
    float regularization, float trust_region_regularization,
    float path_diagonal_scale, float irls_epsilon, float* candidate_knots,
    cudaStream_t stream);

cudaError_t launch_seed_fitter_accept_candidate(
    const float* candidate_knots, const float* candidate_velocity,
    const float* candidate_acceleration, const float* candidate_jerk,
    const float* candidate_dt, float* current_knots,
    float* current_velocity, float* current_acceleration,
    float* current_jerk, float* current_dt, int seeds, int steps, int dof,
    int knot_count, cudaStream_t stream);

}  // namespace curobo::robot::detail
