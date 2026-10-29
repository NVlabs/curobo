// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "trajectory_seed_fitter_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

#include "trajectory/seed_fitter/seed_fitter_kernel.cuh"

namespace curobo::robot::detail {

namespace {

int grid_for(size_t elements) {
  return static_cast<int>((elements +
                           curobo::trajectory::seed_fitter::THREADS_PER_BLOCK -
                           1) /
                          curobo::trajectory::seed_fitter::THREADS_PER_BLOCK);
}

}  // namespace

cudaError_t launch_seed_fitter_resample_dense(
    const float* target_path, const float* arc_progress, float* arc_target,
    int seeds, int steps, int dof, cudaStream_t stream) {
  using curobo::trajectory::seed_fitter::THREADS_PER_BLOCK;
  curobo::trajectory::seed_fitter::resample_dense_kernel
      <<<seeds, THREADS_PER_BLOCK, 2 * steps * sizeof(float),
         stream>>>(target_path, arc_progress, arc_target, seeds, steps, dof);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  using curobo::trajectory::seed_fitter::THREADS_PER_BLOCK;
  const size_t elements =
      static_cast<size_t>(seeds) * knot_count * dof;
  curobo::trajectory::seed_fitter::initialize_fit_kernel
      <<<grid_for(elements), THREADS_PER_BLOCK, 0, stream>>>(
          arc_target, boundary_position, start_position, start_velocity,
          start_acceleration, start_jerk, goal_position, fit_matrix,
          start_position_coefficient, start_velocity_coefficient,
          start_acceleration_coefficient, start_jerk_coefficient,
          goal_position_coefficient, position_residual, knots, seeds, steps,
          dof, knot_count);
  return cudaGetLastError();
}

cudaError_t launch_seed_fitter_required_dt(
    const float* velocity, const float* acceleration, const float* jerk,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, int seeds, int steps, int dof,
    float trajectory_dt, float dt_epsilon, float* required_dt,
    cudaStream_t stream) {
  using curobo::trajectory::seed_fitter::THREADS_PER_BLOCK;
  curobo::trajectory::seed_fitter::required_dt_kernel
      <<<seeds, THREADS_PER_BLOCK, 0, stream>>>(
          velocity, acceleration, jerk, max_velocity, max_acceleration,
          max_jerk, steps, dof, trajectory_dt, dt_epsilon, required_dt);
  return cudaGetLastError();
}

cudaError_t launch_seed_fitter_derivative_scale(
    const float* current_velocity, const float* current_acceleration,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, const float* basis_velocity,
    const float* basis_acceleration, const float* basis_jerk, int seeds,
    int steps, int dof, int knots, float path_normal_trace,
    float timing_regularization, float irls_epsilon,
    float* velocity_trace_workspace, float* acceleration_trace_workspace,
    float* jerk_trace_workspace, float* derivative_scale,
    cudaStream_t stream) {
  using curobo::trajectory::seed_fitter::THREADS_PER_BLOCK;
  curobo::trajectory::seed_fitter::derivative_scale_kernel<<<seeds, THREADS_PER_BLOCK, 0, stream>>>(
      current_velocity, current_acceleration, max_velocity, max_acceleration, max_jerk,
      basis_velocity, basis_acceleration, basis_jerk, seeds, steps, dof, knots - 1,
      path_normal_trace, timing_regularization, irls_epsilon, velocity_trace_workspace,
      acceleration_trace_workspace, jerk_trace_workspace, derivative_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  using curobo::trajectory::seed_fitter::THREADS_PER_BLOCK;
  const int free_knots = knots - 1;
  const size_t shared_values = static_cast<size_t>(2 * steps) +
                               static_cast<size_t>(free_knots * free_knots) +
                               static_cast<size_t>(free_knots);
  curobo::trajectory::seed_fitter::solve_irls_kernel
      <<<seeds * dof, THREADS_PER_BLOCK, shared_values * sizeof(float),
         stream>>>(
          position_residual, boundary_velocity, boundary_acceleration,
          boundary_jerk, current_velocity, current_acceleration, current_knots,
          goal_position, max_velocity, max_acceleration, max_jerk,
          basis_position, basis_velocity, basis_acceleration, basis_jerk,
          path_normal, derivative_scale, steps, dof, knots, regularization,
          trust_region_regularization, path_diagonal_scale, irls_epsilon,
          candidate_knots);
  return cudaGetLastError();
}

cudaError_t launch_seed_fitter_accept_candidate(
    const float* candidate_knots, const float* candidate_velocity,
    const float* candidate_acceleration, const float* candidate_jerk,
    const float* candidate_dt, float* current_knots,
    float* current_velocity, float* current_acceleration,
    float* current_jerk, float* current_dt, int seeds, int steps, int dof,
    int knot_count, cudaStream_t stream) {
  using curobo::trajectory::seed_fitter::THREADS_PER_BLOCK;
  const size_t elements = static_cast<size_t>(seeds) * steps * dof;
  curobo::trajectory::seed_fitter::accept_candidate_kernel
      <<<grid_for(elements), THREADS_PER_BLOCK, 0, stream>>>(
          candidate_knots, candidate_velocity, candidate_acceleration,
          candidate_jerk, candidate_dt, current_knots, current_velocity,
          current_acceleration, current_jerk, current_dt, seeds, steps, dof,
          knot_count);
  curobo::trajectory::seed_fitter::accept_candidate_dt_kernel
      <<<grid_for(static_cast<size_t>(seeds)), THREADS_PER_BLOCK, 0, stream>>>(
          candidate_dt, current_dt, seeds);
  return cudaGetLastError();
}

}  // namespace curobo::robot::detail
