// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bspline/bspline_kernel.cuh"

namespace curobo_robot_runtime {

// Assemble each state gradient before the shared B-spline boundary handling.
template<bool AddCollision, bool AddEffort>
struct TrajectoryGradient {
  const float* base;
  const float* collision;
  const float* effort;

  __device__ __forceinline__ float operator[](int index) const {
    float value = base[index];
    if constexpr (AddCollision) value += collision[index];
    if constexpr (AddEffort) value += effort[index];
    return value;
  }
};

template<int Degree, bool UseEffort>
__global__ void trajectory_gradient_kernel(
    float* out_grad_knots, const float* grad_position,
    const float* grad_velocity, const float* grad_acceleration,
    const float* grad_jerk, const float* traj_dt, const int32_t* dt_idx,
    const uint8_t* use_implicit_goal_state, int batch_size, int horizon,
    int dof, int n_knots, const float* collision_gradient,
    const float* effort_position, const float* effort_velocity,
    const float* effort_acceleration) {
  using namespace curobo::trajectory::bspline;
  bspline_backward_impl<Degree, float, BasisBackend::MATRIX>(
      out_grad_knots,
      TrajectoryGradient<true, UseEffort>{grad_position, collision_gradient, effort_position},
      TrajectoryGradient<false, UseEffort>{grad_velocity, nullptr, effort_velocity},
      TrajectoryGradient<false, UseEffort>{grad_acceleration, nullptr, effort_acceleration},
      grad_jerk, traj_dt, dt_idx, use_implicit_goal_state, batch_size,
      horizon, dof, n_knots);
}

}  // namespace curobo_robot_runtime
