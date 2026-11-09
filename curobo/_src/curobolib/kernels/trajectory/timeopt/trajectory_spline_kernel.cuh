// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "timeopt/timeopt_path_helpers.cuh"
namespace curobo::trajectory::timeopt {
// Complete cardinal cubic controls, independent of model and optimizer state.
__global__ void interpolate_spline_kernel(
    const float* controls, const float* knot_dt, const float* timing,
    const float* time, const float* polynomial, const float* nodes,
    const float* weights, float* output, int count, int dof, int samples,
    int time_points, float output_dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= samples) return;
  const int spans = count - 3, plane = samples * dof;
  const float duration = timing ? time[time_points - 1] : knot_dt[0] * spans;
  const float requested = i == samples - 1 ? duration : fminf(duration, i * output_dt);
  float x[3] = {duration > 0 ? 1.0f / (duration * duration) : 0.0f, 0, 0};
  const float s = duration == 0 ? 0.0f : (timing
      ? progress_at_time(polynomial, timing, nodes, weights, time, 0,
                         time_points - 1, requested, x)
      : requested / duration);
  const float coordinate = fminf(1.0f, fmaxf(0.0f, s)) * spans;
  const int segment = min(static_cast<int>(coordinate), spans - 1);
  bspline::BSplineContext<3> context(coordinate - segment, 1.0f / spans);
  const float speed = sqrtf(fmaxf(x[0], 0.0f));
  for (int j = 0; j < dof; ++j) {
    float support[4], centered[4];
    for (int k = 0; k < 4; ++k) support[k] = controls[(segment + k) * dof + j];
    for (int k = 0; k < 4; ++k) centered[k] = support[k] - support[0];
    const float p = context.compute_position(support);
    const float v = context.compute_velocity(centered);
    const float a = context.compute_acceleration(centered);
    const float jerk = context.compute_jerk(centered);
    output[i * dof + j] = p;
    output[plane + i * dof + j] = v * speed;
    output[2 * plane + i * dof + j] = a * x[0] + 0.5f * v * x[1];
    output[3 * plane + i * dof + j] = speed * (jerk * x[0] + 1.5f * a * x[1] + 0.5f * v * x[2]);
  }
}
}  // namespace curobo::trajectory::timeopt
