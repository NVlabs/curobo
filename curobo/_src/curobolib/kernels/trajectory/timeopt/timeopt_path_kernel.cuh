// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "bspline/bspline_boundary_constraint.cuh"
#include "bspline/bspline_context.cuh"

namespace curobo::trajectory::timeopt {

// Evaluate native action controls with the original four boundary controls.
__device__ __forceinline__ void sample_joint(const float *controls, const float *start,
                                             const float *goal, int batch, int joint, int dof,
                                             int knots, int start_offset, int segment, float local,
                                             float *values) {
  float support[4], centered[4];
  for (int k = 0; k < 4; ++k) {
    const int ci = segment + start_offset + k;
    support[k] = ci < 4 ? start[ci]
                        : (ci >= knots + 3 ? goal[batch * dof + joint]
                                           : controls[(batch * knots + ci - 4) * dof + joint]);
  }
  for (int k = 0; k < 4; ++k) centered[k] = support[k] - support[0];
  bspline::BSplineContext<3> context(local, 1.0f / (knots + 3 - start_offset));
  values[0] = context.compute_position(support);
  values[1] = context.compute_velocity(centered);
  values[2] = context.compute_acceleration(centered);
  values[3] = context.compute_jerk(centered);
}

// The static polynomial is exported from the same clamped cubic timing basis.
__device__ __forceinline__ void timing_state(const float *polynomial, const float *output_timing,
                                             int batch, float s, float *x) {
  const int span = min(28, max(0, static_cast<int>(s * 29.0f)));
  const float local = s - static_cast<float>(span) / 29.0f;
  float a[4] = {};
  for (int power = 0; power < 4; ++power) {
    for (int k = 0; k < 4; ++k) {
      a[power] +=
          polynomial[(span * 4 + power) * 32 + span + k] * output_timing[batch * 32 + span + k];
    }
  }
  x[0] = ((a[0] * local + a[1]) * local + a[2]) * local + a[3];
  x[1] = (3.0f * a[0] * local + 2.0f * a[1]) * local + a[2];
  x[2] = 6.0f * a[0] * local + 2.0f * a[1];
}

__global__ void sample_kernel(const float *controls, const float *start, const float *goal,
                              const float *output_timing, const float *polynomial,
                              const float *nodes, const float *time_weights,
                              const float *output_time, const float *sample_times,
                              const bool *success, const bool *stationary, float *sample_joints,
                              int batch, int samples, int dof, int knots, int intervals,
                              int start_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch * samples) return;
  const int b = idx / samples;
  const int plane = batch * samples * dof;
  const float requested = sample_times[idx];
  const float *time = output_time + b * (intervals + 1);
  const float duration = time[intervals];
  const float target = fminf(duration, fmaxf(0.0f, requested));
  int lo = 0, hi = intervals;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (time[mid] <= target)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int cell = min(lo, intervals - 1);
  const float left = static_cast<float>(cell) / intervals;
  const float right = static_cast<float>(cell + 1) / intervals;
  float s =
      left + (right - left) * (target - time[cell]) / fmaxf(time[cell + 1] - time[cell], 1e-20f);
  float x[3];
  for (int iteration = 0; iteration < 5; ++iteration) {
    float elapsed = 0.0f;
    for (int k = 0; k < 6; ++k) {
      timing_state(polynomial, output_timing, b, left + (s - left) * nodes[k], x);
      elapsed += rsqrtf(fmaxf(x[0], 1e-20f)) * time_weights[k];
    }
    timing_state(polynomial, output_timing, b, s, x);
    s = fmaxf(left, fminf(right, s - (time[cell] + (s - left) * elapsed - target) *
                                         sqrtf(fmaxf(x[0], 1e-20f))));
  }
  s = target <= 0.0f ? 0.0f : (target >= duration ? 1.0f : s);
  timing_state(polynomial, output_timing, b, s, x);
  const float speed = sqrtf(fmaxf(0.0f, x[0]));
  const int spans = knots + 3 - start_offset;
  const float coordinate = s * spans;
  const int segment = min(static_cast<int>(coordinate), spans - 1);
  const bool tail = requested < 0.0f || requested > duration || stationary[b];
  for (int j = 0; j < dof; ++j) {
    float boundary[4], v[4];
    for (int k = 0; k < 4; ++k) boundary[k] = start[(b * 4 + k) * dof + j];
    sample_joint(controls, boundary, goal, b, j, dof, knots, start_offset, segment,
                 coordinate - segment, v);
    const float values[4] = {v[0], v[1] * speed, v[2] * x[0] + 0.5f * v[1] * x[1],
                             speed * (v[3] * x[0] + 1.5f * v[2] * x[1] + 0.5f * v[1] * x[2])};
    for (int k = 0; k < 4; ++k) {
      sample_joints[k * plane + idx * dof + j] =
          !success[b] || !isfinite(requested) ? nanf("") : (tail && k > 0 ? 0.0f : values[k]);
    }
  }
}
}  // namespace curobo::trajectory::timeopt
