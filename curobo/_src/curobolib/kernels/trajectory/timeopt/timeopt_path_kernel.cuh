// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "timeopt/timeopt_path_helpers.cuh"
namespace curobo::trajectory::timeopt {
__global__ void sample_with_progress_kernel(const float *controls, const float *start, const float *goal,
                              const float *output_timing, const float *polynomial,
                              const float *nodes, const float *time_weights,
                              const float *output_time, const float *sample_times,
                              const bool *success, const bool *stationary, float *sample_joints,
                              float *sample_progress,
                              int batch, int samples, int dof, int knots, int intervals,
                              int start_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch * samples) return;
  const int b = idx / samples;
  const int plane = batch * samples * dof;
  const float requested = sample_times[idx];
  const float *time = output_time + b * (intervals + 1);
  const float duration = time[intervals];
  float x[3];
  const float s = progress_at_time(polynomial, output_timing, nodes, time_weights,
                                 time, b, intervals, requested, x);
  const float speed = sqrtf(fmaxf(0.0f, x[0]));
  const int spans = knots + 3 - start_offset;
  const float coordinate = s * spans;
  sample_progress[idx] = fminf(1.0f, fmaxf(0.0f,
      (coordinate + start_offset - 1.0f) / (knots + 2)));
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
