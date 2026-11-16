// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/spline_trajectory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace curobo::robot {
namespace {
std::array<float, 3> timing_state(const float *controls, float progress) {
  const float s = std::clamp(progress, 0.0F, std::nextafter(1.0F, 0.0F));
  float knots[36], basis[35]{}, derivative[35]{}, second[35]{};
  for (int i = 0; i < 36; ++i) knots[i] = std::clamp((i - 3) / 29.0F, 0.0F, 1.0F);
  for (int i = 0; i < 35; ++i) basis[i] = knots[i] <= s && s < knots[i + 1] ? 1.0F : 0.0F;
  for (int degree = 1; degree <= 3; ++degree) {
    float next[35]{}, d[35]{}, dd[35]{};
    for (int i = 0; i < 35 - degree; ++i) {
      const float a = knots[i + degree] - knots[i];
      const float b = knots[i + degree + 1] - knots[i + 1];
      if (a > 0) {
        next[i] += (s - knots[i]) / a * basis[i];
        d[i] += degree / a * basis[i];
        dd[i] += degree / a * derivative[i];
      }
      if (b > 0) {
        next[i] += (knots[i + degree + 1] - s) / b * basis[i + 1];
        d[i] -= degree / b * basis[i + 1];
        dd[i] -= degree / b * derivative[i + 1];
      }
    }
    std::copy(next, next + 35, basis);
    std::copy(d, d + 35, derivative);
    std::copy(dd, dd + 35, second);
  }
  std::array<float, 3> result{};
  for (int i = 0; i < 32; ++i) {
    result[0] += controls[i] * basis[i];
    result[1] += controls[i] * derivative[i];
    result[2] += controls[i] * second[i];
  }
  return result;
}
float progress_at_time(const TimeOptTiming &timing, float seconds) {
  const int intervals = timing.time_points - 1;
  const float duration = timing.time[intervals];
  if (seconds <= 0 || duration == 0) return 0;
  if (seconds >= duration) return 1;
  const int cell = std::clamp(
      static_cast<int>(std::upper_bound(timing.time, timing.time + timing.time_points, seconds) -
                       timing.time) -
          1,
      0, intervals - 1);
  const float left = static_cast<float>(cell) / intervals;
  const float right = static_cast<float>(cell + 1) / intervals;
  float s = left + (right - left) * (seconds - timing.time[cell]) /
                       std::max(timing.time[cell + 1] - timing.time[cell], 1E-20F);
  constexpr float nodes[]{.0337652429F, .1693953068F, .380690407F,
                          .619309593F,  .830604693F,  .966234757F};
  constexpr float weights[]{.0856622462F, .180380787F, .233956967F,
                            .233956967F,  .180380787F, .0856622462F};
  for (int iteration = 0; iteration < 5; ++iteration) {
    float elapsed = 0;
    for (int k = 0; k < 6; ++k)
      elapsed +=
          weights[k] / std::sqrt(std::max(
                           timing_state(timing.controls, left + (s - left) * nodes[k])[0], 1E-20F));
    const float speed = std::sqrt(std::max(timing_state(timing.controls, s)[0], 1E-20F));
    s = std::clamp(s - (timing.time[cell] + (s - left) * elapsed - seconds) * speed, left, right);
  }
  return s;
}
}  // namespace

SplineState SplineTrajectory::sample(float seconds) const {
  if (device_ != -1 || !std::isfinite(seconds))
    throw std::invalid_argument("sample requires a CPU spline and finite seconds");
  const float end = duration();
  if (!std::isfinite(end) || end < 0) throw std::invalid_argument("Invalid spline duration");
  float progress = 0;
  std::array<float, 3> x{};
  if (const auto *timing = std::get_if<TimeOptTiming>(&timing_)) {
    progress = progress_at_time(*timing, seconds);
    x = timing_state(timing->controls, progress);
  } else if (end > 0) {
    progress = std::clamp(seconds / end, 0.0F, 1.0F);
    x[0] = 1 / (end * end);
  }
  const int spans = count_ - 3;
  const float coordinate = progress * spans;
  const int segment = std::min(static_cast<int>(coordinate), spans - 1);
  const float u = coordinate - segment, v = 1 - u;
  const float b[]{v * v * v / 6, (3 * u * u * u - 6 * u * u + 4) / 6,
                  (-3 * u * u * u + 3 * u * u + 3 * u + 1) / 6, u * u * u / 6};
  const float d[]{-v * v / 2, (3 * u * u - 4 * u) / 2, (-3 * u * u + 2 * u + 1) / 2, u * u / 2};
  const float dd[]{v, 3 * u - 2, 1 - 3 * u, u};
  constexpr float ddd[]{-1, 3, -3, 1};
  const float speed = std::sqrt(std::max(x[0], 0.0F));
  SplineState out;
  out.position.resize(num_dof());
  out.velocity.resize(num_dof());
  out.acceleration.resize(num_dof());
  out.jerk.resize(num_dof());
  for (int j = 0; j < num_dof(); ++j) {
    float p = 0, qs = 0, qss = 0, qsss = 0;
    const float origin = controls_[segment * num_dof() + j];
    for (int k = 0; k < 4; ++k) {
      const float q = controls_[(segment + k) * num_dof() + j];
      p += b[k] * q;
      qs += d[k] * (q - origin) * spans;
      qss += dd[k] * (q - origin) * spans * spans;
      qsss += ddd[k] * (q - origin) * spans * spans * spans;
    }
    out.position[j] = p;
    if (seconds >= 0 && seconds <= end) {
      out.velocity[j] = qs * speed;
      out.acceleration[j] = qss * x[0] + .5F * qs * x[1];
      out.jerk[j] = speed * (qsss * x[0] + 1.5F * qss * x[1] + .5F * qs * x[2]);
    }
  }
  return out;
}
}  // namespace curobo::robot
