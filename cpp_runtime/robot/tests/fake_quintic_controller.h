// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mpc_test {
constexpr int DOF = 7;
using Joints = std::array<float, DOF>;
struct State {
  Joints q{}, v{}, a{}, jerk{};
};

/** Offline-only reference generator. No transport, hardware or robot dynamics. */
class FakeQuinticController {
 public:
  // Comparison tolerance only; fitting retains the exact commanded start q/v/a.
  static constexpr float HANDOFF_POSITION_TOLERANCE = 0.05F;
  static constexpr float HANDOFF_VELOCITY_TOLERANCE = 0.1F;
  static constexpr float HANDOFF_ACCELERATION_TOLERANCE = 0.2F;
  struct Segment {
    float dt{};
    std::array<Joints, 6> coefficient{};
    State evaluate(float time) const {
      const float x = std::clamp(time / dt, 0.0F, 1.0F);
      State state;
      for (int j = 0; j < DOF; ++j) {
        const auto c = [&](int k) { return coefficient[k][j]; };
        state.q[j] = c(0) + x * (c(1) + x * (c(2) + x * (c(3) + x * (c(4) + x * c(5)))));
        state.v[j] = (c(1) + x * (2 * c(2) + x * (3 * c(3) + x * (4 * c(4) + x * 5 * c(5))))) / dt;
        state.a[j] = (2 * c(2) + x * (6 * c(3) + x * (12 * c(4) + x * 20 * c(5)))) / (dt * dt);
        state.jerk[j] = (6 * c(3) + x * (24 * c(4) + x * 60 * c(5))) / (dt * dt * dt);
      }
      return state;
    }
  };

  explicit FakeQuinticController(const Joints &initial) { hold_.q = initial; }

  static Segment fit(const State &start, const State &end, float dt) {
    Segment segment;
    segment.dt = dt;
    for (int j = 0; j < DOF; ++j) {
      auto &c = segment.coefficient;
      c[0][j] = start.q[j];
      c[1][j] = start.v[j] * dt;
      c[2][j] = start.a[j] * dt * dt * 0.5F;
      const float p = end.q[j] - c[0][j] - c[1][j] - c[2][j];
      const float v = end.v[j] * dt - c[1][j] - 2 * c[2][j];
      const float a = end.a[j] * dt * dt - 2 * c[2][j];
      c[3][j] = 10 * p - 4 * v + 0.5F * a;
      c[4][j] = -15 * p + 7 * v - a;
      c[5][j] = 6 * p - 3 * v + 0.5F * a;
    }
    return segment;
  }

  State predict(float seconds = 0.0F) const {
    const float time = cursor_ + seconds;
    if (!segments_.empty()) {
      const float dt = segments_.front().dt;
      // All segments have uniform duration. Snap roundoff at a knot to its
      // right-hand segment, whose jerk defines continuation at that boundary.
      const int index = static_cast<int>(std::floor(time / dt + 1.e-5F));
      if (index < static_cast<int>(segments_.size()))
        return segments_[index].evaluate(time - index * dt);
    }
    return hold_;
  }

  void advance(float seconds) { cursor_ += seconds; }
  /** Offline scheduler: exact elapsed time since the accepted replacement. */
  void seek(float seconds) { cursor_ = seconds; }

  void accept(const std::vector<float> &q, const std::vector<float> &v,
              const std::vector<float> &a, float dt, int samples_per_segment = 1) {
    const int steps = static_cast<int>(q.size()) / DOF;
    if (samples_per_segment <= 0 || (steps - 1) % samples_per_segment != 0)
      throw std::invalid_argument("Quintic segments must align with input samples");
    const State start = predict();
    for (int j = 0; j < DOF; ++j) {
      if (std::abs(q[j] - start.q[j]) > HANDOFF_POSITION_TOLERANCE ||
          std::abs(v[j] - start.v[j]) > HANDOFF_VELOCITY_TOLERANCE ||
          std::abs(a[j] - start.a[j]) > HANDOFF_ACCELERATION_TOLERANCE)
        throw std::runtime_error("Discontinuous quintic handoff");
    }
    segments_.clear();
    State previous = start;
    for (int i = samples_per_segment; i < steps; i += samples_per_segment) {
      State next;
      for (int j = 0; j < DOF; ++j) {
        next.q[j] = q[i * DOF + j];
        next.v[j] = v[i * DOF + j];
        next.a[j] = a[i * DOF + j];
      }
      segments_.push_back(fit(previous, next, dt * samples_per_segment));
      previous = next;
    }
    hold_ = previous;
    hold_.v.fill(0);
    hold_.a.fill(0);
    cursor_ = 0;
    stopping_ = false;
  }

  void decelerate(float duration) {
    if (stopping_) return;  // Never restart the braking clock on repeated failures.
    const State start = predict();
    State stop;
    for (int j = 0; j < DOF; ++j)
      stop.q[j] =
          start.q[j] + 0.5F * duration * start.v[j] + duration * duration * start.a[j] / 12.0F;
    segments_ = {fit(start, stop, duration)};
    hold_ = stop;
    cursor_ = 0;
    stopping_ = true;
  }

  bool stopping() const { return stopping_; }
  float remaining() const {
    float time = -cursor_;
    for (const auto &segment : segments_) time += segment.dt;
    return std::max(time, 0.0F);
  }

 private:
  State hold_;
  std::vector<Segment> segments_;
  float cursor_{0};
  bool stopping_{false};
};
}  // namespace mpc_test
