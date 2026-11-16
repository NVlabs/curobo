// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cuda_check.h"
#include "robot_runtime.h"
#include "trajectory/bspline/bspline_interpolation.cuh"

namespace {
constexpr int DEGREE = 3;
constexpr int KNOTS = 16;
constexpr int SHIFT = 4;
constexpr int SAMPLES_PER_KNOT = 4;
constexpr int STEPS = (KNOTS + DEGREE + 1) * SAMPLES_PER_KNOT + 1;
constexpr int HANDOFF_SAMPLE = SHIFT * SAMPLES_PER_KNOT;
// Stop before the new trajectory's terminal boundary changes the overlap.
constexpr int OVERLAP_SAMPLES = STEPS - HANDOFF_SAMPLE - (DEGREE + 1) * SAMPLES_PER_KNOT;

// One thread evaluates both trajectories with the production cubic kernel.
__global__ void continuation(float *output, float dt, bool pinned, bool preserve_jerk) {
  float knots[KNOTS], shifted[KNOTS];
  for (int k = 0; k < KNOTS; ++k) knots[k] = -2.6F + 0.00002F * k * k * k;
  const float start = -2.6F, zero = 0.F, goal = knots[KNOTS - 1];
  const uint8_t implicit = static_cast<uint8_t>(pinned);
  float *q = output, *v = q + STEPS, *a = v + STEPS, *j = a + STEPS;
  float *sq = j + STEPS, *sv = sq + STEPS, *sa = sv + STEPS, *sj = sa + STEPS;
  for (int i = 0; i < STEPS; ++i) {
    curobo::trajectory::bspline::interpolate_bspline_trajectory<float, DEGREE>(
        q, v, a, j, nullptr, knots, &start, &zero, &zero, &zero, &goal, &zero, &zero, &zero, dt,
        &implicit, 1, STEPS, STEPS, 1, 0, i, 0, 0, 0, KNOTS);
  }
  for (int k = 0; k < KNOTS; ++k) shifted[k] = k + SHIFT < KNOTS ? knots[k + SHIFT] : goal;
  const float start_jerk = preserve_jerk ? j[HANDOFF_SAMPLE] : 0.F;
  for (int i = 0; i < STEPS; ++i) {
    curobo::trajectory::bspline::interpolate_bspline_trajectory<float, DEGREE>(
        sq, sv, sa, sj, nullptr, shifted, q + HANDOFF_SAMPLE, v + HANDOFF_SAMPLE,
        a + HANDOFF_SAMPLE, &start_jerk, &goal, &zero, &zero, &zero, dt, &implicit, 1, STEPS,
        STEPS, 1, 0, i, 0, 0, 0, KNOTS);
  }
}

void check_continuation(curobo_robot_runtime::DeviceBuffer &device, float dt, bool pinned,
                        bool preserve_jerk) {
  std::array<float, 8 * STEPS> host{};
  continuation<<<1, 1>>>(reinterpret_cast<float *>(device.ptr()), dt, pinned, preserve_jerk);
  CRR_CUDA_CHECK(cudaGetLastError());
  CRR_CUDA_CHECK(cudaMemcpy(host.data(), reinterpret_cast<void *>(device.ptr()), sizeof(host),
                            cudaMemcpyDeviceToHost));
  for (float value : host) {
    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite spline continuation");
  }
  std::array<float, 4> error{};
  for (int derivative = 0; derivative < 4; ++derivative) {
    for (int sample = 0; sample < OVERLAP_SAMPLES; ++sample) {
      error[derivative] = std::max(error[derivative],
                                   std::abs(host[(derivative + 4) * STEPS + sample] -
                                            host[derivative * STEPS + sample + HANDOFF_SAMPLE]));
    }
  }
  std::cout << "dt=" << dt << " pinned=" << pinned << " preserve_jerk=" << preserve_jerk
            << " q/v/a/j error=";
  for (float value : error) std::cout << value << ' ';
  std::cout << '\n';

  // Float32 tolerances include cancellation at the fastest sample period.
  constexpr std::array<float, 4> TOLERANCES{2.e-5F, 0.001F, 0.2F, 5.F};
  if (preserve_jerk) {
    for (int derivative = 0; derivative < 4; ++derivative) {
      if (error[derivative] > TOLERANCES[derivative])
        throw std::runtime_error("Integer-shift continuation changed");
    }
  } else if (dt < 0.01F && error[2] < TOLERANCES[2]) {
    throw std::runtime_error("Zero-jerk negative control did not expose the seam");
  }
}
}  // namespace

int main() {
  try {
    curobo_robot_runtime::DeviceBuffer device;
    device.allocate(8 * STEPS * sizeof(float));
    for (float dt : {0.025F, 0.00292936F}) {
      for (bool pinned : {false, true}) {
        for (bool preserve_jerk : {false, true})
          check_continuation(device, dt, pinned, preserve_jerk);
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
