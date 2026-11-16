/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/curobo_constants.h"

namespace curobo {
namespace optimization {
namespace line_search {

constexpr int STATUS_BACKTRACK = 5;
constexpr float BACKTRACK_FACTOR = 0.1F;

__device__ __forceinline__ void check_best_convergence(
    const float best_cost,
    const float current_cost,
    const float cost_delta_threshold,
    const float cost_relative_threshold,
    const int current_iteration,
    const int convergence_iteration,
    int& best_iteration,
    bool& update_best,
    bool& converged)
{
  const float cost_delta = best_cost - current_cost;
  const float cost_relative = cost_delta / (best_cost + curobo::common::fp32Precision);

  update_best = cost_delta > cost_delta_threshold && cost_relative > cost_relative_threshold;
  best_iteration = update_best ? current_iteration : best_iteration;
  converged = best_iteration + convergence_iteration < current_iteration;
}

}  // namespace line_search
}  // namespace optimization
}  // namespace curobo
