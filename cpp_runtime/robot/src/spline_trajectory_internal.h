// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "curobo_robot_runtime/spline_trajectory.h"
namespace curobo::robot::detail {
SplineTrajectory native_spline(const float *actions, const float *position, const float *velocity,
                               const float *acceleration, const float *jerk, const float *dt,
                               int knots, int steps, std::vector<std::string> names, int device,
                               cudaStream_t stream);
SplineTrajectory timeopt_spline(const float *actions, const float *start_controls,
                                const float *goal, const float *timing, const float *time,
                                int knots, int points, bool reference,
                                std::vector<std::string> names, int device, cudaStream_t stream);
}  // namespace curobo::robot::detail
