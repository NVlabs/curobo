// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/trajectory_seed_fitter.h"

#include "trajectory_seed_fitter_runtime.h"

#include <utility>

namespace curobo::robot {

TrajectorySeedFitter::TrajectorySeedFitter(
    const std::string& artifact_directory,
    const TrajectorySeedFitterConfig& config)
    : runtime_(std::make_unique<detail::TrajectorySeedFitterRuntime>(
          artifact_directory, config)) {}

TrajectorySeedFitter::~TrajectorySeedFitter() = default;

TrajectorySeedFitter::TrajectorySeedFitter(
    TrajectorySeedFitter&& other) noexcept = default;

TrajectorySeedFitter& TrajectorySeedFitter::operator=(
    TrajectorySeedFitter&& other) noexcept = default;

CuroboRobotInfo TrajectorySeedFitter::robot_info() const {
  return runtime_->robot_info();
}

TrajectorySeedFitterInfo TrajectorySeedFitter::info() const {
  return runtime_->info();
}

const TrajectorySeedFitterConfig& TrajectorySeedFitter::config() const {
  return runtime_->config();
}

void TrajectorySeedFitter::fit(
    const TrajectorySeedFitterRequest& request) const {
  runtime_->fit(request);
}

}  // namespace curobo::robot
