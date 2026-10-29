// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/trajectory_optimization.h"

#include "trajectory_optimizer_runtime.h"

#include <utility>

namespace curobo::robot {

TrajectoryOptimizer::TrajectoryOptimizer(
    const std::string& artifact_directory,
    const TrajectoryOptimizationConfig& config,
    const Scene* scene)
    : runtime_(std::make_unique<detail::TrajectoryOptimizerRuntime>(
          artifact_directory, config, scene)) {}

TrajectoryOptimizer::~TrajectoryOptimizer() = default;

TrajectoryOptimizer::TrajectoryOptimizer(TrajectoryOptimizer&& other) noexcept =
    default;

TrajectoryOptimizer& TrajectoryOptimizer::operator=(
    TrajectoryOptimizer&& other) noexcept = default;

CuroboRobotInfo TrajectoryOptimizer::robot_info() const {
  return runtime_->robot_info();
}

TrajectoryOptimizationInfo TrajectoryOptimizer::info() const {
  return runtime_->info();
}

const TrajectoryOptimizationConfig& TrajectoryOptimizer::config() const {
  return runtime_->config();
}

int TrajectoryOptimizer::attachment_sphere_capacity(
    const std::string& link_name) const {
  return runtime_->attachment_sphere_capacity(link_name);
}

void TrajectoryOptimizer::attach_object(const SphereAttachment& attachment) {
  runtime_->attach_object(attachment);
}

void TrajectoryOptimizer::detach_object(const std::string& link_name) {
  runtime_->detach_object(link_name);
}

void TrajectoryOptimizer::set_link_collision_mask(const std::string& link_name,
                                                  uint32_t layers) {
  runtime_->set_link_collision_mask(link_name, layers);
}

void TrajectoryOptimizer::set_link_collision_enabled(
    const std::string& link_name, bool enabled) {
  runtime_->set_link_collision_enabled(link_name, enabled);
}

void TrajectoryOptimizer::update_link_inertial(
    const LinkInertialUpdate& update) {
  runtime_->update_link_inertial(update);
}

void TrajectoryOptimizer::update_links_inertial(
    const LinkInertialUpdates& updates) {
  runtime_->update_links_inertial(updates);
}

void TrajectoryOptimizer::optimize(
    const TrajectoryOptimizationRequest& request) const {
  runtime_->optimize(request);
}

TrajectoryOptimizationCandidates TrajectoryOptimizer::candidates() const {
  return runtime_->candidates();
}

TimeOptTrajectory TrajectoryOptimizer::timeopt_trajectory() const {
  return runtime_->timeopt_trajectory();
}
void TrajectoryOptimizer::set_execution_speed_scale(float scale) {
  runtime_->set_execution_speed_scale(scale);
}

const int32_t *TrajectoryOptimizer::validate(const TrajectoryValidationRequest &request) const {
  return runtime_->validate(request);
}

}  // namespace curobo::robot
