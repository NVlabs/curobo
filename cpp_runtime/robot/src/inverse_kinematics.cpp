// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/inverse_kinematics.h"

#include "ik_runtime.h"

#include <utility>

namespace curobo::robot {

InverseKinematics::InverseKinematics(
    const std::string& artifact_directory,
    const InverseKinematicsConfig& config, const Scene* scene)
    : runtime_(
          std::make_unique<detail::IkRuntime>(artifact_directory, config, scene)) {}

InverseKinematics::~InverseKinematics() = default;

InverseKinematics::InverseKinematics(
    const InverseKinematics& shared_robot, const InverseKinematicsConfig& config)
    : runtime_(std::make_unique<detail::IkRuntime>(*shared_robot.runtime_, config)) {}

InverseKinematics::InverseKinematics(InverseKinematics&& other) noexcept = default;

InverseKinematics& InverseKinematics::operator=(
    InverseKinematics&& other) noexcept = default;

CuroboRobotInfo InverseKinematics::info() const { return runtime_->info(); }

const InverseKinematicsConfig& InverseKinematics::config() const {
  return runtime_->config();
}

int InverseKinematics::attachment_sphere_capacity(
    const std::string& link_name) const {
  return runtime_->attachment_sphere_capacity(link_name);
}

void InverseKinematics::attach_object(const SphereAttachment& attachment) {
  runtime_->attach_object(attachment);
}

void InverseKinematics::detach_object(const std::string& link_name) {
  runtime_->detach_object(link_name);
}

void InverseKinematics::set_link_collision_mask(const std::string& link_name,
                                                uint32_t layers) {
  runtime_->set_link_collision_mask(link_name, layers);
}

void InverseKinematics::set_link_collision_enabled(
    const std::string& link_name, bool enabled) {
  runtime_->set_link_collision_enabled(link_name, enabled);
}

void InverseKinematics::update_link_inertial(
    const LinkInertialUpdate& update) {
  runtime_->update_link_inertial(update);
}

void InverseKinematics::update_links_inertial(
    const LinkInertialUpdates& updates) {
  runtime_->update_links_inertial(updates);
}

void InverseKinematics::solve(const IkSolveRequest& request) const {
  runtime_->solve(request);
}

bool InverseKinematics::solve_with_host_success(
    const IkSolveRequest& request, uint8_t* host_success,
    std::size_t host_success_count) const {
  return runtime_->solve(request, host_success, host_success_count);
}

}  // namespace curobo::robot
