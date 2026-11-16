// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/rrt.h"

#include "rrt_runtime.h"

#include <utility>

namespace curobo::robot {

RrtPlanner::RrtPlanner(const std::string& artifact_directory,
                       const RrtConfig& config, const Scene* scene)
    : runtime_(std::make_unique<detail::RrtRuntime>(artifact_directory, config,
                                                    scene)) {}

RrtPlanner::~RrtPlanner() = default;

RrtPlanner::RrtPlanner(RrtPlanner&& other) noexcept = default;

RrtPlanner& RrtPlanner::operator=(RrtPlanner&& other) noexcept = default;

CuroboRobotInfo RrtPlanner::robot_info() const { return runtime_->robot_info(); }

RrtInfo RrtPlanner::info() const { return runtime_->info(); }

const RrtConfig& RrtPlanner::config() const { return runtime_->config(); }

void RrtPlanner::reset_seed() { runtime_->reset_seed(); }

int RrtPlanner::attachment_sphere_capacity(
    const std::string& link_name) const {
  return runtime_->attachment_sphere_capacity(link_name);
}

void RrtPlanner::attach_object(const SphereAttachment& attachment) {
  runtime_->attach_object(attachment);
}

void RrtPlanner::detach_object(const std::string& link_name) {
  runtime_->detach_object(link_name);
}

void RrtPlanner::set_link_collision_mask(const std::string& link_name,
                                         uint32_t layers) {
  runtime_->set_link_collision_mask(link_name, layers);
}

void RrtPlanner::set_link_collision_enabled(const std::string& link_name,
                                            bool enabled) {
  runtime_->set_link_collision_enabled(link_name, enabled);
}

void RrtPlanner::update_link_inertial(const LinkInertialUpdate& update) {
  runtime_->update_link_inertial(update);
}

void RrtPlanner::update_links_inertial(
    const LinkInertialUpdates& updates) {
  runtime_->update_links_inertial(updates);
}

void RrtPlanner::plan(const RrtRequest& request) const {
  runtime_->plan(request);
}

}  // namespace curobo::robot
