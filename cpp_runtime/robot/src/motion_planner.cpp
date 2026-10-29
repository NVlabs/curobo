// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/motion_planner.h"

#include "motion_planner_runtime.h"

#include <utility>

namespace curobo::robot {

TrajectoryOptimizationCandidates MotionPlanner::candidates() const {
  return runtime_->candidates();
}

TimeOptTrajectory MotionPlanner::timeopt_trajectory() const {
  return runtime_->timeopt_trajectory();
}
void MotionPlanner::set_execution_speed_scale(float scale) {
  runtime_->set_execution_speed_scale(scale);
}

const int32_t *MotionPlanner::validate(const TrajectoryValidationRequest &request) const {
  return runtime_->validate(request);
}

MotionPlanner::MotionPlanner(const std::string& artifact_directory,
                             const MotionPlanningConfig& config,
                             const Scene* scene)
    : runtime_(std::make_unique<detail::MotionPlannerRuntime>(
          artifact_directory, config, scene)) {}

MotionPlanner::~MotionPlanner() = default;

MotionPlanner::MotionPlanner(MotionPlanner&& other) noexcept = default;

MotionPlanner& MotionPlanner::operator=(MotionPlanner&& other) noexcept =
    default;

CuroboRobotInfo MotionPlanner::robot_info() const {
  return runtime_->robot_info();
}

MotionPlanningInfo MotionPlanner::info() const { return runtime_->info(); }

const MotionPlanningConfig& MotionPlanner::config() const {
  return runtime_->config();
}

double MotionPlanner::last_ik_time() const { return runtime_->last_ik_time(); }

double MotionPlanner::last_rrt_time() const {
  return runtime_->last_rrt_time();
}

int MotionPlanner::last_rrt_calls() const {
  return runtime_->last_rrt_calls();
}

int MotionPlanner::attachment_sphere_capacity(
    const std::string& link_name) const {
  return runtime_->attachment_sphere_capacity(link_name);
}

void MotionPlanner::attach_object(const SphereAttachment& attachment) {
  runtime_->attach_object(attachment);
}

void MotionPlanner::set_link_collision_mask(const std::string& link_name, uint32_t layers) {
  runtime_->set_link_collision_mask(link_name, layers);
}

void MotionPlanner::detach_object(const std::string& link_name) {
  runtime_->detach_object(link_name);
}

void MotionPlanner::set_collision_mask(uint32_t layers) { runtime_->set_collision_mask(layers); }

void MotionPlanner::update_link_inertial(
    const LinkInertialUpdate& update) {
  runtime_->update_link_inertial(update);
}

void MotionPlanner::update_links_inertial(
    const LinkInertialUpdates& updates) {
  runtime_->update_links_inertial(updates);
}

void MotionPlanner::plan(const MotionPlanningRequest& request) const {
  runtime_->plan(request);
}

void MotionPlanner::plan_pose(
    const PoseMotionPlanningRequest& request) const {
  runtime_->plan_pose(request);
}

std::unique_ptr<InverseKinematics> MotionPlanner::create_inverse_kinematics(
    const InverseKinematicsConfig& config) const {
  return runtime_->create_inverse_kinematics(config);
}

GraspPlanningStatus MotionPlanner::plan_grasp(
    const GraspPlanningRequest& request) const {
  return runtime_->plan_grasp(request);
}

}  // namespace curobo::robot
