// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/kinematics.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace curobo::robot {
namespace {

void throw_on_error(int status) {
  if (status != 0) {
    throw std::runtime_error(curobo_robot_last_error());
  }
}

std::string checked_name(const char* name) {
  if (name == nullptr) {
    throw std::runtime_error(curobo_robot_last_error());
  }
  return name;
}

}  // namespace

Kinematics::Kinematics(const std::string& artifact_directory, int device_ordinal) {
  throw_on_error(curobo_robot_create(artifact_directory.c_str(), device_ordinal, &runtime_));
}

Kinematics::~Kinematics() { curobo_robot_destroy(runtime_); }

Kinematics::Kinematics(Kinematics&& other) noexcept : runtime_(other.runtime_) {
  other.runtime_ = nullptr;
}

Kinematics& Kinematics::operator=(Kinematics&& other) noexcept {
  if (this != &other) {
    curobo_robot_destroy(runtime_);
    runtime_ = other.runtime_;
    other.runtime_ = nullptr;
  }
  return *this;
}

CuroboRobotInfo Kinematics::info() const {
  CuroboRobotInfo result{};
  throw_on_error(curobo_robot_get_info(runtime_, &result));
  return result;
}

std::string Kinematics::joint_name(int joint_index) const {
  return checked_name(curobo_robot_joint_name(runtime_, joint_index));
}

std::string Kinematics::tool_frame_name(int tool_frame_index) const {
  return checked_name(curobo_robot_tool_frame_name(runtime_, tool_frame_index));
}

void Kinematics::forward(const float* joint_positions, float* link_positions,
                         float* link_quaternions, float* link_jacobians,
                         int batch_size, int horizon, cudaStream_t stream) const {
  const CuroboRobotKinematicsRequest request{
      reinterpret_cast<std::uintptr_t>(joint_positions),
      reinterpret_cast<std::uintptr_t>(link_positions),
      reinterpret_cast<std::uintptr_t>(link_quaternions),
      reinterpret_cast<std::uintptr_t>(link_jacobians),
      reinterpret_cast<std::uintptr_t>(stream),
      batch_size,
      horizon,
  };
  throw_on_error(curobo_robot_compute_kinematics(runtime_, &request));
}

}  // namespace curobo::robot
