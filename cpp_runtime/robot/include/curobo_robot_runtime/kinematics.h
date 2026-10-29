// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <string>

#include "curobo_robot_runtime/api.h"

namespace curobo::robot {

/** Typed C++ interface for artifact-backed forward kinematics and Jacobians. */
class Kinematics final {
 public:
  explicit Kinematics(const std::string& artifact_directory, int device_ordinal = 0);
  ~Kinematics();

  Kinematics(const Kinematics&) = delete;
  Kinematics& operator=(const Kinematics&) = delete;
  Kinematics(Kinematics&& other) noexcept;
  Kinematics& operator=(Kinematics&& other) noexcept;

  CuroboRobotInfo info() const;
  std::string joint_name(int joint_index) const;
  std::string tool_frame_name(int tool_frame_index) const;

  /**
   * Enqueue fused FK and Jacobian computation on stream.
   *
   * All pointers refer to contiguous CUDA float32 memory. See
   * CuroboRobotKinematicsRequest for tensor shapes and conventions.
   */
  void forward(const float* joint_positions, float* link_positions,
               float* link_quaternions, float* link_jacobians, int batch_size,
               int horizon, cudaStream_t stream = nullptr) const;

 private:
  CuroboRobotRuntime* runtime_{nullptr};
};

}  // namespace curobo::robot
