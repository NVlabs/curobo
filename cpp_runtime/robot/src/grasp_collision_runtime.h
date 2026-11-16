// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attachment_runtime.h"
#include "curobo_robot_runtime/motion_planner.h"

namespace curobo::robot::detail {

enum class GraspCollisionPhase { Approach, Grasp, Lift };

/** Persistent scratch: never replaces shared geometry or captured pointers. */
struct GraspCollisionWorkspace {
  explicit GraspCollisionWorkspace(RobotCollisionModel& model);
  RobotCollisionModel& model;
  curobo_robot_runtime::DeviceBuffer saved_spheres;
  curobo_robot_runtime::DeviceBuffer saved_masks;
  curobo_robot_runtime::DeviceBuffer fit_valid;
};

/** Request-local GPU override. Does not mutate attachment/perception ownership. */
class GraspCollisionScope final {
 public:
  GraspCollisionScope(GraspCollisionWorkspace& workspace, const GraspCollisionPolicy& policy,
                      cudaStream_t stream);
  ~GraspCollisionScope() noexcept;
  GraspCollisionScope(const GraspCollisionScope&) = delete;
  GraspCollisionScope& operator=(const GraspCollisionScope&) = delete;
  void set_phase(GraspCollisionPhase phase);
  void activate_lift(const float* selected_position, const float* selected_quaternion);
  void restore();

 private:
  GraspCollisionWorkspace& workspace_;
  const GraspCollisionPolicy& policy_;
  cudaStream_t stream_;
  bool active_{false};
};

}  // namespace curobo::robot::detail
