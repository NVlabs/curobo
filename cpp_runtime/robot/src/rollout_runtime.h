// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>

#include <string>
#include <vector>

#include "curobo_robot_runtime/inertial.h"
#include "robot_runtime.h"

namespace curobo::robot::detail {

enum class RolloutProfile { IK, RRT, TRAJOPT };

/** Shared inverse-dynamics model used by native solver rollouts. */
class RolloutRuntime final {
 public:
  RolloutRuntime(std::string artifact_directory, int device_ordinal,
                 const curobo_robot_runtime::RobotManifest& robot_manifest,
                 RolloutProfile profile);

  RolloutRuntime(const RolloutRuntime&) = delete;
  RolloutRuntime& operator=(const RolloutRuntime&) = delete;

  int num_links() const { return manifest_.num_links; }
  int cache_floats_per_link() const {
    return profile_.cache_floats_per_link;
  }
  CUdeviceptr effort_bounds() const { return effort_bounds_.ptr(); }

  void launch_inverse_dynamics(CUdeviceptr position, CUdeviceptr velocity,
                               CUdeviceptr acceleration, CUdeviceptr effort,
                               CUdeviceptr forward_cache, int states,
                               CUstream stream) const;
  void launch_inverse_dynamics_backward(
      CUdeviceptr effort_gradient, CUdeviceptr position, CUdeviceptr velocity,
      CUdeviceptr forward_cache, CUdeviceptr gradient_position,
      CUdeviceptr gradient_velocity, CUdeviceptr gradient_acceleration,
      int states, CUstream stream) const;

  void update_link_inertial(const LinkInertialUpdate& update);
  void update_links_inertial(const LinkInertialUpdates& updates);

 private:
  void recompute_reduced_model(const std::vector<float>& original_masses_com,
                               const std::vector<float>& original_inertias,
                               std::vector<float>* reduced_masses_com,
                               std::vector<float>* reduced_inertias) const;

  std::string artifact_directory_;
  std::vector<std::string> link_names_;
  curobo_robot_runtime::RolloutManifest manifest_;
  curobo_robot_runtime::RolloutKernelProfile profile_;
  std::vector<float> original_masses_com_;
  std::vector<float> original_inertias_;
  std::vector<int16_t> external_force_owners_;
  std::vector<float> external_force_transforms_;

  curobo_robot_runtime::DeviceBuffer fixed_transforms_;
  curobo_robot_runtime::DeviceBuffer link_masses_com_;
  curobo_robot_runtime::DeviceBuffer link_inertias_;
  curobo_robot_runtime::DeviceBuffer joint_map_type_;
  curobo_robot_runtime::DeviceBuffer joint_map_;
  curobo_robot_runtime::DeviceBuffer link_map_;
  curobo_robot_runtime::DeviceBuffer chain_map_;
  curobo_robot_runtime::DeviceBuffer joint_offset_map_;
  curobo_robot_runtime::DeviceBuffer gravity_;
  curobo_robot_runtime::DeviceBuffer level_starts_;
  curobo_robot_runtime::DeviceBuffer level_links_;
  curobo_robot_runtime::DeviceBuffer effort_bounds_;
  curobo_robot_runtime::KernelModule forward_kernel_;
  curobo_robot_runtime::KernelModule backward_kernel_;
};

}  // namespace curobo::robot::detail
