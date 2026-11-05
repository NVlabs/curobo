// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>

#include <string>

#include "curobo_robot_runtime/trajectory_seed_fitter.h"
#include "robot_runtime.h"

namespace curobo::robot::detail {

class TrajectorySeedFitterRuntime final {
 public:
  TrajectorySeedFitterRuntime(std::string artifact_directory,
                              const TrajectorySeedFitterConfig& config);
  ~TrajectorySeedFitterRuntime();

  CuroboRobotInfo robot_info() const;
  TrajectorySeedFitterInfo info() const;
  const TrajectorySeedFitterConfig& config() const { return config_; }
  void fit(const TrajectorySeedFitterRequest& request) const;

 private:
  void validate_artifacts() const;
  void load_constants();
  void load_kernels();
  void allocate_workspace();
  void fit_impl(const TrajectorySeedFitterRequest& request) const;
  static bool same_graph_request(const TrajectorySeedFitterRequest& left,
                                 const TrajectorySeedFitterRequest& right);
  void destroy_graph() noexcept;
  void launch_bspline(CUdeviceptr knots, CUdeviceptr start_position,
                      CUdeviceptr start_velocity,
                      CUdeviceptr start_acceleration, CUdeviceptr start_jerk,
                      CUdeviceptr goal_position, CUdeviceptr out_position,
                      CUdeviceptr out_velocity,
                      CUdeviceptr out_acceleration, CUdeviceptr out_jerk,
                      CUstream stream) const;

  std::string artifact_directory_;
  TrajectorySeedFitterConfig config_;
  curobo_robot_runtime::MotionPlanningManifest manifest_;
  curobo_robot_runtime::TrajectoryOptimizationManifest trajopt_manifest_;

  curobo_robot_runtime::DeviceBuffer arc_progress_;
  curobo_robot_runtime::DeviceBuffer fit_matrix_;
  curobo_robot_runtime::DeviceBuffer start_position_coefficient_;
  curobo_robot_runtime::DeviceBuffer start_velocity_coefficient_;
  curobo_robot_runtime::DeviceBuffer start_acceleration_coefficient_;
  curobo_robot_runtime::DeviceBuffer start_jerk_coefficient_;
  curobo_robot_runtime::DeviceBuffer goal_position_coefficient_;
  curobo_robot_runtime::DeviceBuffer basis_position_;
  curobo_robot_runtime::DeviceBuffer basis_velocity_;
  curobo_robot_runtime::DeviceBuffer basis_acceleration_;
  curobo_robot_runtime::DeviceBuffer basis_jerk_;
  curobo_robot_runtime::DeviceBuffer path_normal_;
  curobo_robot_runtime::DeviceBuffer max_velocity_;
  curobo_robot_runtime::DeviceBuffer max_acceleration_;
  curobo_robot_runtime::DeviceBuffer max_jerk_;

  mutable curobo_robot_runtime::DeviceBuffer zero_controls_;
  mutable curobo_robot_runtime::DeviceBuffer zero_state_;
  mutable curobo_robot_runtime::DeviceBuffer lane_indices_;
  mutable curobo_robot_runtime::DeviceBuffer implicit_goal_;
  mutable curobo_robot_runtime::DeviceBuffer trajectory_dt_;
  mutable curobo_robot_runtime::DeviceBuffer dense_dt_;
  mutable curobo_robot_runtime::DeviceBuffer arc_target_;
  mutable curobo_robot_runtime::DeviceBuffer position_residual_;
  mutable curobo_robot_runtime::DeviceBuffer boundary_position_;
  mutable curobo_robot_runtime::DeviceBuffer boundary_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer boundary_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer boundary_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer current_knots_;
  mutable curobo_robot_runtime::DeviceBuffer current_position_;
  mutable curobo_robot_runtime::DeviceBuffer current_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer current_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer current_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer current_required_dt_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_knots_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_position_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_required_dt_;
  mutable curobo_robot_runtime::DeviceBuffer derivative_scale_;
  mutable curobo_robot_runtime::DeviceBuffer velocity_trace_workspace_;
  mutable curobo_robot_runtime::DeviceBuffer acceleration_trace_workspace_;
  mutable curobo_robot_runtime::DeviceBuffer jerk_trace_workspace_;

  curobo_robot_runtime::KernelModule bspline_forward_kernel_;
  mutable cudaGraph_t graph_{nullptr};
  mutable cudaGraphExec_t graph_executable_{nullptr};
  mutable TrajectorySeedFitterRequest graph_request_{};
};

}  // namespace curobo::robot::detail
