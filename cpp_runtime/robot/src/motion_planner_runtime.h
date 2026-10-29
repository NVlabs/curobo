// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/trajectory_seed_fitter.h"
#include "robot_runtime.h"

namespace curobo::robot::detail {

class TrajectoryOptimizerRuntime;
class RrtRuntime;
struct GraspCollisionWorkspace;

class MotionPlannerRuntime final {
 public:
  MotionPlannerRuntime(std::string artifact_directory,
                       const MotionPlanningConfig& config,
                       const Scene* scene);
  ~MotionPlannerRuntime();

  MotionPlannerRuntime(const MotionPlannerRuntime&) = delete;
  MotionPlannerRuntime& operator=(const MotionPlannerRuntime&) = delete;

  CuroboRobotInfo robot_info() const;
  MotionPlanningInfo info() const;
  const MotionPlanningConfig& config() const { return config_; }
  double last_ik_time() const { return last_ik_time_seconds_; }
  double last_rrt_time() const { return last_rrt_time_seconds_; }
  int last_rrt_calls() const { return last_rrt_calls_; }
  int attachment_sphere_capacity(const std::string& link_name) const;
  void attach_object(const SphereAttachment& attachment);
  void set_link_collision_mask(const std::string& link_name, uint32_t layers);
  void set_collision_mask(uint32_t layers);
  void detach_object(const std::string& link_name);
  void update_link_inertial(const LinkInertialUpdate& update);
  void update_links_inertial(const LinkInertialUpdates& updates) const;
  void plan(const MotionPlanningRequest& request) const;
  void plan_pose(const PoseMotionPlanningRequest& request) const;
  TrajectoryOptimizationCandidates candidates() const;
  TimeOptTrajectory timeopt_trajectory() const;
  void set_execution_speed_scale(float scale);
  const int32_t *validate(const TrajectoryValidationRequest &request) const;
  std::unique_ptr<InverseKinematics> create_inverse_kinematics(
      const InverseKinematicsConfig& config) const;
  GraspPlanningStatus plan_grasp(const GraspPlanningRequest& request) const;

 private:
  friend class MpcRuntime;
  struct CudaGraphStage final {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t executable{nullptr};
  };

  struct TrajoptGraph final {
    CudaGraphStage setup{};
    CudaGraphStage pass{};
    CudaGraphStage refinement_pass{};
    CudaGraphStage base_accumulate{};
    CudaGraphStage finetune_prepare{};
    CudaGraphStage finetune_accumulate{};
    CudaGraphStage finalize{};
    TrajectoryOptimizationRequest request{};
    bool has_finetune_stages{false};
    bool initialized{false};
  };

  struct TrajoptGraphCache final {
    TrajoptGraph direct{};
    TrajoptGraph rrt{};
  };

  void validate_config() const;
  void load_fit_artifact();
  void allocate_workspace();
  void fit_rrt_path(const MotionPlanningRequest& request,
                    CUstream stream) const;
  void fit_seed_knots(const MotionPlanningRequest& request,
                      bool use_linear_path, CUstream stream) const;
  bool plan_impl(const MotionPlanningRequest& request,
                 bool activate_repaired_seeds,
                 TrajoptGraphCache* graph_cache) const;
  void plan_pose_impl(const PoseMotionPlanningRequest& request,
                      TrajoptGraphCache* graph_cache,
                      const float* known_goal_configuration = nullptr) const;
  void set_rrt_outputs(const MotionPlanningRequest& request, bool used_rrt,
                       CUstream stream) const;
  void finish_rrt_failure(const MotionPlanningRequest& request,
                          CUstream stream) const;
  void finish_pose_failure(const PoseMotionPlanningRequest& request,
                           cudaStream_t stream, int attempts) const;
  void copy_pose_outputs(const PoseMotionPlanningRequest& request, int attempts,
                         cudaStream_t stream) const;
  void set_collision_links(const std::vector<std::string>& links,
                           bool enabled) const;
  bool run_goalset_selection(const GraspPlanningRequest& request,
                             cudaStream_t stream) const;
  bool plan_linear_phase(const float* goal_positions,
                         const float* goal_quaternions,
                         const float* start_position,
                         const float* start_velocity,
                         const float* start_acceleration,
                         const std::array<float, 3>& direction,
                         bool direction_in_goal_frame,
                         int finetune_attempts,
                         const GraspPhaseOutput& output,
                         cudaStream_t stream,
                         TrajoptGraphCache* graph_cache,
                         const float* known_goal_configuration) const;
  GraspPhaseOutput grasp_phase_workspace() const;
  void copy_grasp_phase_output(const GraspPhaseOutput& output,
                               cudaStream_t stream) const;
  void optimize_trajectory(const TrajectoryOptimizationRequest& request,
                           TrajoptGraph* graph) const;
  static bool same_graph_request(const TrajectoryOptimizationRequest& left,
                                 const TrajectoryOptimizationRequest& right);
  static void destroy_graph(TrajoptGraph* graph) noexcept;
  static void destroy_graph_cache(TrajoptGraphCache* graph_cache) noexcept;

  std::string artifact_directory_;
  const Scene* scene_handle_{nullptr};
  MotionPlanningConfig config_;
  curobo_robot_runtime::MotionPlanningManifest manifest_;
  std::unique_ptr<RrtRuntime> rrt_;
  std::unique_ptr<TrajectoryOptimizerRuntime> trajectory_optimizer_;
  std::unique_ptr<TrajectorySeedFitter> seed_fitter_;
  std::unique_ptr<InverseKinematics> inverse_kinematics_;
  std::unique_ptr<InverseKinematics> grasp_inverse_kinematics_;
  std::unique_ptr<GraspCollisionWorkspace> grasp_collision_workspace_;

  curobo_robot_runtime::DeviceBuffer arc_progress_;

  mutable curobo_robot_runtime::DeviceBuffer rrt_path_;
  mutable curobo_robot_runtime::DeviceBuffer rrt_path_count_;
  mutable curobo_robot_runtime::DeviceBuffer rrt_success_;
  mutable curobo_robot_runtime::DeviceBuffer rrt_iterations_;
  mutable curobo_robot_runtime::DeviceBuffer dense_path_;
  mutable curobo_robot_runtime::DeviceBuffer fitted_seed_knots_;
  mutable curobo_robot_runtime::DeviceBuffer fitted_seed_starts_;
  mutable curobo_robot_runtime::DeviceBuffer fitted_seed_start_velocities_;
  mutable curobo_robot_runtime::DeviceBuffer fitted_seed_start_accelerations_;
  mutable curobo_robot_runtime::DeviceBuffer fitted_seed_goals_;
  mutable curobo_robot_runtime::DeviceBuffer fitted_seed_required_dt_;
  mutable curobo_robot_runtime::DeviceBuffer trajopt_selected_seed_;
  mutable curobo_robot_runtime::DeviceBuffer ik_solutions_;
  mutable curobo_robot_runtime::DeviceBuffer ik_success_;
  mutable curobo_robot_runtime::DeviceBuffer ik_position_error_;
  mutable curobo_robot_runtime::DeviceBuffer ik_orientation_error_;
  mutable curobo_robot_runtime::DeviceBuffer ik_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer ik_goalset_index_;
  mutable curobo_robot_runtime::DeviceBuffer pose_selected_seed_;
  mutable curobo_robot_runtime::DeviceBuffer staged_start_position_;
  mutable curobo_robot_runtime::DeviceBuffer staged_start_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer staged_start_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer staged_goal_position_;
  mutable curobo_robot_runtime::DeviceBuffer staged_seed_start_positions_;
  mutable curobo_robot_runtime::DeviceBuffer staged_seed_goal_positions_;
  mutable curobo_robot_runtime::DeviceBuffer staged_seed_valid_;
  mutable curobo_robot_runtime::DeviceBuffer staged_tool_goal_positions_;
  mutable curobo_robot_runtime::DeviceBuffer staged_tool_goal_quaternions_;
  mutable curobo_robot_runtime::DeviceBuffer staged_ik_seed_positions_;
  mutable curobo_robot_runtime::DeviceBuffer staged_selected_goal_position_;
  mutable curobo_robot_runtime::DeviceBuffer staged_position_error_;
  mutable curobo_robot_runtime::DeviceBuffer staged_orientation_error_;
  mutable curobo_robot_runtime::DeviceBuffer staged_selected_seed_;
  mutable curobo_robot_runtime::DeviceBuffer selected_grasp_position_;
  mutable curobo_robot_runtime::DeviceBuffer selected_grasp_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer selected_grasp_configuration_;
  mutable curobo_robot_runtime::DeviceBuffer grasp_goalset_position_;
  mutable curobo_robot_runtime::DeviceBuffer grasp_goalset_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer grasp_phase_start_;
  mutable curobo_robot_runtime::DeviceBuffer grasp_phase_start_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer grasp_phase_start_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer grasp_phase_goal_position_;
  mutable curobo_robot_runtime::DeviceBuffer grasp_phase_goal_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer pose_joint_goal_;
  mutable curobo_robot_runtime::DeviceBuffer approach_goal_position_;
  mutable curobo_robot_runtime::DeviceBuffer approach_goal_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer lift_goal_position_;
  mutable curobo_robot_runtime::DeviceBuffer lift_goal_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer selection_knots_;
  mutable curobo_robot_runtime::DeviceBuffer selection_position_;
  mutable curobo_robot_runtime::DeviceBuffer selection_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer selection_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer selection_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer selection_success_;
  mutable curobo_robot_runtime::DeviceBuffer selection_cost_;
  mutable curobo_robot_runtime::DeviceBuffer selection_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer selection_dt_;
  mutable curobo_robot_runtime::DeviceBuffer selection_used_rrt_;
  mutable curobo_robot_runtime::DeviceBuffer selection_rrt_iterations_;
  mutable curobo_robot_runtime::DeviceBuffer selection_planning_attempts_;
  mutable double last_ik_time_seconds_{0.0};
  mutable double last_rrt_time_seconds_{0.0};
  mutable int last_rrt_calls_{0};
  mutable TrajoptGraphCache default_trajopt_graphs_;
  mutable TrajoptGraphCache goalset_trajopt_graphs_;
  mutable TrajoptGraphCache grasp_approach_trajopt_graphs_;
  mutable TrajoptGraphCache grasp_insert_trajopt_graphs_;
  mutable TrajoptGraphCache grasp_lift_trajopt_graphs_;
  cudaStream_t internal_stream_{nullptr};

  curobo_robot_runtime::KernelModule resample_path_kernel_;
  curobo_robot_runtime::KernelModule prepare_seed_paths_kernel_;
};

}  // namespace curobo::robot::detail
