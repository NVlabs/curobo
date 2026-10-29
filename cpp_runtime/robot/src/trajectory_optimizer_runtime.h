// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>

#include <string>

#include "curobo_robot_runtime/trajectory_optimization.h"
#include "curobo_robot_runtime/trajectory_timeopt.h"
#include "robot_runtime.h"

namespace curobo::robot::detail {

class SceneRuntime;
struct RobotCollisionModel;
class MotionPlannerRuntime;
class RolloutStreams;
class RolloutRuntime;

class TrajectoryOptimizerRuntime final {
 public:
  TrajectoryOptimizerRuntime(
      std::string artifact_directory,
      const TrajectoryOptimizationConfig& config, const Scene* scene,
      std::shared_ptr<RobotCollisionModel> collision = nullptr);
  ~TrajectoryOptimizerRuntime();

  TrajectoryOptimizerRuntime(const TrajectoryOptimizerRuntime&) = delete;
  TrajectoryOptimizerRuntime& operator=(const TrajectoryOptimizerRuntime&) = delete;

  CuroboRobotInfo robot_info() const;
  TrajectoryOptimizationInfo info() const;
  const TrajectoryOptimizationConfig& config() const { return config_; }
  int attachment_sphere_capacity(const std::string& link_name) const;
  void attach_object(const SphereAttachment& attachment);
  void detach_object(const std::string& link_name);
  void set_link_collision_mask(const std::string& link, uint32_t layers);
  void set_link_collision_enabled(const std::string& link_name, bool enabled);
  void update_link_inertial(const LinkInertialUpdate& update);
  void update_links_inertial(const LinkInertialUpdates& updates);
  void optimize(const TrajectoryOptimizationRequest& request) const;
  TrajectoryOptimizationCandidates candidates() const;
  TimeOptTrajectory timeopt_trajectory() const;
  void set_execution_speed_scale(float scale);
  const int32_t *validate(const TrajectoryValidationRequest &request,
                          bool timeopt = false) const;

 private:
  friend class MotionPlannerRuntime;
  friend class MpcRuntime;
  // MPC uploads persistent criteria when changing phases; replay must not
  // overwrite them with host scalar arguments captured during the first solve.
  bool prepare_pose_criteria_{true};

  // Graphable stages used by MotionPlannerRuntime. The optimization pass is
  // uses the initial or refinement iteration budget for the requested stage.
  void validate_request(const TrajectoryOptimizationRequest& request) const;
  int resolved_finetune_attempts(
      const TrajectoryOptimizationRequest& request) const;
  void begin_optimization(const TrajectoryOptimizationRequest& request) const;
  void run_optimization_pass(
      const TrajectoryOptimizationRequest& request, bool refinement = false) const;
  void accumulate_optimization_pass(
      bool first_pass, const TrajectoryOptimizationRequest& request) const;
  void prepare_finetune_pass(
      const TrajectoryOptimizationRequest& request) const;
  bool base_pass_succeeded(cudaStream_t stream) const;
  void finish_optimization(
      const TrajectoryOptimizationRequest& request) const;
  void validate_config() const;
  void validate_scene() const;
  void load_constants();
  void load_kernels();
  void allocate_workspace();
  void initialize_actions(const TrajectoryOptimizationRequest& request,
                          CUstream stream) const;
  void compute_required_dt(CUdeviceptr knots, float scale,
                           const TrajectoryOptimizationRequest& request,
                           CUstream stream) const;
  void expand_trajectory_dt(int trajectories, CUstream stream) const;
  void evaluate(CUdeviceptr knots, int trajectories, CUdeviceptr costs,
                CUdeviceptr knot_gradients,
                const TrajectoryOptimizationRequest& request,
                CUstream stream) const;
  void launch_bspline(CUdeviceptr knots, int trajectories,
                      const TrajectoryOptimizationRequest& request,
                      CUstream stream) const;
  void launch_kinematics(int trajectories, CUstream stream) const;
  void launch_kinematics_to(
      int trajectories, int horizon, CUdeviceptr joint_position,
      CUdeviceptr link_position, CUdeviceptr link_quaternion,
      CUdeviceptr robot_spheres, CUdeviceptr global_cumulative,
      CUstream stream) const;
  void launch_pose(int trajectories,
                   const TrajectoryOptimizationRequest& request,
                   CUstream stream) const;
  void launch_world_collision(int trajectories, CUstream stream) const;
  void launch_metrics_world_collision(int trajectories, CUstream stream) const;
  void launch_metrics_collisions(int trajectories, CUstream stream) const;
  void launch_discrete_world_collision_to(
      int trajectories, int horizon, CUdeviceptr robot_spheres,
      CUdeviceptr collision_cost, CUdeviceptr collision_gradient,
      bool validation_kernels, CUstream stream) const;
  void launch_self_collision(int trajectories, CUstream stream) const;
  void launch_self_collision_to(
      int trajectories, int horizon, CUdeviceptr robot_spheres,
      CUdeviceptr collision_cost, CUdeviceptr collision_gradient,
      CUdeviceptr sparse_index, CUdeviceptr weight, CUstream stream) const;
  void launch_kinematics_backward(
      int trajectories, const TrajectoryOptimizationRequest& request,
      CUstream stream) const;
  void launch_cspace(int trajectories, CUstream stream) const;
  void combine_rollout(int trajectories, CUdeviceptr costs,
                       bool pose_constraint_active, CUstream stream) const;
  void launch_bspline_backward(int trajectories, CUdeviceptr knot_gradients,
                               CUstream stream) const;
  void prepare_initial_state(bool clear_optimizer_state, CUstream stream) const;
  void prepare_search_actions(CUstream stream) const;
  void line_search(CUstream stream) const;
  void lbfgs_step(CUstream stream) const;
  void optimize_pass(const TrajectoryOptimizationRequest& request,
                     CUstream stream, int iterations) const;
  void validate_interpolated(CUdeviceptr knots, CUstream stream) const;
  void validate_states(CUstream stream, bool timeopt = false) const;
  void accumulate(bool first_pass,
                  const TrajectoryOptimizationRequest& request,
                  CUstream stream) const;
  void finalize(const TrajectoryOptimizationRequest& request,
                CUstream stream) const;

  std::string artifact_directory_;
  TrajectoryOptimizationConfig config_;
  // MPC keeps bound weights unscaled; regularization is dt-normalized in both.
  bool retime_cspace_weights_{true};
  const SceneRuntime* scene_{nullptr};
  curobo_robot_runtime::RobotRuntimeImpl robot_;
  curobo_robot_runtime::TrajectoryOptimizationManifest manifest_;
  std::shared_ptr<RobotCollisionModel> collision_;
  std::unique_ptr<RolloutRuntime> rollout_;
  std::unique_ptr<TrajectoryTimeOpt> timeopt_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_boundaries_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_knot_dt_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_times_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_dt_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_counts_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_states_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_metadata_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_count_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_timing_controls_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_spatial_controls_;
  mutable curobo_robot_runtime::DeviceBuffer timeopt_scaled_states_;
  curobo_robot_runtime::DeviceBuffer execution_speed_scale_;

  curobo_robot_runtime::DeviceBuffer position_bounds_;
  curobo_robot_runtime::DeviceBuffer velocity_bounds_;
  curobo_robot_runtime::DeviceBuffer acceleration_bounds_;
  curobo_robot_runtime::DeviceBuffer jerk_bounds_;
  curobo_robot_runtime::DeviceBuffer state_bound_weight_;
  curobo_robot_runtime::DeviceBuffer state_activation_distance_;
  curobo_robot_runtime::DeviceBuffer state_regularization_weight_;
  curobo_robot_runtime::DeviceBuffer cspace_target_weight_;
  curobo_robot_runtime::DeviceBuffer cspace_non_terminal_weight_;
  curobo_robot_runtime::DeviceBuffer cspace_target_dof_weight_;
  curobo_robot_runtime::DeviceBuffer tool_pose_weight_;
  curobo_robot_runtime::DeviceBuffer world_collision_weight_;
  curobo_robot_runtime::DeviceBuffer self_collision_weight_;
  curobo_robot_runtime::DeviceBuffer metrics_world_collision_weight_;
  curobo_robot_runtime::DeviceBuffer metrics_self_collision_weight_;
  curobo_robot_runtime::DeviceBuffer validation_self_collision_weight_;
  curobo_robot_runtime::DeviceBuffer line_search_scale_;
  curobo_robot_runtime::DeviceBuffer optimization_step_max_;
  curobo_robot_runtime::DeviceBuffer self_collision_padding_;
  curobo_robot_runtime::DeviceBuffer self_collision_pairs_;
  curobo_robot_runtime::DeviceBuffer collision_activation_distance_;
  curobo_robot_runtime::DeviceBuffer metrics_collision_activation_distance_;
  curobo_robot_runtime::DeviceBuffer trajectory_dt_;
  curobo_robot_runtime::DeviceBuffer state_dt_;
  curobo_robot_runtime::DeviceBuffer seed_goal_indices_;
  curobo_robot_runtime::DeviceBuffer candidate_goal_indices_;
  curobo_robot_runtime::DeviceBuffer zero_joint_state_;
  curobo_robot_runtime::DeviceBuffer start_indices_;
  curobo_robot_runtime::DeviceBuffer implicit_goal_;
  curobo_robot_runtime::DeviceBuffer scene_indices_;

  mutable curobo_robot_runtime::DeviceBuffer seed_actions_;
  mutable curobo_robot_runtime::DeviceBuffer resolved_start_positions_;
  mutable curobo_robot_runtime::DeviceBuffer resolved_start_velocities_;
  mutable curobo_robot_runtime::DeviceBuffer resolved_start_accelerations_;
  // Zero for ordinary TrajOpt; MPC supplies commanded jerk for exact continuation.
  mutable curobo_robot_runtime::DeviceBuffer resolved_start_jerks_;
  mutable curobo_robot_runtime::DeviceBuffer resolved_goal_positions_;
  mutable curobo_robot_runtime::DeviceBuffer active_seed_valid_;
  mutable curobo_robot_runtime::DeviceBuffer action_;
  mutable curobo_robot_runtime::DeviceBuffer gradient_;
  mutable curobo_robot_runtime::DeviceBuffer cost_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_action_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_cost_;
  mutable curobo_robot_runtime::DeviceBuffer best_action_;
  mutable curobo_robot_runtime::DeviceBuffer best_cost_;
  mutable curobo_robot_runtime::DeviceBuffer best_iteration_;
  mutable curobo_robot_runtime::DeviceBuffer current_iteration_;
  mutable curobo_robot_runtime::DeviceBuffer converged_;
  mutable curobo_robot_runtime::DeviceBuffer step_direction_;
  mutable curobo_robot_runtime::DeviceBuffer reference_action_;
  mutable curobo_robot_runtime::DeviceBuffer reference_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer history_rho_;
  mutable curobo_robot_runtime::DeviceBuffer history_y_;
  mutable curobo_robot_runtime::DeviceBuffer history_s_;
  mutable curobo_robot_runtime::DeviceBuffer history_head_;
  mutable curobo_robot_runtime::DeviceBuffer history_count_;
  mutable curobo_robot_runtime::DeviceBuffer curvature_update_valid_;
  mutable curobo_robot_runtime::DeviceBuffer search_actions_;
  mutable curobo_robot_runtime::DeviceBuffer search_gradients_;
  mutable curobo_robot_runtime::DeviceBuffer search_costs_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_indices_;
  mutable curobo_robot_runtime::DeviceBuffer selected_indices_;
  mutable curobo_robot_runtime::DeviceBuffer line_search_index_;
  mutable curobo_robot_runtime::DeviceBuffer line_search_status_;
  curobo_robot_runtime::DeviceBuffer line_search_beta_;

  mutable curobo_robot_runtime::DeviceBuffer dense_position_;
  mutable curobo_robot_runtime::DeviceBuffer dense_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer dense_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer dense_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer dense_dt_;
  mutable curobo_robot_runtime::DeviceBuffer required_dt_per_timestep_;
  mutable curobo_robot_runtime::DeviceBuffer interpolation_horizon_;
  mutable curobo_robot_runtime::DeviceBuffer interpolation_dt_;
  mutable curobo_robot_runtime::DeviceBuffer validation_position_;
  mutable curobo_robot_runtime::DeviceBuffer validation_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer validation_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer validation_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer validation_dt_;
  mutable curobo_robot_runtime::DeviceBuffer validation_link_position_;
  mutable curobo_robot_runtime::DeviceBuffer validation_link_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer validation_global_cumulative_;
  mutable curobo_robot_runtime::DeviceBuffer validation_robot_spheres_;
  mutable curobo_robot_runtime::DeviceBuffer validation_world_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer validation_world_collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer validation_self_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer validation_self_collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer validation_self_collision_sparse_index_;
  mutable curobo_robot_runtime::DeviceBuffer validation_violation_;
  mutable curobo_robot_runtime::DeviceBuffer validation_joint_effort_;
  mutable curobo_robot_runtime::DeviceBuffer validation_rnea_forward_cache_;
  mutable curobo_robot_runtime::DeviceBuffer link_position_;
  mutable curobo_robot_runtime::DeviceBuffer link_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer global_cumulative_;
  mutable curobo_robot_runtime::DeviceBuffer robot_spheres_;
  mutable curobo_robot_runtime::DeviceBuffer zero_link_position_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer zero_link_quaternion_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer pose_position_metric_;
  mutable curobo_robot_runtime::DeviceBuffer pose_rotation_metric_;
  mutable curobo_robot_runtime::DeviceBuffer pose_tolerance_;
  mutable curobo_robot_runtime::DeviceBuffer pose_error_frame_;
  mutable curobo_robot_runtime::DeviceBuffer pose_cost_;
  mutable curobo_robot_runtime::DeviceBuffer pose_position_distance_;
  mutable curobo_robot_runtime::DeviceBuffer pose_orientation_distance_;
  mutable curobo_robot_runtime::DeviceBuffer pose_position_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer pose_quaternion_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer pose_goalset_index_;
  mutable curobo_robot_runtime::DeviceBuffer world_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer world_collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_pair_distance_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_sparse_index_;
  mutable curobo_robot_runtime::DeviceBuffer collision_joint_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer cspace_cost_;
  mutable curobo_robot_runtime::DeviceBuffer cspace_gradient_position_;
  mutable curobo_robot_runtime::DeviceBuffer cspace_gradient_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer cspace_gradient_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer cspace_gradient_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer joint_effort_;
  mutable curobo_robot_runtime::DeviceBuffer effort_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer effort_gradient_position_;
  mutable curobo_robot_runtime::DeviceBuffer effort_gradient_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer effort_gradient_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer rnea_forward_cache_;
  mutable curobo_robot_runtime::DeviceBuffer retained_action_;
  mutable curobo_robot_runtime::DeviceBuffer retained_position_;
  mutable curobo_robot_runtime::DeviceBuffer retained_velocity_;
  mutable curobo_robot_runtime::DeviceBuffer retained_acceleration_;
  mutable curobo_robot_runtime::DeviceBuffer retained_jerk_;
  mutable curobo_robot_runtime::DeviceBuffer retained_success_;
  mutable curobo_robot_runtime::DeviceBuffer retained_pose_valid_;
  mutable curobo_robot_runtime::DeviceBuffer retained_cost_;
  mutable curobo_robot_runtime::DeviceBuffer retained_rank_score_;
  mutable curobo_robot_runtime::DeviceBuffer retained_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer retained_dt_;
  mutable curobo_robot_runtime::DeviceBuffer finetune_enabled_;
  mutable curobo_robot_runtime::DeviceBuffer selected_seed_;

  curobo_robot_runtime::KernelModule kinematics_kernel_;
  curobo_robot_runtime::KernelModule kinematics_backward_kernel_;
  curobo_robot_runtime::KernelModule bspline_forward_kernel_;
  curobo_robot_runtime::KernelModule bspline_single_dt_kernel_;
  curobo_robot_runtime::KernelModule bspline_backward_kernel_;
  curobo_robot_runtime::KernelModule time_scaled_dt_kernel_;
  curobo_robot_runtime::KernelModule cspace_kernel_;
  curobo_robot_runtime::KernelModule pose_kernel_;
  curobo_robot_runtime::KernelModule collision_cuboid_kernel_;
  curobo_robot_runtime::KernelModule collision_capsule_kernel_;
  curobo_robot_runtime::KernelModule collision_cylinder_kernel_;
  curobo_robot_runtime::KernelModule collision_mesh_kernel_;
  curobo_robot_runtime::KernelModule collision_voxel_kernel_;
  curobo_robot_runtime::KernelModule metrics_collision_cuboid_kernel_;
  curobo_robot_runtime::KernelModule metrics_collision_capsule_kernel_;
  curobo_robot_runtime::KernelModule metrics_collision_cylinder_kernel_;
  curobo_robot_runtime::KernelModule metrics_collision_mesh_kernel_;
  curobo_robot_runtime::KernelModule metrics_collision_voxel_kernel_;
  curobo_robot_runtime::KernelModule validation_collision_cuboid_kernel_;
  curobo_robot_runtime::KernelModule validation_collision_capsule_kernel_;
  curobo_robot_runtime::KernelModule validation_collision_cylinder_kernel_;
  curobo_robot_runtime::KernelModule validation_collision_mesh_kernel_;
  curobo_robot_runtime::KernelModule validation_collision_voxel_kernel_;
  curobo_robot_runtime::KernelModule self_collision_kernel_;
  curobo_robot_runtime::KernelModule initialize_actions_kernel_;
  curobo_robot_runtime::KernelModule prepare_pose_criteria_kernel_;
  curobo_robot_runtime::KernelModule expand_trajectory_dt_kernel_;
  curobo_robot_runtime::KernelModule reduce_required_dt_kernel_;
  curobo_robot_runtime::KernelModule calculate_interpolation_horizon_kernel_;
  curobo_robot_runtime::KernelModule validate_interpolated_kernel_;
  curobo_robot_runtime::KernelModule prepare_initial_state_kernel_;
  curobo_robot_runtime::KernelModule prepare_search_actions_kernel_;
  curobo_robot_runtime::KernelModule combine_rollout_kernel_;
  curobo_robot_runtime::KernelModule accumulate_kernel_;
  curobo_robot_runtime::KernelModule finalize_kernel_;
  curobo_robot_runtime::KernelModule line_search_kernel_;
  curobo_robot_runtime::KernelModule lbfgs_kernel_;
  std::unique_ptr<RolloutStreams> rollout_streams_;
};

}  // namespace curobo::robot::detail
