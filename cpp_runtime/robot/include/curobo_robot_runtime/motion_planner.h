// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "curobo_robot_runtime/api.h"
#include "curobo_robot_runtime/inverse_kinematics.h"
#include "curobo_robot_runtime/rrt.h"
#include "curobo_robot_runtime/trajectory_optimization.h"

namespace curobo::robot {

class Scene;

namespace detail {
class MotionPlannerRuntime;
class MpcRuntime;
}

/** Policy for one joint-space motion-planning pipeline. */
struct MotionPlanningConfig {
  TrajectoryOptimizationConfig trajectory_optimization{};
  RrtConfig rrt{};
  bool attempt_direct_trajectory_optimization{true};
  bool use_cuda_graph{true};
  /** IK ranking policy; cspace_rank_weight favors the current configuration. */
  InverseKinematicsConfig inverse_kinematics = [] {
    InverseKinematicsConfig config;
    config.cspace_rank_weight = 5.0F;
    return config;
  }();
};

/** Fixed artifact dimensions needed to allocate motion-planning outputs. */
struct MotionPlanningInfo {
  int num_dof{0};
  int num_knots{0};
  int trajectory_steps{0};
  int max_rrt_path_nodes{0};
};

/**
 * CUDA tensors for one joint-space motion-planning request.
 *
 * Start and goal have shape `[num_dof]` and are the common fallback endpoints.
 * Optional start velocity and acceleration have shape `[num_dof]`; null
 * pointers request zero boundary derivatives.
 * Optional per-seed starts and goals have shape `[num_seeds, num_dof]` and are
 * paired across RRT and TrajOpt. `seed_valid`, when non-null, has shape
 * `[num_seeds]` and masks unusable pairs such as failed IK solutions.
 * Optimized knots have shape `[num_knots, num_dof]`; trajectory outputs have
 * shape `[trajectory_steps, num_dof]`. All status and cost outputs contain one
 * element. `selected_seed` identifies the chosen endpoint pair.
 * `trajectory_dt` is the selected uniform timestep. `used_rrt` is one when the
 * selected TrajOpt seed came from a successful RRT path.
 */
struct MotionPlanningRequest {
  const float* start_position{nullptr};
  const float* goal_position{nullptr};
  const float* seed_goal_positions{nullptr};
  float* optimized_knots{nullptr};
  float* trajectory_position{nullptr};
  float* trajectory_velocity{nullptr};
  float* trajectory_acceleration{nullptr};
  float* trajectory_jerk{nullptr};
  uint8_t* success{nullptr};
  float* cost{nullptr};
  float* collision_cost{nullptr};
  float* trajectory_dt{nullptr};
  uint8_t* used_rrt{nullptr};
  int32_t* rrt_iterations{nullptr};
  cudaStream_t stream{nullptr};
  const float* seed_start_positions{nullptr};
  const uint8_t* seed_valid{nullptr};
  int32_t* selected_seed{nullptr};
  /** Host-side stage policy; defaults preserve joint-space plan behavior. */
  bool attempt_direct_trajectory_optimization{true};
  bool allow_rrt{true};
  int finetune_attempts{-1};
  float finetune_dt_scale{-1.0F};
  /** Optional Cartesian-line objective applied by the TrajOpt stage. */
  const float* goal_tool_positions{nullptr};
  const float* goal_tool_quaternions{nullptr};
  bool constrain_tool_path{false};
  std::array<float, 3> free_position_direction{0.0F, 0.0F, 1.0F};
  bool direction_in_goal_frame{true};
  float path_position_tolerance{-1.0F};
  float path_orientation_tolerance{-1.0F};
  const float* start_velocity{nullptr};
  const float* start_acceleration{nullptr};
};

/**
 * CUDA tensors for Python-compatible single-pose motion planning.
 *
 * Goal positions and wxyz quaternions have shapes `[num_tool_frames, 3]` and
 * `[num_tool_frames, 4]`. Current position has shape `[num_dof]`. Optional IK
 * seeds have shape `[inverse_kinematics.num_seeds, num_dof]`. Trajectory
 * outputs match `MotionPlanningRequest`. Optional selected goal position,
 * pose errors, selected seed, and planning-attempt count are scalar or
 * `[num_dof]` outputs as appropriate. Optional current velocity and
 * acceleration have shape `[num_dof]`; null pointers request a static start.
 */
struct PoseMotionPlanningRequest {
  const float* goal_positions{nullptr};
  const float* goal_quaternions{nullptr};
  const float* current_position{nullptr};
  const float* ik_seed_positions{nullptr};
  float* selected_goal_position{nullptr};
  float* optimized_knots{nullptr};
  float* trajectory_position{nullptr};
  float* trajectory_velocity{nullptr};
  float* trajectory_acceleration{nullptr};
  float* trajectory_jerk{nullptr};
  uint8_t* success{nullptr};
  float* cost{nullptr};
  float* collision_cost{nullptr};
  float* trajectory_dt{nullptr};
  float* position_error{nullptr};
  float* orientation_error{nullptr};
  uint8_t* used_rrt{nullptr};
  int32_t* rrt_iterations{nullptr};
  int32_t* selected_seed{nullptr};
  int32_t* planning_attempts{nullptr};
  cudaStream_t stream{nullptr};
  int max_attempts{5};
  int enable_graph_attempt{1};
  int finetune_attempts{2};
  float finetune_dt_scale{0.6F};
  /** Constrain intermediate tool poses to a line ending at the goal pose. */
  bool constrain_tool_path{false};
  std::array<float, 3> free_position_direction{0.0F, 0.0F, 1.0F};
  bool direction_in_goal_frame{true};
  float path_position_tolerance{-1.0F};
  float path_orientation_tolerance{-1.0F};
  const float* current_velocity{nullptr};
  const float* current_acceleration{nullptr};
};

/** Cartesian axis used to offset and constrain grasp phases. */
enum class CartesianAxis : int32_t { X = 0, Y = 1, Z = 2 };

/** Host-visible terminal status for a synchronous grasp-planning request. */
enum class GraspPlanningStatus : int32_t {
  SUCCESS = 0,
  GOALSET_UNREACHABLE = 1,
  APPROACH_FAILED = 2,
  GRASP_FAILED = 3,
  LIFT_FAILED = 4,
};

/** CUDA outputs for one preserved grasp phase. */
struct GraspPhaseOutput {
  float* optimized_knots{nullptr};
  float* trajectory_position{nullptr};
  float* trajectory_velocity{nullptr};
  float* trajectory_acceleration{nullptr};
  float* trajectory_jerk{nullptr};
  uint8_t* success{nullptr};
  float* cost{nullptr};
  float* collision_cost{nullptr};
  float* trajectory_dt{nullptr};
  uint8_t* used_rrt{nullptr};
  int32_t* rrt_iterations{nullptr};
  int32_t* selected_seed{nullptr};
  int32_t* planning_attempts{nullptr};
  /** Optional owned executable controls, copied before planning the next phase.
   * Spatial: [num_knots+6,num_dof]; timing: [32]; duration: CUDA scalar.
   * Null outputs are ignored; disabled timeopt and unplanned phases clear them. */
  float* spatial_controls{nullptr};
  float* timing_controls{nullptr};
  float* timeopt_duration{nullptr};
};

/** Per-phase routing for a single-tool grasp and a 32-sphere attachment cache. */
struct GraspCollisionPolicy {
  bool enabled{false};
  uint32_t common_layers{1U};
  uint32_t full_layer{2U};
  uint32_t without_target_layer{4U};
  /** Optional objects-only field for contact links during grasp and lift.
   * Excludes common layers too, permitting contact with the environment. */
  uint32_t contact_layer{0U};
  std::vector<std::string> contact_links{};
  /** Must be fixed at the first grasp tool frame, with identity transform. */
  std::string attachment_link{"attached_object"};
  /** Optional finite float32 [N,3] voxel centers in the robot base frame.
   * Null keeps non-contact links on the full field and leaves geometry intact.
   * Requires contact_layer when null. The caller
   * owns this CUDA buffer and both ESDFs until synchronous plan_grasp returns.
   * All three must come from one immutable mapping snapshot. Storage must be
   * on the planner's device and ready on GraspPlanningRequest::stream. */
  const float* target_points{nullptr};
  int target_point_count{0};
  /** Adds the selected TSDF voxel half-diagonal to the fitted AABB cover. */
  float target_voxel_radius{0.0F};
};

/**
 * Select a goal, approach, insert, then lift. Candidate positions and wxyz
 * quaternions have shapes [num_tool_frames,num_grasps,3] and
 * [num_tool_frames,num_grasps,4]. Current state and selected configuration
 * have shape [num_dof]; null velocity/acceleration request a static start.
 * Phase outputs use the fixed dimensions in MotionPlanningInfo.
 */
struct GraspPlanningRequest {
  /** Layer routing replaces whole-world contact-link disabling when enabled. */
  GraspCollisionPolicy collision_policy{};
  const float* grasp_positions{nullptr};
  const float* grasp_quaternions{nullptr};
  const float* current_position{nullptr};
  int num_grasps{1};
  GraspPhaseOutput approach{};
  GraspPhaseOutput grasp{};
  GraspPhaseOutput lift{};
  uint8_t* success{nullptr};
  uint8_t* goalset_success{nullptr};
  int32_t* goalset_index{nullptr};
  float* selected_grasp_configuration{nullptr};
  float* goalset_position_error{nullptr};
  float* goalset_orientation_error{nullptr};
  float* goalset_collision_cost{nullptr};
  cudaStream_t stream{nullptr};
  CartesianAxis grasp_approach_axis{CartesianAxis::Z};
  float grasp_approach_offset{-0.15F};
  bool grasp_approach_in_tool_frame{true};
  CartesianAxis grasp_lift_axis{CartesianAxis::Z};
  float grasp_lift_offset{-0.15F};
  bool grasp_lift_in_tool_frame{true};
  bool plan_approach_to_grasp{true};
  bool plan_grasp_to_lift{true};
  /** Use artifact contact links only when the explicit list is empty. */
  bool use_default_collision_links{true};
  std::vector<std::string> disable_collision_links{};
  /** Select among IK goalset solutions using the complete motion pipeline. */
  bool use_trajopt_for_goalset_selection{false};
  /** Run one timestep-refinement pass for insertion and lift. */
  bool use_double_pass_trajopt_for_constrained_phases{false};
  const float* current_velocity{nullptr};
  const float* current_acceleration{nullptr};
  /** Optional inertial updates before approach, insertion, and lift. Empty
   * entries preserve the current model. The last applied updates remain active
   * after this call; set the payload explicitly before subsequent planning. */
  std::array<LinkInertialUpdates, 3> phase_inertial_updates{};
  /** Optional actual execution speeds for approach, insertion, and lift.
   * Zero preserves the optimizer's configured speed for that phase. */
  std::array<float, 3> phase_execution_speed_scales{};
};

/**
 * Single-request joint-space motion planner.
 *
 * The planner first tries straight-seeded TrajOpt when configured, then falls
 * back to RRT, geometric B-spline fitting, and a final TrajOpt pass. Calls are
 * synchronous because stage selection observes device status on the host. When
 * enabled, CUDA graphs cache the direct and RRT-seeded TrajOpt stages for
 * stable request addresses; changing an address recaptures the corresponding
 * graph. One instance must not execute concurrent calls. The Scene must
 * outlive the planner.
 */
class MotionPlanner final {
 public:
  explicit MotionPlanner(const std::string& artifact_directory,
                         const MotionPlanningConfig& config = {},
                         const Scene* scene = nullptr);
  ~MotionPlanner();

  MotionPlanner(const MotionPlanner&) = delete;
  MotionPlanner& operator=(const MotionPlanner&) = delete;
  MotionPlanner(MotionPlanner&& other) noexcept;
  MotionPlanner& operator=(MotionPlanner&& other) noexcept;

  CuroboRobotInfo robot_info() const;
  MotionPlanningInfo info() const;
  const MotionPlanningConfig& config() const;
  /** Synchronized wall time spent in RRT by the last planning request. */
  double last_rrt_time() const;
  /** Host seconds spent solving IK in the last pose plan. */
  double last_ik_time() const;
  /** Number of RRT calls made by the last planning request. */
  int last_rrt_calls() const;

  /** Return the artifact's preallocated collision-sphere slots for a link. */
  int attachment_sphere_capacity(const std::string& link_name) const;

  /** Install the same attachment in the composed IK, RRT, and TrajOpt stages. */
  void attach_object(const SphereAttachment& attachment);
  /** Shared by ordinary/grasp IK, RRT and TrajOpt; does not affect
   * self-collision. */
  void set_link_collision_mask(const std::string& link_name, uint32_t layers);
  /** Set world-collision layers for every shared pre-FK sphere. */
  void set_collision_mask(uint32_t layers);

  /** Restore one link's exported collision spheres in every planning stage. */
  void detach_object(const std::string& link_name = "attached_object");

  /** Update the same inertial model in IK, RRT, and TrajOpt. */
  void update_link_inertial(const LinkInertialUpdate& update);

  /** Synchronously update the same inertial model in every planning stage. */
  void update_links_inertial(const LinkInertialUpdates& updates);

  /** Run the configured motion-planning pipeline synchronously. */
  void plan(const MotionPlanningRequest& request) const;

  /** Run single-pose IK followed by fitted TrajOpt and optional RRT fallback. */
  void plan_pose(const PoseMotionPlanningRequest& request) const;

  /** Borrow all retained TrajOpt seeds from the last successful plan. */
  TrajectoryOptimizationCandidates candidates() const;

  /** Borrow executable states and timing controls selected inside the retry loop. */
  TimeOptTrajectory timeopt_trajectory() const;
  void set_execution_speed_scale(float scale);

  /** Validate sampled states using this planner's model, scene, and streams. */
  const int32_t *validate(const TrajectoryValidationRequest &request) const;
  /** Create private IK workspace sharing this planner's live robot and Scene. */
  std::unique_ptr<InverseKinematics> create_inverse_kinematics(
      const InverseKinematicsConfig& config) const;

  /**
   * Select a grasp and synchronously plan approach, insertion, and lift.
   * Dynamic inputs and phase outputs are staged through fixed internal CUDA
   * addresses so changing caller buffers does not recapture phase graphs.
   */
  GraspPlanningStatus plan_grasp(const GraspPlanningRequest& request) const;

 private:
  friend class detail::MpcRuntime;
  std::unique_ptr<detail::MotionPlannerRuntime> runtime_;
};

}  // namespace curobo::robot
