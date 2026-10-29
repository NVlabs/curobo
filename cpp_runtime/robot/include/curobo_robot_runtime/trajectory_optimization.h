// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "curobo_robot_runtime/attachment.h"
#include "curobo_robot_runtime/api.h"
#include "curobo_robot_runtime/inertial.h"

namespace curobo::robot {

class Scene;

namespace detail {
class TrajectoryOptimizerRuntime;
}

/** Fixed workspace and convergence policy for joint-space TrajOpt. */
struct TrajectoryOptimizationConfig {
  int device_ordinal{0};
  int num_seeds{4};
  int max_iterations{25};
  /** Iterations in each refinement pass after the initial solve. */
  int refinement_iterations{25};
  int finetune_attempts{2};
  float trajectory_dt{0.15F};
  /** Scale the input seed's required timestep for the first solve. */
  float initial_dt_scale{1.25F};
  float finetune_dt_scale{0.6F};
  float collision_activation_distance{0.0025F};
  float constraint_tolerance{0.0F};
  float position_tolerance{0.005F};
  float orientation_tolerance{0.05F};
  bool check_self_collision{true};
  /** Shrink failed probe fans on GPU without advancing L-BFGS history. */
  bool line_search_backtracking{false};
  /** Compute RNEA torque costs and enforce exported joint effort limits. */
  bool check_effort_limits{false};
  /** Optimize timing for all retained seeds before validation and duration selection. */
  bool enable_timeopt{false};
  /** Also validate timeopt candidates at this constant execution speed. */
  float execution_speed_scale{1.0F};
};

/** Borrowed executable trajectory for the selected timeopt seed.
 * States are [capacity,num_dof]; dt, duration, and sample_count are CUDA scalars.
 * timing_controls is [num_timing_controls] for the same selected seed: physical
 * cubic squared-speed controls x(s)=(ds/dt)^2 on s in [0,1], not timestamps.
 * spatial_controls is [num_spatial_controls,num_dof], the complete uniform
 * cardinal cubic path with num_spatial_controls-3 spans, including boundaries.
 * Only sample_count states are executable. Valid after a successful solve until
 * the next solve; all pointers are null when timeopt is disabled. Native knot
 * and state outputs keep their original geometric-trajectory interpretation.
 * Failed timeopt seed selection clears the controls and sets sample_count to zero.
 */
struct TimeOptTrajectory {
  const float* position{nullptr};
  const float* velocity{nullptr};
  const float* acceleration{nullptr};
  const float* jerk{nullptr};
  const float* dt{nullptr};
  const float* duration{nullptr};
  const int32_t* sample_count{nullptr};
  int capacity{0};
  const float* timing_controls{nullptr};
  int num_timing_controls{0};
  const float* spatial_controls{nullptr};
  int num_spatial_controls{0};
};

/** Fixed artifact dimensions needed to allocate request outputs. */
struct TrajectoryOptimizationInfo {
  int num_dof{0};
  int num_knots{0};
  int trajectory_steps{0};
  int num_tool_frames{0};
};

/** Borrowed per-seed solutions, valid until the next optimization call.
 * Knots are [B,K,D], states [B,T,D], and dt/success [B]. dt is the state
 * timestep; multiply by (T-1)/(K+4) to recover the knot interval.
 */
struct TrajectoryOptimizationCandidates {
  const float *knots{nullptr};
  const float *position{nullptr};
  const float *velocity{nullptr};
  const float *acceleration{nullptr};
  const float *jerk{nullptr};
  const float *dt{nullptr};
  const uint8_t *success{nullptr};
};

/** Validate externally sampled states against the optimizer's model and scene.
 * Each input is [num_seeds, num_samples, num_dof]. The sample count must match
 * the artifact's fixed interpolation buffer. This checks finite states, joint
 * bounds, world/self collision, and configured effort limits.
 */
struct TrajectoryValidationRequest {
  const float *position{nullptr};
  const float *velocity{nullptr};
  const float *acceleration{nullptr};
  const float *jerk{nullptr};
  int num_samples{0};
  cudaStream_t stream{nullptr};
};

/**
 * CUDA tensors for one joint-space trajectory optimization request.
 *
 * Start and goal have shape `[num_dof]` and are the common fallback endpoints.
 * Optional start velocity and acceleration have shape `[num_dof]`; null
 * pointers request zero boundary derivatives.
 * Optional per-seed starts and goals have shape `[num_seeds, num_dof]` and
 * define paired trajectory candidates. Optional seed knots have shape
 * `[num_seeds, num_knots, num_dof]`; a null seed pointer requests straight
 * joint-space seeds. `seed_valid`, when non-null, has shape `[num_seeds]` and
 * masks unusable pairs such as failed IK solutions. Output knots have shape
 * `[num_knots, num_dof]`, trajectory state outputs have shape
 * `[trajectory_steps, num_dof]`, and scalar outputs contain one element.
 */
struct TrajectoryOptimizationRequest {
  const float* start_position{nullptr};
  const float* goal_position{nullptr};
  const float* seed_knots{nullptr};
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
  int32_t* selected_seed{nullptr};
  cudaStream_t stream{nullptr};
  const float* seed_start_positions{nullptr};
  const uint8_t* seed_valid{nullptr};
  /** Negative values use the optimizer configuration defaults. */
  int finetune_attempts{-1};
  float finetune_dt_scale{-1.0F};
  /** Optional single-pose Cartesian path objective for constrained phases. */
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
 * Standalone, fixed-shape cuRobo joint-space trajectory optimizer.
 *
 * Calls are asynchronous and CUDA graph capturable after construction. One
 * instance must not execute concurrent optimize calls. The Scene must outlive
 * the optimizer. Pose IK and graph search are deliberately separate: their
 * joint goals and fitted paths can be passed through this request.
 */
class TrajectoryOptimizer final {
 public:
  explicit TrajectoryOptimizer(
      const std::string& artifact_directory,
      const TrajectoryOptimizationConfig& config = {},
      const Scene* scene = nullptr);
  ~TrajectoryOptimizer();

  TrajectoryOptimizer(const TrajectoryOptimizer&) = delete;
  TrajectoryOptimizer& operator=(const TrajectoryOptimizer&) = delete;
  TrajectoryOptimizer(TrajectoryOptimizer&& other) noexcept;
  TrajectoryOptimizer& operator=(TrajectoryOptimizer&& other) noexcept;

  CuroboRobotInfo robot_info() const;
  TrajectoryOptimizationInfo info() const;
  const TrajectoryOptimizationConfig& config() const;

  /** Return the artifact's preallocated collision-sphere slots for a link. */
  int attachment_sphere_capacity(const std::string& link_name) const;

  /** Synchronously install link-local attached-object collision geometry. */
  void attach_object(const SphereAttachment& attachment);

  /** Synchronously restore one link's exported collision spheres. */
  void detach_object(const std::string& link_name = "attached_object");

  /** Temporarily enable or disable all collision spheres assigned to a link. */
  /** Select world obstacle layers without changing self-collision geometry. */
  void set_link_collision_mask(const std::string& link_name, uint32_t layers);
  void set_link_collision_enabled(const std::string& link_name, bool enabled);

  /** Replace any subset of one link's mass, COM, or inertia. */
  void update_link_inertial(const LinkInertialUpdate& update);

  /** Atomically replace inertial properties for one or more links. */
  void update_links_inertial(const LinkInertialUpdates& updates);

  /** Enqueue multi-seed B-spline L-BFGS and fixed-count dt refinement. */
  void optimize(const TrajectoryOptimizationRequest& request) const;

  TrajectoryOptimizationCandidates candidates() const;

  TimeOptTrajectory timeopt_trajectory() const;
  /** Synchronously update the scale for future timeopt solves; never call
   * concurrently with a solve. Samples and timing controls remain nominal. */
  void set_execution_speed_scale(float scale);

  /** Enqueue validation; return borrowed [num_seeds] flags (zero is valid).
   * Uses the same parallel streams as optimization and is graph capturable.
   */
  const int32_t *validate(const TrajectoryValidationRequest &request) const;

 private:
  std::unique_ptr<detail::TrajectoryOptimizerRuntime> runtime_;
};

}  // namespace curobo::robot
