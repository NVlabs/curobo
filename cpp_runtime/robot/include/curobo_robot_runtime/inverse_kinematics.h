// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "curobo_robot_runtime/attachment.h"
#include "curobo_robot_runtime/api.h"
#include "curobo_robot_runtime/inertial.h"

namespace curobo::robot {

class Scene;

namespace detail {
class IkRuntime;
class MotionPlannerRuntime;
}

/** Fixed workspace and convergence policy for an inverse-kinematics solver. */
struct InverseKinematicsConfig {
  int device_ordinal{0};
  int max_batch_size{1};
  int max_goalset{64};
  int num_seeds{32};
  /** Zero uses the iteration count exported from Python's L-BFGS IK profile. */
  int lbfgs_iterations{0};
  float position_tolerance{0.005F};
  float orientation_tolerance{0.05F};
  float position_weight{1.0F};
  float orientation_weight{1.0F};
  float joint_limit_weight{1.0F};
  float position_rank_weight{200.0F};
  float orientation_rank_weight{20.0F};
  float cspace_rank_weight{1.0F};
  float initial_damping{1.0F};
  float damping_factor{2.0F};
  float minimum_damping{1.0E-5F};
  float maximum_damping{1.0E10F};
  float minimum_trust_ratio{1.0E-5F};
  /** L-BFGS collision activation distance; negative uses the exported value. */
  float optimizer_collision_activation_distance{-1.0F};
  /** Collision activation distance used to validate returned solutions. */
  float collision_activation_distance{0.0F};
  bool check_self_collision{true};
  /** Evaluate static gravity and payload torque during L-BFGS and validation. */
  bool check_effort_limits{false};
  bool use_cuda_graph{true};
  /** Shrink failed L-BFGS probe fans on GPU without advancing history. */
  bool line_search_backtracking{false};
  /** Return after LM when every batch item has at least one valid solution. */
  bool exit_early{true};
  /** Pose-only LM seed-refinement iterations run before L-BFGS. */
  int lm_iterations{16};
};

/**
 * Device tensors for one or more IK goals sharing the solver's Scene.
 *
 * Inputs and outputs are contiguous CUDA storage. Goal shapes are
 * `[batch_size, num_tool_frames, num_goalset, 3]` and
 * `[batch_size, num_tool_frames, num_goalset, 4]`; quaternions use wxyz order.
 * Optional seed positions have shape `[batch_size, num_seeds, num_dof]`. When
 * seeds are null, deterministic artifact seeds are used. Optional current
 * positions have shape `[batch_size, num_dof]`; they become the first LM seed
 * and participate in result ranking. Output shapes are
 * `[batch_size, num_solutions, num_dof]` for
 * solution, `[batch_size, num_solutions]` for every metric, and
 * `[batch_size, num_solutions, num_tool_frames]` for selected goalset indices.
 * The index output is required when `num_goalset` is greater than one.
 */
struct IkSolveRequest {
  const float* goal_positions{nullptr};
  const float* goal_quaternions{nullptr};
  const float* seed_positions{nullptr};
  const float* current_positions{nullptr};
  float* solution{nullptr};
  uint8_t* success{nullptr};
  float* position_error{nullptr};
  float* orientation_error{nullptr};
  float* collision_cost{nullptr};
  int32_t* goalset_index{nullptr};
  int batch_size{1};
  int num_goalset{1};
  int num_solutions{1};
  cudaStream_t stream{nullptr};
  /** Rotate deterministic artifact seeds without recapturing the CUDA graph. */
  int seed_offset{0};
};

/**
 * Multi-seed artifact-backed inverse kinematics for one deployment Scene.
 *
 * The fixed workspace is owned by this object. CUDA graph replay is enabled by
 * default and recaptured when request addresses or dimensions change. One
 * instance must not execute concurrent solve calls. Every solve first runs
 * pose-only LM seed refinement, without collision evaluation. When
 * `exit_early` is enabled, the solver evaluates and ranks those seeds,
 * synchronizes once to inspect their validity, and returns when every batch item
 * has at least one valid solution. Otherwise it continues with collision-aware
 * L-BFGS.
 */
class InverseKinematics final {
 public:
  explicit InverseKinematics(
      const std::string& artifact_directory,
      const InverseKinematicsConfig& config = {},
      const Scene* scene = nullptr);
  /**
   * Create independent workspace/graphs sharing the source's robot and Scene.
   * Link-local spheres (before FK), attachments, collision enable flags, and
   * inertial updates are shared. FK outputs are private to each solver.
   * Synchronize outstanding solves before modifying the shared model.
   */
  InverseKinematics(const InverseKinematics& shared_robot,
                    const InverseKinematicsConfig& config);
  ~InverseKinematics();

  InverseKinematics(const InverseKinematics&) = delete;
  InverseKinematics& operator=(const InverseKinematics&) = delete;
  InverseKinematics(InverseKinematics&& other) noexcept;
  InverseKinematics& operator=(InverseKinematics&& other) noexcept;

  CuroboRobotInfo info() const;
  const InverseKinematicsConfig& config() const;

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

  /** Run LM stage selection and enqueue L-BFGS only when it is still needed. */
  void solve(const IkSolveRequest& request) const;

 private:
  friend class detail::MotionPlannerRuntime;

  bool solve_with_host_success(const IkSolveRequest& request,
                               uint8_t* host_success,
                               std::size_t host_success_count) const;
  std::unique_ptr<detail::IkRuntime> runtime_;
};

}  // namespace curobo::robot
