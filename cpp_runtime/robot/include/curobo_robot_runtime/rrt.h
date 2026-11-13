// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>

#include "curobo_robot_runtime/attachment.h"
#include "curobo_robot_runtime/api.h"
#include "curobo_robot_runtime/inertial.h"

namespace curobo::robot {

class Scene;

namespace detail {
class RrtRuntime;
}

/** Runtime policy for one bounded bidirectional-RRT request (up to 128 plans). */
struct RrtConfig {
  int device_ordinal{0};
  int max_iterations{32};
  int max_plans{32};
  bool check_self_collision{true};
  /** Reject states outside static gravity and payload torque limits. */
  bool check_effort_limits{false};
  /** Additional world clearance in meters, applied during geometric search. */
  float collision_activation_distance{0.0F};
};

/** Fixed artifact dimensions needed to allocate RRT outputs. */
struct RrtInfo {
  int num_dof{0};
  int samples_per_iteration{0};
  int max_iterations{0};
  int max_nodes_per_tree{0};
  int max_path_nodes{0};
  int max_plans{0};
};

/**
 * CUDA tensors for one joint-space planning request.
 *
 * Start and goal contain `num_plans` paired configurations with shape
 * `[num_plans, num_dof]`. The planner searches all valid pairs together and
 * stops after a complete RRT stage finds at least one path. `plan_valid`, when
 * non-null, has shape `[num_plans]` and masks unusable pairs such as failed IK
 * solutions. Paths have shape `[num_plans, max_path_nodes, num_dof]`; only the
 * leading `path_count[i]` rows of each plan are valid. Status and iteration
 * outputs have shape `[num_plans]`.
 */
struct RrtRequest {
  const float* start_position{nullptr};
  const float* goal_position{nullptr};
  float* path{nullptr};
  int32_t* path_count{nullptr};
  uint8_t* success{nullptr};
  int32_t* iterations{nullptr};
  cudaStream_t stream{nullptr};
  int num_plans{1};
  const uint8_t* plan_valid{nullptr};
};

/**
 * Standalone bounded bidirectional RRT for one set of paired candidates.
 *
 * The planner synchronizes the request stream while coordinating the bounded
 * search and therefore is not end-to-end CUDA graph capturable. GPU tree,
 * steering, collision, and batched path finalization use reusable CUDA graphs
 * over fixed-capacity buffers and perform no allocation during planning. The
 * graphs are rebuilt when a scene obstacle type crosses between empty and
 * populated. One instance must not execute concurrent calls. The Scene must
 * outlive the planner.
 */
class RrtPlanner final {
 public:
  explicit RrtPlanner(const std::string& artifact_directory,
                      const RrtConfig& config = {},
                      const Scene* scene = nullptr);
  ~RrtPlanner();

  RrtPlanner(const RrtPlanner&) = delete;
  RrtPlanner& operator=(const RrtPlanner&) = delete;
  RrtPlanner(RrtPlanner&& other) noexcept;
  RrtPlanner& operator=(RrtPlanner&& other) noexcept;

  CuroboRobotInfo robot_info() const;
  RrtInfo info() const;
  const RrtConfig& config() const;

  /** Reset buffered Halton sampling to its exported initial state. */
  void reset_seed();

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

  /** Run one synchronous bounded bidirectional-RRT candidate-set search. */
  void plan(const RrtRequest& request) const;

 private:
  std::unique_ptr<detail::RrtRuntime> runtime_;
};

}  // namespace curobo::robot
