// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

#include "curobo_robot_runtime/rrt.h"
#include "robot_runtime.h"

namespace curobo::robot::detail {

class SceneRuntime;
struct RobotCollisionModel;
class RolloutRuntime;
class RolloutStreams;

class RrtRuntime final {
 public:
  RrtRuntime(std::string artifact_directory, const RrtConfig& config,
             const Scene* scene,
             std::shared_ptr<RobotCollisionModel> collision = nullptr);
  ~RrtRuntime();

  RrtRuntime(const RrtRuntime&) = delete;
  RrtRuntime& operator=(const RrtRuntime&) = delete;

  CuroboRobotInfo robot_info() const;
  RrtInfo info() const;
  const RrtConfig& config() const { return config_; }
  int attachment_sphere_capacity(const std::string& link_name) const;
  void attach_object(const SphereAttachment& attachment);
  void detach_object(const std::string& link_name);
  void set_link_collision_mask(const std::string& link, uint32_t layers);
  void set_link_collision_enabled(const std::string& link_name, bool enabled);
  void update_link_inertial(const LinkInertialUpdate& update);
  void update_links_inertial(const LinkInertialUpdates& updates);
  void reset_seed();
  void plan(const RrtRequest& request) const;

 private:
  enum class SearchGraphKind { DIRECT, START_GROWTH, GOAL_GROWTH };

  // Graphs bind only planner-owned fixed-address workspace and are reused when
  // caller input/output addresses change.
  struct SearchGraph {
    SearchGraphKind kind{SearchGraphKind::DIRECT};
    int plans{0};
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t executable{nullptr};
  };

  struct FinalizationGraph {
    int plans{0};
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t executable{nullptr};
  };

  void validate_config() const;
  void validate_scene() const;
  uint32_t scene_collision_mask() const;
  void refresh_scene_graphs() const;
  void load_constants();
  void load_kernels();
  void allocate_workspace();
  void initialize(const RrtRequest& request, int plans, CUstream stream) const;
  void launch_kinematics(CUdeviceptr positions, int states,
                         CUstream stream) const;
  void launch_world_collision(int states, CUstream stream) const;
  void launch_self_collision(int states, CUstream stream) const;
  void evaluate_positions(CUdeviceptr positions, int states,
                          CUstream stream) const;
  void classify_states(CUdeviceptr positions, int states, CUdeviceptr feasible,
                       CUstream stream) const;
  void nearest(CUdeviceptr tree_nodes, CUdeviceptr tree_count,
               CUdeviceptr desired, CUdeviceptr starts, CUdeviceptr parents,
               int plans, int queries_per_plan, CUstream stream) const;
  void prepare_samples(CUdeviceptr samples, int plans, CUstream stream) const;
  CUdeviceptr next_samples(int plans) const;
  void run_edges(CUdeviceptr starts, CUdeviceptr desired, int edges,
                 int candidates_per_plan, bool use_plan_mask,
                 CUdeviceptr endpoints, CUdeviceptr reached,
                 CUdeviceptr progressed, CUstream stream) const;
  void append(CUdeviceptr tree_nodes, CUdeviceptr tree_parents,
              CUdeviceptr tree_count, CUdeviceptr candidates,
              CUdeviceptr candidate_parents, CUdeviceptr candidate_valid,
              CUdeviceptr assigned_slots, CUdeviceptr write_mask,
              int plans, int candidates_per_plan, CUstream stream) const;
  void select_connection(bool start_is_active, int plans,
                         CUstream stream) const;
  void run_direct_edges(int plans, CUstream stream) const;
  void run_direct_stage(int plans, CUstream stream) const;
  void run_growth_stage(bool start_is_active, int plans,
                        CUstream stream) const;
  void launch_search_graph(SearchGraphKind kind, int plans,
                           CUstream stream) const;
  static void destroy_search_graph(SearchGraph* graph) noexcept;
  void run_shortcut_stage(int plans, CUstream stream) const;
  void run_finalization_stage(int plans, CUstream stream) const;
  void launch_finalization_graph(int plans, CUstream stream) const;
  static void destroy_finalization_graph(FinalizationGraph* graph) noexcept;
  void finish_direct(const RrtRequest& request,
                     const std::vector<uint8_t>& direct_success,
                     CUstream stream) const;
  bool finalize_paths(const RrtRequest& request, int plans, int iterations,
                      CUstream stream) const;
  void finish_failure(const RrtRequest& request, int plans, int iterations,
                      CUstream stream) const;

  std::string artifact_directory_;
  RrtConfig config_;
  const SceneRuntime* scene_{nullptr};
  curobo_robot_runtime::RobotRuntimeImpl robot_;
  curobo_robot_runtime::RrtManifest manifest_;
  std::shared_ptr<RobotCollisionModel> collision_;
  std::unique_ptr<RolloutRuntime> rollout_;

  curobo_robot_runtime::DeviceBuffer position_lower_;
  curobo_robot_runtime::DeviceBuffer position_upper_;
  curobo_robot_runtime::DeviceBuffer effort_bounds_fallback_;
  curobo_robot_runtime::DeviceBuffer distance_weight_;
  curobo_robot_runtime::DeviceBuffer samples_;
  curobo_robot_runtime::DeviceBuffer shortcut_pair_sources_;
  curobo_robot_runtime::DeviceBuffer shortcut_pair_destinations_;
  curobo_robot_runtime::DeviceBuffer shortcut_edge_lookup_;
  curobo_robot_runtime::DeviceBuffer self_collision_padding_;
  curobo_robot_runtime::DeviceBuffer self_collision_pairs_;
  curobo_robot_runtime::DeviceBuffer collision_weight_;
  curobo_robot_runtime::DeviceBuffer collision_activation_distance_;
  curobo_robot_runtime::DeviceBuffer scene_indices_;

  mutable curobo_robot_runtime::DeviceBuffer start_nodes_;
  mutable curobo_robot_runtime::DeviceBuffer goal_nodes_;
  mutable curobo_robot_runtime::DeviceBuffer start_parents_;
  mutable curobo_robot_runtime::DeviceBuffer goal_parents_;
  mutable curobo_robot_runtime::DeviceBuffer start_count_;
  mutable curobo_robot_runtime::DeviceBuffer goal_count_;
  mutable curobo_robot_runtime::DeviceBuffer desired_;
  mutable curobo_robot_runtime::DeviceBuffer active_starts_;
  mutable curobo_robot_runtime::DeviceBuffer active_parents_;
  mutable curobo_robot_runtime::DeviceBuffer active_endpoints_;
  mutable curobo_robot_runtime::DeviceBuffer active_reached_;
  mutable curobo_robot_runtime::DeviceBuffer active_progressed_;
  mutable curobo_robot_runtime::DeviceBuffer active_slots_;
  mutable curobo_robot_runtime::DeviceBuffer active_written_;
  mutable curobo_robot_runtime::DeviceBuffer passive_starts_;
  mutable curobo_robot_runtime::DeviceBuffer passive_parents_;
  mutable curobo_robot_runtime::DeviceBuffer passive_endpoints_;
  mutable curobo_robot_runtime::DeviceBuffer passive_reached_;
  mutable curobo_robot_runtime::DeviceBuffer passive_progressed_;
  mutable curobo_robot_runtime::DeviceBuffer passive_slots_;
  mutable curobo_robot_runtime::DeviceBuffer passive_written_;
  mutable curobo_robot_runtime::DeviceBuffer targets_;
  mutable curobo_robot_runtime::DeviceBuffer line_points_;
  mutable curobo_robot_runtime::DeviceBuffer line_valid_;
  mutable curobo_robot_runtime::DeviceBuffer connection_indices_;
  mutable curobo_robot_runtime::DeviceBuffer growth_status_;
  mutable curobo_robot_runtime::DeviceBuffer terminal_positions_;
  mutable curobo_robot_runtime::DeviceBuffer state_feasible_;
  mutable curobo_robot_runtime::DeviceBuffer active_plans_;
  mutable curobo_robot_runtime::DeviceBuffer direct_point_count_;
  mutable curobo_robot_runtime::DeviceBuffer direct_terminal_feasible_;
  mutable curobo_robot_runtime::DeviceBuffer reconstruction_path_lengths_;

  mutable curobo_robot_runtime::DeviceBuffer shortcut_raw_path_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_path_count_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_edge_offsets_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_edge_steps_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_edge_tested_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_edge_visible_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_packed_states_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_dp_hops_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_dp_cost_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_parents_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_output_path_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_output_indices_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_status_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_path_lengths_;
  mutable curobo_robot_runtime::DeviceBuffer shortcut_success_;

  mutable curobo_robot_runtime::DeviceBuffer link_position_;
  mutable curobo_robot_runtime::DeviceBuffer link_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer global_cumulative_;
  mutable curobo_robot_runtime::DeviceBuffer robot_spheres_;
  mutable curobo_robot_runtime::DeviceBuffer world_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer world_collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_cost_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_pair_distance_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_sparse_index_;
  mutable curobo_robot_runtime::DeviceBuffer joint_effort_;
  mutable curobo_robot_runtime::DeviceBuffer zero_derivatives_;
  mutable curobo_robot_runtime::DeviceBuffer rnea_forward_cache_;
  mutable std::vector<SearchGraph> search_graphs_;
  mutable std::vector<FinalizationGraph> finalization_graphs_;
  mutable uint32_t active_scene_collision_mask_{0};
  mutable size_t sample_cursor_{0};
  mutable size_t large_sample_cursor_{0};
  cudaStream_t graph_capture_stream_{nullptr};

  curobo_robot_runtime::KernelModule kinematics_kernel_;
  curobo_robot_runtime::KernelModule collision_cuboid_kernel_;
  curobo_robot_runtime::KernelModule collision_capsule_kernel_;
  curobo_robot_runtime::KernelModule collision_cylinder_kernel_;
  curobo_robot_runtime::KernelModule collision_mesh_kernel_;
  curobo_robot_runtime::KernelModule collision_voxel_kernel_;
  curobo_robot_runtime::KernelModule self_collision_kernel_;
  curobo_robot_runtime::KernelModule nearest_target_kernel_;
  curobo_robot_runtime::KernelModule packed_append_kernel_;
  curobo_robot_runtime::KernelModule reconstruct_path_kernel_;
  curobo_robot_runtime::KernelModule prepare_samples_kernel_;
  curobo_robot_runtime::KernelModule steer_edges_kernel_;
  curobo_robot_runtime::KernelModule evaluate_edges_kernel_;
  curobo_robot_runtime::KernelModule prepare_direct_edges_kernel_;
  curobo_robot_runtime::KernelModule interpolate_direct_edges_kernel_;
  curobo_robot_runtime::KernelModule reduce_direct_edges_kernel_;
  curobo_robot_runtime::KernelModule classify_states_kernel_;
  curobo_robot_runtime::KernelModule select_connection_kernel_;
  curobo_robot_runtime::KernelModule shortcut_setup_kernel_;
  curobo_robot_runtime::KernelModule shortcut_fill_edges_kernel_;
  curobo_robot_runtime::KernelModule shortcut_reduce_edges_kernel_;
  curobo_robot_runtime::KernelModule shortcut_select_path_kernel_;
  std::unique_ptr<RolloutStreams> rollout_streams_;
};

}  // namespace curobo::robot::detail
