// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "curobo_robot_runtime/api.h"

namespace curobo_robot_runtime {

struct TensorManifest {
  std::string file;
  std::string dtype;
  size_t bytes{0};
};

struct KernelManifest {
  std::string cubin;
  std::string symbol;
  std::string source;
  int threads_per_batch{0};
  int max_batches_per_block{0};
  int max_threads_per_block{0};
  int max_dynamic_shared_memory{0};
};

struct RobotManifest {
  int schema_version{0};
  std::string arch;
  int num_links{0};
  int num_dof{0};
  int num_tool_frames{0};
  std::vector<std::string> link_names;
  std::vector<std::string> joint_names;
  std::vector<std::string> tool_frames;
  TensorManifest fixed_transforms;
  TensorManifest joint_map_type;
  TensorManifest joint_map;
  TensorManifest link_map;
  TensorManifest tool_frame_map;
  TensorManifest joint_offset_map;
  TensorManifest tool_joint_links_data;
  TensorManifest tool_joint_links_offsets;
  TensorManifest joint_link_influence_words;
  TensorManifest joint_links_data;
  TensorManifest joint_links_offsets;
  TensorManifest link_masses_com;
  KernelManifest kernel;
};

struct IkKernelManifest {
  std::string cubin;
  std::string symbol;
  std::string source;
  int shared_memory{0};
  int block_dim{0};
  int max_obstacles{0};
};

struct RolloutKernelProfile {
  int threads_per_batch{0};
  int cache_floats_per_link{0};
  IkKernelManifest forward;
  IkKernelManifest backward;
};

struct RolloutManifest {
  int schema_version{0};
  std::string arch;
  int num_dof{0};
  int num_links{0};
  int num_original_links{0};
  int num_levels{0};
  int level_link_indices{0};
  TensorManifest fixed_transforms;
  TensorManifest link_masses_com;
  TensorManifest link_inertias;
  TensorManifest joint_map_type;
  TensorManifest joint_map;
  TensorManifest link_map;
  TensorManifest chain_map;
  TensorManifest joint_offset_map;
  TensorManifest gravity;
  TensorManifest level_starts;
  TensorManifest level_links;
  TensorManifest original_link_masses_com;
  TensorManifest original_link_inertias;
  TensorManifest external_force_owners;
  TensorManifest external_force_transforms;
  TensorManifest effort_bounds;
  RolloutKernelProfile ik;
  RolloutKernelProfile rrt;
  RolloutKernelProfile trajopt;
};

struct IkManifest {
  int schema_version{0};
  std::string arch;
  int num_links{0};
  int num_dof{0};
  int num_tool_frames{0};
  int num_spheres{0};
  int num_residuals{0};
  int max_seeds{0};
  int num_seed_samples{0};
  int max_goalset{0};
  int num_self_collision_pairs{0};
  int max_cuboids{0};
  int max_capsules{0};
  int max_cylinders{0};
  int max_meshes{0};
  int max_voxel_grids{0};
  int history{0};
  int num_line_search{0};
  int optimizer_iterations{0};
  int convergence_iteration{0};
  float epsilon{0.0F};
  float initial_step_scale{0.0F};
  float armijo_threshold{0.0F};
  float curvature_threshold{0.0F};
  float cost_delta_threshold{0.0F};
  float cost_relative_threshold{0.0F};
  float position_weight{0.0F};
  float orientation_weight{0.0F};
  TensorManifest joint_position_lower;
  TensorManifest joint_position_upper;
  TensorManifest default_joint_position;
  TensorManifest seed_positions;
  TensorManifest link_spheres;
  TensorManifest sphere_collision_mask;
  TensorManifest link_sphere_map;
  TensorManifest self_collision_padding;
  TensorManifest self_collision_pairs;
  TensorManifest position_bounds;
  TensorManifest velocity_bounds;
  TensorManifest cspace_weight;
  TensorManifest cspace_activation_distance;
  TensorManifest cspace_target_weight;
  TensorManifest cspace_target_dof_weight;
  TensorManifest cspace_regularization_weight;
  TensorManifest start_cspace_weight;
  TensorManifest start_cspace_dof_weight;
  TensorManifest world_collision_weight;
  TensorManifest self_collision_weight;
  TensorManifest optimizer_collision_activation_distance;
  TensorManifest line_search_scale;
  TensorManifest optimization_step_max;
  KernelManifest kinematics;
  IkKernelManifest kinematics_backward;
  IkKernelManifest initialize;
  IkKernelManifest error;
  IkKernelManifest update;
  IkKernelManifest rank;
  IkKernelManifest finalize;
  IkKernelManifest levenberg_marquardt;
  IkKernelManifest cspace;
  IkKernelManifest prepare_initial_state;
  IkKernelManifest prepare_search_actions;
  IkKernelManifest prepare_search_actions_backtracking;
  IkKernelManifest combine_lbfgs_rollout;
  IkKernelManifest line_search;
  IkKernelManifest line_search_backtracking;
  IkKernelManifest lbfgs;
  IkKernelManifest collision_cuboid;
  IkKernelManifest collision_capsule;
  IkKernelManifest collision_cylinder;
  IkKernelManifest collision_mesh;
  IkKernelManifest collision_voxel;
  IkKernelManifest self_collision;
};

struct TrajectoryOptimizationManifest {
  int schema_version{0};
  std::string arch;
  int num_dof{0};
  int num_links{0};
  int num_tool_frames{0};
  int num_spheres{0};
  int num_self_collision_pairs{0};
  int num_knots{0};
  int trajectory_steps{0};
  int interpolation_buffer_size{0};
  int spline_degree{0};
  int max_seeds{0};
  int history{0};
  int num_line_search{0};
  int convergence_iteration{0};
  int max_cuboids{0};
  int max_capsules{0};
  int max_cylinders{0};
  int max_meshes{0};
  int max_voxel_grids{0};
  float base_dt{0.0F};
  float minimum_dt{0.0F};
  float maximum_dt{0.0F};
  float interpolation_dt{0.0F};
  float epsilon{0.0F};
  float initial_step_scale{0.0F};
  float armijo_threshold{0.0F};
  float curvature_threshold{0.0F};
  float cost_delta_threshold{0.0F};
  float cost_relative_threshold{0.0F};
  TensorManifest position_bounds;
  TensorManifest velocity_bounds;
  TensorManifest acceleration_bounds;
  TensorManifest jerk_bounds;
  TensorManifest state_bound_weight;
  TensorManifest state_activation_distance;
  TensorManifest state_regularization_weight;
  TensorManifest cspace_target_weight;
  TensorManifest cspace_non_terminal_weight;
  TensorManifest cspace_target_dof_weight;
  TensorManifest tool_pose_weight;
  TensorManifest world_collision_weight;
  TensorManifest self_collision_weight;
  TensorManifest metrics_world_collision_weight;
  TensorManifest metrics_self_collision_weight;
  TensorManifest validation_self_collision_weight;
  TensorManifest line_search_scale;
  TensorManifest optimization_step_max;
  TensorManifest link_spheres;
  TensorManifest sphere_collision_mask;
  TensorManifest link_sphere_map;
  TensorManifest self_collision_padding;
  TensorManifest self_collision_pairs;
  KernelManifest kinematics;
  IkKernelManifest kinematics_backward;
  IkKernelManifest bspline_forward;
  IkKernelManifest bspline_single_dt;
  IkKernelManifest bspline_backward;
  IkKernelManifest time_scaled_dt;
  IkKernelManifest cspace;
  IkKernelManifest pose;
  IkKernelManifest collision_cuboid;
  IkKernelManifest collision_capsule;
  IkKernelManifest collision_cylinder;
  IkKernelManifest collision_mesh;
  IkKernelManifest collision_voxel;
  IkKernelManifest metrics_collision_cuboid;
  IkKernelManifest metrics_collision_capsule;
  IkKernelManifest metrics_collision_cylinder;
  IkKernelManifest metrics_collision_mesh;
  IkKernelManifest metrics_collision_voxel;
  IkKernelManifest validation_collision_cuboid;
  IkKernelManifest validation_collision_capsule;
  IkKernelManifest validation_collision_cylinder;
  IkKernelManifest validation_collision_mesh;
  IkKernelManifest validation_collision_voxel;
  IkKernelManifest self_collision;
  IkKernelManifest initialize_actions;
  IkKernelManifest prepare_pose_criteria;
  IkKernelManifest expand_trajectory_dt;
  IkKernelManifest reduce_required_dt;
  IkKernelManifest calculate_interpolation_horizon;
  IkKernelManifest validate_interpolated;
  IkKernelManifest prepare_initial_state;
  IkKernelManifest prepare_search_actions;
  IkKernelManifest prepare_search_actions_backtracking;
  IkKernelManifest combine_rollout;
  IkKernelManifest accumulate;
  IkKernelManifest finalize;
  IkKernelManifest line_search;
  IkKernelManifest line_search_backtracking;
  IkKernelManifest lbfgs;
};

struct RrtManifest {
  int schema_version{0};
  std::string arch;
  int num_dof{0};
  int num_links{0};
  int num_tool_frames{0};
  int num_spheres{0};
  int num_self_collision_pairs{0};
  int samples_per_iteration{0};
  int max_iterations{0};
  int max_nodes_per_tree{0};
  int fixed_steer_steps{0};
  int direct_steer_steps{0};
  int max_path_nodes{0};
  int sample_buffer_size{0};
  int max_batched_lanes{0};
  int shortcut_path_capacity{0};
  int shortcut_state_capacity{0};
  int shortcut_edge_capacity{0};
  int max_cuboids{0};
  int max_capsules{0};
  int max_cylinders{0};
  int max_meshes{0};
  int max_voxel_grids{0};
  float cspace_resolution{0.0F};
  float max_extension_distance{0.0F};
  float similarity_threshold{0.0F};
  TensorManifest position_lower;
  TensorManifest position_upper;
  TensorManifest distance_weight;
  TensorManifest samples;
  TensorManifest shortcut_pair_sources;
  TensorManifest shortcut_pair_destinations;
  TensorManifest shortcut_edge_lookup;
  TensorManifest link_spheres;
  TensorManifest sphere_collision_mask;
  TensorManifest link_sphere_map;
  TensorManifest self_collision_padding;
  TensorManifest self_collision_pairs;
  KernelManifest kinematics;
  IkKernelManifest collision_cuboid;
  IkKernelManifest collision_capsule;
  IkKernelManifest collision_cylinder;
  IkKernelManifest collision_mesh;
  IkKernelManifest collision_voxel;
  IkKernelManifest self_collision;
  IkKernelManifest nearest_target;
  IkKernelManifest packed_append;
  IkKernelManifest reconstruct_path;
  IkKernelManifest prepare_samples;
  IkKernelManifest steer_edges;
  IkKernelManifest evaluate_edges;
  IkKernelManifest prepare_direct_edges;
  IkKernelManifest interpolate_direct_edges;
  IkKernelManifest reduce_direct_edges;
  IkKernelManifest classify_states;
  IkKernelManifest select_connection;
  IkKernelManifest shortcut_setup;
  IkKernelManifest shortcut_fill_edges;
  IkKernelManifest shortcut_reduce_edges;
  IkKernelManifest shortcut_select_path;
};

struct MotionPlanningManifest {
  int schema_version{0};
  std::string arch;
  std::string fit;
  int num_dof{0};
  int num_knots{0};
  int trajectory_steps{0};
  int interpolation_steps{0};
  int max_rrt_path_nodes{0};
  int max_trajopt_seeds{0};
  std::vector<std::string> grasp_contact_link_names;
  int irls_iterations{0};
  float regularization{0.0F};
  float timing_regularization{0.0F};
  float trust_region_regularization{0.0F};
  float irls_epsilon{0.0F};
  float dt_epsilon{0.0F};
  float trajectory_dt{0.0F};
  float path_normal_trace{0.0F};
  float path_diagonal_scale{0.0F};
  TensorManifest arc_progress;
  TensorManifest fit_matrix;
  TensorManifest start_coefficient;
  TensorManifest start_velocity_coefficient;
  TensorManifest start_acceleration_coefficient;
  TensorManifest start_jerk_coefficient;
  TensorManifest goal_coefficient;
  TensorManifest basis_position;
  TensorManifest basis_velocity;
  TensorManifest basis_acceleration;
  TensorManifest basis_jerk;
  TensorManifest path_normal;
  TensorManifest max_velocity;
  TensorManifest max_acceleration;
  TensorManifest max_jerk;
  IkKernelManifest resample_path;
  IkKernelManifest prepare_seed_paths;
  IkKernelManifest resample_dense;
  IkKernelManifest initialize_fit;
  IkKernelManifest accept_candidate;
};

struct TimeOptManifest {
  int num_dof{0};
  int num_knots{0};
  int reference{0};
  int intervals{0};
  int points{0};
  int tiles{0};
  std::map<std::string, TensorManifest> tensors;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void allocate(size_t bytes);
  void upload(const std::vector<uint8_t>& data);
  void clear(cudaStream_t stream = nullptr) const;
  CUdeviceptr ptr() const { return reinterpret_cast<CUdeviceptr>(pointer_); }
  size_t bytes() const { return bytes_; }

 private:
  void release();

  void* pointer_{nullptr};
  size_t bytes_{0};
};

class KernelModule {
 public:
  KernelModule() = default;
  ~KernelModule();

  KernelModule(const KernelModule&) = delete;
  KernelModule& operator=(const KernelModule&) = delete;

  void load(const std::string& cubin_path, const std::string& symbol);
  void set_max_dynamic_shared_memory(int bytes) const;
  void launch(unsigned int grid_x, unsigned int block_x, unsigned int shared_memory,
              CUstream stream, const std::vector<void*>& arguments) const;

 private:
  CUmodule module_{nullptr};
  CUfunction function_{nullptr};
};

class RobotRuntimeImpl {
 public:
  RobotRuntimeImpl(std::string artifact_dir, int device_ordinal);
  ~RobotRuntimeImpl();

  CuroboRobotInfo info() const;
  const std::string& joint_name(int index) const;
  const std::string& tool_frame_name(int index) const;
  void compute_kinematics(const CuroboRobotKinematicsRequest& request) const;

  struct ModelDeviceView {
    CUdeviceptr fixed_transforms{0};
    CUdeviceptr joint_map_type{0};
    CUdeviceptr joint_map{0};
    CUdeviceptr link_map{0};
    CUdeviceptr tool_frame_map{0};
    CUdeviceptr joint_offset_map{0};
    CUdeviceptr tool_joint_links_data{0};
    CUdeviceptr tool_joint_links_offsets{0};
    CUdeviceptr joint_link_influence_words{0};
    CUdeviceptr joint_links_data{0};
    CUdeviceptr joint_links_offsets{0};
    CUdeviceptr link_masses_com{0};
  };

  const RobotManifest& manifest() const { return manifest_; }
  ModelDeviceView model_device_view() const;

 private:
  void load_model();
  void validate_device_architecture() const;

  std::string artifact_dir_;
  int device_ordinal_{0};
  RobotManifest manifest_;
  DeviceBuffer fixed_transforms_;
  DeviceBuffer joint_map_type_;
  DeviceBuffer joint_map_;
  DeviceBuffer link_map_;
  DeviceBuffer tool_frame_map_;
  DeviceBuffer joint_offset_map_;
  DeviceBuffer tool_joint_links_data_;
  DeviceBuffer tool_joint_links_offsets_;
  DeviceBuffer joint_link_influence_words_;
  DeviceBuffer joint_links_data_;
  DeviceBuffer joint_links_offsets_;
  DeviceBuffer link_masses_com_;
  KernelModule kernel_;
};

RobotManifest load_manifest(const std::string& artifact_dir);
TimeOptManifest load_timeopt_manifest(const std::string &artifact_dir, int knots, bool reference);
IkManifest load_ik_manifest(const std::string& artifact_dir);
RolloutManifest load_rollout_manifest(const std::string& artifact_dir);
TrajectoryOptimizationManifest load_trajectory_optimization_manifest(
    const std::string& artifact_dir);
RrtManifest load_rrt_manifest(const std::string& artifact_dir);
MotionPlanningManifest load_motion_planning_manifest(
    const std::string& artifact_dir);
std::vector<uint8_t> load_tensor_data(const std::string& artifact_dir,
                                      const TensorManifest& tensor,
                                      const std::string& name);

}  // namespace curobo_robot_runtime
