// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "trajectory_optimizer_runtime.h"

#include "attachment_runtime.h"
#include "cuda_check.h"
#include "curobo_robot_runtime/scene.h"
#include "rollout_runtime.h"
#include "rollout_streams.h"
#include "scene_runtime.h"
#include "trajectory_timeopt_cuda.h"
#include "warp_abi.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace curobo::robot::detail {
namespace {

size_t checked_product(size_t left, size_t right, const char* description) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    throw std::invalid_argument(std::string(description) + " size overflow");
  }
  return left * right;
}

template <typename ValueType>
std::vector<uint8_t> as_bytes(const std::vector<ValueType>& values) {
  std::vector<uint8_t> result(values.size() * sizeof(ValueType));
  if (!result.empty()) {
    std::memcpy(result.data(), values.data(), result.size());
  }
  return result;
}

unsigned int grid_for(size_t count, int block_size) {
  if (count == 0 || block_size <= 0) {
    throw std::invalid_argument("kernel launch dimensions must be positive");
  }
  const auto blocks =
      (count + static_cast<size_t>(block_size) - 1) /
      static_cast<size_t>(block_size);
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument("kernel grid exceeds the CUDA launch range");
  }
  return static_cast<unsigned int>(blocks);
}

void load_kernel(const std::string& artifact_directory,
                 const curobo_robot_runtime::IkKernelManifest& manifest,
                 curobo_robot_runtime::KernelModule* kernel) {
  if (manifest.cubin.empty()) {
    return;
  }
  kernel->load(
      (std::filesystem::path(artifact_directory) / manifest.cubin).string(),
      manifest.symbol);
}

WarpArray array_1d(CUdeviceptr pointer, int count, int element_bytes) {
  return warp_array_1d(static_cast<uint64_t>(pointer), count, element_bytes);
}

WarpArray array_2d(CUdeviceptr pointer, int rows, int columns,
                   int element_bytes) {
  return warp_array_2d(static_cast<uint64_t>(pointer), rows, columns,
                       element_bytes);
}

CuboidDataWarp make_cuboid_data(const ObstacleDeviceView& view) {
  return CuboidDataWarp{
      array_2d(view.params, view.capacity, view.param_stride, sizeof(float)),
      array_2d(view.world_bounds, view.capacity, 8, sizeof(float)),
      array_2d(view.inverse_poses, view.capacity, 8, sizeof(float)),
      array_1d(view.enabled, view.capacity, sizeof(uint8_t)),
      array_1d(view.count, 1, sizeof(int32_t)),
      view.capacity,
      1,
  };
}

CapsuleDataWarp make_capsule_data(const ObstacleDeviceView& view) {
  return CapsuleDataWarp{
      array_2d(view.params, view.capacity, view.param_stride, sizeof(float)),
      array_2d(view.world_bounds, view.capacity, 8, sizeof(float)),
      array_2d(view.inverse_poses, view.capacity, 8, sizeof(float)),
      array_1d(view.enabled, view.capacity, sizeof(uint8_t)),
      array_1d(view.count, 1, sizeof(int32_t)),
      view.capacity,
      1,
  };
}

MeshDataWarp make_mesh_data(const MeshDeviceView& view) {
  return MeshDataWarp{
      array_1d(view.mesh_ids, view.capacity, sizeof(uint64_t)),
      array_2d(view.dimensions, view.capacity, 4, sizeof(float)),
      array_2d(view.world_bounds, view.capacity, 8, sizeof(float)),
      array_2d(view.inverse_poses, view.capacity, 8, sizeof(float)),
      array_1d(view.enabled, view.capacity, sizeof(uint8_t)),
      array_1d(view.count, 1, sizeof(int32_t)),
      view.capacity,
      1,
      view.max_distance,
  };
}

VoxelDataWarp make_voxel_data(const VoxelDeviceView& view) {
  return VoxelDataWarp{
      array_2d(view.params, view.capacity, 4, sizeof(float)),
      array_2d(view.dimensions, view.capacity, 4, sizeof(float)),
      array_2d(view.inverse_poses, view.capacity, 8, sizeof(float)),
      array_1d(view.enabled, view.capacity, sizeof(uint8_t)),
      array_1d(view.features,
               view.capacity * view.max_voxels_per_grid, sizeof(uint16_t)),
      array_1d(view.count, 1, sizeof(int32_t)),
      view.max_voxels_per_grid,
      view.capacity,
      1,
      view.max_distance,
  };
}

template <typename ObstacleData>
void launch_swept_collision(
    const curobo_robot_runtime::KernelModule& kernel,
    const curobo_robot_runtime::IkKernelManifest& manifest,
    const ObstacleData& obstacle_data, CUdeviceptr obstacle_layers,
    CUdeviceptr sphere_collision_mask, CUdeviceptr robot_spheres,
    CUdeviceptr weight, CUdeviceptr activation_distance, CUdeviceptr speed_dt,
    CUdeviceptr scene_indices, CUdeviceptr collision_cost,
    CUdeviceptr collision_gradient, int trajectories, int horizon,
    int num_spheres, bool overwrite, CUstream stream) {
  const int states = trajectories * horizon;
  const auto bounds = warp_tiled_bounds_2d(states, manifest.block_dim);
  auto spheres =
      array_1d(robot_spheres, states * num_spheres, 4 * sizeof(float));
  auto weight_array = array_1d(weight, 1, sizeof(float));
  auto activation_array = array_1d(activation_distance, 1, sizeof(float));
  auto dt_array = array_1d(speed_dt, trajectories, sizeof(float));
  auto scene_index_array = array_1d(scene_indices, trajectories, sizeof(int32_t));
  auto cost_array = array_1d(collision_cost, states, sizeof(float));
  auto gradient_array =
      array_1d(collision_gradient, states * num_spheres * 4, sizeof(float));
  uint8_t use_multiple_scenes = 0;
  uint8_t overwrite_output = overwrite ? 1U : 0U;
  uint8_t sum_distance = 1U;
  auto obstacle_layer_array =
      array_1d(obstacle_layers, obstacle_data.capacity, sizeof(uint32_t));
  auto sphere_mask_array =
      array_2d(sphere_collision_mask, 1, num_spheres, sizeof(uint32_t));
  auto mutable_obstacle_data = obstacle_data;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &mutable_obstacle_data,
      &spheres,
      &weight_array,
      &activation_array,
      &dt_array,
      &scene_index_array,
      &cost_array,
      &gradient_array,
      &use_multiple_scenes,
      &overwrite_output,
      &sum_distance,
      &obstacle_layer_array,
      &sphere_mask_array,
  };
  kernel.launch(static_cast<unsigned int>(states),
                static_cast<unsigned int>(manifest.block_dim),
                static_cast<unsigned int>(manifest.shared_memory), stream,
                arguments);
}

template <typename ObstacleData>
void launch_discrete_collision(
    const curobo_robot_runtime::KernelModule& kernel,
    const curobo_robot_runtime::IkKernelManifest& manifest,
    const ObstacleData& obstacle_data, CUdeviceptr obstacle_layers,
    CUdeviceptr sphere_collision_mask, CUdeviceptr robot_spheres,
    CUdeviceptr weight, CUdeviceptr activation_distance,
    CUdeviceptr scene_indices, CUdeviceptr collision_cost,
    CUdeviceptr collision_gradient, int trajectories, int horizon,
    int num_spheres, bool overwrite, CUstream stream) {
  const int states = trajectories * horizon;
  const auto bounds = warp_tiled_bounds_2d(states, manifest.block_dim);
  auto spheres =
      array_1d(robot_spheres, states * num_spheres, 4 * sizeof(float));
  auto weight_array = array_1d(weight, 1, sizeof(float));
  auto activation_array = array_1d(activation_distance, 1, sizeof(float));
  auto scene_index_array = array_1d(scene_indices, trajectories, sizeof(int32_t));
  auto cost_array = array_1d(collision_cost, states, sizeof(float));
  auto gradient_array =
      array_1d(collision_gradient, states * num_spheres * 4, sizeof(float));
  uint8_t use_multiple_scenes = 0;
  uint8_t overwrite_output = overwrite ? 1U : 0U;
  uint8_t sum_distance = 1U;
  auto obstacle_layer_array =
      array_1d(obstacle_layers, obstacle_data.capacity, sizeof(uint32_t));
  auto sphere_mask_array =
      array_2d(sphere_collision_mask, 1, num_spheres, sizeof(uint32_t));
  auto mutable_obstacle_data = obstacle_data;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &mutable_obstacle_data,
      &spheres,
      &weight_array,
      &activation_array,
      &scene_index_array,
      &cost_array,
      &gradient_array,
      &use_multiple_scenes,
      &overwrite_output,
      &sum_distance,
      &obstacle_layer_array,
      &sphere_mask_array,
  };
  kernel.launch(static_cast<unsigned int>(states),
                static_cast<unsigned int>(manifest.block_dim),
                static_cast<unsigned int>(manifest.shared_memory), stream,
                arguments);
}

}  // namespace

TrajectoryOptimizerRuntime::TrajectoryOptimizerRuntime(
    std::string artifact_directory, const TrajectoryOptimizationConfig& config,
    const Scene* scene, std::shared_ptr<RobotCollisionModel> collision)
    : artifact_directory_(std::move(artifact_directory)),
      config_(config),
      scene_(scene == nullptr ? nullptr : scene->runtime_.get()),
      robot_(artifact_directory_, config.device_ordinal),
      manifest_(curobo_robot_runtime::load_trajectory_optimization_manifest(
          artifact_directory_)) {
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  const auto rollout_manifest =
      std::filesystem::path(artifact_directory_) / "robot_rollout_manifest.json";
  if (std::filesystem::exists(rollout_manifest)) {
    rollout_ = std::make_unique<RolloutRuntime>(
        artifact_directory_, config_.device_ordinal,
        robot_.manifest(), RolloutProfile::TRAJOPT);
  } else if (config_.check_effort_limits || config_.enable_timeopt) {
    throw std::runtime_error(
        "TrajOpt effort limits require a robot rollout artifact");
  }
  validate_config();
  validate_scene();
  collision_ = std::move(collision);
  load_constants();
  load_kernels();
  allocate_workspace();
  rollout_streams_ = std::make_unique<RolloutStreams>();
  if (config_.enable_timeopt) {
    TrajectoryTimeOptConfig timeopt_config;
    timeopt_config.device_ordinal = config_.device_ordinal;
    timeopt_config.num_seeds = config_.num_seeds;
    timeopt_config.num_knots = manifest_.num_knots;
    timeopt_config.sample_count = manifest_.interpolation_buffer_size;
    if (timeopt_config.sample_count < 2) {
      throw std::invalid_argument("timeopt requires at least two interpolation samples");
    }
    timeopt_.reset(new TrajectoryTimeOpt(artifact_directory_, timeopt_config, *rollout_));
    const size_t batch = config_.num_seeds, dof = manifest_.num_dof;
    const size_t capacity = manifest_.interpolation_buffer_size;
    timeopt_boundaries_.allocate(4 * batch * dof * sizeof(float));
    timeopt_knot_dt_.allocate(batch * sizeof(float));
    timeopt_times_.allocate(batch * capacity * sizeof(float));
    timeopt_dt_.allocate(batch * sizeof(float));
    timeopt_counts_.allocate(batch * sizeof(int32_t));
    timeopt_states_.allocate(4 * capacity * dof * sizeof(float));
    timeopt_metadata_.allocate(2 * sizeof(float));
    timeopt_count_.allocate(sizeof(int32_t));
    timeopt_count_.clear();
    timeopt_timing_controls_.allocate(TIMEOPT_CONTROL_COUNT * sizeof(float));
    timeopt_timing_controls_.clear();
    timeopt_spatial_controls_.allocate((manifest_.num_knots + 6) * dof * sizeof(float));
    timeopt_spatial_controls_.clear();
    timeopt_scaled_states_.allocate(3 * batch * capacity * dof * sizeof(float));
    execution_speed_scale_.allocate(sizeof(float));
    set_execution_speed_scale(config_.execution_speed_scale);
  }
}

TrajectoryOptimizerRuntime::~TrajectoryOptimizerRuntime() {
  cudaSetDevice(config_.device_ordinal);
}

CuroboRobotInfo TrajectoryOptimizerRuntime::robot_info() const {
  return robot_.info();
}

TrajectoryOptimizationInfo TrajectoryOptimizerRuntime::info() const {
  return TrajectoryOptimizationInfo{
      manifest_.num_dof,
      manifest_.num_knots,
      manifest_.trajectory_steps,
      manifest_.num_tool_frames,
  };
}

int TrajectoryOptimizerRuntime::attachment_sphere_capacity(
    const std::string& link_name) const {
  return collision_->attachment->capacity(link_name);
}

void TrajectoryOptimizerRuntime::attach_object(
    const SphereAttachment& attachment) {
  collision_->attachment->attach(attachment);
}

void TrajectoryOptimizerRuntime::detach_object(const std::string& link_name) {
  collision_->attachment->detach(link_name);
}

void TrajectoryOptimizerRuntime::update_link_inertial(
    const LinkInertialUpdate& update) {
  if (rollout_ == nullptr) {
    throw std::runtime_error("TrajOpt artifact has no inverse-dynamics model");
  }
  rollout_->update_link_inertial(update);
}

void TrajectoryOptimizerRuntime::update_links_inertial(
    const LinkInertialUpdates& updates) {
  if (rollout_ == nullptr) {
    throw std::runtime_error("TrajOpt artifact has no inverse-dynamics model");
  }
  rollout_->update_links_inertial(updates);
}

void TrajectoryOptimizerRuntime::set_link_collision_mask(
    const std::string& link, uint32_t layers) {
  collision_->attachment->set_collision_mask(link, layers,
                                             collision_->sphere_collision_mask);
}

void TrajectoryOptimizerRuntime::set_link_collision_enabled(
    const std::string& link_name, bool enabled) {
  collision_->attachment->set_collision_enabled(link_name, enabled);
}

void TrajectoryOptimizerRuntime::validate_config() const {
  const auto robot_info_value = robot_.info();
  if (manifest_.arch != robot_.manifest().arch ||
      manifest_.num_dof != robot_info_value.num_dof ||
      manifest_.num_links != robot_info_value.num_links ||
      manifest_.num_tool_frames != robot_info_value.num_tool_frames) {
    throw std::runtime_error(
        "robot TrajOpt artifact does not match its kinematics artifact");
  }
  const auto model = robot_.model_device_view();
  if (model.joint_link_influence_words == 0 || model.joint_links_data == 0 ||
      model.joint_links_offsets == 0 || model.link_masses_com == 0) {
    throw std::runtime_error(
        "TrajOpt artifact requires reverse-kinematics topology tensors");
  }
  if (config_.num_seeds <= 0 || config_.num_seeds > manifest_.max_seeds ||
      config_.max_iterations <= 0 || config_.refinement_iterations <= 0 ||
      config_.finetune_attempts < 0) {
    throw std::invalid_argument(
        "TrajOpt seeds or iteration counts are outside the artifact capacity");
  }
  for (const float value : {
           config_.trajectory_dt,
           config_.initial_dt_scale,
           config_.finetune_dt_scale,
           config_.collision_activation_distance,
           config_.constraint_tolerance,
           config_.position_tolerance,
           config_.orientation_tolerance,
       }) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("TrajOpt scalar configuration must be finite");
    }
  }
  if (config_.trajectory_dt <= 0.0F ||
      config_.trajectory_dt < manifest_.minimum_dt ||
      config_.trajectory_dt > manifest_.maximum_dt ||
      config_.initial_dt_scale <= 0.0F ||
      config_.finetune_dt_scale <= 0.0F ||
      config_.finetune_dt_scale > 1.0F ||
      config_.collision_activation_distance < 0.0F ||
      config_.constraint_tolerance < 0.0F ||
      config_.position_tolerance <= 0.0F ||
      config_.orientation_tolerance <= 0.0F) {
    throw std::invalid_argument(
        "TrajOpt scalar configuration is outside its valid range");
  }
  const int total_knots =
      manifest_.num_knots + manifest_.spline_degree + 1;
  const int sparse_interpolation_steps =
      (manifest_.trajectory_steps - 1) / total_knots;
  const int maximum_steps_per_waypoint = static_cast<int>(std::ceil(
                                             manifest_.maximum_dt *
                                             sparse_interpolation_steps /
                                             manifest_.interpolation_dt)) +
                                         1;
  const int required_validation_steps =
      total_knots * maximum_steps_per_waypoint + 1;
  if (sparse_interpolation_steps <= 0 ||
      (manifest_.trajectory_steps - 1) % total_knots != 0 ||
      manifest_.interpolation_buffer_size < required_validation_steps) {
    throw std::runtime_error(
        "TrajOpt interpolation buffer is too small for the exported dt range");
  }
}

void TrajectoryOptimizerRuntime::validate_scene() const {
  if (scene_ == nullptr) {
    return;
  }
  const auto& scene_config = scene_->config();
  if (scene_config.device_ordinal != config_.device_ordinal) {
    throw std::invalid_argument("TrajOpt and Scene must use the same CUDA device");
  }
  if (scene_config.max_cuboids > manifest_.max_cuboids ||
      scene_config.max_capsules > manifest_.max_capsules ||
      scene_config.max_cylinders > manifest_.max_cylinders ||
      scene_config.max_meshes > manifest_.max_meshes ||
      scene_config.max_voxel_grids > manifest_.max_voxel_grids) {
    throw std::invalid_argument(
        "Scene capacity exceeds the TrajOpt artifact capacity");
  }
}

void TrajectoryOptimizerRuntime::load_constants() {
  const auto load = [&](const curobo_robot_runtime::TensorManifest& tensor,
                        const char* name,
                        curobo_robot_runtime::DeviceBuffer* destination) {
    destination->upload(curobo_robot_runtime::load_tensor_data(
        artifact_directory_, tensor, name));
  };
  load(manifest_.position_bounds, "position_bounds", &position_bounds_);
  load(manifest_.velocity_bounds, "velocity_bounds", &velocity_bounds_);
  load(manifest_.acceleration_bounds, "acceleration_bounds",
       &acceleration_bounds_);
  load(manifest_.jerk_bounds, "jerk_bounds", &jerk_bounds_);
  load(manifest_.state_bound_weight, "state_bound_weight",
       &state_bound_weight_);
  load(manifest_.state_activation_distance, "state_activation_distance",
       &state_activation_distance_);
  load(manifest_.state_regularization_weight, "state_regularization_weight",
       &state_regularization_weight_);
  load(manifest_.cspace_target_weight, "cspace_target_weight",
       &cspace_target_weight_);
  load(manifest_.cspace_non_terminal_weight, "cspace_non_terminal_weight",
       &cspace_non_terminal_weight_);
  load(manifest_.cspace_target_dof_weight, "cspace_target_dof_weight",
       &cspace_target_dof_weight_);
  load(manifest_.tool_pose_weight, "tool_pose_weight", &tool_pose_weight_);
  load(manifest_.world_collision_weight, "world_collision_weight",
       &world_collision_weight_);
  load(manifest_.self_collision_weight, "self_collision_weight",
       &self_collision_weight_);
  load(manifest_.metrics_world_collision_weight, "metrics_world_collision_weight",
       &metrics_world_collision_weight_);
  load(manifest_.metrics_self_collision_weight, "metrics_self_collision_weight",
       &metrics_self_collision_weight_);
  load(manifest_.validation_self_collision_weight,
       "validation_self_collision_weight", &validation_self_collision_weight_);
  load(manifest_.line_search_scale, "line_search_scale",
       &line_search_scale_);
  load(manifest_.optimization_step_max, "optimization_step_max",
       &optimization_step_max_);
  const auto link_spheres = curobo_robot_runtime::load_tensor_data(
      artifact_directory_, manifest_.link_spheres, "link_spheres");
  const auto link_sphere_map = curobo_robot_runtime::load_tensor_data(
      artifact_directory_, manifest_.link_sphere_map, "link_sphere_map");
  if (collision_ == nullptr) {
    collision_ = std::make_shared<RobotCollisionModel>(
        config_.device_ordinal, robot_.manifest().link_names, link_spheres,
        link_sphere_map,
        curobo_robot_runtime::load_tensor_data(artifact_directory_,
                                               manifest_.sphere_collision_mask,
                                               "sphere_collision_mask"));
  } else if (collision_->device_ordinal != config_.device_ordinal ||
             collision_->link_spheres.bytes() != link_spheres.size() ||
             collision_->link_sphere_map.bytes() != link_sphere_map.size()) {
    throw std::invalid_argument(
        "shared collision model does not match solver geometry");
  }
  if (manifest_.num_self_collision_pairs > 0) {
    load(manifest_.self_collision_padding, "self_collision_padding",
         &self_collision_padding_);
    load(manifest_.self_collision_pairs, "self_collision_pairs",
         &self_collision_pairs_);
  }
  collision_activation_distance_.upload(
      as_bytes(std::vector<float>{config_.collision_activation_distance}));
  metrics_collision_activation_distance_.upload(
      as_bytes(std::vector<float>{0.0F}));
  interpolation_dt_.upload(
      as_bytes(std::vector<float>{manifest_.interpolation_dt}));
  const int candidates = manifest_.max_seeds * manifest_.num_line_search;
  trajectory_dt_.allocate(
      static_cast<size_t>(manifest_.max_seeds) * sizeof(float));
  state_dt_.allocate(static_cast<size_t>(candidates) * sizeof(float));
  zero_joint_state_.upload(as_bytes(std::vector<float>(
      static_cast<size_t>(manifest_.max_seeds) * manifest_.num_dof, 0.0F)));
  std::vector<int32_t> seed_goal_indices(
      static_cast<size_t>(manifest_.max_seeds));
  for (int seed = 0; seed < manifest_.max_seeds; ++seed) {
    seed_goal_indices.at(static_cast<size_t>(seed)) = seed;
  }
  seed_goal_indices_.upload(as_bytes(seed_goal_indices));
  std::vector<int32_t> candidate_goal_indices(
      static_cast<size_t>(candidates));
  for (int candidate = 0; candidate < candidates; ++candidate) {
    candidate_goal_indices.at(static_cast<size_t>(candidate)) =
        candidate / manifest_.num_line_search;
  }
  start_indices_.upload(as_bytes(candidate_goal_indices));
  candidate_goal_indices_.upload(as_bytes(candidate_goal_indices));
  implicit_goal_.upload(as_bytes(std::vector<uint8_t>(
      static_cast<size_t>(manifest_.max_seeds), 1U)));
  scene_indices_.upload(
      as_bytes(std::vector<int32_t>(static_cast<size_t>(candidates), 0)));
}

void TrajectoryOptimizerRuntime::load_kernels() {
  kinematics_kernel_.load(
      (std::filesystem::path(artifact_directory_) / manifest_.kinematics.cubin)
          .string(),
      manifest_.kinematics.symbol);
  load_kernel(artifact_directory_, manifest_.kinematics_backward,
              &kinematics_backward_kernel_);
  load_kernel(artifact_directory_, manifest_.bspline_forward,
              &bspline_forward_kernel_);
  load_kernel(artifact_directory_, manifest_.bspline_single_dt,
              &bspline_single_dt_kernel_);
  load_kernel(artifact_directory_, manifest_.bspline_backward,
              &bspline_backward_kernel_);
  load_kernel(artifact_directory_, manifest_.time_scaled_dt,
              &time_scaled_dt_kernel_);
  load_kernel(artifact_directory_, manifest_.cspace, &cspace_kernel_);
  load_kernel(artifact_directory_, manifest_.pose, &pose_kernel_);
  load_kernel(artifact_directory_, manifest_.collision_cuboid,
              &collision_cuboid_kernel_);
  load_kernel(artifact_directory_, manifest_.collision_capsule,
              &collision_capsule_kernel_);
  load_kernel(artifact_directory_, manifest_.collision_cylinder,
              &collision_cylinder_kernel_);
  load_kernel(artifact_directory_, manifest_.collision_mesh,
              &collision_mesh_kernel_);
  load_kernel(artifact_directory_, manifest_.collision_voxel,
              &collision_voxel_kernel_);
  load_kernel(artifact_directory_, manifest_.metrics_collision_cuboid,
              &metrics_collision_cuboid_kernel_);
  load_kernel(artifact_directory_, manifest_.metrics_collision_capsule,
              &metrics_collision_capsule_kernel_);
  load_kernel(artifact_directory_, manifest_.metrics_collision_cylinder,
              &metrics_collision_cylinder_kernel_);
  load_kernel(artifact_directory_, manifest_.metrics_collision_mesh,
              &metrics_collision_mesh_kernel_);
  load_kernel(artifact_directory_, manifest_.metrics_collision_voxel,
              &metrics_collision_voxel_kernel_);
  load_kernel(artifact_directory_, manifest_.validation_collision_cuboid,
              &validation_collision_cuboid_kernel_);
  load_kernel(artifact_directory_, manifest_.validation_collision_capsule,
              &validation_collision_capsule_kernel_);
  load_kernel(artifact_directory_, manifest_.validation_collision_cylinder,
              &validation_collision_cylinder_kernel_);
  load_kernel(artifact_directory_, manifest_.validation_collision_mesh,
              &validation_collision_mesh_kernel_);
  load_kernel(artifact_directory_, manifest_.validation_collision_voxel,
              &validation_collision_voxel_kernel_);
  load_kernel(artifact_directory_, manifest_.self_collision,
              &self_collision_kernel_);
  load_kernel(artifact_directory_, manifest_.initialize_actions,
              &initialize_actions_kernel_);
  load_kernel(artifact_directory_, manifest_.prepare_pose_criteria,
              &prepare_pose_criteria_kernel_);
  load_kernel(artifact_directory_, manifest_.expand_trajectory_dt,
              &expand_trajectory_dt_kernel_);
  load_kernel(artifact_directory_, manifest_.reduce_required_dt,
              &reduce_required_dt_kernel_);
  load_kernel(artifact_directory_, manifest_.calculate_interpolation_horizon,
              &calculate_interpolation_horizon_kernel_);
  load_kernel(artifact_directory_, manifest_.validate_interpolated,
              &validate_interpolated_kernel_);
  load_kernel(artifact_directory_, manifest_.prepare_initial_state,
              &prepare_initial_state_kernel_);
  if (config_.line_search_backtracking) {
    manifest_.prepare_search_actions = manifest_.prepare_search_actions_backtracking;
    manifest_.line_search = manifest_.line_search_backtracking;
  }
  load_kernel(artifact_directory_, manifest_.prepare_search_actions,
              &prepare_search_actions_kernel_);
  load_kernel(artifact_directory_, manifest_.combine_rollout,
              &combine_rollout_kernel_);
  load_kernel(artifact_directory_, manifest_.accumulate, &accumulate_kernel_);
  load_kernel(artifact_directory_, manifest_.finalize, &finalize_kernel_);
  load_kernel(artifact_directory_, manifest_.line_search, &line_search_kernel_);
  load_kernel(artifact_directory_, manifest_.lbfgs, &lbfgs_kernel_);
  if (manifest_.self_collision.shared_memory > 48000) {
    self_collision_kernel_.set_max_dynamic_shared_memory(
        manifest_.self_collision.shared_memory);
  }
  if (manifest_.lbfgs.shared_memory > 48000) {
    lbfgs_kernel_.set_max_dynamic_shared_memory(manifest_.lbfgs.shared_memory);
  }
}

void TrajectoryOptimizerRuntime::allocate_workspace() {
  const size_t max_seeds = static_cast<size_t>(manifest_.max_seeds);
  const size_t candidates = checked_product(
      max_seeds, static_cast<size_t>(manifest_.num_line_search),
      "TrajOpt candidate");
  const size_t opt_dim = checked_product(
      static_cast<size_t>(manifest_.num_knots),
      static_cast<size_t>(manifest_.num_dof), "TrajOpt action");
  const size_t states = checked_product(
      candidates, static_cast<size_t>(manifest_.trajectory_steps),
      "TrajOpt states");
  const size_t state_dof = checked_product(
      states, static_cast<size_t>(manifest_.num_dof), "TrajOpt state DOF");
  const size_t state_spheres = checked_product(
      states, static_cast<size_t>(manifest_.num_spheres),
      "TrajOpt state spheres");
  const size_t state_tools = checked_product(
      states, static_cast<size_t>(manifest_.num_tool_frames),
      "TrajOpt state tools");
  const size_t validation_states = checked_product(
      max_seeds, static_cast<size_t>(manifest_.interpolation_buffer_size),
      "TrajOpt validation states");
  const size_t validation_state_dof = checked_product(
      validation_states, static_cast<size_t>(manifest_.num_dof),
      "TrajOpt validation state DOF");
  const size_t validation_state_spheres = checked_product(
      validation_states, static_cast<size_t>(manifest_.num_spheres),
      "TrajOpt validation state spheres");
  const size_t validation_state_tools = checked_product(
      validation_states, static_cast<size_t>(manifest_.num_tool_frames),
      "TrajOpt validation state tools");

  const auto allocate_float = [](curobo_robot_runtime::DeviceBuffer* buffer,
                                 size_t count) {
    buffer->allocate(checked_product(count, sizeof(float), "float buffer"));
  };
  allocate_float(&seed_actions_, max_seeds * opt_dim);
  allocate_float(&resolved_start_positions_, max_seeds * manifest_.num_dof);
  allocate_float(&resolved_start_velocities_, max_seeds * manifest_.num_dof);
  allocate_float(&resolved_start_accelerations_,
                 max_seeds * manifest_.num_dof);
  allocate_float(&resolved_start_jerks_, max_seeds * manifest_.num_dof);
  resolved_start_jerks_.clear();
  allocate_float(&resolved_goal_positions_, max_seeds * manifest_.num_dof);
  active_seed_valid_.allocate(max_seeds * sizeof(uint8_t));
  for (auto* buffer : {
           &action_,
           &gradient_,
           &exploration_action_,
           &exploration_gradient_,
           &best_action_,
           &step_direction_,
           &reference_action_,
           &reference_gradient_,
       }) {
    allocate_float(buffer, max_seeds * opt_dim);
  }
  for (auto* buffer : {&cost_, &exploration_cost_, &best_cost_}) {
    allocate_float(buffer, max_seeds);
  }
  best_iteration_.allocate(max_seeds * sizeof(int32_t));
  current_iteration_.allocate(max_seeds * sizeof(int32_t));
  converged_.allocate(max_seeds * sizeof(uint8_t));
  history_head_.allocate(max_seeds * sizeof(int32_t));
  history_count_.allocate(max_seeds * sizeof(int32_t));
  curvature_update_valid_.allocate(max_seeds * sizeof(bool));
  allocate_float(&history_rho_, max_seeds * manifest_.history);
  allocate_float(&history_y_, max_seeds * manifest_.history * opt_dim);
  allocate_float(&history_s_, max_seeds * manifest_.history * opt_dim);
  allocate_float(&search_actions_, candidates * opt_dim);
  allocate_float(&search_gradients_, candidates * opt_dim);
  allocate_float(&search_costs_, candidates);
  exploration_indices_.allocate(
      max_seeds * manifest_.num_line_search * sizeof(int32_t));
  selected_indices_.allocate(
      max_seeds * manifest_.num_line_search * sizeof(int32_t));
  line_search_index_.allocate(max_seeds * sizeof(int32_t));
  line_search_status_.allocate(max_seeds * sizeof(int32_t));
  line_search_beta_.allocate(max_seeds * sizeof(float));

  allocate_float(&dense_position_, state_dof);
  allocate_float(&dense_velocity_, state_dof);
  allocate_float(&dense_acceleration_, state_dof);
  allocate_float(&dense_jerk_, state_dof);
  allocate_float(&dense_dt_, candidates);
  allocate_float(&required_dt_per_timestep_,
                 max_seeds * manifest_.trajectory_steps);
  interpolation_horizon_.allocate(max_seeds * sizeof(int32_t));
  allocate_float(&validation_position_, validation_state_dof);
  allocate_float(&validation_velocity_, validation_state_dof);
  allocate_float(&validation_acceleration_, validation_state_dof);
  allocate_float(&validation_jerk_, validation_state_dof);
  allocate_float(&validation_dt_, max_seeds);
  allocate_float(&validation_link_position_, validation_state_tools * 3);
  allocate_float(&validation_link_quaternion_, validation_state_tools * 4);
  allocate_float(&validation_global_cumulative_,
                 validation_states * manifest_.num_links * 12);
  allocate_float(&validation_robot_spheres_, validation_state_spheres * 4);
  allocate_float(&validation_world_collision_cost_, validation_states);
  allocate_float(&validation_world_collision_gradient_,
                 validation_state_spheres * 4);
  allocate_float(&validation_self_collision_cost_, validation_states);
  allocate_float(&validation_self_collision_gradient_,
                 validation_state_spheres * 4);
  validation_self_collision_sparse_index_.allocate(
      validation_state_spheres * sizeof(uint8_t));
  validation_violation_.allocate(max_seeds * sizeof(int32_t));
  allocate_float(&link_position_, state_tools * 3);
  allocate_float(&link_quaternion_, state_tools * 4);
  allocate_float(&global_cumulative_, states * manifest_.num_links * 12);
  allocate_float(&robot_spheres_, state_spheres * 4);
  allocate_float(&zero_link_position_gradient_, state_tools * 3);
  allocate_float(&zero_link_quaternion_gradient_, state_tools * 4);
  allocate_float(&pose_position_metric_,
                 2 * manifest_.num_tool_frames * 9);
  allocate_float(&pose_rotation_metric_,
                 2 * manifest_.num_tool_frames * 9);
  allocate_float(&pose_tolerance_, 2 * manifest_.num_tool_frames * 2);
  pose_error_frame_.allocate(
      static_cast<size_t>(manifest_.num_tool_frames) * sizeof(int32_t));
  allocate_float(&pose_cost_, state_tools * 2);
  allocate_float(&pose_position_distance_, state_tools);
  allocate_float(&pose_orientation_distance_, state_tools);
  allocate_float(&pose_position_gradient_, state_tools * 3);
  allocate_float(&pose_quaternion_gradient_, state_tools * 4);
  pose_goalset_index_.allocate(state_tools * sizeof(int32_t));
  allocate_float(&world_collision_cost_, states);
  allocate_float(&world_collision_gradient_, state_spheres * 4);
  allocate_float(&self_collision_cost_, states);
  allocate_float(&self_collision_gradient_, state_spheres * 4);
  allocate_float(&self_collision_pair_distance_, 1);
  self_collision_sparse_index_.allocate(state_spheres * sizeof(uint8_t));
  allocate_float(&collision_joint_gradient_, state_dof);
  allocate_float(&cspace_cost_, state_dof);
  allocate_float(&cspace_gradient_position_, state_dof);
  allocate_float(&cspace_gradient_velocity_, state_dof);
  allocate_float(&cspace_gradient_acceleration_, state_dof);
  allocate_float(&cspace_gradient_jerk_, state_dof);
  if (config_.check_effort_limits) {
    allocate_float(&joint_effort_, state_dof);
    allocate_float(&effort_gradient_, state_dof);
    allocate_float(&effort_gradient_position_, state_dof);
    allocate_float(&effort_gradient_velocity_, state_dof);
    allocate_float(&effort_gradient_acceleration_, state_dof);
    allocate_float(
        &rnea_forward_cache_,
        states * static_cast<size_t>(rollout_->num_links()) *
            static_cast<size_t>(rollout_->cache_floats_per_link()));
  }
  if (config_.check_effort_limits || config_.enable_timeopt) {
    allocate_float(&validation_joint_effort_, validation_state_dof);
    allocate_float(
        &validation_rnea_forward_cache_,
        validation_states * static_cast<size_t>(rollout_->num_links()) *
            static_cast<size_t>(rollout_->cache_floats_per_link()));
  }

  allocate_float(&retained_action_, max_seeds * opt_dim);
  allocate_float(&retained_position_,
                 max_seeds * manifest_.trajectory_steps * manifest_.num_dof);
  allocate_float(&retained_velocity_,
                 max_seeds * manifest_.trajectory_steps * manifest_.num_dof);
  allocate_float(&retained_acceleration_,
                 max_seeds * manifest_.trajectory_steps * manifest_.num_dof);
  allocate_float(&retained_jerk_,
                 max_seeds * manifest_.trajectory_steps * manifest_.num_dof);
  retained_success_.allocate(max_seeds * sizeof(uint8_t));
  retained_pose_valid_.allocate(max_seeds * sizeof(uint8_t));
  allocate_float(&retained_cost_, max_seeds);
  allocate_float(&retained_rank_score_, max_seeds);
  allocate_float(&retained_collision_cost_, max_seeds);
  allocate_float(&retained_dt_, max_seeds);
  finetune_enabled_.allocate(sizeof(int32_t));
  selected_seed_.allocate(sizeof(int32_t));

  zero_link_position_gradient_.clear();
  zero_link_quaternion_gradient_.clear();
  pose_cost_.clear();
  pose_position_distance_.clear();
  pose_orientation_distance_.clear();
  pose_position_gradient_.clear();
  pose_quaternion_gradient_.clear();
  world_collision_gradient_.clear();
  self_collision_gradient_.clear();
  self_collision_sparse_index_.clear();
  validation_world_collision_gradient_.clear();
  validation_self_collision_gradient_.clear();
  validation_self_collision_sparse_index_.clear();
  retained_action_.clear();
  retained_position_.clear();
  retained_velocity_.clear();
  retained_acceleration_.clear();
  retained_jerk_.clear();
  retained_success_.clear();
  retained_cost_.clear();
  retained_collision_cost_.clear();
  retained_dt_.clear();
  if (config_.check_effort_limits) {
    joint_effort_.clear();
    effort_gradient_.clear();
    effort_gradient_position_.clear();
    effort_gradient_velocity_.clear();
    effort_gradient_acceleration_.clear();
  }
}

void TrajectoryOptimizerRuntime::initialize_actions(
    const TrajectoryOptimizationRequest& request, CUstream stream) const {
  const int opt_dim = manifest_.num_knots * manifest_.num_dof;
  const auto bounds = warp_bounds_2d(config_.num_seeds, opt_dim);
  auto start = warp_array_1d(reinterpret_cast<uint64_t>(request.start_position),
                             manifest_.num_dof, sizeof(float));
  const CUdeviceptr start_velocity_pointer =
      request.start_velocity == nullptr
          ? zero_joint_state_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.start_velocity));
  auto start_velocity = array_1d(start_velocity_pointer, manifest_.num_dof,
                                 sizeof(float));
  const CUdeviceptr start_acceleration_pointer =
      request.start_acceleration == nullptr
          ? zero_joint_state_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.start_acceleration));
  auto start_acceleration = array_1d(start_acceleration_pointer,
                                     manifest_.num_dof, sizeof(float));
  auto goal = warp_array_1d(reinterpret_cast<uint64_t>(request.goal_position),
                            manifest_.num_dof, sizeof(float));
  const CUdeviceptr request_start_pointer =
      request.seed_start_positions == nullptr
          ? static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.start_position))
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.seed_start_positions));
  auto seed_starts =
      array_2d(request_start_pointer, config_.num_seeds, manifest_.num_dof,
               sizeof(float));
  const CUdeviceptr request_goal_pointer =
      request.seed_goal_positions == nullptr
          ? static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.goal_position))
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.seed_goal_positions));
  auto seed_goals =
      array_2d(request_goal_pointer, config_.num_seeds, manifest_.num_dof,
               sizeof(float));
  const CUdeviceptr request_valid_pointer =
      request.seed_valid == nullptr
          ? active_seed_valid_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.seed_valid));
  auto request_seed_valid =
      array_1d(request_valid_pointer, config_.num_seeds, sizeof(uint8_t));
  const CUdeviceptr request_seed_pointer =
      request.seed_knots == nullptr
          ? seed_actions_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.seed_knots));
  auto request_seeds =
      array_2d(request_seed_pointer, config_.num_seeds, opt_dim, sizeof(float));
  auto actions =
      array_2d(seed_actions_.ptr(), config_.num_seeds, opt_dim, sizeof(float));
  auto resolved_starts = array_2d(resolved_start_positions_.ptr(),
                                  config_.num_seeds, manifest_.num_dof,
                                  sizeof(float));
  auto resolved_start_velocities =
      array_2d(resolved_start_velocities_.ptr(), config_.num_seeds,
               manifest_.num_dof, sizeof(float));
  auto resolved_start_accelerations =
      array_2d(resolved_start_accelerations_.ptr(), config_.num_seeds,
               manifest_.num_dof, sizeof(float));
  auto resolved_goals = array_2d(resolved_goal_positions_.ptr(),
                                 config_.num_seeds, manifest_.num_dof,
                                 sizeof(float));
  auto active_seed_valid =
      array_1d(active_seed_valid_.ptr(), config_.num_seeds, sizeof(uint8_t));
  uint8_t use_request_seeds = request.seed_knots == nullptr ? 0U : 1U;
  uint8_t use_seed_starts = request.seed_start_positions == nullptr ? 0U : 1U;
  uint8_t use_seed_goals = request.seed_goal_positions == nullptr ? 0U : 1U;
  uint8_t use_seed_valid = request.seed_valid == nullptr ? 0U : 1U;
  float initial_dt = config_.trajectory_dt;
  auto trajectory_dt =
      array_1d(trajectory_dt_.ptr(), config_.num_seeds, sizeof(float));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &start,
      &start_velocity,
      &start_acceleration,
      &goal,
      &seed_starts,
      &seed_goals,
      &request_seed_valid,
      &request_seeds,
      &actions,
      &resolved_starts,
      &resolved_start_velocities,
      &resolved_start_accelerations,
      &resolved_goals,
      &active_seed_valid,
      &use_request_seeds,
      &use_seed_starts,
      &use_seed_goals,
      &use_seed_valid,
      &initial_dt,
      &trajectory_dt,
  };
  const size_t elements =
      static_cast<size_t>(config_.num_seeds) * static_cast<size_t>(opt_dim);
  initialize_actions_kernel_.launch(
      grid_for(elements, manifest_.initialize_actions.block_dim),
      static_cast<unsigned int>(manifest_.initialize_actions.block_dim),
      static_cast<unsigned int>(manifest_.initialize_actions.shared_memory),
      stream, arguments);
}

void TrajectoryOptimizerRuntime::compute_required_dt(
    CUdeviceptr knots, float scale,
    const TrajectoryOptimizationRequest& request, CUstream stream) const {
  (void)request;
  CUdeviceptr output = required_dt_per_timestep_.ptr();
  CUdeviceptr start = resolved_start_positions_.ptr();
  CUdeviceptr start_velocity = resolved_start_velocities_.ptr();
  CUdeviceptr start_acceleration = resolved_start_accelerations_.ptr();
  CUdeviceptr start_jerk = resolved_start_jerks_.ptr();
  CUdeviceptr goal = resolved_goal_positions_.ptr();
  CUdeviceptr zero = zero_joint_state_.ptr();
  CUdeviceptr start_indices = seed_goal_indices_.ptr();
  CUdeviceptr goal_indices = seed_goal_indices_.ptr();
  CUdeviceptr trajectory_dt = trajectory_dt_.ptr();
  CUdeviceptr implicit_goal = implicit_goal_.ptr();
  const auto dof_bytes =
      static_cast<CUdeviceptr>(manifest_.num_dof * sizeof(float));
  CUdeviceptr max_velocity = velocity_bounds_.ptr() + dof_bytes;
  CUdeviceptr max_acceleration = acceleration_bounds_.ptr() + dof_bytes;
  CUdeviceptr max_jerk = jerk_bounds_.ptr() + dof_bytes;
  float epsilon = 1.0E-3F;
  float minimum_dt = manifest_.minimum_dt;
  float maximum_dt = manifest_.maximum_dt;
  int batch_size = config_.num_seeds;
  int horizon = manifest_.trajectory_steps;
  int dof = manifest_.num_dof;
  int n_knots = manifest_.num_knots;
  std::vector<void*> arguments{
      &output,
      &knots,
      &start,
      &start_velocity,
      &start_acceleration,
      &start_jerk,
      &goal,
      &zero,
      &zero,
      &zero,
      &start_indices,
      &goal_indices,
      &trajectory_dt,
      &implicit_goal,
      &max_velocity,
      &max_acceleration,
      &max_jerk,
      &epsilon,
      &minimum_dt,
      &maximum_dt,
      &batch_size,
      &horizon,
      &dof,
      &n_knots,
  };
  const size_t threads = static_cast<size_t>(batch_size) *
                         static_cast<size_t>(horizon) * 32;
  time_scaled_dt_kernel_.launch(
      grid_for(threads, manifest_.time_scaled_dt.block_dim),
      static_cast<unsigned int>(manifest_.time_scaled_dt.block_dim),
      static_cast<unsigned int>(manifest_.time_scaled_dt.shared_memory), stream,
      arguments);

  const auto bounds = warp_bounds_1d(config_.num_seeds);
  auto required_dt = array_1d(required_dt_per_timestep_.ptr(),
                              config_.num_seeds * horizon, sizeof(float));
  auto output_dt =
      array_1d(trajectory_dt_.ptr(), config_.num_seeds, sizeof(float));
  std::vector<void*> reduce_arguments{
      const_cast<WarpLaunchBounds1*>(&bounds),
      &required_dt,
      &output_dt,
      &scale,
      &minimum_dt,
      &maximum_dt,
  };
  reduce_required_dt_kernel_.launch(
      grid_for(config_.num_seeds, manifest_.reduce_required_dt.block_dim),
      static_cast<unsigned int>(manifest_.reduce_required_dt.block_dim),
      static_cast<unsigned int>(manifest_.reduce_required_dt.shared_memory),
      stream, reduce_arguments);
}

void TrajectoryOptimizerRuntime::expand_trajectory_dt(
    int trajectories, CUstream stream) const {
  int repeat_count = 0;
  if (trajectories == config_.num_seeds) {
    repeat_count = 1;
  } else if (trajectories ==
             config_.num_seeds * manifest_.num_line_search) {
    repeat_count = manifest_.num_line_search;
  } else {
    throw std::invalid_argument(
        "TrajOpt trajectory count cannot be mapped to active seeds");
  }
  const auto bounds = warp_bounds_1d(trajectories);
  auto trajectory_dt =
      array_1d(trajectory_dt_.ptr(), config_.num_seeds, sizeof(float));
  auto expanded_dt = array_1d(state_dt_.ptr(), trajectories, sizeof(float));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds),
      &trajectory_dt,
      &expanded_dt,
      &repeat_count,
  };
  expand_trajectory_dt_kernel_.launch(
      grid_for(trajectories, manifest_.expand_trajectory_dt.block_dim),
      static_cast<unsigned int>(manifest_.expand_trajectory_dt.block_dim),
      static_cast<unsigned int>(manifest_.expand_trajectory_dt.shared_memory),
      stream, arguments);
}

void TrajectoryOptimizerRuntime::launch_bspline(
    CUdeviceptr knots, int trajectories,
    const TrajectoryOptimizationRequest& request, CUstream stream) const {
  (void)request;
  CUdeviceptr output_position = dense_position_.ptr();
  CUdeviceptr output_velocity = dense_velocity_.ptr();
  CUdeviceptr output_acceleration = dense_acceleration_.ptr();
  CUdeviceptr output_jerk = dense_jerk_.ptr();
  CUdeviceptr output_dt = dense_dt_.ptr();
  CUdeviceptr start = resolved_start_positions_.ptr();
  CUdeviceptr start_velocity = resolved_start_velocities_.ptr();
  CUdeviceptr start_acceleration = resolved_start_accelerations_.ptr();
  CUdeviceptr start_jerk = resolved_start_jerks_.ptr();
  CUdeviceptr goal = resolved_goal_positions_.ptr();
  CUdeviceptr zero = zero_joint_state_.ptr();
  CUdeviceptr start_indices =
      trajectories == config_.num_seeds ? seed_goal_indices_.ptr()
                                        : start_indices_.ptr();
  CUdeviceptr goal_indices =
      trajectories == config_.num_seeds ? seed_goal_indices_.ptr()
                                        : candidate_goal_indices_.ptr();
  CUdeviceptr trajectory_dt = trajectory_dt_.ptr();
  CUdeviceptr implicit_goal = implicit_goal_.ptr();
  int batch_size = trajectories;
  int horizon = manifest_.trajectory_steps;
  int dof = manifest_.num_dof;
  int n_knots = manifest_.num_knots;
  std::vector<void*> arguments{
      &output_position,
      &output_velocity,
      &output_acceleration,
      &output_jerk,
      &output_dt,
      &knots,
      &start,
      &start_velocity,
      &start_acceleration,
      &start_jerk,
      &goal,
      &zero,
      &zero,
      &zero,
      &start_indices,
      &goal_indices,
      &trajectory_dt,
      &implicit_goal,
      &batch_size,
      &horizon,
      &dof,
      &n_knots,
  };
  const size_t elements = static_cast<size_t>(trajectories) *
                          static_cast<size_t>(horizon) *
                          static_cast<size_t>(dof);
  bspline_forward_kernel_.launch(
      grid_for(elements, manifest_.bspline_forward.block_dim),
      static_cast<unsigned int>(manifest_.bspline_forward.block_dim),
      static_cast<unsigned int>(manifest_.bspline_forward.shared_memory), stream,
      arguments);
}

void TrajectoryOptimizerRuntime::launch_kinematics(int trajectories,
                                                   CUstream stream) const {
  launch_kinematics_to(
      trajectories, manifest_.trajectory_steps, dense_position_.ptr(),
      link_position_.ptr(), link_quaternion_.ptr(), robot_spheres_.ptr(),
      global_cumulative_.ptr(), stream);
}

void TrajectoryOptimizerRuntime::launch_kinematics_to(
    int trajectories, int horizon, CUdeviceptr joint_position,
    CUdeviceptr link_position, CUdeviceptr link_quaternion,
    CUdeviceptr robot_spheres, CUdeviceptr global_cumulative,
    CUstream stream) const {
  int states = trajectories * horizon;
  const auto& policy = manifest_.kinematics;
  auto model = robot_.model_device_view();
  const int shared_per_batch = manifest_.num_links * 12 * sizeof(float) * 2;
  const int batches_per_block = std::min(
      {states, policy.max_batches_per_block,
       policy.max_threads_per_block / policy.threads_per_batch,
       policy.max_dynamic_shared_memory / shared_per_batch});
  if (batches_per_block <= 0) {
    throw std::runtime_error(
        "TrajOpt kinematics launch policy produced an empty block");
  }
  const auto block =
      static_cast<unsigned int>(batches_per_block * policy.threads_per_batch);
  const auto grid = static_cast<unsigned int>(
      (states + batches_per_block - 1) / batches_per_block);
  const auto shared =
      static_cast<unsigned int>(batches_per_block * shared_per_batch);

  CUdeviceptr spheres = robot_spheres;
  CUdeviceptr null_pointer = 0;
  CUdeviceptr link_spheres = collision_->link_spheres.ptr();
  CUdeviceptr link_sphere_map = collision_->link_sphere_map.ptr();
  CUdeviceptr scene_indices = scene_indices_.ptr();
  int batch_size = states;
  int num_spheres = manifest_.num_spheres;
  int num_scenes = 1;
  int num_links = manifest_.num_links;
  int num_dof = manifest_.num_dof;
  int num_tool_frames = manifest_.num_tool_frames;
  std::vector<void*> arguments{
      &link_position,
      &link_quaternion,
      &spheres,
      &null_pointer,
      &global_cumulative,
      &joint_position,
      &model.fixed_transforms,
      &link_spheres,
      &model.link_masses_com,
      &model.joint_map_type,
      &model.joint_map,
      &model.link_map,
      &model.tool_frame_map,
      &link_sphere_map,
      &model.joint_offset_map,
      &scene_indices,
      &batch_size,
      &horizon,
      &num_spheres,
      &num_scenes,
      &num_links,
      &num_dof,
      &num_tool_frames,
  };
  kinematics_kernel_.launch(grid, block, shared, stream, arguments);
}

void TrajectoryOptimizerRuntime::launch_pose(
    int trajectories, const TrajectoryOptimizationRequest& request,
    CUstream stream) const {
  int states = trajectories * manifest_.trajectory_steps;
  const int state_tools = states * manifest_.num_tool_frames;
  if (!request.constrain_tool_path) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(pose_cost_.ptr()), 0,
        static_cast<size_t>(state_tools) * 2 * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
    return;
  }
  if (!prepare_pose_criteria_) {
    // The pose kernel returns early for zero metrics. Clear its outputs so
    // zero-weight c-space phases cannot reuse a previous Cartesian gradient.
    for (auto* output : {&pose_cost_, &pose_position_gradient_, &pose_quaternion_gradient_,
                         &pose_position_distance_, &pose_orientation_distance_}) {
      output->clear(reinterpret_cast<cudaStream_t>(stream));
    }
  }

  float path_position_tolerance =
      request.path_position_tolerance < 0.0F
          ? config_.position_tolerance
          : request.path_position_tolerance;
  float path_orientation_tolerance =
      request.path_orientation_tolerance < 0.0F
          ? config_.orientation_tolerance
          : request.path_orientation_tolerance;
  int num_tool_frames = manifest_.num_tool_frames;
  const int criteria_count = 2 * num_tool_frames;
  const auto criteria_bounds = warp_bounds_1d(criteria_count);
  auto position_metric =
      array_1d(pose_position_metric_.ptr(), criteria_count, 9 * sizeof(float));
  auto rotation_metric =
      array_1d(pose_rotation_metric_.ptr(), criteria_count, 9 * sizeof(float));
  auto pose_tolerance =
      array_1d(pose_tolerance_.ptr(), criteria_count, 2 * sizeof(float));
  auto error_frame =
      array_1d(pose_error_frame_.ptr(), num_tool_frames, sizeof(int32_t));
  float direction_x = request.free_position_direction.at(0);
  float direction_y = request.free_position_direction.at(1);
  float direction_z = request.free_position_direction.at(2);
  uint8_t direction_in_goal_frame =
      request.direction_in_goal_frame ? 1U : 0U;
  std::vector<void*> criteria_arguments{
      const_cast<WarpLaunchBounds1*>(&criteria_bounds),
      &position_metric,
      &rotation_metric,
      &pose_tolerance,
      &error_frame,
      &direction_x,
      &direction_y,
      &direction_z,
      &path_position_tolerance,
      &path_orientation_tolerance,
      &direction_in_goal_frame,
      &num_tool_frames,
  };
  if (prepare_pose_criteria_) {
    prepare_pose_criteria_kernel_.launch(
        grid_for(criteria_count, manifest_.prepare_pose_criteria.block_dim),
        static_cast<unsigned int>(manifest_.prepare_pose_criteria.block_dim),
        static_cast<unsigned int>(manifest_.prepare_pose_criteria.shared_memory),
        stream, criteria_arguments);
  }

  const auto pose_bounds = warp_bounds_1d(state_tools);
  auto current_position =
      array_1d(link_position_.ptr(), state_tools, 3 * sizeof(float));
  auto current_quaternion =
      array_1d(link_quaternion_.ptr(), state_tools, 4 * sizeof(float));
  auto goal_position = warp_array_1d(
      reinterpret_cast<uint64_t>(request.goal_tool_positions),
      num_tool_frames, 3 * sizeof(float));
  auto goal_quaternion = warp_array_1d(
      reinterpret_cast<uint64_t>(request.goal_tool_quaternions),
      num_tool_frames, 4 * sizeof(float));
  auto goal_indices =
      array_1d(scene_indices_.ptr(), trajectories, sizeof(int32_t));
  auto weight = array_1d(tool_pose_weight_.ptr(), 2, sizeof(float));
  auto cost = array_1d(pose_cost_.ptr(), state_tools * 2, sizeof(float));
  auto position_distance =
      array_1d(pose_position_distance_.ptr(), state_tools, sizeof(float));
  auto orientation_distance =
      array_1d(pose_orientation_distance_.ptr(), state_tools, sizeof(float));
  auto position_gradient = array_1d(
      pose_position_gradient_.ptr(), state_tools, 3 * sizeof(float));
  auto quaternion_gradient = array_1d(
      pose_quaternion_gradient_.ptr(), state_tools, 4 * sizeof(float));
  auto goalset_index =
      array_1d(pose_goalset_index_.ptr(), state_tools, sizeof(int32_t));
  int batch_size = trajectories;
  int horizon = manifest_.trajectory_steps;
  std::vector<void*> pose_arguments{
      const_cast<WarpLaunchBounds1*>(&pose_bounds),
      &current_position,
      &current_quaternion,
      &goal_position,
      &goal_quaternion,
      &goal_indices,
      &weight,
      &position_metric,
      &rotation_metric,
      &pose_tolerance,
      &error_frame,
      &cost,
      &position_distance,
      &orientation_distance,
      &position_gradient,
      &quaternion_gradient,
      &goalset_index,
      &batch_size,
      &horizon,
      &num_tool_frames,
  };
  pose_kernel_.launch(grid_for(state_tools, manifest_.pose.block_dim),
                      static_cast<unsigned int>(manifest_.pose.block_dim),
                      static_cast<unsigned int>(manifest_.pose.shared_memory),
                      stream, pose_arguments);
}

void TrajectoryOptimizerRuntime::launch_world_collision(
    int trajectories, CUstream stream) const {
  const int states = trajectories * manifest_.trajectory_steps;
  if (scene_ == nullptr) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_cost_.ptr()), 0,
        static_cast<size_t>(states) * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_gradient_.ptr()), 0,
        static_cast<size_t>(states) * manifest_.num_spheres * 4 * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
    return;
  }
  const auto view = scene_->device_view();
  bool overwrite = true;
  const auto launch =
      [&](int capacity, const auto& obstacle_data, CUdeviceptr layers,
          const curobo_robot_runtime::IkKernelManifest& kernel_manifest,
          const curobo_robot_runtime::KernelModule& kernel) {
        if (capacity == 0 || kernel_manifest.cubin.empty()) {
          return;
        }
        launch_swept_collision(
            kernel, kernel_manifest, obstacle_data, layers,
            collision_->sphere_collision_mask.ptr(), robot_spheres_.ptr(),
            world_collision_weight_.ptr(), collision_activation_distance_.ptr(),
            state_dt_.ptr(), scene_indices_.ptr(), world_collision_cost_.ptr(),
            world_collision_gradient_.ptr(), trajectories,
            manifest_.trajectory_steps, manifest_.num_spheres, overwrite,
            stream);
        overwrite = false;
      };
  launch(view.cuboids.capacity, make_cuboid_data(view.cuboids),
         view.cuboids.obstacle_layers, manifest_.collision_cuboid,
         collision_cuboid_kernel_);
  launch(view.capsules.capacity, make_capsule_data(view.capsules),
         view.capsules.obstacle_layers, manifest_.collision_capsule,
         collision_capsule_kernel_);
  launch(view.cylinders.capacity, make_capsule_data(view.cylinders),
         view.cylinders.obstacle_layers, manifest_.collision_cylinder,
         collision_cylinder_kernel_);
  launch(view.meshes.capacity, make_mesh_data(view.meshes),
         view.meshes.obstacle_layers, manifest_.collision_mesh,
         collision_mesh_kernel_);
  launch(view.voxel_grids.capacity, make_voxel_data(view.voxel_grids),
         view.voxel_grids.obstacle_layers, manifest_.collision_voxel,
         collision_voxel_kernel_);
  if (overwrite) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_cost_.ptr()), 0,
        static_cast<size_t>(states) * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_gradient_.ptr()), 0,
        static_cast<size_t>(states) * manifest_.num_spheres * 4 * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
  }
}

void TrajectoryOptimizerRuntime::launch_metrics_world_collision(
    int trajectories, CUstream stream) const {
  launch_discrete_world_collision_to(
      trajectories, manifest_.trajectory_steps, robot_spheres_.ptr(),
      world_collision_cost_.ptr(), world_collision_gradient_.ptr(), false,
      stream);
}

void TrajectoryOptimizerRuntime::launch_metrics_collisions(
    int trajectories, CUstream stream) const {
  auto* streams = rollout_streams_.get();
  const auto primary = reinterpret_cast<cudaStream_t>(stream);
  const auto world = streams->cuda_stream(RolloutStream::WORLD);
  const auto self = streams->cuda_stream(RolloutStream::SELF);
  streams->record(RolloutEvent::STATE_READY, primary);
  streams->wait(world, RolloutEvent::STATE_READY);
  streams->wait(self, RolloutEvent::STATE_READY);
  launch_metrics_world_collision(
      trajectories, streams->driver_stream(RolloutStream::WORLD));
  streams->record(RolloutEvent::WORLD_DONE, world);
  launch_self_collision_to(
      trajectories, manifest_.trajectory_steps, robot_spheres_.ptr(),
      self_collision_cost_.ptr(), self_collision_gradient_.ptr(),
      self_collision_sparse_index_.ptr(), metrics_self_collision_weight_.ptr(),
      streams->driver_stream(RolloutStream::SELF));
  streams->record(RolloutEvent::SELF_DONE, self);
  streams->wait(primary, RolloutEvent::WORLD_DONE);
  streams->wait(primary, RolloutEvent::SELF_DONE);
}

void TrajectoryOptimizerRuntime::launch_discrete_world_collision_to(
    int trajectories, int horizon, CUdeviceptr robot_spheres,
    CUdeviceptr collision_cost, CUdeviceptr collision_gradient,
    bool validation_kernels, CUstream stream) const {
  const int states = trajectories * horizon;
  if (scene_ == nullptr) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(collision_cost), 0,
        static_cast<size_t>(states) * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
    return;
  }
  const auto view = scene_->device_view();
  bool overwrite = true;
  const auto launch =
      [&](int capacity, const auto& obstacle_data, CUdeviceptr layers,
          const curobo_robot_runtime::IkKernelManifest& kernel_manifest,
          const curobo_robot_runtime::KernelModule& kernel) {
        if (capacity == 0 || kernel_manifest.cubin.empty()) {
          return;
        }
        launch_discrete_collision(
            kernel, kernel_manifest, obstacle_data, layers,
            collision_->sphere_collision_mask.ptr(), robot_spheres,
            metrics_world_collision_weight_.ptr(),
            metrics_collision_activation_distance_.ptr(), scene_indices_.ptr(),
            collision_cost, collision_gradient, trajectories, horizon,
            manifest_.num_spheres, overwrite, stream);
        overwrite = false;
      };
  const auto& cuboid_manifest =
      validation_kernels ? manifest_.validation_collision_cuboid
                         : manifest_.metrics_collision_cuboid;
  const auto& capsule_manifest =
      validation_kernels ? manifest_.validation_collision_capsule
                         : manifest_.metrics_collision_capsule;
  const auto& cylinder_manifest =
      validation_kernels ? manifest_.validation_collision_cylinder
                         : manifest_.metrics_collision_cylinder;
  const auto& mesh_manifest =
      validation_kernels ? manifest_.validation_collision_mesh
                         : manifest_.metrics_collision_mesh;
  const auto& voxel_manifest =
      validation_kernels ? manifest_.validation_collision_voxel
                         : manifest_.metrics_collision_voxel;
  launch(view.cuboids.capacity, make_cuboid_data(view.cuboids),
         view.cuboids.obstacle_layers, cuboid_manifest,
         validation_kernels ? validation_collision_cuboid_kernel_
                            : metrics_collision_cuboid_kernel_);
  launch(view.capsules.capacity, make_capsule_data(view.capsules),
         view.capsules.obstacle_layers, capsule_manifest,
         validation_kernels ? validation_collision_capsule_kernel_
                            : metrics_collision_capsule_kernel_);
  launch(view.cylinders.capacity, make_capsule_data(view.cylinders),
         view.cylinders.obstacle_layers, cylinder_manifest,
         validation_kernels ? validation_collision_cylinder_kernel_
                            : metrics_collision_cylinder_kernel_);
  launch(view.meshes.capacity, make_mesh_data(view.meshes),
         view.meshes.obstacle_layers, mesh_manifest,
         validation_kernels ? validation_collision_mesh_kernel_
                            : metrics_collision_mesh_kernel_);
  launch(view.voxel_grids.capacity, make_voxel_data(view.voxel_grids),
         view.voxel_grids.obstacle_layers, voxel_manifest,
         validation_kernels ? validation_collision_voxel_kernel_
                            : metrics_collision_voxel_kernel_);
  if (overwrite) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(collision_cost), 0,
        static_cast<size_t>(states) * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
  }
}

void TrajectoryOptimizerRuntime::launch_self_collision(
    int trajectories, CUstream stream) const {
  launch_self_collision_to(
      trajectories, manifest_.trajectory_steps, robot_spheres_.ptr(),
      self_collision_cost_.ptr(), self_collision_gradient_.ptr(),
      self_collision_sparse_index_.ptr(), self_collision_weight_.ptr(),
      stream);
}

void TrajectoryOptimizerRuntime::launch_self_collision_to(
    int trajectories, int horizon, CUdeviceptr robot_spheres,
    CUdeviceptr collision_cost, CUdeviceptr collision_gradient,
    CUdeviceptr sparse_index, CUdeviceptr weight, CUstream stream) const {
  const int states = trajectories * horizon;
  if (!config_.check_self_collision ||
      manifest_.num_self_collision_pairs == 0) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(collision_cost), 0,
        static_cast<size_t>(states) * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(collision_gradient), 0,
        static_cast<size_t>(states) * manifest_.num_spheres * 4 * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)));
    return;
  }
  CUdeviceptr distance = collision_cost;
  CUdeviceptr gradient = collision_gradient;
  CUdeviceptr pair_distance = self_collision_pair_distance_.ptr();
  CUdeviceptr spheres = robot_spheres;
  CUdeviceptr padding = self_collision_padding_.ptr();
  CUdeviceptr pairs = self_collision_pairs_.ptr();
  int batch_size = trajectories;
  int num_spheres = manifest_.num_spheres;
  int num_pairs = manifest_.num_self_collision_pairs;
  std::vector<void*> arguments{
      &distance,
      &gradient,
      &pair_distance,
      &sparse_index,
      &spheres,
      &padding,
      &weight,
      &pairs,
      &batch_size,
      &horizon,
      &num_spheres,
      &num_pairs,
  };
  self_collision_kernel_.launch(
      static_cast<unsigned int>(states),
      static_cast<unsigned int>(manifest_.self_collision.block_dim),
      static_cast<unsigned int>(manifest_.self_collision.shared_memory), stream,
      arguments);
}

void TrajectoryOptimizerRuntime::launch_kinematics_backward(
    int trajectories, const TrajectoryOptimizationRequest& request,
    CUstream stream) const {
  const int states = trajectories * manifest_.trajectory_steps;
  auto model = robot_.model_device_view();
  CUdeviceptr output = collision_joint_gradient_.ptr();
  CUdeviceptr position_gradient = request.constrain_tool_path
                                      ? pose_position_gradient_.ptr()
                                      : zero_link_position_gradient_.ptr();
  CUdeviceptr quaternion_gradient = request.constrain_tool_path
                                        ? pose_quaternion_gradient_.ptr()
                                        : zero_link_quaternion_gradient_.ptr();
  CUdeviceptr sphere_gradient = world_collision_gradient_.ptr();
  CUdeviceptr sphere_gradient_addend = self_collision_gradient_.ptr();
  CUdeviceptr null_pointer = 0;
  CUdeviceptr global_cumulative = global_cumulative_.ptr();
  CUdeviceptr link_spheres = collision_->link_spheres.ptr();
  CUdeviceptr link_sphere_map = collision_->link_sphere_map.ptr();
  CUdeviceptr scene_indices = scene_indices_.ptr();
  int batch_size = states;
  int horizon = manifest_.trajectory_steps;
  int num_scenes = 1;
  std::vector<void*> arguments{
      &output,
      &position_gradient,
      &quaternion_gradient,
      &sphere_gradient,
      &sphere_gradient_addend,
      &null_pointer,
      &null_pointer,
      &global_cumulative,
      &link_spheres,
      &model.link_masses_com,
      &model.joint_map_type,
      &model.joint_link_influence_words,
      &model.joint_links_data,
      &model.joint_links_offsets,
      &model.joint_offset_map,
      &model.tool_frame_map,
      &link_sphere_map,
      &scene_indices,
      &batch_size,
      &horizon,
      &num_scenes,
  };
  kinematics_backward_kernel_.launch(
      static_cast<unsigned int>(states),
      static_cast<unsigned int>(manifest_.kinematics_backward.block_dim),
      static_cast<unsigned int>(manifest_.kinematics_backward.shared_memory),
      stream, arguments);
}

void TrajectoryOptimizerRuntime::launch_cspace(int trajectories,
                                               CUstream stream) const {
  const int states = trajectories * manifest_.trajectory_steps;
  const int values = states * manifest_.num_dof;
  const auto bounds = warp_bounds_1d(values);
  auto position = array_1d(dense_position_.ptr(), values, sizeof(float));
  auto velocity = array_1d(dense_velocity_.ptr(), values, sizeof(float));
  auto acceleration = array_1d(dense_acceleration_.ptr(), values, sizeof(float));
  auto jerk = array_1d(dense_jerk_.ptr(), values, sizeof(float));
  auto effort = array_1d(joint_effort_.ptr(), values, sizeof(float));
  auto dt = array_1d(state_dt_.ptr(), trajectories, sizeof(float));
  auto target = array_1d(resolved_goal_positions_.ptr(),
                         config_.num_seeds * manifest_.num_dof,
                         sizeof(float));
  const auto target_index_pointer =
      trajectories == config_.num_seeds ? seed_goal_indices_.ptr()
                                        : candidate_goal_indices_.ptr();
  auto target_indices =
      array_1d(target_index_pointer, trajectories, sizeof(int32_t));
  auto position_bounds =
      array_1d(position_bounds_.ptr(), 2 * manifest_.num_dof, sizeof(float));
  auto velocity_bounds =
      array_1d(velocity_bounds_.ptr(), 2 * manifest_.num_dof, sizeof(float));
  auto acceleration_bounds =
      array_1d(acceleration_bounds_.ptr(), 2 * manifest_.num_dof, sizeof(float));
  auto jerk_bounds =
      array_1d(jerk_bounds_.ptr(), 2 * manifest_.num_dof, sizeof(float));
  const auto effort_bounds_pointer =
      rollout_ == nullptr ? position_bounds_.ptr() : rollout_->effort_bounds();
  auto effort_bounds = array_1d(effort_bounds_pointer,
                                2 * manifest_.num_dof, sizeof(float));
  auto bound_weight = array_1d(state_bound_weight_.ptr(), 5, sizeof(float));
  auto activation =
      array_1d(state_activation_distance_.ptr(), 5, sizeof(float));
  auto regularization =
      array_1d(state_regularization_weight_.ptr(), 5, sizeof(float));
  auto target_weight = array_1d(cspace_target_weight_.ptr(), 1, sizeof(float));
  auto non_terminal_weight =
      array_1d(cspace_non_terminal_weight_.ptr(), 1, sizeof(float));
  auto target_dof_weight = array_1d(cspace_target_dof_weight_.ptr(),
                                    manifest_.num_dof, sizeof(float));
  auto output_cost = array_1d(cspace_cost_.ptr(), values, sizeof(float));
  auto gradient_position =
      array_1d(cspace_gradient_position_.ptr(), values, sizeof(float));
  auto gradient_velocity =
      array_1d(cspace_gradient_velocity_.ptr(), values, sizeof(float));
  auto gradient_acceleration =
      array_1d(cspace_gradient_acceleration_.ptr(), values, sizeof(float));
  auto gradient_jerk =
      array_1d(cspace_gradient_jerk_.ptr(), values, sizeof(float));
  auto gradient_effort =
      array_1d(effort_gradient_.ptr(), values, sizeof(float));
  bool compute_effort = config_.check_effort_limits;
  uint8_t write_state_gradient = 1U;
  uint8_t write_effort_gradient = config_.check_effort_limits ? 1U : 0U;
  int batch_size = trajectories;
  int horizon = manifest_.trajectory_steps;
  int dof = manifest_.num_dof;
  bool retime_weights = retime_cspace_weights_;
  bool retime_regularization = true;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds),
      &position,
      &velocity,
      &acceleration,
      &jerk,
      &effort,
      &dt,
      &target,
      &target_indices,
      &position_bounds,
      &velocity_bounds,
      &acceleration_bounds,
      &jerk_bounds,
      &effort_bounds,
      &bound_weight,
      &activation,
      &regularization,
      &target_weight,
      &non_terminal_weight,
      &target_dof_weight,
      &output_cost,
      &gradient_position,
      &gradient_velocity,
      &gradient_acceleration,
      &gradient_jerk,
      &gradient_effort,
      &compute_effort,
      &write_state_gradient,
      &write_effort_gradient,
      &batch_size,
      &horizon,
      &dof,
      &retime_weights,
      &retime_regularization,
  };
  cspace_kernel_.launch(grid_for(values, manifest_.cspace.block_dim),
                        static_cast<unsigned int>(manifest_.cspace.block_dim),
                        static_cast<unsigned int>(manifest_.cspace.shared_memory),
                        stream, arguments);
}

void TrajectoryOptimizerRuntime::combine_rollout(
    int trajectories, CUdeviceptr costs, bool pose_constraint_active,
    CUstream stream) const {
  constexpr int block_size = 128;
  const int state_values =
      trajectories * manifest_.trajectory_steps * manifest_.num_dof;
  const int states = trajectories * manifest_.trajectory_steps;
  const auto bounds = warp_bounds_2d(trajectories, block_size);
  auto cspace_cost =
      array_1d(cspace_cost_.ptr(), state_values, sizeof(float));
  auto gradient_position =
      array_1d(cspace_gradient_position_.ptr(), state_values, sizeof(float));
  auto gradient_velocity =
      array_1d(cspace_gradient_velocity_.ptr(), state_values, sizeof(float));
  auto gradient_acceleration =
      array_1d(cspace_gradient_acceleration_.ptr(), state_values, sizeof(float));
  auto gradient_jerk =
      array_1d(cspace_gradient_jerk_.ptr(), state_values, sizeof(float));
  auto effort_gradient_position =
      array_1d(effort_gradient_position_.ptr(), state_values, sizeof(float));
  auto effort_gradient_velocity =
      array_1d(effort_gradient_velocity_.ptr(), state_values, sizeof(float));
  auto effort_gradient_acceleration =
      array_1d(effort_gradient_acceleration_.ptr(), state_values, sizeof(float));
  auto collision_gradient =
      array_1d(collision_joint_gradient_.ptr(), state_values, sizeof(float));
  auto world_cost =
      array_1d(world_collision_cost_.ptr(), states, sizeof(float));
  auto self_cost =
      array_1d(self_collision_cost_.ptr(), states, sizeof(float));
  auto pose_cost = array_1d(
      pose_cost_.ptr(), states * manifest_.num_tool_frames * 2, sizeof(float));
  auto output_cost = array_1d(costs, trajectories, sizeof(float));
  int num_tool_frames = manifest_.num_tool_frames;
  uint8_t use_pose_constraint = pose_constraint_active ? 1U : 0U;
  uint8_t use_effort = config_.check_effort_limits ? 1U : 0U;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &cspace_cost,
      &gradient_position,
      &gradient_velocity,
      &gradient_acceleration,
      &gradient_jerk,
      &effort_gradient_position,
      &effort_gradient_velocity,
      &effort_gradient_acceleration,
      &collision_gradient,
      &world_cost,
      &self_cost,
      &pose_cost,
      &num_tool_frames,
      &use_pose_constraint,
      &use_effort,
      &gradient_position,
      &gradient_velocity,
      &gradient_acceleration,
      &gradient_jerk,
      &output_cost,
  };
  const size_t threads =
      static_cast<size_t>(trajectories) * static_cast<size_t>(block_size);
  combine_rollout_kernel_.launch(grid_for(threads, block_size),
      static_cast<unsigned int>(manifest_.combine_rollout.block_dim),
      static_cast<unsigned int>(manifest_.combine_rollout.shared_memory), stream,
      arguments);
}

void TrajectoryOptimizerRuntime::launch_bspline_backward(
    int trajectories, CUdeviceptr knot_gradients, CUstream stream) const {
  CUdeviceptr position_gradient = cspace_gradient_position_.ptr();
  CUdeviceptr velocity_gradient = cspace_gradient_velocity_.ptr();
  CUdeviceptr acceleration_gradient = cspace_gradient_acceleration_.ptr();
  CUdeviceptr jerk_gradient = cspace_gradient_jerk_.ptr();
  CUdeviceptr trajectory_dt = trajectory_dt_.ptr();
  CUdeviceptr dt_indices =
      trajectories == config_.num_seeds ? seed_goal_indices_.ptr()
                                        : candidate_goal_indices_.ptr();
  CUdeviceptr implicit_goal = implicit_goal_.ptr();
  int batch_size = trajectories;
  int horizon = manifest_.trajectory_steps - 1;
  int dof = manifest_.num_dof;
  int n_knots = manifest_.num_knots;
  std::vector<void*> arguments{
      &knot_gradients,
      &position_gradient,
      &velocity_gradient,
      &acceleration_gradient,
      &jerk_gradient,
      &trajectory_dt,
      &dt_indices,
      &implicit_goal,
      &batch_size,
      &horizon,
      &dof,
      &n_knots,
  };
  const int total_knots = n_knots + manifest_.spline_degree + 1;
  const int interpolation_steps = horizon / total_knots;
  if (interpolation_steps <= 0 || interpolation_steps > 32 ||
      horizon % total_knots != 0) {
    throw std::runtime_error("TrajOpt artifact has an invalid B-spline layout");
  }
  const int knots_per_warp = 32 / interpolation_steps;
  const int warps_for_knots =
      (n_knots + knots_per_warp - 1) / knots_per_warp;
  const int threads_for_knots = warps_for_knots * 32;
  const size_t threads = static_cast<size_t>(trajectories) *
                         static_cast<size_t>(dof) *
                         static_cast<size_t>(threads_for_knots);
  bspline_backward_kernel_.launch(
      grid_for(threads, manifest_.bspline_backward.block_dim),
      static_cast<unsigned int>(manifest_.bspline_backward.block_dim),
      static_cast<unsigned int>(manifest_.bspline_backward.shared_memory), stream,
      arguments);
}

void TrajectoryOptimizerRuntime::evaluate(
    CUdeviceptr knots, int trajectories, CUdeviceptr costs,
    CUdeviceptr knot_gradients,
    const TrajectoryOptimizationRequest& request, CUstream stream) const {
  expand_trajectory_dt(trajectories, stream);
  launch_bspline(knots, trajectories, request, stream);
  auto* streams = rollout_streams_.get();
  const auto primary = reinterpret_cast<cudaStream_t>(stream);
  const auto dynamics = streams->cuda_stream(RolloutStream::DYNAMICS);
  const auto kinematics = streams->cuda_stream(RolloutStream::KINEMATICS);
  const auto cspace = streams->cuda_stream(RolloutStream::CSPACE);
  const auto pose = streams->cuda_stream(RolloutStream::POSE);
  const auto world = streams->cuda_stream(RolloutStream::WORLD);
  const auto self = streams->cuda_stream(RolloutStream::SELF);
  streams->record(RolloutEvent::STATE_READY, primary);

  if (config_.check_effort_limits) {
    const int states = trajectories * manifest_.trajectory_steps;
    streams->wait(dynamics, RolloutEvent::STATE_READY);
    rollout_->launch_inverse_dynamics(
        dense_position_.ptr(), dense_velocity_.ptr(),
        dense_acceleration_.ptr(), joint_effort_.ptr(),
        rnea_forward_cache_.ptr(), states,
        streams->driver_stream(RolloutStream::DYNAMICS));
    streams->record(RolloutEvent::DYNAMICS_READY, dynamics);
    streams->wait(cspace, RolloutEvent::DYNAMICS_READY);
  } else {
    streams->wait(cspace, RolloutEvent::STATE_READY);
  }
  launch_cspace(trajectories, streams->driver_stream(RolloutStream::CSPACE));
  if (config_.check_effort_limits) {
    const int states = trajectories * manifest_.trajectory_steps;
    rollout_->launch_inverse_dynamics_backward(
        effort_gradient_.ptr(), dense_position_.ptr(), dense_velocity_.ptr(),
        rnea_forward_cache_.ptr(), effort_gradient_position_.ptr(),
        effort_gradient_velocity_.ptr(), effort_gradient_acceleration_.ptr(),
        states, streams->driver_stream(RolloutStream::CSPACE));
  }
  streams->record(RolloutEvent::CSPACE_DONE, cspace);

  streams->wait(kinematics, RolloutEvent::STATE_READY);
  launch_kinematics(
      trajectories, streams->driver_stream(RolloutStream::KINEMATICS));
  streams->record(RolloutEvent::KINEMATICS_READY, kinematics);
  for (const auto cost_stream : {pose, world, self}) {
    streams->wait(cost_stream, RolloutEvent::KINEMATICS_READY);
  }
  launch_pose(trajectories, request,
              streams->driver_stream(RolloutStream::POSE));
  streams->record(RolloutEvent::POSE_DONE, pose);
  launch_world_collision(trajectories,
                         streams->driver_stream(RolloutStream::WORLD));
  streams->record(RolloutEvent::WORLD_DONE, world);
  launch_self_collision(trajectories,
                        streams->driver_stream(RolloutStream::SELF));
  streams->record(RolloutEvent::SELF_DONE, self);
  for (const auto event : {
           RolloutEvent::WORLD_DONE,
           RolloutEvent::SELF_DONE,
       }) {
    streams->wait(kinematics, event);
  }
  if (request.constrain_tool_path) {
    streams->wait(kinematics, RolloutEvent::POSE_DONE);
  }
  launch_kinematics_backward(
      trajectories, request,
      streams->driver_stream(RolloutStream::KINEMATICS));
  streams->record(RolloutEvent::KINEMATICS_BACKWARD_DONE, kinematics);

  streams->wait(primary, RolloutEvent::POSE_DONE);
  streams->wait(primary, RolloutEvent::CSPACE_DONE);
  streams->wait(primary, RolloutEvent::KINEMATICS_BACKWARD_DONE);
  combine_rollout(trajectories, costs, request.constrain_tool_path, stream);
  launch_bspline_backward(trajectories, knot_gradients, stream);
}

void TrajectoryOptimizerRuntime::prepare_initial_state(
    bool clear_optimizer_state, CUstream stream) const {
  auto beta = array_1d(line_search_beta_.ptr(), config_.num_seeds, sizeof(float));
  if (clear_optimizer_state) {
    history_rho_.clear(reinterpret_cast<cudaStream_t>(stream));
    history_y_.clear(reinterpret_cast<cudaStream_t>(stream));
    history_s_.clear(reinterpret_cast<cudaStream_t>(stream));
  }
  const int opt_dim = manifest_.num_knots * manifest_.num_dof;
  const auto bounds = warp_bounds_1d(config_.num_seeds);
  auto seed_action =
      array_2d(seed_actions_.ptr(), config_.num_seeds, opt_dim, sizeof(float));
  auto seed_cost =
      array_1d(search_costs_.ptr(), config_.num_seeds, sizeof(float));
  auto seed_gradient = array_2d(search_gradients_.ptr(), config_.num_seeds,
                                opt_dim, sizeof(float));
  auto action = array_2d(action_.ptr(), config_.num_seeds, opt_dim, sizeof(float));
  auto gradient =
      array_2d(gradient_.ptr(), config_.num_seeds, opt_dim, sizeof(float));
  auto cost = array_1d(cost_.ptr(), config_.num_seeds, sizeof(float));
  auto exploration_action = array_2d(exploration_action_.ptr(),
                                     config_.num_seeds, opt_dim, sizeof(float));
  auto exploration_gradient = array_2d(exploration_gradient_.ptr(),
                                       config_.num_seeds, opt_dim, sizeof(float));
  auto exploration_cost =
      array_1d(exploration_cost_.ptr(), config_.num_seeds, sizeof(float));
  auto best_action =
      array_2d(best_action_.ptr(), config_.num_seeds, opt_dim, sizeof(float));
  auto best_cost =
      array_1d(best_cost_.ptr(), config_.num_seeds, sizeof(float));
  auto best_iteration =
      array_1d(best_iteration_.ptr(), config_.num_seeds, sizeof(int32_t));
  auto current_iteration =
      array_1d(current_iteration_.ptr(), config_.num_seeds, sizeof(int32_t));
  auto converged =
      array_1d(converged_.ptr(), config_.num_seeds, sizeof(uint8_t));
  auto step_direction = array_2d(step_direction_.ptr(), config_.num_seeds,
                                 opt_dim, sizeof(float));
  auto reference_action = array_2d(reference_action_.ptr(), config_.num_seeds,
                                   opt_dim, sizeof(float));
  auto reference_gradient = array_2d(reference_gradient_.ptr(),
                                     config_.num_seeds, opt_dim, sizeof(float));
  auto history_head =
      array_1d(history_head_.ptr(), config_.num_seeds, sizeof(int32_t));
  auto history_count =
      array_1d(history_count_.ptr(), config_.num_seeds, sizeof(int32_t));
  auto curvature =
      array_1d(curvature_update_valid_.ptr(), config_.num_seeds, sizeof(bool));
  auto step_max =
      array_1d(optimization_step_max_.ptr(), opt_dim, sizeof(float));
  float initial_step_scale = manifest_.initial_step_scale;
  uint8_t clear_state = clear_optimizer_state ? 1U : 0U;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds),
      &seed_action,
      &seed_cost,
      &seed_gradient,
      &action,
      &gradient,
      &cost,
      &exploration_action,
      &exploration_gradient,
      &exploration_cost,
      &best_action,
      &best_cost,
      &best_iteration,
      &current_iteration,
      &converged,
      &step_direction,
      &reference_action,
      &reference_gradient,
      &history_head,
      &history_count,
      &curvature,
      &step_max,
      &initial_step_scale,
      &clear_state,
      &beta,
  };
  prepare_initial_state_kernel_.launch(
      grid_for(config_.num_seeds, manifest_.prepare_initial_state.block_dim),
      static_cast<unsigned int>(manifest_.prepare_initial_state.block_dim),
      static_cast<unsigned int>(manifest_.prepare_initial_state.shared_memory),
      stream, arguments);
}

void TrajectoryOptimizerRuntime::prepare_search_actions(CUstream stream) const {
  auto beta = array_1d(line_search_beta_.ptr(), config_.num_seeds, sizeof(float));
  const int opt_dim = manifest_.num_knots * manifest_.num_dof;
  const int candidates = config_.num_seeds * manifest_.num_line_search;
  const auto bounds = warp_bounds_2d(candidates, opt_dim);
  auto exploration = array_2d(exploration_action_.ptr(), config_.num_seeds,
                              opt_dim, sizeof(float));
  auto direction = array_2d(step_direction_.ptr(), config_.num_seeds, opt_dim,
                            sizeof(float));
  auto scales = array_1d(line_search_scale_.ptr(), manifest_.num_line_search,
                         sizeof(float));
  auto search =
      array_2d(search_actions_.ptr(), candidates, opt_dim, sizeof(float));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &exploration,
      &direction,
      &scales,
      &beta,
      &search,
  };
  const size_t elements =
      static_cast<size_t>(candidates) * static_cast<size_t>(opt_dim);
  prepare_search_actions_kernel_.launch(
      grid_for(elements, manifest_.prepare_search_actions.block_dim),
      static_cast<unsigned int>(manifest_.prepare_search_actions.block_dim),
      static_cast<unsigned int>(manifest_.prepare_search_actions.shared_memory),
      stream, arguments);
}

void TrajectoryOptimizerRuntime::line_search(CUstream stream) const {
  CUdeviceptr beta = line_search_beta_.ptr();
  CUdeviceptr best_cost = best_cost_.ptr();
  CUdeviceptr best_action = best_action_.ptr();
  CUdeviceptr best_iteration = best_iteration_.ptr();
  CUdeviceptr current_iteration = current_iteration_.ptr();
  CUdeviceptr converged = converged_.ptr();
  int convergence_iteration = manifest_.convergence_iteration;
  float cost_delta_threshold = manifest_.cost_delta_threshold;
  float cost_relative_threshold = manifest_.cost_relative_threshold;
  CUdeviceptr exploration_cost = exploration_cost_.ptr();
  CUdeviceptr exploration_action = exploration_action_.ptr();
  CUdeviceptr exploration_gradient = exploration_gradient_.ptr();
  CUdeviceptr exploration_indices = exploration_indices_.ptr();
  CUdeviceptr selected_cost = cost_.ptr();
  CUdeviceptr selected_action = action_.ptr();
  CUdeviceptr selected_gradient = gradient_.ptr();
  CUdeviceptr selected_indices = selected_indices_.ptr();
  CUdeviceptr search_cost = search_costs_.ptr();
  CUdeviceptr search_action = search_actions_.ptr();
  CUdeviceptr search_gradient = search_gradients_.ptr();
  CUdeviceptr step_direction = step_direction_.ptr();
  CUdeviceptr search_magnitudes = line_search_scale_.ptr();
  CUdeviceptr line_search_index = line_search_index_.ptr();
  CUdeviceptr line_search_status = line_search_status_.ptr();
  CUdeviceptr curvature = curvature_update_valid_.ptr();
  float armijo_threshold = manifest_.armijo_threshold;
  float curvature_threshold = manifest_.curvature_threshold;
  int opt_dim = manifest_.num_knots * manifest_.num_dof;
  int batch_size = config_.num_seeds;
  std::vector<void*> arguments{
      &best_cost,
      &best_action,
      &best_iteration,
      &current_iteration,
      &converged,
      &convergence_iteration,
      &cost_delta_threshold,
      &cost_relative_threshold,
      &exploration_cost,
      &exploration_action,
      &exploration_gradient,
      &exploration_indices,
      &selected_cost,
      &selected_action,
      &selected_gradient,
      &selected_indices,
      &search_cost,
      &search_action,
      &search_gradient,
      &step_direction,
      &search_magnitudes,
      &line_search_index,
      &line_search_status,
      &curvature,
      &armijo_threshold,
      &curvature_threshold,
      &opt_dim,
      &batch_size,
      &beta,
  };
  line_search_kernel_.launch(
      static_cast<unsigned int>(config_.num_seeds),
      static_cast<unsigned int>(manifest_.line_search.block_dim),
      static_cast<unsigned int>(manifest_.line_search.shared_memory), stream,
      arguments);
}

void TrajectoryOptimizerRuntime::lbfgs_step(CUstream stream) const {
  CUdeviceptr line_search_status = line_search_status_.ptr();
  CUdeviceptr direction = step_direction_.ptr();
  CUdeviceptr rho = history_rho_.ptr();
  CUdeviceptr y = history_y_.ptr();
  CUdeviceptr s = history_s_.ptr();
  CUdeviceptr q = exploration_action_.ptr();
  CUdeviceptr reference_action = reference_action_.ptr();
  CUdeviceptr reference_gradient = reference_gradient_.ptr();
  CUdeviceptr gradient = exploration_gradient_.ptr();
  CUdeviceptr step_max = optimization_step_max_.ptr();
  float epsilon = manifest_.epsilon;
  int batch_size = config_.num_seeds;
  int history = manifest_.history;
  int opt_dim = manifest_.num_knots * manifest_.num_dof;
  bool stable_mode = true;
  CUdeviceptr history_head = history_head_.ptr();
  CUdeviceptr history_count = history_count_.ptr();
  bool scale_step = true;
  CUdeviceptr curvature = curvature_update_valid_.ptr();
  std::vector<void*> arguments{
      &direction,
      &rho,
      &y,
      &s,
      &q,
      &reference_action,
      &reference_gradient,
      &gradient,
      &step_max,
      &epsilon,
      &batch_size,
      &history,
      &opt_dim,
      &stable_mode,
      &history_head,
      &history_count,
      &scale_step,
      &curvature,
      &line_search_status,
  };
  lbfgs_kernel_.launch(static_cast<unsigned int>(config_.num_seeds),
                       static_cast<unsigned int>(manifest_.lbfgs.block_dim),
                       static_cast<unsigned int>(manifest_.lbfgs.shared_memory),
                       stream, arguments);
}

void TrajectoryOptimizerRuntime::optimize_pass(
    const TrajectoryOptimizationRequest& request, CUstream stream, int iterations) const {
  evaluate(seed_actions_.ptr(), config_.num_seeds, search_costs_.ptr(),
           search_gradients_.ptr(), request, stream);
  prepare_initial_state(false, stream);
  const int candidates = config_.num_seeds * manifest_.num_line_search;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    prepare_search_actions(stream);
    evaluate(search_actions_.ptr(), candidates, search_costs_.ptr(),
             search_gradients_.ptr(), request, stream);
    line_search(stream);
    lbfgs_step(stream);
  }
  evaluate(best_action_.ptr(), config_.num_seeds, search_costs_.ptr(),
           search_gradients_.ptr(), request, stream);
  launch_metrics_collisions(config_.num_seeds, stream);
}

TrajectoryOptimizationCandidates TrajectoryOptimizerRuntime::candidates() const {
  return {reinterpret_cast<const float *>(retained_action_.ptr()),
          reinterpret_cast<const float *>(retained_position_.ptr()),
          reinterpret_cast<const float *>(retained_velocity_.ptr()),
          reinterpret_cast<const float *>(retained_acceleration_.ptr()),
          reinterpret_cast<const float *>(retained_jerk_.ptr()),
          reinterpret_cast<const float *>(retained_dt_.ptr()),
          reinterpret_cast<const uint8_t *>(retained_success_.ptr())};
}

const int32_t *TrajectoryOptimizerRuntime::validate(
    const TrajectoryValidationRequest &request, bool timeopt) const {
  if (!request.position || !request.velocity || !request.acceleration || !request.jerk ||
      request.num_samples != manifest_.interpolation_buffer_size) {
    throw std::invalid_argument("validation requires fixed-shape position and derivatives");
  }
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  const size_t bytes = static_cast<size_t>(config_.num_seeds) * request.num_samples *
                       manifest_.num_dof * sizeof(float);
  const auto copy = [&](CUdeviceptr target, const float *source) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(target), source, bytes,
                                   cudaMemcpyDeviceToDevice, request.stream));
  };
  copy(validation_position_.ptr(), request.position);
  copy(validation_velocity_.ptr(), request.velocity);
  copy(validation_acceleration_.ptr(), request.acceleration);
  copy(validation_jerk_.ptr(), request.jerk);
  validate_states(reinterpret_cast<CUstream>(request.stream), timeopt);
  return reinterpret_cast<const int32_t *>(validation_violation_.ptr());
}

void TrajectoryOptimizerRuntime::validate_interpolated(
    CUdeviceptr knots, CUstream stream) const {
  int total_knots =
      manifest_.num_knots + manifest_.spline_degree + 1;
  int sparse_interpolation_steps =
      (manifest_.trajectory_steps - 1) / total_knots;
  const auto horizon_bounds = warp_bounds_1d(config_.num_seeds);
  auto trajectory_dt =
      array_1d(trajectory_dt_.ptr(), config_.num_seeds, sizeof(float));
  auto interpolation_horizon = array_1d(
      interpolation_horizon_.ptr(), config_.num_seeds, sizeof(int32_t));
  float interpolation_dt = manifest_.interpolation_dt;
  std::vector<void*> horizon_arguments{
      const_cast<WarpLaunchBounds1*>(&horizon_bounds),
      &trajectory_dt,
      &interpolation_horizon,
      &interpolation_dt,
      &sparse_interpolation_steps,
      &total_knots,
  };
  calculate_interpolation_horizon_kernel_.launch(
      grid_for(config_.num_seeds,
               manifest_.calculate_interpolation_horizon.block_dim),
      static_cast<unsigned int>(
          manifest_.calculate_interpolation_horizon.block_dim),
      static_cast<unsigned int>(
          manifest_.calculate_interpolation_horizon.shared_memory),
      stream, horizon_arguments);

  CUdeviceptr output_position = validation_position_.ptr();
  CUdeviceptr output_velocity = validation_velocity_.ptr();
  CUdeviceptr output_acceleration = validation_acceleration_.ptr();
  CUdeviceptr output_jerk = validation_jerk_.ptr();
  CUdeviceptr output_dt = validation_dt_.ptr();
  CUdeviceptr start = resolved_start_positions_.ptr();
  CUdeviceptr start_velocity = resolved_start_velocities_.ptr();
  CUdeviceptr start_acceleration = resolved_start_accelerations_.ptr();
  CUdeviceptr start_jerk = resolved_start_jerks_.ptr();
  CUdeviceptr goal = resolved_goal_positions_.ptr();
  CUdeviceptr zero = zero_joint_state_.ptr();
  CUdeviceptr indices = seed_goal_indices_.ptr();
  CUdeviceptr interpolation_dt_pointer = interpolation_dt_.ptr();
  CUdeviceptr knot_dt_pointer = trajectory_dt_.ptr();
  CUdeviceptr implicit_goal = implicit_goal_.ptr();
  CUdeviceptr interpolation_horizon_pointer = interpolation_horizon_.ptr();
  int batch_size = config_.num_seeds;
  int validation_horizon = manifest_.interpolation_buffer_size;
  int dof = manifest_.num_dof;
  int n_knots = manifest_.num_knots;
  std::vector<void*> interpolation_arguments{
      &output_position,
      &output_velocity,
      &output_acceleration,
      &output_jerk,
      &output_dt,
      &knots,
      &knot_dt_pointer,
      &start,
      &start_velocity,
      &start_acceleration,
      &start_jerk,
      &goal,
      &zero,
      &zero,
      &zero,
      &indices,
      &indices,
      &interpolation_dt_pointer,
      &implicit_goal,
      &interpolation_horizon_pointer,
      &batch_size,
      &validation_horizon,
      &dof,
      &n_knots,
  };
  const size_t interpolation_values =
      static_cast<size_t>(config_.num_seeds) * validation_horizon * dof;
  bspline_single_dt_kernel_.launch(
      grid_for(interpolation_values, manifest_.bspline_single_dt.block_dim),
      static_cast<unsigned int>(manifest_.bspline_single_dt.block_dim),
      static_cast<unsigned int>(manifest_.bspline_single_dt.shared_memory),
      stream, interpolation_arguments);

  validate_states(stream);
}

void TrajectoryOptimizerRuntime::validate_states(CUstream stream, bool timeopt) const {
  const bool check_effort = config_.check_effort_limits || timeopt;
  const int validation_horizon = manifest_.interpolation_buffer_size;
  const size_t interpolation_values =
      static_cast<size_t>(config_.num_seeds) * validation_horizon * manifest_.num_dof;
  auto* streams = rollout_streams_.get();
  const auto primary = reinterpret_cast<cudaStream_t>(stream);
  const auto dynamics = streams->cuda_stream(RolloutStream::DYNAMICS);
  const auto kinematics = streams->cuda_stream(RolloutStream::KINEMATICS);
  const auto world = streams->cuda_stream(RolloutStream::WORLD);
  const auto self = streams->cuda_stream(RolloutStream::SELF);
  streams->record(RolloutEvent::STATE_READY, primary);

  if (check_effort) {
    streams->wait(dynamics, RolloutEvent::STATE_READY);
    rollout_->launch_inverse_dynamics(
        validation_position_.ptr(), validation_velocity_.ptr(),
        validation_acceleration_.ptr(), validation_joint_effort_.ptr(),
        validation_rnea_forward_cache_.ptr(),
        config_.num_seeds * validation_horizon,
        streams->driver_stream(RolloutStream::DYNAMICS));
    streams->record(RolloutEvent::DYNAMICS_READY, dynamics);
  }

  streams->wait(kinematics, RolloutEvent::STATE_READY);
  launch_kinematics_to(
      config_.num_seeds, validation_horizon, validation_position_.ptr(),
      validation_link_position_.ptr(), validation_link_quaternion_.ptr(),
      validation_robot_spheres_.ptr(),
      validation_global_cumulative_.ptr(),
      streams->driver_stream(RolloutStream::KINEMATICS));
  streams->record(RolloutEvent::KINEMATICS_READY, kinematics);
  streams->wait(world, RolloutEvent::KINEMATICS_READY);
  streams->wait(self, RolloutEvent::KINEMATICS_READY);
  launch_discrete_world_collision_to(
      config_.num_seeds, validation_horizon,
      validation_robot_spheres_.ptr(),
      validation_world_collision_cost_.ptr(),
      validation_world_collision_gradient_.ptr(), true,
      streams->driver_stream(RolloutStream::WORLD));
  streams->record(RolloutEvent::WORLD_DONE, world);
  launch_self_collision_to(
      config_.num_seeds, validation_horizon,
      validation_robot_spheres_.ptr(),
      validation_self_collision_cost_.ptr(),
      validation_self_collision_gradient_.ptr(),
      validation_self_collision_sparse_index_.ptr(),
      validation_self_collision_weight_.ptr(),
      streams->driver_stream(RolloutStream::SELF));
  streams->record(RolloutEvent::SELF_DONE, self);
  streams->wait(primary, RolloutEvent::WORLD_DONE);
  streams->wait(primary, RolloutEvent::SELF_DONE);
  if (check_effort) {
    streams->wait(primary, RolloutEvent::DYNAMICS_READY);
  }

  validation_violation_.clear(reinterpret_cast<cudaStream_t>(stream));
  auto position =
      array_1d(validation_position_.ptr(), interpolation_values, sizeof(float));
  auto velocity =
      array_1d(validation_velocity_.ptr(), interpolation_values, sizeof(float));
  auto acceleration = array_1d(
      validation_acceleration_.ptr(), interpolation_values, sizeof(float));
  auto jerk =
      array_1d(validation_jerk_.ptr(), interpolation_values, sizeof(float));
  auto effort = array_1d(validation_joint_effort_.ptr(), interpolation_values,
                         sizeof(float));
  const int validation_states = config_.num_seeds * validation_horizon;
  auto world_collision = array_1d(
      validation_world_collision_cost_.ptr(), validation_states, sizeof(float));
  auto self_collision = array_1d(
      validation_self_collision_cost_.ptr(), validation_states, sizeof(float));
  const auto dof_bytes =
      static_cast<CUdeviceptr>(manifest_.num_dof * sizeof(float));
  auto position_lower =
      array_1d(position_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto position_upper = array_1d(position_bounds_.ptr() + dof_bytes,
                                 manifest_.num_dof, sizeof(float));
  auto velocity_lower =
      array_1d(velocity_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto velocity_upper = array_1d(velocity_bounds_.ptr() + dof_bytes,
                                 manifest_.num_dof, sizeof(float));
  auto acceleration_lower =
      array_1d(acceleration_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto acceleration_upper = array_1d(
      acceleration_bounds_.ptr() + dof_bytes, manifest_.num_dof, sizeof(float));
  auto jerk_lower =
      array_1d(jerk_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto jerk_upper = array_1d(jerk_bounds_.ptr() + dof_bytes,
                             manifest_.num_dof, sizeof(float));
  const auto effort_bounds_pointer =
      rollout_ == nullptr ? position_bounds_.ptr() : rollout_->effort_bounds();
  auto effort_lower =
      array_1d(effort_bounds_pointer, manifest_.num_dof, sizeof(float));
  auto effort_upper = array_1d(effort_bounds_pointer + dof_bytes,
                               manifest_.num_dof, sizeof(float));
  auto violation =
      array_1d(validation_violation_.ptr(), config_.num_seeds, sizeof(int32_t));
  float constraint_tolerance = config_.constraint_tolerance;
  uint8_t effort_active = check_effort ? 1U : 0U;
  const auto validation_bounds = warp_bounds_1d(interpolation_values);
  std::vector<void*> validation_arguments{
      const_cast<WarpLaunchBounds1*>(&validation_bounds),
      &position,
      &velocity,
      &acceleration,
      &jerk,
      &effort,
      &world_collision,
      &self_collision,
      &position_lower,
      &position_upper,
      &velocity_lower,
      &velocity_upper,
      &acceleration_lower,
      &acceleration_upper,
      &jerk_lower,
      &jerk_upper,
      &effort_lower,
      &effort_upper,
      &violation,
      &constraint_tolerance,
      &effort_active,
  };
  validate_interpolated_kernel_.launch(
      grid_for(interpolation_values, manifest_.validate_interpolated.block_dim),
      static_cast<unsigned int>(manifest_.validate_interpolated.block_dim),
      static_cast<unsigned int>(manifest_.validate_interpolated.shared_memory),
      stream, validation_arguments);
}

void TrajectoryOptimizerRuntime::accumulate(
    bool first_pass, const TrajectoryOptimizationRequest& request,
    CUstream stream) const {
  const int opt_dim = manifest_.num_knots * manifest_.num_dof;
  const int values =
      config_.num_seeds * manifest_.trajectory_steps * manifest_.num_dof;
  const int states = config_.num_seeds * manifest_.trajectory_steps;
  const auto bounds = warp_tiled_bounds_2d(config_.num_seeds, manifest_.accumulate.block_dim);
  auto knots =
      array_2d(best_action_.ptr(), config_.num_seeds, opt_dim, sizeof(float));
  auto costs = array_1d(search_costs_.ptr(), config_.num_seeds, sizeof(float));
  auto position = array_1d(dense_position_.ptr(), values, sizeof(float));
  auto velocity = array_1d(dense_velocity_.ptr(), values, sizeof(float));
  auto acceleration = array_1d(dense_acceleration_.ptr(), values, sizeof(float));
  auto jerk = array_1d(dense_jerk_.ptr(), values, sizeof(float));
  auto effort = array_1d(joint_effort_.ptr(), values, sizeof(float));
  auto world_cost =
      array_1d(world_collision_cost_.ptr(), states, sizeof(float));
  auto self_cost =
      array_1d(self_collision_cost_.ptr(), states, sizeof(float));
  const int state_tools = states * manifest_.num_tool_frames;
  auto pose_position_distance = array_1d(
      pose_position_distance_.ptr(), state_tools, sizeof(float));
  auto pose_orientation_distance = array_1d(
      pose_orientation_distance_.ptr(), state_tools, sizeof(float));
  const auto dof_bytes =
      static_cast<CUdeviceptr>(manifest_.num_dof * sizeof(float));
  auto position_lower =
      array_1d(position_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto position_upper = array_1d(position_bounds_.ptr() + dof_bytes,
                                 manifest_.num_dof, sizeof(float));
  auto velocity_lower =
      array_1d(velocity_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto velocity_upper = array_1d(velocity_bounds_.ptr() + dof_bytes,
                                 manifest_.num_dof, sizeof(float));
  auto acceleration_lower =
      array_1d(acceleration_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto acceleration_upper = array_1d(acceleration_bounds_.ptr() + dof_bytes,
                                     manifest_.num_dof, sizeof(float));
  auto jerk_lower =
      array_1d(jerk_bounds_.ptr(), manifest_.num_dof, sizeof(float));
  auto jerk_upper = array_1d(jerk_bounds_.ptr() + dof_bytes,
                             manifest_.num_dof, sizeof(float));
  const auto effort_bounds_pointer =
      rollout_ == nullptr ? position_bounds_.ptr() : rollout_->effort_bounds();
  auto effort_lower =
      array_1d(effort_bounds_pointer, manifest_.num_dof, sizeof(float));
  auto effort_upper = array_1d(effort_bounds_pointer + dof_bytes,
                               manifest_.num_dof, sizeof(float));
  auto trajectory_dt =
      array_1d(trajectory_dt_.ptr(), config_.num_seeds, sizeof(float));
  auto retained_knots = array_2d(retained_action_.ptr(), config_.num_seeds,
                                 opt_dim, sizeof(float));
  auto retained_position = array_1d(retained_position_.ptr(), values, sizeof(float));
  auto retained_velocity = array_1d(retained_velocity_.ptr(), values, sizeof(float));
  auto retained_acceleration =
      array_1d(retained_acceleration_.ptr(), values, sizeof(float));
  auto retained_jerk = array_1d(retained_jerk_.ptr(), values, sizeof(float));
  auto retained_success =
      array_1d(retained_success_.ptr(), config_.num_seeds, sizeof(uint8_t));
  auto retained_pose_valid =
      array_1d(retained_pose_valid_.ptr(), config_.num_seeds, sizeof(uint8_t));
  auto retained_cost =
      array_1d(retained_cost_.ptr(), config_.num_seeds, sizeof(float));
  auto retained_rank_score =
      array_1d(retained_rank_score_.ptr(), config_.num_seeds, sizeof(float));
  auto retained_collision_cost = array_1d(
      retained_collision_cost_.ptr(), config_.num_seeds, sizeof(float));
  auto retained_dt =
      array_1d(retained_dt_.ptr(), config_.num_seeds, sizeof(float));
  auto active_seed_valid =
      array_1d(active_seed_valid_.ptr(), config_.num_seeds, sizeof(uint8_t));
  auto interpolated_violation = array_1d(
      validation_violation_.ptr(), config_.num_seeds, sizeof(int32_t));
  auto finetune_enabled =
      array_1d(finetune_enabled_.ptr(), 1, sizeof(int32_t));
  uint8_t is_first_pass = first_pass ? 1U : 0U;
  float constraint_tolerance = config_.constraint_tolerance;
  float pose_position_tolerance = config_.position_tolerance;
  float pose_orientation_tolerance = config_.orientation_tolerance;
  float path_position_tolerance =
      request.path_position_tolerance < 0.0F
          ? config_.position_tolerance
          : request.path_position_tolerance;
  float path_orientation_tolerance =
      request.path_orientation_tolerance < 0.0F
          ? config_.orientation_tolerance
          : request.path_orientation_tolerance;
  int num_tool_frames = manifest_.num_tool_frames;
  uint8_t pose_constraint_active = request.constrain_tool_path ? 1U : 0U;
  uint8_t effort_active = config_.check_effort_limits ? 1U : 0U;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &knots,
      &costs,
      &position,
      &velocity,
      &acceleration,
      &jerk,
      &effort,
      &world_cost,
      &self_cost,
      &pose_position_distance,
      &pose_orientation_distance,
      &position_lower,
      &position_upper,
      &velocity_lower,
      &velocity_upper,
      &acceleration_lower,
      &acceleration_upper,
      &jerk_lower,
      &jerk_upper,
      &effort_lower,
      &effort_upper,
      &trajectory_dt,
      &retained_knots,
      &retained_position,
      &retained_velocity,
      &retained_acceleration,
      &retained_jerk,
      &retained_success,
      &retained_pose_valid,
      &retained_cost,
      &retained_rank_score,
      &retained_collision_cost,
      &retained_dt,
      &active_seed_valid,
      &interpolated_violation,
      &finetune_enabled,
      &is_first_pass,
      &constraint_tolerance,
      &pose_position_tolerance,
      &pose_orientation_tolerance,
      &path_position_tolerance,
      &path_orientation_tolerance,
      &num_tool_frames,
      &pose_constraint_active,
      &effort_active,
  };
  accumulate_kernel_.launch(
      static_cast<unsigned int>(config_.num_seeds),
      static_cast<unsigned int>(manifest_.accumulate.block_dim),
      static_cast<unsigned int>(manifest_.accumulate.shared_memory), stream,
      arguments);
}

void TrajectoryOptimizerRuntime::finalize(
    const TrajectoryOptimizationRequest& request, CUstream stream) const {
  const int opt_dim = manifest_.num_knots * manifest_.num_dof;
  const int values =
      config_.num_seeds * manifest_.trajectory_steps * manifest_.num_dof;
  const auto bounds = warp_tiled_bounds_2d(1, manifest_.finalize.block_dim);
  auto retained_knots = array_2d(retained_action_.ptr(), config_.num_seeds,
                                 opt_dim, sizeof(float));
  auto retained_position = array_1d(retained_position_.ptr(), values, sizeof(float));
  auto retained_velocity = array_1d(retained_velocity_.ptr(), values, sizeof(float));
  auto retained_acceleration =
      array_1d(retained_acceleration_.ptr(), values, sizeof(float));
  auto retained_jerk = array_1d(retained_jerk_.ptr(), values, sizeof(float));
  auto retained_success =
      array_1d(retained_success_.ptr(), config_.num_seeds, sizeof(uint8_t));
  auto retained_cost =
      array_1d(retained_cost_.ptr(), config_.num_seeds, sizeof(float));
  auto retained_rank_score =
      array_1d(retained_rank_score_.ptr(), config_.num_seeds, sizeof(float));
  auto retained_collision_cost = array_1d(
      retained_collision_cost_.ptr(), config_.num_seeds, sizeof(float));
  auto retained_dt =
      array_1d(retained_dt_.ptr(), config_.num_seeds, sizeof(float));
  auto output_knots = warp_array_1d(
      reinterpret_cast<uint64_t>(request.optimized_knots), opt_dim, sizeof(float));
  const int output_values = manifest_.trajectory_steps * manifest_.num_dof;
  auto output_position = warp_array_1d(
      reinterpret_cast<uint64_t>(request.trajectory_position), output_values,
      sizeof(float));
  auto output_velocity = warp_array_1d(
      reinterpret_cast<uint64_t>(request.trajectory_velocity), output_values,
      sizeof(float));
  auto output_acceleration = warp_array_1d(
      reinterpret_cast<uint64_t>(request.trajectory_acceleration), output_values,
      sizeof(float));
  auto output_jerk = warp_array_1d(
      reinterpret_cast<uint64_t>(request.trajectory_jerk), output_values,
      sizeof(float));
  auto output_success = warp_array_1d(
      reinterpret_cast<uint64_t>(request.success), 1, sizeof(uint8_t));
  auto output_cost = warp_array_1d(
      reinterpret_cast<uint64_t>(request.cost), 1, sizeof(float));
  auto output_collision_cost = warp_array_1d(
      reinterpret_cast<uint64_t>(request.collision_cost), 1, sizeof(float));
  auto output_dt = warp_array_1d(
      reinterpret_cast<uint64_t>(request.trajectory_dt), 1, sizeof(float));
  const auto selected_seed_pointer =
      request.selected_seed == nullptr
          ? selected_seed_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.selected_seed));
  auto output_selected_seed =
      array_1d(selected_seed_pointer, 1, sizeof(int32_t));
  int active_seeds = config_.num_seeds;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &retained_knots,
      &retained_position,
      &retained_velocity,
      &retained_acceleration,
      &retained_jerk,
      &retained_success,
      &retained_cost,
      &retained_rank_score,
      &retained_collision_cost,
      &retained_dt,
      &output_knots,
      &output_position,
      &output_velocity,
      &output_acceleration,
      &output_jerk,
      &output_success,
      &output_cost,
      &output_collision_cost,
      &output_dt,
      &output_selected_seed,
      &active_seeds,
  };
  finalize_kernel_.launch(1,
                          static_cast<unsigned int>(manifest_.finalize.block_dim),
                          static_cast<unsigned int>(manifest_.finalize.shared_memory),
                          stream, arguments);
}

void TrajectoryOptimizerRuntime::validate_request(
    const TrajectoryOptimizationRequest& request) const {
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  if (request.start_position == nullptr || request.goal_position == nullptr ||
      request.optimized_knots == nullptr ||
      request.trajectory_position == nullptr ||
      request.trajectory_velocity == nullptr ||
      request.trajectory_acceleration == nullptr ||
      request.trajectory_jerk == nullptr || request.success == nullptr ||
      request.cost == nullptr || request.collision_cost == nullptr ||
      request.trajectory_dt == nullptr) {
    throw std::invalid_argument(
        "TrajOpt input and output CUDA addresses must be non-null");
  }
  if (request.finetune_attempts < -1 ||
      (request.finetune_dt_scale != -1.0F &&
       (!std::isfinite(request.finetune_dt_scale) ||
        request.finetune_dt_scale <= 0.0F ||
        request.finetune_dt_scale > 1.0F))) {
    throw std::invalid_argument(
        "TrajOpt request finetuning override is outside its valid range");
  }
  if (request.constrain_tool_path) {
    float direction_norm_squared = 0.0F;
    for (const float value : request.free_position_direction) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "TrajOpt Cartesian free direction must be finite");
      }
      direction_norm_squared += value * value;
    }
    if (request.goal_tool_positions == nullptr ||
        request.goal_tool_quaternions == nullptr ||
        !(direction_norm_squared > 0.0F)) {
      throw std::invalid_argument(
          "constrained TrajOpt needs goal tool poses and a nonzero free direction");
    }
    for (const float value : {request.path_position_tolerance,
                              request.path_orientation_tolerance}) {
      if (!std::isfinite(value) || (value < 0.0F && value != -1.0F) ||
          value == 0.0F) {
        throw std::invalid_argument(
            "TrajOpt Cartesian path tolerances must be positive or -1");
      }
    }
  }
}

int TrajectoryOptimizerRuntime::resolved_finetune_attempts(
    const TrajectoryOptimizationRequest& request) const {
  return request.finetune_attempts < 0 ? config_.finetune_attempts
                                      : request.finetune_attempts;
}

void TrajectoryOptimizerRuntime::begin_optimization(
    const TrajectoryOptimizationRequest& request) const {
  const auto stream = reinterpret_cast<CUstream>(request.stream);
  initialize_actions(request, stream);
  // The first pass and refinement use independent timestep reductions.
  compute_required_dt(seed_actions_.ptr(), config_.initial_dt_scale, request, stream);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  finetune_enabled_.clear(cuda_stream);
  // The reusable pass graph must preserve L-BFGS history for finetuning. Clear
  // all history once here so its first replay still has base-pass semantics.
  history_rho_.clear(cuda_stream);
  history_y_.clear(cuda_stream);
  history_s_.clear(cuda_stream);
  history_head_.clear(cuda_stream);
  history_count_.clear(cuda_stream);
}

void TrajectoryOptimizerRuntime::run_optimization_pass(
    const TrajectoryOptimizationRequest& request, bool refinement) const {
  const auto stream = reinterpret_cast<CUstream>(request.stream);
  optimize_pass(request, stream, refinement ? config_.refinement_iterations
                                          : config_.max_iterations);
  // Validate and retain the optimized trajectory at its required dt on every
  // pass. This is part of Python's base TrajOpt contract, not only finetuning.
  compute_required_dt(best_action_.ptr(), 1.0F, request, stream);
  evaluate(best_action_.ptr(), config_.num_seeds, search_costs_.ptr(),
           search_gradients_.ptr(), request, stream);
  launch_metrics_world_collision(config_.num_seeds, stream);
  validate_interpolated(best_action_.ptr(), stream);
}

void TrajectoryOptimizerRuntime::accumulate_optimization_pass(
    bool first_pass, const TrajectoryOptimizationRequest& request) const {
  accumulate(first_pass, request, reinterpret_cast<CUstream>(request.stream));
}

void TrajectoryOptimizerRuntime::prepare_finetune_pass(
    const TrajectoryOptimizationRequest& request) const {
  const auto action_bytes = static_cast<size_t>(config_.num_seeds) *
                            manifest_.num_knots * manifest_.num_dof *
                            sizeof(float);
  const auto dt_bytes =
      static_cast<size_t>(config_.num_seeds) * sizeof(float);
  const auto stream = request.stream;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(seed_actions_.ptr()),
      reinterpret_cast<void*>(retained_action_.ptr()), action_bytes,
      cudaMemcpyDeviceToDevice, stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(trajectory_dt_.ptr()),
      reinterpret_cast<void*>(retained_dt_.ptr()), dt_bytes,
      cudaMemcpyDeviceToDevice, stream));
  const float finetune_dt_scale =
      request.finetune_dt_scale < 0.0F ? config_.finetune_dt_scale
                                       : request.finetune_dt_scale;
  compute_required_dt(seed_actions_.ptr(), finetune_dt_scale, request,
                      reinterpret_cast<CUstream>(stream));
}

bool TrajectoryOptimizerRuntime::base_pass_succeeded(
    cudaStream_t stream) const {
  int32_t enabled = 0;
  CRR_CUDA_CHECK(cudaMemcpyAsync(&enabled,
                                 reinterpret_cast<const void*>(
                                     finetune_enabled_.ptr()),
                                 sizeof(enabled), cudaMemcpyDeviceToHost,
                                 stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return enabled != 0;
}

void TrajectoryOptimizerRuntime::finish_optimization(
    const TrajectoryOptimizationRequest& request) const {
  const auto stream = request.stream;
  if (!timeopt_) {
    finalize(request, reinterpret_cast<CUstream>(stream));
    return;
  }
  const int batch = config_.num_seeds, dof = manifest_.num_dof;
  const int capacity = manifest_.interpolation_buffer_size;
  const int interpolation_steps =
      (manifest_.trajectory_steps - 1) / (manifest_.num_knots + manifest_.spline_degree + 1);
  auto* boundaries = reinterpret_cast<float*>(timeopt_boundaries_.ptr());
  auto* knot_dt = reinterpret_cast<float*>(timeopt_knot_dt_.ptr());
  CRR_CUDA_CHECK(launch_timeopt_boundaries(
      reinterpret_cast<const float*>(retained_position_.ptr()),
      reinterpret_cast<const float*>(retained_dt_.ptr()), request.start_velocity,
      request.start_acceleration, batch, dof, manifest_.trajectory_steps,
      interpolation_steps, boundaries, knot_dt, stream));
  TrajectoryTimeOptRequest timeopt;
  timeopt.knots = reinterpret_cast<const float*>(retained_action_.ptr());
  timeopt.start_position = boundaries;
  timeopt.goal_position = boundaries + batch * dof;
  timeopt.start_velocity = boundaries + 2 * batch * dof;
  timeopt.start_acceleration = boundaries + 3 * batch * dof;
  timeopt.knot_dt = knot_dt;
  timeopt.stream = stream;
  const auto result = timeopt_->timeopt(timeopt);
  auto* times = reinterpret_cast<float*>(timeopt_times_.ptr());
  auto* dt = reinterpret_cast<float*>(timeopt_dt_.ptr());
  auto* counts = reinterpret_cast<int32_t*>(timeopt_counts_.ptr());
  CRR_CUDA_CHECK(launch_timeopt_times(result.duration, batch, capacity,
      manifest_.interpolation_dt, times, dt, counts, stream));
  const auto samples = timeopt_->sample(times, stream);
  auto* scaled = reinterpret_cast<float*>(timeopt_scaled_states_.ptr());
  const size_t values = static_cast<size_t>(batch) * capacity * dof;
  CRR_CUDA_CHECK(launch_timeopt_scale(samples,
      reinterpret_cast<const float*>(execution_speed_scale_.ptr()), values, scaled, stream));
  const TrajectoryValidationRequest validation{samples.position, scaled,
      scaled + values, scaled + 2 * values, capacity, stream};
  const auto* violation = validate(validation, true);
  // Dynamic infeasibility may be repaired; preserve seed and pose constraints.
  CRR_CUDA_CHECK(launch_timeopt_rank(result.success, violation, counts,
      reinterpret_cast<const uint8_t*>(active_seed_valid_.ptr()),
      reinterpret_cast<const uint8_t*>(retained_pose_valid_.ptr()), result.duration, batch,
      reinterpret_cast<uint8_t*>(retained_success_.ptr()),
      reinterpret_cast<float*>(retained_rank_score_.ptr()), stream));
  finalize(request, reinterpret_cast<CUstream>(stream));
  const auto* selected = request.selected_seed ? request.selected_seed
      : reinterpret_cast<const int32_t*>(selected_seed_.ptr());
  CRR_CUDA_CHECK(launch_timeopt_output(samples, result.timing_controls, dt, result.duration, counts,
      selected, request.success, capacity, dof,
      reinterpret_cast<float*>(timeopt_states_.ptr()),
      reinterpret_cast<float*>(timeopt_metadata_.ptr()),
      reinterpret_cast<int32_t*>(timeopt_count_.ptr()),
      reinterpret_cast<float*>(timeopt_timing_controls_.ptr()), stream));
  CRR_CUDA_CHECK(launch_timeopt_spatial_output(
      timeopt.knots, boundaries, knot_dt, selected, request.success, batch,
      manifest_.num_knots, dof,
      reinterpret_cast<float*>(timeopt_spatial_controls_.ptr()), stream));
}

void TrajectoryOptimizerRuntime::set_execution_speed_scale(float scale) {
  if (!timeopt_ || !std::isfinite(scale) || scale <= 0.0F || scale > 1.0F) {
    throw std::invalid_argument("execution speed requires enabled timeopt and a scale in (0,1]");
  }
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  CRR_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(execution_speed_scale_.ptr()), &scale,
                            sizeof(scale), cudaMemcpyHostToDevice));
  config_.execution_speed_scale = scale;
}

TimeOptTrajectory TrajectoryOptimizerRuntime::timeopt_trajectory() const {
  if (!timeopt_) return {};
  const auto* states = reinterpret_cast<const float*>(timeopt_states_.ptr());
  const auto* metadata = reinterpret_cast<const float*>(timeopt_metadata_.ptr());
  const int capacity = manifest_.interpolation_buffer_size;
  const size_t values = static_cast<size_t>(capacity) * manifest_.num_dof;
  return {states, states + values, states + 2 * values, states + 3 * values,
          metadata, metadata + 1, reinterpret_cast<const int32_t*>(timeopt_count_.ptr()), capacity,
          reinterpret_cast<const float*>(timeopt_timing_controls_.ptr()), TIMEOPT_CONTROL_COUNT,
          reinterpret_cast<const float*>(timeopt_spatial_controls_.ptr()), manifest_.num_knots + 6};
}

void TrajectoryOptimizerRuntime::optimize(
    const TrajectoryOptimizationRequest& request) const {
  validate_request(request);
  const int finetune_attempts = resolved_finetune_attempts(request);
  begin_optimization(request);
  for (int attempt = 0; attempt <= finetune_attempts; ++attempt) {
    run_optimization_pass(request, attempt > 0);
    accumulate_optimization_pass(attempt == 0, request);
    if (attempt < finetune_attempts) {
      prepare_finetune_pass(request);
    }
  }
  finish_optimization(request);
}

}  // namespace curobo::robot::detail
