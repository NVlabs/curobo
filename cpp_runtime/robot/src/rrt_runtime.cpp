// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rrt_runtime.h"

#include "attachment_runtime.h"
#include "cuda_check.h"
#include "curobo_robot_runtime/scene.h"
#include "rollout_runtime.h"
#include "rollout_streams.h"
#include "scene_runtime.h"
#include "warp_abi.h"

#include <algorithm>
#include <array>
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

constexpr int MAX_RRT_PLANS = 128;
constexpr uint32_t CUBOID_COLLISION_BIT = 1U << 0U;
constexpr uint32_t CAPSULE_COLLISION_BIT = 1U << 1U;
constexpr uint32_t CYLINDER_COLLISION_BIT = 1U << 2U;
constexpr uint32_t MESH_COLLISION_BIT = 1U << 3U;
constexpr uint32_t VOXEL_COLLISION_BIT = 1U << 4U;

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
void launch_collision_kernel(
    const curobo_robot_runtime::KernelModule& kernel,
    const curobo_robot_runtime::IkKernelManifest& manifest,
    const ObstacleData& obstacle_data, CUdeviceptr obstacle_layers,
    CUdeviceptr sphere_collision_mask, CUdeviceptr robot_spheres,
    CUdeviceptr weight, CUdeviceptr activation_distance,
    CUdeviceptr scene_indices, CUdeviceptr collision_cost,
    CUdeviceptr collision_gradient, int states, int num_spheres, bool overwrite,
    CUstream stream) {
  const auto bounds = warp_tiled_bounds_2d(states, manifest.block_dim);
  auto spheres =
      array_1d(robot_spheres, states * num_spheres, 4 * sizeof(float));
  auto weight_array = array_1d(weight, 1, sizeof(float));
  auto activation_array = array_1d(activation_distance, 1, sizeof(float));
  auto scene_index_array = array_1d(scene_indices, states, sizeof(int32_t));
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

RrtRuntime::RrtRuntime(std::string artifact_directory, const RrtConfig& config,
                       const Scene* scene,
                       std::shared_ptr<RobotCollisionModel> collision)
    : artifact_directory_(std::move(artifact_directory)),
      config_(config),
      scene_(scene == nullptr ? nullptr : scene->runtime_.get()),
      robot_(artifact_directory_, config.device_ordinal),
      manifest_(curobo_robot_runtime::load_rrt_manifest(artifact_directory_)) {
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  const auto rollout_manifest =
      std::filesystem::path(artifact_directory_) / "robot_rollout_manifest.json";
  if (std::filesystem::exists(rollout_manifest)) {
    rollout_ = std::make_unique<RolloutRuntime>(
        artifact_directory_, config_.device_ordinal,
        robot_.manifest(), RolloutProfile::RRT);
  } else if (config_.check_effort_limits) {
    throw std::runtime_error(
        "RRT effort limits require a robot rollout artifact");
  }
  validate_config();
  validate_scene();
  collision_ = std::move(collision);
  load_constants();
  load_kernels();
  allocate_workspace();
  rollout_streams_ = std::make_unique<RolloutStreams>();
  CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&graph_capture_stream_,
                                            cudaStreamNonBlocking));
}

RrtRuntime::~RrtRuntime() {
  cudaSetDevice(config_.device_ordinal);
  for (auto& graph : search_graphs_) {
    destroy_search_graph(&graph);
  }
  for (auto& graph : finalization_graphs_) {
    destroy_finalization_graph(&graph);
  }
  if (graph_capture_stream_ != nullptr) {
    cudaStreamDestroy(graph_capture_stream_);
  }
}

CuroboRobotInfo RrtRuntime::robot_info() const { return robot_.info(); }

int RrtRuntime::attachment_sphere_capacity(
    const std::string& link_name) const {
  return collision_->attachment->capacity(link_name);
}

void RrtRuntime::attach_object(const SphereAttachment& attachment) {
  collision_->attachment->attach(attachment);
}

void RrtRuntime::detach_object(const std::string& link_name) {
  collision_->attachment->detach(link_name);
}

void RrtRuntime::set_link_collision_mask(const std::string& link,
                                         uint32_t layers) {
  collision_->attachment->set_collision_mask(link, layers,
                                             collision_->sphere_collision_mask);
}

void RrtRuntime::set_link_collision_enabled(const std::string& link_name,
                                            bool enabled) {
  collision_->attachment->set_collision_enabled(link_name, enabled);
}

void RrtRuntime::update_link_inertial(const LinkInertialUpdate& update) {
  if (rollout_ == nullptr) {
    throw std::runtime_error("RRT artifact has no inverse-dynamics model");
  }
  rollout_->update_link_inertial(update);
}

void RrtRuntime::update_links_inertial(
    const LinkInertialUpdates& updates) {
  if (rollout_ == nullptr) {
    throw std::runtime_error("RRT artifact has no inverse-dynamics model");
  }
  rollout_->update_links_inertial(updates);
}

void RrtRuntime::reset_seed() {
  sample_cursor_ = 0;
  large_sample_cursor_ = 0;
}

RrtInfo RrtRuntime::info() const {
  return RrtInfo{
      manifest_.num_dof,
      manifest_.samples_per_iteration,
      manifest_.max_iterations,
      manifest_.max_nodes_per_tree,
      manifest_.max_path_nodes,
      config_.max_plans,
  };
}

void RrtRuntime::validate_config() const {
  if (!std::isfinite(config_.collision_activation_distance) ||
      config_.collision_activation_distance < 0.0F) {
    throw std::invalid_argument("RRT collision activation distance must be finite and nonnegative");
  }
  const auto robot_info_value = robot_.info();
  if (manifest_.arch != robot_.manifest().arch ||
      manifest_.num_dof != robot_info_value.num_dof ||
      manifest_.num_links != robot_info_value.num_links ||
      manifest_.num_tool_frames != robot_info_value.num_tool_frames) {
    throw std::runtime_error(
        "robot RRT artifact does not match its kinematics artifact");
  }
  if (robot_.model_device_view().link_masses_com == 0) {
    throw std::runtime_error("RRT artifact requires sphere kinematics tensors");
  }
  if (config_.max_iterations <= 0 || config_.max_plans <= 0 ||
      config_.max_plans > MAX_RRT_PLANS ||
      config_.max_plans > manifest_.max_batched_lanes ||
      config_.max_iterations > manifest_.max_iterations) {
    throw std::invalid_argument(
        "RRT max_iterations or max_plans is outside the supported range");
  }
  if (manifest_.shortcut_state_capacity < 2 * config_.max_plans) {
    throw std::invalid_argument(
        "RRT shortcut capacity must provide two states per plan");
  }
}

void RrtRuntime::validate_scene() const {
  if (scene_ == nullptr) {
    return;
  }
  const auto& scene_config = scene_->config();
  if (scene_config.device_ordinal != config_.device_ordinal) {
    throw std::invalid_argument("RRT and Scene must use the same CUDA device");
  }
  if (scene_config.max_cuboids > manifest_.max_cuboids ||
      scene_config.max_capsules > manifest_.max_capsules ||
      scene_config.max_cylinders > manifest_.max_cylinders ||
      scene_config.max_meshes > manifest_.max_meshes ||
      scene_config.max_voxel_grids > manifest_.max_voxel_grids) {
    throw std::invalid_argument("Scene capacity exceeds the RRT artifact capacity");
  }
}

uint32_t RrtRuntime::scene_collision_mask() const {
  if (scene_ == nullptr) {
    return 0;
  }
  uint32_t mask = 0;
  if (scene_->obstacle_count(ObstacleType::CUBOID) > 0) {
    mask |= CUBOID_COLLISION_BIT;
  }
  if (scene_->obstacle_count(ObstacleType::CAPSULE) > 0 ||
      scene_->obstacle_count(ObstacleType::SPHERE) > 0) {
    mask |= CAPSULE_COLLISION_BIT;
  }
  if (scene_->obstacle_count(ObstacleType::CYLINDER) > 0) {
    mask |= CYLINDER_COLLISION_BIT;
  }
  if (scene_->obstacle_count(ObstacleType::MESH) > 0) {
    mask |= MESH_COLLISION_BIT;
  }
  if (scene_->obstacle_count(ObstacleType::VOXEL_GRID) > 0) {
    mask |= VOXEL_COLLISION_BIT;
  }
  return mask;
}

void RrtRuntime::refresh_scene_graphs() const {
  const auto current_mask = scene_collision_mask();
  if (current_mask == active_scene_collision_mask_) {
    return;
  }
  for (auto& graph : search_graphs_) {
    destroy_search_graph(&graph);
  }
  search_graphs_.clear();
  for (auto& graph : finalization_graphs_) {
    destroy_finalization_graph(&graph);
  }
  finalization_graphs_.clear();
  active_scene_collision_mask_ = current_mask;
}

void RrtRuntime::load_constants() {
  const auto load = [&](const curobo_robot_runtime::TensorManifest& tensor,
                        const char* name,
                        curobo_robot_runtime::DeviceBuffer* destination) {
    destination->upload(curobo_robot_runtime::load_tensor_data(
        artifact_directory_, tensor, name));
  };
  load(manifest_.position_lower, "position_lower", &position_lower_);
  load(manifest_.position_upper, "position_upper", &position_upper_);
  std::vector<float> effort_bounds_fallback(
      static_cast<size_t>(2 * manifest_.num_dof));
  std::fill_n(effort_bounds_fallback.begin(), manifest_.num_dof,
              -std::numeric_limits<float>::infinity());
  std::fill_n(effort_bounds_fallback.begin() + manifest_.num_dof,
              manifest_.num_dof, std::numeric_limits<float>::infinity());
  effort_bounds_fallback_.upload(as_bytes(effort_bounds_fallback));
  load(manifest_.distance_weight, "distance_weight", &distance_weight_);
  load(manifest_.samples, "samples", &samples_);
  load(manifest_.shortcut_pair_sources, "shortcut_pair_sources",
       &shortcut_pair_sources_);
  load(manifest_.shortcut_pair_destinations, "shortcut_pair_destinations",
       &shortcut_pair_destinations_);
  load(manifest_.shortcut_edge_lookup, "shortcut_edge_lookup",
       &shortcut_edge_lookup_);
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
  collision_weight_.upload(as_bytes(std::vector<float>{1.0F}));
  collision_activation_distance_.upload(
      as_bytes(std::vector<float>{config_.collision_activation_distance}));
  const int max_states = std::max(
      {2 * config_.max_plans,
       config_.max_plans * manifest_.samples_per_iteration *
           manifest_.fixed_steer_steps,
       2 * config_.max_plans * manifest_.direct_steer_steps,
       manifest_.shortcut_state_capacity});
  scene_indices_.upload(
      as_bytes(std::vector<int32_t>(static_cast<size_t>(max_states), 0)));
}

void RrtRuntime::load_kernels() {
  kinematics_kernel_.load(
      (std::filesystem::path(artifact_directory_) / manifest_.kinematics.cubin)
          .string(),
      manifest_.kinematics.symbol);
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
  load_kernel(artifact_directory_, manifest_.self_collision,
              &self_collision_kernel_);
  load_kernel(artifact_directory_, manifest_.nearest_target,
              &nearest_target_kernel_);
  load_kernel(artifact_directory_, manifest_.packed_append,
              &packed_append_kernel_);
  load_kernel(artifact_directory_, manifest_.reconstruct_path,
              &reconstruct_path_kernel_);
  load_kernel(artifact_directory_, manifest_.prepare_samples,
              &prepare_samples_kernel_);
  load_kernel(artifact_directory_, manifest_.steer_edges,
              &steer_edges_kernel_);
  load_kernel(artifact_directory_, manifest_.evaluate_edges,
              &evaluate_edges_kernel_);
  load_kernel(artifact_directory_, manifest_.prepare_direct_edges,
              &prepare_direct_edges_kernel_);
  load_kernel(artifact_directory_, manifest_.interpolate_direct_edges,
              &interpolate_direct_edges_kernel_);
  load_kernel(artifact_directory_, manifest_.reduce_direct_edges,
              &reduce_direct_edges_kernel_);
  load_kernel(artifact_directory_, manifest_.classify_states,
              &classify_states_kernel_);
  load_kernel(artifact_directory_, manifest_.select_connection,
              &select_connection_kernel_);
  load_kernel(artifact_directory_, manifest_.shortcut_setup,
              &shortcut_setup_kernel_);
  load_kernel(artifact_directory_, manifest_.shortcut_fill_edges,
              &shortcut_fill_edges_kernel_);
  load_kernel(artifact_directory_, manifest_.shortcut_reduce_edges,
              &shortcut_reduce_edges_kernel_);
  load_kernel(artifact_directory_, manifest_.shortcut_select_path,
              &shortcut_select_path_kernel_);
  if (manifest_.num_self_collision_pairs > 0 &&
      manifest_.self_collision.shared_memory > 48000) {
    self_collision_kernel_.set_max_dynamic_shared_memory(
        manifest_.self_collision.shared_memory);
  }
}

void RrtRuntime::allocate_workspace() {
  const size_t dof = static_cast<size_t>(manifest_.num_dof);
  const size_t candidates =
      static_cast<size_t>(manifest_.samples_per_iteration);
  const size_t plans = static_cast<size_t>(config_.max_plans);
  const size_t tree_capacity =
      static_cast<size_t>(manifest_.max_nodes_per_tree);
  const size_t max_states = std::max(
      {checked_product(2, plans, "RRT terminal states"),
       checked_product(
           checked_product(plans, candidates, "RRT candidate edges"),
           static_cast<size_t>(manifest_.fixed_steer_steps),
           "RRT steering states"),
       checked_product(
           checked_product(2, plans, "RRT direct edges"),
           static_cast<size_t>(manifest_.direct_steer_steps),
           "RRT direct states"),
       static_cast<size_t>(manifest_.shortcut_state_capacity)});
  const auto allocate_float = [](curobo_robot_runtime::DeviceBuffer* buffer,
                                 size_t count) {
    buffer->allocate(checked_product(count, sizeof(float), "float buffer"));
  };
  allocate_float(&start_nodes_, plans * tree_capacity * dof);
  allocate_float(&goal_nodes_, plans * tree_capacity * dof);
  start_parents_.allocate(plans * tree_capacity * sizeof(int64_t));
  goal_parents_.allocate(plans * tree_capacity * sizeof(int64_t));
  start_count_.allocate(plans * sizeof(int32_t));
  goal_count_.allocate(plans * sizeof(int32_t));
  for (auto* buffer : {
           &desired_,
           &active_starts_,
           &active_endpoints_,
           &passive_starts_,
           &passive_endpoints_,
           &targets_,
       }) {
    allocate_float(buffer, plans * candidates * dof);
  }
  for (auto* buffer : {&active_parents_, &active_slots_, &passive_parents_,
                       &passive_slots_}) {
    buffer->allocate(plans * candidates * sizeof(int64_t));
  }
  for (auto* buffer : {&active_reached_, &active_progressed_,
                       &active_written_, &passive_reached_,
                       &passive_progressed_, &passive_written_}) {
    buffer->allocate(plans * candidates * sizeof(bool));
  }
  allocate_float(&line_points_, max_states * dof);
  line_valid_.allocate(max_states * sizeof(bool));
  connection_indices_.allocate(2 * plans * sizeof(int64_t));
  growth_status_.allocate(5 * plans * sizeof(int64_t));
  allocate_float(&terminal_positions_, 2 * plans * dof);
  state_feasible_.allocate(max_states * sizeof(bool));
  active_plans_.allocate(plans * sizeof(bool));
  direct_point_count_.allocate(sizeof(int32_t));
  direct_terminal_feasible_.allocate(2 * plans * sizeof(bool));
  allocate_float(&reconstruction_path_lengths_, plans);

  const size_t shortcut_path_capacity =
      static_cast<size_t>(manifest_.shortcut_path_capacity);
  const size_t shortcut_edge_capacity =
      static_cast<size_t>(manifest_.shortcut_edge_capacity);
  const size_t shortcut_state_capacity =
      static_cast<size_t>(manifest_.shortcut_state_capacity);
  const size_t raw_path_capacity = static_cast<size_t>(manifest_.max_path_nodes);
  allocate_float(&shortcut_raw_path_, plans * raw_path_capacity * dof);
  shortcut_path_count_.allocate(plans * sizeof(int32_t));
  shortcut_edge_offsets_.allocate(plans * shortcut_edge_capacity *
                                  sizeof(int32_t));
  shortcut_edge_steps_.allocate(plans * shortcut_edge_capacity *
                                sizeof(int32_t));
  shortcut_edge_tested_.allocate(plans * shortcut_edge_capacity * sizeof(bool));
  shortcut_edge_visible_.allocate(plans * shortcut_edge_capacity *
                                  sizeof(bool));
  allocate_float(&shortcut_packed_states_, shortcut_state_capacity * dof);
  shortcut_dp_hops_.allocate(plans * shortcut_path_capacity * sizeof(int32_t));
  allocate_float(&shortcut_dp_cost_, plans * shortcut_path_capacity);
  shortcut_parents_.allocate(plans * shortcut_path_capacity * sizeof(int32_t));
  allocate_float(&shortcut_output_path_, plans * raw_path_capacity * dof);
  shortcut_output_indices_.allocate(plans * raw_path_capacity * sizeof(int32_t));
  shortcut_status_.allocate(plans * 5 * sizeof(int32_t));
  allocate_float(&shortcut_path_lengths_, plans);
  shortcut_success_.allocate(plans * sizeof(bool));

  allocate_float(&link_position_,
                 max_states * manifest_.num_tool_frames * 3);
  allocate_float(&link_quaternion_,
                 max_states * manifest_.num_tool_frames * 4);
  allocate_float(&global_cumulative_,
                 max_states * manifest_.num_links * 12);
  allocate_float(&robot_spheres_, max_states * manifest_.num_spheres * 4);
  allocate_float(&world_collision_cost_, max_states);
  allocate_float(&world_collision_gradient_,
                 max_states * manifest_.num_spheres * 4);
  allocate_float(&self_collision_cost_, max_states);
  allocate_float(&self_collision_gradient_,
                 max_states * manifest_.num_spheres * 4);
  allocate_float(&self_collision_pair_distance_, 1);
  self_collision_sparse_index_.allocate(
      max_states * manifest_.num_spheres * sizeof(uint8_t));
  allocate_float(&joint_effort_, max_states * dof);
  allocate_float(&zero_derivatives_, max_states * dof);
  if (config_.check_effort_limits) {
    allocate_float(
        &rnea_forward_cache_,
        max_states * static_cast<size_t>(rollout_->num_links()) *
            static_cast<size_t>(rollout_->cache_floats_per_link()));
  }
  // The tiled nearest-neighbor kernel loads a complete final tile before
  // masking lanes beyond each tree count.
  start_nodes_.clear();
  goal_nodes_.clear();
  desired_.clear();
  active_starts_.clear();
  passive_starts_.clear();
  world_collision_gradient_.clear();
  self_collision_gradient_.clear();
  self_collision_sparse_index_.clear();
  joint_effort_.clear();
  zero_derivatives_.clear();
  shortcut_packed_states_.clear();
  shortcut_status_.clear();
}

void RrtRuntime::initialize(const RrtRequest& request, int plans,
                            CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const size_t dof_bytes =
      static_cast<size_t>(manifest_.num_dof) * sizeof(float);
  const size_t tree_pitch =
      static_cast<size_t>(manifest_.max_nodes_per_tree) * dof_bytes;
  CRR_CUDA_CHECK(cudaMemcpy2DAsync(
      reinterpret_cast<void*>(start_nodes_.ptr()), tree_pitch,
      request.start_position, dof_bytes, dof_bytes,
      static_cast<size_t>(plans), cudaMemcpyDeviceToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpy2DAsync(
      reinterpret_cast<void*>(goal_nodes_.ptr()), tree_pitch,
      request.goal_position, dof_bytes, dof_bytes,
      static_cast<size_t>(plans), cudaMemcpyDeviceToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemsetAsync(
      reinterpret_cast<void*>(start_parents_.ptr()), 0xff,
      start_parents_.bytes(), cuda_stream));
  CRR_CUDA_CHECK(cudaMemsetAsync(
      reinterpret_cast<void*>(goal_parents_.ptr()), 0xff,
      goal_parents_.bytes(), cuda_stream));
  const std::vector<int32_t> one(static_cast<size_t>(plans), 1);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(start_count_.ptr()), one.data(),
      one.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(goal_count_.ptr()), one.data(),
      one.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice, cuda_stream));
  const std::vector<int64_t> no_connection(static_cast<size_t>(2 * plans), -1);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(connection_indices_.ptr()), no_connection.data(),
      no_connection.size() * sizeof(int64_t), cudaMemcpyHostToDevice,
      cuda_stream));
  const size_t endpoint_bytes = static_cast<size_t>(plans) * dof_bytes;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(terminal_positions_.ptr()), request.start_position,
      endpoint_bytes, cudaMemcpyDeviceToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(terminal_positions_.ptr() + endpoint_bytes),
      request.goal_position, endpoint_bytes, cudaMemcpyDeviceToDevice,
      cuda_stream));
  CRR_CUDA_CHECK(cudaMemsetAsync(request.success, 0,
                                 static_cast<size_t>(plans) * sizeof(uint8_t),
                                 cuda_stream));
  CRR_CUDA_CHECK(cudaMemsetAsync(request.path_count, 0,
                                 static_cast<size_t>(plans) * sizeof(int32_t),
                                 cuda_stream));
  CRR_CUDA_CHECK(cudaMemsetAsync(request.iterations, 0,
                                 static_cast<size_t>(plans) * sizeof(int32_t),
                                 cuda_stream));
}

void RrtRuntime::launch_kinematics(CUdeviceptr positions, int states,
                                   CUstream stream) const {
  const auto& policy = manifest_.kinematics;
  auto model = robot_.model_device_view();
  const int shared_per_batch = manifest_.num_links * 12 * sizeof(float) * 2;
  const int batches_per_block = std::min(
      {states, policy.max_batches_per_block,
       policy.max_threads_per_block / policy.threads_per_batch,
       policy.max_dynamic_shared_memory / shared_per_batch});
  if (batches_per_block <= 0) {
    throw std::runtime_error("RRT kinematics launch policy is invalid");
  }
  const auto block =
      static_cast<unsigned int>(batches_per_block * policy.threads_per_batch);
  const auto grid = static_cast<unsigned int>(
      (states + batches_per_block - 1) / batches_per_block);
  const auto shared =
      static_cast<unsigned int>(batches_per_block * shared_per_batch);

  CUdeviceptr link_position = link_position_.ptr();
  CUdeviceptr link_quaternion = link_quaternion_.ptr();
  CUdeviceptr spheres = robot_spheres_.ptr();
  CUdeviceptr null_pointer = 0;
  CUdeviceptr global_cumulative = global_cumulative_.ptr();
  CUdeviceptr link_spheres = collision_->link_spheres.ptr();
  CUdeviceptr link_sphere_map = collision_->link_sphere_map.ptr();
  CUdeviceptr scene_indices = scene_indices_.ptr();
  int batch_size = states;
  int horizon = 1;
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
      &positions,
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

void RrtRuntime::launch_world_collision(int states, CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (scene_ == nullptr) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_cost_.ptr()), 0,
        static_cast<size_t>(states) * sizeof(float), cuda_stream));
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_gradient_.ptr()), 0,
        static_cast<size_t>(states) * manifest_.num_spheres * 4 * sizeof(float),
        cuda_stream));
    return;
  }
  const auto view = scene_->device_view();
  bool overwrite = true;
  const auto launch =
      [&](const auto& obstacle_data, CUdeviceptr layers,
          const curobo_robot_runtime::IkKernelManifest& kernel_manifest,
          const curobo_robot_runtime::KernelModule& kernel) {
        if (kernel_manifest.cubin.empty()) {
          return;
        }
        launch_collision_kernel(
            kernel, kernel_manifest, obstacle_data, layers,
            collision_->sphere_collision_mask.ptr(), robot_spheres_.ptr(),
            collision_weight_.ptr(), collision_activation_distance_.ptr(),
            scene_indices_.ptr(), world_collision_cost_.ptr(),
            world_collision_gradient_.ptr(), states, manifest_.num_spheres,
            overwrite, stream);
        overwrite = false;
      };
  if ((active_scene_collision_mask_ & CUBOID_COLLISION_BIT) != 0U) {
    launch(make_cuboid_data(view.cuboids), view.cuboids.obstacle_layers,
           manifest_.collision_cuboid, collision_cuboid_kernel_);
  }
  if ((active_scene_collision_mask_ & CAPSULE_COLLISION_BIT) != 0U) {
    launch(make_capsule_data(view.capsules), view.capsules.obstacle_layers,
           manifest_.collision_capsule, collision_capsule_kernel_);
  }
  if ((active_scene_collision_mask_ & CYLINDER_COLLISION_BIT) != 0U) {
    launch(make_capsule_data(view.cylinders), view.cylinders.obstacle_layers,
           manifest_.collision_cylinder, collision_cylinder_kernel_);
  }
  if ((active_scene_collision_mask_ & MESH_COLLISION_BIT) != 0U) {
    launch(make_mesh_data(view.meshes), view.meshes.obstacle_layers,
           manifest_.collision_mesh, collision_mesh_kernel_);
  }
  if ((active_scene_collision_mask_ & VOXEL_COLLISION_BIT) != 0U) {
    launch(make_voxel_data(view.voxel_grids), view.voxel_grids.obstacle_layers,
           manifest_.collision_voxel, collision_voxel_kernel_);
  }
  if (overwrite) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_cost_.ptr()), 0,
        static_cast<size_t>(states) * sizeof(float), cuda_stream));
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(world_collision_gradient_.ptr()), 0,
        static_cast<size_t>(states) * manifest_.num_spheres * 4 * sizeof(float),
        cuda_stream));
  }
}

void RrtRuntime::launch_self_collision(int states, CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (!config_.check_self_collision ||
      manifest_.num_self_collision_pairs == 0) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(self_collision_cost_.ptr()), 0,
        static_cast<size_t>(states) * sizeof(float), cuda_stream));
    return;
  }
  CUdeviceptr distance = self_collision_cost_.ptr();
  CUdeviceptr gradient = self_collision_gradient_.ptr();
  CUdeviceptr pair_distance = self_collision_pair_distance_.ptr();
  CUdeviceptr sparse_index = self_collision_sparse_index_.ptr();
  CUdeviceptr spheres = robot_spheres_.ptr();
  CUdeviceptr padding = self_collision_padding_.ptr();
  CUdeviceptr weight = collision_weight_.ptr();
  CUdeviceptr pairs = self_collision_pairs_.ptr();
  int batch_size = states;
  int horizon = 1;
  int num_spheres = manifest_.num_spheres;
  int num_pairs = manifest_.num_self_collision_pairs;
  std::vector<void*> arguments{
      &distance, &gradient, &pair_distance, &sparse_index, &spheres, &padding,
      &weight,   &pairs,    &batch_size,    &horizon,      &num_spheres,
      &num_pairs,
  };
  self_collision_kernel_.launch(
      static_cast<unsigned int>(states),
      static_cast<unsigned int>(manifest_.self_collision.block_dim),
      static_cast<unsigned int>(manifest_.self_collision.shared_memory), stream,
      arguments);
}

void RrtRuntime::evaluate_positions(CUdeviceptr positions, int states,
                                    CUstream stream) const {
  auto* streams = rollout_streams_.get();
  const auto primary = reinterpret_cast<cudaStream_t>(stream);
  const auto dynamics = streams->cuda_stream(RolloutStream::DYNAMICS);
  const auto kinematics = streams->cuda_stream(RolloutStream::KINEMATICS);
  const auto world = streams->cuda_stream(RolloutStream::WORLD);
  const auto self = streams->cuda_stream(RolloutStream::SELF);
  streams->record(RolloutEvent::STATE_READY, primary);

  if (config_.check_effort_limits) {
    streams->wait(dynamics, RolloutEvent::STATE_READY);
    rollout_->launch_inverse_dynamics(
        positions, zero_derivatives_.ptr(), zero_derivatives_.ptr(),
        joint_effort_.ptr(), rnea_forward_cache_.ptr(), states,
        streams->driver_stream(RolloutStream::DYNAMICS));
    streams->record(RolloutEvent::DYNAMICS_READY, dynamics);
  }

  streams->wait(kinematics, RolloutEvent::STATE_READY);
  launch_kinematics(positions, states,
                    streams->driver_stream(RolloutStream::KINEMATICS));
  streams->record(RolloutEvent::KINEMATICS_READY, kinematics);
  streams->wait(world, RolloutEvent::KINEMATICS_READY);
  streams->wait(self, RolloutEvent::KINEMATICS_READY);
  launch_world_collision(states,
                         streams->driver_stream(RolloutStream::WORLD));
  streams->record(RolloutEvent::WORLD_DONE, world);
  launch_self_collision(states,
                        streams->driver_stream(RolloutStream::SELF));
  streams->record(RolloutEvent::SELF_DONE, self);
  streams->wait(primary, RolloutEvent::WORLD_DONE);
  streams->wait(primary, RolloutEvent::SELF_DONE);
  if (config_.check_effort_limits) {
    streams->wait(primary, RolloutEvent::DYNAMICS_READY);
  }
}

void RrtRuntime::classify_states(CUdeviceptr positions, int states,
                                 CUdeviceptr feasible, CUstream stream) const {
  const auto bounds = warp_bounds_1d(states);
  auto position_array =
      array_2d(positions, states, manifest_.num_dof, sizeof(float));
  auto world_cost =
      array_1d(world_collision_cost_.ptr(), states, sizeof(float));
  auto self_cost = array_1d(self_collision_cost_.ptr(), states, sizeof(float));
  auto effort = array_1d(joint_effort_.ptr(),
                         states * manifest_.num_dof, sizeof(float));
  auto lower =
      array_1d(position_lower_.ptr(), manifest_.num_dof, sizeof(float));
  auto upper =
      array_1d(position_upper_.ptr(), manifest_.num_dof, sizeof(float));
  const auto dof_bytes =
      static_cast<CUdeviceptr>(manifest_.num_dof * sizeof(float));
  const auto effort_bounds =
      rollout_ == nullptr ? effort_bounds_fallback_.ptr()
                          : rollout_->effort_bounds();
  auto effort_lower =
      array_1d(effort_bounds, manifest_.num_dof, sizeof(float));
  auto effort_upper = array_1d(effort_bounds + dof_bytes,
                               manifest_.num_dof, sizeof(float));
  auto feasible_array = array_1d(feasible, states, sizeof(bool));
  uint8_t effort_active = config_.check_effort_limits ? 1U : 0U;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds), &position_array, &world_cost,
      &self_cost, &effort, &lower, &upper, &effort_lower, &effort_upper,
      &feasible_array, &effort_active,
  };
  classify_states_kernel_.launch(
      grid_for(states, manifest_.classify_states.block_dim),
      static_cast<unsigned int>(manifest_.classify_states.block_dim),
      static_cast<unsigned int>(manifest_.classify_states.shared_memory), stream,
      arguments);
}

void RrtRuntime::nearest(CUdeviceptr tree_nodes, CUdeviceptr tree_count,
                         CUdeviceptr desired, CUdeviceptr starts,
                         CUdeviceptr parents, int plans,
                         int queries_per_plan, CUstream stream) const {
  const int queries = plans * queries_per_plan;
  const auto bounds = warp_tiled_bounds_2d(
      queries, manifest_.nearest_target.block_dim);
  auto nodes = array_2d(tree_nodes, plans * manifest_.max_nodes_per_tree,
                        manifest_.num_dof, sizeof(float));
  auto count = array_1d(tree_count, plans, sizeof(int32_t));
  auto desired_array =
      array_2d(desired, queries, manifest_.num_dof, sizeof(float));
  auto weights =
      array_1d(distance_weight_.ptr(), manifest_.num_dof, sizeof(float));
  auto starts_array =
      array_2d(starts, queries, manifest_.num_dof, sizeof(float));
  auto parent_array = array_1d(parents, queries, sizeof(int64_t));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds), &nodes,  &count, &desired_array,
      &weights,                                &starts_array, &parent_array,
  };
  nearest_target_kernel_.launch(
      static_cast<unsigned int>(queries),
      static_cast<unsigned int>(manifest_.nearest_target.block_dim),
      static_cast<unsigned int>(manifest_.nearest_target.shared_memory), stream,
      arguments);
}

void RrtRuntime::prepare_samples(CUdeviceptr samples, int plans,
                                 CUstream stream) const {
  int candidates = manifest_.samples_per_iteration;
  const int edges = plans * candidates;
  const auto bounds = warp_bounds_2d(edges, manifest_.num_dof);
  auto sample_array = array_2d(samples, edges, manifest_.num_dof,
                               sizeof(float));
  auto desired_array = array_2d(desired_.ptr(), edges, manifest_.num_dof,
                                sizeof(float));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &sample_array,
      &desired_array,
      &candidates,
  };
  prepare_samples_kernel_.launch(
      grid_for(static_cast<size_t>(edges) * manifest_.num_dof,
               manifest_.prepare_samples.block_dim),
      static_cast<unsigned int>(manifest_.prepare_samples.block_dim),
      static_cast<unsigned int>(manifest_.prepare_samples.shared_memory),
      stream, arguments);
}

CUdeviceptr RrtRuntime::next_samples(int plans) const {
  const size_t rows = checked_product(
      static_cast<size_t>(plans),
      static_cast<size_t>(manifest_.samples_per_iteration),
      "RRT sample request");
  const size_t dof = static_cast<size_t>(manifest_.num_dof);
  const size_t row_bytes = checked_product(dof, sizeof(float),
                                           "RRT sample row");
  const size_t storage_rows = samples_.bytes() / row_bytes;
  size_t* cursor = nullptr;
  size_t cycle = 0;
  if (rows <= static_cast<size_t>(manifest_.sample_buffer_size / 2)) {
    cursor = &sample_cursor_;
    cycle = static_cast<size_t>(manifest_.sample_buffer_size);
  } else {
    cursor = &large_sample_cursor_;
    cycle = storage_rows;
  }
  if (rows > cycle) {
    throw std::invalid_argument("RRT sample request exceeds artifact storage");
  }
  if (*cursor + rows > cycle) {
    *cursor = 0;
  }
  const size_t offset = *cursor;
  *cursor += rows;
  return samples_.ptr() + checked_product(offset, row_bytes,
                                          "RRT sample offset");
}

void RrtRuntime::run_edges(CUdeviceptr starts, CUdeviceptr desired, int edges,
                           int candidates_per_plan, bool use_plan_mask,
                           CUdeviceptr endpoints, CUdeviceptr reached,
                           CUdeviceptr progressed, CUstream stream) const {
  const auto steer_bounds =
      warp_tiled_bounds_2d(edges, manifest_.steer_edges.block_dim);
  auto starts_array =
      array_2d(starts, edges, manifest_.num_dof, sizeof(float));
  auto desired_array =
      array_2d(desired, edges, manifest_.num_dof, sizeof(float));
  auto weights =
      array_1d(distance_weight_.ptr(), manifest_.num_dof, sizeof(float));
  auto targets_array =
      array_2d(targets_.ptr(), edges, manifest_.num_dof, sizeof(float));
  const int states = edges * manifest_.fixed_steer_steps;
  auto lines =
      array_2d(line_points_.ptr(), states, manifest_.num_dof, sizeof(float));
  auto line_valid = array_1d(line_valid_.ptr(), states, sizeof(bool));
  auto precomputed_feasible =
      array_1d(state_feasible_.ptr(), states, sizeof(bool));
  float max_extension_distance = manifest_.max_extension_distance;
  float cspace_resolution = manifest_.cspace_resolution;
  std::vector<void*> steer_arguments{
      const_cast<WarpLaunchBounds2*>(&steer_bounds),
      &starts_array,
      &desired_array,
      &weights,
      &targets_array,
      &lines,
      &line_valid,
      &max_extension_distance,
      &cspace_resolution,
  };
  steer_edges_kernel_.launch(
      static_cast<unsigned int>(edges),
      static_cast<unsigned int>(manifest_.steer_edges.block_dim),
      static_cast<unsigned int>(manifest_.steer_edges.shared_memory), stream,
      steer_arguments);

  evaluate_positions(line_points_.ptr(), states, stream);
  auto world_cost =
      array_1d(world_collision_cost_.ptr(), states, sizeof(float));
  auto self_cost = array_1d(self_collision_cost_.ptr(), states, sizeof(float));
  auto effort = array_1d(joint_effort_.ptr(),
                         states * manifest_.num_dof, sizeof(float));
  auto lower =
      array_1d(position_lower_.ptr(), manifest_.num_dof, sizeof(float));
  auto upper =
      array_1d(position_upper_.ptr(), manifest_.num_dof, sizeof(float));
  const auto dof_bytes =
      static_cast<CUdeviceptr>(manifest_.num_dof * sizeof(float));
  const auto effort_bounds =
      rollout_ == nullptr ? effort_bounds_fallback_.ptr()
                          : rollout_->effort_bounds();
  auto effort_lower =
      array_1d(effort_bounds, manifest_.num_dof, sizeof(float));
  auto effort_upper = array_1d(effort_bounds + dof_bytes,
                               manifest_.num_dof, sizeof(float));
  auto endpoint_array =
      array_2d(endpoints, edges, manifest_.num_dof, sizeof(float));
  auto reached_array = array_1d(reached, edges, sizeof(bool));
  auto progressed_array = array_1d(progressed, edges, sizeof(bool));
  auto plan_active = array_1d(
      active_plans_.ptr(),
      use_plan_mask ? edges / candidates_per_plan : 1, sizeof(bool));
  uint8_t mask_plans = use_plan_mask ? 1U : 0U;
  float similarity_threshold = manifest_.similarity_threshold;
  uint8_t effort_active = config_.check_effort_limits ? 1U : 0U;
  uint8_t use_precomputed_feasible = 0U;
  const auto evaluate_bounds =
      warp_tiled_bounds_2d(edges, manifest_.evaluate_edges.block_dim);
  std::vector<void*> evaluate_arguments{
      const_cast<WarpLaunchBounds2*>(&evaluate_bounds),
      &starts_array,
      &desired_array,
      &targets_array,
      &lines,
      &line_valid,
      &precomputed_feasible,
      &world_cost,
      &self_cost,
      &effort,
      &lower,
      &upper,
      &effort_lower,
      &effort_upper,
      &plan_active,
      &candidates_per_plan,
      &mask_plans,
      &similarity_threshold,
      &effort_active,
      &use_precomputed_feasible,
      &endpoint_array,
      &reached_array,
      &progressed_array,
  };
  evaluate_edges_kernel_.launch(
      static_cast<unsigned int>(edges),
      static_cast<unsigned int>(manifest_.evaluate_edges.block_dim),
      static_cast<unsigned int>(manifest_.evaluate_edges.shared_memory), stream,
      evaluate_arguments);
}

void RrtRuntime::append(CUdeviceptr tree_nodes, CUdeviceptr tree_parents,
                        CUdeviceptr tree_count, CUdeviceptr candidates,
                        CUdeviceptr candidate_parents,
                        CUdeviceptr candidate_valid,
                        CUdeviceptr assigned_slots, CUdeviceptr write_mask,
                        int plans, int candidates_per_plan,
                        CUstream stream) const {
  const int candidate_count = plans * candidates_per_plan;
  const auto bounds = warp_tiled_bounds_2d(
      plans, manifest_.packed_append.block_dim);
  auto nodes = array_2d(tree_nodes, plans * manifest_.max_nodes_per_tree,
                        manifest_.num_dof, sizeof(float));
  auto parents =
      array_1d(tree_parents, plans * manifest_.max_nodes_per_tree,
               sizeof(int64_t));
  auto count = array_1d(tree_count, plans, sizeof(int32_t));
  auto candidate_array =
      array_2d(candidates, candidate_count, manifest_.num_dof, sizeof(float));
  auto candidate_parent_array =
      array_1d(candidate_parents, candidate_count, sizeof(int64_t));
  auto valid_array =
      array_1d(candidate_valid, candidate_count, sizeof(bool));
  auto slots = array_1d(assigned_slots, candidate_count, sizeof(int64_t));
  auto written = array_1d(write_mask, candidate_count, sizeof(bool));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds), &nodes, &parents, &count,
      &candidate_array, &candidate_parent_array, &valid_array, &slots,
      &written,
  };
  packed_append_kernel_.launch(
      static_cast<unsigned int>(plans),
      static_cast<unsigned int>(manifest_.packed_append.block_dim),
      static_cast<unsigned int>(manifest_.packed_append.shared_memory), stream,
      arguments);
}

void RrtRuntime::select_connection(bool start_is_active, int plans,
                                   CUstream stream) const {
  const auto bounds = warp_bounds_1d(plans);
  const int candidates = manifest_.samples_per_iteration;
  auto connected =
      array_1d(passive_reached_.ptr(), plans * candidates, sizeof(bool));
  auto active_slots =
      array_1d(active_slots_.ptr(), plans * candidates, sizeof(int64_t));
  auto active_written =
      array_1d(active_written_.ptr(), plans * candidates, sizeof(bool));
  auto passive_slots =
      array_1d(passive_slots_.ptr(), plans * candidates, sizeof(int64_t));
  auto passive_written =
      array_1d(passive_written_.ptr(), plans * candidates, sizeof(bool));
  auto passive_parents =
      array_1d(passive_parents_.ptr(), plans * candidates, sizeof(int64_t));
  auto start_counts = array_1d(start_count_.ptr(), plans, sizeof(int32_t));
  auto goal_counts = array_1d(goal_count_.ptr(), plans, sizeof(int32_t));
  auto plan_active = array_1d(active_plans_.ptr(), plans, sizeof(bool));
  auto connection =
      array_1d(connection_indices_.ptr(), 2 * plans, sizeof(int64_t));
  auto status = array_2d(growth_status_.ptr(), plans, 5, sizeof(int64_t));
  uint8_t start_active = start_is_active ? 1U : 0U;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds),
      &connected,
      &active_slots,
      &active_written,
      &passive_slots,
      &passive_written,
      &passive_parents,
      &start_counts,
      &goal_counts,
      &plan_active,
      &connection,
      &start_active,
      &status,
  };
  select_connection_kernel_.launch(
      grid_for(plans, manifest_.select_connection.block_dim),
      static_cast<unsigned int>(manifest_.select_connection.block_dim),
      static_cast<unsigned int>(manifest_.select_connection.shared_memory),
      stream, arguments);
}

void RrtRuntime::run_direct_edges(int plans, CUstream stream) const {
  const int edges = 2 * plans;
  const int states = edges * manifest_.direct_steer_steps;
  auto starts = array_2d(active_starts_.ptr(), edges, manifest_.num_dof,
                         sizeof(float));
  auto desired =
      array_2d(desired_.ptr(), edges, manifest_.num_dof, sizeof(float));
  auto weights =
      array_1d(distance_weight_.ptr(), manifest_.num_dof, sizeof(float));
  auto point_count = array_1d(direct_point_count_.ptr(), 1, sizeof(int32_t));
  int edge_count = edges;
  float cspace_resolution = manifest_.cspace_resolution;
  const auto prepare_bounds = warp_bounds_1d(1);
  std::vector<void*> prepare_arguments{
      const_cast<WarpLaunchBounds1*>(&prepare_bounds), &starts, &desired,
      &weights, &edge_count, &cspace_resolution, &point_count,
  };
  prepare_direct_edges_kernel_.launch(
      grid_for(1, manifest_.prepare_direct_edges.block_dim),
      static_cast<unsigned int>(manifest_.prepare_direct_edges.block_dim),
      static_cast<unsigned int>(manifest_.prepare_direct_edges.shared_memory),
      stream, prepare_arguments);

  auto lines =
      array_2d(line_points_.ptr(), states, manifest_.num_dof, sizeof(float));
  auto line_valid = array_1d(line_valid_.ptr(), states, sizeof(bool));
  const auto interpolate_bounds =
      warp_bounds_2d(edges, manifest_.direct_steer_steps);
  std::vector<void*> interpolate_arguments{
      const_cast<WarpLaunchBounds2*>(&interpolate_bounds), &starts, &desired,
      &point_count, &lines, &line_valid,
  };
  interpolate_direct_edges_kernel_.launch(
      grid_for(static_cast<size_t>(states),
               manifest_.interpolate_direct_edges.block_dim),
      static_cast<unsigned int>(manifest_.interpolate_direct_edges.block_dim),
      static_cast<unsigned int>(
          manifest_.interpolate_direct_edges.shared_memory),
      stream, interpolate_arguments);

  evaluate_positions(line_points_.ptr(), states, stream);
  classify_states(line_points_.ptr(), states, state_feasible_.ptr(), stream);

  auto feasible = array_1d(state_feasible_.ptr(), states, sizeof(bool));
  auto plan_active = array_1d(active_plans_.ptr(), plans, sizeof(bool));
  auto endpoints = array_2d(active_endpoints_.ptr(), edges,
                            manifest_.num_dof, sizeof(float));
  auto reached = array_1d(active_reached_.ptr(), edges, sizeof(bool));
  auto progressed = array_1d(active_progressed_.ptr(), edges, sizeof(bool));
  auto terminal_feasible =
      array_1d(direct_terminal_feasible_.ptr(), edges, sizeof(bool));
  int plan_count = plans;
  float similarity_threshold = manifest_.similarity_threshold;
  const auto reduce_bounds = warp_bounds_1d(edges);
  std::vector<void*> reduce_arguments{
      const_cast<WarpLaunchBounds1*>(&reduce_bounds),
      &starts,
      &lines,
      &line_valid,
      &feasible,
      &weights,
      &point_count,
      &plan_active,
      &plan_count,
      &similarity_threshold,
      &endpoints,
      &reached,
      &progressed,
      &terminal_feasible,
  };
  reduce_direct_edges_kernel_.launch(
      grid_for(edges, manifest_.reduce_direct_edges.block_dim),
      static_cast<unsigned int>(manifest_.reduce_direct_edges.block_dim),
      static_cast<unsigned int>(manifest_.reduce_direct_edges.shared_memory),
      stream, reduce_arguments);
}

void RrtRuntime::run_direct_stage(int plans, CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const size_t endpoint_bytes = static_cast<size_t>(plans) *
                                manifest_.num_dof * sizeof(float);
  const auto terminal_starts = terminal_positions_.ptr();
  const auto terminal_goals = terminal_starts + endpoint_bytes;
  const auto reverse_starts = active_starts_.ptr() + endpoint_bytes;
  const auto reverse_goals = desired_.ptr() + endpoint_bytes;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(active_starts_.ptr()),
      reinterpret_cast<const void*>(terminal_starts), endpoint_bytes,
      cudaMemcpyDeviceToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(reverse_starts),
      reinterpret_cast<const void*>(terminal_goals), endpoint_bytes,
      cudaMemcpyDeviceToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(desired_.ptr()),
      reinterpret_cast<const void*>(terminal_goals), endpoint_bytes,
      cudaMemcpyDeviceToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(reverse_goals),
      reinterpret_cast<const void*>(terminal_starts), endpoint_bytes,
      cudaMemcpyDeviceToDevice, cuda_stream));
  run_direct_edges(plans, stream);

  const size_t dof_values = static_cast<size_t>(plans) * manifest_.num_dof;
  append(start_nodes_.ptr(), start_parents_.ptr(), start_count_.ptr(),
         active_endpoints_.ptr(), active_parents_.ptr(),
         active_progressed_.ptr(), active_slots_.ptr(), active_written_.ptr(),
         plans, 1, stream);
  append(goal_nodes_.ptr(), goal_parents_.ptr(), goal_count_.ptr(),
         active_endpoints_.ptr() + dof_values * sizeof(float),
         active_parents_.ptr() + static_cast<size_t>(plans) * sizeof(int64_t),
         active_progressed_.ptr() + static_cast<size_t>(plans) * sizeof(bool),
         active_slots_.ptr() + static_cast<size_t>(plans) * sizeof(int64_t),
         active_written_.ptr() + static_cast<size_t>(plans) * sizeof(bool),
         plans, 1, stream);
}

void RrtRuntime::run_growth_stage(bool start_is_active, int plans,
                                  CUstream stream) const {
  const CUdeviceptr active_nodes =
      start_is_active ? start_nodes_.ptr() : goal_nodes_.ptr();
  const CUdeviceptr active_tree_parents =
      start_is_active ? start_parents_.ptr() : goal_parents_.ptr();
  const CUdeviceptr active_count =
      start_is_active ? start_count_.ptr() : goal_count_.ptr();
  const CUdeviceptr passive_nodes =
      start_is_active ? goal_nodes_.ptr() : start_nodes_.ptr();
  const CUdeviceptr passive_tree_parents =
      start_is_active ? goal_parents_.ptr() : start_parents_.ptr();
  const CUdeviceptr passive_count =
      start_is_active ? goal_count_.ptr() : start_count_.ptr();
  const int candidates = manifest_.samples_per_iteration;
  const int edges = plans * candidates;

  nearest(active_nodes, active_count, desired_.ptr(), active_starts_.ptr(),
          active_parents_.ptr(), plans, candidates, stream);
  run_edges(active_starts_.ptr(), desired_.ptr(), edges, candidates, true,
            active_endpoints_.ptr(), active_reached_.ptr(),
            active_progressed_.ptr(), stream);
  append(active_nodes, active_tree_parents, active_count,
         active_endpoints_.ptr(), active_parents_.ptr(),
         active_progressed_.ptr(), active_slots_.ptr(), active_written_.ptr(),
         plans, candidates, stream);

  nearest(passive_nodes, passive_count, active_endpoints_.ptr(),
          passive_starts_.ptr(), passive_parents_.ptr(), plans, candidates,
          stream);
  run_edges(passive_starts_.ptr(), active_endpoints_.ptr(), edges, candidates,
            true, passive_endpoints_.ptr(), passive_reached_.ptr(),
            passive_progressed_.ptr(), stream);
  append(passive_nodes, passive_tree_parents, passive_count,
         passive_endpoints_.ptr(), passive_parents_.ptr(),
         passive_progressed_.ptr(), passive_slots_.ptr(),
         passive_written_.ptr(), plans, candidates, stream);
  select_connection(start_is_active, plans, stream);
}

void RrtRuntime::destroy_search_graph(SearchGraph* graph) noexcept {
  if (graph->executable != nullptr) {
    cudaGraphExecDestroy(graph->executable);
  }
  if (graph->graph != nullptr) {
    cudaGraphDestroy(graph->graph);
  }
  *graph = {};
}

void RrtRuntime::run_shortcut_stage(int plans, CUstream stream) const {
  const int path_capacity = manifest_.shortcut_path_capacity;
  const int raw_path_capacity = manifest_.max_path_nodes;
  const int edge_capacity = manifest_.shortcut_edge_capacity;
  const int state_capacity = manifest_.shortcut_state_capacity;
  auto raw_path = array_2d(shortcut_raw_path_.ptr(), plans * raw_path_capacity,
                           manifest_.num_dof, sizeof(float));
  auto path_count =
      array_1d(shortcut_path_count_.ptr(), plans, sizeof(int32_t));
  auto weights =
      array_1d(distance_weight_.ptr(), manifest_.num_dof, sizeof(float));
  auto pair_sources = array_1d(shortcut_pair_sources_.ptr(), edge_capacity,
                               sizeof(int32_t));
  auto pair_destinations =
      array_1d(shortcut_pair_destinations_.ptr(), edge_capacity,
               sizeof(int32_t));
  const int batched_edges = plans * edge_capacity;
  auto edge_offsets = array_1d(shortcut_edge_offsets_.ptr(), batched_edges,
                               sizeof(int32_t));
  auto edge_steps = array_1d(shortcut_edge_steps_.ptr(), batched_edges,
                             sizeof(int32_t));
  auto edge_tested =
      array_1d(shortcut_edge_tested_.ptr(), batched_edges, sizeof(bool));
  auto status = array_1d(shortcut_status_.ptr(), plans * 5, sizeof(int32_t));
  const auto setup_bounds = warp_bounds_1d(plans);
  std::vector<void*> setup_arguments{
      const_cast<WarpLaunchBounds1*>(&setup_bounds),
      &raw_path,
      &path_count,
      &weights,
      &pair_sources,
      &pair_destinations,
      &edge_offsets,
      &edge_steps,
      &edge_tested,
      &status,
  };
  shortcut_setup_kernel_.launch(
      grid_for(plans, manifest_.shortcut_setup.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_setup.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_setup.shared_memory), stream,
      setup_arguments);

  auto packed_states = array_2d(shortcut_packed_states_.ptr(), state_capacity,
                                manifest_.num_dof, sizeof(float));
  const auto fill_bounds = warp_bounds_1d(batched_edges);
  std::vector<void*> fill_arguments{
      const_cast<WarpLaunchBounds1*>(&fill_bounds),
      &raw_path,
      &pair_sources,
      &pair_destinations,
      &edge_offsets,
      &edge_steps,
      &edge_tested,
      &packed_states,
  };
  shortcut_fill_edges_kernel_.launch(
      grid_for(batched_edges, manifest_.shortcut_fill_edges.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_fill_edges.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_fill_edges.shared_memory),
      stream, fill_arguments);

  evaluate_positions(shortcut_packed_states_.ptr(), state_capacity, stream);
  classify_states(shortcut_packed_states_.ptr(), state_capacity,
                  state_feasible_.ptr(), stream);

  auto feasible = array_1d(state_feasible_.ptr(), state_capacity, sizeof(bool));
  auto edge_visible =
      array_1d(shortcut_edge_visible_.ptr(), batched_edges, sizeof(bool));
  const auto reduce_bounds = warp_bounds_1d(batched_edges);
  std::vector<void*> reduce_arguments{
      const_cast<WarpLaunchBounds1*>(&reduce_bounds), &feasible, &edge_offsets,
      &edge_steps, &edge_tested, &edge_visible,
  };
  shortcut_reduce_edges_kernel_.launch(
      grid_for(batched_edges, manifest_.shortcut_reduce_edges.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_reduce_edges.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_reduce_edges.shared_memory),
      stream, reduce_arguments);

  auto edge_lookup = array_1d(shortcut_edge_lookup_.ptr(),
                              path_capacity * path_capacity, sizeof(int32_t));
  auto dp_hops =
      array_1d(shortcut_dp_hops_.ptr(), plans * path_capacity, sizeof(int32_t));
  auto dp_cost =
      array_1d(shortcut_dp_cost_.ptr(), plans * path_capacity, sizeof(float));
  auto parents =
      array_1d(shortcut_parents_.ptr(), plans * path_capacity, sizeof(int32_t));
  auto output_path =
      array_2d(shortcut_output_path_.ptr(), plans * raw_path_capacity,
               manifest_.num_dof, sizeof(float));
  auto output_indices =
      array_1d(shortcut_output_indices_.ptr(), plans * raw_path_capacity,
               sizeof(int32_t));
  auto raw_path_length =
      array_1d(reconstruction_path_lengths_.ptr(), plans, sizeof(float));
  auto path_length =
      array_1d(shortcut_path_lengths_.ptr(), plans, sizeof(float));
  auto success = array_1d(shortcut_success_.ptr(), plans, sizeof(bool));
  const auto select_bounds = warp_bounds_1d(plans);
  std::vector<void*> select_arguments{
      const_cast<WarpLaunchBounds1*>(&select_bounds),
      &raw_path,
      &path_count,
      &raw_path_length,
      &weights,
      &edge_lookup,
      &edge_visible,
      &dp_hops,
      &dp_cost,
      &parents,
      &output_path,
      &output_indices,
      &status,
      &path_length,
      &path_count,
      &success,
  };
  shortcut_select_path_kernel_.launch(
      grid_for(plans, manifest_.shortcut_select_path.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_select_path.block_dim),
      static_cast<unsigned int>(manifest_.shortcut_select_path.shared_memory),
      stream, select_arguments);
}

void RrtRuntime::run_finalization_stage(int plans, CUstream stream) const {
  const auto bounds = warp_bounds_1d(plans);
  auto start_nodes =
      array_2d(start_nodes_.ptr(), plans * manifest_.max_nodes_per_tree,
               manifest_.num_dof, sizeof(float));
  auto start_parents =
      array_1d(start_parents_.ptr(), plans * manifest_.max_nodes_per_tree,
               sizeof(int64_t));
  auto goal_nodes =
      array_2d(goal_nodes_.ptr(), plans * manifest_.max_nodes_per_tree,
               manifest_.num_dof, sizeof(float));
  auto goal_parents =
      array_1d(goal_parents_.ptr(), plans * manifest_.max_nodes_per_tree,
               sizeof(int64_t));
  auto connection =
      array_1d(connection_indices_.ptr(), 2 * plans, sizeof(int64_t));
  auto weights =
      array_1d(distance_weight_.ptr(), manifest_.num_dof, sizeof(float));
  auto output_path =
      array_2d(shortcut_raw_path_.ptr(), plans * manifest_.max_path_nodes,
               manifest_.num_dof, sizeof(float));
  auto output_count =
      array_1d(shortcut_path_count_.ptr(), plans, sizeof(int32_t));
  auto path_lengths =
      array_1d(reconstruction_path_lengths_.ptr(), plans, sizeof(float));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds), &start_nodes, &start_parents,
      &goal_nodes, &goal_parents, &connection, &weights, &output_path,
      &output_count, &path_lengths,
  };
  reconstruct_path_kernel_.launch(
      grid_for(plans, manifest_.reconstruct_path.block_dim),
      static_cast<unsigned int>(manifest_.reconstruct_path.block_dim),
      static_cast<unsigned int>(manifest_.reconstruct_path.shared_memory),
      stream, arguments);
  run_shortcut_stage(plans, stream);
}

void RrtRuntime::destroy_finalization_graph(FinalizationGraph* graph) noexcept {
  if (graph->executable != nullptr) {
    cudaGraphExecDestroy(graph->executable);
  }
  if (graph->graph != nullptr) {
    cudaGraphDestroy(graph->graph);
  }
  *graph = {};
}

void RrtRuntime::launch_finalization_graph(int plans, CUstream stream) const {
  auto graph = std::find_if(
      finalization_graphs_.begin(), finalization_graphs_.end(),
      [plans](const FinalizationGraph& value) { return value.plans == plans; });
  if (graph == finalization_graphs_.end()) {
    FinalizationGraph captured;
    captured.plans = plans;
    CRR_CUDA_CHECK(cudaStreamSynchronize(graph_capture_stream_));
    CRR_CUDA_CHECK(cudaStreamBeginCapture(graph_capture_stream_,
                                          cudaStreamCaptureModeThreadLocal));
    bool capturing = true;
    try {
      run_finalization_stage(
          plans, reinterpret_cast<CUstream>(graph_capture_stream_));
      CRR_CUDA_CHECK(
          cudaStreamEndCapture(graph_capture_stream_, &captured.graph));
      capturing = false;
      CRR_CUDA_CHECK(
          cudaGraphInstantiate(&captured.executable, captured.graph, 0));
    } catch (...) {
      if (capturing) {
        cudaGraph_t abandoned_graph = nullptr;
        cudaStreamEndCapture(graph_capture_stream_, &abandoned_graph);
        if (abandoned_graph != nullptr) {
          cudaGraphDestroy(abandoned_graph);
        }
      }
      destroy_finalization_graph(&captured);
      throw;
    }
    finalization_graphs_.push_back(captured);
    graph = finalization_graphs_.end() - 1;
  }
  CRR_CUDA_CHECK(cudaGraphLaunch(
      graph->executable, reinterpret_cast<cudaStream_t>(stream)));
}

void RrtRuntime::launch_search_graph(SearchGraphKind kind, int plans,
                                     CUstream stream) const {
  auto graph = std::find_if(
      search_graphs_.begin(), search_graphs_.end(),
      [kind, plans](const SearchGraph& value) {
        return value.kind == kind && value.plans == plans;
      });
  if (graph == search_graphs_.end()) {
    SearchGraph captured;
    captured.kind = kind;
    captured.plans = plans;
    CRR_CUDA_CHECK(cudaStreamSynchronize(graph_capture_stream_));
    CRR_CUDA_CHECK(cudaStreamBeginCapture(graph_capture_stream_,
                                          cudaStreamCaptureModeThreadLocal));
    bool capturing = true;
    try {
      const auto capture_stream =
          reinterpret_cast<CUstream>(graph_capture_stream_);
      if (kind == SearchGraphKind::DIRECT) {
        run_direct_stage(plans, capture_stream);
      } else {
        run_growth_stage(kind == SearchGraphKind::START_GROWTH, plans,
                         capture_stream);
      }
      CRR_CUDA_CHECK(
          cudaStreamEndCapture(graph_capture_stream_, &captured.graph));
      capturing = false;
      CRR_CUDA_CHECK(
          cudaGraphInstantiate(&captured.executable, captured.graph, 0));
    } catch (...) {
      if (capturing) {
        cudaGraph_t abandoned_graph = nullptr;
        cudaStreamEndCapture(graph_capture_stream_, &abandoned_graph);
        if (abandoned_graph != nullptr) {
          cudaGraphDestroy(abandoned_graph);
        }
      }
      destroy_search_graph(&captured);
      throw;
    }
    search_graphs_.push_back(captured);
    graph = search_graphs_.end() - 1;
  }
  CRR_CUDA_CHECK(cudaGraphLaunch(
      graph->executable, reinterpret_cast<cudaStream_t>(stream)));
}

void RrtRuntime::finish_direct(
    const RrtRequest& request, const std::vector<uint8_t>& direct_success,
    CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const size_t dof_bytes =
      static_cast<size_t>(manifest_.num_dof) * sizeof(float);
  const int plans = request.num_plans;
  std::vector<int32_t> path_counts(static_cast<size_t>(plans), 0);
  for (int plan = 0; plan < plans; ++plan) {
    if (direct_success.at(static_cast<size_t>(plan)) == 0U) {
      continue;
    }
    const size_t endpoint_offset = static_cast<size_t>(plan) * dof_bytes;
    const size_t path_offset =
        static_cast<size_t>(plan) * manifest_.max_path_nodes * dof_bytes;
    auto* output = reinterpret_cast<uint8_t*>(request.path) + path_offset;
    const auto* start = reinterpret_cast<const uint8_t*>(request.start_position) +
                        endpoint_offset;
    const auto* goal = reinterpret_cast<const uint8_t*>(request.goal_position) +
                       endpoint_offset;
    CRR_CUDA_CHECK(cudaMemcpyAsync(output, start, dof_bytes,
                                   cudaMemcpyDeviceToDevice, cuda_stream));
    CRR_CUDA_CHECK(cudaMemcpyAsync(output + dof_bytes, goal, dof_bytes,
                                   cudaMemcpyDeviceToDevice, cuda_stream));
    path_counts.at(static_cast<size_t>(plan)) = 2;
  }
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.success, direct_success.data(), direct_success.size(),
      cudaMemcpyHostToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.path_count, path_counts.data(),
      path_counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
      cuda_stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

bool RrtRuntime::finalize_paths(const RrtRequest& request, int plans,
                                int iterations, CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  launch_finalization_graph(plans, stream);
  std::vector<uint8_t> success(static_cast<size_t>(plans), 0U);
  const size_t path_values =
      static_cast<size_t>(plans) * manifest_.max_path_nodes * manifest_.num_dof;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.path, reinterpret_cast<const void*>(shortcut_output_path_.ptr()),
      path_values * sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.path_count,
      reinterpret_cast<const void*>(shortcut_path_count_.ptr()),
      static_cast<size_t>(plans) * sizeof(int32_t), cudaMemcpyDeviceToDevice,
      cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.success, reinterpret_cast<const void*>(shortcut_success_.ptr()),
      static_cast<size_t>(plans) * sizeof(uint8_t), cudaMemcpyDeviceToDevice,
      cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      success.data(), reinterpret_cast<const void*>(shortcut_success_.ptr()),
      success.size(), cudaMemcpyDeviceToHost, cuda_stream));
  const std::vector<int32_t> iteration_counts(static_cast<size_t>(plans),
                                               iterations);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.iterations, iteration_counts.data(),
      iteration_counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
      cuda_stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  return std::any_of(success.begin(), success.end(),
                     [](uint8_t value) { return value != 0U; });
}

void RrtRuntime::finish_failure(const RrtRequest& request, int plans,
                                int iterations, CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CRR_CUDA_CHECK(cudaMemsetAsync(
      request.success, 0, static_cast<size_t>(plans) * sizeof(uint8_t),
      cuda_stream));
  CRR_CUDA_CHECK(cudaMemsetAsync(
      request.path_count, 0, static_cast<size_t>(plans) * sizeof(int32_t),
      cuda_stream));
  const std::vector<int32_t> iteration_counts(static_cast<size_t>(plans),
                                               iterations);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.iterations, iteration_counts.data(),
      iteration_counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
      cuda_stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void RrtRuntime::plan(const RrtRequest& request) const {
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  if (request.start_position == nullptr || request.goal_position == nullptr ||
      request.path == nullptr || request.path_count == nullptr ||
      request.success == nullptr || request.iterations == nullptr) {
    throw std::invalid_argument("RRT input and output CUDA addresses must be non-null");
  }
  if (request.num_plans <= 0 || request.num_plans > config_.max_plans) {
    throw std::invalid_argument("RRT num_plans exceeds the configured workspace");
  }
  const int plans = request.num_plans;
  const auto cuda_stream = request.stream;
  const auto stream = reinterpret_cast<CUstream>(cuda_stream);
  refresh_scene_graphs();
  initialize(request, plans, stream);

  std::vector<uint8_t> requested(static_cast<size_t>(plans), 1U);
  if (request.plan_valid != nullptr) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        requested.data(), request.plan_valid, requested.size(),
        cudaMemcpyDeviceToHost, cuda_stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    for (auto& value : requested) {
      value = value == 0U ? 0U : 1U;
    }
  }
  if (std::none_of(requested.begin(), requested.end(),
                   [](uint8_t value) { return value != 0U; })) {
    finish_failure(request, plans, 0, stream);
    return;
  }

  std::vector<uint8_t> active = requested;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(active_plans_.ptr()), active.data(), active.size(),
      cudaMemcpyHostToDevice, cuda_stream));

  active_reached_.clear(cuda_stream);
  active_progressed_.clear(cuda_stream);
  passive_reached_.clear(cuda_stream);
  passive_progressed_.clear(cuda_stream);
  active_parents_.clear(cuda_stream);
  passive_parents_.clear(cuda_stream);
  launch_search_graph(SearchGraphKind::DIRECT, plans, stream);
  std::vector<uint8_t> terminal_feasible(static_cast<size_t>(2 * plans), 0U);
  std::vector<uint8_t> direct_reached(static_cast<size_t>(plans), 0U);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      terminal_feasible.data(),
      reinterpret_cast<void*>(direct_terminal_feasible_.ptr()),
      terminal_feasible.size(), cudaMemcpyDeviceToHost, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      direct_reached.data(), reinterpret_cast<void*>(active_reached_.ptr()),
      direct_reached.size(), cudaMemcpyDeviceToHost, cuda_stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  for (int plan = 0; plan < plans; ++plan) {
    const auto index = static_cast<size_t>(plan);
    active.at(index) =
        requested.at(index) != 0U && terminal_feasible.at(index) != 0U &&
                terminal_feasible.at(static_cast<size_t>(plans + plan)) != 0U
            ? 1U
            : 0U;
    direct_reached.at(index) =
        active.at(index) != 0U && direct_reached.at(index) != 0U ? 1U : 0U;
  }
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(active_plans_.ptr()), active.data(), active.size(),
      cudaMemcpyHostToDevice, cuda_stream));
  if (std::none_of(active.begin(), active.end(),
                   [](uint8_t value) { return value != 0U; })) {
    finish_failure(request, plans, 0, stream);
    return;
  }
  if (std::any_of(direct_reached.begin(), direct_reached.end(),
                  [](uint8_t value) { return value != 0U; })) {
    finish_direct(request, direct_reached, stream);
    return;
  }

  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    prepare_samples(next_samples(plans), plans, stream);
    const bool start_is_active = iteration % 2 == 0;
    launch_search_graph(start_is_active ? SearchGraphKind::START_GROWTH
                                        : SearchGraphKind::GOAL_GROWTH,
                        plans, stream);

    std::vector<int64_t> connection(static_cast<size_t>(2 * plans), -1);
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        connection.data(), reinterpret_cast<void*>(connection_indices_.ptr()),
        connection.size() * sizeof(int64_t), cudaMemcpyDeviceToHost,
        cuda_stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    bool connected = false;
    for (int plan = 0; plan < plans; ++plan) {
      const auto offset = static_cast<size_t>(2 * plan);
      connected = connected ||
                  (connection.at(offset) >= 0 &&
                   connection.at(offset + 1) >= 0);
    }
    if (connected) {
      if (finalize_paths(request, plans, iteration + 1, stream)) {
        return;
      }
    }
  }
  finish_failure(request, plans, config_.max_iterations, stream);
}

}  // namespace curobo::robot::detail
