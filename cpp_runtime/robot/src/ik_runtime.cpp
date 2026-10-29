// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "ik_runtime.h"

#include "attachment_runtime.h"
#include "cuda_check.h"
#include "curobo_robot_runtime/scene.h"
#include "rollout_runtime.h"
#include "rollout_streams.h"
#include "scene_runtime.h"
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
  if (count == 0) {
    throw std::invalid_argument("kernel launch size must be positive");
  }
  const auto blocks = (count + static_cast<size_t>(block_size) - 1) /
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
  kernel->load((std::filesystem::path(artifact_directory) / manifest.cubin).string(),
               manifest.symbol);
}

WarpArray array_1d(CUdeviceptr pointer, int count, int element_bytes) {
  return warp_array_1d(static_cast<uint64_t>(pointer), count, element_bytes);
}

WarpArray array_2d(CUdeviceptr pointer, int rows, int columns, int element_bytes) {
  return warp_array_2d(static_cast<uint64_t>(pointer), rows, columns, element_bytes);
}

WarpArray array_3d(CUdeviceptr pointer, int first, int second, int third,
                   int element_bytes) {
  return warp_array_3d(static_cast<uint64_t>(pointer), first, second, third,
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
      array_1d(view.features, view.capacity * view.max_voxels_per_grid,
               sizeof(uint16_t)),
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
    CUdeviceptr scene_indices, CUdeviceptr collision_distance,
    CUdeviceptr collision_gradient, int problems, int num_spheres,
    bool overwrite, CUstream stream) {
  const auto bounds = warp_tiled_bounds_2d(problems, manifest.block_dim);
  auto spheres = array_1d(robot_spheres, problems * num_spheres, 4 * sizeof(float));
  auto weight_array = array_1d(weight, 1, sizeof(float));
  auto activation_array = array_1d(activation_distance, 1, sizeof(float));
  auto scene_index_array = array_1d(scene_indices, problems, sizeof(int32_t));
  auto distance_array = array_1d(collision_distance, problems, sizeof(float));
  auto gradient_array =
      array_1d(collision_gradient, problems * num_spheres * 4, sizeof(float));
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
      &distance_array,
      &gradient_array,
      &use_multiple_scenes,
      &overwrite_output,
      &sum_distance,
      &obstacle_layer_array,
      &sphere_mask_array,
  };
  kernel.launch(static_cast<unsigned int>(problems),
                static_cast<unsigned int>(manifest.block_dim),
                static_cast<unsigned int>(manifest.shared_memory), stream, arguments);
}

}  // namespace

bool has_valid_solution_for_each_batch(const std::vector<uint8_t>& success,
                                       int batch_size, int num_solutions) {
  if (batch_size <= 0 || num_solutions <= 0) {
    throw std::invalid_argument(
        "IK batch size and solution count must be positive");
  }
  const auto expected_count = checked_product(
      static_cast<size_t>(batch_size), static_cast<size_t>(num_solutions),
      "IK success output");
  if (success.size() != expected_count) {
    throw std::invalid_argument(
        "IK success output does not match its batch dimensions");
  }
  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    const auto begin = success.cbegin() +
                       static_cast<size_t>(batch_index) *
                           static_cast<size_t>(num_solutions);
    const auto end = begin + num_solutions;
    if (std::none_of(begin, end,
                     [](uint8_t value) { return value != 0U; })) {
      return false;
    }
  }
  return true;
}

// Geometry lives before FK: every solver reads the same link-local spheres,
// but writes its transformed spheres/Jacobians into its own workspace.
struct IkModelParameters {
  IkModelParameters(const std::string& artifact_directory, int device_ordinal);

  int device_ordinal_;
  curobo_robot_runtime::RobotRuntimeImpl robot_;
  curobo_robot_runtime::IkManifest manifest_;
  std::unique_ptr<RolloutRuntime> rollout_;
  curobo_robot_runtime::DeviceBuffer joint_position_lower_;
  curobo_robot_runtime::DeviceBuffer joint_position_upper_;
  curobo_robot_runtime::DeviceBuffer default_joint_position_;
  curobo_robot_runtime::DeviceBuffer seed_positions_;
  std::shared_ptr<RobotCollisionModel> collision_;
  curobo_robot_runtime::DeviceBuffer self_collision_padding_;
  curobo_robot_runtime::DeviceBuffer self_collision_pairs_;
  curobo_robot_runtime::DeviceBuffer position_bounds_;
  curobo_robot_runtime::DeviceBuffer velocity_bounds_;
};

IkModelParameters::IkModelParameters(const std::string& artifact_directory,
                                     int device_ordinal)
    : device_ordinal_(device_ordinal),
      robot_(artifact_directory, device_ordinal),
      manifest_(curobo_robot_runtime::load_ik_manifest(artifact_directory)) {
  const auto rollout_manifest =
      std::filesystem::path(artifact_directory) / "robot_rollout_manifest.json";
  if (std::filesystem::exists(rollout_manifest)) {
    rollout_ = std::make_unique<RolloutRuntime>(
        artifact_directory, device_ordinal, robot_.manifest(), RolloutProfile::IK);
  }
  const auto load = [&](const curobo_robot_runtime::TensorManifest& tensor,
                        const char* name,
                        curobo_robot_runtime::DeviceBuffer* destination) {
    destination->upload(curobo_robot_runtime::load_tensor_data(
        artifact_directory, tensor, name));
  };
  load(manifest_.joint_position_lower, "joint_position_lower", &joint_position_lower_);
  load(manifest_.joint_position_upper, "joint_position_upper", &joint_position_upper_);
  load(manifest_.default_joint_position, "default_joint_position", &default_joint_position_);
  load(manifest_.seed_positions, "seed_positions", &seed_positions_);
  const auto link_spheres = curobo_robot_runtime::load_tensor_data(
      artifact_directory, manifest_.link_spheres, "link_spheres");
  const auto link_sphere_map = curobo_robot_runtime::load_tensor_data(
      artifact_directory, manifest_.link_sphere_map, "link_sphere_map");
  collision_ = std::make_shared<RobotCollisionModel>(
      device_ordinal, robot_.manifest().link_names, link_spheres,
      link_sphere_map,
      curobo_robot_runtime::load_tensor_data(artifact_directory,
                                             manifest_.sphere_collision_mask,
                                             "sphere_collision_mask"));
  if (manifest_.num_self_collision_pairs > 0) {
    load(manifest_.self_collision_padding, "self_collision_padding", &self_collision_padding_);
    load(manifest_.self_collision_pairs, "self_collision_pairs", &self_collision_pairs_);
  }
  load(manifest_.position_bounds, "ik_position_bounds", &position_bounds_);
  load(manifest_.velocity_bounds, "ik_velocity_bounds", &velocity_bounds_);
}

IkRuntime::IkRuntime(std::string artifact_directory,
                     const InverseKinematicsConfig& config, const Scene* scene)
    : IkRuntime(artifact_directory, config,
                scene == nullptr ? nullptr : scene->runtime_.get(),
                std::make_shared<IkModelParameters>(artifact_directory, config.device_ordinal)) {}

IkRuntime::IkRuntime(const IkRuntime& shared_robot, const InverseKinematicsConfig& config)
    : IkRuntime(shared_robot.artifact_directory_, config, shared_robot.scene_, shared_robot.model_) {}

IkRuntime::IkRuntime(std::string artifact_directory,
                     const InverseKinematicsConfig& config, const SceneRuntime* scene,
                     std::shared_ptr<IkModelParameters> model)
    : artifact_directory_(std::move(artifact_directory)), config_(config), scene_(scene),
      model_(std::move(model)), manifest_(model_->manifest_),
      prepare_search_actions_spec_(config.line_search_backtracking
          ? manifest_.prepare_search_actions_backtracking : manifest_.prepare_search_actions),
      line_search_spec_(config.line_search_backtracking
          ? manifest_.line_search_backtracking : manifest_.line_search) {
  if (config_.device_ordinal != model_->device_ordinal_) {
    throw std::invalid_argument("Shared IK model and solver must use the same CUDA device");
  }
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  if (model_->rollout_ == nullptr && config_.check_effort_limits) {
    throw std::runtime_error(
        "IK effort limits require a robot rollout artifact");
  }
  if (config_.lbfgs_iterations == 0) {
    config_.lbfgs_iterations = manifest_.optimizer_iterations;
  }
  if (config_.lm_iterations == 0) {
    config_.lm_iterations = 16;
  }
  validate_config();
  validate_scene();
  load_constants();
  load_kernels();
  allocate_workspace();
  rollout_streams_ = std::make_unique<RolloutStreams>();
  if (config_.use_cuda_graph) {
    CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&graph_capture_stream_,
                                              cudaStreamNonBlocking));
  }
}

IkRuntime::~IkRuntime() {
  cudaSetDevice(config_.device_ordinal);
  for (const auto* cache : {&lm_graph_, &lbfgs_graph_}) {
    if (cache->initialized) {
      cudaStreamSynchronize(cache->last_stream);
    }
  }
  destroy_graph(&lm_graph_);
  destroy_graph(&lbfgs_graph_);
  if (graph_capture_stream_ != nullptr) {
    cudaStreamDestroy(graph_capture_stream_);
  }
}

CuroboRobotInfo IkRuntime::info() const { return model_->robot_.info(); }

int IkRuntime::attachment_sphere_capacity(
    const std::string& link_name) const {
  return model_->collision_->attachment->capacity(link_name);
}

void IkRuntime::attach_object(const SphereAttachment& attachment) {
  model_->collision_->attachment->attach(attachment);
}

void IkRuntime::detach_object(const std::string& link_name) {
  model_->collision_->attachment->detach(link_name);
}

void IkRuntime::update_link_inertial(const LinkInertialUpdate& update) {
  if (model_->rollout_ == nullptr) {
    throw std::runtime_error("IK artifact has no inverse-dynamics model");
  }
  model_->rollout_->update_link_inertial(update);
}

void IkRuntime::update_links_inertial(const LinkInertialUpdates& updates) {
  if (model_->rollout_ == nullptr) {
    throw std::runtime_error("IK artifact has no inverse-dynamics model");
  }
  model_->rollout_->update_links_inertial(updates);
}

std::shared_ptr<RobotCollisionModel> IkRuntime::collision_model() const {
  return model_->collision_;
}

void IkRuntime::set_link_collision_mask(const std::string& link,
                                        uint32_t layers) {
  model_->collision_->attachment->set_collision_mask(
      link, layers, model_->collision_->sphere_collision_mask);
}

void IkRuntime::set_link_collision_enabled(const std::string& link_name,
                                           bool enabled) {
  model_->collision_->attachment->set_collision_enabled(link_name, enabled);
}

void IkRuntime::validate_config() const {
  const auto robot_info = model_->robot_.info();
  if (manifest_.arch != model_->robot_.manifest().arch ||
      manifest_.num_links != robot_info.num_links ||
      manifest_.num_dof != robot_info.num_dof ||
      manifest_.num_tool_frames != robot_info.num_tool_frames) {
    throw std::runtime_error("robot IK artifact does not match its kinematics artifact");
  }
  if (config_.max_batch_size <= 0 || config_.max_goalset <= 0 ||
      config_.num_seeds <= 0 || config_.lm_iterations <= 0 ||
      config_.lbfgs_iterations <= 0) {
    throw std::invalid_argument(
        "IK workspace dimensions and optimizer iteration counts must be positive");
  }
  if (config_.num_seeds > manifest_.max_seeds) {
    throw std::invalid_argument("IK num_seeds exceeds the artifact seed capacity");
  }
  if (config_.max_goalset > manifest_.max_goalset) {
    throw std::invalid_argument(
        "IK max_goalset exceeds the artifact goalset capacity");
  }
  const auto model = model_->robot_.model_device_view();
  if (model.joint_link_influence_words == 0 || model.joint_links_data == 0 ||
      model.joint_links_offsets == 0 || model.link_masses_com == 0) {
    throw std::runtime_error(
        "L-BFGS IK requires reverse-kinematics topology tensors");
  }
  for (const auto value : {
           config_.position_tolerance,
           config_.orientation_tolerance,
           config_.position_weight,
           config_.orientation_weight,
           config_.joint_limit_weight,
           config_.position_rank_weight,
           config_.orientation_rank_weight,
           config_.cspace_rank_weight,
           config_.initial_damping,
           config_.damping_factor,
           config_.minimum_damping,
           config_.maximum_damping,
           config_.minimum_trust_ratio,
           config_.optimizer_collision_activation_distance,
           config_.collision_activation_distance,
       }) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("IK scalar configuration must be finite");
    }
  }
  if (config_.position_tolerance <= 0.0F ||
      config_.orientation_tolerance <= 0.0F ||
      config_.position_weight <= 0.0F ||
      config_.orientation_weight <= 0.0F ||
      config_.joint_limit_weight < 0.0F ||
      config_.position_rank_weight < 0.0F ||
      config_.orientation_rank_weight < 0.0F ||
      config_.cspace_rank_weight < 0.0F || config_.initial_damping <= 0.0F ||
      config_.damping_factor <= 1.0F || config_.minimum_damping <= 0.0F ||
      config_.maximum_damping < config_.minimum_damping ||
      (config_.optimizer_collision_activation_distance < 0.0F &&
       config_.optimizer_collision_activation_distance != -1.0F) ||
      config_.collision_activation_distance < 0.0F) {
    throw std::invalid_argument("IK scalar configuration is outside its valid range");
  }
}

void IkRuntime::validate_scene() const {
  if (scene_ == nullptr) {
    return;
  }
  const auto& scene_config = scene_->config();
  if (scene_config.device_ordinal != config_.device_ordinal) {
    throw std::invalid_argument("IK and Scene must use the same CUDA device");
  }
  if (scene_config.max_cuboids > manifest_.max_cuboids ||
      scene_config.max_capsules > manifest_.max_capsules ||
      scene_config.max_cylinders > manifest_.max_cylinders ||
      scene_config.max_meshes > manifest_.max_meshes ||
      scene_config.max_voxel_grids > manifest_.max_voxel_grids) {
    throw std::invalid_argument("Scene capacity exceeds the IK artifact capacity");
  }
}

void IkRuntime::load_constants() {
  const auto load = [&](const curobo_robot_runtime::TensorManifest& tensor,
                        const char* name,
                        curobo_robot_runtime::DeviceBuffer* destination) {
    destination->upload(curobo_robot_runtime::load_tensor_data(
        artifact_directory_, tensor, name));
  };
  seed_offset_.allocate(sizeof(int32_t));
  std::vector<float> inactive_effort_bounds(
      static_cast<size_t>(2 * manifest_.num_dof));
  std::fill_n(inactive_effort_bounds.begin(), manifest_.num_dof,
              -std::numeric_limits<float>::infinity());
  std::fill_n(inactive_effort_bounds.begin() + manifest_.num_dof,
              manifest_.num_dof, std::numeric_limits<float>::infinity());
  inactive_effort_bounds_.upload(as_bytes(inactive_effort_bounds));
  load(manifest_.cspace_weight, "ik_cspace_weight", &cspace_weight_);
  load(manifest_.cspace_activation_distance,
       "ik_cspace_activation_distance", &cspace_activation_distance_);
  load(manifest_.cspace_target_weight, "ik_cspace_target_weight",
       &cspace_target_weight_);
  load(manifest_.cspace_target_dof_weight,
       "ik_cspace_target_dof_weight", &cspace_target_dof_weight_);
  load(manifest_.cspace_regularization_weight,
       "ik_cspace_regularization_weight", &cspace_regularization_weight_);
  load(manifest_.start_cspace_weight, "ik_start_cspace_weight",
       &start_cspace_weight_);
  load(manifest_.start_cspace_dof_weight, "ik_start_cspace_dof_weight",
       &start_cspace_dof_weight_);
  load(manifest_.world_collision_weight, "ik_world_collision_weight",
       &optimizer_world_collision_weight_);
  load(manifest_.self_collision_weight, "ik_self_collision_weight",
       &optimizer_self_collision_weight_);
  load(manifest_.optimizer_collision_activation_distance,
       "ik_optimizer_collision_activation_distance",
       &optimizer_collision_activation_distance_);
  if (config_.optimizer_collision_activation_distance >= 0.0F) {
    optimizer_collision_activation_distance_.upload(as_bytes(std::vector<float>{
        config_.optimizer_collision_activation_distance}));
  }
  load(manifest_.line_search_scale, "ik_line_search_scale",
       &line_search_scale_);
  load(manifest_.optimization_step_max, "ik_optimization_step_max",
       &optimization_step_max_);
  collision_weight_.upload(as_bytes(std::vector<float>{1.0F}));
  collision_activation_distance_.upload(
      as_bytes(std::vector<float>{config_.collision_activation_distance}));
  zero_state_.upload(as_bytes(std::vector<float>(manifest_.num_dof, 0.0F)));
  const auto max_evaluations =
      checked_product(
          checked_product(static_cast<size_t>(config_.max_batch_size),
                          static_cast<size_t>(config_.num_seeds), "IK problem"),
          static_cast<size_t>(manifest_.num_line_search), "IK evaluation");
  zero_indices_.upload(
      as_bytes(std::vector<int32_t>(max_evaluations, 0)));
  scene_indices_.upload(
      as_bytes(std::vector<int32_t>(max_evaluations, 0)));
}

void IkRuntime::load_kernels() {
  kinematics_kernel_.load(
      (std::filesystem::path(artifact_directory_) / manifest_.kinematics.cubin)
          .string(),
      manifest_.kinematics.symbol);
  load_kernel(artifact_directory_, manifest_.kinematics_backward,
              &kinematics_backward_kernel_);
  load_kernel(artifact_directory_, manifest_.initialize, &initialize_kernel_);
  load_kernel(artifact_directory_, manifest_.error, &error_kernel_);
  load_kernel(artifact_directory_, manifest_.update, &update_kernel_);
  load_kernel(artifact_directory_, manifest_.finalize, &finalize_kernel_);
  load_kernel(artifact_directory_, manifest_.rank, &rank_kernel_);
  load_kernel(artifact_directory_, manifest_.levenberg_marquardt, &lm_kernel_);
  load_kernel(artifact_directory_, manifest_.cspace, &cspace_kernel_);
  load_kernel(artifact_directory_, manifest_.prepare_initial_state,
              &prepare_initial_state_kernel_);
  load_kernel(artifact_directory_, prepare_search_actions_spec_,
              &prepare_search_actions_kernel_);
  load_kernel(artifact_directory_, manifest_.combine_lbfgs_rollout,
              &combine_lbfgs_rollout_kernel_);
  load_kernel(artifact_directory_, line_search_spec_,
              &line_search_kernel_);
  load_kernel(artifact_directory_, manifest_.lbfgs, &lbfgs_kernel_);
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
  if (manifest_.num_self_collision_pairs > 0 &&
      manifest_.self_collision.shared_memory > 48000) {
    self_collision_kernel_.set_max_dynamic_shared_memory(
        manifest_.self_collision.shared_memory);
  }
  if (manifest_.lbfgs.shared_memory > 48000) {
    lbfgs_kernel_.set_max_dynamic_shared_memory(manifest_.lbfgs.shared_memory);
  }
}

void IkRuntime::allocate_workspace() {
  const auto max_problems =
      checked_product(static_cast<size_t>(config_.max_batch_size),
                      static_cast<size_t>(config_.num_seeds), "IK problem");
  const auto max_evaluations =
      checked_product(max_problems, static_cast<size_t>(manifest_.num_line_search),
                      "IK evaluation");
  const auto dof = static_cast<size_t>(manifest_.num_dof);
  const auto links = static_cast<size_t>(manifest_.num_tool_frames);
  const auto spheres = static_cast<size_t>(manifest_.num_spheres);
  const auto residuals = static_cast<size_t>(manifest_.num_residuals);
  const auto allocate_float = [](curobo_robot_runtime::DeviceBuffer* buffer,
                                 size_t count) {
    buffer->allocate(checked_product(count, sizeof(float), "IK float buffer"));
  };

  allocate_float(&joint_position_, max_problems * dof);
  allocate_float(&candidate_position_, max_problems * dof);
  allocate_float(&link_position_, max_evaluations * links * 3);
  allocate_float(&link_quaternion_, max_evaluations * links * 4);
  allocate_float(&link_jacobian_, max_evaluations * links * 6 * dof);
  allocate_float(&global_cumulative_, max_evaluations * manifest_.num_links * 12);
  allocate_float(&robot_spheres_, max_evaluations * spheres * 4);
  allocate_float(&zero_link_position_gradient_, max_evaluations * links * 3);
  allocate_float(&zero_link_quaternion_gradient_, max_evaluations * links * 4);
  allocate_float(&scaled_jacobian_, max_evaluations * residuals * dof);
  allocate_float(&jt_error_, max_evaluations * dof);
  for (auto* buffer : {
           &current_error_,
           &candidate_error_,
           &current_position_error_,
           &candidate_position_error_,
           &current_orientation_error_,
           &rank_score_,
           &candidate_orientation_error_,
           &damping_,
           &predicted_reduction_,
           &collision_distance_,
           &self_collision_distance_,
       }) {
    allocate_float(buffer, max_evaluations);
  }
  allocate_float(&collision_gradient_, max_evaluations * spheres * 4);
  allocate_float(&self_collision_gradient_, max_evaluations * spheres * 4);
  allocate_float(&self_collision_pair_distance_, 1);
  self_collision_sparse_index_.allocate(
      checked_product(max_evaluations, spheres, "IK self-collision sparse index"));
  allocate_float(&collision_joint_gradient_, max_evaluations * dof);
  allocate_float(&cspace_cost_, max_evaluations * dof);
  allocate_float(&cspace_gradient_, max_evaluations * dof);
  allocate_float(&zero_effort_, max_evaluations * dof);
  allocate_float(&zero_effort_gradient_, max_evaluations * dof);
  allocate_float(&zero_derivatives_, max_evaluations * dof);
  allocate_float(&effort_position_gradient_, max_evaluations * dof);
  allocate_float(&effort_velocity_gradient_, max_evaluations * dof);
  allocate_float(&effort_acceleration_gradient_, max_evaluations * dof);
  if (config_.check_effort_limits) {
    allocate_float(
        &rnea_forward_cache_,
        max_evaluations * static_cast<size_t>(model_->rollout_->num_links()) *
            static_cast<size_t>(model_->rollout_->cache_floats_per_link()));
  }
  rank_success_.allocate(config_.max_batch_size * config_.num_seeds * sizeof(uint8_t));
  goalset_index_.allocate(
      checked_product(max_evaluations, links, "IK goalset index") *
      sizeof(int32_t));
  output_goalset_index_.allocate(
      checked_product(max_problems, links,
                      "IK output goalset index") *
      sizeof(int32_t));

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
    allocate_float(buffer, max_problems * dof);
  }
  for (auto* buffer : {&cost_, &exploration_cost_, &best_cost_}) {
    allocate_float(buffer, max_problems);
  }
  best_iteration_.allocate(max_problems * sizeof(int32_t));
  current_iteration_.allocate(max_problems * sizeof(int32_t));
  converged_.allocate(max_problems * sizeof(uint8_t));
  history_head_.allocate(max_problems * sizeof(int32_t));
  history_count_.allocate(max_problems * sizeof(int32_t));
  curvature_update_valid_.allocate(max_problems * sizeof(bool));
  allocate_float(&history_rho_, max_problems * manifest_.history);
  allocate_float(&history_y_, max_problems * manifest_.history * dof);
  allocate_float(&history_s_, max_problems * manifest_.history * dof);
  allocate_float(&search_actions_, max_evaluations * dof);
  exploration_indices_.allocate(
      max_problems * manifest_.num_line_search * sizeof(int32_t));
  selected_indices_.allocate(
      max_problems * manifest_.num_line_search * sizeof(int32_t));
  line_search_index_.allocate(max_problems * sizeof(int32_t));
  line_search_status_.allocate(max_problems * sizeof(int32_t));
  line_search_beta_.allocate(max_problems * sizeof(float));

  zero_link_position_gradient_.clear();
  zero_link_quaternion_gradient_.clear();
  zero_effort_.clear();
  zero_effort_gradient_.clear();
  zero_derivatives_.clear();
  effort_position_gradient_.clear();
  collision_gradient_.clear();
  self_collision_gradient_.clear();
  self_collision_sparse_index_.clear();
}

void IkRuntime::launch_kinematics(CUdeviceptr joint_position, int problems,
                                  CUstream stream) const {
  const auto& policy = manifest_.kinematics;
  const auto& robot_manifest = model_->robot_.manifest();
  auto model = model_->robot_.model_device_view();
  const int shared_memory_per_batch = robot_manifest.num_links * 12 * sizeof(float) * 2;
  const int batches_per_block = std::min(
      {problems, policy.max_batches_per_block,
       policy.max_threads_per_block / policy.threads_per_batch,
       policy.max_dynamic_shared_memory / shared_memory_per_batch});
  if (batches_per_block <= 0) {
    throw std::runtime_error("IK kinematics launch policy produced an empty block");
  }
  const auto block =
      static_cast<unsigned int>(batches_per_block * policy.threads_per_batch);
  const auto grid = static_cast<unsigned int>(
      (problems + batches_per_block - 1) / batches_per_block);
  const auto shared =
      static_cast<unsigned int>(batches_per_block * shared_memory_per_batch);

  CUdeviceptr link_position = link_position_.ptr();
  CUdeviceptr link_quaternion = link_quaternion_.ptr();
  CUdeviceptr spheres = robot_spheres_.ptr();
  CUdeviceptr null_pointer = 0;
  CUdeviceptr jacobian = link_jacobian_.ptr();
  CUdeviceptr global_cumulative = global_cumulative_.ptr();
  CUdeviceptr fixed_transforms = model.fixed_transforms;
  CUdeviceptr link_spheres = model_->collision_->link_spheres.ptr();
  CUdeviceptr joint_map_type = model.joint_map_type;
  CUdeviceptr joint_map = model.joint_map;
  CUdeviceptr link_map = model.link_map;
  CUdeviceptr tool_frame_map = model.tool_frame_map;
  CUdeviceptr link_sphere_map = model_->collision_->link_sphere_map.ptr();
  CUdeviceptr tool_joint_links_data = model.tool_joint_links_data;
  CUdeviceptr tool_joint_links_offsets = model.tool_joint_links_offsets;
  CUdeviceptr joint_offset = model.joint_offset_map;
  int batch_size = problems;
  int horizon = 1;
  int num_spheres = manifest_.num_spheres;
  int num_scenes = 1;
  int num_links = robot_manifest.num_links;
  int num_dof = manifest_.num_dof;
  int num_tool_frames = manifest_.num_tool_frames;
  std::vector<void*> arguments{
      &link_position,
      &link_quaternion,
      &spheres,
      &null_pointer,
      &jacobian,
      &global_cumulative,
      &joint_position,
      &fixed_transforms,
      &link_spheres,
      &null_pointer,
      &joint_map_type,
      &joint_map,
      &link_map,
      &tool_frame_map,
      &link_sphere_map,
      &tool_joint_links_data,
      &tool_joint_links_offsets,
      &joint_offset,
      &null_pointer,
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

void IkRuntime::launch_error(CUdeviceptr joint_position, CUdeviceptr error,
                             CUdeviceptr position_error,
                             CUdeviceptr orientation_error,
                             const IkSolveRequest& request, int problems,
                             bool include_joint_limits,
                             CUstream stream) const {
  const auto bounds = warp_tiled_bounds_2d(problems, manifest_.error.block_dim);
  auto current_positions = array_1d(
      link_position_.ptr(), problems * manifest_.num_tool_frames, 3 * sizeof(float));
  auto current_quaternions = array_1d(
      link_quaternion_.ptr(), problems * manifest_.num_tool_frames, 4 * sizeof(float));
  auto goal_positions = warp_array_1d(
      reinterpret_cast<uint64_t>(request.goal_positions),
      request.batch_size * manifest_.num_tool_frames * request.num_goalset,
      3 * sizeof(float));
  auto goal_quaternions = warp_array_1d(
      reinterpret_cast<uint64_t>(request.goal_quaternions),
      request.batch_size * manifest_.num_tool_frames * request.num_goalset,
      4 * sizeof(float));
  auto goalset_index = array_1d(
      goalset_index_.ptr(), problems * manifest_.num_tool_frames,
      sizeof(int32_t));
  auto raw_jacobian = array_3d(
      link_jacobian_.ptr(), problems, manifest_.num_tool_frames * 6,
      manifest_.num_dof, sizeof(float));
  auto joints = array_2d(joint_position, problems, manifest_.num_dof, sizeof(float));
  auto lower = array_1d(model_->joint_position_lower_.ptr(), manifest_.num_dof, sizeof(float));
  auto upper = array_1d(model_->joint_position_upper_.ptr(), manifest_.num_dof, sizeof(float));
  auto scaled_jacobian = array_3d(scaled_jacobian_.ptr(), problems,
                                  manifest_.num_residuals, manifest_.num_dof,
                                  sizeof(float));
  auto gradient = array_2d(jt_error_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto error_array = array_1d(error, problems, sizeof(float));
  auto position_error_array = array_1d(position_error, problems, sizeof(float));
  auto orientation_error_array = array_1d(orientation_error, problems, sizeof(float));
  int num_seeds = problems / request.batch_size;
  int num_goalset = request.num_goalset;
  float position_weight = config_.position_weight;
  float orientation_weight = config_.orientation_weight;
  float joint_limit_weight = config_.joint_limit_weight;
  if (!include_joint_limits) {
    position_weight = manifest_.position_weight;
    orientation_weight = manifest_.orientation_weight;
  }
  uint8_t include_joint_limit_cost = include_joint_limits ? 1U : 0U;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &current_positions,
      &current_quaternions,
      &goal_positions,
      &goal_quaternions,
      &goalset_index,
      &raw_jacobian,
      &joints,
      &lower,
      &upper,
      &scaled_jacobian,
      &gradient,
      &error_array,
      &position_error_array,
      &orientation_error_array,
      &num_seeds,
      &num_goalset,
      &position_weight,
      &orientation_weight,
      &joint_limit_weight,
      &include_joint_limit_cost,
  };
  const int error_block_size = manifest_.error.block_dim;
  error_kernel_.launch(static_cast<unsigned int>(problems),
                       static_cast<unsigned int>(error_block_size),
                       static_cast<unsigned int>(manifest_.error.shared_memory), stream,
                       arguments);
}

void IkRuntime::launch_collision(int problems, bool optimizer_cost,
                                 CUstream stream) const {
  if (scene_ == nullptr || manifest_.num_spheres == 0) {
    collision_distance_.clear(reinterpret_cast<cudaStream_t>(stream));
    collision_gradient_.clear(reinterpret_cast<cudaStream_t>(stream));
    return;
  }
  const auto view = scene_->device_view();
  const auto weight = optimizer_cost ? optimizer_world_collision_weight_.ptr()
                                     : collision_weight_.ptr();
  const auto activation_distance =
      optimizer_cost ? optimizer_collision_activation_distance_.ptr()
                     : collision_activation_distance_.ptr();
  bool overwrite = true;
  const auto launch =
      [&](const auto& obstacle_data, CUdeviceptr layers,
          const curobo_robot_runtime::IkKernelManifest& kernel_manifest,
          const curobo_robot_runtime::KernelModule& kernel) {
        if (kernel_manifest.cubin.empty()) {
          return;
        }
        launch_collision_kernel(kernel, kernel_manifest, obstacle_data, layers,
                                model_->collision_->sphere_collision_mask.ptr(),
                                robot_spheres_.ptr(), weight,
                                activation_distance, scene_indices_.ptr(),
                                collision_distance_.ptr(),
                                collision_gradient_.ptr(), problems,
                                manifest_.num_spheres, overwrite, stream);
        overwrite = false;
      };
  launch(make_cuboid_data(view.cuboids), view.cuboids.obstacle_layers,
         manifest_.collision_cuboid, collision_cuboid_kernel_);
  launch(make_capsule_data(view.capsules), view.capsules.obstacle_layers,
         manifest_.collision_capsule, collision_capsule_kernel_);
  launch(make_capsule_data(view.cylinders), view.cylinders.obstacle_layers,
         manifest_.collision_cylinder, collision_cylinder_kernel_);
  launch(make_mesh_data(view.meshes), view.meshes.obstacle_layers,
         manifest_.collision_mesh, collision_mesh_kernel_);
  launch(make_voxel_data(view.voxel_grids), view.voxel_grids.obstacle_layers,
         manifest_.collision_voxel, collision_voxel_kernel_);
  if (overwrite) {
    collision_distance_.clear(reinterpret_cast<cudaStream_t>(stream));
    collision_gradient_.clear(reinterpret_cast<cudaStream_t>(stream));
  }
}

void IkRuntime::launch_self_collision(int problems, bool optimizer_cost,
                                      CUstream stream) const {
  if (!config_.check_self_collision || manifest_.num_self_collision_pairs == 0) {
    self_collision_distance_.clear(reinterpret_cast<cudaStream_t>(stream));
    self_collision_gradient_.clear(reinterpret_cast<cudaStream_t>(stream));
    return;
  }
  CUdeviceptr distance = self_collision_distance_.ptr();
  CUdeviceptr gradient = self_collision_gradient_.ptr();
  CUdeviceptr pair_distance = self_collision_pair_distance_.ptr();
  CUdeviceptr sparse_index = self_collision_sparse_index_.ptr();
  CUdeviceptr spheres = robot_spheres_.ptr();
  CUdeviceptr padding = model_->self_collision_padding_.ptr();
  CUdeviceptr weight = optimizer_cost ? optimizer_self_collision_weight_.ptr()
                                      : collision_weight_.ptr();
  CUdeviceptr pairs = model_->self_collision_pairs_.ptr();
  int batch_size = problems;
  int horizon = 1;
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
      static_cast<unsigned int>(problems),
      static_cast<unsigned int>(manifest_.self_collision.block_dim),
      static_cast<unsigned int>(manifest_.self_collision.shared_memory), stream,
      arguments);
}

void IkRuntime::launch_kinematics_backward(int problems,
                                           CUstream stream) const {
  const bool has_collision_gradient =
      manifest_.num_spheres > 0 &&
      (scene_ != nullptr ||
       (config_.check_self_collision &&
        manifest_.num_self_collision_pairs > 0));
  if (!has_collision_gradient) {
    collision_joint_gradient_.clear(reinterpret_cast<cudaStream_t>(stream));
    return;
  }

  auto model = model_->robot_.model_device_view();
  CUdeviceptr output = collision_joint_gradient_.ptr();
  CUdeviceptr zero_position = zero_link_position_gradient_.ptr();
  CUdeviceptr zero_quaternion = zero_link_quaternion_gradient_.ptr();
  CUdeviceptr sphere_gradient = collision_gradient_.ptr();
  CUdeviceptr sphere_gradient_addend = self_collision_gradient_.ptr();
  CUdeviceptr null_pointer = 0;
  CUdeviceptr global_cumulative = global_cumulative_.ptr();
  CUdeviceptr link_spheres = model_->collision_->link_spheres.ptr();
  CUdeviceptr link_sphere_map = model_->collision_->link_sphere_map.ptr();
  CUdeviceptr scene_indices = scene_indices_.ptr();
  int batch_size = problems;
  int horizon = 1;
  int num_scenes = 1;
  std::vector<void*> arguments{
      &output,
      &zero_position,
      &zero_quaternion,
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
      static_cast<unsigned int>(problems),
      static_cast<unsigned int>(manifest_.kinematics_backward.block_dim),
      static_cast<unsigned int>(manifest_.kinematics_backward.shared_memory),
      stream, arguments);
}

void IkRuntime::launch_cspace(CUdeviceptr joint_position, int problems,
                              CUstream stream) const {
  const int values = problems * manifest_.num_dof;
  const auto bounds = warp_bounds_1d(values);
  auto position = array_1d(joint_position, values, sizeof(float));
  auto effort = array_1d(zero_effort_.ptr(), values, sizeof(float));
  auto target = array_1d(model_->default_joint_position_.ptr(), manifest_.num_dof,
                         sizeof(float));
  auto target_indices =
      array_1d(zero_indices_.ptr(), problems, sizeof(int32_t));
  auto position_bounds =
      array_1d(model_->position_bounds_.ptr(), 2 * manifest_.num_dof, sizeof(float));
  const auto effort_bounds_pointer =
      config_.check_effort_limits ? model_->rollout_->effort_bounds()
                                  : inactive_effort_bounds_.ptr();
  auto effort_bounds = array_1d(effort_bounds_pointer,
                                2 * manifest_.num_dof, sizeof(float));
  auto weight = array_1d(cspace_weight_.ptr(), 2, sizeof(float));
  auto activation =
      array_1d(cspace_activation_distance_.ptr(), 2, sizeof(float));
  auto target_weight =
      array_1d(cspace_target_weight_.ptr(), 1, sizeof(float));
  auto target_dof_weight = array_1d(
      cspace_target_dof_weight_.ptr(), manifest_.num_dof, sizeof(float));
  auto regularization =
      array_1d(cspace_regularization_weight_.ptr(), 2, sizeof(float));
  auto current_position = array_1d(model_->default_joint_position_.ptr(),
                                   manifest_.num_dof, sizeof(float));
  auto current_velocity =
      array_1d(zero_state_.ptr(), manifest_.num_dof, sizeof(float));
  auto current_indices =
      array_1d(zero_indices_.ptr(), problems, sizeof(int32_t));
  auto velocity_bounds =
      array_1d(model_->velocity_bounds_.ptr(), 2 * manifest_.num_dof, sizeof(float));
  auto state_dt = array_1d(zero_state_.ptr(), 1, sizeof(float));
  auto output_cost = array_1d(cspace_cost_.ptr(), values, sizeof(float));
  auto output_gradient =
      array_1d(cspace_gradient_.ptr(), values, sizeof(float));
  auto output_effort_gradient =
      array_1d(zero_effort_gradient_.ptr(), values, sizeof(float));
  uint8_t write_state_gradient = 1U;
  uint8_t write_effort_gradient =
      config_.check_effort_limits ? 1U : 0U;
  int horizon = 1;
  int dof = manifest_.num_dof;
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds),
      &position,
      &effort,
      &target,
      &target_indices,
      &position_bounds,
      &effort_bounds,
      &weight,
      &activation,
      &target_weight,
      &target_dof_weight,
      &regularization,
      &current_position,
      &current_velocity,
      &current_indices,
      &velocity_bounds,
      &state_dt,
      &output_cost,
      &output_gradient,
      &output_effort_gradient,
      &write_state_gradient,
      &write_effort_gradient,
      &problems,
      &horizon,
      &dof,
  };
  cspace_kernel_.launch(
      grid_for(static_cast<size_t>(values), manifest_.cspace.block_dim),
      static_cast<unsigned int>(manifest_.cspace.block_dim),
      static_cast<unsigned int>(manifest_.cspace.shared_memory), stream,
      arguments);
}

void IkRuntime::evaluate_lbfgs(CUdeviceptr joint_position, int problems,
                               const IkSolveRequest& request,
                               CUstream stream) const {
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
    streams->wait(dynamics, RolloutEvent::STATE_READY);
    model_->rollout_->launch_inverse_dynamics(
        joint_position, zero_derivatives_.ptr(), zero_derivatives_.ptr(),
        zero_effort_.ptr(), rnea_forward_cache_.ptr(), problems,
        streams->driver_stream(RolloutStream::DYNAMICS));
    streams->record(RolloutEvent::DYNAMICS_READY, dynamics);
    streams->wait(cspace, RolloutEvent::DYNAMICS_READY);
  } else {
    streams->wait(cspace, RolloutEvent::STATE_READY);
  }
  launch_cspace(joint_position, problems,
                streams->driver_stream(RolloutStream::CSPACE));
  if (config_.check_effort_limits) {
    model_->rollout_->launch_inverse_dynamics_backward(
        zero_effort_gradient_.ptr(), joint_position, zero_derivatives_.ptr(),
        rnea_forward_cache_.ptr(), effort_position_gradient_.ptr(),
        effort_velocity_gradient_.ptr(), effort_acceleration_gradient_.ptr(),
        problems, streams->driver_stream(RolloutStream::CSPACE));
  }
  streams->record(RolloutEvent::CSPACE_DONE, cspace);

  streams->wait(kinematics, RolloutEvent::STATE_READY);
  launch_kinematics(joint_position, problems,
                    streams->driver_stream(RolloutStream::KINEMATICS));
  streams->record(RolloutEvent::KINEMATICS_READY, kinematics);
  for (const auto cost_stream : {pose, world, self}) {
    streams->wait(cost_stream, RolloutEvent::KINEMATICS_READY);
  }
  launch_error(joint_position, current_error_.ptr(),
               current_position_error_.ptr(), current_orientation_error_.ptr(),
               request, problems, false,
               streams->driver_stream(RolloutStream::POSE));
  streams->record(RolloutEvent::POSE_DONE, pose);
  launch_collision(problems, true,
                   streams->driver_stream(RolloutStream::WORLD));
  streams->record(RolloutEvent::WORLD_DONE, world);
  launch_self_collision(problems, true,
                        streams->driver_stream(RolloutStream::SELF));
  streams->record(RolloutEvent::SELF_DONE, self);
  for (const auto event : {
           RolloutEvent::WORLD_DONE,
           RolloutEvent::SELF_DONE,
       }) {
    streams->wait(kinematics, event);
  }
  launch_kinematics_backward(
      problems, streams->driver_stream(RolloutStream::KINEMATICS));
  streams->record(RolloutEvent::KINEMATICS_BACKWARD_DONE, kinematics);
  streams->wait(primary, RolloutEvent::POSE_DONE);
  streams->wait(primary, RolloutEvent::CSPACE_DONE);
  streams->wait(primary, RolloutEvent::KINEMATICS_BACKWARD_DONE);

  const auto bounds = warp_bounds_1d(problems);
  auto pose_cost = array_1d(current_error_.ptr(), problems, sizeof(float));
  auto pose_gradient =
      array_2d(jt_error_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto cspace_cost = array_1d(
      cspace_cost_.ptr(), problems * manifest_.num_dof, sizeof(float));
  auto cspace_gradient = array_1d(
      cspace_gradient_.ptr(), problems * manifest_.num_dof, sizeof(float));
  auto effort_position_gradient = array_1d(
      effort_position_gradient_.ptr(), problems * manifest_.num_dof,
      sizeof(float));
  auto collision_gradient = array_2d(collision_joint_gradient_.ptr(), problems,
                                     manifest_.num_dof, sizeof(float));
  auto world_cost =
      array_1d(collision_distance_.ptr(), problems, sizeof(float));
  auto self_cost =
      array_1d(self_collision_distance_.ptr(), problems, sizeof(float));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds1*>(&bounds),
      &pose_cost,
      &pose_gradient,
      &cspace_cost,
      &cspace_gradient,
      &effort_position_gradient,
      &collision_gradient,
      &world_cost,
      &self_cost,
  };
  combine_lbfgs_rollout_kernel_.launch(
      grid_for(static_cast<size_t>(problems),
               manifest_.combine_lbfgs_rollout.block_dim),
      static_cast<unsigned int>(manifest_.combine_lbfgs_rollout.block_dim),
      static_cast<unsigned int>(manifest_.combine_lbfgs_rollout.shared_memory),
      stream, arguments);
}

void IkRuntime::prepare_lbfgs(int problems, CUstream stream) const {
  history_rho_.clear(reinterpret_cast<cudaStream_t>(stream));
  history_y_.clear(reinterpret_cast<cudaStream_t>(stream));
  history_s_.clear(reinterpret_cast<cudaStream_t>(stream));
  const auto bounds = warp_bounds_1d(problems);
  auto seed_action =
      array_2d(joint_position_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto seed_cost = array_1d(current_error_.ptr(), problems, sizeof(float));
  auto seed_gradient =
      array_2d(jt_error_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto action = array_2d(action_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto gradient =
      array_2d(gradient_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto cost = array_1d(cost_.ptr(), problems, sizeof(float));
  auto exploration_action = array_2d(
      exploration_action_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto exploration_gradient = array_2d(
      exploration_gradient_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto exploration_cost =
      array_1d(exploration_cost_.ptr(), problems, sizeof(float));
  auto best_action =
      array_2d(best_action_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto best_cost = array_1d(best_cost_.ptr(), problems, sizeof(float));
  auto best_iteration =
      array_1d(best_iteration_.ptr(), problems, sizeof(int32_t));
  auto current_iteration =
      array_1d(current_iteration_.ptr(), problems, sizeof(int32_t));
  auto converged = array_1d(converged_.ptr(), problems, sizeof(uint8_t));
  auto step_direction = array_2d(
      step_direction_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto reference_action = array_2d(
      reference_action_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto reference_gradient = array_2d(
      reference_gradient_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto history_head =
      array_1d(history_head_.ptr(), problems, sizeof(int32_t));
  auto history_count =
      array_1d(history_count_.ptr(), problems, sizeof(int32_t));
  auto curvature =
      array_1d(curvature_update_valid_.ptr(), problems, sizeof(bool));
  auto step_max = array_1d(optimization_step_max_.ptr(), manifest_.num_dof,
                           sizeof(float));
  float initial_step_scale = manifest_.initial_step_scale;
  auto beta = array_1d(line_search_beta_.ptr(), problems, sizeof(float));
  uint8_t clear_optimizer_state = 1U;
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
      &clear_optimizer_state,
      &beta,
  };
  prepare_initial_state_kernel_.launch(
      grid_for(static_cast<size_t>(problems),
               manifest_.prepare_initial_state.block_dim),
      static_cast<unsigned int>(manifest_.prepare_initial_state.block_dim),
      static_cast<unsigned int>(manifest_.prepare_initial_state.shared_memory),
      stream, arguments);
}

void IkRuntime::prepare_search_actions(int problems, CUstream stream) const {
  auto beta = array_1d(line_search_beta_.ptr(), problems, sizeof(float));
  const int candidates = problems * manifest_.num_line_search;
  const auto bounds = warp_bounds_2d(candidates, manifest_.num_dof);
  auto exploration = array_2d(
      exploration_action_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto direction = array_2d(
      step_direction_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto scales = array_1d(
      line_search_scale_.ptr(), manifest_.num_line_search, sizeof(float));
  auto search = array_2d(
      search_actions_.ptr(), candidates, manifest_.num_dof, sizeof(float));
  std::vector<void*> arguments{
      const_cast<WarpLaunchBounds2*>(&bounds),
      &exploration,
      &direction,
      &scales,
      &beta,
      &search,
  };
  const size_t elements = static_cast<size_t>(candidates) * manifest_.num_dof;
  prepare_search_actions_kernel_.launch(
      grid_for(elements, prepare_search_actions_spec_.block_dim),
      static_cast<unsigned int>(prepare_search_actions_spec_.block_dim),
      static_cast<unsigned int>(prepare_search_actions_spec_.shared_memory),
      stream, arguments);
}

void IkRuntime::line_search(int problems, CUstream stream) const {
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
  CUdeviceptr search_cost = current_error_.ptr();
  CUdeviceptr search_action = search_actions_.ptr();
  CUdeviceptr search_gradient = jt_error_.ptr();
  CUdeviceptr step_direction = step_direction_.ptr();
  CUdeviceptr search_magnitudes = line_search_scale_.ptr();
  CUdeviceptr line_search_index = line_search_index_.ptr();
  CUdeviceptr line_search_status = line_search_status_.ptr();
  CUdeviceptr curvature = curvature_update_valid_.ptr();
  float armijo_threshold = manifest_.armijo_threshold;
  float curvature_threshold = manifest_.curvature_threshold;
  int opt_dim = manifest_.num_dof;
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
      &problems,
      &beta,
  };
  line_search_kernel_.launch(
      static_cast<unsigned int>(problems),
      static_cast<unsigned int>(line_search_spec_.block_dim),
      static_cast<unsigned int>(line_search_spec_.shared_memory), stream,
      arguments);
}

void IkRuntime::lbfgs_step(int problems, CUstream stream) const {
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
  int history = manifest_.history;
  int opt_dim = manifest_.num_dof;
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
      &problems,
      &history,
      &opt_dim,
      &stable_mode,
      &history_head,
      &history_count,
      &scale_step,
      &curvature,
      &line_search_status,
  };
  lbfgs_kernel_.launch(
      static_cast<unsigned int>(problems),
      static_cast<unsigned int>(manifest_.lbfgs.block_dim),
      static_cast<unsigned int>(manifest_.lbfgs.shared_memory), stream,
      arguments);
}

bool IkRuntime::same_graph_request(const IkSolveRequest& left,
                                   const IkSolveRequest& right) {
  return left.goal_positions == right.goal_positions &&
         left.goal_quaternions == right.goal_quaternions &&
         left.seed_positions == right.seed_positions &&
         left.current_positions == right.current_positions &&
         left.solution == right.solution && left.success == right.success &&
         left.position_error == right.position_error &&
         left.orientation_error == right.orientation_error &&
         left.collision_cost == right.collision_cost &&
         left.goalset_index == right.goalset_index &&
         left.batch_size == right.batch_size &&
         left.num_goalset == right.num_goalset &&
         left.num_solutions == right.num_solutions;
}

void IkRuntime::destroy_graph(GraphCache* cache) noexcept {
  if (cache->executable != nullptr) {
    cudaGraphExecDestroy(cache->executable);
  }
  if (cache->graph != nullptr) {
    cudaGraphDestroy(cache->graph);
  }
  *cache = {};
}

void IkRuntime::launch_graph_stage(GraphCache* cache,
                                   const IkSolveRequest& request,
                                   bool lm_stage) const {
  if (cache->initialized && same_graph_request(cache->request, request)) {
    CRR_CUDA_CHECK(cudaGraphLaunch(cache->executable, request.stream));
    cache->last_stream = request.stream;
    return;
  }

  if (cache->initialized) {
    CRR_CUDA_CHECK(cudaStreamSynchronize(cache->last_stream));
  }
  destroy_graph(cache);
  CRR_CUDA_CHECK(cudaStreamSynchronize(graph_capture_stream_));
  auto capture_request = request;
  capture_request.stream = graph_capture_stream_;
  CRR_CUDA_CHECK(cudaStreamBeginCapture(graph_capture_stream_,
                                        cudaStreamCaptureModeThreadLocal));
  try {
    if (lm_stage) {
      solve_lm(capture_request,
               reinterpret_cast<CUstream>(graph_capture_stream_));
    } else {
      solve_lbfgs(capture_request,
                  reinterpret_cast<CUstream>(graph_capture_stream_));
    }
    CRR_CUDA_CHECK(cudaStreamEndCapture(graph_capture_stream_, &cache->graph));
    CRR_CUDA_CHECK(
        cudaGraphInstantiate(&cache->executable, cache->graph, 0));
  } catch (...) {
    cudaGraph_t abandoned_graph = nullptr;
    cudaStreamEndCapture(graph_capture_stream_, &abandoned_graph);
    if (abandoned_graph != nullptr) {
      cudaGraphDestroy(abandoned_graph);
    }
    destroy_graph(cache);
    throw;
  }
  cache->request = request;
  cache->initialized = true;
  CRR_CUDA_CHECK(cudaGraphLaunch(cache->executable, request.stream));
  cache->last_stream = request.stream;
}

bool IkRuntime::lm_result_can_exit_early(
    const IkSolveRequest& request, cudaStream_t stream, uint8_t* host_success,
    std::size_t host_success_count) const {
  const auto output_count = checked_product(
      static_cast<size_t>(request.batch_size),
      static_cast<size_t>(request.num_solutions), "IK success output");
  if (host_success != nullptr && host_success_count < output_count) {
    throw std::invalid_argument("IK host success buffer is too small");
  }
  std::vector<uint8_t> owned_success;
  if (host_success == nullptr) {
    owned_success.resize(output_count, 0U);
    host_success = owned_success.data();
  }
  CRR_CUDA_CHECK(cudaMemcpyAsync(host_success, request.success, output_count,
                                 cudaMemcpyDeviceToHost, stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int batch_index = 0; batch_index < request.batch_size; ++batch_index) {
    const auto begin =
        host_success + static_cast<size_t>(batch_index) * request.num_solutions;
    const auto end = begin + request.num_solutions;
    if (std::none_of(begin, end,
                     [](uint8_t value) { return value != 0U; })) {
      return false;
    }
  }
  return true;
}

bool IkRuntime::solve(const IkSolveRequest& request, uint8_t* host_success,
                      std::size_t host_success_count) const {
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  if (request.goal_positions == nullptr || request.goal_quaternions == nullptr ||
      request.solution == nullptr || request.success == nullptr ||
      request.position_error == nullptr || request.orientation_error == nullptr ||
      request.collision_cost == nullptr) {
    throw std::invalid_argument("IK input and output CUDA addresses must be non-null");
  }
  if (request.batch_size <= 0 || request.batch_size > config_.max_batch_size) {
    throw std::invalid_argument("IK batch_size exceeds the configured workspace");
  }
  if (request.num_goalset <= 0 || request.num_goalset > config_.max_goalset) {
    throw std::invalid_argument("IK num_goalset exceeds the configured workspace");
  }
  if (request.num_solutions <= 0 ||
      request.num_solutions > config_.num_seeds) {
    throw std::invalid_argument(
        "IK num_solutions must be within the configured seed count");
  }
  if (request.seed_offset < 0) {
    throw std::invalid_argument("IK seed_offset must be non-negative");
  }
  if (request.num_goalset > 1 && request.goalset_index == nullptr) {
    throw std::invalid_argument(
        "IK goalset_index output is required when num_goalset is greater than one");
  }
  const auto problem_count =
      static_cast<int64_t>(request.batch_size) * config_.num_seeds;
  if (problem_count > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("IK problem count exceeds the kernel integer range");
  }
  // Ordered on the solve stream before graph replay; no host staging lifetime
  // or captured scalar argument. The offset allocation stays fixed per solver.
  CRR_CU_CHECK(cuMemsetD32Async(seed_offset_.ptr(),
                               request.seed_offset % manifest_.num_seed_samples, 1,
                               reinterpret_cast<CUstream>(request.stream)));
  if (!config_.use_cuda_graph) {
    return solve_eager(request, reinterpret_cast<CUstream>(request.stream),
                       host_success, host_success_count);
  }
  launch_graph_stage(&lm_graph_, request, true);
  if (config_.exit_early &&
      lm_result_can_exit_early(request, request.stream, host_success,
                               host_success_count)) {
    return true;
  }
  launch_graph_stage(&lbfgs_graph_, request, false);
  return false;
}

bool IkRuntime::solve_eager(const IkSolveRequest& request, CUstream stream,
                            uint8_t* host_success,
                            std::size_t host_success_count) const {
  solve_lm(request, stream);
  if (config_.exit_early && lm_result_can_exit_early(
                                request, reinterpret_cast<cudaStream_t>(stream),
                                host_success, host_success_count)) {
    return true;
  }
  solve_lbfgs(request, stream);
  return false;
}

void IkRuntime::solve_lm(const IkSolveRequest& request, CUstream stream) const {
  const int problems = request.batch_size * config_.num_seeds;

  const auto initialize_bounds =
      warp_tiled_bounds_2d(problems, manifest_.num_dof);
  auto template_seeds = array_2d(model_->seed_positions_.ptr(), manifest_.num_seed_samples,
                                 manifest_.num_dof, sizeof(float));
  const auto request_seed_pointer =
      request.seed_positions == nullptr
          ? model_->seed_positions_.ptr()
          : static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(request.seed_positions));
  auto request_seeds =
      array_2d(request_seed_pointer, problems, manifest_.num_dof, sizeof(float));
  const auto current_seed_pointer =
      request.current_positions == nullptr
          ? model_->default_joint_position_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.current_positions));
  auto current_seeds = array_2d(current_seed_pointer, request.batch_size,
                                manifest_.num_dof, sizeof(float));
  auto default_seed = array_1d(model_->default_joint_position_.ptr(),
                               manifest_.num_dof, sizeof(float));
  auto lower = array_1d(model_->joint_position_lower_.ptr(), manifest_.num_dof, sizeof(float));
  auto upper = array_1d(model_->joint_position_upper_.ptr(), manifest_.num_dof, sizeof(float));
  auto joints = array_2d(joint_position_.ptr(), problems, manifest_.num_dof, sizeof(float));
  auto damping = array_1d(damping_.ptr(), problems, sizeof(float));
  int num_seeds = config_.num_seeds;
  int max_template_seeds = manifest_.num_seed_samples;
  auto seed_offset = array_1d(seed_offset_.ptr(), 1, sizeof(int32_t));
  uint8_t use_request_seeds = request.seed_positions == nullptr ? 0U : 1U;
  uint8_t use_current_seed = request.current_positions == nullptr ? 0U : 1U;
  float initial_damping = config_.initial_damping;
  std::vector<void*> initialize_arguments{
      const_cast<WarpLaunchBounds2*>(&initialize_bounds),
      &template_seeds,
      &request_seeds,
      &current_seeds,
      &default_seed,
      &lower,
      &upper,
      &joints,
      &damping,
      &num_seeds,
      &max_template_seeds,
      &seed_offset,
      &use_request_seeds,
      &use_current_seed,
      &initial_damping,
  };
  const auto joint_values =
      static_cast<size_t>(problems) * static_cast<size_t>(manifest_.num_dof);
  const int initialize_block_size = manifest_.initialize.block_dim;
  initialize_kernel_.launch(grid_for(joint_values, initialize_block_size),
                            static_cast<unsigned int>(initialize_block_size),
                            static_cast<unsigned int>(manifest_.initialize.shared_memory),
                            stream, initialize_arguments);

  // Match Python IK staging: pose-only LM produces seeds for collision-aware
  // L-BFGS. Collision kernels are intentionally absent from this LM loop.
  for (int iteration = 0; iteration < config_.lm_iterations; ++iteration) {
    launch_kinematics(joint_position_.ptr(), problems, stream);
    launch_error(joint_position_.ptr(), current_error_.ptr(),
                 current_position_error_.ptr(),
                 current_orientation_error_.ptr(), request, problems, true,
                 stream);
    if (iteration > 0) {
      // SeedIterationStateManager carries the last candidate objective into
      // the next trust-ratio calculation even when it retains the previous
      // joint state. Preserve that Python LM state transition while the
      // recomputed Jacobian and gradient continue to follow the retained
      // joints.
      CRR_CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<void*>(static_cast<uintptr_t>(current_error_.ptr())),
          reinterpret_cast<const void*>(
              static_cast<uintptr_t>(candidate_error_.ptr())),
          static_cast<size_t>(problems) * sizeof(float),
          cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
    }
    const int lm_block_size = manifest_.levenberg_marquardt.block_dim;
    const auto lm_bounds = warp_tiled_bounds_1d(problems, lm_block_size);
    auto jacobian = array_3d(scaled_jacobian_.ptr(), problems,
                             manifest_.num_residuals, manifest_.num_dof,
                             sizeof(float));
    auto gradient = array_2d(jt_error_.ptr(), problems, manifest_.num_dof,
                             sizeof(float));
    auto damping_values = array_1d(damping_.ptr(), problems, sizeof(float));
    auto current_joints = array_2d(joint_position_.ptr(), problems,
                                   manifest_.num_dof, sizeof(float));
    auto candidate_joints = array_2d(candidate_position_.ptr(), problems,
                                     manifest_.num_dof, sizeof(float));
    auto predicted =
        array_1d(predicted_reduction_.ptr(), problems, sizeof(float));
    std::vector<void*> lm_arguments{
        const_cast<WarpLaunchBounds1*>(&lm_bounds),
        &jacobian,
        &gradient,
        &damping_values,
        &current_joints,
        &candidate_joints,
        &predicted,
    };
    lm_kernel_.launch(
        static_cast<unsigned int>(problems),
        static_cast<unsigned int>(lm_block_size),
        static_cast<unsigned int>(manifest_.levenberg_marquardt.shared_memory),
        stream, lm_arguments);

    launch_kinematics(candidate_position_.ptr(), problems, stream);
    launch_error(candidate_position_.ptr(), candidate_error_.ptr(),
                 candidate_position_error_.ptr(),
                 candidate_orientation_error_.ptr(), request, problems, true,
                 stream);

    const auto update_bounds = warp_bounds_1d(problems);
    auto current_error =
        array_1d(current_error_.ptr(), problems, sizeof(float));
    auto candidate_error =
        array_1d(candidate_error_.ptr(), problems, sizeof(float));
    float minimum_trust_ratio = config_.minimum_trust_ratio;
    float damping_factor = config_.damping_factor;
    float minimum_damping = config_.minimum_damping;
    float maximum_damping = config_.maximum_damping;
    std::vector<void*> update_arguments{
        const_cast<WarpLaunchBounds1*>(&update_bounds),
        &current_joints,
        &candidate_joints,
        &current_error,
        &candidate_error,
        &predicted,
        &damping_values,
        &minimum_trust_ratio,
        &damping_factor,
        &minimum_damping,
        &maximum_damping,
    };
    const int update_block_size = manifest_.update.block_dim;
    update_kernel_.launch(
        grid_for(static_cast<size_t>(problems), update_block_size),
        static_cast<unsigned int>(update_block_size),
        static_cast<unsigned int>(manifest_.update.shared_memory), stream,
        update_arguments);
  }

  if (config_.exit_early) {
    finalize_solution(joint_position_.ptr(), request, problems, stream);
  }
}

void IkRuntime::solve_lbfgs(const IkSolveRequest& request,
                            CUstream stream) const {
  const int problems = request.batch_size * config_.num_seeds;
  evaluate_lbfgs(joint_position_.ptr(), problems, request, stream);
  prepare_lbfgs(problems, stream);
  const int candidates = problems * manifest_.num_line_search;
  for (int iteration = 0; iteration < config_.lbfgs_iterations; ++iteration) {
    prepare_search_actions(problems, stream);
    evaluate_lbfgs(search_actions_.ptr(), candidates, request, stream);
    line_search(problems, stream);
    lbfgs_step(problems, stream);
  }
  const CUdeviceptr final_joint_position = best_action_.ptr();

  finalize_solution(final_joint_position, request, problems, stream);
}

void IkRuntime::finalize_solution(CUdeviceptr final_joint_position,
                                  const IkSolveRequest& request, int problems,
                                  CUstream stream) const {
  auto* streams = rollout_streams_.get();
  const auto primary = reinterpret_cast<cudaStream_t>(stream);
  const auto dynamics = streams->cuda_stream(RolloutStream::DYNAMICS);
  const auto kinematics = streams->cuda_stream(RolloutStream::KINEMATICS);
  const auto pose = streams->cuda_stream(RolloutStream::POSE);
  const auto world = streams->cuda_stream(RolloutStream::WORLD);
  const auto self = streams->cuda_stream(RolloutStream::SELF);
  streams->record(RolloutEvent::STATE_READY, primary);

  if (config_.check_effort_limits) {
    streams->wait(dynamics, RolloutEvent::STATE_READY);
    model_->rollout_->launch_inverse_dynamics(
        final_joint_position, zero_derivatives_.ptr(), zero_derivatives_.ptr(),
        zero_effort_.ptr(), rnea_forward_cache_.ptr(), problems,
        streams->driver_stream(RolloutStream::DYNAMICS));
    streams->record(RolloutEvent::DYNAMICS_READY, dynamics);
  }

  streams->wait(kinematics, RolloutEvent::STATE_READY);
  launch_kinematics(final_joint_position, problems,
                    streams->driver_stream(RolloutStream::KINEMATICS));
  streams->record(RolloutEvent::KINEMATICS_READY, kinematics);
  for (const auto cost_stream : {pose, world, self}) {
    streams->wait(cost_stream, RolloutEvent::KINEMATICS_READY);
  }
  launch_error(final_joint_position, current_error_.ptr(),
               current_position_error_.ptr(), current_orientation_error_.ptr(), request,
               problems, false, streams->driver_stream(RolloutStream::POSE));
  streams->record(RolloutEvent::POSE_DONE, pose);
  launch_collision(problems, false,
                   streams->driver_stream(RolloutStream::WORLD));
  streams->record(RolloutEvent::WORLD_DONE, world);
  launch_self_collision(problems, false,
                        streams->driver_stream(RolloutStream::SELF));
  streams->record(RolloutEvent::SELF_DONE, self);
  for (const auto event : {
           RolloutEvent::POSE_DONE,
           RolloutEvent::WORLD_DONE,
           RolloutEvent::SELF_DONE,
       }) {
    streams->wait(primary, event);
  }
  if (config_.check_effort_limits) {
    streams->wait(primary, RolloutEvent::DYNAMICS_READY);
  }

  const int output_count = request.batch_size * request.num_solutions;
  int num_solutions = request.num_solutions;
  const auto finalize_bounds = warp_tiled_bounds_2d(output_count, manifest_.finalize.block_dim);
  auto final_joints =
      array_2d(final_joint_position, problems, manifest_.num_dof, sizeof(float));
  const auto current_position_pointer =
      request.current_positions == nullptr
          ? model_->default_joint_position_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.current_positions));
  auto current_positions = array_2d(current_position_pointer,
                                    request.batch_size, manifest_.num_dof,
                                    sizeof(float));
  auto position_bounds = array_1d(model_->position_bounds_.ptr(),
                                  2 * manifest_.num_dof, sizeof(float));
  auto position_errors =
      array_1d(current_position_error_.ptr(), problems, sizeof(float));
  auto orientation_errors =
      array_1d(current_orientation_error_.ptr(), problems, sizeof(float));
  auto collision_costs = array_1d(collision_distance_.ptr(), problems, sizeof(float));
  auto self_collision_costs =
      array_1d(self_collision_distance_.ptr(), problems, sizeof(float));
  auto effort =
      array_1d(zero_effort_.ptr(), problems * manifest_.num_dof, sizeof(float));
  const auto effort_bounds_pointer =
      config_.check_effort_limits ? model_->rollout_->effort_bounds()
                                  : inactive_effort_bounds_.ptr();
  auto effort_bounds = array_1d(effort_bounds_pointer,
                                2 * manifest_.num_dof, sizeof(float));
  auto goalset_indices = array_1d(
      goalset_index_.ptr(), problems * manifest_.num_tool_frames,
      sizeof(int32_t));
  auto solution = warp_array_2d(reinterpret_cast<uint64_t>(request.solution),
                                output_count, manifest_.num_dof, sizeof(float));
  auto success = warp_array_1d(reinterpret_cast<uint64_t>(request.success),
                               output_count, sizeof(uint8_t));
  auto output_position_error = warp_array_1d(
      reinterpret_cast<uint64_t>(request.position_error), output_count,
      sizeof(float));
  auto output_orientation_error = warp_array_1d(
      reinterpret_cast<uint64_t>(request.orientation_error), output_count,
      sizeof(float));
  auto output_collision_cost = warp_array_1d(
      reinterpret_cast<uint64_t>(request.collision_cost), output_count,
      sizeof(float));
  const auto output_goalset_pointer =
      request.goalset_index == nullptr
          ? output_goalset_index_.ptr()
          : static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(request.goalset_index));
  auto output_goalset_indices = array_1d(
      output_goalset_pointer,
      output_count * manifest_.num_tool_frames, sizeof(int32_t));
  auto start_cspace_weight =
      array_1d(start_cspace_weight_.ptr(), 1, sizeof(float));
  auto start_cspace_dof_weight = array_1d(
      start_cspace_dof_weight_.ptr(), manifest_.num_dof, sizeof(float));
  uint8_t use_current_position =
      request.current_positions == nullptr ? 0U : 1U;
  int num_seeds = config_.num_seeds;
  float position_tolerance = config_.position_tolerance;
  float orientation_tolerance = config_.orientation_tolerance;
  float position_rank_weight = config_.position_rank_weight;
  float orientation_rank_weight = config_.orientation_rank_weight;
  float cspace_rank_weight = config_.cspace_rank_weight;
  uint8_t effort_active = config_.check_effort_limits ? 1U : 0U;
  const auto rank_bounds = warp_bounds_1d(problems);
  auto rank_score = array_1d(rank_score_.ptr(), problems, sizeof(float));
  auto rank_success = array_1d(rank_success_.ptr(), problems, sizeof(uint8_t));
  std::vector<void*> rank_arguments{
      const_cast<WarpLaunchBounds1*>(&rank_bounds),
      &final_joints,
      &current_positions,
      &position_bounds,
      &position_errors,
      &orientation_errors,
      &collision_costs,
      &self_collision_costs,
      &effort,
      &effort_bounds,
      &start_cspace_weight,
      &start_cspace_dof_weight,
      &rank_success,
      &rank_score,
      &num_seeds,
      &use_current_position,
      &position_tolerance,
      &orientation_tolerance,
      &position_rank_weight,
      &orientation_rank_weight,
      &cspace_rank_weight,
      &effort_active,
  };
  rank_kernel_.launch(grid_for(static_cast<size_t>(problems), manifest_.rank.block_dim),
                      static_cast<unsigned int>(manifest_.rank.block_dim),
                      static_cast<unsigned int>(manifest_.rank.shared_memory), stream,
                      rank_arguments);
  std::vector<void*> finalize_arguments{
      const_cast<WarpLaunchBounds2*>(&finalize_bounds),
      &final_joints,
      &position_errors,
      &orientation_errors,
      &collision_costs,
      &self_collision_costs,
      &goalset_indices,
      &rank_success,
      &rank_score,
      &solution,
      &success,
      &output_position_error,
      &output_orientation_error,
      &output_collision_cost,
      &output_goalset_indices,
      &num_seeds,
      &num_solutions,
  };
  finalize_kernel_.launch(
      static_cast<unsigned int>(output_count),
      static_cast<unsigned int>(manifest_.finalize.block_dim),
      static_cast<unsigned int>(manifest_.finalize.shared_memory), stream,
      finalize_arguments);
}

}  // namespace curobo::robot::detail
