// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/scene.h"

#include "scene_runtime.h"

#include <utility>

namespace curobo::robot {

Scene::Scene(const SceneConfig& config)
    : runtime_(std::make_unique<detail::SceneRuntime>(config)) {}

Scene::~Scene() = default;

Scene::Scene(Scene&& other) noexcept = default;

Scene& Scene::operator=(Scene&& other) noexcept = default;

const SceneConfig& Scene::config() const { return runtime_->config(); }

int Scene::obstacle_count(ObstacleType type) const {
  return runtime_->obstacle_count(type);
}

bool Scene::mesh_support_available() const noexcept {
  return runtime_ != nullptr && runtime_->mesh_support_available();
}

ObstacleHandle Scene::add_cuboid(const Cuboid& cuboid) {
  return runtime_->add_cuboid(cuboid);
}

ObstacleHandle Scene::add_capsule(const Capsule& capsule) {
  return runtime_->add_capsule(capsule);
}

ObstacleHandle Scene::add_cylinder(const Cylinder& cylinder) {
  return runtime_->add_cylinder(cylinder);
}

ObstacleHandle Scene::add_sphere(const Sphere& sphere) {
  return runtime_->add_sphere(sphere);
}

ObstacleHandle Scene::add_mesh(const Mesh& mesh) {
  return runtime_->add_mesh(mesh);
}

ObstacleHandle Scene::add_voxel_grid(const VoxelGrid& voxel_grid) {
  return runtime_->add_voxel_grid(voxel_grid);
}

uint64_t Scene::voxel_grid_cuda_address(ObstacleHandle handle) const {
  return runtime_->voxel_grid_cuda_address(handle);
}

void Scene::update_voxel_grid_cuda(ObstacleHandle handle,
                                   uint64_t source_esdf_fp16,
                                   std::size_t voxel_count,
                                   uint64_t stream) {
  runtime_->update_voxel_grid_cuda(handle, source_esdf_fp16, voxel_count, stream);
}

void Scene::update_obstacle_pose(ObstacleHandle handle, const Pose& pose) {
  runtime_->update_obstacle_pose(handle, pose);
}

void Scene::set_obstacle_enabled(ObstacleHandle handle, bool enabled) {
  runtime_->set_obstacle_enabled(handle, enabled);
}

void Scene::set_obstacle_layers(ObstacleHandle handle, uint32_t layers) {
  runtime_->set_obstacle_layers(handle, layers);
}

void Scene::clear() { runtime_->clear(); }

}  // namespace curobo::robot
