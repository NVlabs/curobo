// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>

#include <memory>

#include "curobo_robot_runtime/scene.h"

namespace curobo::robot::detail {

struct ObstacleDeviceView {
  CUdeviceptr params{0};
  CUdeviceptr world_bounds{0};
  CUdeviceptr inverse_poses{0};
  CUdeviceptr enabled{0};
  CUdeviceptr obstacle_layers{0};
  CUdeviceptr count{0};
  int capacity{0};
  int param_stride{0};
};

struct MeshDeviceView {
  CUdeviceptr mesh_ids{0};
  CUdeviceptr dimensions{0};
  CUdeviceptr world_bounds{0};
  CUdeviceptr inverse_poses{0};
  CUdeviceptr enabled{0};
  CUdeviceptr obstacle_layers{0};
  CUdeviceptr count{0};
  int capacity{0};
  float max_distance{0.1F};
};

struct VoxelDeviceView {
  CUdeviceptr params{0};
  CUdeviceptr dimensions{0};
  CUdeviceptr inverse_poses{0};
  CUdeviceptr enabled{0};
  CUdeviceptr obstacle_layers{0};
  CUdeviceptr features{0};
  CUdeviceptr count{0};
  int capacity{0};
  int max_voxels_per_grid{0};
  float max_distance{100.0F};
};

/** Internal single-scene buffers consumed by robot collision solvers. */
struct SceneDeviceView {
  ObstacleDeviceView cuboids;
  ObstacleDeviceView capsules;
  ObstacleDeviceView cylinders;
  MeshDeviceView meshes;
  VoxelDeviceView voxel_grids;
};

class SceneRuntime final {
 public:
  explicit SceneRuntime(const SceneConfig& config);
  ~SceneRuntime();

  SceneRuntime(const SceneRuntime&) = delete;
  SceneRuntime& operator=(const SceneRuntime&) = delete;

  const SceneConfig& config() const;
  SceneDeviceView device_view() const;
  int obstacle_count(ObstacleType type) const;
  bool mesh_support_available() const noexcept;

  ObstacleHandle add_cuboid(const Cuboid& cuboid);
  ObstacleHandle add_capsule(const Capsule& capsule);
  ObstacleHandle add_cylinder(const Cylinder& cylinder);
  ObstacleHandle add_sphere(const Sphere& sphere);
  ObstacleHandle add_mesh(const Mesh& mesh);
  ObstacleHandle add_voxel_grid(const VoxelGrid& voxel_grid);
  uint64_t voxel_grid_cuda_address(ObstacleHandle handle) const;
  void update_voxel_grid_cuda(ObstacleHandle handle, uint64_t source_esdf_fp16,
                              std::size_t voxel_count, uint64_t stream);

  void update_obstacle_pose(ObstacleHandle handle, const Pose& pose);
  void set_obstacle_enabled(ObstacleHandle handle, bool enabled);
  /** Set unsigned layer membership in stable scene-owned GPU storage. */
  void set_obstacle_layers(ObstacleHandle handle, uint32_t layers);
  void clear();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace curobo::robot::detail
