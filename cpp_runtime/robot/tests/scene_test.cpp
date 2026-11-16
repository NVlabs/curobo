// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/scene.h"
#include "scene_runtime.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {

using curobo::robot::Capsule;
using curobo::robot::Cuboid;
using curobo::robot::Cylinder;
using curobo::robot::Mesh;
using curobo::robot::ObstacleType;
using curobo::robot::Pose;
using curobo::robot::Scene;
using curobo::robot::SceneConfig;
using curobo::robot::Sphere;
using curobo::robot::VoxelGrid;

void check(bool condition) {
  if (!condition) {
    throw std::runtime_error("scene test check failed");
  }
}

template <typename ValueType, size_t Size>
std::array<ValueType, Size> download(CUdeviceptr pointer) {
  std::array<ValueType, Size> values{};
  const auto status = cudaMemcpy(values.data(), reinterpret_cast<const void*>(pointer),
                                 sizeof(values), cudaMemcpyDeviceToHost);
  check(status == cudaSuccess);
  return values;
}

template <typename Function>
void expect_invalid_argument(Function&& function) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw);
}

template <typename Function>
void expect_runtime_error(Function&& function) {
  bool threw = false;
  try {
    function();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw);
}

void test_analytic_obstacles() {
  SceneConfig config;
  config.max_cuboids = 1;
  config.max_capsules = 2;
  config.max_cylinders = 1;
  Scene scene(config);

  Cuboid cuboid;
  cuboid.name = "table";
  cuboid.dimensions = {1.0F, 0.8F, 0.1F};
  cuboid.pose.position = {0.0F, 0.0F, -0.05F};
  const auto cuboid_handle = scene.add_cuboid(cuboid);

  Sphere sphere;
  sphere.name = "ball";
  sphere.radius = 0.2F;
  const auto sphere_handle = scene.add_sphere(sphere);

  Capsule capsule;
  capsule.name = "rail";
  capsule.base = {-0.5F, 0.0F, 0.0F};
  capsule.tip = {0.5F, 0.0F, 0.0F};
  capsule.radius = 0.05F;
  scene.add_capsule(capsule);

  Cylinder cylinder;
  cylinder.name = "post";
  cylinder.radius = 0.1F;
  cylinder.height = 0.8F;
  const auto cylinder_handle = scene.add_cylinder(cylinder);

  check(cuboid_handle.valid());
  check(scene.obstacle_count(ObstacleType::CUBOID) == 1);
  check(scene.obstacle_count(ObstacleType::SPHERE) == 1);
  check(scene.obstacle_count(ObstacleType::CAPSULE) == 1);
  check(scene.obstacle_count(ObstacleType::CYLINDER) == 1);

  Pose moved;
  moved.position = {0.2F, 0.3F, 0.4F};
  scene.update_obstacle_pose(cuboid_handle, moved);
  scene.set_obstacle_enabled(sphere_handle, false);

  expect_runtime_error([&] { scene.add_cuboid(cuboid); });
  expect_invalid_argument([&] { scene.update_obstacle_pose({}, moved); });

  scene.clear();
  check(scene.obstacle_count(ObstacleType::CUBOID) == 0);
  check(scene.obstacle_count(ObstacleType::SPHERE) == 0);
  check(scene.obstacle_count(ObstacleType::CAPSULE) == 0);
  check(scene.obstacle_count(ObstacleType::CYLINDER) == 0);
  expect_invalid_argument([&] { scene.set_obstacle_enabled(cylinder_handle, true); });
}

void test_validation_and_move() {
  SceneConfig config;
  config.max_cuboids = 1;
  Scene scene(config);

  Cuboid invalid;
  invalid.name = "invalid";
  invalid.dimensions = {1.0F, 0.0F, 1.0F};
  expect_invalid_argument([&] { scene.add_cuboid(invalid); });

  Cuboid first;
  first.name = "first";
  first.dimensions = {1.0F, 1.0F, 1.0F};
  const auto first_handle = scene.add_cuboid(first);

  Scene moved(std::move(scene));
  check(moved.obstacle_count(ObstacleType::CUBOID) == 1);

  Scene other(config);
  Cuboid second;
  second.name = "second";
  const auto other_handle = other.add_cuboid(second);
  check(other_handle.valid());
  expect_invalid_argument([&] { other.set_obstacle_enabled(first_handle, false); });
}

void test_device_layout() {
  SceneConfig config;
  config.max_cuboids = 1;
  config.max_capsules = 0;
  config.max_cylinders = 0;
  config.max_meshes = 0;
  curobo::robot::detail::SceneRuntime scene(config);

  Cuboid cuboid;
  cuboid.name = "box";
  cuboid.pose.position = {1.0F, 2.0F, 3.0F};
  cuboid.dimensions = {2.0F, 4.0F, 6.0F};
  const auto handle = scene.add_cuboid(cuboid);

  const auto view = scene.device_view();
  check(view.cuboids.capacity == 1);
  check(view.cuboids.param_stride == 4);
  check(download<float, 4>(view.cuboids.params) ==
        std::array<float, 4>{2.0F, 4.0F, 6.0F, 0.0F});
  check(download<float, 8>(view.cuboids.inverse_poses) ==
        std::array<float, 8>{-1.0F, -2.0F, -3.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                             0.0F});
  check(download<float, 8>(view.cuboids.world_bounds) ==
        std::array<float, 8>{1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F, 0.0F});
  check(download<uint8_t, 1>(view.cuboids.enabled).at(0) == 1U);
  check(download<int32_t, 1>(view.cuboids.count).at(0) == 1);
  const auto layers = view.cuboids.obstacle_layers;
  check(download<uint32_t, 1>(layers).at(0) == 1U);
  scene.set_obstacle_layers(handle, 0x80000004U);
  check(download<uint32_t, 1>(layers).at(0) == 0x80000004U);
  check(scene.device_view().cuboids.obstacle_layers == layers);

  Pose updated_pose;
  updated_pose.quaternion = {0.82812649F, 0.0F, 0.0F, -0.56054151F};
  scene.update_obstacle_pose(handle, updated_pose);
  const auto updated_inverse_pose =
      download<float, 8>(scene.device_view().cuboids.inverse_poses);
  check(updated_inverse_pose.at(3) == updated_pose.quaternion.at(0));
  check(updated_inverse_pose.at(4) == -updated_pose.quaternion.at(1));
  check(updated_inverse_pose.at(5) == -updated_pose.quaternion.at(2));
  check(updated_inverse_pose.at(6) == -updated_pose.quaternion.at(3));
  scene.clear();
  scene.add_cuboid(cuboid);
  check(scene.device_view().cuboids.obstacle_layers == layers);
  check(download<uint32_t, 1>(layers).at(0) == 1U);
  expect_invalid_argument([&] { scene.set_obstacle_layers(handle, 1U); });
}

void test_voxel_grid() {
  SceneConfig config;
  config.max_voxel_grids = 2;
  config.max_voxels_per_grid = 8;
  Scene scene(config);

  VoxelGrid grid;
  grid.name = "esdf";
  grid.dimensions = {0.2F, 0.2F, 0.2F};
  grid.voxel_size = 0.1F;
  grid.esdf = {-0.05F, 0.05F, 0.05F, 0.05F, 0.05F, 0.05F, 0.05F, 0.05F};
  scene.add_voxel_grid(grid);
  check(scene.obstacle_count(ObstacleType::VOXEL_GRID) == 1);

  grid.name = "bad_esdf";
  grid.esdf.pop_back();
  expect_invalid_argument([&] { scene.add_voxel_grid(grid); });
}

void test_cuda_voxel_grid_update() {
  SceneConfig config;
  config.max_voxel_grids = 1;
  config.max_voxels_per_grid = 8;
  curobo::robot::detail::SceneRuntime scene(config);

  VoxelGrid grid;
  grid.name = "live_esdf";
  grid.dimensions = {0.2F, 0.2F, 0.2F};
  grid.voxel_size = 0.1F;
  grid.esdf.assign(8, 0.0F);
  const auto handle = scene.add_voxel_grid(grid);
  check(scene.voxel_grid_cuda_address(handle) ==
        static_cast<uint64_t>(scene.device_view().voxel_grids.features));

  const std::array<uint16_t, 8> source{
      0x2E66U, 0x3266U, 0x34CDU, 0x3666U,
      0x3800U, 0x38CDU, 0x399AU, 0x3A66U};
  uint16_t* device_source = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device_source), sizeof(source)) ==
        cudaSuccess);
  check(cudaMemcpy(device_source, source.data(), sizeof(source),
                   cudaMemcpyHostToDevice) == cudaSuccess);
  scene.update_voxel_grid_cuda(
      handle, reinterpret_cast<uint64_t>(device_source), source.size(), 0);
  check(cudaDeviceSynchronize() == cudaSuccess);

  const auto actual = download<uint16_t, 8>(scene.device_view().voxel_grids.features);
  check(actual == source);
  check(cudaFree(device_source) == cudaSuccess);

  expect_invalid_argument(
      [&] { scene.update_voxel_grid_cuda(handle, 0, source.size(), 0); });
  expect_invalid_argument([&] {
    scene.update_voxel_grid_cuda(handle, reinterpret_cast<uint64_t>(device_source),
                                 source.size() - 1, 0);
  });
}

void test_mesh_if_available() {
  SceneConfig config;
  config.max_meshes = 1;
  Scene scene(config);

  Mesh mesh;
  mesh.name = "tetrahedron";
  mesh.vertices = {
      {0.0F, 0.0F, 0.0F},
      {1.0F, 0.0F, 0.0F},
      {0.0F, 1.0F, 0.0F},
      {0.0F, 0.0F, 1.0F},
  };
  mesh.triangles = {
      {0, 2, 1},
      {0, 1, 3},
      {0, 3, 2},
      {1, 2, 3},
  };
  if (!scene.mesh_support_available()) {
    expect_runtime_error([&] { scene.add_mesh(mesh); });
    return;
  }
  const auto handle = scene.add_mesh(mesh);
  check(handle.valid());
  check(scene.obstacle_count(ObstacleType::MESH) == 1);
  scene.clear();
  check(scene.add_mesh(mesh).valid());
}

}  // namespace

int main() {
  test_analytic_obstacles();
  test_validation_and_move();
  test_device_layout();
  test_voxel_grid();
  test_cuda_voxel_grid_update();
  test_mesh_if_available();
  return EXIT_SUCCESS;
}
