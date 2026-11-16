// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/sphere_fit.h"
#include "attachment_runtime.h"
#include "robot_runtime.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

using curobo::robot::Mesh;
using curobo::robot::SphereFitConfig;
using curobo::robot::SphereAttachment;
using curobo::robot::detail::LinkSphereAttachment;
using curobo::robot::fit_spheres_to_mesh;

void check(bool condition) {
  if (!condition) {
    throw std::runtime_error("sphere fit test check failed");
  }
}

Mesh unit_cube() {
  Mesh mesh;
  mesh.name = "cube";
  mesh.vertices = {
      {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
      {1.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F},
      {0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 1.0F},
      {1.0F, 1.0F, 1.0F}, {0.0F, 1.0F, 1.0F},
  };
  mesh.triangles = {
      {0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
      {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5},
      {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7},
  };
  return mesh;
}

void test_cube_fit() {
  SphereFitConfig config;
  config.num_spheres = 8;
  const auto result = fit_spheres_to_mesh(unit_cube(), config);
  check(result.spheres.size() == 8);
  check(result.candidate_count == 8);
  check(std::abs(result.voxel_pitch - 0.5F) < 1.0E-6F);
  for (const auto& sphere : result.spheres) {
    check(sphere.center.at(0) > 0.0F && sphere.center.at(0) < 1.0F);
    check(sphere.center.at(1) > 0.0F && sphere.center.at(1) < 1.0F);
    check(sphere.center.at(2) > 0.0F && sphere.center.at(2) < 1.0F);
    check(std::abs(sphere.radius - 0.25F) < 1.0E-5F);
  }
}

void test_pose_is_not_applied() {
  auto mesh = unit_cube();
  mesh.pose.position = {10.0F, -5.0F, 2.0F};
  const auto result = fit_spheres_to_mesh(mesh, SphereFitConfig{});
  check(!result.spheres.empty());
  for (const auto& sphere : result.spheres) {
    check(sphere.center.at(0) < 1.0F);
  }
}

void test_open_mesh_rejected() {
  auto mesh = unit_cube();
  mesh.triangles.pop_back();
  bool threw = false;
  try {
    static_cast<void>(fit_spheres_to_mesh(mesh));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw);
}

void test_link_attachment_transform_and_reset() {
  const std::vector<float> reference{
      0.0F, 0.0F, 0.0F, 0.1F,
      0.0F, 0.0F, 0.0F, -100.0F,
      0.0F, 0.0F, 0.0F, -100.0F,
  };
  const std::vector<int16_t> sphere_map{0, 1, 1};
  std::vector<uint8_t> reference_bytes(reference.size() * sizeof(float));
  std::vector<uint8_t> map_bytes(sphere_map.size() * sizeof(int16_t));
  std::memcpy(reference_bytes.data(), reference.data(), reference_bytes.size());
  std::memcpy(map_bytes.data(), sphere_map.data(), map_bytes.size());
  curobo_robot_runtime::DeviceBuffer device_spheres;
  device_spheres.upload(reference_bytes);
  LinkSphereAttachment updater(0, {"base", "attached_object"},
                               reference_bytes, map_bytes, &device_spheres);
  check(updater.capacity("attached_object") == 2);

  SphereAttachment attachment;
  attachment.object_in_link.position = {1.0F, 2.0F, 3.0F};
  constexpr float HALF_SQRT_TWO = 0.7071067811865475F;
  attachment.object_in_link.quaternion =
      {HALF_SQRT_TWO, 0.0F, 0.0F, HALF_SQRT_TWO};
  attachment.spheres = {{{1.0F, 0.0F, 0.0F}, 0.2F}};
  updater.attach(attachment);

  std::vector<float> attached(reference.size());
  check(cudaMemcpy(attached.data(), reinterpret_cast<void*>(device_spheres.ptr()),
                   attached.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  check(std::abs(attached.at(4) - 1.0F) < 1.0E-5F);
  check(std::abs(attached.at(5) - 3.0F) < 1.0E-5F);
  check(std::abs(attached.at(6) - 3.0F) < 1.0E-5F);
  check(std::abs(attached.at(7) - 0.2F) < 1.0E-5F);
  check(attached.at(11) == -100.0F);

  updater.detach("attached_object");
  std::vector<float> detached(reference.size());
  check(cudaMemcpy(detached.data(), reinterpret_cast<void*>(device_spheres.ptr()),
                   detached.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  check(detached == reference);
}

}  // namespace

int main() {
  test_cube_fit();
  test_pose_is_not_applied();
  test_open_mesh_rejected();
  test_link_attachment_transform_and_reset();
  return EXIT_SUCCESS;
}
