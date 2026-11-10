// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/pose_calibration.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace curobo::robot;

int main(int argc, char** argv) {
  int devices = 0;
  if (argc != 2 || argv[1][0] == '\0' || cudaGetDeviceCount(&devices) != cudaSuccess ||
      devices == 0)
    return 77;
  CalibrationMesh mesh;
  // Three unequal boxes form an asymmetric articulated-looking target.
  for (int box = 0; box < 3; ++box) {
    int offset = static_cast<int>(mesh.vertices.size());
    for (int i = 0; i < 8; ++i)
      mesh.vertices.push_back({.21F * box + (i & 1 ? .12F : 0.F), .06F * box + (i & 2 ? .09F : 0.F),
                               .17F * box + (i & 4 ? .19F + .03F * box : 0.F)});
    for (auto face : {std::array<int, 3>{0, 2, 3},
                      {0, 3, 1},
                      {4, 5, 7},
                      {4, 7, 6},
                      {0, 1, 5},
                      {0, 5, 4},
                      {2, 6, 7},
                      {2, 7, 3},
                      {0, 4, 6},
                      {0, 6, 2},
                      {1, 3, 7},
                      {1, 7, 5}}) {
      for (auto& index : face) index += offset;
      mesh.triangles.push_back(face);
    }
  }
  std::vector<std::array<float, 3>> observed;
  const float angle = .23F, c = std::cos(angle), s = std::sin(angle);
  for (const auto& face : mesh.triangles) {
    auto a = mesh.vertices[face[0]], b = mesh.vertices[face[1]], d = mesh.vertices[face[2]];
    for (int i = 1; i < 18; ++i)
      for (int j = 1; j < 18 - i; ++j) {
        float u = i / 18.F, v = j / 18.F;
        std::array<float, 3> p{};
        for (int k = 0; k < 3; ++k) p[k] = a[k] + u * (b[k] - a[k]) + v * (d[k] - a[k]);
        observed.push_back({c * p[0] - s * p[1] + .3F, s * p[0] + c * p[1] - .2F, p[2] + 1.2F});
      }
  }
  PoseCalibrationConfig config;
  PoseCalibration detector(argv[1], config);
  std::array<float, 16> seed{1, 0, 0, .31F, 0, 1, 0, -.21F, 0, 0, 1, 1.21F, 0, 0, 0, 1};
  for (bool global : {false, true}) {
    auto result = detector.detect(mesh, observed, global ? std::nullopt : std::optional(seed));
    std::cout << (global ? "global" : "seeded") << " success=" << result.success
              << " icp=" << result.icp_mean_distance << " surface=" << result.surface_rms
              << " ratio=" << result.valid_ratio << " error=" << result.error << '\n';
    if (!result.success || result.surface_rms > .003F)
      throw std::runtime_error("registration failed");
    auto t = result.camera_from_base;
    if (std::abs(t[3] - .3F) > .005F || std::abs(t[7] + .2F) > .005F ||
        std::abs(t[11] - 1.2F) > .005F || std::abs(t[0] - c) > .015F || std::abs(t[4] - s) > .015F)
      throw std::runtime_error("wrong transform direction or inaccurate recovery");
  }
  auto bad = seed;
  bad[0] = 2;
  bool rejected = false;
  try {
    detector.detect(mesh, observed, bad);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("non-rigid seed accepted");
  auto far = seed;
  far[3] = 10;
  if (detector.detect(mesh, observed, far).success)
    throw std::runtime_error("no-overlap seed accepted");
  std::cout << "native calibration tests passed\n";
}
