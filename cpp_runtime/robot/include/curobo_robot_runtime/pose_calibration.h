// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace curobo::robot {

/** Coarse-to-fine point-to-plane ICP and unsigned mesh-distance LM policy. */
struct PoseCalibrationConfig {
  int rotation_samples{32};
  int coarse_mesh_points{500};
  int fine_mesh_points{2000};
  int coarse_observed_points{2000};
  int fine_observed_points{10000};
  int coarse_iterations{50};
  int fine_iterations{50};
  float coarse_distance{0.5F};
  float fine_distance{0.01F};
  float icp_huber_delta{0.02F};
  bool refine_surface{true};
  int surface_points{5000};
  int surface_iterations{10};
  float surface_distance{0.2F};
  float surface_huber_delta{0.1F};
  float minimum_valid_ratio{0.1F};
  unsigned int random_seed{0};
};

/** Posed visual triangles in robot-base coordinates, at measured joints. */
struct CalibrationMesh {
  std::vector<std::array<float, 3>> vertices;
  std::vector<std::array<int, 3>> triangles;
};

/** A candidate only: this API neither moves a robot nor applies calibration. */
struct PoseCalibrationResult {
  /** Row-major T_camera_base: p_camera = R * p_base + t. */
  std::array<float, 16> camera_from_base{};
  bool success{false};
  std::string error;
  float icp_mean_distance{0};
  /** Robust RMS of unsigned mesh-distance residuals, after Huber weighting. */
  float surface_rms{0};
  float valid_ratio{0};
  int fine_iterations{0};
  int surface_iterations{0};
  int best_hypothesis{-1};
};

/** GPU registration using target-exported mesh query kernels, without Python.
 *
 * Each call owns its input copies and synchronizes its private stream before
 * returning. Calls on one instance must be serialized. Invalid input throws;
 * insufficient registration overlap returns success=false. Geometry is supplied
 * at measured joints so no robot YAML, mesh loader, or Python FK is needed here.
 */
class PoseCalibration final {
 public:
  explicit PoseCalibration(const std::string &kernel_directory, PoseCalibrationConfig config = {});
  ~PoseCalibration();
  PoseCalibration(const PoseCalibration &) = delete;
  PoseCalibration &operator=(const PoseCalibration &) = delete;

  /** Register camera-frame observed points; omit the seed for global search. */
  PoseCalibrationResult detect(
      const CalibrationMesh &mesh, const std::vector<std::array<float, 3>> &observed_points,
      const std::optional<std::array<float, 16>> &camera_from_base = std::nullopt);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace curobo::robot
