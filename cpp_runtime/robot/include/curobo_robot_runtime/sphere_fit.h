// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "curobo_robot_runtime/attachment.h"
#include "curobo_robot_runtime/scene.h"

namespace curobo::robot {

/** Policy for deterministic voxel-volume sphere fitting. */
struct SphereFitConfig {
  /** Maximum number of fitted spheres returned. */
  int num_spheres{8};
  /** Discard candidates whose mesh clearance is no greater than this value. */
  float minimum_radius{0.0F};
};

/** Result of fitting inscribed spheres to a closed triangle mesh. */
struct SphereFitResult {
  std::vector<CollisionSphere> spheres;
  /** Number of interior voxel centers before truncation. */
  int candidate_count{0};
  /** Isotropic pitch used to choose the bounding-box grid resolution. */
  float voxel_pitch{0.0F};
};

/**
 * Fit inscribed spheres using a deterministic bounding-box voxel grid.
 *
 * The mesh must be watertight. Vertices and returned sphere centers are in the
 * mesh-local frame; `mesh.pose` is intentionally not applied. Fitting runs on
 * the CPU because attachment is an infrequent scene mutation rather than a
 * planning-loop operation.
 */
SphereFitResult fit_spheres_to_mesh(const Mesh& mesh,
                                    const SphereFitConfig& config = {});

}  // namespace curobo::robot
