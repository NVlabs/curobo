// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <string>
#include <vector>

#include "curobo_robot_runtime/scene.h"

namespace curobo::robot {

/** One collision sphere expressed in an object's local frame. */
struct CollisionSphere {
  std::array<float, 3> center{0.0F, 0.0F, 0.0F};
  float radius{0.0F};
};

/**
 * Collision geometry to install in the preallocated sphere slots of one link.
 *
 * `object_in_link` maps object-local sphere centers into the target link frame.
 * Unused sphere slots are disabled. Robot artifacts must preallocate enough
 * slots for `link_name`, for example with `extra_collision_spheres`.
 */
struct SphereAttachment {
  std::string link_name{"attached_object"};
  Pose object_in_link{};
  std::vector<CollisionSphere> spheres;
};

}  // namespace curobo::robot
