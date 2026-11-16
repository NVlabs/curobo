// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace curobo::robot {

/** Optional inertial-property replacements for one exported robot link. */
struct LinkInertialUpdate {
  std::string link_name;
  std::optional<float> mass{};
  std::optional<std::array<float, 3>> center_of_mass{};
  /** Symmetric inertia as [ixx, iyy, izz, ixy, ixz, iyz]. */
  std::optional<std::array<float, 6>> inertia{};
};

using LinkInertialUpdates = std::vector<LinkInertialUpdate>;

}  // namespace curobo::robot
