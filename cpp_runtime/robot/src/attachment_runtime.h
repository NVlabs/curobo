// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "curobo_robot_runtime/attachment.h"
#include "robot_runtime.h"

namespace curobo::robot::detail {

/** Maintains the shared robot model's stable buffer of link-local collision
 * spheres. */
class LinkSphereAttachment final {
 public:
  LinkSphereAttachment(
      int device_ordinal, std::vector<std::string> link_names,
      const std::vector<uint8_t>& reference_spheres,
      const std::vector<uint8_t>& link_sphere_map,
      curobo_robot_runtime::DeviceBuffer* device_spheres);

  int capacity(const std::string& link_name) const;
  const int32_t* device_slots(const std::string& link_name) const;
  void attach(const SphereAttachment& attachment);
  void detach(const std::string& link_name);
  void set_collision_enabled(const std::string& link_name, bool enabled);
  void set_collision_mask(const std::string& link_name, uint32_t layers,
                          curobo_robot_runtime::DeviceBuffer& masks) const;

 private:
  const std::vector<size_t>& slots(const std::string& link_name) const;
  void upload() const;

  int device_ordinal_{0};
  curobo_robot_runtime::DeviceBuffer* device_spheres_{nullptr};
  std::vector<float> reference_spheres_;
  std::vector<float> current_spheres_;
  std::unordered_map<std::string, std::vector<size_t>> link_slots_;
  std::unordered_map<std::string, size_t> link_slot_offsets_;
  curobo_robot_runtime::DeviceBuffer device_slots_;
  curobo_robot_runtime::DeviceBuffer nominal_spheres_;
  std::unordered_set<std::string> disabled_links_;
};

/** Shared pre-FK geometry and layer policy; solver FK outputs stay private. */
struct RobotCollisionModel final {
  RobotCollisionModel(int device_ordinal, const std::vector<std::string>& link_names,
                      const std::vector<uint8_t>& spheres, const std::vector<uint8_t>& sphere_map,
                      const std::vector<uint8_t>& masks);

  const int device_ordinal;
  curobo_robot_runtime::DeviceBuffer link_spheres;
  curobo_robot_runtime::DeviceBuffer link_sphere_map;
  curobo_robot_runtime::DeviceBuffer sphere_collision_mask;
  std::unique_ptr<LinkSphereAttachment> attachment;
};

}  // namespace curobo::robot::detail
