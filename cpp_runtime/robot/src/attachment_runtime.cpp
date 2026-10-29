// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "attachment_runtime.h"

#include "cuda_check.h"
#include "motion_planner_cuda.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace curobo::robot::detail {
namespace {

std::array<float, 3> transform_point(const Pose& pose,
                                     const std::array<float, 3>& point) {
  for (const auto value : pose.position) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("attachment position must be finite");
    }
  }
  float quaternion_norm_squared = 0.0F;
  for (const auto value : pose.quaternion) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("attachment quaternion must be finite");
    }
    quaternion_norm_squared += value * value;
  }
  if (!(quaternion_norm_squared > 0.0F)) {
    throw std::invalid_argument("attachment quaternion must have nonzero norm");
  }
  const float inverse_norm = 1.0F / std::sqrt(quaternion_norm_squared);
  const float w = pose.quaternion.at(0) * inverse_norm;
  const std::array<float, 3> axis{
      pose.quaternion.at(1) * inverse_norm,
      pose.quaternion.at(2) * inverse_norm,
      pose.quaternion.at(3) * inverse_norm,
  };
  const std::array<float, 3> twice_cross{
      2.0F * (axis.at(1) * point.at(2) - axis.at(2) * point.at(1)),
      2.0F * (axis.at(2) * point.at(0) - axis.at(0) * point.at(2)),
      2.0F * (axis.at(0) * point.at(1) - axis.at(1) * point.at(0)),
  };
  const std::array<float, 3> second_cross{
      axis.at(1) * twice_cross.at(2) - axis.at(2) * twice_cross.at(1),
      axis.at(2) * twice_cross.at(0) - axis.at(0) * twice_cross.at(2),
      axis.at(0) * twice_cross.at(1) - axis.at(1) * twice_cross.at(0),
  };
  return {
      pose.position.at(0) + point.at(0) + w * twice_cross.at(0) +
          second_cross.at(0),
      pose.position.at(1) + point.at(1) + w * twice_cross.at(1) +
          second_cross.at(1),
      pose.position.at(2) + point.at(2) + w * twice_cross.at(2) +
          second_cross.at(2),
  };
}

}  // namespace

LinkSphereAttachment::LinkSphereAttachment(
    int device_ordinal, std::vector<std::string> link_names,
    const std::vector<uint8_t>& reference_spheres,
    const std::vector<uint8_t>& link_sphere_map,
    curobo_robot_runtime::DeviceBuffer* device_spheres)
    : device_ordinal_(device_ordinal), device_spheres_(device_spheres) {
  if (device_spheres_ == nullptr ||
      reference_spheres.size() % (4 * sizeof(float)) != 0) {
    throw std::invalid_argument("invalid attachment sphere storage");
  }
  const size_t sphere_count = reference_spheres.size() / (4 * sizeof(float));
  if (link_sphere_map.size() != sphere_count * sizeof(int16_t)) {
    throw std::invalid_argument("attachment sphere map size does not match spheres");
  }
  reference_spheres_.resize(sphere_count * 4);
  std::memcpy(reference_spheres_.data(), reference_spheres.data(),
              reference_spheres.size());
  current_spheres_ = reference_spheres_;

  std::vector<int16_t> sphere_links(sphere_count);
  if (!link_sphere_map.empty()) {
    std::memcpy(sphere_links.data(), link_sphere_map.data(),
                link_sphere_map.size());
  }
  for (size_t sphere_index = 0; sphere_index < sphere_count; ++sphere_index) {
    const int link_index = sphere_links.at(sphere_index);
    if (link_index < 0 ||
        static_cast<size_t>(link_index) >= link_names.size()) {
      if (link_names.empty()) {
        continue;
      }
      throw std::invalid_argument("attachment sphere map has an invalid link index");
    }
    const auto& link_name = link_names.at(static_cast<size_t>(link_index));
    if (link_name.empty()) {
      throw std::invalid_argument("attachment metadata contains an empty link name");
    }
    link_slots_[link_name].push_back(sphere_index);
  }
  std::vector<int32_t> packed_slots;
  for (const auto& entry : link_slots_) {
    link_slot_offsets_[entry.first] = packed_slots.size();
    for (const auto slot : entry.second) {
      packed_slots.push_back(static_cast<int32_t>(slot));
    }
  }
  CRR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  nominal_spheres_.upload(reference_spheres);
  std::vector<uint8_t> slot_bytes(packed_slots.size() * sizeof(int32_t));
  if (!slot_bytes.empty()) {
    std::memcpy(slot_bytes.data(), packed_slots.data(), slot_bytes.size());
    device_slots_.upload(slot_bytes);
  }
}

const std::vector<size_t>& LinkSphereAttachment::slots(
    const std::string& link_name) const {
  if (link_slots_.empty()) {
    throw std::runtime_error(
        "robot artifact has no link attachment metadata; re-export it with the current exporter");
  }
  const auto found = link_slots_.find(link_name);
  if (found == link_slots_.end() || found->second.empty()) {
    throw std::invalid_argument("robot link has no collision-sphere slots: " +
                                link_name);
  }
  return found->second;
}

int LinkSphereAttachment::capacity(const std::string& link_name) const {
  return static_cast<int>(slots(link_name).size());
}

const int32_t* LinkSphereAttachment::device_slots(const std::string& link_name) const {
  slots(link_name);  // Validate before looking up the packed GPU range.
  return reinterpret_cast<const int32_t*>(device_slots_.ptr()) + link_slot_offsets_.at(link_name);
}

void LinkSphereAttachment::attach(const SphereAttachment& attachment) {
  const auto& target_slots = slots(attachment.link_name);
  if (attachment.spheres.empty()) {
    throw std::invalid_argument("an attachment must contain at least one sphere");
  }
  if (attachment.spheres.size() > target_slots.size()) {
    throw std::invalid_argument(
        "attachment sphere count exceeds the link's artifact capacity");
  }

  for (const auto sphere_index : target_slots) {
    const size_t offset = sphere_index * 4;
    current_spheres_.at(offset) = 0.0F;
    current_spheres_.at(offset + 1) = 0.0F;
    current_spheres_.at(offset + 2) = 0.0F;
    current_spheres_.at(offset + 3) = -100.0F;
  }
  for (size_t index = 0; index < attachment.spheres.size(); ++index) {
    const auto& sphere = attachment.spheres.at(index);
    for (const auto coordinate : sphere.center) {
      if (!std::isfinite(coordinate)) {
        throw std::invalid_argument("attachment sphere centers must be finite");
      }
    }
    if (!std::isfinite(sphere.radius) || sphere.radius <= 0.0F) {
      throw std::invalid_argument("attachment sphere radii must be finite and positive");
    }
    const auto center = transform_point(attachment.object_in_link, sphere.center);
    const size_t offset = target_slots.at(index) * 4;
    current_spheres_.at(offset) = center.at(0);
    current_spheres_.at(offset + 1) = center.at(1);
    current_spheres_.at(offset + 2) = center.at(2);
    current_spheres_.at(offset + 3) = sphere.radius;
  }
  upload();
}

void LinkSphereAttachment::detach(const std::string& link_name) {
  for (const auto sphere_index : slots(link_name)) {
    const size_t offset = sphere_index * 4;
    for (size_t component = 0; component < 4; ++component) {
      current_spheres_.at(offset + component) =
          reference_spheres_.at(offset + component);
    }
  }
  upload();
}

void LinkSphereAttachment::set_collision_enabled(const std::string& link_name,
                                                 bool enabled) {
  const auto& target_slots = slots(link_name);
  if (enabled) {
    if (disabled_links_.erase(link_name) == 0) {
      return;
    }
  } else {
    if (!disabled_links_.insert(link_name).second) {
      return;
    }
  }
  CRR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  CRR_CUDA_CHECK(launch_set_sphere_collision_enabled(
      reinterpret_cast<float*>(device_spheres_->ptr()),
      reinterpret_cast<const float*>(nominal_spheres_.ptr()),
      reinterpret_cast<const int32_t*>(device_slots_.ptr()) +
          link_slot_offsets_.at(link_name),
      static_cast<int>(target_slots.size()), enabled, nullptr));
  // Public attachment updates are synchronous, including for nonblocking users.
  CRR_CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void LinkSphereAttachment::upload() const {
  const size_t bytes = current_spheres_.size() * sizeof(float);
  if (device_spheres_->bytes() != bytes) {
    throw std::runtime_error("attachment device sphere storage changed size");
  }
  CRR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  if (bytes != 0) {
    CRR_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(nominal_spheres_.ptr()),
                              current_spheres_.data(), bytes,
                              cudaMemcpyHostToDevice));
    auto upload_spheres = current_spheres_;
    for (const auto& link_name : disabled_links_) {
      for (const auto sphere_index : slots(link_name)) {
        upload_spheres.at(sphere_index * 4 + 3) = -100.0F;
      }
    }
    CRR_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(device_spheres_->ptr()),
                              upload_spheres.data(), bytes,
                              cudaMemcpyHostToDevice));
  }
}

void LinkSphereAttachment::set_collision_mask(const std::string& link_name, uint32_t layers,
                                              curobo_robot_runtime::DeviceBuffer& masks) const {
  const auto& target_slots = slots(link_name);
  CRR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  CRR_CUDA_CHECK(launch_set_sphere_collision_mask(
      reinterpret_cast<uint32_t*>(masks.ptr()),
      reinterpret_cast<const int32_t*>(device_slots_.ptr()) + link_slot_offsets_.at(link_name),
      static_cast<int>(target_slots.size()), layers, nullptr));
  CRR_CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

RobotCollisionModel::RobotCollisionModel(int device, const std::vector<std::string>& link_names,
                                         const std::vector<uint8_t>& spheres,
                                         const std::vector<uint8_t>& sphere_map,
                                         const std::vector<uint8_t>& masks)
    : device_ordinal(device) {
  if (spheres.size() % (4 * sizeof(float)) != 0 || masks.size() != spheres.size() / 4) {
    throw std::invalid_argument("collision masks must have one uint32 per sphere");
  }
  CRR_CUDA_CHECK(cudaSetDevice(device));
  link_spheres.upload(spheres);
  link_sphere_map.upload(sphere_map);
  sphere_collision_mask.upload(masks);
  attachment = std::make_unique<LinkSphereAttachment>(device, link_names, spheres, sphere_map,
                                                      &link_spheres);
}

}  // namespace curobo::robot::detail
