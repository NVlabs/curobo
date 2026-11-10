// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "grasp_collision_runtime.h"

#include <cmath>
#include <stdexcept>

#include "cuda_check.h"
#include "motion_planner_cuda.h"

namespace curobo::robot::detail {
namespace {
void copy(curobo_robot_runtime::DeviceBuffer& out, const curobo_robot_runtime::DeviceBuffer& in,
          cudaStream_t stream) {
  CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(out.ptr()),
                                 reinterpret_cast<const void*>(in.ptr()), in.bytes(),
                                 cudaMemcpyDeviceToDevice, stream));
}
bool single_bit(uint32_t bits) { return bits && !(bits & (bits - 1)); }
}  // namespace

GraspCollisionWorkspace::GraspCollisionWorkspace(RobotCollisionModel& collision)
    : model(collision) {
  saved_spheres.allocate(model.link_spheres.bytes());
  saved_masks.allocate(model.sphere_collision_mask.bytes());
  fit_valid.allocate(sizeof(uint8_t));
}

GraspCollisionScope::GraspCollisionScope(GraspCollisionWorkspace& workspace,
                                         const GraspCollisionPolicy& policy, cudaStream_t stream)
    : workspace_(workspace), policy_(policy), stream_(stream) {
  if (!policy.enabled) return;
  if (!single_bit(policy.full_layer) || !single_bit(policy.without_target_layer) ||
      (policy.full_layer & policy.without_target_layer) ||
      (policy.common_layers & (policy.full_layer | policy.without_target_layer)) ||
      (policy.contact_layer &&
       (!single_bit(policy.contact_layer) ||
        (policy.contact_layer & (policy.common_layers | policy.full_layer |
                                 (policy.target_points ? policy.without_target_layer : 0U))))) ||
      (!policy.target_points && (!policy.contact_layer || policy.target_point_count != 0)) ||
      (policy.target_points && policy.target_point_count <= 0) ||
      !std::isfinite(policy.target_voxel_radius) || policy.target_voxel_radius < 0 ||
      policy.contact_links.empty()) {
    throw std::invalid_argument("invalid layered grasp collision policy or target points");
  }
  auto& model = workspace_.model;
  if (policy.target_points && model.attachment->capacity(policy.attachment_link) != 32) {
    throw std::invalid_argument("provisional grasp attachment requires 32 reserved spheres");
  }
  for (const auto& link : policy.contact_links) {
    model.attachment->capacity(link);
    if (link == policy.attachment_link)
      throw std::invalid_argument("attachment cannot be a contact link");
  }
  copy(workspace_.saved_spheres, model.link_spheres, stream_);
  copy(workspace_.saved_masks, model.sphere_collision_mask, stream_);
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream_));
  active_ = true;
  if (!policy.target_points) return;
  try {
    CRR_CUDA_CHECK(launch_set_sphere_collision_enabled(
        reinterpret_cast<float*>(model.link_spheres.ptr()), nullptr,
        model.attachment->device_slots(policy.attachment_link), 32, false, stream_));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream_));
  } catch (...) {
    restore();
    throw;
  }
}

GraspCollisionScope::~GraspCollisionScope() noexcept {
  // Explicit restore reports CUDA errors on normal exits; unwinding must not
  // throw a second exception. A device failure still fails the planning call.
  try {
    restore();
  } catch (...) {
  }
}

void GraspCollisionScope::restore() {
  if (!active_) return;
  copy(workspace_.model.link_spheres, workspace_.saved_spheres, stream_);
  copy(workspace_.model.sphere_collision_mask, workspace_.saved_masks, stream_);
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream_));
  active_ = false;
}

void GraspCollisionScope::set_phase(GraspCollisionPhase phase) {
  if (!active_) return;
  auto& model = workspace_.model;
  const uint32_t layers =
      policy_.common_layers |
      (phase == GraspCollisionPhase::Lift && policy_.target_points
           ? policy_.without_target_layer : policy_.full_layer);
  auto* masks = reinterpret_cast<uint32_t*>(model.sphere_collision_mask.ptr());
  CRR_CUDA_CHECK(launch_set_sphere_collision_mask(
      masks, nullptr, static_cast<int>(model.sphere_collision_mask.bytes() / sizeof(uint32_t)),
      layers, stream_));
  if (phase == GraspCollisionPhase::Grasp ||
      (phase == GraspCollisionPhase::Lift && policy_.contact_layer)) {
    for (const auto& link : policy_.contact_links) {
      CRR_CUDA_CHECK(launch_set_sphere_collision_mask(
          masks, model.attachment->device_slots(link), model.attachment->capacity(link),
          policy_.contact_layer ? policy_.contact_layer
                                : policy_.common_layers | policy_.without_target_layer, stream_));
    }
  }
  // Solvers may launch collision work on auxiliary streams.
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void GraspCollisionScope::activate_lift(const float* position, const float* quaternion) {
  if (!active_) return;
  if (!policy_.target_points) {
    set_phase(GraspCollisionPhase::Lift);
    return;
  }
  auto& model = workspace_.model;
  CRR_CUDA_CHECK(launch_fit_grasp_attachment(
      policy_.target_points, policy_.target_point_count, position, quaternion,
      policy_.target_voxel_radius, model.attachment->device_slots(policy_.attachment_link),
      reinterpret_cast<float*>(model.link_spheres.ptr()),
      reinterpret_cast<uint8_t*>(workspace_.fit_valid.ptr()), stream_));
  uint8_t valid = 0;
  CRR_CUDA_CHECK(cudaMemcpyAsync(&valid, reinterpret_cast<void*>(workspace_.fit_valid.ptr()),
                                 sizeof(valid), cudaMemcpyDeviceToHost, stream_));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (!valid) throw std::invalid_argument("non-finite provisional attachment points or tool pose");
  set_phase(GraspCollisionPhase::Lift);
}

}  // namespace curobo::robot::detail
