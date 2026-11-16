// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda_check.h"
#include "grasp_collision_runtime.h"

namespace {
template <typename T>
std::vector<uint8_t> bytes(const std::vector<T>& values) {
  std::vector<uint8_t> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}
template <typename T>
std::vector<T> read(const curobo_robot_runtime::DeviceBuffer& buffer) {
  std::vector<T> result(buffer.bytes() / sizeof(T));
  CRR_CUDA_CHECK(cudaMemcpy(result.data(), reinterpret_cast<void*>(buffer.ptr()), buffer.bytes(),
                            cudaMemcpyDeviceToHost));
  return result;
}
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    using namespace curobo::robot;
    using namespace curobo::robot::detail;
    std::vector<float> spheres(34 * 4, 0.0F);
    std::vector<int16_t> links(34, 2);
    links[0] = 0;
    links[1] = 1;
    for (int i = 0; i < 34; ++i) spheres[i * 4 + 3] = i < 2 ? .02F : -100.0F;
    spheres[8] = .1F;
    spheres[11] = .015F;  // Even an existing observed attachment must be restored.
    const std::vector<uint32_t> masks(34, 0x80000001U);
    RobotCollisionModel model(0, {"housing", "finger", "attached_object"}, bytes(spheres),
                              bytes(links), bytes(masks));
    GraspCollisionWorkspace workspace(model);
    GraspCollisionPolicy policy;
    policy.enabled = true;
    policy.contact_links = {"finger"};
    curobo_robot_runtime::DeviceBuffer points, position, quaternion;
    points.upload(bytes<float>({1.F, 0.F, 0.F, 1.04F, .04F, .04F}));
    position.upload(bytes<float>({1.F, 0.F, 0.F}));
    quaternion.upload(bytes<float>({1.F, 0.F, 0.F, 1.F}));  // Non-unit, 90 degrees about Z.
    policy.target_points = reinterpret_cast<const float*>(points.ptr());
    policy.target_point_count = 2;
    const auto sphere_ptr = model.link_spheres.ptr();
    const auto mask_ptr = model.sphere_collision_mask.ptr();
    for (bool throw_during_lift : {false, true}) {
      try {
        GraspCollisionScope scope(workspace, policy, nullptr);
        scope.set_phase(GraspCollisionPhase::Grasp);
        auto actual = read<uint32_t>(model.sphere_collision_mask);
        require(actual[0] == 3U && actual[1] == 5U, "grasp routing");
        require(read<float>(model.link_spheres)[11] < 0, "inactive provisional attachment");
        scope.set_phase(GraspCollisionPhase::Approach);
        actual = read<uint32_t>(model.sphere_collision_mask);
        require(actual[0] == 3U && actual[1] == 3U, "transit must use full field");
        scope.set_phase(GraspCollisionPhase::Lift);
        for (auto mask : read<uint32_t>(model.sphere_collision_mask))
          require(mask == 5U, "unattached lift must use target-subtracted field");
        for (int i = 2; i < 34; ++i)
          require(read<float>(model.link_spheres)[i * 4 + 3] < 0,
                  "lift phase alone must not activate payload spheres");
        scope.activate_lift(reinterpret_cast<const float*>(position.ptr()),
                            reinterpret_cast<const float*>(quaternion.ptr()));
        actual = read<uint32_t>(model.sphere_collision_mask);
        for (auto mask : actual) require(mask == 5U, "lift routing");
        const auto fitted = read<float>(model.link_spheres);
        require(fitted[11] > 0 && fitted[8] < .04F && fitted[9] < 0,
                "fit must use selected tool frame");
        for (const auto point :
             {std::vector<float>{0.F, 0.F, 0.F}, std::vector<float>{.04F, -.04F, .04F}}) {
          bool covered = false;
          for (int i = 2; i < 34; ++i) {
            float squared = 0;
            for (int axis = 0; axis < 3; ++axis)
              squared += std::pow(point[axis] - fitted[i * 4 + axis], 2);
            covered |= std::sqrt(squared) <= fitted[i * 4 + 3] + 1.e-6F;
          }
          require(covered, "provisional cover omitted target geometry");
        }
        if (throw_during_lift) throw std::runtime_error("simulate failed planning");
      } catch (const std::runtime_error& error) {
        if (!throw_during_lift || std::string(error.what()) != "simulate failed planning") throw;
      }
      require(read<float>(model.link_spheres) == spheres, "restore exact geometry");
      require(read<uint32_t>(model.sphere_collision_mask) == masks, "restore exact policy");
      require(
          model.link_spheres.ptr() == sphere_ptr && model.sphere_collision_mask.ptr() == mask_ptr,
          "stable graph addresses");
    }
    // Object-only finger field, without provisional payload or target subtraction
    // for the housing. Geometry (including self collision) stays enabled.
    auto finger_policy = policy;
    finger_policy.contact_layer = 4U;
    finger_policy.target_points = nullptr;
    finger_policy.target_point_count = 0;
    {
      GraspCollisionScope scope(workspace, finger_policy, nullptr);
      for (auto phase : {GraspCollisionPhase::Grasp, GraspCollisionPhase::Lift}) {
        scope.set_phase(phase);
        const auto actual = read<uint32_t>(model.sphere_collision_mask);
        require(actual[0] == 3U && actual[1] == 4U, "objects-only finger routing");
        require(read<float>(model.link_spheres) == spheres, "finger mode preserves geometry");
      }
      scope.set_phase(GraspCollisionPhase::Approach);
      for (auto mask : read<uint32_t>(model.sphere_collision_mask))
        require(mask == 3U, "finger mode transit checks full scene");
    }
    require(read<uint32_t>(model.sphere_collision_mask) == masks, "finger mode restores masks");
    require(read<float>(model.link_spheres) == spheres, "finger mode restores geometry");
    // The additional field also overrides fingers in the target-subtraction path.
    finger_policy.target_points = policy.target_points;
    finger_policy.contact_layer = 8U;
    finger_policy.target_point_count = policy.target_point_count;
    {
      GraspCollisionScope scope(workspace, finger_policy, nullptr);
      scope.set_phase(GraspCollisionPhase::Lift);
      const auto actual = read<uint32_t>(model.sphere_collision_mask);
      require(actual[0] == 5U && actual[1] == 8U, "subtracted lift with objects-only fingers");
    }
    points.upload(bytes<float>({std::numeric_limits<float>::quiet_NaN(), 0.F, 0.F, 1.F, 0.F, 0.F}));
    policy.target_points = reinterpret_cast<const float*>(points.ptr());
    bool invalid_rejected = false;
    try {
      GraspCollisionScope scope(workspace, policy, nullptr);
      scope.activate_lift(reinterpret_cast<const float*>(position.ptr()),
                          reinterpret_cast<const float*>(quaternion.ptr()));
    } catch (const std::invalid_argument&) {
      invalid_rejected = true;
    }
    require(invalid_rejected, "non-finite points must fail planning");
    require(read<float>(model.link_spheres) == spheres, "failed fit must restore geometry");
    require(read<uint32_t>(model.sphere_collision_mask) == masks, "failed fit must restore masks");
    std::cout << "grasp collision policy and restoration passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
