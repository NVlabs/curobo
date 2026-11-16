// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "attachment_runtime.h"
#include "cuda_check.h"
#include "curobo_robot_runtime/kinematics.h"
#include "grasp_collision_runtime.h"
#include "ik_runtime.h"
#include "rrt_runtime.h"
#include "trajectory_optimizer_runtime.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T>
class Buffer {
 public:
  explicit Buffer(size_t count) { storage_.allocate(count * sizeof(T)); }
  T* data() const { return reinterpret_cast<T*>(storage_.ptr()); }
  T value() const {
    T result{};
    CRR_CUDA_CHECK(cudaMemcpy(&result, data(), sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }

 private:
  curobo_robot_runtime::DeviceBuffer storage_;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: collision_layers_test ARTIFACT_DIRECTORY");
    const std::string artifact = argv[1];
    const auto manifest = curobo_robot_runtime::load_ik_manifest(artifact);
    const int dof = manifest.num_dof;
    curobo::robot::Scene scene;
    curobo::robot::Cuboid box;
    box.name = "world";
    box.dimensions = {20.0F, 20.0F, 20.0F};
    const auto obstacle = scene.add_cuboid(box);
    scene.set_obstacle_layers(obstacle, 0x80000004U);
    curobo::robot::InverseKinematicsConfig config;
    config.check_self_collision = false;
    config.exit_early = false;
    config.lm_iterations = 1;
    config.lbfgs_iterations = 1;
    curobo::robot::detail::IkRuntime ik(artifact, config, &scene);
    curobo::robot::detail::IkRuntime goalset(ik, config);
    auto model = ik.collision_model();
    require(goalset.collision_model() == model, "goalset must share model ownership");
    const auto sphere_pointer = model->link_spheres.ptr();
    const auto mask_pointer = model->sphere_collision_mask.ptr();

    curobo::robot::RrtConfig rrt_config;
    curobo::robot::detail::RrtRuntime rrt(artifact, rrt_config, &scene, model);
    curobo::robot::TrajectoryOptimizationConfig trajopt_config;
    curobo::robot::detail::TrajectoryOptimizerRuntime trajopt(artifact, trajopt_config, &scene,
                                                              model);
    // Updates from every solver must reach the same pre-FK buffer.
    for (auto update : {0x80000000U, 4U, 1U}) {
      if (update == 0x80000000U) {
        rrt.set_link_collision_mask("attached_object", update);
      } else if (update == 4U) {
        trajopt.set_link_collision_mask("attached_object", update);
      } else {
        goalset.set_link_collision_mask("attached_object", update);
      }
      const auto map_bytes = curobo_robot_runtime::load_tensor_data(
          artifact, manifest.link_sphere_map, "link_sphere_map");
      std::vector<int16_t> links(manifest.num_spheres);
      std::memcpy(links.data(), map_bytes.data(), map_bytes.size());
      const auto robot_manifest = curobo_robot_runtime::load_manifest(artifact);
      std::vector<uint32_t> masks(manifest.num_spheres);
      CRR_CUDA_CHECK(cudaMemcpy(masks.data(), reinterpret_cast<void*>(mask_pointer),
                                masks.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      for (int i = 0; i < manifest.num_spheres; ++i) {
        if (robot_manifest.link_names.at(links[i]) == "attached_object") {
          require(masks[i] == update, "solver update did not reach shared IK masks");
        }
      }
    }
    curobo::robot::SphereAttachment attachment;
    attachment.spheres = {{{0.0F, 0.0F, 0.0F}, 0.02F}};
    trajopt.attach_object(attachment);
    require(model->link_spheres.ptr() == sphere_pointer, "attachment replaced sphere storage");
    std::vector<float> attached(manifest.num_spheres * 4);
    CRR_CUDA_CHECK(cudaMemcpy(attached.data(), reinterpret_cast<void*>(sphere_pointer),
                              attached.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto original =
        curobo_robot_runtime::load_tensor_data(artifact, manifest.link_spheres, "link_spheres");
    require(std::memcmp(attached.data(), original.data(), original.size()) != 0,
            "TrajOpt attachment did not reach shared IK geometry");
    rrt.detach_object("attached_object");
    CRR_CUDA_CHECK(cudaMemcpy(attached.data(), reinterpret_cast<void*>(sphere_pointer),
                              attached.size() * sizeof(float), cudaMemcpyDeviceToHost));
    require(std::memcmp(attached.data(), original.data(), original.size()) == 0,
            "RRT detachment did not restore shared IK geometry");

    curobo_robot_runtime::DeviceBuffer current;
    current.upload(curobo_robot_runtime::load_tensor_data(artifact, manifest.default_joint_position,
                                                          "default_joint_position"));
    Buffer<float> positions(3), quaternions(4), jacobian(6 * dof), solution(dof);
    Buffer<float> position_error(1), orientation_error(1), collision_cost(1);
    Buffer<uint8_t> success(1);
    curobo::robot::Kinematics fk(artifact);
    fk.forward(reinterpret_cast<float*>(current.ptr()), positions.data(), quaternions.data(),
               jacobian.data(), 1, 1);
    curobo::robot::IkSolveRequest request;
    request.current_positions = reinterpret_cast<float*>(current.ptr());
    request.goal_positions = positions.data();
    request.goal_quaternions = quaternions.data();
    request.solution = solution.data();
    request.position_error = position_error.data();
    request.orientation_error = orientation_error.data();
    request.collision_cost = collision_cost.data();
    request.success = success.data();
    const auto cost = [&](const auto& solver) {
      solver.solve(request);
      CRR_CUDA_CHECK(cudaDeviceSynchronize());
      return collision_cost.value();
    };
    require(cost(ik) == 0.0F, "common masks must ignore the alternate layer");
    require(cost(goalset) == 0.0F, "shared IK must agree on initial masks");
    // Replay the same graphs after changing only GPU metadata values.
    CRR_CU_CHECK(cuMemsetD32(mask_pointer, 0x80000000U, manifest.num_spheres));
    const float hit = cost(ik);
    require(hit > 1.0F && cost(goalset) > 1.0F, "captured IK did not read new masks");
    CRR_CU_CHECK(cuMemsetD32(mask_pointer, 0x80000004U, manifest.num_spheres));
    require(cost(ik) == hit, "multiple matching bits counted an obstacle twice");
    scene.set_obstacle_layers(obstacle, 2U);
    require(cost(ik) == 0.0F, "captured graph did not see updated obstacle membership");
    require(model->sphere_collision_mask.ptr() == mask_pointer, "mask storage changed");
    if (ik.attachment_sphere_capacity("attached_object") == 32) {
      using namespace curobo::robot::detail;
      const auto robot_manifest = curobo_robot_runtime::load_manifest(artifact);
      const auto map_bytes = curobo_robot_runtime::load_tensor_data(
          artifact, manifest.link_sphere_map, "link_sphere_map");
      std::vector<int16_t> sphere_links(manifest.num_spheres);
      std::memcpy(sphere_links.data(), map_bytes.data(), map_bytes.size());
      curobo::robot::GraspCollisionPolicy policy;
      policy.enabled = true;
      policy.target_points = positions.data();
      policy.target_point_count = 1;
      policy.target_voxel_radius = .002F;
      for (int link = 0; link < robot_manifest.num_links; ++link) {
        const auto& name = robot_manifest.link_names.at(link);
        if (name != "attached_object" &&
            std::find(sphere_links.begin(), sphere_links.end(), link) != sphere_links.end())
          policy.contact_links.push_back(name);
      }
      GraspCollisionWorkspace workspace(*model);
      {
        GraspCollisionScope scope(workspace, policy, nullptr);
        scope.set_phase(GraspCollisionPhase::Grasp);
        require(cost(ik) == 0.F, "contact spheres must ignore target-only geometry");
        scope.set_phase(GraspCollisionPhase::Approach);
        require(cost(ik) > 1.F, "transit must not ignore target geometry");
        scope.set_phase(GraspCollisionPhase::Grasp);
        scene.set_obstacle_layers(obstacle, 1U);
        require(cost(ik) > 1.F, "contact must retain COMMON obstacle checks");
        scene.set_obstacle_layers(obstacle, 4U);
        require(cost(ik) > 1.F, "contact must retain table and neighbor checks");
        scope.activate_lift(positions.data(), quaternions.data());
        require(cost(ik) > 1.F, "lift must check without-target environment");
      }
      scene.set_obstacle_layers(obstacle, 2U);
      policy.contact_links.resize(1);
      {
        GraspCollisionScope scope(workspace, policy, nullptr);
        scope.set_phase(GraspCollisionPhase::Grasp);
        require(cost(ik) > 1.F, "non-contact arm and housing must still check target");
      }
      require(cost(ik) == 0.F, "phase routing leaked after scope exit");
    }
    std::cout << "collision layers and shared model replay passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
