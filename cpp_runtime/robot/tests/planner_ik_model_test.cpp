// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/scene.h"
#include "cuda_check.h"
#include "robot_runtime.h"

#include <array>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: planner_ik_model_test YAM_ARTIFACT");
    curobo::robot::Scene scene;
    curobo::robot::MotionPlanningConfig config;
    config.inverse_kinematics.check_effort_limits = true;
    config.inverse_kinematics.max_goalset = 100;
    curobo::robot::MotionPlanner planner(argv[1], config, &scene);
    auto ik = planner.create_inverse_kinematics(config.inverse_kinematics);
    std::array<curobo_robot_runtime::DeviceBuffer, 9> storage;
    for (auto& buffer : storage) buffer.allocate(64 * sizeof(float));
    auto data = [&](int index) { return reinterpret_cast<float*>(storage[index].ptr()); };
    const std::array<float, 6> q{.2F,.6F,.8F,.2F,.3F,.1F};
    CRR_CUDA_CHECK(cudaMemcpy(data(0), q.data(), sizeof(q), cudaMemcpyHostToDevice));
    curobo::robot::Kinematics fk(argv[1]);
    fk.forward(data(0), data(1), data(2), data(3), 1, 1);
    curobo::robot::IkSolveRequest request;
    request.current_positions = data(0);
    request.goal_positions = data(1);
    request.goal_quaternions = data(2);
    request.solution = data(4);
    request.success = reinterpret_cast<uint8_t*>(data(5));
    request.position_error = data(6);
    request.orientation_error = data(7);
    request.collision_cost = data(8);
    auto expect = [&](bool expected, const char* label) {
      ik->solve(request);
      uint8_t success = 0;
      CRR_CUDA_CHECK(cudaMemcpy(&success, request.success, sizeof(success), cudaMemcpyDeviceToHost));
      if ((success != 0) != expected) throw std::runtime_error(label);
    };
    expect(true, "initial empty scene failed");
    curobo::robot::Cuboid obstacle;
    obstacle.name = "blocking_world";
    obstacle.dimensions = {4,4,4};
    const auto handle = scene.add_cuboid(obstacle);
    expect(false, "shared IK ignored new scene obstacle");
    scene.set_obstacle_enabled(handle, false);
    expect(true, "shared IK did not recover after obstacle disabled");
    curobo::robot::SphereAttachment attachment;
    attachment.link_name = "attached_object";
    attachment.spheres = {{{0,0,0}, 10}};
    planner.attach_object(attachment);
    expect(false, "shared IK ignored attached geometry");
    planner.detach_object();
    expect(true, "shared IK did not recover after detach");
    curobo::robot::LinkInertialUpdate payload;
    payload.link_name = "attached_object";
    payload.mass = 1000;
    payload.center_of_mass = std::array<float,3>{.5F,0,.1F};
    payload.inertia = std::array<float,6>{};
    planner.update_link_inertial(payload);
    expect(false, "shared IK ignored payload torque");
    payload.mass = .1F;
    planner.update_link_inertial(payload);
    expect(true, "shared IK did not recover after payload update");
    std::cout << "PASS: live scene, attachment, and inertia shared through CUDA graphs\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
