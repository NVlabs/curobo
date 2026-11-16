// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_robot_runtime/rrt.h"
#include "curobo_robot_runtime/scene.h"

namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename ValueType>
class DeviceValue final {
 public:
  explicit DeviceValue(size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          count * sizeof(ValueType)));
  }

  ~DeviceValue() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceValue(const DeviceValue&) = delete;
  DeviceValue& operator=(const DeviceValue&) = delete;

  ValueType* get() { return pointer_; }

  void upload(const ValueType* values) {
    check_cuda(cudaMemcpy(pointer_, values, count_ * sizeof(ValueType),
                          cudaMemcpyHostToDevice));
  }

  void download(ValueType* values) const {
    check_cuda(cudaMemcpy(values, pointer_, count_ * sizeof(ValueType),
                          cudaMemcpyDeviceToHost));
  }

 private:
  ValueType* pointer_{nullptr};
  size_t count_{0};
};

void require_near(float actual, float expected, float tolerance,
                  const char* description) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(description) + " mismatch: actual=" +
                             std::to_string(actual) + " expected=" +
                             std::to_string(expected));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument("usage: curobo_robot_rrt_test ARTIFACT_DIRECTORY");
    }
    curobo::robot::RrtConfig config;
    config.check_effort_limits = true;
    curobo::robot::RrtPlanner direct_planner(argv[1], config);
    const auto info = direct_planner.info();
    if (info.num_dof != 7 || info.samples_per_iteration != 24 ||
        info.max_iterations < config.max_iterations || info.max_path_nodes < 2 ||
        info.max_plans != config.max_plans) {
      throw std::runtime_error("RRT test expects the default Franka artifact");
    }
    curobo::robot::LinkInertialUpdate payload;
    payload.link_name = "attached_object";
    payload.mass = 0.1F;
    payload.center_of_mass = std::array<float, 3>{0.0F, 0.0F, 0.1F};
    payload.inertia =
        std::array<float, 6>{0.001F, 0.001F, 0.001F, 0.0F, 0.0F, 0.0F};
    direct_planner.update_links_inertial({payload});

    const std::array<float, 7> start{
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F};
    const std::array<float, 7> goal{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    DeviceValue<float> start_device(start.size());
    DeviceValue<float> goal_device(goal.size());
    start_device.upload(start.data());
    goal_device.upload(goal.data());
    DeviceValue<float> path(
        static_cast<size_t>(info.max_path_nodes * info.num_dof));
    DeviceValue<int32_t> path_count(1);
    DeviceValue<uint8_t> success(1);
    DeviceValue<int32_t> iterations(1);
    curobo::robot::RrtRequest request{
        start_device.get(), goal_device.get(), path.get(), path_count.get(),
        success.get(), iterations.get(), nullptr};
    direct_planner.plan(request);

    uint8_t success_value = 0;
    int32_t path_count_value = 0;
    int32_t iteration_value = -1;
    success.download(&success_value);
    path_count.download(&path_count_value);
    iterations.download(&iteration_value);
    if (success_value != 1 || path_count_value != 2 || iteration_value != 0) {
      throw std::runtime_error("RRT did not take the collision-free direct path");
    }
    std::vector<float> path_host(
        static_cast<size_t>(info.max_path_nodes * info.num_dof));
    path.download(path_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(path_host.at(static_cast<size_t>(joint)),
                   start.at(static_cast<size_t>(joint)), 1.0E-5F, "RRT start");
      require_near(path_host.at(static_cast<size_t>(info.num_dof + joint)),
                   goal.at(static_cast<size_t>(joint)), 1.0E-5F, "RRT goal");
    }

    request.goal_position = start_device.get();
    direct_planner.plan(request);
    success.download(&success_value);
    path_count.download(&path_count_value);
    if (success_value != 1 || path_count_value != 2) {
      throw std::runtime_error("RRT rejected an identical start and goal");
    }
    request.goal_position = goal_device.get();

    const std::array<float, 7> long_direct_goal{
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, -0.8F};
    goal_device.upload(long_direct_goal.data());
    direct_planner.plan(request);
    success.download(&success_value);
    path_count.download(&path_count_value);
    iterations.download(&iteration_value);
    if (success_value != 1 || path_count_value != 2 || iteration_value != 0) {
      throw std::runtime_error(
          "RRT capped a collision-free direct edge at its growth distance");
    }
    goal_device.upload(goal.data());

    const std::array<float, 14> paired_starts{
        100.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.0F,   -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F};
    const std::array<float, 14> paired_goals{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F,
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    const std::array<uint8_t, 2> paired_valid{0U, 1U};
    DeviceValue<float> paired_starts_device(paired_starts.size());
    DeviceValue<float> paired_goals_device(paired_goals.size());
    DeviceValue<uint8_t> paired_valid_device(paired_valid.size());
    DeviceValue<float> paired_path(
        static_cast<size_t>(2 * info.max_path_nodes * info.num_dof));
    DeviceValue<int32_t> paired_path_count(2);
    DeviceValue<uint8_t> paired_success(2);
    DeviceValue<int32_t> paired_iterations(2);
    paired_starts_device.upload(paired_starts.data());
    paired_goals_device.upload(paired_goals.data());
    paired_valid_device.upload(paired_valid.data());
    curobo::robot::RrtRequest paired_request{
        paired_starts_device.get(), paired_goals_device.get(),
        paired_path.get(), paired_path_count.get(), paired_success.get(),
        paired_iterations.get(), nullptr, 2, paired_valid_device.get()};
    direct_planner.plan(paired_request);
    std::array<uint8_t, 2> paired_success_host{0U, 0U};
    std::array<int32_t, 2> paired_count_host{0, 0};
    paired_success.download(paired_success_host.data());
    paired_path_count.download(paired_count_host.data());
    if (paired_success_host != std::array<uint8_t, 2>{0U, 1U} ||
        paired_count_host != std::array<int32_t, 2>{0, 2}) {
      throw std::runtime_error("RRT did not preserve paired candidate status");
    }
    std::vector<float> paired_path_host(
        static_cast<size_t>(2 * info.max_path_nodes * info.num_dof));
    paired_path.download(paired_path_host.data());
    const size_t selected_path_offset =
        static_cast<size_t>(info.max_path_nodes * info.num_dof);
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(
          paired_path_host.at(selected_path_offset + joint),
          start.at(static_cast<size_t>(joint)), 1.0E-5F,
          "paired RRT start");
      require_near(
          paired_path_host.at(selected_path_offset + info.num_dof + joint),
          goal.at(static_cast<size_t>(joint)), 1.0E-5F,
          "paired RRT goal");
    }
    const std::array<float, 14> two_valid_starts{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F,
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F};
    const std::array<uint8_t, 2> two_valid{1U, 1U};
    paired_starts_device.upload(two_valid_starts.data());
    paired_valid_device.upload(two_valid.data());
    direct_planner.plan(paired_request);
    paired_success.download(paired_success_host.data());
    paired_path_count.download(paired_count_host.data());
    if (paired_success_host != std::array<uint8_t, 2>{1U, 1U} ||
        paired_count_host != std::array<int32_t, 2>{2, 2}) {
      throw std::runtime_error("RRT did not retain all direct candidate paths");
    }
    paired_starts_device.upload(paired_starts.data());
    paired_valid_device.upload(paired_valid.data());

    curobo::robot::Cuboid obstacle;
    obstacle.name = "mid_path_obstacle";
    obstacle.pose.position = {0.218F, 0.126F, 0.723F};
    obstacle.dimensions = {0.02F, 0.02F, 0.02F};

    // This fixture uses one cuboid and must also support cuboid-only artifacts.
    curobo::robot::SceneConfig scene_config;
    scene_config.max_cuboids = 1;
    scene_config.max_capsules = 0;
    scene_config.max_cylinders = 0;
    scene_config.max_meshes = 0;
    curobo::robot::Scene changing_scene(scene_config);
    curobo::robot::RrtPlanner changing_scene_planner(argv[1], config,
                                                     &changing_scene);
    changing_scene_planner.plan(request);
    success.download(&success_value);
    path_count.download(&path_count_value);
    if (success_value != 1 || path_count_value != 2) {
      throw std::runtime_error("RRT rejected a direct path in an empty scene");
    }
    changing_scene.add_cuboid(obstacle);
    changing_scene_planner.reset_seed();
    changing_scene_planner.plan(request);
    success.download(&success_value);
    path_count.download(&path_count_value);
    iterations.download(&iteration_value);
    if (success_value != 1 || path_count_value <= 2 || iteration_value <= 0) {
      throw std::runtime_error(
          "RRT did not refresh CUDA graphs after adding an obstacle type");
    }

    curobo::robot::Scene scene(scene_config);
    scene.add_cuboid(obstacle);
    curobo::robot::RrtPlanner obstacle_planner(argv[1], config, &scene);
    obstacle_planner.plan(request);
    success.download(&success_value);
    path_count.download(&path_count_value);
    iterations.download(&iteration_value);
    if (success_value != 1 || path_count_value <= 2 || iteration_value <= 0 ||
        iteration_value > config.max_iterations) {
      throw std::runtime_error(
          "RRT did not find a bounded path around the cuboid: success=" +
          std::to_string(success_value) + " path_count=" +
          std::to_string(path_count_value) + " iterations=" +
          std::to_string(iteration_value));
    }
    path.download(path_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(path_host.at(static_cast<size_t>(joint)),
                   start.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "obstacle RRT start");
      const auto goal_offset = static_cast<size_t>(
          (path_count_value - 1) * info.num_dof + joint);
      require_near(path_host.at(goal_offset), goal.at(static_cast<size_t>(joint)),
                   1.0E-5F, "obstacle RRT goal");
    }
    const int32_t first_obstacle_path_count = path_count_value;
    const int32_t first_obstacle_iterations = iteration_value;
    obstacle_planner.reset_seed();
    obstacle_planner.plan(request);
    success.download(&success_value);
    path_count.download(&path_count_value);
    iterations.download(&iteration_value);
    if (success_value != 1 || path_count_value != first_obstacle_path_count ||
        iteration_value != first_obstacle_iterations) {
      throw std::runtime_error("RRT CUDA graph replay changed the growth result");
    }
    obstacle_planner.plan(paired_request);
    paired_success.download(paired_success_host.data());
    paired_path_count.download(paired_count_host.data());
    std::array<int32_t, 2> paired_iterations_host{0, 0};
    paired_iterations.download(paired_iterations_host.data());
    if (paired_success_host != std::array<uint8_t, 2>{0U, 1U} ||
        paired_count_host[0] != 0 || paired_count_host[1] <= 2 ||
        paired_iterations_host[1] <= 0) {
      throw std::runtime_error(
          "batched RRT growth did not solve the valid endpoint pair");
    }
    paired_path.download(paired_path_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(
          paired_path_host.at(selected_path_offset + joint),
          start.at(static_cast<size_t>(joint)), 1.0E-5F,
          "batched obstacle RRT start");
      const size_t goal_offset =
          selected_path_offset +
          static_cast<size_t>((paired_count_host[1] - 1) * info.num_dof + joint);
      require_near(paired_path_host.at(goal_offset),
                   goal.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "batched obstacle RRT goal");
    }
    const auto obstacle_path = path_host;
    const int32_t obstacle_path_count = path_count_value;
    for (int edge = 0; edge + 1 < obstacle_path_count; ++edge) {
      const auto source_offset = static_cast<size_t>(edge * info.num_dof);
      const auto target_offset =
          static_cast<size_t>((edge + 1) * info.num_dof);
      start_device.upload(obstacle_path.data() + source_offset);
      goal_device.upload(obstacle_path.data() + target_offset);
      obstacle_planner.plan(request);
      success.download(&success_value);
      path_count.download(&path_count_value);
      if (success_value != 1 || path_count_value != 2) {
        std::ostringstream message;
        message << "RRT returned a collision-invalid path edge " << edge
                << " of " << obstacle_path_count - 1 << "; start=[";
        for (int joint = 0; joint < info.num_dof; ++joint) {
          message << (joint == 0 ? "" : ", ")
                  << obstacle_path.at(source_offset + joint);
        }
        message << "], goal=[";
        for (int joint = 0; joint < info.num_dof; ++joint) {
          message << (joint == 0 ? "" : ", ")
                  << obstacle_path.at(target_offset + joint);
        }
        message << "], success=" << static_cast<int>(success_value)
                << ", path_count=" << path_count_value;
        throw std::runtime_error(message.str());
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
