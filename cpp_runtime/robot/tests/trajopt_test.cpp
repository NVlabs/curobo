// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_robot_runtime/scene.h"
#include "curobo_robot_runtime/trajectory_optimization.h"

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
  const ValueType* get() const { return pointer_; }

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
    throw std::runtime_error(std::string(description) + " mismatch: " +
                             std::to_string(actual) + " versus " +
                             std::to_string(expected));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: curobo_robot_trajopt_test ARTIFACT_DIRECTORY");
    }
    curobo::robot::TrajectoryOptimizationConfig config;
    config.num_seeds = 4;
    config.max_iterations = 50;
    config.finetune_attempts = 0;
    config.check_effort_limits = true;
    curobo::robot::TrajectoryOptimizer optimizer(argv[1], config);
    const auto info = optimizer.info();
    if (info.num_dof != 7 || info.num_knots != 16 ||
        info.trajectory_steps != 81) {
      throw std::runtime_error("TrajOpt test expects the default Franka artifact");
    }
    curobo::robot::LinkInertialUpdate payload;
    payload.link_name = "attached_object";
    payload.mass = 0.1F;
    payload.center_of_mass = std::array<float, 3>{0.0F, 0.0F, 0.1F};
    payload.inertia =
        std::array<float, 6>{0.001F, 0.001F, 0.001F, 0.0F, 0.0F, 0.0F};
    optimizer.update_link_inertial(payload);

    const std::array<float, 7> start{
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F};
    const std::array<float, 7> goal{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    DeviceValue<float> start_device(start.size());
    DeviceValue<float> goal_device(goal.size());
    start_device.upload(start.data());
    goal_device.upload(goal.data());

    const size_t knot_values =
        static_cast<size_t>(info.num_knots * info.num_dof);
    const size_t trajectory_values =
        static_cast<size_t>(info.trajectory_steps * info.num_dof);
    DeviceValue<float> knots(knot_values);
    DeviceValue<float> position(trajectory_values);
    DeviceValue<float> velocity(trajectory_values);
    DeviceValue<float> acceleration(trajectory_values);
    DeviceValue<float> jerk(trajectory_values);
    DeviceValue<uint8_t> success(1);
    DeviceValue<float> cost(1);
    DeviceValue<float> collision_cost(1);
    DeviceValue<float> trajectory_dt(1);
    DeviceValue<int32_t> selected_seed(1);
    curobo::robot::TrajectoryOptimizationRequest request{
        start_device.get(),
        goal_device.get(),
        nullptr,
        nullptr,
        knots.get(),
        position.get(),
        velocity.get(),
        acceleration.get(),
        jerk.get(),
        success.get(),
        cost.get(),
        collision_cost.get(),
        trajectory_dt.get(),
        selected_seed.get(),
        nullptr,
    };
    optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());

    uint8_t success_value = 0;
    float cost_value = 0.0F;
    float collision_cost_value = 0.0F;
    float trajectory_dt_value = 0.0F;
    int32_t selected_seed_value = -1;
    success.download(&success_value);
    cost.download(&cost_value);
    collision_cost.download(&collision_cost_value);
    trajectory_dt.download(&trajectory_dt_value);
    selected_seed.download(&selected_seed_value);
    if (success_value != 1 || !std::isfinite(cost_value) ||
        collision_cost_value != 0.0F ||
        !std::isfinite(trajectory_dt_value) || trajectory_dt_value <= 0.0F ||
        trajectory_dt_value > config.trajectory_dt ||
        selected_seed_value < 0 || selected_seed_value >= config.num_seeds) {
      throw std::runtime_error(
          "TrajOpt failed: success=" + std::to_string(success_value) +
          " cost=" + std::to_string(cost_value) +
          " collision_cost=" + std::to_string(collision_cost_value));
    }
    std::vector<float> position_host(trajectory_values);
    position.download(position_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(position_host.at(static_cast<size_t>(joint)),
                   start.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "trajectory start");
      const size_t end_index =
          static_cast<size_t>((info.trajectory_steps - 1) * info.num_dof + joint);
      require_near(position_host.at(end_index), goal.at(static_cast<size_t>(joint)),
                   1.0E-5F, "trajectory goal");
    }

    const std::array<float, 7> start_velocity{
        0.01F, -0.01F, 0.005F, -0.01F, 0.005F, 0.01F, -0.005F};
    const std::array<float, 7> start_acceleration{
        0.02F, -0.02F, 0.01F, -0.02F, 0.01F, 0.02F, -0.01F};
    DeviceValue<float> start_velocity_device(start_velocity.size());
    DeviceValue<float> start_acceleration_device(start_acceleration.size());
    start_velocity_device.upload(start_velocity.data());
    start_acceleration_device.upload(start_acceleration.data());
    request.start_velocity = start_velocity_device.get();
    request.start_acceleration = start_acceleration_device.get();
    optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    if (success_value != 1) {
      throw std::runtime_error(
          "TrajOpt rejected a non-static start boundary");
    }
    std::vector<float> velocity_host(trajectory_values);
    std::vector<float> acceleration_host(trajectory_values);
    velocity.download(velocity_host.data());
    acceleration.download(acceleration_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(velocity_host.at(static_cast<size_t>(joint)),
                   start_velocity.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "trajectory start velocity");
      require_near(acceleration_host.at(static_cast<size_t>(joint)),
                   start_acceleration.at(static_cast<size_t>(joint)), 1.0E-4F,
                   "trajectory start acceleration");
    }
    request.start_velocity = nullptr;
    request.start_acceleration = nullptr;

    cudaStream_t graph_stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    check_cuda(cudaStreamCreate(&graph_stream));
    request.stream = graph_stream;
    check_cuda(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal));
    optimizer.optimize(request);
    check_cuda(cudaStreamEndCapture(graph_stream, &graph));
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, 0));
    check_cuda(cudaGraphLaunch(graph_exec, graph_stream));
    check_cuda(cudaStreamSynchronize(graph_stream));
    success.download(&success_value);
    if (success_value != 1) {
      throw std::runtime_error("CUDA Graph replay did not produce a valid trajectory");
    }
    velocity.download(velocity_host.data());
    acceleration.download(acceleration_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(velocity_host.at(static_cast<size_t>(joint)), 0.0F,
                   1.0E-5F, "default trajectory start velocity");
      require_near(acceleration_host.at(static_cast<size_t>(joint)), 0.0F,
                   1.0E-4F, "default trajectory start acceleration");
    }
    check_cuda(cudaGraphExecDestroy(graph_exec));
    check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(graph_stream));

    auto request_seed_config = config;
    request_seed_config.num_seeds = 1;
    request_seed_config.max_iterations = 1;
    request_seed_config.check_effort_limits = false;
    curobo::robot::TrajectoryOptimizer request_seed_optimizer(
        argv[1], request_seed_config);
    request.seed_knots = knots.get();
    request.stream = nullptr;
    request_seed_optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    if (success_value != 1) {
      throw std::runtime_error("TrajOpt rejected a caller-provided valid seed");
    }
    request.seed_knots = nullptr;

    auto finetune_config = config;
    finetune_config.finetune_attempts = 2;
    curobo::robot::TrajectoryOptimizer finetune_optimizer(
        argv[1], finetune_config);
    finetune_optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    trajectory_dt.download(&trajectory_dt_value);
    if (success_value != 1 || !(trajectory_dt_value < config.trajectory_dt) ||
        !std::isfinite(trajectory_dt_value)) {
      throw std::runtime_error(
          "TrajOpt dt refinement did not retain a faster feasible trajectory");
    }
    check_cuda(cudaStreamCreate(&graph_stream));
    request.stream = graph_stream;
    check_cuda(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal));
    finetune_optimizer.optimize(request);
    check_cuda(cudaStreamEndCapture(graph_stream, &graph));
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, 0));
    check_cuda(cudaGraphLaunch(graph_exec, graph_stream));
    check_cuda(cudaStreamSynchronize(graph_stream));
    success.download(&success_value);
    if (success_value != 1) {
      throw std::runtime_error(
          "dt-refined CUDA Graph replay did not produce a valid trajectory");
    }
    check_cuda(cudaGraphExecDestroy(graph_exec));
    check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(graph_stream));
    request.stream = nullptr;

    const std::array<float, 14> seed_goals{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F,
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    const std::array<float, 14> seed_starts{
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    DeviceValue<float> seed_goals_device(seed_goals.size());
    DeviceValue<float> seed_starts_device(seed_starts.size());
    seed_goals_device.upload(seed_goals.data());
    seed_starts_device.upload(seed_starts.data());
    auto many_seed_config = config;
    many_seed_config.num_seeds = 2;
    many_seed_config.max_iterations = 1;
    curobo::robot::TrajectoryOptimizer many_seed_optimizer(
        argv[1], many_seed_config);
    request.seed_goal_positions = seed_goals_device.get();
    request.seed_start_positions = seed_starts_device.get();
    many_seed_optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    selected_seed.download(&selected_seed_value);
    if (success_value != 1 || selected_seed_value != 1) {
      throw std::runtime_error(
          "TrajOpt did not select the best per-seed goal trajectory");
    }
    position.download(position_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      const size_t end_index = static_cast<size_t>(
          (info.trajectory_steps - 1) * info.num_dof + joint);
      require_near(position_host.at(static_cast<size_t>(joint)),
                   goal.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "selected per-seed trajectory start");
      require_near(position_host.at(end_index), goal.at(static_cast<size_t>(joint)),
                   1.0E-5F, "selected per-seed trajectory goal");
    }
    const std::array<uint8_t, 2> seed_valid{1U, 0U};
    DeviceValue<uint8_t> seed_valid_device(seed_valid.size());
    seed_valid_device.upload(seed_valid.data());
    request.seed_valid = seed_valid_device.get();
    many_seed_optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    selected_seed.download(&selected_seed_value);
    if (success_value != 1 || selected_seed_value != 0) {
      throw std::runtime_error("TrajOpt selected a masked endpoint pair");
    }
    position.download(position_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      const size_t end_index = static_cast<size_t>(
          (info.trajectory_steps - 1) * info.num_dof + joint);
      require_near(position_host.at(static_cast<size_t>(joint)),
                   start.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "masked per-seed trajectory start");
      require_near(position_host.at(end_index), goal.at(static_cast<size_t>(joint)),
                   1.0E-5F, "masked per-seed trajectory goal");
    }
    request.seed_valid = nullptr;
    request.seed_start_positions = nullptr;
    request.seed_goal_positions = nullptr;

    curobo::robot::SceneConfig scene_config;
    scene_config.max_cuboids = 1;
    scene_config.max_capsules = 0;
    scene_config.max_cylinders = 0;
    scene_config.max_meshes = 0;
    curobo::robot::Scene scene(scene_config);
    curobo::robot::Cuboid obstacle;
    obstacle.name = "mid_path_obstacle";
    obstacle.pose.position = {0.218F, 0.126F, 0.723F};
    obstacle.dimensions = {0.02F, 0.02F, 0.02F};
    scene.add_cuboid(obstacle);
    auto one_step_config = config;
    one_step_config.max_iterations = 1;
    curobo::robot::TrajectoryOptimizer one_step_optimizer(
        argv[1], one_step_config, &scene);
    request.stream = nullptr;
    one_step_optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    collision_cost.download(&collision_cost_value);
    if (success_value != 0 || !(collision_cost_value > 0.0F)) {
      throw std::runtime_error(
          "the obstacle test does not intersect the straight-path seed");
    }

    check_cuda(cudaStreamCreate(&graph_stream));
    request.stream = graph_stream;
    check_cuda(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal));
    one_step_optimizer.optimize(request);
    check_cuda(cudaStreamEndCapture(graph_stream, &graph));
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, 0));
    check_cuda(cudaGraphLaunch(graph_exec, graph_stream));
    check_cuda(cudaStreamSynchronize(graph_stream));
    scene.clear();
    obstacle.pose.position = {4.0F, 4.0F, 4.0F};
    scene.add_cuboid(obstacle);
    check_cuda(cudaGraphLaunch(graph_exec, graph_stream));
    check_cuda(cudaStreamSynchronize(graph_stream));
    success.download(&success_value);
    collision_cost.download(&collision_cost_value);
    if (success_value != 1 || collision_cost_value != 0.0F) {
      throw std::runtime_error(
          "TrajOpt graph replay did not consume the replaced scene cache");
    }
    check_cuda(cudaGraphExecDestroy(graph_exec));
    check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(graph_stream));
    request.stream = nullptr;
    scene.clear();
    obstacle.pose.position = {0.218F, 0.126F, 0.723F};
    scene.add_cuboid(obstacle);

    curobo::robot::TrajectoryOptimizer collision_optimizer(
        argv[1], config, &scene);
    request.stream = nullptr;
    collision_optimizer.optimize(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    collision_cost.download(&collision_cost_value);
    if (success_value != 1 || collision_cost_value != 0.0F) {
      throw std::runtime_error(
          "TrajOpt did not find a collision-free path around the cuboid: success=" +
          std::to_string(success_value) + " collision_cost=" +
          std::to_string(collision_cost_value));
    }
    // Integrated timeopt must preserve endpoints and observe payload changes
    // through a captured solve, including its shared inverse-dynamics model.
    auto timeopt_config = config;
    timeopt_config.max_iterations = 25;
    timeopt_config.finetune_attempts = 2;
    timeopt_config.enable_timeopt = true;
    curobo::robot::TrajectoryOptimizer timeopt_optimizer(argv[1], timeopt_config);
    const auto disabled_timeopt = optimizer.timeopt_trajectory();
    if (disabled_timeopt.position != nullptr ||
        disabled_timeopt.timing_controls != nullptr ||
        disabled_timeopt.num_timing_controls != 0 || disabled_timeopt.spatial_controls != nullptr ||
        disabled_timeopt.num_spatial_controls != 0) {
      throw std::runtime_error("disabled timeopt exposed an executable trajectory");
    }
    payload.mass = 0.1F;
    timeopt_optimizer.update_link_inertial(payload);
    check_cuda(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking));
    request.stream = graph_stream;
    timeopt_optimizer.optimize(request);
    check_cuda(cudaStreamSynchronize(graph_stream));
    check_cuda(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeThreadLocal));
    timeopt_optimizer.optimize(request);
    check_cuda(cudaStreamEndCapture(graph_stream, &graph));
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, 0));
    float graph_duration = 0.0F;
    for (const float mass : {0.1F, 3.0F}) {
      payload.mass = mass;
      timeopt_optimizer.update_link_inertial(payload);
      check_cuda(cudaGraphLaunch(graph_exec, graph_stream));
      check_cuda(cudaStreamSynchronize(graph_stream));
      success.download(&success_value);
      const auto executable = timeopt_optimizer.timeopt_trajectory();
      if (executable.timing_controls == nullptr || executable.num_timing_controls != 32) {
        throw std::runtime_error("timeopt did not expose selected timing controls");
      }
      if (executable.spatial_controls == nullptr || executable.num_spatial_controls != info.num_knots + 6) {
        throw std::runtime_error("timeopt did not expose complete spatial controls");
      }
      std::vector<float> timing(executable.num_timing_controls);
      check_cuda(cudaMemcpy(timing.data(), executable.timing_controls,
                            timing.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for (const float value : timing) {
        if (!std::isfinite(value) || value <= 0.0F) {
          throw std::runtime_error("selected timing controls must be finite and positive");
        }
      }
      int32_t count = 0;
      float dt = 0.0F;
      check_cuda(cudaMemcpy(&count, executable.sample_count, sizeof(count), cudaMemcpyDeviceToHost));
      check_cuda(cudaMemcpy(&dt, executable.dt, sizeof(dt), cudaMemcpyDeviceToHost));
      check_cuda(cudaMemcpy(&graph_duration, executable.duration, sizeof(float), cudaMemcpyDeviceToHost));
      if (success_value != 1 || count < 2 || count > executable.capacity ||
          !(dt > 0.0F) || dt > 0.025001F) {
        throw std::runtime_error("timeopt graph output failed validation");
      }
      require_near(dt * (count - 1), graph_duration, 1.0E-5F, "timeopt duration");
      std::vector<float> samples(static_cast<size_t>(count) * info.num_dof);
      check_cuda(cudaMemcpy(samples.data(), executable.position, samples.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
      for (int joint = 0; joint < info.num_dof; ++joint) {
        require_near(samples[joint], start[joint], 2.0E-5F, "timeopt start");
        require_near(samples[(count - 1) * info.num_dof + joint], goal[joint],
                     2.0E-5F, "timeopt endpoint");
      }
    }
    curobo::robot::TrajectoryOptimizer fresh_optimizer(argv[1], timeopt_config);
    fresh_optimizer.update_link_inertial(payload);
    fresh_optimizer.optimize(request);
    check_cuda(cudaStreamSynchronize(graph_stream));
    float fresh_duration = 0.0F;
    check_cuda(cudaMemcpy(&fresh_duration, fresh_optimizer.timeopt_trajectory().duration,
                          sizeof(float), cudaMemcpyDeviceToHost));
    require_near(graph_duration, fresh_duration, 1.0E-4F, "payload replay versus fresh model");
    // A captured solve reads the updated scale from fixed device storage.
    timeopt_optimizer.set_execution_speed_scale(0.5F);
    check_cuda(cudaGraphLaunch(graph_exec, graph_stream));
    check_cuda(cudaStreamSynchronize(graph_stream));
    success.download(&success_value);
    if (!success_value) throw std::runtime_error("scaled timeopt graph failed validation");
    check_cuda(cudaMemcpy(&graph_duration, timeopt_optimizer.timeopt_trajectory().duration,
                          sizeof(float), cudaMemcpyDeviceToHost));
    fresh_optimizer.set_execution_speed_scale(0.5F);
    fresh_optimizer.optimize(request);
    check_cuda(cudaStreamSynchronize(graph_stream));
    check_cuda(cudaMemcpy(&fresh_duration, fresh_optimizer.timeopt_trajectory().duration,
                          sizeof(float), cudaMemcpyDeviceToHost));
    require_near(graph_duration, fresh_duration, 1.0E-4F, "speed replay versus fresh model");
    bool rejected_scale = false;
    try { timeopt_optimizer.set_execution_speed_scale(0.0F); }
    catch (const std::invalid_argument&) { rejected_scale = true; }
    if (!rejected_scale) throw std::runtime_error("invalid execution scale was accepted");
    check_cuda(cudaGraphExecDestroy(graph_exec));
    check_cuda(cudaGraphDestroy(graph));
    const std::array<uint8_t, 4> invalid_seeds{0, 0, 0, 0};
    DeviceValue<uint8_t> invalid_seed_device(invalid_seeds.size());
    invalid_seed_device.upload(invalid_seeds.data());
    request.seed_valid = invalid_seed_device.get();
    timeopt_optimizer.optimize(request);
    check_cuda(cudaStreamSynchronize(graph_stream));
    success.download(&success_value);
    int32_t invalid_count = -1;
    check_cuda(cudaMemcpy(&invalid_count, timeopt_optimizer.timeopt_trajectory().sample_count,
                          sizeof(invalid_count), cudaMemcpyDeviceToHost));
    if (success_value != 0 || invalid_count != 0) {
      throw std::runtime_error("timeopt accepted masked seeds");
    }
    const auto rejected_timeopt = timeopt_optimizer.timeopt_trajectory();
    std::vector<float> rejected_timing(rejected_timeopt.num_timing_controls, -1.0F);
    check_cuda(cudaMemcpy(rejected_timing.data(), rejected_timeopt.timing_controls,
                          rejected_timing.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (const float value : rejected_timing) {
      if (value != 0.0F) {
        throw std::runtime_error("failed timeopt retained stale timing controls");
      }
    }
    check_cuda(cudaStreamDestroy(graph_stream));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
