// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_robot_runtime/inverse_kinematics.h"
#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/scene.h"

namespace {
namespace robot = curobo::robot;

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename ValueType>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(ValueType)));
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ValueType* get() { return data_; }
  void upload(const std::vector<ValueType>& values) {
    if (values.size() != count_) throw std::runtime_error("upload size mismatch");
    check_cuda(cudaMemcpy(data_, values.data(), count_ * sizeof(ValueType), cudaMemcpyHostToDevice));
  }
  std::vector<ValueType> download() const {
    std::vector<ValueType> values(count_);
    check_cuda(cudaMemcpy(values.data(), data_, count_ * sizeof(ValueType), cudaMemcpyDeviceToHost));
    return values;
  }
 private:
  ValueType* data_{nullptr};
  size_t count_{0};
};

std::vector<float> read_values(std::istream& input, int count) {
  std::vector<float> result(count);
  for (auto& value : result) input >> value;
  if (!input) throw std::runtime_error("truncated benchmark input");
  return result;
}

void load_scene(std::istream& input, robot::Scene& scene) {
  int count = 0;
  input >> count;
  if (!input || count < 0 || count > scene.config().max_cuboids)
    throw std::runtime_error("invalid benchmark obstacle count");
  scene.clear();
  for (int i = 0; i < count; ++i) {
    const auto values = read_values(input, 10);
    robot::Cuboid box;
    box.name = std::to_string(i);
    std::copy_n(values.begin(), 3, box.pose.position.begin());
    std::copy_n(values.begin() + 3, 4, box.pose.quaternion.begin());
    std::copy_n(values.begin() + 7, 3, box.dimensions.begin());
    scene.add_cuboid(box);
  }
}

template <typename Function>
double measure(Function function) {
  check_cuda(cudaDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  function();
  check_cuda(cudaDeviceSynchronize());
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void run_ik(const std::string& artifact, std::istream& input, std::ostream& output,
            bool collision_free, bool humanoid) {
  robot::Scene scene;
  load_scene(input, scene);
  int batch_size = 0, batches = 0;
  input >> batch_size >> batches;
  if (!input || batch_size <= 0 || batches <= 0) throw std::runtime_error("invalid IK dimensions");
  robot::InverseKinematicsConfig config;
  config.max_batch_size = batch_size;
  config.check_self_collision = collision_free;
  if (humanoid) config.lbfgs_iterations = 160;
  robot::InverseKinematics solver(artifact, config, collision_free ? &scene : nullptr);
  robot::Kinematics kinematics(artifact);
  const auto info = solver.info();
  DeviceBuffer<float> joints(batch_size * info.num_dof);
  DeviceBuffer<float> positions(batch_size * info.num_tool_frames * 3);
  DeviceBuffer<float> quaternions(batch_size * info.num_tool_frames * 4);
  DeviceBuffer<float> jacobians(batch_size * info.num_tool_frames * 6 * info.num_dof);
  DeviceBuffer<float> solution(batch_size * info.num_dof), p_error(batch_size), q_error(batch_size);
  DeviceBuffer<float> collision(batch_size);
  DeviceBuffer<uint8_t> success(batch_size);
  robot::IkSolveRequest request;
  request.batch_size = batch_size;
  request.goal_positions = positions.get();
  request.goal_quaternions = quaternions.get();
  request.solution = solution.get();
  request.success = success.get();
  request.position_error = p_error.get();
  request.orientation_error = q_error.get();
  request.collision_cost = collision.get();
  for (int batch = 0; batch < batches; ++batch) {
    joints.upload(read_values(input, batch_size * info.num_dof));
    kinematics.forward(joints.get(), positions.get(), quaternions.get(), jacobians.get(), batch_size, 1);
    check_cuda(cudaDeviceSynchronize());
    if (batch == 0) {
      // Warm the measured instance using the production early-exit policy.
      for (int i = 0; i < 3; ++i) solver.solve(request);
      check_cuda(cudaDeviceSynchronize());
    }
    const double elapsed = measure([&] { solver.solve(request); });
    const auto valid = success.download();
    const auto pe = p_error.download(), qe = q_error.download();
    for (int i = 0; i < batch_size; ++i) {
      output << "{\"batch\":" << batch << ",\"query\":" << i
             << ",\"time_s\":" << elapsed << ",\"success\":" << int(valid[i]);
      if (valid[i])
        output << ",\"position_error_m\":" << pe[i] << ",\"orientation_error_rad\":" << qe[i];
      output << "}\n";
    }
    output.flush();
  }
}

void write_states(std::ostream& output, const float* pointer, size_t count) {
  std::vector<float> values(count);
  check_cuda(cudaMemcpy(values.data(), pointer, count * sizeof(float), cudaMemcpyDeviceToHost));
  output.write(reinterpret_cast<const char*>(values.data()), count * sizeof(float));
}

void run_motion(const std::string& artifact, std::istream& input, std::ostream& output,
                const std::string& output_path, bool dynamics, int max_cuboids) {
  robot::SceneConfig scene_config;
  scene_config.max_cuboids = max_cuboids;
  scene_config.max_capsules = 0;
  scene_config.max_cylinders = 0;
  scene_config.max_meshes = 0;
  scene_config.max_voxel_grids = 0;
  robot::Scene scene(scene_config);
  robot::MotionPlanningConfig config;
  config.trajectory_optimization.check_effort_limits = dynamics;
  config.inverse_kinematics.check_effort_limits = dynamics;
  config.rrt.check_effort_limits = dynamics;
  robot::MotionPlanner planner(artifact, config, &scene);
  robot::LinkInertialUpdate payload;
  payload.link_name = "attached_object";
  payload.mass = dynamics ? 3.0F : 0.0F;
  planner.update_link_inertial(payload);
  const auto info = planner.info();
  DeviceBuffer<float> start(info.num_dof), positions(3), quaternions(4);
  DeviceBuffer<float> knots(info.num_knots * info.num_dof);
  DeviceBuffer<float> q(info.trajectory_steps * info.num_dof), v(info.trajectory_steps * info.num_dof);
  DeviceBuffer<float> a(info.trajectory_steps * info.num_dof), j(info.trajectory_steps * info.num_dof);
  DeviceBuffer<float> cost(1), collision(1), dt(1), pe(1), qe(1);
  DeviceBuffer<uint8_t> success(1), used_rrt(1);
  DeviceBuffer<int32_t> attempts(1), rrt_iterations(1);
  robot::PoseMotionPlanningRequest request;
  request.current_position = start.get();
  request.goal_positions = positions.get();
  request.goal_quaternions = quaternions.get();
  request.optimized_knots = knots.get();
  request.trajectory_position = q.get();
  request.trajectory_velocity = v.get();
  request.trajectory_acceleration = a.get();
  request.trajectory_jerk = j.get();
  request.success = success.get();
  request.cost = cost.get();
  request.collision_cost = collision.get();
  request.trajectory_dt = dt.get();
  request.position_error = pe.get();
  request.orientation_error = qe.get();
  request.used_rrt = used_rrt.get();
  request.planning_attempts = attempts.get();
  request.rrt_iterations = rrt_iterations.get();
  std::ofstream states(output_path + ".states.f32", std::ios::binary);
  if (!states) throw std::runtime_error("cannot open trajectory output");
  int problems = 0;
  input >> problems;
  if (!input || problems <= 0) throw std::runtime_error("invalid motion count");
  for (int index = 0; index < problems; ++index) {
    start.upload(read_values(input, info.num_dof));
    positions.upload(read_values(input, 3));
    quaternions.upload(read_values(input, 4));
    load_scene(input, scene);
    if (index == 0) {
      for (const int graph_attempt : {0, 1, 1}) {
        request.enable_graph_attempt = graph_attempt;
        planner.plan_pose(request);
      }
    }
    const double elapsed = measure([&] { planner.plan_pose(request); });
    const bool valid = success.download()[0] != 0;
    output << "{\"index\":" << index << ",\"success\":" << int(valid)
           << ",\"plan_time_s\":" << elapsed << ",\"ik_time_s\":" << planner.last_ik_time()
           << ",\"rrt_time_s\":" << planner.last_rrt_time()
           << ",\"rrt_calls\":" << planner.last_rrt_calls()
           << ",\"attempts\":" << attempts.download()[0];
    if (valid) {
      const auto trajectory = planner.timeopt_trajectory();
      int count = 0;
      float sample_dt = 0, duration = 0;
      check_cuda(cudaMemcpy(&count, trajectory.sample_count, sizeof(count), cudaMemcpyDeviceToHost));
      check_cuda(cudaMemcpy(&sample_dt, trajectory.dt, sizeof(float), cudaMemcpyDeviceToHost));
      check_cuda(cudaMemcpy(&duration, trajectory.duration, sizeof(float), cudaMemcpyDeviceToHost));
      if (count < 2 || count > trajectory.capacity) throw std::runtime_error("invalid executable trajectory");
      const auto offset = static_cast<int64_t>(states.tellp()) / sizeof(float);
      for (const auto* pointer : {trajectory.position, trajectory.velocity, trajectory.acceleration, trajectory.jerk})
        write_states(states, pointer, static_cast<size_t>(count) * info.num_dof);
      output << ",\"position_error_m\":" << pe.download()[0]
             << ",\"orientation_error_rad\":" << qe.download()[0]
             << ",\"used_rrt\":" << int(used_rrt.download()[0])
             << ",\"state_offset\":" << offset << ",\"steps\":" << count
             << ",\"dt\":" << sample_dt << ",\"motion_time_s\":" << duration;
    }
    output << "}\n";
    output.flush();
    states.flush();
    if ((index + 1) % 50 == 0 || index + 1 == problems)
      std::cerr << index + 1 << '/' << problems << " completed\n";
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 7 || argc > 8) throw std::invalid_argument(
        "usage: curobo_benchmark MODE ARTIFACT INPUT OUTPUT COLLISION_OR_DYNAMICS HUMANOID [MAX_CUBOIDS]");
    const std::string mode = argv[1];
    if ((mode == "ik" && argc != 7) || (mode == "motion" && argc != 8))
      throw std::invalid_argument("motion mode requires MAX_CUBOIDS; IK mode does not");
    std::ifstream input(argv[3]);
    std::ofstream output(argv[4]);
    if (!input || !output) throw std::runtime_error("cannot open benchmark input/output");
    output << std::setprecision(10);
    if (mode == "ik") run_ik(argv[2], input, output, std::stoi(argv[5]) != 0, std::stoi(argv[6]) != 0);
    else if (mode == "motion") run_motion(argv[2], input, output, argv[4], std::stoi(argv[5]) != 0, std::stoi(argv[7]));
    else throw std::invalid_argument("unknown benchmark mode");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
