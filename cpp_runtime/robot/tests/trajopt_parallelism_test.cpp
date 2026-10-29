// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/trajectory_optimization.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <typename ValueType>
class Buffer {
 public:
  explicit Buffer(size_t count) : count_(count) {
    check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(ValueType)));
  }
  ~Buffer() { cudaFree(data_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ValueType* data() const { return data_; }
  void upload(const std::vector<ValueType>& values) {
    if (values.size() != count_) throw std::runtime_error("buffer size mismatch");
    check(cudaMemcpy(data_, values.data(), count_ * sizeof(ValueType), cudaMemcpyHostToDevice));
  }
  void append(std::vector<float>& result) const {
    std::vector<ValueType> values(count_);
    check(cudaMemcpy(values.data(), data_, count_ * sizeof(ValueType), cudaMemcpyDeviceToHost));
    for (auto value : values) result.push_back(static_cast<float>(value));
  }
 private:
  ValueType* data_{nullptr};
  size_t count_;
};

std::vector<float> evaluate(const std::string& artifact, bool graph) {
  std::vector<float> result;
  for (const int seeds : {1, 4}) {
    for (const int finetune : {0, 1}) {
      curobo::robot::TrajectoryOptimizationConfig config;
      config.num_seeds = seeds;
      config.max_iterations = 6;
      config.finetune_attempts = finetune;
      config.check_effort_limits = finetune != 0;
      curobo::robot::TrajectoryOptimizer optimizer(artifact, config);
      const auto info = optimizer.info();
      const int dof = info.num_dof;
      const int states = info.trajectory_steps * dof;
      const int variables = info.num_knots * dof;
      std::vector<float> start(dof), goal(dof);
      std::ifstream input(artifact + "/default_joint_position.f32", std::ios::binary);
      input.read(reinterpret_cast<char*>(start.data()), dof * sizeof(float));
      if (!input) throw std::runtime_error("cannot read home position");
      goal = start;
      goal[0] += 0.05F;
      Buffer<float> start_buffer(dof), goal_buffer(dof), knots(variables);
      Buffer<float> position(states), velocity(states), acceleration(states), jerk(states);
      Buffer<float> cost(1), collision(1), dt(1);
      Buffer<uint8_t> success(1), valid(seeds);
      Buffer<int32_t> selected(1);
      start_buffer.upload(start);
      goal_buffer.upload(goal);
      curobo::robot::TrajectoryOptimizationRequest request;
      request.start_position = start_buffer.data();
      request.goal_position = goal_buffer.data();
      request.optimized_knots = knots.data();
      request.trajectory_position = position.data();
      request.trajectory_velocity = velocity.data();
      request.trajectory_acceleration = acceleration.data();
      request.trajectory_jerk = jerk.data();
      request.success = success.data();
      request.cost = cost.data();
      request.collision_cost = collision.data();
      request.trajectory_dt = dt.data();
      request.selected_seed = selected.data();
      request.seed_valid = valid.data();
      cudaStream_t stream = nullptr;
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      request.stream = stream;
      cudaGraph_t captured = nullptr;
      cudaGraphExec_t executable = nullptr;
      valid.upload(std::vector<uint8_t>(seeds, 1U));
      if (graph) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        optimizer.optimize(request);
        check(cudaStreamEndCapture(stream, &captured));
        check(cudaGraphInstantiate(&executable, captured, 0));
      }
      for (const int mode : {0, 1, 2}) {
        std::vector<uint8_t> active(seeds, mode == 0 ? 1U : 0U);
        if (mode == 1) active.at(static_cast<size_t>(seeds - 1)) = 1U;
        valid.upload(active);
        for (int repeat = 0; repeat < 2; ++repeat) {
          if (graph) check(cudaGraphLaunch(executable, stream));
          else optimizer.optimize(request);
        }
        check(cudaStreamSynchronize(stream));
        success.append(result);
        selected.append(result);
        cost.append(result);
        collision.append(result);
        dt.append(result);
        knots.append(result);
        position.append(result);
        velocity.append(result);
        acceleration.append(result);
        jerk.append(result);
      }
      if (graph) {
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(captured));
      }
      check(cudaStreamDestroy(stream));
    }
  }
  return result;
}

void compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("reference size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    const float tolerance = 1.0e-3F + 1.0e-4F * std::abs(expected[i]);
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
      throw std::runtime_error("TrajOpt mismatch at " + std::to_string(i) + ": " +
                               std::to_string(actual[i]) + " vs " + std::to_string(expected[i]));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 4) {
      throw std::runtime_error("usage: trajopt_parallelism_test ARTIFACT [--write-reference|--reference FILE]");
    }
    const auto result = evaluate(argv[1], true);
    compare(evaluate(argv[1], false), result);
    if (argc == 4 && std::string(argv[2]) == "--write-reference") {
      std::ofstream output(argv[3]);
      output << std::setprecision(9);
      for (float value : result) output << value << '\n';
      if (!output) throw std::runtime_error("cannot write reference");
    } else if (argc == 4 && std::string(argv[2]) == "--reference") {
      std::ifstream input(argv[3]);
      std::vector<float> expected;
      float value;
      while (input >> value) expected.push_back(value);
      compare(result, expected);
    } else if (argc != 2) {
      throw std::runtime_error("unknown reference option");
    }
    std::cout << "PASS TrajOpt validity masks, finetuning, effort and graph replay ("
              << result.size() << " values)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
