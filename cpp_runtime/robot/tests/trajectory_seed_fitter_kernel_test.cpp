// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "trajectory_seed_fitter_cuda.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class DeviceArray final {
 public:
  explicit DeviceArray(const std::vector<float> &values) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), values.size() * sizeof(float)));
    check_cuda(
        cudaMemcpy(data_, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  ~DeviceArray() { cudaFree(data_); }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  float *data() const { return data_; }

 private:
  float *data_{nullptr};
};

void check_manufactured_solution(int knots) {
  constexpr int SEEDS = 2;
  constexpr int DOF = 3;
  constexpr int INTERPOLATION_STEPS = 4;
  const int free_knots = knots - 1;
  const int steps = (knots + 4) * INTERPOLATION_STEPS + 1;
  const int states = SEEDS * steps * DOF;
  std::array<std::vector<float>, 4> basis;
  for (auto &values : basis) values.resize(steps * free_knots);
  std::vector<float> path_normal(free_knots * free_knots);
  std::vector<float> expected(SEEDS * knots * DOF);
  std::vector<float> goal(SEEDS * DOF);
  for (int seed = 0; seed < SEEDS; ++seed) {
    for (int joint = 0; joint < DOF; ++joint) {
      goal[seed * DOF + joint] = 0.4F + 0.1F * joint + 0.05F * seed;
      for (int control = 0; control < knots; ++control) {
        expected[(seed * knots + control) * DOF + joint] =
            control == free_knots ? goal[seed * DOF + joint]
                                  : 0.2F * std::sin(0.1F * control + 0.3F * joint) + 0.05F * seed;
      }
    }
  }
  std::vector<float> residual(states);
  std::array<std::vector<float>, 3> boundary;
  for (auto &values : boundary) values.resize(states);
  for (int step = 0; step < steps; ++step) {
    const int span = step / INTERPOLATION_STEPS;
    const int first = std::max(0, span - 4);
    const int end = std::min(free_knots, span);
    for (int control = first; control < end; ++control) {
      const int index = step * free_knots + control;
      const int delta = span - control;
      basis[0][index] = 0.1F * delta + 0.02F * (step % INTERPOLATION_STEPS);
      basis[1][index] = 0.1F * (delta - 2);
      basis[2][index] = delta % 2 == 0 ? 0.05F : -0.05F;
      basis[3][index] = 0.03F * (delta - 1);
    }
    for (int control = first; control < end; ++control) {
      const int index = step * free_knots + control;
      for (int column = first; column < end; ++column) {
        path_normal[control * free_knots + column] +=
            basis[0][index] * basis[0][step * free_knots + column];
      }
      for (int seed = 0; seed < SEEDS; ++seed) {
        for (int joint = 0; joint < DOF; ++joint) {
          const int state = (seed * steps + step) * DOF + joint;
          const float value = expected[(seed * knots + control) * DOF + joint];
          // These responses make the known controls solve every weighted normal system.
          residual[state] += basis[0][index] * value;
          for (int derivative = 0; derivative < 3; ++derivative) {
            boundary[derivative][state] -= basis[derivative + 1][index] * value;
          }
        }
      }
    }
  }
  float path_trace = 0.0F;
  for (int control = 0; control < free_knots; ++control) {
    path_trace += path_normal[control * free_knots + control];
  }
  DeviceArray positions(basis[0]), velocities(basis[1]), accelerations(basis[2]), jerks(basis[3]);
  DeviceArray normal(path_normal), controls(expected), goals(goal), position_residual(residual);
  DeviceArray boundary_velocity(boundary[0]), boundary_acceleration(boundary[1]),
      boundary_jerk(boundary[2]);
  DeviceArray current_velocity(std::vector<float>(states, 0.2F));
  DeviceArray current_acceleration(std::vector<float>(states, 0.1F));
  DeviceArray limits(std::vector<float>(DOF, 1.0F));
  DeviceArray velocity_trace{std::vector<float>(states)};
  DeviceArray acceleration_trace{std::vector<float>(states)};
  DeviceArray jerk_trace{std::vector<float>(states)};
  DeviceArray scales{std::vector<float>(SEEDS)};
  DeviceArray output(std::vector<float>(expected.size(), NAN));
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  using namespace curobo::robot::detail;
  check_cuda(launch_seed_fitter_derivative_scale(
      current_velocity.data(), current_acceleration.data(), limits.data(), limits.data(),
      limits.data(), velocities.data(), accelerations.data(), jerks.data(), SEEDS, steps, DOF,
      knots, INTERPOLATION_STEPS, path_trace, 1.0F, 1.0E-6F, velocity_trace.data(),
      acceleration_trace.data(), jerk_trace.data(), scales.data(), stream));
  check_cuda(launch_seed_fitter_solve_irls(
      position_residual.data(), boundary_velocity.data(), boundary_acceleration.data(),
      boundary_jerk.data(), current_velocity.data(), current_acceleration.data(), controls.data(),
      goals.data(), limits.data(), limits.data(), limits.data(), positions.data(),
      velocities.data(), accelerations.data(), jerks.data(), normal.data(), scales.data(), SEEDS,
      steps, DOF, knots, INTERPOLATION_STEPS, 0.0F, 0.2F, 1.0F, 1.0E-6F, output.data(), stream));
  check_cuda(cudaStreamEndCapture(stream, &graph));
  check_cuda(cudaGraphInstantiateWithFlags(&executable, graph, 0));
  for (int replay = 0; replay < 2; ++replay) {
    check_cuda(cudaGraphLaunch(executable, stream));
    check_cuda(cudaStreamSynchronize(stream));
    std::vector<float> actual(expected.size());
    check_cuda(cudaMemcpy(actual.data(), output.data(), actual.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t index = 0; index < actual.size(); ++index) {
      if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected[index]) > 2.0E-4F) {
        throw std::runtime_error("banded seed-fitter solve differs from manufactured solution");
      }
    }
  }
  check_cuda(cudaGraphExecDestroy(executable));
  check_cuda(cudaGraphDestroy(graph));
  check_cuda(cudaStreamDestroy(stream));
}

}  // namespace

int main() {
  try {
    for (int knots : {2, 12, 16, 64, 128, 129, 256, 512, 1024}) {
      check_manufactured_solution(knots);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
