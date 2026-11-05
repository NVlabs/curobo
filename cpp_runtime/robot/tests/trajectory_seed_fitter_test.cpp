// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_robot_runtime/trajectory_seed_fitter.h"

namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename ValueType>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          count * sizeof(ValueType)));
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  ValueType* get() { return pointer_; }

  void upload(const std::vector<ValueType>& values) {
    if (values.size() != count_) {
      throw std::invalid_argument("seed-fitter test upload size mismatch");
    }
    check_cuda(cudaMemcpy(pointer_, values.data(), count_ * sizeof(ValueType),
                          cudaMemcpyHostToDevice));
  }

  std::vector<ValueType> download() const {
    std::vector<ValueType> values(count_);
    check_cuda(cudaMemcpy(values.data(), pointer_, count_ * sizeof(ValueType),
                          cudaMemcpyDeviceToHost));
    return values;
  }

 private:
  ValueType* pointer_{nullptr};
  size_t count_{0};
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4) {
      throw std::invalid_argument(
          "usage: curobo_robot_seed_fitter_test ARTIFACT_DIRECTORY [OUTPUT] "
          "[NUM_SEEDS]");
    }
    const int seeds = argc == 4 ? std::stoi(argv[3]) : 4;
    if (seeds <= 0) {
      throw std::invalid_argument("NUM_SEEDS must be positive");
    }
    curobo::robot::TrajectorySeedFitterConfig config;
    config.num_seeds = seeds;
    curobo::robot::TrajectorySeedFitter fitter(argv[1], config);
    const auto info = fitter.info();
    if (info.num_dof != 7 || info.num_knots != 16 ||
        info.trajectory_steps != 81 || info.irls_iterations != 2) {
      throw std::runtime_error(
          "seed-fitter test expects the default Franka artifact");
    }

    const int dof = info.num_dof;
    const int steps = info.trajectory_steps;
    const int knots = info.num_knots;
    std::vector<float> start(static_cast<size_t>(seeds * dof));
    std::vector<float> goal(start.size());
    std::vector<float> target(static_cast<size_t>(seeds * steps * dof));
    for (int seed = 0; seed < seeds; ++seed) {
      for (int joint = 0; joint < dof; ++joint) {
        const float start_value = -0.5F + static_cast<float>(joint) / 6.0F;
        const float goal_value =
            0.5F - static_cast<float>(joint) / 6.0F + 0.05F * seed;
        start[static_cast<size_t>(seed * dof + joint)] = start_value;
        goal[static_cast<size_t>(seed * dof + joint)] = goal_value;
        for (int step = 0; step < steps; ++step) {
          const float progress = static_cast<float>(step) / (steps - 1);
          const float curved_progress = progress * progress;
          const float bend = -0.3F + static_cast<float>(joint) / 10.0F;
          target[static_cast<size_t>((seed * steps + step) * dof + joint)] =
              start_value * (1.0F - curved_progress) +
              goal_value * curved_progress +
              std::sin(3.14159265358979323846F * curved_progress) * bend;
        }
      }
    }

    DeviceBuffer<float> target_device(target.size());
    DeviceBuffer<float> start_device(start.size());
    DeviceBuffer<float> goal_device(goal.size());
    DeviceBuffer<float> knots_device(
        static_cast<size_t>(seeds * knots * dof));
    DeviceBuffer<float> dt_device(seeds);
    target_device.upload(target);
    start_device.upload(start);
    goal_device.upload(goal);
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    curobo::robot::TrajectorySeedFitterRequest request;
    request.target_position = target_device.get();
    request.start_position = start_device.get();
    request.goal_position = goal_device.get();
    request.fitted_knots = knots_device.get();
    request.required_dt = dt_device.get();
    request.stream = stream;
    fitter.fit(request);
    fitter.fit(request);
    check_cuda(cudaStreamSynchronize(stream));

    const auto fitted = knots_device.download();
    const auto required_dt = dt_device.download();
    for (int repetition = 0; repetition < 32; ++repetition) {
      fitter.fit(request);
      check_cuda(cudaStreamSynchronize(stream));
      if (knots_device.download() != fitted ||
          dt_device.download() != required_dt) {
        throw std::runtime_error(
            "seed fitter changed output for identical inputs");
      }
    }
    for (int seed = 0; seed < seeds; ++seed) {
      if (!std::isfinite(required_dt.at(static_cast<size_t>(seed))) ||
          required_dt.at(static_cast<size_t>(seed)) <= 0.0F) {
        throw std::runtime_error("seed fitter returned an invalid timestep");
      }
      for (int joint = 0; joint < dof; ++joint) {
        const auto output_index = static_cast<size_t>(
            (seed * knots + knots - 1) * dof + joint);
        const auto goal_index = static_cast<size_t>(seed * dof + joint);
        if (!std::isfinite(fitted.at(output_index)) ||
            std::abs(fitted.at(output_index) - goal.at(goal_index)) > 1.0E-6F) {
          throw std::runtime_error("seed fitter did not pin the goal knot");
        }
      }
    }

    curobo::robot::TrajectorySeedFitterConfig single_config;
    single_config.num_seeds = 1;
    curobo::robot::TrajectorySeedFitter single_fitter(argv[1], single_config);
    DeviceBuffer<float> single_knots(static_cast<size_t>(knots * dof));
    DeviceBuffer<float> single_dt(1);
    for (int seed = 0; seed < seeds; ++seed) {
      auto single_request = request;
      single_request.target_position = target_device.get() + seed * steps * dof;
      single_request.start_position = start_device.get() + seed * dof;
      single_request.goal_position = goal_device.get() + seed * dof;
      single_request.fitted_knots = single_knots.get();
      single_request.required_dt = single_dt.get();
      single_fitter.fit(single_request);
      check_cuda(cudaStreamSynchronize(stream));
      const auto expected = single_knots.download();
      if (!std::equal(expected.begin(), expected.end(), fitted.begin() + seed * knots * dof) ||
          single_dt.download().front() != required_dt.at(seed)) {
        throw std::runtime_error("batched fit differs from independent fit");
      }
    }

    std::vector<float> start_velocity(start.size());
    std::vector<float> start_acceleration(start.size());
    std::vector<float> start_jerk(start.size(), 0.0F);
    for (int seed = 0; seed < seeds; ++seed) {
      for (int joint = 0; joint < dof; ++joint) {
        const auto index = static_cast<size_t>(seed * dof + joint);
        start_velocity[index] = -0.4F + 0.8F * joint / (dof - 1);
        start_acceleration[index] = 2.0F - 4.0F * joint / (dof - 1);
      }
    }
    DeviceBuffer<float> velocity_device(start_velocity.size());
    DeviceBuffer<float> acceleration_device(start_acceleration.size());
    DeviceBuffer<float> jerk_device(start_jerk.size());
    velocity_device.upload(start_velocity);
    acceleration_device.upload(start_acceleration);
    jerk_device.upload(start_jerk);
    request.start_velocity = velocity_device.get();
    request.start_acceleration = acceleration_device.get();
    request.start_jerk = jerk_device.get();
    fitter.fit(request);
    check_cuda(cudaStreamSynchronize(stream));
    const auto moving_fitted = knots_device.download();
    const auto moving_required_dt = dt_device.download();
    cudaGraph_t outer_graph = nullptr;
    cudaGraphExec_t outer_executable = nullptr;
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    fitter.fit(request);
    check_cuda(cudaStreamEndCapture(stream, &outer_graph));
    check_cuda(cudaGraphInstantiate(&outer_executable, outer_graph, 0));
    check_cuda(cudaGraphLaunch(outer_executable, stream));
    check_cuda(cudaStreamSynchronize(stream));
    check_cuda(cudaGraphExecDestroy(outer_executable));
    check_cuda(cudaGraphDestroy(outer_graph));
    if (knots_device.download() != moving_fitted || dt_device.download() != moving_required_dt) {
      throw std::runtime_error("outer graph capture changed fitted controls or timing");
    }
    float maximum_change = 0.0F;
    for (size_t index = 0; index < fitted.size(); ++index) {
      maximum_change =
          std::max(maximum_change, std::abs(fitted[index] - moving_fitted[index]));
    }
    if (maximum_change <= 1.0E-4F) {
      throw std::runtime_error("seed fitter ignored start derivatives");
    }
    for (float dt : moving_required_dt) {
      if (!std::isfinite(dt) || dt <= 0.0F) {
        throw std::runtime_error(
            "moving-state seed fitter returned an invalid timestep");
      }
    }
    if (argc == 3) {
      std::ofstream output(argv[2], std::ios::binary);
      output.write(reinterpret_cast<const char*>(target.data()),
                   static_cast<std::streamsize>(target.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(start.data()),
                   static_cast<std::streamsize>(start.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(goal.data()),
                   static_cast<std::streamsize>(goal.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(fitted.data()),
                   static_cast<std::streamsize>(fitted.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(required_dt.data()),
                   static_cast<std::streamsize>(required_dt.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(start_velocity.data()),
                   static_cast<std::streamsize>(start_velocity.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(start_acceleration.data()),
                   static_cast<std::streamsize>(start_acceleration.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(start_jerk.data()),
                   static_cast<std::streamsize>(start_jerk.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(moving_fitted.data()),
                   static_cast<std::streamsize>(moving_fitted.size() * sizeof(float)));
      output.write(reinterpret_cast<const char*>(moving_required_dt.data()),
                   static_cast<std::streamsize>(moving_required_dt.size() * sizeof(float)));
      if (!output) {
        throw std::runtime_error("failed to write seed-fitter test output");
      }
    }
    check_cuda(cudaStreamDestroy(stream));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
