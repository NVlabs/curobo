// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/trajectory_timeopt.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "cuda_check.h"
#include "robot_runtime.h"

using curobo_robot_runtime::DeviceBuffer;

static DeviceBuffer load(const std::filesystem::path &path, size_t count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || static_cast<size_t>(in.tellg()) != count * sizeof(float)) {
    throw std::runtime_error("invalid fixture: " + path.string());
  }
  std::vector<uint8_t> bytes(count * sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  DeviceBuffer buffer;
  buffer.upload(bytes);
  return buffer;
}

template <typename Value>
static void save(const std::filesystem::path &path, const Value *pointer, size_t count) {
  std::vector<Value> values(count);
  CRR_CUDA_CHECK(cudaMemcpy(values.data(), pointer, count * sizeof(Value), cudaMemcpyDeviceToHost));
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(values.data()), count * sizeof(Value));
}

int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument("usage: timeopt_test ARTIFACT FIXTURE KNOTS REFERENCE SAMPLES INTERVALS");
    curobo::robot::TrajectoryTimeOptConfig config;
    config.num_knots = std::stoi(argv[3]);
    config.reference_timing = std::stoi(argv[4]) != 0;
    config.sample_count = std::stoi(argv[5]);
    config.num_intervals = std::stoi(argv[6]);
    const std::filesystem::path root(argv[2]);
    cudaStream_t stream = nullptr;
    CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (bool graph : {false, true}) {
      config.use_cuda_graph = graph;
      curobo::robot::TrajectoryTimeOpt timeopt(argv[1], config);
      const int dof = timeopt.num_dof(), batch = config.num_seeds;
      auto knots = load(root / "knots.bin", batch * config.num_knots * dof);
      auto start = load(root / "start.bin", batch * dof);
      auto goal = load(root / "goal.bin", batch * dof);
      auto dt = load(root / "dt.bin", batch);
      auto velocity = load(root / "velocity.bin", batch * dof);
      auto acceleration = load(root / "acceleration.bin", batch * dof);
      auto jerk = load(root / "jerk.bin", batch * dof);
      auto times = load(root / "times.bin", batch * config.sample_count);
      curobo::robot::TrajectoryTimeOptRequest request;
      request.knots = reinterpret_cast<float *>(knots.ptr());
      request.start_position = reinterpret_cast<float *>(start.ptr());
      request.goal_position = reinterpret_cast<float *>(goal.ptr());
      request.stream = stream;
      if (config.reference_timing) {
        request.knot_dt = reinterpret_cast<float *>(dt.ptr());
        request.start_velocity = reinterpret_cast<float *>(velocity.ptr());
        request.start_acceleration = reinterpret_cast<float *>(acceleration.ptr());
        request.start_jerk = reinterpret_cast<float *>(jerk.ptr());
      }
      bool rejected = false;
      try {
        timeopt.sample(reinterpret_cast<float *>(times.ptr()), stream);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("sample before timeopt was accepted");
      const auto invalid = curobo::robot::TrajectoryTimeOptRequest{};
      rejected = false;
      try {
        timeopt.timeopt(invalid);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("null inputs were accepted");
      curobo::robot::TrajectoryTimeOptResult result;
      curobo::robot::TrajectoryTimeOptSamples samples;
      for (int replay = 0; replay < 3; ++replay) {
        result = timeopt.timeopt(request);
        samples = timeopt.sample(reinterpret_cast<float *>(times.ptr()), stream);
      }
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      // Capture the complete public calls, including their input copies.
      cudaGraph_t outer = nullptr;
      cudaGraphExec_t executable = nullptr;
      CRR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      timeopt.timeopt(request);
      timeopt.sample(reinterpret_cast<float *>(times.ptr()), stream);
      CRR_CUDA_CHECK(cudaStreamEndCapture(stream, &outer));
      CRR_CUDA_CHECK(cudaGraphInstantiate(&executable, outer, 0));
      CRR_CUDA_CHECK(cudaGraphLaunch(executable, stream));
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      CRR_CUDA_CHECK(cudaGraphExecDestroy(executable));
      CRR_CUDA_CHECK(cudaGraphDestroy(outer));
      // A new input address must be copied on replay; graph nodes must not
      // retain the caller's original pointer. Also exercise default-stream replay.
      auto replacement = load(root / "knots.bin", batch * config.num_knots * dof);
      CRR_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(knots.ptr()), 0,
                                     batch * config.num_knots * dof * sizeof(float), stream));
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      request.knots = reinterpret_cast<float *>(replacement.ptr());
      request.stream = nullptr;
      result = timeopt.timeopt(request);
      samples = timeopt.sample(reinterpret_cast<float *>(times.ptr()));
      CRR_CUDA_CHECK(cudaDeviceSynchronize());
      const auto prefix = graph ? "graph_" : "eager_";
      const auto output = [&](const char *name) {
        return root / (std::string(prefix) + name + ".bin");
      };
      save(output("timing"), result.timing_controls, batch * 32);
      save(output("time"), result.time, batch * (config.num_intervals + 1));
      save(output("duration"), result.duration, batch);
      save(output("uniform"), result.uniform_duration, batch);
      save(output("success"), result.success, batch);
      save(output("status"), result.status, batch);
      save(output("states"), result.position, 4 * batch * (config.num_intervals + 1) * dof);
      save(output("torque"), result.torque, batch * (config.num_intervals + 1) * dof);
      save(output("samples"), samples.position, 4 * batch * config.sample_count * dof);
    }
    CRR_CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "timeopt fixtures, graph replay, and outer capture passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
