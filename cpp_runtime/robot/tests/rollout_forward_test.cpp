// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rollout_runtime.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void upload(curobo_robot_runtime::DeviceBuffer& buffer, const std::vector<float>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  buffer.upload(bytes);
}

void compare(const std::string& artifact, int states, bool verify_no_cache) {
  const auto robot = curobo_robot_runtime::load_manifest(artifact);
  curobo::robot::detail::RolloutRuntime rollout(
      artifact, 0, robot, curobo::robot::detail::RolloutProfile::TRAJOPT);
  const auto count = static_cast<size_t>(states) * robot.num_dof;
  std::vector<float> q(count), v(count), a(count);
  for (size_t i = 0; i < count; ++i) {
    q[i] = 0.4F * std::sin(0.013F * i);
    v[i] = 0.2F * std::cos(0.023F * i);
    a[i] = 0.3F * std::sin(0.031F * i);
  }
  curobo_robot_runtime::DeviceBuffer position, velocity, acceleration, expected, actual, cache;
  upload(position, q);
  upload(velocity, v);
  upload(acceleration, a);
  expected.allocate(count * sizeof(float));
  actual.allocate(count * sizeof(float));
  cache.allocate(static_cast<size_t>(states) * rollout.num_links() *
                 rollout.cache_floats_per_link() * sizeof(float));
  curobo_robot_runtime::DeviceBuffer forward_only_cache;
  forward_only_cache.allocate(cache.bytes());
  std::vector<float> sentinel(cache.bytes() / sizeof(float), -12345.0F);
  upload(forward_only_cache, sentinel);
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto run = [&]() {
    rollout.launch_inverse_dynamics(position.ptr(), velocity.ptr(), acceleration.ptr(),
        expected.ptr(), cache.ptr(), states, reinterpret_cast<CUstream>(stream));
    rollout.launch_inverse_dynamics(position.ptr(), velocity.ptr(), acceleration.ptr(),
        actual.ptr(), forward_only_cache.ptr(), states, reinterpret_cast<CUstream>(stream), false);
  };
  run();
  check(cudaStreamSynchronize(stream));
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  run();
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  for (const float mass : {0.0F, 3.0F, 1.0F}) {
    curobo::robot::LinkInertialUpdate update;
    update.link_name = "attached_object";
    update.mass = mass;
    rollout.update_link_inertial(update);
    for (int replay = 0; replay < 2; ++replay) {
      check(cudaGraphLaunch(executable, stream));
      check(cudaStreamSynchronize(stream));
      std::vector<float> reference(count), result(count);
      check(cudaMemcpy(reference.data(), reinterpret_cast<void*>(expected.ptr()),
                       count * sizeof(float), cudaMemcpyDeviceToHost));
      check(cudaMemcpy(result.data(), reinterpret_cast<void*>(actual.ptr()),
                       count * sizeof(float), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(result[i]) ||
            std::abs(result[i] - reference[i]) > 1e-4F + 1e-4F * std::abs(reference[i])) {
          throw std::runtime_error("forward-only RNEA torque mismatch");
        }
      }
      if (verify_no_cache) {
        std::vector<float> contents(sentinel.size());
        check(cudaMemcpy(contents.data(), reinterpret_cast<void*>(forward_only_cache.ptr()),
                         forward_only_cache.bytes(), cudaMemcpyDeviceToHost));
        if (contents != sentinel) throw std::runtime_error("forward-only RNEA wrote its cache");
      }
    }
  }
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) throw std::runtime_error("usage: rollout_forward_test ARTIFACT [--no-cache]");
    check(cudaSetDevice(0));
    for (int states : {1, 324, 972, 1024, 1025, 4000, 4096, 4097, 6912, 8192, 8193}) {
      compare(argv[1], states, argc == 3);
    }
    std::cout << "forward-only RNEA graph replay and payload parity passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
