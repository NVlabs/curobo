// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda_check.h"
#include "mpc_cuda.h"
#include "robot_runtime.h"

namespace {
using curobo_robot_runtime::DeviceBuffer;
float *ptr(DeviceBuffer &buffer) { return reinterpret_cast<float *>(buffer.ptr()); }
const int *shift_ptr(DeviceBuffer &buffer) { return reinterpret_cast<const int *>(buffer.ptr()); }
void upload_shift(DeviceBuffer &buffer, int value) {
  buffer.allocate(sizeof(int));
  CRR_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(buffer.ptr()), &value, sizeof(int),
                            cudaMemcpyHostToDevice));
}
void upload(DeviceBuffer &buffer, const std::vector<float> &values) {
  if (buffer.bytes() != values.size() * sizeof(float))
    buffer.allocate(values.size() * sizeof(float));
  CRR_CUDA_CHECK(cudaMemcpy(ptr(buffer), values.data(), buffer.bytes(), cudaMemcpyHostToDevice));
}
void check(DeviceBuffer &buffer, const std::vector<float> &expected) {
  std::vector<float> actual(expected.size());
  CRR_CUDA_CHECK(cudaMemcpy(actual.data(), ptr(buffer), buffer.bytes(), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1.0e-5F)
      throw std::runtime_error("MPC shift mismatch at " + std::to_string(i));
  }
}

void check_path_acceptance() {
  DeviceBuffer position, rotation, current, goal, quaternion, direction, frame, enforce;
  upload(position, {1, 2, 3, 4, 5, 6});
  upload(rotation, {0.01F, 0.02F, 0.03F, 0.04F, 0.05F, 0.06F});
  upload(goal, {2, 3, 4, -1, 2, 3});
  upload(current, {2.003F, 3.004F, 4.03F, -0.97F, 2.003F, 3.004F, 2, 3, 4.10F, -0.90F, 2, 3,
                   2.006F, 3, 4.3F, -0.7F, 2.007F, 3});
  const float half_sqrt = std::sqrt(0.5F);
  upload(quaternion, {1, 0, 0, 0, half_sqrt, 0, half_sqrt, 0});
  upload(direction, {0, 0, 1});
  frame.allocate(2 * sizeof(int32_t));
  const int32_t goal_frames[2]{1, 1};
  CRR_CUDA_CHECK(cudaMemcpy(ptr(frame), goal_frames, frame.bytes(), cudaMemcpyHostToDevice));
  enforce.allocate(sizeof(uint8_t));
  CRR_CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(enforce.ptr()), 1, 1));
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CRR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CRR_CUDA_CHECK(curobo::robot::detail::launch_mpc_pose_acceptance(
      ptr(position), ptr(rotation), ptr(current), ptr(goal), ptr(quaternion), ptr(direction),
      reinterpret_cast<const int32_t *>(frame.ptr()),
      reinterpret_cast<const uint8_t *>(enforce.ptr()), 3, 2, stream));
  CRR_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CRR_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  auto replay = [&] {
    CRR_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  replay();
  // Each tool's goal-frame Z axis has different world orientation. Remaining
  // axial distance does not contribute, but transverse/rotation errors remain.
  check(position, {0.005F, 0.005F, 0, 0, 0.006F, 0.007F});
  check(rotation, {0.01F, 0.02F, 0.03F, 0.04F, 0.05F, 0.06F});
  // Replacing previously weighted distances cannot affect geometric acceptance.
  upload(position, {100, 100, 100, 100, 100, 100});
  replay();
  check(position, {0.005F, 0.005F, 0, 0, 0.006F, 0.007F});
  CRR_CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(frame.ptr()), 0, frame.bytes()));
  replay();
  check(position, {0.005F, std::hypot(0.03F, 0.003F), 0, 0.10F, 0.006F, std::hypot(0.3F, 0.007F)});
  // Direction updates stay GPU-resident and replay the same captured kernel.
  upload(direction, {1, 0, 0});
  replay();
  check(position, {std::hypot(0.004F, 0.03F), 0.005F, 0.10F, 0, 0.3F, 0.007F});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  upload(current, {nan, 0, 0, -1, 2, 3, 2, 3, 4, -1, 2, 3, 2, 3, 4, -1, 2, 3});
  replay();
  float invalid_distance = 0.F;
  CRR_CUDA_CHECK(
      cudaMemcpy(&invalid_distance, ptr(position), sizeof(float), cudaMemcpyDeviceToHost));
  if (!std::isinf(invalid_distance) || invalid_distance < 0.F)
    throw std::runtime_error("Nonfinite path position was not rejected");
  CRR_CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(enforce.ptr()), 0, 1));
  replay();
  check(position, {0, 0, 0, 0, 0, 0});
  check(rotation, {0, 0, 0, 0, 0, 0});
  CRR_CUDA_CHECK(cudaGraphExecDestroy(executable));
  CRR_CUDA_CHECK(cudaGraphDestroy(graph));
  CRR_CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "MPC unweighted path acceptance and goal-frame graph replay passed\n";
}
void check_batch_kernels() {
  using namespace curobo::robot::detail;
  DeviceBuffer seed, batch, q, v, a, j, limits, world, self, spheres, violation;
  upload(seed, {1, 2, 3});
  upload(batch, std::vector<float>(12, 0.F));
  CRR_CUDA_CHECK(launch_mpc_broadcast(ptr(seed), ptr(batch), 3, 4, nullptr));
  check(batch, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  upload(q, {0, 0, 0, 0, 0, 0, nan, 0});
  upload(v, std::vector<float>(8, 0.F));
  upload(a, {0, 0, 0, 0, 0, 2, 0, 0});
  upload(j, std::vector<float>(8, 0.F));
  upload(limits, {-1, 1});
  upload(world, {0, 0, 1, 0, 0, 0, 0, 0});
  upload(self, std::vector<float>(8, 0.F));
  // Large changes BETWEEN lanes must not become within-trajectory sweep errors.
  upload(spheres, {0, 0, 0, 1, 0, 0, 0, 1, 100, 0, 0, 1, 100, 0, 0, 1,
                   200, 0, 0, 1, 200, 0, 0, 1, 300, 0, 0, 1, 300, 0, 0, 1});
  violation.allocate(4 * sizeof(int));
  violation.clear();
  CRR_CUDA_CHECK(launch_mpc_validate(ptr(q), ptr(v), ptr(a), ptr(j), ptr(limits),
      ptr(limits), ptr(limits), ptr(limits), ptr(world), ptr(self), ptr(spheres),
      1, .01F, reinterpret_cast<int *>(violation.ptr()), 2, 1, nullptr, 4));
  std::array<int, 4> flags{};
  CRR_CUDA_CHECK(cudaMemcpy(flags.data(), reinterpret_cast<void *>(violation.ptr()),
                            sizeof(flags), cudaMemcpyDeviceToHost));
  if (flags != std::array<int, 4>{0, 16, 4, 1})
    throw std::runtime_error("MPC candidate validation leaked between lanes");
  DeviceBuffer success, cost, dt, rank;
  success.allocate(4);
  CRR_CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(success.ptr()), 1, 4));
  upload(cost, {100, 1, nan, 0});
  upload(dt, {.025F, .00625F, .0125F, .02F});
  upload(rank, {0, 0, 0, 0});
  CRR_CUDA_CHECK(launch_mpc_rank(reinterpret_cast<uint8_t *>(success.ptr()), ptr(cost),
                                ptr(dt), ptr(rank), 4, nullptr));
  std::array<float, 4> ranks;
  std::array<uint8_t, 4> valid;
  CRR_CUDA_CHECK(cudaMemcpy(ranks.data(), ptr(rank), sizeof(ranks), cudaMemcpyDeviceToHost));
  CRR_CUDA_CHECK(cudaMemcpy(valid.data(), reinterpret_cast<void *>(success.ptr()),
                            sizeof(valid), cudaMemcpyDeviceToHost));
  if (valid != std::array<uint8_t, 4>{1, 1, 0, 1} || !std::isinf(ranks[2]) ||
      ranks[0] != .025F || ranks[1] != .00625F || ranks[3] != .02F)
    throw std::runtime_error("MPC dt ranking used raw cost or retained NaN");
  std::cout << "MPC four-lane broadcast, isolated validation and dt ranking passed\n";
}
}  // namespace

int main() {
  try {
    DeviceBuffer input, output, goal, shift;
    upload(input, {1, 2, 3, 4, 5, 6});
    upload(output, {0, 0, 0, 0, 0, 0});
    upload(goal, {7, 8});
    upload_shift(shift, 1);
    CRR_CUDA_CHECK(curobo::robot::detail::launch_mpc_shift(ptr(input), ptr(output), ptr(goal),
                                                           shift_ptr(shift), 1, 3, 2, nullptr));
    check(output, {3, 4, 5, 6, 7, 8});
    CRR_CUDA_CHECK(curobo::robot::detail::launch_mpc_shift(ptr(input), ptr(output), nullptr,
                                                           shift_ptr(shift), 1, 3, 2, nullptr));
    check(output, {3, 4, 5, 6, 0, 0});
    upload_shift(shift, 100);
    CRR_CUDA_CHECK(curobo::robot::detail::launch_mpc_shift(ptr(input), ptr(output), ptr(goal),
                                                           shift_ptr(shift), 1, 3, 2, nullptr));
    check(output, {7, 8, 7, 8, 7, 8});

    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CRR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CRR_CUDA_CHECK(curobo::robot::detail::launch_mpc_shift(ptr(input), ptr(output), ptr(goal),
                                                           shift_ptr(shift), 1, 3, 2, stream));
    CRR_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CRR_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    for (int offset : {0, 1}) {
      CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(shift.ptr()), &offset, sizeof(int),
                                     cudaMemcpyHostToDevice, stream));
      CRR_CUDA_CHECK(cudaGraphLaunch(executable, stream));
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      check(output, offset == 0 ? std::vector<float>{1, 2, 3, 4, 5, 6}
                                : std::vector<float>{3, 4, 5, 6, 7, 8});
    }
    CRR_CUDA_CHECK(cudaGraphExecDestroy(executable));
    CRR_CUDA_CHECK(cudaGraphDestroy(graph));
    CRR_CUDA_CHECK(cudaStreamDestroy(stream));
    DeviceBuffer implicit;
    implicit.allocate(sizeof(uint8_t));
    for (int pinned : {1, 0, 1}) {
      upload(goal, {7, 8});
      CRR_CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(implicit.ptr()), pinned, 1));
      CRR_CUDA_CHECK(curobo::robot::detail::launch_mpc_tail(
          ptr(input), ptr(goal), reinterpret_cast<const uint8_t *>(implicit.ptr()), 3, 2,
          nullptr));
      check(goal, pinned ? std::vector<float>{7, 8} : std::vector<float>{5, 6});
    }
    std::cout << "MPC integer shift and tail padding checks passed\n";
    check_path_acceptance();
    check_batch_kernels();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
