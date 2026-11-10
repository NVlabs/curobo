// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda_check.h"
#include "robot_runtime.h"
#include "trajectory_timeopt_cuda.h"

namespace {
using curobo_robot_runtime::DeviceBuffer;
using curobo::robot::detail::TIMEOPT_CONTROL_COUNT;

template <typename ValueType>
DeviceBuffer upload(const std::vector<ValueType>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(ValueType));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  DeviceBuffer buffer;
  buffer.upload(bytes);
  return buffer;
}

template <typename ValueType>
std::vector<ValueType> download(const DeviceBuffer& buffer, size_t count) {
  std::vector<ValueType> values(count);
  CRR_CUDA_CHECK(cudaMemcpy(values.data(), reinterpret_cast<const void*>(buffer.ptr()),
                           count * sizeof(ValueType), cudaMemcpyDeviceToHost));
  return values;
}

void test_output(int capacity, int dof) {
  constexpr int BATCH = 4;
  const int values = capacity * dof;
  std::vector<float> input(BATCH * values);
  std::vector<float> timing(BATCH * TIMEOPT_CONTROL_COUNT);
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i + 1);
  for (size_t i = 0; i < timing.size(); ++i) timing[i] = static_cast<float>(i + 101);
  auto sample_device = upload(input);
  auto timing_device = upload(timing);
  auto dt = upload(std::vector<float>{0.01F, 0.02F, 0.03F, 0.04F});
  auto duration = upload(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
  auto counts = upload(std::vector<int32_t>{capacity, capacity, capacity, capacity});
  auto selected = upload(std::vector<int32_t>{3});
  auto success = upload(std::vector<uint8_t>{1});
  auto states = upload(std::vector<float>(4 * values, -1.0F));
  auto metadata = upload(std::vector<float>(2, -1.0F));
  auto count = upload(std::vector<int32_t>{-1});
  auto output_timing = upload(std::vector<float>(TIMEOPT_CONTROL_COUNT, -1.0F));
  constexpr int KNOTS = 16;
  std::vector<float> controls(BATCH * KNOTS * dof), boundaries(4 * BATCH * dof);
  for (size_t i = 0; i < controls.size(); ++i) controls[i] = static_cast<float>(i);
  for (int b = 0; b < BATCH; ++b) for (int j = 0; j < dof; ++j) {
    boundaries[b*dof+j] = static_cast<float>(b+j);
    boundaries[BATCH*dof+b*dof+j] = static_cast<float>(b+j+10);
    boundaries[2*BATCH*dof+b*dof+j] = 0.3F;
    boundaries[3*BATCH*dof+b*dof+j] = 0.2F;
  }
  auto control_device = upload(controls), boundary_device = upload(boundaries);
  auto output_spatial = upload(std::vector<float>((KNOTS+6)*dof, -1.0F));
  const auto* sample_pointer = reinterpret_cast<const float*>(sample_device.ptr());
  curobo::robot::TrajectoryTimeOptSamples samples{
      sample_pointer, sample_pointer, sample_pointer, sample_pointer};
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CRR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CRR_CUDA_CHECK(curobo::robot::detail::launch_timeopt_output(
      samples, reinterpret_cast<const float*>(timing_device.ptr()),
      reinterpret_cast<const float*>(dt.ptr()), reinterpret_cast<const float*>(duration.ptr()),
      reinterpret_cast<const int32_t*>(counts.ptr()),
      reinterpret_cast<const int32_t*>(selected.ptr()),
      reinterpret_cast<const uint8_t*>(success.ptr()), capacity, dof,
      reinterpret_cast<float*>(states.ptr()), reinterpret_cast<float*>(metadata.ptr()),
      reinterpret_cast<int32_t*>(count.ptr()), reinterpret_cast<float*>(output_timing.ptr()), stream));
  CRR_CUDA_CHECK(curobo::robot::detail::launch_timeopt_spatial_output(
      reinterpret_cast<const float*>(control_device.ptr()),
      reinterpret_cast<const float*>(boundary_device.ptr()), reinterpret_cast<const float*>(dt.ptr()),
      reinterpret_cast<const int32_t*>(selected.ptr()), reinterpret_cast<const uint8_t*>(success.ptr()),
      BATCH, KNOTS, dof, reinterpret_cast<float*>(output_spatial.ptr()), stream));
  CRR_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CRR_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  for (const int32_t seed : {3, 1, -1}) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(selected.ptr()), &seed,
                                  sizeof(seed), cudaMemcpyHostToDevice, stream));
    CRR_CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto actual_timing = download<float>(output_timing, TIMEOPT_CONTROL_COUNT);
    const auto actual_states = download<float>(states, 4 * values);
    const auto actual_metadata = download<float>(metadata, 2);
    const auto actual_count = download<int32_t>(count, 1);
    const auto spatial = download<float>(output_spatial, (KNOTS+6)*dof);
    const float velocity_coefficient[4] = {-1.0F,0.0F,1.0F,2.0F};
    const float acceleration_coefficient[4] = {1.0F/3.0F,-1.0F/6.0F,1.0F/3.0F,11.0F/6.0F};
    for (int k = 0; k < KNOTS+6; ++k) for (int j = 0; j < dof; ++j) {
      float expected = 0.0F;
      if (seed >= 0) {
        const float interval = 0.01F * (seed+1);
        expected = k < 4 ? static_cast<float>(seed+j) + velocity_coefficient[k]*.3F*interval +
            acceleration_coefficient[k]*.2F*interval*interval
            : (k >= KNOTS+3 ? static_cast<float>(seed+j+10) : controls[(seed*KNOTS+k-4)*dof+j]);
      }
      if (std::abs(spatial[k*dof+j]-expected) > 1e-5F)
        throw std::runtime_error("complete spatial controls did not follow native boundaries and seed");
    }
    for (int i = 0; i < TIMEOPT_CONTROL_COUNT; ++i) {
      if (actual_timing[i] != (seed < 0 ? 0.0F : timing[seed * TIMEOPT_CONTROL_COUNT + i])) {
        throw std::runtime_error("timing controls did not follow the selected seed");
      }
    }
    for (int i = 0; i < 4 * values; ++i) {
      if (actual_states[i] != (seed < 0 ? 0.0F : input[seed * values + i % values])) {
        throw std::runtime_error("sample states and timing controls selected different seeds");
      }
    }
    if (actual_count[0] != (seed < 0 ? 0 : capacity) ||
        actual_metadata[1] != (seed < 0 ? 0.0F : static_cast<float>(seed + 1))) {
      throw std::runtime_error("timeopt output metadata did not follow the selected seed");
    }
  }
  const int32_t seed = 2;
  const uint8_t rejected = 0;
  CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(selected.ptr()), &seed,
                                sizeof(seed), cudaMemcpyHostToDevice, stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(success.ptr()), &rejected,
                                sizeof(rejected), cudaMemcpyHostToDevice, stream));
  CRR_CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  for (const float value : download<float>(output_timing, TIMEOPT_CONTROL_COUNT)) {
    if (value != 0.0F) throw std::runtime_error("rejected output retained timing controls");
  }
  CRR_CUDA_CHECK(cudaGraphExecDestroy(executable));
  CRR_CUDA_CHECK(cudaGraphDestroy(graph));
  CRR_CUDA_CHECK(cudaStreamDestroy(stream));
}
}  // namespace

int main() {
  try {
    test_output(2, 1);
    test_output(81, 7);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
