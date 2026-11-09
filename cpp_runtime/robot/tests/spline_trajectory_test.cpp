// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/spline_trajectory.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace curobo::robot;
namespace {
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
std::vector<float> download(const InterpolatedTrajectory &value) {
  std::vector<float> values(4 * value.sample_count * value.num_dof);
  require(cudaMemcpy(values.data(), value.position, values.size() * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "download failed");
  return values;
}
void dump(std::ofstream &file, const InterpolatedTrajectory &result,
          const std::vector<float> &data) {
  if (!file.is_open()) return;
  file.write(reinterpret_cast<const char *>(&result.sample_count), sizeof(int));
  file.write(reinterpret_cast<const char *>(&result.dt), sizeof(float));
  file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}
}  // namespace
int main(int argc, char **argv) {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    std::ofstream file;
    if (argc > 1) file.open(argv[1], std::ios::binary);
    std::vector<float> points{0, 1, 2, 3, 4, 5, 6};
    float knot_dt = .1F;
    SplineTrajectory host(points.data(), 7, {"joint"}, UniformTiming{&knot_dt}, -1);
    auto retained = host.clone();
    points[0] = 99;
    require(retained.control_points()[0] == 0, "CPU clone aliases source");
    auto spline = retained.to(0);
    auto uniform = spline.interpolate(.03F);
    auto data = download(uniform);
    require(uniform.sample_count == 15 && uniform.dt <= .03F, "bad uniform grid");
    for (int i = 0; i < uniform.sample_count; ++i) {
      require(std::abs(data[i] - (1 + 10 * i * uniform.dt)) < 2e-5F, "bad uniform position");
      require(std::abs(data[uniform.sample_count + i] - 10) < 2e-4F, "bad uniform velocity");
    }
    dump(file, uniform, data);
    auto copy = spline.clone().to(-1);
    require(copy.control_points()[6] == 6 && copy.joint_names()[0] == "joint",
            "GPU clone/transfer failed");
    std::vector<float> timing(32), times(512);
    auto knot = [](int index) { return std::clamp((index - 3) / 29.0F, 0.0F, 1.0F); };
    for (int i = 0; i < 32; ++i) timing[i] = 1 + (knot(i + 1) + knot(i + 2) + knot(i + 3)) / 3;
    for (int i = 0; i < 512; ++i) times[i] = 2 * (std::sqrt(1 + i / 511.0F) - 1);
    SplineTrajectory optimized(retained.control_points(), 7, {"joint"},
                               TimeOptTiming{timing.data(), times.data(), 512}, -1);
    auto gpu = optimized.to(0);
    auto retimed = gpu.interpolate(.01F);
    data = download(retimed);
    for (int i = 0; i < retimed.sample_count; ++i) {
      const float t = i * retimed.dt, s = (1 + t / 2) * (1 + t / 2) - 1;
      require(std::abs(data[i] - (1 + 4 * s)) < 4e-5F, "bad retimed position");
      require(std::abs(data[retimed.sample_count + i] - 4 * (1 + t / 2)) < 4e-5F,
              "bad retimed velocity");
      require(std::abs(data[2 * retimed.sample_count + i] - 2) < 4e-4F, "bad retimed acceleration");
      require(std::abs(data[3 * retimed.sample_count + i]) < 3e-3F, "bad retimed jerk");
    }
    dump(file, retimed, data);
    std::cout << "Uniform, TimeOpt, clone, and transfer checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
