// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "motion_planner_cuda.h"
#include "robot_runtime.h"
#include "cuda_check.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
template <typename Value>
class Buffer {
 public:
  explicit Buffer(const std::vector<Value>& values) { upload(values); }
  void upload(const std::vector<Value>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(Value));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    buffer_.upload(bytes);
  }
  Value* data() const { return reinterpret_cast<Value*>(buffer_.ptr()); }
  std::vector<Value> read() const {
    std::vector<Value> values(buffer_.bytes() / sizeof(Value));
    CRR_CUDA_CHECK(cudaMemcpy(values.data(), data(), buffer_.bytes(), cudaMemcpyDeviceToHost));
    return values;
  }
 private:
  curobo_robot_runtime::DeviceBuffer buffer_;
};

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
}  // namespace

int main() {
  try {
    Buffer<float> positions({99, 99, 99, 1, 2, 3, 99, 99, 99, 4, 5, 6});
    // Non-unit quaternion: a 90-degree rotation around Z for both tools.
    Buffer<float> quaternions({2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2});
    Buffer<int32_t> indices({0, 0, 1, 1}), selected_index({-1});
    Buffer<uint8_t> valid({0});
    Buffer<float> selected(std::vector<float>(6)), approach(std::vector<float>(6));
    Buffer<float> lift(std::vector<float>(6)), selected_q(std::vector<float>(8));
    Buffer<float> approach_q(std::vector<float>(8)), lift_q(std::vector<float>(8));
    curobo::robot::detail::GraspPosePreparation request{
        positions.data(), quaternions.data(), indices.data(), 1, 2, 2,
        0, 2, -0.15F, 0.15F, true, false,
        selected.data(), selected_q.data(), approach.data(), approach_q.data(),
        lift.data(), lift_q.data(), selected_index.data(), valid.data()};
    const auto prepare = [&]() {
      CRR_CUDA_CHECK(curobo::robot::detail::launch_prepare_grasp_poses(request, nullptr));
      CRR_CUDA_CHECK(cudaDeviceSynchronize());
      return valid.read().at(0) != 0;
    };
    require(prepare() && selected_index.read().at(0) == 1, "selected goal was not gathered");
    const auto p = selected.read(), a = approach.read(), l = lift.read();
    for (int tool = 0; tool < 2; ++tool) {
      for (int axis = 0; axis < 3; ++axis) {
        const int index = tool * 3 + axis;
        require(std::abs(p.at(index) - float(index + 1)) < 1.0e-6F, "wrong selected pose");
        require(std::abs(a.at(index) - p.at(index) + (axis == 1 ? 0.15F : 0)) < 1.0e-6F,
                "tool-frame approach axis was not rotated");
        require(std::abs(l.at(index) - p.at(index) - (axis == 2 ? 0.15F : 0)) < 1.0e-6F,
                "base-frame lift axis was rotated");
      }
    }
    require(selected_q.read() == approach_q.read() && selected_q.read() == lift_q.read(),
            "phase orientations differ");
    indices.upload({0, 0, 2, 2});
    request.goal_indices = indices.data();
    require(!prepare() && selected_index.read().at(0) == -1, "invalid goal index accepted");
    indices.upload({0, 0, 1, 1});
    request.goal_indices = indices.data();
    quaternions.upload(std::vector<float>(16, 0));
    request.quaternions = quaternions.data();
    require(!prepare(), "zero quaternion accepted");
    quaternions.upload(std::vector<float>(16, std::numeric_limits<float>::quiet_NaN()));
    request.quaternions = quaternions.data();
    require(!prepare(), "nonfinite quaternion accepted");
    std::cout << "PASS device grasp pose selection, tool/base offsets and invalid inputs\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
