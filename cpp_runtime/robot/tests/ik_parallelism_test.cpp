// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/inverse_kinematics.h"
#include "curobo_robot_runtime/kinematics.h"
#include "cuda_check.h"
#include "robot_runtime.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
class Buffer {
 public:
  explicit Buffer(size_t count) : count_(count) { storage_.allocate(count * sizeof(T)); }
  T* data() const { return reinterpret_cast<T*>(storage_.ptr()); }
  void upload(const std::vector<T>& values) {
    if (values.size() != count_) throw std::runtime_error("buffer size mismatch");
    CRR_CUDA_CHECK(cudaMemcpy(data(), values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> values(count_);
    CRR_CUDA_CHECK(cudaMemcpy(values.data(), data(), count_ * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
 private:
  size_t count_;
  curobo_robot_runtime::DeviceBuffer storage_;
};

template <typename T>
void append(std::vector<float>& results, const std::vector<T>& values) {
  for (const auto value : values) results.push_back(static_cast<float>(value));
}

std::vector<float> evaluate(const std::string& artifact, bool graph) {
  const auto manifest = curobo_robot_runtime::load_ik_manifest(artifact);
  const int dof = manifest.num_dof;
  const int links = manifest.num_tool_frames;
  curobo_robot_runtime::DeviceBuffer default_position;
  default_position.upload(curobo_robot_runtime::load_tensor_data(
      artifact, manifest.default_joint_position, "default_joint_position"));
  std::vector<float> home(dof);
  CRR_CUDA_CHECK(cudaMemcpy(home.data(), reinterpret_cast<void*>(default_position.ptr()),
                           dof * sizeof(float), cudaMemcpyDeviceToHost));
  curobo::robot::Kinematics fk(artifact);
  Buffer<float> fk_position(links * 3), fk_quaternion(links * 4), jacobian(links * 6 * dof);
  fk.forward(reinterpret_cast<float*>(default_position.ptr()), fk_position.data(),
             fk_quaternion.data(), jacobian.data(), 1, 1);
  const auto home_position = fk_position.download();
  const auto home_quaternion = fk_quaternion.download();
  std::vector<float> results;
  for (const int batch : {1, 2}) {
    for (const int goals : {1, 17, 100, 129, 257}) {
      if (goals > manifest.max_goalset) continue;
      for (const int seeds : {1, 17, 32, 64, 129}) {
        if (seeds > manifest.max_seeds) continue;
        curobo::robot::InverseKinematicsConfig config;
        config.max_batch_size = batch;
        config.max_goalset = goals;
        config.num_seeds = seeds;
        config.lm_iterations = 1;
        config.lbfgs_iterations = 1;
        config.exit_early = false;
        config.check_effort_limits = goals == 17;
        config.use_cuda_graph = graph;
        curobo::robot::InverseKinematics ik(artifact, config);
        Buffer<float> positions(batch * links * goals * 3);
        Buffer<float> quaternions(batch * links * goals * 4);
        Buffer<float> current(batch * dof), seed_positions(batch * seeds * dof);
        std::vector<float> p(batch * links * goals * 3), q(batch * links * goals * 4);
        std::vector<float> start(batch * dof), seed_values(batch * seeds * dof);
        for (int b = 0; b < batch; ++b) {
          for (int j = 0; j < dof; ++j) {
            start[b * dof + j] = home[j] + 0.15F * b;
            for (int s = 0; s < seeds; ++s) {
              // Duplicate seeds exercise stable ties; others require actual refinement.
              seed_values[(b * seeds + s) * dof + j] =
                  home[j] + 0.025F * static_cast<float>((s / 2 + j * b) % 7);
            }
          }
          for (int link = 0; link < links; ++link) {
            for (int g = 0; g < goals; ++g) {
              const int pose = (b * links + link) * goals + g;
              for (int k = 0; k < 3; ++k) {
                // Goal 0 and the last goal tie. Batch 1 is deliberately unreachable.
                p[pose * 3 + k] = home_position[link * 3 + k] + 10.0F * b +
                    ((g == goals - 1) ? 0.0F : 0.002F * g);
              }
              for (int k = 0; k < 4; ++k) q[pose * 4 + k] = home_quaternion[link * 4 + k];
            }
          }
        }
        positions.upload(p);
        quaternions.upload(q);
        current.upload(start);
        seed_positions.upload(seed_values);
        for (const int solutions : {1, seeds}) {
          const int count = batch * solutions;
          Buffer<float> solution(count * dof), position_error(count), orientation_error(count);
          Buffer<float> collision_cost(count);
          Buffer<uint8_t> success(count);
          Buffer<int32_t> goal_index(count * links);
          curobo::robot::IkSolveRequest request;
          request.goal_positions = positions.data();
          request.goal_quaternions = quaternions.data();
          request.current_positions = goals == 1 ? nullptr : current.data();
          request.seed_positions = seed_positions.data();
          request.solution = solution.data();
          request.success = success.data();
          request.position_error = position_error.data();
          request.orientation_error = orientation_error.data();
          request.collision_cost = collision_cost.data();
          request.goalset_index = goal_index.data();
          request.batch_size = batch;
          request.num_goalset = goals;
          request.num_solutions = solutions;
          // Repeat with identical addresses to cover graph replay, not only capture.
          ik.solve(request);
          ik.solve(request);
          CRR_CUDA_CHECK(cudaDeviceSynchronize());
          append(results, solution.download());
          append(results, success.download());
          append(results, goal_index.download());
          append(results, position_error.download());
          append(results, orientation_error.download());
          append(results, collision_cost.download());
        }
      }
    }
  }
  return results;
}

void compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("reference size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1.0e-5F) {
      throw std::runtime_error("IK mismatch at " + std::to_string(i) + ": " +
                               std::to_string(actual[i]) + " vs " + std::to_string(expected[i]));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 4) {
      throw std::runtime_error("usage: ik_parallelism_test ARTIFACT [--write-reference|--reference FILE]");
    }
    const auto result = evaluate(argv[1], true);
    compare(evaluate(argv[1], false), result);
    if (argc == 4) {
      if (std::string(argv[2]) == "--write-reference") {
        std::ofstream output(argv[3]);
        output << std::setprecision(9);
        for (float value : result) output << value << '\n';
        if (!output) throw std::runtime_error("cannot write reference");
      } else if (std::string(argv[2]) == "--reference") {
        std::ifstream input(argv[3]);
        std::vector<float> expected;
        float value;
        while (input >> value) expected.push_back(value);
        compare(result, expected);
      } else {
        throw std::runtime_error("unknown reference option");
      }
    }
    std::cout << "PASS IK batch, goalset ties, top-k, effort checks and graph parity ("
              << result.size() << " values)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
