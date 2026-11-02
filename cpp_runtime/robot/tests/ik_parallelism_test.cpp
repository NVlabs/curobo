// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/inverse_kinematics.h"
#include "curobo_robot_runtime/kinematics.h"
#include "cuda_check.h"
#include "robot_runtime.h"

#include <algorithm>
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
        // Exercise batched top-k LM compaction alongside the equal-count path.
        const int lm_seeds = seeds == 17 ? std::min(32, manifest.max_seeds) : seeds;
        config.lm_num_seeds = lm_seeds;
        // Multiple steps also exercise retention when the selected goal changes.
        config.lm_iterations = 3;
        config.lbfgs_iterations = 1;
        config.exit_early = false;
        config.check_effort_limits = goals == 17;
        config.use_cuda_graph = graph;
        curobo::robot::InverseKinematics ik(artifact, config);
        Buffer<float> positions(batch * links * goals * 3);
        Buffer<float> quaternions(batch * links * goals * 4);
        Buffer<float> current(batch * dof), seed_positions(batch * lm_seeds * dof);
        std::vector<float> p(batch * links * goals * 3), q(batch * links * goals * 4);
        std::vector<float> start(batch * dof), seed_values(batch * lm_seeds * dof);
        for (int b = 0; b < batch; ++b) {
          for (int j = 0; j < dof; ++j) {
            start[b * dof + j] = home[j] + 0.15F * b;
            for (int s = 0; s < lm_seeds; ++s) {
              // Duplicate seeds exercise stable ties; others require actual refinement.
              seed_values[(b * lm_seeds + s) * dof + j] =
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

// Expose every retained LM seed before L-BFGS can hide a state-selection error.
std::vector<float> evaluate_lm(const std::string& artifact, bool graph) {
  const auto manifest = curobo_robot_runtime::load_ik_manifest(artifact);
  const int dof = manifest.num_dof;
  const int links = manifest.num_tool_frames;
  const int batch = 2;
  const int seeds = std::min(17, manifest.max_seeds);
  const int count = batch * seeds;
  curobo_robot_runtime::DeviceBuffer home_buffer;
  home_buffer.upload(curobo_robot_runtime::load_tensor_data(
      artifact, manifest.default_joint_position, "default_joint_position"));
  std::vector<float> home(dof);
  CRR_CUDA_CHECK(cudaMemcpy(home.data(), reinterpret_cast<void*>(home_buffer.ptr()),
                           dof * sizeof(float), cudaMemcpyDeviceToHost));
  curobo::robot::Kinematics fk(artifact);
  Buffer<float> positions(batch * links * 3), quaternions(batch * links * 4);
  Buffer<float> raw_jacobian(batch * links * 6 * dof), targets(batch * dof);
  Buffer<float> initial(count * dof), solution(count * dof);
  Buffer<float> position_error(count), orientation_error(count), collision_cost(count);
  Buffer<uint8_t> success(count);
  std::vector<float> results;
  // Zero requests the runtime default of 16 iterations.
  for (const int iterations : {0, 1, 2, 16}) {
    for (const bool reject_all : {false, true}) {
      curobo::robot::InverseKinematicsConfig config;
      config.max_batch_size = batch;
      config.num_seeds = seeds;
      config.lm_iterations = iterations;
      config.use_cuda_graph = graph;
      config.check_self_collision = false;
      // Make every in-bounds seed valid so the public solve returns after LM.
      config.position_tolerance = 100.0F;
      config.orientation_tolerance = 100.0F;
      config.minimum_trust_ratio = reject_all ? 1.0e30F : 1.0e-5F;
      curobo::robot::InverseKinematics ik(artifact, config);
      curobo::robot::IkSolveRequest request;
      request.goal_positions = positions.data();
      request.goal_quaternions = quaternions.data();
      request.seed_positions = initial.data();
      request.solution = solution.data();
      request.success = success.data();
      request.position_error = position_error.data();
      request.orientation_error = orientation_error.data();
      request.collision_cost = collision_cost.data();
      request.batch_size = batch;
      request.num_solutions = seeds;
      for (int replay = 0; replay < 2; ++replay) {
        std::vector<float> target(batch * dof), seed_values(count * dof);
        for (int b = 0; b < batch; ++b) {
          for (int j = 0; j < dof; ++j) {
            target[b * dof + j] = home[j] + 0.04F * (b + replay + 1);
            for (int seed = 0; seed < seeds; ++seed) {
              seed_values[(b * seeds + seed) * dof + j] =
                  home[j] - 0.01F * (1 + (seed + j + replay) % 7);
            }
          }
        }
        targets.upload(target);
        initial.upload(seed_values);
        fk.forward(targets.data(), positions.data(), quaternions.data(),
                   raw_jacobian.data(), batch, 1);
        ik.solve(request);
        const auto valid = success.download();
        if (std::any_of(valid.begin(), valid.end(), [](uint8_t v) { return v != 1U; })) {
          throw std::runtime_error("LM retention test did not return valid seeds");
        }
        const auto selected = solution.download();
        if (reject_all) {
          // Ranking may reorder seeds, but rejected steps must retain each input.
          for (int b = 0; b < batch; ++b) {
            for (int out = 0; out < seeds; ++out) {
              bool found = false;
              for (int in = 0; in < seeds; ++in) {
                bool equal = true;
                for (int j = 0; j < dof; ++j) {
                  equal = equal && selected[(b * seeds + out) * dof + j] ==
                                       seed_values[(b * seeds + in) * dof + j];
                }
                found = found || equal;
              }
              if (!found) throw std::runtime_error("rejected LM step changed a seed");
            }
          }
        }
        append(results, selected);
        append(results, valid);
        append(results, position_error.download());
        append(results, orientation_error.download());
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
    auto result = evaluate(argv[1], true);
    compare(evaluate(argv[1], false), result);
    const auto lm_result = evaluate_lm(argv[1], true);
    compare(evaluate_lm(argv[1], false), lm_result);
    append(result, lm_result);
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
    std::cout << "PASS IK batch, goalset ties, top-k, effort, LM retention and graph parity ("
              << result.size() << " values)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
