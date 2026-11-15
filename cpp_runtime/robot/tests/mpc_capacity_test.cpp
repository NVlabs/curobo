// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/mpc.h"
#include "curobo_robot_runtime/scene.h"
#include "robot_runtime.h"
#include "trajectory_optimizer_runtime.h"

namespace {
using namespace curobo::robot;
using curobo_robot_runtime::DeviceBuffer;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

float *data(DeviceBuffer &buffer) { return reinterpret_cast<float *>(buffer.ptr()); }

void check(const std::string &artifact) {
  const auto manifest = curobo_robot_runtime::load_trajectory_optimization_manifest(artifact);
  require(manifest.num_dof == 7, "capacity fixture requires Franka");
  SceneConfig scene_config;
  scene_config.max_cuboids = 1;
  scene_config.max_capsules = scene_config.max_cylinders = scene_config.max_meshes = 0;
  scene_config.max_voxel_grids = 0;
  Scene scene(scene_config);
  Cuboid obstacle;
  obstacle.name = "far_obstacle";
  obstacle.pose.position = {10.0F, 10.0F, 10.0F};
  obstacle.dimensions = {0.1F, 0.1F, 0.1F};
  scene.add_cuboid(obstacle);
  MotionPlanningConfig config;
  config.trajectory_optimization.num_knots = manifest.num_knots;
  config.trajectory_optimization.num_seeds = 4;
  config.rrt.max_plans = 4;
  MotionPlanner planner(artifact, config, &scene);
  const std::vector<float> holding{0, -1.3F, 0, -2.5F, 0, 1.5F, 0.8F};
  DeviceBuffer position, zero, seed;
  position.allocate(7 * sizeof(float));
  zero.allocate(7 * sizeof(float));
  zero.clear();
  CRR_CUDA_CHECK(
      cudaMemcpy(data(position), holding.data(), position.bytes(), cudaMemcpyHostToDevice));
  std::vector<float> controls;
  for (int k = 0; k < manifest.num_knots; ++k)
    controls.insert(controls.end(), holding.begin(), holding.end());
  seed.allocate(controls.size() * sizeof(float));
  CRR_CUDA_CHECK(cudaMemcpy(data(seed), controls.data(), seed.bytes(), cudaMemcpyHostToDevice));

  for (float dt : {0.005F, 0.025F, 0.05F}) {
    MpcConfig mpc_config;
    mpc_config.trajectory_dt = dt;
    MpcController mpc(planner, mpc_config);
    mpc.reset(data(position), data(seed));
    for (int replay = 0; replay < 2; ++replay) {
      const auto result = mpc.step(data(position), data(zero), data(zero), data(zero), 0);
      require(result.feasible, "holding MPC failed with independent validation capacity");
    }
    require(mpc.graph_capture_count() == 1, "MPC validation recaptured its graph");
    const auto before = mpc.candidate_dts();
    bool rejected = false;
    try {
      mpc.update_dt(0.1F);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected && mpc.candidate_dts() == before, "rejected timing update changed MPC state");
    mpc.update_dt(0.005F);
    mpc.reset(data(position), data(seed));
    require(mpc.step(data(position), data(zero), data(zero), data(zero), 0).feasible,
            "supported timing update failed");
    require(mpc.graph_capture_count() == 1, "supported timing update recaptured MPC");
  }

  // Exercise validation beyond the exported capacity and seed indexing near
  // the end of the larger buffer, using original pairs and repeated calls.
  const int capacity = manifest.interpolation_buffer_size + 97;
  auto optimizer_config = config.trajectory_optimization;
  optimizer_config.enable_timeopt = false;
  detail::TrajectoryOptimizerRuntime optimizer(artifact, optimizer_config, &scene, nullptr,
                                               capacity, 0.004F);
  std::vector<float> samples;
  for (int i = 0; i < 4 * capacity; ++i)
    samples.insert(samples.end(), holding.begin(), holding.end());
  samples[(3 * capacity - 1) * 7] = 100.0F;
  DeviceBuffer sampled_position, sampled_zero;
  sampled_position.allocate(samples.size() * sizeof(float));
  sampled_zero.allocate(samples.size() * sizeof(float));
  sampled_zero.clear();
  CRR_CUDA_CHECK(cudaMemcpy(data(sampled_position), samples.data(), sampled_position.bytes(),
                            cudaMemcpyHostToDevice));
  TrajectoryValidationRequest request{data(sampled_position), data(sampled_zero),
                                      data(sampled_zero), data(sampled_zero), capacity};
  for (int replay = 0; replay < 2; ++replay) {
    const auto *violations = optimizer.validate(request);
    int actual[4]{};
    CRR_CUDA_CHECK(cudaMemcpy(actual, violations, sizeof(actual), cudaMemcpyDeviceToHost));
    require(actual[0] == 0 && actual[1] == 0 && actual[2] != 0 && actual[3] == 0,
            "dynamic validation attributed a violation to the wrong seed");
  }
  std::cout << "MPC fixed-timing capacity and replay passed for " << manifest.num_knots
            << " knots\n";
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) return 77;
  try {
    for (int i = 1; i < argc; ++i) check(argv[i]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
