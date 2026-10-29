// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>

#include "cuda_check.h"
#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/mpc.h"
#include "curobo_robot_runtime/scene.h"
#include "fake_quintic_controller.h"
#include "robot_runtime.h"

namespace {
using namespace curobo::robot;
using namespace mpc_test;
using curobo_robot_runtime::DeviceBuffer;

static_assert(!InverseKinematicsConfig{}.line_search_backtracking);
static_assert(!TrajectoryOptimizationConfig{}.line_search_backtracking);
static_assert(MpcConfig{}.line_search_backtracking);

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

class Buffer {
 public:
  explicit Buffer(size_t count) : size_(count) { storage_.allocate(count * sizeof(float)); }
  float *data() const { return reinterpret_cast<float *>(storage_.ptr()); }
  void upload(const float *source) {
    CRR_CUDA_CHECK(cudaMemcpy(data(), source, size_ * sizeof(float), cudaMemcpyHostToDevice));
    // Tests also consume these buffers on nonblocking streams. Pageable host
    // uploads must complete before those streams read the default-stream copy.
    CRR_CUDA_CHECK(cudaStreamSynchronize(nullptr));
  }
  std::vector<float> download() const {
    std::vector<float> result(size_);
    CRR_CUDA_CHECK(
        cudaMemcpy(result.data(), data(), size_ * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
  }
  float first() const { return download()[0]; }

 private:
  DeviceBuffer storage_;
  size_t size_;
};

std::vector<float> download(const float *data, int count) {
  std::vector<float> result(count);
  CRR_CUDA_CHECK(cudaMemcpy(result.data(), data, count * sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

class Simulation {
 public:
  Simulation(MpcController &mpc, int steps)
      : mpc_(mpc),
        steps_(steps),
        q_(steps * DOF),
        v_(steps * DOF),
        a_(steps * DOF),
        j_(steps * DOF) {}

  bool validate(const FakeQuinticController &controller, float duration) {
    // Check the actual spline in <=4 ms increments, including long replacements.
    constexpr float WINDOW = 0.004F * 80;
    for (float offset = 0; offset < std::max(duration, 0.004F); offset += WINDOW) {
      auto window = controller;
      window.advance(offset);
      if (!validate_window(window, std::min(WINDOW, std::max(duration - offset, 0.004F))))
        return false;
    }
    return true;
  }

 private:
  bool validate_window(const FakeQuinticController &controller, float duration) {
    std::vector<float> q(steps_ * DOF), v(q.size()), a(q.size()), j(q.size());
    const float dt = std::max(duration, 0.004F) / (steps_ - 1);
    for (int i = 0; i < steps_; ++i) {
      const auto state = controller.predict(i * dt);
      for (int k = 0; k < DOF; ++k) {
        q[i * DOF + k] = state.q[k];
        v[i * DOF + k] = state.v[k];
        a[i * DOF + k] = state.a[k];
        j[i * DOF + k] = state.jerk[k];
      }
    }
    q_.upload(q.data());
    v_.upload(v.data());
    a_.upload(a.data());
    j_.upload(j.data());
    return mpc_.validate_trajectory(q_.data(), v_.data(), a_.data(), j_.data(), dt);
  }

  MpcController &mpc_;
  int steps_;
  Buffer q_, v_, a_, j_;
};

float max_abs(const Joints &state) {
  float result = 0;
  for (const float v : state) result = std::max(result, std::abs(v));
  return result;
}
float error(const Joints &q, const Joints &target) {
  float result = 0;
  for (int j = 0; j < DOF; ++j) result = std::max(result, std::abs(q[j] - target[j]));
  return result;
}

bool check_moving_dt_batch(MotionPlanner &planner, const Joints &initial,
                           const Joints &goal, const std::vector<float> &planned_q,
                           const std::vector<float> &planned_v,
                           const std::vector<float> &planned_a, float planned_dt,
                           float speed) {
  // A poor seed at a MOVING handoff, anchored to the initial plan's timing.
  // The accepted nominal command is never stretched or replaced by a failure.
  MpcConfig config;
  config.trajectory_dt = planned_dt / speed;
  MpcController mpc(planner, config);
  const auto shape = mpc.info();
  Buffer q(DOF), v(DOF), a(DOF), j(DOF), target(DOF), seed(shape.num_knots * DOF);
  FakeQuinticController controller(initial);
  auto scaled_v = planned_v, scaled_a = planned_a;
  for (auto &value : scaled_v) value *= speed;
  for (auto &value : scaled_a) value *= speed * speed;
  controller.accept(planned_q, scaled_v, scaled_a, planned_dt / speed, 4);
  const float handoff = 4 * 4 * planned_dt / speed;
  const auto predicted = controller.predict(handoff);
  require(max_abs(predicted.v) > .01F, "Recovery fixture must be moving");
  q.upload(predicted.q.data());
  v.upload(predicted.v.data());
  a.upload(predicted.a.data());
  j.upload(predicted.jerk.data());
  target.upload(goal.data());
  std::vector<float> flat(shape.num_knots * DOF);
  for (int k = 0; k < shape.num_knots; ++k)
    std::copy(predicted.q.begin(), predicted.q.end(), flat.begin() + k * DOF);
  seed.upload(flat.data());
  mpc.reset(target.data(), seed.data());
  const auto fixed_dts = mpc.candidate_dts();
  // Warm the graph, then measure a single four-lane replay.
  mpc.step(q.data(), v.data(), a.data(), j.data(), 0);
  mpc.reset(target.data(), seed.data());
  const auto begin = std::chrono::steady_clock::now();
  auto result = mpc.step(q.data(), v.data(), a.data(), j.data(), 0);
  const float solve_ms = std::chrono::duration<float, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  std::cout << "Moving batch speed=" << max_abs(predicted.v)
            << " feasible=" << result.feasible << " lane=" << result.selected_candidate
            << " knot_dt=" << result.trajectory_dt * 4 << " solve_ms=" << solve_ms
            << " candidates=";
  for (float dt : fixed_dts) std::cout << dt * 4 << ',';
  std::cout << '\n';
  require(error(controller.predict(handoff).q, predicted.q) == 0.F,
          "Batch modified the executing command");
  const auto first_q = download(result.position, shape.trajectory_steps * DOF);
  const auto repeated = mpc.step(q.data(), v.data(), a.data(), j.data(), 0);
  require(result.selected_candidate == repeated.selected_candidate,
          "Unaccepted proposal changed the next batch's seed");
  const auto repeated_q = download(repeated.position, shape.trajectory_steps * DOF);
  for (size_t i = 0; i < first_q.size(); ++i)
    require(std::abs(first_q[i] - repeated_q[i]) < 1.e-5F,
            "A rejected proposal leaked into the warm start");
  require(mpc.candidate_dts() == fixed_dts, "A solve changed the phase timing candidates");
  if (!result.feasible) {
    require(result.selected_candidate == -1, "Infeasible batch selected a lane");
    return false;
  }
  controller.advance(handoff);
  const int count = shape.trajectory_steps * DOF;
  controller.accept(download(result.position, count), download(result.velocity, count),
                     download(result.acceleration, count), result.trajectory_dt, 4);
  Simulation simulation(mpc, shape.trajectory_steps);
  require(simulation.validate(controller, controller.remaining()),
          "Retimed moving replacement failed swept quintic validation");
  controller.advance(controller.remaining());
  require(error(controller.predict().q, goal) < 1.e-4F, "Retry changed the final joint goal");
  mpc.accept();
  require(mpc.graph_capture_count() == 1, "Moving batch recaptured graph");
  std::cout << "Moving batch recovery: knot_dt=" << result.trajectory_dt * 4
            << " endpoint_error=" << error(controller.predict().q, goal)
            << " graph_captures=1\n";
  return true;
}

void check_stationary_cartesian_recovery(MotionPlanner &planner, Kinematics &fk) {
  // Stopped commanded state from the offline contact fixture. The previous
  // uphill fallback poisoned L-BFGS history and produced exactly zero motion.
  const Joints stopped{0.350124F, 0.51522F, -0.17277F, -2.56166F, 0.875876F, 3.03133F, 0.112436F};
  const Joints zero{};
  Buffer q(DOF), v(DOF), a(DOF), j(DOF), pose(3), quat(4), jacobian(6 * DOF);
  for (float dt : {0.00292936F, 0.00496348F, 0.01F, 0.025F}) {
    MpcConfig config;
    config.trajectory_dt = dt;
    MpcController mpc(planner, config);
    const auto shape = mpc.info();
    Buffer seed(shape.num_knots * DOF);
    std::vector<float> flat(shape.num_knots * DOF);
    for (int k = 0; k < shape.num_knots; ++k)
      std::copy(stopped.begin(), stopped.end(), flat.begin() + k * DOF);
    seed.upload(flat.data());
    q.upload(stopped.data());
    v.upload(zero.data());
    a.upload(zero.data());
    j.upload(zero.data());
    fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    auto goal = pose.download();
    const float start_z = goal[2];
    goal[2] += 0.005F;
    pose.upload(goal.data());
    MpcCartesianConfig objective;
    objective.along_direction_weight = 1.F;
    objective.constrain_path = false;
    mpc.reset_cartesian(pose.data(), quat.data(), seed.data(), objective);
    bool accepted = false;
    const int advance = 4 * (shape.trajectory_steps - 1) / (shape.num_knots + 4);
    for (int iteration = 0; iteration < 60; ++iteration) {
      const auto result = mpc.step(q.data(), v.data(), a.data(), j.data(), accepted ? 4 : 0);
      accepted = result.feasible;
      if (!accepted) continue;
      mpc.accept();
      for (auto pair :
           {std::make_pair(q.data(), result.position), std::make_pair(v.data(), result.velocity),
            std::make_pair(a.data(), result.acceleration), std::make_pair(j.data(), result.jerk)})
        CRR_CUDA_CHECK(cudaMemcpy(pair.first, pair.second + advance * DOF, DOF * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
    }
    fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    const float progress = pose.download()[2] - start_z;
    std::cout << "Stationary recovery dt=" << dt << " progress_mm=" << 1000.F * progress
              << " graph_captures=" << mpc.graph_capture_count() << '\n';
    require(std::abs(progress - 0.005F) < 0.0015F, "Small Cartesian recovery stalled");
    require(mpc.graph_capture_count() == 1, "Backtracking recaptured the CUDA graph");
  }
}

void check_lift_progress_with_path_validation(MotionPlanner &planner, Kinematics &fk) {
  // Late-lift commanded state from the contact fixture's 10 mm shortfall.
  const Joints stopped{0.448016F, 0.424049F, -0.286237F, -2.55736F,
                       0.828449F, 3.07415F,  0.154869F};
  const Joints zero{};
  MpcConfig config;
  config.trajectory_dt = 0.00496053F;
  MpcController mpc(planner, config);
  const auto shape = mpc.info();
  Buffer q(DOF), v(DOF), a(DOF), j(DOF), pose(3), quat(4), jacobian(6 * DOF);
  Buffer seed(shape.num_knots * DOF);
  std::vector<float> flat(shape.num_knots * DOF);
  for (int k = 0; k < shape.num_knots; ++k)
    std::copy(stopped.begin(), stopped.end(), flat.begin() + k * DOF);
  seed.upload(flat.data());
  q.upload(stopped.data());
  v.upload(zero.data());
  a.upload(zero.data());
  j.upload(zero.data());
  fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
  const auto start = pose.download();
  auto goal = start;
  goal[2] += 0.0104184F;
  pose.upload(goal.data());
  MpcCartesianConfig objective;
  require(objective.along_direction_weight == 1.F,
          "Running pose cost must default to full weight");
  require(objective.constrain_path, "Lift regression must retain path validation");
  mpc.reset_cartesian(pose.data(), quat.data(), seed.data(), objective);
  const int advance = 4 * (shape.trajectory_steps - 1) / (shape.num_knots + 4);
  float max_lateral_error = 0.F;
  for (int iteration = 0; iteration < 100; ++iteration) {
    const auto result = mpc.step(q.data(), v.data(), a.data(), j.data(), iteration ? 4 : 0);
    require(result.feasible, "Remaining lift distance was treated as a path violation");
    mpc.accept();
    for (auto pair :
         {std::make_pair(q.data(), result.position), std::make_pair(v.data(), result.velocity),
          std::make_pair(a.data(), result.acceleration), std::make_pair(j.data(), result.jerk)})
      CRR_CUDA_CHECK(cudaMemcpy(pair.first, pair.second + advance * DOF, DOF * sizeof(float),
                                cudaMemcpyDeviceToDevice));
    fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    const auto actual = pose.download();
    max_lateral_error =
        std::max(max_lateral_error, std::hypot(actual[0] - start[0], actual[1] - start[1]));
  }
  const float remaining = goal[2] - pose.download()[2];
  require(std::abs(remaining) < 0.002F, "Lift retained its 10 mm shortfall");
  require(max_lateral_error < 0.005F, "Lift departed from its straight-line path");
  require(mpc.graph_capture_count() == 1, "Lift recaptured the CUDA graph");
  // A lateral offset is still a violation, regardless of forward progress.
  auto invalid_goal = pose.download();
  invalid_goal[0] += 0.02F;
  pose.upload(invalid_goal.data());
  mpc.update_cartesian_target(pose.data(), quat.data());
  require(!mpc.step(q.data(), v.data(), a.data(), j.data(), 4).feasible,
          "Lateral path violation was accepted");
  fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
  const auto rotation = quat.download();
  const float sine = std::sin(0.1F), cosine = std::cos(0.1F);
  const std::array<float, 4> invalid_rotation{
      rotation[0] * cosine - rotation[3] * sine, rotation[1] * cosine + rotation[2] * sine,
      rotation[2] * cosine - rotation[1] * sine, rotation[3] * cosine + rotation[0] * sine};
  quat.upload(invalid_rotation.data());
  mpc.update_cartesian_target(pose.data(), quat.data());
  // No shift: the preceding rejected candidate was never executed.
  require(!mpc.step(q.data(), v.data(), a.data(), j.data(), 0).feasible,
          "Orientation path violation was accepted");
  require(mpc.graph_capture_count() == 1, "Path update recaptured the CUDA graph");
  std::cout << "Lift with path checks remaining_mm=" << 1000.F * remaining
            << " max_lateral_mm=" << 1000.F * max_lateral_error << " graph_captures=1\n";
}
}  // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: curobo_robot_mpc_test ARTIFACT_DIRECTORY");
    constexpr int GRID_SIDE = 64;
    constexpr int GRID_COUNT = GRID_SIDE * GRID_SIDE * GRID_SIDE;
    constexpr float VOXEL_SIZE = 0.025F;
    SceneConfig scene_config;
    scene_config.max_voxel_grids = 1;
    scene_config.max_voxels_per_grid = GRID_COUNT;
    Scene scene(scene_config);
    VoxelGrid grid;
    grid.name = "live_esdf";
    grid.pose.position = {0.3F, 0.0F, 0.5F};
    grid.dimensions = {1.6F, 1.6F, 1.6F};
    grid.voxel_size = VOXEL_SIZE;
    grid.esdf.assign(GRID_COUNT, 2.0F);
    const auto grid_handle = scene.add_voxel_grid(grid);
    scene.set_obstacle_enabled(grid_handle, false);
    DeviceBuffer esdf;
    esdf.allocate(GRID_COUNT * sizeof(__half));
    Cuboid obstacle;
    obstacle.name = "temporary_goal_blocker";
    obstacle.pose.position = {10, 10, 10};
    obstacle.dimensions = {0.08F, 0.08F, 0.08F};
    const auto blocker = scene.add_cuboid(obstacle);
    MotionPlanningConfig config;
    config.trajectory_optimization.num_seeds = 4;
    config.rrt.max_plans = 4;
    config.trajectory_optimization.finetune_attempts = 0;
    MotionPlanner planner(argv[1], config, &scene);
    Kinematics fk(argv[1]);
    check_lift_progress_with_path_validation(planner, fk);
    check_stationary_cartesian_recovery(planner, fk);
    const auto info = planner.info();
    require(info.num_dof == DOF, "MPC fake controller test expects Franka");
    const Joints initial{0, -1.3F, 0, -2.5F, 0, 1.5F, 0.8F};
    const Joints requested_goal{0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    Buffer q(DOF), v(DOF), a(DOF), start_jerk(DOF), target(DOF), pose(3), quat(4),
        jacobian(6 * DOF);
    q.upload(initial.data());
    target.upload(requested_goal.data());
    const Joints zero{};
    v.upload(zero.data());
    a.upload(zero.data());
    fk.forward(target.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    const auto goal_xyz = pose.download();
    const auto update_esdf = [&](bool blocked) {
      std::vector<__half> values(GRID_COUNT);
      for (int x = 0; x < GRID_SIDE; ++x) {
        for (int y = 0; y < GRID_SIDE; ++y) {
          for (int z = 0; z < GRID_SIDE; ++z) {
            const int indices[3]{x, y, z};
            float outside_squared = 0, inside = -std::numeric_limits<float>::infinity();
            for (int axis = 0; axis < 3; ++axis) {
              const float coordinate =
                  grid.pose.position[axis] + (indices[axis] - (GRID_SIDE - 1) * 0.5F) * VOXEL_SIZE;
              const float d = std::abs(coordinate - goal_xyz[axis]) - 0.04F;
              outside_squared += std::max(d, 0.0F) * std::max(d, 0.0F);
              inside = std::max(inside, d);
            }
            values[(x * GRID_SIDE + y) * GRID_SIDE + z] =
                __float2half(blocked ? std::sqrt(outside_squared) + std::min(inside, 0.0F) : 2.0F);
          }
        }
      }
      CRR_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(esdf.ptr()), values.data(), esdf.bytes(),
                                cudaMemcpyHostToDevice));
      scene.update_voxel_grid_cuda(grid_handle, esdf.ptr(), GRID_COUNT, 0);
    };
    const int count = info.trajectory_steps * DOF;
    Buffer knots(info.num_knots * DOF), positions(count), velocities(count), accelerations(count),
        jerk(count);
    Buffer cost(1), collision(1), dt(1), position_error(1), orientation_error(1);
    DeviceBuffer success, used_rrt, rrt_iterations;
    success.allocate(sizeof(uint8_t));
    used_rrt.allocate(sizeof(uint8_t));
    rrt_iterations.allocate(sizeof(int32_t));
    PoseMotionPlanningRequest request;
    request.current_position = q.data();
    request.goal_positions = pose.data();
    request.goal_quaternions = quat.data();
    request.selected_goal_position = target.data();
    request.optimized_knots = knots.data();
    request.trajectory_position = positions.data();
    request.trajectory_velocity = velocities.data();
    request.trajectory_acceleration = accelerations.data();
    request.trajectory_jerk = jerk.data();
    request.success = reinterpret_cast<uint8_t *>(success.ptr());
    request.used_rrt = reinterpret_cast<uint8_t *>(used_rrt.ptr());
    request.rrt_iterations = reinterpret_cast<int32_t *>(rrt_iterations.ptr());
    request.cost = cost.data();
    request.collision_cost = collision.data();
    request.trajectory_dt = dt.data();
    request.position_error = position_error.data();
    request.orientation_error = orientation_error.data();
    planner.plan_pose(request);
    uint8_t planned = 0;
    CRR_CUDA_CHECK(cudaMemcpy(&planned, request.success, 1, cudaMemcpyDeviceToHost));
    require(planned != 0, "Initial plan_pose failed");
    Joints goal;
    const auto goal_values = target.download();
    std::copy(goal_values.begin(), goal_values.end(), goal.begin());
    MpcConfig mpc_config;
    mpc_config.trajectory_dt = dt.first();
    MpcController mpc(planner, mpc_config);
    const auto phase_timings = mpc.candidate_dts();
    bool early_accept_rejected = false;
    try {
      mpc.accept();
    } catch (const std::logic_error &) {
      early_accept_rejected = true;
    }
    require(early_accept_rejected, "MPC accepted without a feasible proposal");
    Simulation simulation(mpc, info.trajectory_steps);
    const auto initial_q = positions.download(), initial_v = velocities.download(),
               initial_a = accelerations.download();
    const float knot_dt =
        mpc_config.trajectory_dt * (info.trajectory_steps - 1) / (info.num_knots + 4);
    const float initial_period = 4.F * knot_dt;
    constexpr float STOP_TIME = 0.2F;
    std::vector<float> times;

    for (const std::string name : {"clear", "blocked_then_removed", "esdf_blocked_then_removed",
                                   "permanently_blocked", "late_results"}) {
      Pose obstacle_pose;
      obstacle_pose.position = {10, 10, 10};
      scene.update_obstacle_pose(blocker, obstacle_pose);
      const bool use_esdf = name == "esdf_blocked_then_removed";
      scene.set_obstacle_enabled(grid_handle, use_esdf);
      if (use_esdf) update_esdf(false);
      mpc.reset(target.data(), knots.data());
      FakeQuinticController controller(initial);
      controller.accept(initial_q, initial_v, initial_a, mpc_config.trajectory_dt, 4);
      float period = initial_period, handoff = knot_dt;
      int accepted = 0, rejected = 0;
      bool previous_accepted = false;
      bool reached = false, stopped = false;
      float peak_v = 0, peak_a = 0, peak_j = 0;
      for (int tick = 0; tick < 250; ++tick) {
        // Insert the recoverable obstacle before the arm nears the goal.
        // Do not assume a particular optimizer's progress rate; the separate
        // late-obstacle test covers an obstacle inside the stopping distance.
        const bool blocked = name.find("blocked") != std::string::npos && tick >= 1 &&
                             (name == "permanently_blocked" || tick < 25);
        obstacle_pose.position = blocked && !use_esdf
                                     ? std::array<float, 3>{goal_xyz[0], goal_xyz[1], goal_xyz[2]}
                                     : std::array<float, 3>{10, 10, 10};
        scene.update_obstacle_pose(blocker, obstacle_pose);
        if (use_esdf && (tick == 1 || tick == 25)) update_esdf(blocked);
        const auto predicted = controller.predict(handoff);
        q.upload(predicted.q.data());
        v.upload(predicted.v.data());
        a.upload(predicted.a.data());
        const auto begin = std::chrono::steady_clock::now();
        start_jerk.upload(predicted.jerk.data());
        const auto candidate = mpc.step(q.data(), v.data(), a.data(), start_jerk.data(),
                                        tick == 0 ? 1 : (previous_accepted ? 4 : 0));
        times.push_back(
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin)
                .count());
        const bool late = name == "late_results" && tick >= 5 && tick < 15;
        require(mpc.candidate_dts() == phase_timings,
                "Scene or solve failure changed the initial phase timing grid");
        auto candidate_q = download(candidate.position, count);
        auto candidate_v = download(candidate.velocity, count);
        auto candidate_a = download(candidate.acceleration, count);
        require(simulation.validate(controller, handoff), name + ": outgoing prefix unsafe");
        controller.advance(handoff);
        bool executable = false;
        auto replacement = controller;
        if (candidate.feasible && !late) {
          for (int j = 0; j < DOF; ++j)
            require(
                std::abs(candidate_q[(info.trajectory_steps - 1) * DOF + j] - goal[j]) < 1.0e-5F,
                "Implicit goal moved during optimization");
          replacement.accept(candidate_q, candidate_v, candidate_a, candidate.trajectory_dt, 4);
          executable = simulation.validate(replacement, replacement.remaining());
        }
        if (executable) {
          controller = std::move(replacement);
          mpc.accept();
          period = 16.F * candidate.trajectory_dt;
          handoff = 4.F * candidate.trajectory_dt;
          ++accepted;
          previous_accepted = true;
        } else {
          previous_accepted = false;
          ++rejected;
          auto braking = controller;
          braking.decelerate(STOP_TIME);
          require(simulation.validate(braking, braking.remaining()),
                  name + ": braking trajectory unsafe at tick " + std::to_string(tick));
          controller = std::move(braking);
        }
        require(simulation.validate(controller, period - handoff),
                name + ": execution interval unsafe");
        for (int substep = 0; substep < 20; ++substep) {
          const auto state = controller.predict();
          peak_v = std::max(peak_v, max_abs(state.v));
          peak_a = std::max(peak_a, max_abs(state.a));
          peak_j = std::max(peak_j, max_abs(state.jerk));
          controller.advance((period - handoff) / 20);
        }
        const auto state = controller.predict();
        if (controller.stopping() && max_abs(state.v) < 1.0e-4F) stopped = true;
        reached = error(state.q, goal) < 0.005F && max_abs(state.v) < 0.01F;
        if (reached || (name == "permanently_blocked" && tick > 30 && stopped)) break;
      }
      // A permanently blocked target may correctly reject every replacement.
      if (name != "permanently_blocked") require(accepted > 0, name + ": no accepted candidates");
      if (name == "permanently_blocked")
        require(stopped && !reached, "Blocked goal did not stop");
      else
        require(reached, name + ": fixed goal not reached; accepted=" + std::to_string(accepted) +
                             " rejected=" + std::to_string(rejected) +
                             " error=" + std::to_string(error(controller.predict().q, goal)));
      if (name != "clear") require(rejected > 0, name + ": failure path was not exercised");
      std::cout << name << ": accepted=" << accepted << " rejected=" << rejected
                << " stopped=" << stopped << " reached=" << reached << " peaks v/a/j=" << peak_v
                << '/' << peak_a << '/' << peak_j << std::endl;
    }
    scene.set_obstacle_enabled(grid_handle, false);
    Pose far;
    far.position = {10, 10, 10};
    scene.update_obstacle_pose(blocker, far);
    const FakeQuinticController holding(initial);
    require(simulation.validate(holding, 0.1F), "Free holding state must validate");

    // Collision geometry remains shared after both validation and solve captures.
    SphereAttachment attachment;
    attachment.spheres = {{{0, 0, 0}, 1.0F}};
    planner.attach_object(attachment);
    require(!simulation.validate(holding, 0.1F), "Attached self-collision was ignored");
    planner.detach_object();
    require(simulation.validate(holding, 0.1F), "Detachment did not reach MPC");

    // Test swept collision independently: only the attached tiny sphere queries
    // the world. Both endpoint states are free, but their swept center crosses
    // a thin cuboid. These samples are a collision-query fixture, not commands.
    attachment.spheres = {{{0, 0, 0}, 0.002F}};
    planner.attach_object(attachment);
    planner.set_collision_mask(0);
    planner.set_link_collision_mask("attached_object", 1);

    // A dynamically inserted obstacle can make a valid stop collide. Detect
    // that failure without executing it; do not claim a physical recovery.
    FakeQuinticController braking(initial);
    braking.accept(initial_q, initial_v, initial_a, mpc_config.trajectory_dt, 4);
    braking.advance(braking.remaining() * 0.5F);
    const auto brake_start = braking.predict();
    braking.decelerate(0.4F);
    require(simulation.validate(braking, braking.remaining()), "Free-space brake is invalid");
    q.upload(braking.predict(braking.remaining()).q.data());
    fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    const auto brake_end = pose.download();
    Cuboid brake_block;
    brake_block.name = "inside_stopping_distance";
    brake_block.dimensions = {0.004F, 0.004F, 0.004F};
    std::copy(brake_end.begin(), brake_end.end(), brake_block.pose.position.begin());
    const auto brake_obstacle = scene.add_cuboid(brake_block);
    require(simulation.validate(FakeQuinticController(brake_start.q), 0.1F),
            "Brake fixture already collides at its initial pose");
    require(!simulation.validate(braking, braking.remaining()),
            "Obstacle inside braking distance was not rejected");
    scene.set_obstacle_enabled(brake_obstacle, false);
    require(simulation.validate(braking, braking.remaining()), "Brake removal did not propagate");
    std::cout << "obstacle inside stopping distance: unsafe brake rejected before execution\n";

    q.upload(initial.data());
    fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    const auto start_xyz = pose.download();
    fk.forward(target.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    const auto end_xyz = pose.download();
    Pose midpoint;
    for (int axis = 0; axis < 3; ++axis)
      midpoint.position[axis] = (start_xyz[axis] + end_xyz[axis]) * 0.5F;
    Pose at_start;
    for (int axis = 0; axis < 3; ++axis) at_start.position[axis] = start_xyz[axis];
    scene.update_obstacle_pose(blocker, at_start);
    require(!simulation.validate(holding, 0.1F), "Payload sphere did not collide at its start");
    scene.update_obstacle_pose(blocker, midpoint);
    require(simulation.validate(holding, 0.1F), "Swept fixture start is not free");
    require(simulation.validate(FakeQuinticController(goal), 0.1F),
            "Swept fixture goal is not free");
    std::vector<float> endpoint_states(count), zeros(count, 0);
    for (int i = 0; i < info.trajectory_steps; ++i)
      for (int joint = 0; joint < DOF; ++joint)
        endpoint_states[i * DOF + joint] =
            i < info.trajectory_steps / 2 ? initial[joint] : goal[joint];
    positions.upload(endpoint_states.data());
    velocities.upload(zeros.data());
    accelerations.upload(zeros.data());
    jerk.upload(zeros.data());
    require(!mpc.validate_trajectory(positions.data(), velocities.data(), accelerations.data(),
                                     jerk.data(), 0.1F),
            "Under-sampled swept collision fixture was accepted");
    scene.update_obstacle_pose(blocker, far);
    require(!mpc.validate_trajectory(positions.data(), velocities.data(), accelerations.data(),
                                     jerk.data(), 0.1F),
            "Large unchecked sphere displacement was accepted without obstacles");
    endpoint_states[0] = std::numeric_limits<float>::quiet_NaN();
    positions.upload(endpoint_states.data());
    require(!mpc.validate_trajectory(positions.data(), velocities.data(), accelerations.data(),
                                     jerk.data(), 0.1F),
            "Non-finite candidate was accepted");
    planner.detach_object();
    planner.set_collision_mask(1);
    std::cout
        << "shared attachment/self-collision, coarse-sweep rejection, and NaN rejection passed"
        << std::endl;

    // Relatch a different goal and execute the captured graph on another stream.
    // A bad state must fail closed; explicit reset must recover its warm history.
    std::vector<float> holding_knots(info.num_knots * DOF);
    for (int i = 0; i < info.num_knots; ++i)
      std::copy(initial.begin(), initial.end(), holding_knots.begin() + i * DOF);
    knots.upload(holding_knots.data());
    target.upload(initial.data());
    q.upload(initial.data());
    v.upload(zero.data());
    a.upload(zero.data());
    start_jerk.upload(zero.data());
    cudaStream_t another_stream;
    CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&another_stream, cudaStreamNonBlocking));
    mpc.reset(target.data(), knots.data(), another_stream);
    auto result = mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), 0, another_stream);
    require(result.feasible, "Goal relatch / alternate-stream replay failed");
    const auto result_q = download(result.position, count);
    for (int joint = 0; joint < DOF; ++joint)
      require(std::abs(result_q[count - DOF + joint] - initial[joint]) < 1.0e-5F,
              "Captured graph retained the old goal");
    auto invalid = initial;
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    q.upload(invalid.data());
    require(!mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), 0, another_stream).feasible,
            "Non-finite MPC start was accepted");
    bool invalid_accept_rejected = false;
    try {
      mpc.accept(another_stream);
    } catch (const std::logic_error &) {
      invalid_accept_rejected = true;
    }
    require(invalid_accept_rejected, "MPC committed an infeasible proposal");
    q.upload(initial.data());
    mpc.reset(target.data(), knots.data(), another_stream);
    require(mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), 0, another_stream).feasible,
            "Reset did not recover from invalid state");
    // Switch the already captured instance to Cartesian control, with no IK.
    fk.forward(q.data(), pose.data(), quat.data(), jacobian.data(), 1, 1);
    mpc.reset_cartesian(pose.data(), quat.data(), knots.data(), {}, another_stream);
    require(mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), 0, another_stream).feasible,
            "Cartesian hold failed");
    auto moved_pose = pose.download();
    moved_pose[2] += 0.01F;
    pose.upload(moved_pose.data());
    mpc.update_cartesian_target(pose.data(), quat.data(), another_stream);
    MpcResult cartesian;
    const int cartesian_advance = 4 * (info.trajectory_steps - 1) / (info.num_knots + 4);
    for (int i = 0; i < 100; ++i) {
      cartesian = mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), i == 0 ? 0 : 4,
                           another_stream);
      require(cartesian.feasible, "Cartesian candidate rejected during tracking");
      mpc.accept(another_stream);
      // Advance the commanded reference, never reset it to a measured pose.
      CRR_CUDA_CHECK(cudaMemcpy(q.data(), cartesian.position + cartesian_advance * DOF,
                                DOF * sizeof(float), cudaMemcpyDeviceToDevice));
      CRR_CUDA_CHECK(cudaMemcpy(v.data(), cartesian.velocity + cartesian_advance * DOF,
                                DOF * sizeof(float), cudaMemcpyDeviceToDevice));
      CRR_CUDA_CHECK(cudaMemcpy(a.data(), cartesian.acceleration + cartesian_advance * DOF,
                                DOF * sizeof(float), cudaMemcpyDeviceToDevice));
      CRR_CUDA_CHECK(cudaMemcpy(start_jerk.data(), cartesian.jerk + cartesian_advance * DOF,
                                DOF * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    std::cout << "Cartesian feasible=" << cartesian.feasible << " cost=" << cartesian.cost
              << " collision=" << cartesian.collision_cost << std::endl;
    const auto cartesian_q = download(cartesian.position, count);
    float endpoint_change = 0.F;
    for (int joint = 0; joint < DOF; ++joint)
      endpoint_change =
          std::max(endpoint_change, std::abs(cartesian_q[count - DOF + joint] - initial[joint]));
    require(endpoint_change > 1.e-3F, "Cartesian endpoint remained pinned to old joint goal");
    target.upload(cartesian_q.data() + count - DOF);
    Buffer reached_pose(3), reached_quat(4);
    fk.forward(target.data(), reached_pose.data(), reached_quat.data(), jacobian.data(), 1, 1);
    const auto reached_xyz = reached_pose.download();
    float error = 0.F;
    for (int axis = 0; axis < 3; ++axis)
      error += std::pow(reached_xyz[axis] - moved_pose[axis], 2.F);
    std::cout << "Cartesian endpoint change=" << endpoint_change << " error=" << std::sqrt(error)
              << std::endl;
    require(cartesian.feasible, "Updated Cartesian target failed");
    require(std::sqrt(error) < 0.005F, "Cartesian endpoint missed its target");

    // A stale Cartesian target must not affect a subsequent c-space hold.
    moved_pose[2] += 1.F;
    pose.upload(moved_pose.data());
    mpc.update_cartesian_target(pose.data(), quat.data(), another_stream);
    target.upload(initial.data());
    q.upload(initial.data());
    v.upload(zero.data());
    a.upload(zero.data());
    start_jerk.upload(zero.data());
    mpc.reset(target.data(), knots.data(), another_stream);
    result = mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), 0, another_stream);
    require(result.feasible, "C-space reset retained Cartesian acceptance criteria");
    const auto restored = download(result.position, count);
    for (int joint = 0; joint < DOF; ++joint)
      require(std::abs(restored[count - DOF + joint] - initial[joint]) < 1.e-5F,
              "C-space reset did not restore implicit goal");
    std::cout << "Cartesian update, free endpoint, and c-space restoration passed\n";
    std::cout << "goal relatch, alternate-stream replay, and invalid-state reset passed\n";
    // Change timing after capture and compare against a fresh solver at that dt.
    const float matching_dt = mpc_config.trajectory_dt * 0.8F;
    mpc.update_dt(matching_dt, another_stream);
    bool needs_reset = false;
    try {
      mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), 0, another_stream);
    } catch (const std::logic_error &) {
      needs_reset = true;
    }
    require(needs_reset, "Changing dt must require a matching phase seed");
    auto matching_config = mpc_config;
    matching_config.trajectory_dt = matching_dt;
    MpcController fresh(planner, matching_config);
    // Exercise nonzero target/smoothness costs, not just an unchanged hold.
    auto timing_goal = initial;
    timing_goal[0] += 0.05F;
    target.upload(timing_goal.data());
    mpc.reset(target.data(), knots.data(), another_stream);
    fresh.reset(target.data(), knots.data(), another_stream);
    for (int invalid_shift : {-1, info.num_knots + 1}) {
      bool rejected = false;
      try {
        mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), invalid_shift, another_stream);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "Out-of-range knot advancement accepted");
    }
    for (int iteration = 0; iteration < 2; ++iteration) {
      const int elapsed = iteration == 0 ? 0 : 2;
      const auto updated =
          mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), elapsed, another_stream);
      const auto expected =
          fresh.step(q.data(), v.data(), a.data(), start_jerk.data(), elapsed, another_stream);
      require(updated.feasible == expected.feasible, "Updated dt acceptance differs");
      require(std::abs(updated.cost - expected.cost) < 0.001F * (1.F + std::abs(expected.cost)),
              "Updated dt cost scaling differs from fresh solver");
      const auto actual_q = download(updated.position, count);
      const auto expected_q = download(expected.position, count);
      for (int i = 0; i < count; ++i)
        require(std::abs(actual_q[i] - expected_q[i]) < 1.e-4F,
                "Updated dt trajectory or warm-start shift differs");
      for (const auto &outputs : {std::make_pair(updated.velocity, expected.velocity),
                                  std::make_pair(updated.acceleration, expected.acceleration),
                                  std::make_pair(updated.jerk, expected.jerk)}) {
        const auto actual = download(outputs.first, count);
        const auto reference = download(outputs.second, count);
        for (int i = 0; i < count; ++i)
          require(std::abs(actual[i] - reference[i]) < 0.001F * (1.F + std::abs(reference[i])),
                  "Updated dt derivative scaling differs");
      }
      if (updated.feasible) {
        mpc.accept(another_stream);
        fresh.accept(another_stream);
      }
    }
    for (float invalid_dt : {0.F, -1.F, .05F, 1000.F, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
      bool rejected = false;
      try {
        mpc.update_dt(invalid_dt);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "Invalid MPC dt accepted");
    }
    require(mpc.graph_capture_count() == 1, "MPC recaptured on scene/state/dt changes");
    std::cout << "Updated dt: fresh-solver cost/q/v/a/jerk parity and graph reuse passed\n";
    const auto timings = mpc.candidate_dts();
    require(std::abs(timings[0] - matching_dt) < 1.e-7F &&
                std::abs(timings[3] - .025F) < 1.e-7F,
            "Timing grid did not retain initial plan dt and 100ms endpoint");
    for (int i = 1; i < 3; ++i)
      require(std::abs(timings[i] / timings[i - 1] - timings[i + 1] / timings[i]) < 1.e-5F,
              "Timing grid is not geometric");
    mpc.update_dt(.002F, another_stream);
    require(mpc.candidate_dts()[0] == .0025F, "Missing 10ms knot floor");
    mpc.update_dt(.0375F, another_stream); // 150ms initial knot spacing, not a hard cap.
    const auto slow_timings = mpc.candidate_dts();
    require(slow_timings.front() == .0375F && slow_timings.back() == .075F,
            "Slow plan must retain its initial timing and extend through twice that spacing");
    for (int i = 1; i < 3; ++i)
      require(std::abs(slow_timings[i] / slow_timings[i - 1] -
                        slow_timings[i + 1] / slow_timings[i]) < 1.e-5F,
              "Slow timing grid is not geometric");
    bool capacity_rejected = false;
    try {
      mpc.update_dt(.05F, another_stream); // 400ms last lane needs >2000 dense samples.
    } catch (const std::invalid_argument &error) {
      capacity_rejected = std::string(error.what()).find("validation buffer") != std::string::npos;
    }
    require(capacity_rejected && mpc.candidate_dts() == slow_timings,
            "Buffer overflow must reject the whole timing update without changing the grid");
    q.upload(initial.data());
    v.upload(zero.data());
    a.upload(zero.data());
    start_jerk.upload(zero.data());
    target.upload(initial.data());
    std::vector<float> hold_knots(info.num_knots * DOF);
    for (int k = 0; k < info.num_knots; ++k)
      std::copy(initial.begin(), initial.end(), hold_knots.begin() + k * DOF);
    knots.upload(hold_knots.data());
    mpc.reset(target.data(), knots.data(), another_stream);
    const auto slow = mpc.step(q.data(), v.data(), a.data(), start_jerk.data(), 0, another_stream);
    require(slow.feasible && slow.trajectory_dt == slow_timings.front(),
            "A slow stationary plan failed or was sped up");
    require(mpc.validate_trajectory(slow.position, slow.velocity, slow.acceleration, slow.jerk,
                                    slow.trajectory_dt, another_stream),
            "Slow trajectory failed dense validation");
    mpc.accept(another_stream);
    require(mpc.graph_capture_count() == 1, "Slow timing update recaptured the solve graph");
    auto slow_config = mpc_config;
    slow_config.trajectory_dt = .0375F;
    MpcController slow_fresh(planner, slow_config);
    require(slow_fresh.candidate_dts() == slow_timings,
            "Constructor and phase-entry timing grids disagree for a slow plan");
    std::cout << "150ms initial knots: fixed 150-300ms batch, 2000-sample guard, graph reuse passed\n";
    for (float speed : {.25F, .5F, 1.F})
      require(check_moving_dt_batch(planner, initial, goal, initial_q, initial_v, initial_a,
                                     mpc_config.trajectory_dt, speed),
              "Fixed timing batch did not recover the moving handoff");
    CRR_CUDA_CHECK(cudaStreamDestroy(another_stream));
    std::sort(times.begin() + 1, times.end());
    std::cout << "MPC solve median_ms=" << times[1 + (times.size() - 1) / 2]
              << " p95_ms=" << times[1 + static_cast<size_t>((times.size() - 2) * .95F)]
              << " graph_captures=" << mpc.graph_capture_count() << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
