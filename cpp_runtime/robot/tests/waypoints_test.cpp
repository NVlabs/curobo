// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/scene.h"
#include "trajectory_waypoints_cuda.h"

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template <typename ValueType>
class Buffer {
 public:
  explicit Buffer(size_t count) : count_(count) {
    check(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(ValueType)));
    check(cudaMemset(data_, 0, count * sizeof(ValueType)));
  }
  ~Buffer() { cudaFree(data_); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  ValueType *get() const { return data_; }
  void upload(const std::vector<ValueType> &values) {
    require(values.size() == count_, "buffer upload size");
    check(cudaMemcpy(data_, values.data(), count_ * sizeof(ValueType), cudaMemcpyHostToDevice));
    // The planner and captured tests use nonblocking streams. Finish pageable
    // host uploads before handing their device pointers to a different stream.
    check(cudaStreamSynchronize(nullptr));
  }
  std::vector<ValueType> read() const {
    std::vector<ValueType> values(count_);
    check(cudaMemcpy(values.data(), data_, count_ * sizeof(ValueType), cudaMemcpyDeviceToHost));
    return values;
  }

 private:
  size_t count_;
  ValueType *data_{nullptr};
};

void test_reference_and_validation(curobo::robot::SplineLayout layout, int count = 32) {
  using namespace curobo::robot::detail;
  Buffer<float> p(count * 3), q(count * 4), sp(3), sq(4), rp(layout.native_horizon * 3),
      rq(layout.native_horizon * 4), progress(5), settings(10);
  Buffer<float> measured_p(5 * 3), measured_q(5 * 4);
  Buffer<int32_t> counts(1), violation(1);
  std::vector<float> positions(count * 3), quaternions(count * 4);
  for (int i = 0; i < count; ++i) {
    positions[i * 3] = float(i + 1) / count;
    quaternions[i * 4] = 1.0F;
  }
  p.upload(positions);
  q.upload(quaternions);
  sq.upload({1, 0, 0, 0});
  check(launch_pose_waypoints(p.get(), q.get(), count, sp.get(), sq.get(), 1, layout, rp.get(),
                              rq.get(), nullptr));
  const auto reference = rp.read();
  for (int i = 0; i <= layout.padding; ++i) {
    require(reference[i * 3] == 0.0F, "start reference must hold");
    require(reference[(layout.native_horizon - 1 - i) * 3] == 1.0F, "end reference must hold");
  }
  for (int i = 0; i < count; ++i) {
    const int step =
        layout.padding + int(std::nearbyint(float(layout.reference_intervals) * (i + 1) / count));
    require(std::abs(reference[step * 3] - positions[i * 3]) < 1.0E-6F,
            "every anchor must survive resampling");
  }
  counts.upload({4});
  progress.upload({0, .25F, .5F, 1, 0});
  measured_p.upload({0, 0, 0, .25F, 0, 0, .5F, 0, 0, 1, 0, 0, 100, 100, 100});
  measured_q.upload({1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
  check(launch_tool_path_settings(settings.get(), ToolPathMode::WAYPOINTS, true, .05F, .26F, .005F,
                                  .05F, 0, 0, 1, true, nullptr));
  auto validate = [&]() {
    check(cudaMemset(violation.get(), 0, sizeof(int32_t)));
    check(launch_validate_tool_path(measured_p.get(), measured_q.get(), rp.get(), rq.get(),
                                    progress.get(), counts.get(), settings.get(), 1, 5, 1, layout,
                                    violation.get(), nullptr));
    return violation.read()[0];
  };
  require(validate() == 0, "dense validation must ignore unused capacity");
  measured_p.upload({0, 0, 0, .75F, 0, 0, .5F, 0, 0, 1, 0, 0, 100, 100, 100});
  require(validate() != 0, "waypoints must reject out-of-order progress along the line");
  measured_p.upload({0, 0, 0, .25F, 0, 0, .5F, 0, 0, 1, 0, 0, 100, 100, 100});
  counts.upload({6});
  require(validate() != 0, "dense validation must reject truncated output capacity");
  counts.upload({4});
  measured_p.upload({0, 0, 0, .25F, .06F, 0, .5F, 0, 0, 1, 0, 0, 100, 100, 100});
  require(validate() != 0, "dense between-anchor deviation must fail");
  measured_p.upload({0, 0, 0, .25F, 0, 0, .5F, 0, 0, 1.01F, 0, 0, 100, 100, 100});
  require(validate() != 0, "final tolerance must be separate from running tolerance");
  measured_p.upload({0, 0, 0, .25F, 0, 0, .5F, 0, 0, 1, 0, 0, 100, 100, 100});
  measured_q.upload({1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
  require(validate() != 0, "orientation tracking must enforce angular tolerance");
  check(launch_tool_path_settings(settings.get(), ToolPathMode::WAYPOINTS, false, .05F, .26F, .005F,
                                  .05F, 0, 0, 1, true, nullptr));
  require(validate() == 0, "position-only tracking must ignore angular error");
  // The generic line-cost API still allows free axial progress.
  check(launch_pose_waypoints(p.get() + (count - 1) * 3, q.get() + (count - 1) * 4, 0, nullptr,
                              nullptr, 1, layout, rp.get(), rq.get(), nullptr));
  check(launch_tool_path_settings(settings.get(), ToolPathMode::LINE, true, .02F, .174532925F,
                                  .005F, .05F, 1, 0, 0, false, nullptr));
  measured_q.upload({1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
  measured_p.upload({.3F, 0, 0, .8F, .019F, 0, .7F, 0, 0, 1, 0, 0, 100, 100, 100});
  require(validate() == 0, "line validation must allow axial progress");
  measured_p.upload({.3F, 0, 0, .8F, .021F, 0, .7F, 0, 0, 1, 0, 0, 100, 100, 100});
  require(validate() != 0, "line validation must enforce 2cm perpendicular tolerance");
}

// Inject a failed final acceptance verdict just before the fallback guard.
// This exercises restoration deterministically without weakening production checks.
void inject_final_failure(cudaGraph_t graph, uint8_t *success) {
  size_t count = 0;
  check(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  check(cudaGraphGetNodes(graph, nodes.data(), &count));
  cudaGraphNode_t fallback = nullptr;
  for (auto node : nodes) {
    cudaGraphNodeType type;
    check(cudaGraphNodeGetType(node, &type));
    size_t children = 0;
#if CUDART_VERSION >= 13000
    check(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &children));
#else
    check(cudaGraphNodeGetDependentNodes(node, nullptr, &children));
#endif
    if (type == cudaGraphNodeTypeConditional && children == 0) fallback = node;
  }
  require(fallback != nullptr, "captured refinement must have a final fallback conditional");
  const auto dependencies = [](cudaGraphNode_t node) {
    size_t size = 0;
#if CUDART_VERSION >= 13000
    check(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &size));
    std::vector<cudaGraphNode_t> result(size);
    check(cudaGraphNodeGetDependencies(node, result.data(), nullptr, &size));
#else
    check(cudaGraphNodeGetDependencies(node, nullptr, &size));
    std::vector<cudaGraphNode_t> result(size);
    check(cudaGraphNodeGetDependencies(node, result.data(), &size));
#endif
    return result;
  };
  const auto guards = dependencies(fallback);
  require(guards.size() == 1, "fallback must follow its device guard");
  const auto predecessors = dependencies(guards[0]);
  cudaMemsetParams clear{};
  clear.dst = success;
  clear.elementSize = 1;
  clear.width = 1;
  clear.height = 1;
  cudaGraphNode_t failure;
  check(cudaGraphAddMemsetNode(&failure, graph, predecessors.data(), predecessors.size(), &clear));
#if CUDART_VERSION >= 13000
  check(cudaGraphAddDependencies(graph, &failure, guards.data(), nullptr, 1));
#else
  check(cudaGraphAddDependencies(graph, &failure, guards.data(), 1));
#endif
}

void test_planning(const char *artifact, int num_knots) {
  using namespace curobo::robot;
  MotionPlanningConfig config;
  config.trajectory_optimization.num_seeds = 4;
  config.trajectory_optimization.num_knots = num_knots;
  SceneConfig scene_config;
  scene_config.max_cuboids = 4;
  scene_config.max_capsules = scene_config.max_cylinders = 0;
  scene_config.max_meshes = scene_config.max_voxel_grids = 0;
  Scene scene(scene_config);
  Cuboid table;
  table.name = "table";
  table.pose.position = {0, 0, -.2F};
  table.dimensions = {4, 4, .2F};
  scene.add_cuboid(table);
  MotionPlanner planner(artifact, config, &scene);
  const auto shape = planner.info();
  const int waypoint_count = num_knots >= 62 ? 256 : 32;
  const int horizon = planner.native_horizon();
  require(horizon == shape.spline_layout.native_horizon && horizon == shape.trajectory_steps,
          "planner must use exported native horizon");
  require(shape.num_knots == num_knots, "planner control count must match config");
  auto mismatched = config;
  mismatched.trajectory_optimization.num_knots = num_knots == 64 ? 63 : num_knots + 1;
  bool mismatch_rejected = false;
  try {
    MotionPlanner invalid(artifact, mismatched);
  } catch (const std::invalid_argument &) {
    mismatch_rejected = true;
  }
  require(mismatch_rejected, "configured control count must match the artifact before planning");
  Kinematics fk(artifact);
  Buffer<float> start(7), goal(7), gp(3), gq(4), jac(42), p(waypoint_count * 3),
      q(waypoint_count * 4);
  start.upload({0, -1.3F, 0, -2.5F, 0, 1.5F, .8F});
  goal.upload({.15F, -1.2F, .1F, -2.4F, .05F, 1.5F, .8F});
  fk.forward(goal.get(), gp.get(), gq.get(), jac.get(), 1, 1);
  const auto end_p = gp.read(), end_q = gq.read();
  fk.forward(start.get(), gp.get(), gq.get(), jac.get(), 1, 1);
  const auto begin_p = gp.read(), begin_q = gq.read();
  std::vector<float> positions(waypoint_count * 3), quaternions(waypoint_count * 4);
  for (int i = 0; i < waypoint_count; ++i) {
    const float t = float(i + 1) / waypoint_count;
    for (int j = 0; j < 3; ++j) positions[i * 3 + j] = (1 - t) * begin_p[j] + t * end_p[j];
    for (int j = 0; j < 4; ++j) quaternions[i * 4 + j] = end_q[j];
  }
  p.upload(positions);
  q.upload(quaternions);
  Buffer<float> outputs(num_knots * 7 + 4 * horizon * 7 + 7 + 5);
  Buffer<uint8_t> status(2);
  Buffer<int32_t> indices(3);
  PoseMotionPlanningRequest pose;
  require(pose.outer_iterations == 2, "normal planning must default to two solve/check passes");
  auto *out = outputs.get();
  pose.optimized_knots = out;
  out += num_knots * 7;
  pose.trajectory_position = out;
  out += horizon * 7;
  pose.trajectory_velocity = out;
  out += horizon * 7;
  pose.trajectory_acceleration = out;
  out += horizon * 7;
  pose.trajectory_jerk = out;
  out += horizon * 7;
  pose.selected_goal_position = out;
  out += 7;
  pose.cost = out++;
  pose.collision_cost = out++;
  pose.trajectory_dt = out++;
  pose.position_error = out++;
  pose.orientation_error = out++;
  pose.success = status.get();
  pose.used_rrt = status.get() + 1;
  pose.selected_seed = indices.get();
  pose.rrt_iterations = indices.get() + 1;
  pose.planning_attempts = indices.get() + 2;
  pose.current_position = start.get();
  pose.goal_positions = p.get() + (waypoint_count - 1) * 3;
  pose.goal_quaternions = q.get() + (waypoint_count - 1) * 4;
  pose.max_attempts = 2;
  // The first normal solve initializes all graph-unsafe dependencies.
  planner.plan_pose(pose);
  require(status.read()[0] == 1, "normal pose warmup must succeed");
  pose.enable_graph_attempt = 0;
  planner.plan_pose(pose);
  require(status.read()[0] == 1 && status.read()[1] == 1 && planner.last_rrt_calls() > 0,
          "configured RRT seed fitting and refinement must succeed");
  pose.enable_graph_attempt = 1;
  WaypointMotionPlanningRequest request;
  require(request.outer_iterations == 4, "waypoints must default to four solve/check passes");
  request.motion = pose;
  request.positions = p.get();
  request.quaternions = q.get();
  request.num_waypoints = waypoint_count;
  for (bool orientation : {true, false, true}) {
    request.track_orientation = orientation;
    planner.plan_pose_waypoints(request);
    require(status.read()[0] == 1, "waypoint planning after normal warmup must succeed");
    require(status.read()[1] == 0, "waypoints must not fall back to unconstrained RRT");
    const auto execution = planner.timeopt_trajectory();
    require(execution.sample_count != nullptr, "waypoints must expose executable dense output");
    int count = 0;
    check(cudaMemcpy(&count, execution.sample_count, sizeof(count), cudaMemcpyDeviceToHost));
    require(count > 1 && count <= execution.capacity, "waypoint dense sample count");
    Buffer<float> tool_p(count * 3), tool_q(count * 4), tool_j(count * 42);
    fk.forward(execution.position, tool_p.get(), tool_q.get(), tool_j.get(), 1, count);
    const auto dense = tool_p.read();
    for (int i = 0; i < count; ++i) {
      float along = 0, denom = 0;
      for (int j = 0; j < 3; ++j) {
        along += (dense[i * 3 + j] - begin_p[j]) * (end_p[j] - begin_p[j]);
        denom += (end_p[j] - begin_p[j]) * (end_p[j] - begin_p[j]);
      }
      const float t = std::fmin(1.0F, std::fmax(0.0F, along / denom));
      float error = 0;
      for (int j = 0; j < 3; ++j) {
        const float d = dense[i * 3 + j] - ((1 - t) * begin_p[j] + t * end_p[j]);
        error += d * d;
      }
      require(std::sqrt(error) <= request.path_position_tolerance + 1E-5F,
              "independent dense FK path check");
    }
  }
  // An unreachable intermediate target exhausts the budget while terminal IK stays valid.
  // Changing only the pass count must invalidate the captured optimization graph.
  const auto reachable_positions = positions;
  positions[(waypoint_count / 2) * 3 + 2] += 3.0F;
  p.upload(positions);
  request.motion.max_attempts = 1;
  request.motion.finetune_attempts = 0;
  for (const int passes : {4, 1, 2, 4}) {
    request.outer_iterations = passes;
    planner.plan_pose_waypoints(request);
    require(status.read()[0] == 0, "unreachable intermediate waypoint must fail");
    int iterations = 0;
    check(cudaMemcpy(&iterations, planner.candidates().initial_iterations, sizeof(iterations),
                     cudaMemcpyDeviceToHost));
    require(iterations == passes * 50, "waypoint solve must honor the requested pass count");
  }
  positions = reachable_positions;
  p.upload(positions);
  request.motion = pose;
  request.outer_iterations = 4;
  planner.plan_pose(pose);
  int normal_iterations = 0;
  check(cudaMemcpy(&normal_iterations, planner.candidates().initial_iterations,
                   sizeof(normal_iterations), cudaMemcpyDeviceToHost));
  require(status.read()[0] == 1 && normal_iterations == 50,
          "waypoint budgets must not change normal planning's default");
  for (const int invalid : {0, -1}) {
    auto invalid_pose = pose;
    invalid_pose.outer_iterations = invalid;
    bool rejected = false;
    try {
      planner.plan_pose(invalid_pose);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "nonpositive outer iteration counts must be rejected");
  }
  // A curved path must depart from the straight joint-space initialization.
  for (int i = 0; i < waypoint_count; ++i)
    positions[i * 3 + 2] += .10F * std::sin(3.141592654F * (i + 1) / waypoint_count);
  p.upload(positions);
  request.path_position_tolerance = .025F;
  request.track_orientation = true;
  planner.plan_pose_waypoints(request);
  require(status.read()[0] == 1, "curved Cartesian path must optimize beyond the straight seed");
  // Position-only IK and tracking must tolerate an unrelated rotation target.
  for (int i = 0; i < waypoint_count; ++i) {
    quaternions[i * 4] = 0;
    quaternions[i * 4 + 1] = 1;
    quaternions[i * 4 + 2] = 0;
    quaternions[i * 4 + 3] = 0;
  }
  q.upload(quaternions);
  request.track_orientation = false;
  planner.plan_pose_waypoints(request);
  require(status.read()[0] == 1,
          "position-only waypoints must ignore terminal and intermediate orientation");
  for (int i = 0; i < waypoint_count; ++i)
    for (int j = 0; j < 4; ++j) quaternions[i * 4 + j] = end_q[j];
  q.upload(quaternions);
  planner.plan_pose(pose);
  require(status.read()[0] == 1, "normal pose after waypoints must succeed");
  Buffer<float> moving(7);
  moving.upload({.01F, 0, 0, 0, 0, 0, 0});
  request.motion.current_velocity = moving.get();
  bool moving_rejected = false;
  try {
    planner.plan_pose_waypoints(request);
  } catch (const std::invalid_argument &) {
    moving_rejected = true;
  }
  require(moving_rejected, "waypoints must reject nonstationary starts");
  request.motion.current_velocity = nullptr;
  request.final_orientation_tolerance = 4.0F;
  bool tolerance_rejected = false;
  try {
    planner.plan_pose_waypoints(request);
  } catch (const std::invalid_argument &) {
    tolerance_rejected = true;
  }
  require(tolerance_rejected, "angular tolerances above pi must be rejected");
  request.final_orientation_tolerance = .05F;
  if (shape.spline_layout.reference_intervals < 256) {
    request.num_waypoints = shape.spline_layout.reference_intervals + 1;
    bool grid_rejected = false;
    try {
      planner.plan_pose_waypoints(request);
    } catch (const std::invalid_argument &) {
      grid_rejected = true;
    }
    require(grid_rejected, "targets exceeding the configured reference grid must be rejected");
  }
  request.num_waypoints = 257;
  bool rejected = false;
  try {
    planner.plan_pose_waypoints(request);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "more than 256 targets must be rejected");

  TrajectoryOptimizationConfig policy;
  policy.num_seeds = 1;
  policy.num_knots = num_knots;
  policy.finetune_attempts = 0;
  policy.enable_timeopt = false;
  TrajectoryOptimizer optimizer(artifact, policy);
  TrajectoryOptimizationRequest opt;
  opt.outer_iterations = 4;
  opt.start_position = start.get();
  opt.goal_position = goal.get();
  opt.optimized_knots = pose.optimized_knots;
  opt.trajectory_position = pose.trajectory_position;
  opt.trajectory_velocity = pose.trajectory_velocity;
  opt.trajectory_acceleration = pose.trajectory_acceleration;
  opt.trajectory_jerk = pose.trajectory_jerk;
  opt.success = pose.success;
  opt.cost = pose.cost;
  opt.collision_cost = pose.collision_cost;
  opt.trajectory_dt = pose.trajectory_dt;
  Buffer<uint8_t> valid(1);
  valid.upload({1});
  opt.seed_valid = valid.get();
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  opt.stream = stream;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  optimizer.optimize(opt);
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, 0));
  for (const int expected : {50, 200, 50}) {
    valid.upload({static_cast<uint8_t>(expected == 50)});
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    int iterations = 0;
    check(cudaMemcpy(&iterations, optimizer.candidates().initial_iterations, sizeof(int),
                     cudaMemcpyDeviceToHost));
    require(iterations == expected,
            "captured adaptive solve must stop at success or 200 iterations");
    require((status.read()[0] != 0) == (expected == 50),
            "captured adaptive success must follow current seed validity");
  }
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));

  valid.upload({1});
  policy.enable_timeopt = true;
  TrajectoryOptimizer initial(artifact, policy);
  opt.stream = stream;
  initial.optimize(opt);
  check(cudaStreamSynchronize(stream));
  require(status.read()[0] == 1, "initial fallback reference must be executable");
  const auto reference_output = outputs.read();
  float reference_duration = 0;
  check(cudaMemcpy(&reference_duration, initial.timeopt_trajectory().duration, sizeof(float),
                   cudaMemcpyDeviceToHost));
  policy.finetune_attempts = 1;
  TrajectoryOptimizer refined(artifact, policy);
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  refined.optimize(opt);
  check(cudaStreamEndCapture(stream, &graph));
  inject_final_failure(graph, opt.success);
  check(cudaGraphInstantiate(&executable, graph, 0));
  check(cudaGraphLaunch(executable, stream));
  check(cudaStreamSynchronize(stream));
  require(status.read()[0] == 1, "failed final refinement must restore initial success");
  const auto restored = outputs.read();
  for (int i = 0; i < num_knots * 7 + 4 * horizon * 7; ++i)
    require(std::abs(restored[i] - reference_output[i]) < 1.0E-5F,
            "fallback must restore initial controls and states");
  float restored_duration = 0;
  check(cudaMemcpy(&restored_duration, refined.timeopt_trajectory().duration, sizeof(float),
                   cudaMemcpyDeviceToHost));
  require(std::abs(restored_duration - reference_duration) < 1.0E-5F,
          "fallback must restore executable timing");
  Buffer<float> moving_velocity(7);
  moving_velocity.upload({.01F, -.01F, .005F, -.01F, .005F, .01F, -.005F});
  opt.start_velocity = moving_velocity.get();
  initial.optimize(opt);
  check(cudaStreamSynchronize(stream));
  require(status.read()[0] == 1, "configured standalone optimizer must retain moving starts");
  std::array<float, 7> first_velocity{};
  check(cudaMemcpy(first_velocity.data(), initial.timeopt_trajectory().velocity, 7 * sizeof(float),
                   cudaMemcpyDeviceToHost));
  const auto expected_velocity = moving_velocity.read();
  for (int joint = 0; joint < 7; ++joint)
    require(std::abs(first_velocity[joint] - expected_velocity[joint]) < 1E-4F,
            "configured retimer must preserve initial velocity");
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
}
}  // namespace
int main(int argc, char **argv) {
  try {
    for (int knots = 12; knots <= 1024; ++knots) {
      const int horizon = 4 * (knots + 4) + 1;
      test_reference_and_validation({knots, 3, 4, horizon, 4, horizon - 9});
    }
    for (int count = 1; count <= 256; ++count) {
      test_reference_and_validation({64, 3, 4, 273, 4, 264}, count);
    }
    if (argc >= 2) test_planning(argv[1], argc == 3 ? std::stoi(argv[2]) : 16);
    std::cout << "waypoint checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
