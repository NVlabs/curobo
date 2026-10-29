// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/mpc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "cuda_check.h"
#include "motion_planner_runtime.h"
#include "mpc_cuda.h"
#include "trajectory_optimizer_runtime.h"

namespace curobo::robot::detail {
namespace {
using curobo_robot_runtime::DeviceBuffer;
constexpr int CANDIDATES = MpcConfig::TIMING_CANDIDATES;
float *data(const DeviceBuffer &value) { return reinterpret_cast<float *>(value.ptr()); }
void copy(DeviceBuffer &destination, const float *source, cudaStream_t stream) {
  CRR_CUDA_CHECK(cudaMemcpyAsync(data(destination), source, destination.bytes(),
                                 cudaMemcpyDeviceToDevice, stream));
}
}  // namespace

class MpcRuntime final {
 public:
  MpcRuntime(const MotionPlanner &planner, const MpcConfig &config) : config_(config) {
    if (planner.runtime_ == nullptr || config.iterations <= 0 ||
        !std::isfinite(config.trajectory_dt) || config.trajectory_dt <= 0.0F ||
        !std::isfinite(config.collision_activation_distance) ||
        config.collision_activation_distance <= 0)
      throw std::invalid_argument("Invalid MPC configuration");
    auto &source = *planner.runtime_;
    TrajectoryOptimizationConfig optimizer_config;
    optimizer_config.device_ordinal = source.config_.trajectory_optimization.device_ordinal;
    optimizer_config.num_seeds = CANDIDATES;
    optimizer_config.max_iterations = config.iterations;
    optimizer_config.line_search_backtracking = config.line_search_backtracking;
    optimizer_config.finetune_attempts = 0;
    optimizer_config.trajectory_dt = config.trajectory_dt;
    optimizer_config.collision_activation_distance = config.collision_activation_distance;
    optimizer_ = std::make_unique<TrajectoryOptimizerRuntime>(
        source.artifact_directory_, optimizer_config, source.scene_handle_,
        source.trajectory_optimizer_->collision_);
    auto &opt = *optimizer_;
    const auto &manifest = opt.manifest_;
    // The current artifact uses cubic B-splines, reconstructed exactly by
    // quintic Hermite execution at matching q/v/a and knot-aligned intervals.
    if (manifest.spline_degree != 3)
      throw std::invalid_argument("MPC currently requires the cubic B-spline artifact");
    for (const auto *kernel :
         {&manifest.collision_cuboid, &manifest.collision_capsule, &manifest.collision_cylinder,
          &manifest.collision_mesh, &manifest.collision_voxel}) {
      if (!kernel->cubin.empty() && kernel->symbol.find("swept") == std::string::npos)
        throw std::invalid_argument("MPC requires swept world-collision kernels");
    }
    load_weights(source.artifact_directory_);
    for (float weight : config.tool_pose_weight)
      if (!std::isfinite(weight) || weight <= 0)
        throw std::invalid_argument("MPC pose weights must be finite and positive");
    opt.prepare_pose_criteria_ = false;
    // Schema-1 export verifies this matches lbfgs_mpc.yml. Native TrajOpt's
    // default scales bound weights by dt; MPC intentionally does not.
    opt.retime_cspace_weights_ = false;
    // Dense validation checks at <=4 ms. The exported fixed buffer must cover
    // the entire trajectory; overflow is rejected by the existing validator.
    opt.manifest_.interpolation_dt = std::min(opt.manifest_.interpolation_dt, 0.004F);
    candidate_dts_ = make_candidate_dts(config.trajectory_dt);
    CRR_CUDA_CHECK(cudaMemcpy(data(opt.interpolation_dt_), &opt.manifest_.interpolation_dt,
                              sizeof(float), cudaMemcpyHostToDevice));
    const auto shape = opt.info();
    const size_t state_bytes = shape.num_dof * sizeof(float);
    for (auto *buffer : {&position_, &velocity_, &acceleration_, &jerk_, &goal_})
      buffer->allocate(state_bytes);
    tool_positions_.allocate(shape.num_tool_frames * 3 * sizeof(float));
    tool_quaternions_.allocate(shape.num_tool_frames * 4 * sizeof(float));
    tool_positions_.clear(nullptr);
    std::vector<float> identity_quaternions(shape.num_tool_frames * 4, 0.F);
    for (int i = 0; i < shape.num_tool_frames; ++i) identity_quaternions[i * 4] = 1.F;
    CRR_CUDA_CHECK(cudaMemcpy(data(tool_quaternions_), identity_quaternions.data(),
                              tool_quaternions_.bytes(), cudaMemcpyHostToDevice));
    for (auto *buffer : {&seed_, &shifted_seed_, &knots_})
      buffer->allocate(shape.num_knots * state_bytes);
    for (auto *buffer :
         {&output_position_, &output_velocity_, &output_acceleration_, &output_jerk_})
      buffer->allocate(shape.trajectory_steps * state_bytes);
    batch_seed_.allocate(CANDIDATES * seed_.bytes());
    shift_.allocate(sizeof(int));
    success_.allocate(sizeof(uint8_t));
    selected_.allocate(sizeof(int32_t));
    violation_.allocate(sizeof(int));
    constrain_path_.allocate(sizeof(uint8_t));
    path_direction_.allocate(3 * sizeof(float));
    for (auto *buffer : {&cost_, &collision_cost_, &dt_}) buffer->allocate(sizeof(float));
    input_dt_.allocate(sizeof(candidate_dts_));
    CRR_CUDA_CHECK(cudaMemcpy(data(input_dt_), candidate_dts_.data(), input_dt_.bytes(),
                              cudaMemcpyHostToDevice));
    request_.start_position = data(position_);
    request_.start_velocity = data(velocity_);
    request_.start_acceleration = data(acceleration_);
    request_.goal_position = data(goal_);
    request_.seed_knots = data(batch_seed_);
    request_.optimized_knots = data(knots_);
    request_.trajectory_position = data(output_position_);
    request_.trajectory_velocity = data(output_velocity_);
    request_.trajectory_acceleration = data(output_acceleration_);
    request_.trajectory_jerk = data(output_jerk_);
    request_.success = reinterpret_cast<uint8_t *>(success_.ptr());
    request_.cost = data(cost_);
    request_.collision_cost = data(collision_cost_);
    request_.trajectory_dt = data(dt_);
    request_.selected_seed = reinterpret_cast<int32_t *>(selected_.ptr());
    request_.finetune_attempts = 0;
    // Always capture the same pose-cost/gradient path. Zero criteria in c-space
    // mode remove pose costs and pose acceptance distances without recapture.
    request_.constrain_tool_path = true;
    request_.goal_tool_positions = data(tool_positions_);
    request_.goal_tool_quaternions = data(tool_quaternions_);
    CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&default_stream_, cudaStreamNonBlocking));
  }

  ~MpcRuntime() {
    cudaSetDevice(optimizer_->config_.device_ordinal);
    if (graph_exec_ != nullptr) cudaGraphExecDestroy(graph_exec_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    if (check_graph_exec_ != nullptr) cudaGraphExecDestroy(check_graph_exec_);
    if (check_graph_ != nullptr) cudaGraphDestroy(check_graph_);
    if (default_stream_ != nullptr) cudaStreamDestroy(default_stream_);
  }

  TrajectoryOptimizationInfo info() const { return optimizer_->info(); }
  int captures() const { return captures_; }
  std::array<float, CANDIDATES> candidate_dts() const { return candidate_dts_; }

  void update_dt(float dt, cudaStream_t stream) {
    const auto timings = make_candidate_dts(dt);
    if (dt == config_.trajectory_dt) return;
    CRR_CUDA_CHECK(cudaSetDevice(optimizer_->config_.device_ordinal));
    stream = resolve_stream(stream);
    CRR_CUDA_CHECK(
        cudaMemcpyAsync(data(input_dt_), timings.data(), input_dt_.bytes(),
                          cudaMemcpyHostToDevice, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    config_.trajectory_dt = dt;
    candidate_dts_ = timings;
    initialized_ = false;  // reset supplies the new seed and clears L-BFGS history.
    pending_ = false;
  }

  void reset(const float *goal, const float *seed, cudaStream_t stream) {
    if (goal == nullptr || seed == nullptr)
      throw std::invalid_argument("MPC reset needs goal and seed");
    CRR_CUDA_CHECK(cudaSetDevice(optimizer_->config_.device_ordinal));
    stream = resolve_stream(stream);
    set_objective(false, {}, stream);
    copy(goal_, goal, stream);
    reset_seed(seed, stream);
  }

  void reset_cartesian(const float *positions, const float *quaternions, const float *seed,
                       const MpcCartesianConfig &config, cudaStream_t stream) {
    if (!positions || !quaternions || !seed)
      throw std::invalid_argument("Cartesian MPC needs targets and seed");
    CRR_CUDA_CHECK(cudaSetDevice(optimizer_->config_.device_ordinal));
    stream = resolve_stream(stream);
    set_objective(true, config, stream);
    copy(tool_positions_, positions, stream);
    copy(tool_quaternions_, quaternions, stream);
    reset_seed(seed, stream);
  }

  void update_cartesian_target(const float *positions, const float *quaternions,
                               cudaStream_t stream) {
    if (!initialized_ || !cartesian_) throw std::logic_error("Reset Cartesian MPC first");
    if (!positions || !quaternions) throw std::invalid_argument("Missing Cartesian targets");
    CRR_CUDA_CHECK(cudaSetDevice(optimizer_->config_.device_ordinal));
    stream = resolve_stream(stream);
    copy(tool_positions_, positions, stream);
    copy(tool_quaternions_, quaternions, stream);
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  MpcResult step(const float *position, const float *velocity, const float *acceleration,
                 const float *jerk, int executed_knots, cudaStream_t stream) {
    if (!initialized_) throw std::logic_error("Call MPC reset before step");
    pending_ = false;
    if (position == nullptr || velocity == nullptr || acceleration == nullptr || jerk == nullptr ||
        executed_knots < 0 || executed_knots > info().num_knots)
      throw std::invalid_argument("MPC needs q/v/a/jerk and an in-range whole-knot shift");
    CRR_CUDA_CHECK(cudaSetDevice(optimizer_->config_.device_ordinal));
    stream = resolve_stream(stream);
    copy(position_, position, stream);
    copy(velocity_, velocity, stream);
    copy(acceleration_, acceleration, stream);
    copy(jerk_, jerk, stream);
    CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(shift_.ptr()), &executed_knots,
                                   sizeof(int), cudaMemcpyHostToDevice, stream));
    if (graph_exec_ == nullptr) {
      // All modules/workspaces are initialized. Capture performs no solve and
      // does not consume the accepted seed an extra time.
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      request_.stream = stream;
      CRR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      try {
        solve();
      } catch (...) {
        cudaGraph_t failed = nullptr;
        cudaStreamEndCapture(stream, &failed);
        if (failed != nullptr) cudaGraphDestroy(failed);
        throw;
      }
      CRR_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
      CRR_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
      ++captures_;
    }
    CRR_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
    uint8_t feasible = 0;
    MpcResult result;
    CRR_CUDA_CHECK(cudaMemcpyAsync(&feasible, request_.success, sizeof(feasible),
                                   cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(
        cudaMemcpyAsync(&result.cost, data(cost_), sizeof(float), cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(cudaMemcpyAsync(&result.collision_cost, data(collision_cost_), sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(cudaMemcpyAsync(&result.trajectory_dt, data(dt_), sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(cudaMemcpyAsync(&result.selected_candidate,
                                   reinterpret_cast<void *>(selected_.ptr()), sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    result.feasible =
        feasible != 0 && std::isfinite(result.cost) && std::isfinite(result.collision_cost);
    result.position = data(output_position_);
    result.velocity = data(output_velocity_);
    result.acceleration = data(output_acceleration_);
    result.jerk = data(output_jerk_);
    pending_ = result.feasible;
    return result;
  }

  void accept(cudaStream_t stream) {
    if (!pending_) throw std::logic_error("No feasible MPC proposal to accept");
    CRR_CUDA_CHECK(cudaSetDevice(optimizer_->config_.device_ordinal));
    stream = resolve_stream(stream);
    copy(seed_, data(knots_), stream);
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    pending_ = false;
  }

  bool validate_trajectory(const float *q, const float *v, const float *a, const float *j,
                           float sample_dt, cudaStream_t stream) {
    if (q == nullptr || v == nullptr || a == nullptr || j == nullptr ||
        !std::isfinite(sample_dt) || sample_dt <= 0)
      throw std::invalid_argument("Trajectory validation requires q/v/a/jerk and positive dt");
    CRR_CUDA_CHECK(cudaSetDevice(optimizer_->config_.device_ordinal));
    stream = resolve_stream(stream);
    auto &opt = *optimizer_;
    const auto bytes = output_position_.bytes();
    for (auto pair :
         {std::make_pair(&opt.dense_position_, q), std::make_pair(&opt.dense_velocity_, v),
          std::make_pair(&opt.dense_acceleration_, a), std::make_pair(&opt.dense_jerk_, j)})
      CRR_CUDA_CHECK(cudaMemcpyAsync(data(*pair.first), pair.second, bytes,
                                     cudaMemcpyDeviceToDevice, stream));
    CRR_CUDA_CHECK(cudaMemcpyAsync(data(opt.state_dt_), &sample_dt, sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    if (check_graph_exec_ == nullptr) {
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      CRR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      try {
        const auto driver_stream = reinterpret_cast<CUstream>(stream);
        opt.launch_kinematics(1, driver_stream);
        opt.launch_world_collision(1, driver_stream);
        opt.launch_self_collision(1, driver_stream);
        violation_.clear(stream);
        CRR_CUDA_CHECK(launch_mpc_validate(
            data(opt.dense_position_), data(opt.dense_velocity_), data(opt.dense_acceleration_),
            data(opt.dense_jerk_), data(opt.position_bounds_), data(opt.velocity_bounds_),
            data(opt.acceleration_bounds_), data(opt.jerk_bounds_),
            data(opt.world_collision_cost_), data(opt.self_collision_cost_),
            data(opt.robot_spheres_), opt.manifest_.num_spheres,
            config_.collision_activation_distance * 2.0F,
            reinterpret_cast<int *>(violation_.ptr()), info().trajectory_steps, info().num_dof,
            stream));
      } catch (...) {
        cudaGraph_t failed = nullptr;
        cudaStreamEndCapture(stream, &failed);
        if (failed != nullptr) cudaGraphDestroy(failed);
        throw;
      }
      CRR_CUDA_CHECK(cudaStreamEndCapture(stream, &check_graph_));
      CRR_CUDA_CHECK(cudaGraphInstantiate(&check_graph_exec_, check_graph_, 0));
    }
    CRR_CUDA_CHECK(cudaGraphLaunch(check_graph_exec_, stream));
    int violation = 0;
    CRR_CUDA_CHECK(cudaMemcpyAsync(&violation, reinterpret_cast<void *>(violation_.ptr()),
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return violation == 0;
  }

 private:
  std::array<float, CANDIDATES> make_candidate_dts(float initial_dt) const {
    const auto &manifest = optimizer_->manifest_;
    const int samples_per_knot = (manifest.trajectory_steps - 1) /
                                 (manifest.num_knots + manifest.spline_degree + 1);
    if (!std::isfinite(initial_dt) || initial_dt <= 0.F)
      throw std::invalid_argument("MPC initial dt must be finite and positive");
    const float first = std::max(initial_dt, MpcConfig::MINIMUM_KNOT_DT / samples_per_knot);
    const float maximum = std::max(MpcConfig::DEFAULT_MAXIMUM_KNOT_DT / samples_per_knot,
                                   MpcConfig::MAXIMUM_DT_SCALE * first);
    std::array<float, CANDIDATES> result;
    for (int i = 0; i < CANDIDATES; ++i) {
      result[i] = i == CANDIDATES - 1 ? maximum :
          first * std::pow(maximum / first, static_cast<float>(i) / (CANDIDATES - 1));
      validate_dt(result[i]);
    }
    return result;
  }

  void validate_dt(float dt) const {
    const auto &manifest = optimizer_->manifest_;
    if (!std::isfinite(dt) || dt <= 0.F || dt < manifest.minimum_dt || dt > manifest.maximum_dt)
      throw std::invalid_argument("MPC dt is outside artifact limits");
    const int intervals = manifest.num_knots + manifest.spline_degree + 1;
    const int samples = (manifest.trajectory_steps - 1) / intervals;
    const float dense_steps =
        (dt * samples + manifest.interpolation_dt) / manifest.interpolation_dt;
    if (std::floor(dense_steps) * intervals + 1 > manifest.interpolation_buffer_size)
      throw std::invalid_argument("MPC horizon exceeds the artifact's 4 ms validation buffer");
  }

  void reset_seed(const float *seed, cudaStream_t stream) {
    copy(seed_, seed, stream);
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    initialized_ = true;
    pending_ = false;
  }

  void set_objective(bool cartesian, const MpcCartesianConfig &config, cudaStream_t stream) {
    auto &opt = *optimizer_;
    const int tools = info().num_tool_frames;
    float length = 0.F;
    for (float value : config.direction) length += value * value;
    if (!std::isfinite(length) || length < 1.e-12F ||
        !std::isfinite(config.along_direction_weight) || config.along_direction_weight < 0.F ||
        config.along_direction_weight > 1.F)
      throw std::invalid_argument("Invalid Cartesian MPC direction/weight");
    std::vector<float> position(2 * tools * 9, 0.F), rotation(position.size(), 0.F);
    // Track continuously without a cost deadband. Acceptance has independent
    // tolerances; reusing them here can postpone small motions indefinitely.
    std::vector<float> tolerance(2 * tools * 2, 0.F);
    std::vector<int32_t> frame(tools, config.direction_in_goal_frame ? 1 : 0);
    std::array<float, 3> direction = config.direction;
    for (float &value : direction) value /= std::sqrt(length);
    if (cartesian) {
      for (int criterion = 0; criterion < 2 * tools; ++criterion) {
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            const float identity = row == col ? 1.F : 0.F;
            const int index = criterion * 9 + row * 3 + col;
            position[index] = identity;
            if (criterion >= tools)
              position[index] -=
                  (1.F - config.along_direction_weight) * direction[row] * direction[col];
            rotation[index] = identity;
          }
        }
      }
    }
    auto upload = [&](DeviceBuffer &buffer, const void *source, size_t bytes) {
      if (buffer.bytes() != bytes) throw std::logic_error("MPC criterion shape mismatch");
      CRR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(buffer.ptr()), source, bytes,
                                     cudaMemcpyHostToDevice, stream));
    };
    upload(opt.pose_position_metric_, position.data(), position.size() * sizeof(float));
    upload(opt.pose_rotation_metric_, rotation.data(), rotation.size() * sizeof(float));
    upload(opt.pose_tolerance_, tolerance.data(), tolerance.size() * sizeof(float));
    upload(opt.pose_error_frame_, frame.data(), frame.size() * sizeof(int32_t));
    upload(path_direction_, direction.data(), sizeof(direction));
    const float cspace_weight = cartesian ? 0.F : cspace_weight_;
    upload(opt.cspace_target_weight_, &cspace_weight, sizeof(float));
    const std::array<float, 2> pose_weight =
        cartesian ? config_.tool_pose_weight : std::array<float, 2>{0.F, 0.F};
    upload(opt.tool_pose_weight_, pose_weight.data(), sizeof(pose_weight));
    CRR_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(opt.implicit_goal_.ptr()),
                                   cartesian ? 0 : 1, opt.implicit_goal_.bytes(), stream));
    CRR_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void *>(constrain_path_.ptr()),
                                   cartesian && config.constrain_path ? 1 : 0, 1, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));  // Host criteria may now expire.
    cartesian_ = cartesian;
  }
  cudaStream_t resolve_stream(cudaStream_t stream) {
    if (stream != nullptr) return stream;
    CRR_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    return default_stream_;
  }
  void load_weights(const std::string &directory) {
    // Schema 1: 5 bounds, 5 activations, 5 smoothness, target,
    // nonterminal target, world, self. Exported directly from lbfgs_mpc.yml.
    std::ifstream file(directory + "/mpc_cost_weights.f32", std::ios::binary);
    std::array<float, 19> values{};
    if (!file.read(reinterpret_cast<char *>(values.data()), sizeof(values)) || file.peek() != EOF)
      throw std::runtime_error(
          "Export MPC weights with tools/export_mpc_weights.py --out ARTIFACT");
    for (const float value : values)
      if (!std::isfinite(value) || value < 0) throw std::runtime_error("Invalid MPC weights");
    auto upload = [&](DeviceBuffer &buffer, int offset, int count) {
      if (buffer.bytes() != count * sizeof(float))
        throw std::runtime_error("MPC weight shape mismatch");
      CRR_CUDA_CHECK(cudaMemcpy(data(buffer), values.data() + offset, buffer.bytes(),
                                cudaMemcpyHostToDevice));
    };
    upload(optimizer_->state_bound_weight_, 0, 5);
    upload(optimizer_->state_activation_distance_, 5, 5);
    upload(optimizer_->state_regularization_weight_, 10, 5);
    upload(optimizer_->cspace_target_weight_, 15, 1);
    cspace_weight_ = values[15];
    upload(optimizer_->cspace_non_terminal_weight_, 16, 1);
    upload(optimizer_->world_collision_weight_, 17, 1);
    upload(optimizer_->self_collision_weight_, 18, 1);
  }

  void solve() {
    auto &opt = *optimizer_;
    const auto shape = info();
    const int dimension = shape.num_knots * shape.num_dof;
    const auto stream = request_.stream;
    const auto *shift = reinterpret_cast<const int *>(shift_.ptr());
    CRR_CUDA_CHECK(launch_mpc_tail(data(seed_), data(goal_),
                                   reinterpret_cast<const uint8_t *>(opt.implicit_goal_.ptr()),
                                   shape.num_knots, shape.num_dof, stream));
    CRR_CUDA_CHECK(launch_mpc_shift(data(seed_), data(shifted_seed_), data(goal_), shift, 1,
                                    shape.num_knots, shape.num_dof, stream));
    // Consume elapsed knots once even if this proposal is rejected. Only
    // accept() replaces this shifted accepted seed with newly optimized knots.
    copy(seed_, data(shifted_seed_), stream);
    CRR_CUDA_CHECK(launch_mpc_broadcast(data(shifted_seed_), data(batch_seed_), dimension,
                                       CANDIDATES, stream));
    CRR_CUDA_CHECK(launch_mpc_broadcast(data(jerk_), data(opt.resolved_start_jerks_),
                                       shape.num_dof, CANDIDATES, stream));
    for (auto *buffer : {&opt.history_s_, &opt.history_y_, &opt.history_rho_,
                         &opt.history_head_, &opt.history_count_})
      buffer->clear(stream);
    opt.initialize_actions(request_, reinterpret_cast<CUstream>(stream));
    // The initializer's host scalar is frozen at graph capture. Override it
    // from persistent GPU input before evaluating splines and costs. The shared
    // Python/Warp c-space kernel scales regularization by dt, dt^2, dt^3;
    // bound/pose/collision weights remain unchanged for the MPC configuration.
    CRR_CUDA_CHECK(cudaMemcpyAsync(data(opt.trajectory_dt_), data(input_dt_), input_dt_.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    // Each lane optimizes the same spatial seed at its fixed phase timing.
    opt.optimize_pass(request_, reinterpret_cast<CUstream>(stream),
                      opt.config_.max_iterations);
    opt.validate_interpolated(opt.best_action_.ptr(), reinterpret_cast<CUstream>(stream));
    // Apply the same sampling-density guard to the native candidate, not only
    // to execution-layer checks. Padding is the terminal hold, as in TrajOpt.
    CRR_CUDA_CHECK(launch_mpc_validate(
        data(opt.validation_position_), data(opt.validation_velocity_),
        data(opt.validation_acceleration_), data(opt.validation_jerk_), data(opt.position_bounds_),
        data(opt.velocity_bounds_), data(opt.acceleration_bounds_), data(opt.jerk_bounds_),
        data(opt.validation_world_collision_cost_), data(opt.validation_self_collision_cost_),
        data(opt.validation_robot_spheres_), opt.manifest_.num_spheres,
        config_.collision_activation_distance * 2.0F,
        reinterpret_cast<int *>(opt.validation_violation_.ptr()),
        opt.manifest_.interpolation_buffer_size, shape.num_dof, stream, CANDIDATES));
    // Native TrajOpt metrics are discrete. Keep the swept check (including
    // activation clearance) in MPC's acceptance decision, not only its cost.
    opt.launch_world_collision(CANDIDATES, reinterpret_cast<CUstream>(stream));
    // MPC may make safe partial progress rather than reach the Cartesian goal
    // within one horizon. Keep path/collision/limit acceptance, but do not use
    // terminal pose convergence as candidate feasibility. Costs are unchanged;
    // measured goal completion belongs to the executor, not this validator.
    CRR_CUDA_CHECK(launch_mpc_pose_acceptance(
        data(opt.pose_position_distance_), data(opt.pose_orientation_distance_),
        data(opt.link_position_), data(tool_positions_), data(tool_quaternions_),
        data(path_direction_), reinterpret_cast<const int32_t *>(opt.pose_error_frame_.ptr()),
        reinterpret_cast<const uint8_t *>(constrain_path_.ptr()), CANDIDATES * shape.trajectory_steps,
        shape.num_tool_frames, stream));
    opt.accumulate_optimization_pass(true, request_);
    CRR_CUDA_CHECK(launch_mpc_rank(reinterpret_cast<uint8_t *>(opt.retained_success_.ptr()),
                                  data(opt.retained_cost_), data(opt.retained_dt_),
                                  data(opt.retained_rank_score_), CANDIDATES, stream));
    opt.finish_optimization(request_);
  }

  MpcConfig config_;
  std::unique_ptr<TrajectoryOptimizerRuntime> optimizer_;
  TrajectoryOptimizationRequest request_{};
  DeviceBuffer position_, velocity_, acceleration_, jerk_, goal_, seed_, shifted_seed_, knots_;
  DeviceBuffer batch_seed_, selected_;
  DeviceBuffer tool_positions_, tool_quaternions_;
  DeviceBuffer constrain_path_, path_direction_;
  DeviceBuffer input_dt_;
  std::array<float, CANDIDATES> candidate_dts_{};
  float cspace_weight_{0.F};
  bool cartesian_{false};
  DeviceBuffer shift_, success_, cost_, collision_cost_, dt_, violation_;
  DeviceBuffer output_position_, output_velocity_, output_acceleration_, output_jerk_;
  cudaGraph_t graph_{nullptr};
  cudaGraphExec_t graph_exec_{nullptr};
  cudaGraph_t check_graph_{nullptr};
  cudaGraphExec_t check_graph_exec_{nullptr};
  cudaStream_t default_stream_{nullptr};
  bool initialized_{false}, pending_{false};
  int captures_{0};
};
}  // namespace curobo::robot::detail

namespace curobo::robot {
MpcController::MpcController(const MotionPlanner &planner, const MpcConfig &config)
    : runtime_(std::make_unique<detail::MpcRuntime>(planner, config)) {}
MpcController::~MpcController() = default;
TrajectoryOptimizationInfo MpcController::info() const { return runtime_->info(); }
void MpcController::update_dt(float dt, cudaStream_t stream) { runtime_->update_dt(dt, stream); }
std::array<float, MpcConfig::TIMING_CANDIDATES> MpcController::candidate_dts() const {
  return runtime_->candidate_dts();
}
void MpcController::accept(cudaStream_t stream) { runtime_->accept(stream); }
void MpcController::reset(const float *goal, const float *seed, cudaStream_t stream) {
  runtime_->reset(goal, seed, stream);
}
void MpcController::reset_cartesian(const float *positions, const float *quaternions,
                                    const float *seed, const MpcCartesianConfig &config,
                                    cudaStream_t stream) {
  runtime_->reset_cartesian(positions, quaternions, seed, config, stream);
}
void MpcController::update_cartesian_target(const float *positions, const float *quaternions,
                                            cudaStream_t stream) {
  runtime_->update_cartesian_target(positions, quaternions, stream);
}
MpcResult MpcController::step(const float *q, const float *v, const float *a, const float *jerk,
                              int knots, cudaStream_t stream) {
  return runtime_->step(q, v, a, jerk, knots, stream);
}
int MpcController::graph_capture_count() const { return runtime_->captures(); }
bool MpcController::validate_trajectory(const float *q, const float *v, const float *a,
                                        const float *j, float dt, cudaStream_t stream) {
  return runtime_->validate_trajectory(q, v, a, j, dt, stream);
}
}  // namespace curobo::robot
