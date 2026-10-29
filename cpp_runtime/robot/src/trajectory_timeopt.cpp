// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "curobo_robot_runtime/trajectory_timeopt.h"
#include "cuda_check.h"
#include "robot_runtime.h"
#include "rollout_runtime.h"
#include "trajectory_timeopt_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace curobo::robot::detail {
namespace {
using curobo_robot_runtime::DeviceBuffer;

std::vector<float> read_floats(const std::string &directory,
                               const curobo_robot_runtime::TensorManifest &tensor, size_t count) {
  if (tensor.dtype != "float32" || tensor.bytes != count * sizeof(float)) {
    throw std::runtime_error("timeopt limit tensor has the wrong shape or dtype");
  }
  const auto bytes = curobo_robot_runtime::load_tensor_data(directory, tensor, "timeopt limits");
  std::vector<float> result(count);
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
}  // namespace

class TrajectoryTimeOptRuntime {
 public:
  TrajectoryTimeOptRuntime(const std::string &directory, const TrajectoryTimeOptConfig &config,
                          RolloutRuntime* shared_rollout = nullptr)
      : config_(config) {
    if (config.num_seeds < 1 || config.num_seeds > 128 || config.num_knots < 2 ||
        config.num_knots > 512 || config.num_intervals < 1 ||
        config.max_iterations < 1 ||
        config.sample_count < 1 || config.sample_count > 1000000 ||
        !std::isfinite(config.limit_scale) || config.limit_scale <= 0 || config.limit_scale > 1 ||
        !std::isfinite(config.jerk_limit_scale) || config.jerk_limit_scale <= 0 ||
        config.jerk_limit_scale > 1) {
      throw std::invalid_argument("invalid timeopt shape, iteration budget, or limit scale");
    }
    CRR_CU_CHECK(cuInit(0));
    CRR_CUDA_CHECK(cudaSetDevice(config.device_ordinal));
    const auto robot = curobo_robot_runtime::load_manifest(directory);
    cudaDeviceProp properties{};
    CRR_CUDA_CHECK(cudaGetDeviceProperties(&properties, config.device_ordinal));
    const auto arch = "sm_" + std::to_string(properties.major) + std::to_string(properties.minor);
    if (robot.arch != arch || robot.num_dof < 1 || robot.num_dof > 64) {
      throw std::invalid_argument("timeopt model architecture or DoF is unsupported");
    }
    const auto tables = curobo_robot_runtime::load_timeopt_manifest(directory, config.num_knots,
                                                                     config.reference_timing);
    const int spans = config.num_knots + 2 + config.reference_timing;
    if (tables.num_dof != robot.num_dof || tables.num_knots != config.num_knots ||
        tables.reference != static_cast<int>(config.reference_timing) ||
        tables.intervals != config.num_intervals ||
        tables.points != config.num_intervals + 1 + 2 * (spans - 1) + 28 || tables.tiles < 1 ||
        tables.tiles > tables.points * robot.num_dof + 32 + config.num_intervals + 1) {
      throw std::invalid_argument("timeopt tables do not match the model and configuration");
    }
    w_.batch = config.num_seeds;
    w_.dof = robot.num_dof;
    w_.knots = config.num_knots;
    w_.points = tables.points;
    w_.intervals = config.num_intervals;
    w_.tiles = tables.tiles;
    w_.seed_tiles = (w_.points * w_.dof + 255) / 256;
    w_.start_offset = !config.reference_timing;
    w_.iterations = config.max_iterations;
    w_.samples = config.sample_count;
    const auto load = [&](const char *name, auto **destination, size_t count) {
      using Value = std::remove_pointer_t<std::remove_reference_t<decltype(*destination)>>;
      const auto &tensor = tables.tensors.at(name);
      const char *dtype = std::is_same_v<Value, int> ? "int32" : "float32";
      if (tensor.bytes != count * sizeof(Value) || tensor.dtype != dtype) {
        throw std::runtime_error(std::string("invalid timeopt table: ") + name);
      }
      buffers_.emplace_back();
      buffers_.back().upload(curobo_robot_runtime::load_tensor_data(directory, tensor, name));
      *destination = reinterpret_cast<Value *>(buffers_.back().ptr());
    };
    load("basis", &w_.basis, 12 * w_.points);
    load("point_first", &w_.point_first, w_.points);
    load("meta", &w_.meta, w_.tiles * 64 * 2);
    load("offsets", &w_.offsets, w_.tiles);
    load("starts", &w_.starts, 30);
    load("time_basis", &w_.time_basis, w_.intervals * 6 * 4);
    load("time_first", &w_.time_first, w_.intervals * 6);
    load("time_weights", &w_.time_weights, 6);
    load("segments", &w_.segments, w_.points);
    load("local", &w_.local, w_.points);
    load("polynomial", &w_.polynomial, 29 * 4 * 32);
    load("nodes", &w_.nodes, 6);
    const auto trajopt = curobo_robot_runtime::load_trajectory_optimization_manifest(directory);
    const auto rollout = curobo_robot_runtime::load_rollout_manifest(directory);
    if (trajopt.num_dof != w_.dof || rollout.num_dof != w_.dof) {
      throw std::invalid_argument("timeopt dynamics and limits have mismatched DoF");
    }
    const auto limit = [&](const auto &tensor, float scale, float **destination) {
      const auto bounds = read_floats(directory, tensor, 2 * w_.dof);
      std::vector<float> values(w_.dof);
      for (int j = 0; j < w_.dof; ++j) {
        values[j] = std::min(-bounds[j], bounds[w_.dof + j]) * scale;
        if (!std::isfinite(values[j]) || values[j] <= 0) {
          throw std::invalid_argument("timeopt requires finite positive derivative limits");
        }
      }
      *destination = upload(values);
    };
    limit(trajopt.velocity_bounds, config.limit_scale, &w_.vlim);
    limit(trajopt.acceleration_bounds, config.limit_scale, &w_.alim);
    limit(trajopt.jerk_bounds, config.jerk_limit_scale, &w_.jlim);
    const auto effort = read_floats(directory, rollout.effort_bounds, 2 * w_.dof);
    std::vector<float> lower(w_.dof), upper(w_.dof);
    for (int j = 0; j < w_.dof; ++j) {
      lower[j] = std::max(effort[j], effort[j] * config.limit_scale);
      upper[j] = std::min(effort[w_.dof + j], effort[w_.dof + j] * config.limit_scale);
      if (!std::isfinite(lower[j]) || !std::isfinite(upper[j]) || lower[j] > upper[j]) {
        throw std::invalid_argument("timeopt requires finite ordered effort limits");
      }
    }
    w_.lower = upload(lower);
    w_.upper = upload(upper);
    if (shared_rollout == nullptr) {
      owned_rollout_ = std::make_unique<RolloutRuntime>(
          directory, config.device_ordinal, robot, RolloutProfile::TRAJOPT);
      shared_rollout = owned_rollout_.get();
    }
    rollout_ = shared_rollout;
    allocate_workspace();
    CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
  }

  ~TrajectoryTimeOptRuntime() {
    cudaSetDevice(config_.device_ordinal);
    for (auto graph : {solve_graph_, sample_graph_})
      if (graph) cudaGraphExecDestroy(graph);
    if (capture_stream_) cudaStreamDestroy(capture_stream_);
  }

  const TrajectoryTimeOptConfig &config() const { return config_; }
  int num_dof() const { return w_.dof; }

  TrajectoryTimeOptResult timeopt(const TrajectoryTimeOptRequest &r) {
    if (!r.knots || !r.start_position || !r.goal_position ||
        config_.reference_timing != (r.knot_dt != nullptr) ||
        (!config_.reference_timing && (r.start_velocity || r.start_acceleration || r.start_jerk))) {
      throw std::invalid_argument("timeopt inputs do not match the reference timing contract");
    }
    CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
    const size_t boundary = static_cast<size_t>(w_.batch) * w_.dof;
    copy(w_.input_knots, r.knots, boundary * w_.knots, r.stream);
    copy(w_.input_start, r.start_position, boundary, r.stream);
    copy(w_.input_goal, r.goal_position, boundary, r.stream);
    copy(w_.input_velocity, r.start_velocity, boundary, r.stream);
    copy(w_.input_acceleration, r.start_acceleration, boundary, r.stream);
    copy(w_.input_jerk, r.start_jerk, boundary, r.stream);
    copy(w_.input_dt, r.knot_dt, w_.batch, r.stream);
    execute(false, r.stream);
    has_result_ = true;
    const size_t plane = boundary * (w_.intervals + 1);
    return {w_.output_timing,
            w_.output_time,
            w_.duration,
            w_.output_joints,
            w_.output_joints + plane,
            w_.output_joints + 2 * plane,
            w_.output_joints + 3 * plane,
            w_.output_torque,
            w_.output_uniform,
            reinterpret_cast<uint8_t *>(w_.success),
            w_.status};
  }

  TrajectoryTimeOptSamples sample(const float *times, cudaStream_t stream) {
    if (!has_result_ || !times)
      throw std::invalid_argument("sample requires a timeopt and CUDA times");
    CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
    copy(w_.sample_times, times, static_cast<size_t>(w_.batch) * w_.samples, stream);
    execute(true, stream);
    const size_t plane = static_cast<size_t>(w_.batch) * w_.samples * w_.dof;
    return {w_.sample_joints, w_.sample_joints + plane, w_.sample_joints + 2 * plane,
            w_.sample_joints + 3 * plane};
  }

  void update_links_inertial(const LinkInertialUpdates &updates) {
    CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
    rollout_->update_links_inertial(updates);
    has_result_ = false;
  }

 private:
  template <typename Value = float>
  Value *allocate(size_t count) {
    buffers_.emplace_back();
    buffers_.back().allocate(count * sizeof(Value));
    return reinterpret_cast<Value *>(buffers_.back().ptr());
  }

  float *upload(const std::vector<float> &values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    buffers_.emplace_back();
    buffers_.back().upload(bytes);
    return reinterpret_cast<float *>(buffers_.back().ptr());
  }

  static void copy(float *out, const float *in, size_t count, cudaStream_t stream) {
    if (in)
      CRR_CUDA_CHECK(
          cudaMemcpyAsync(out, in, count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    else
      CRR_CUDA_CHECK(cudaMemsetAsync(out, 0, count * sizeof(float), stream));
  }

  void allocate_workspace() {
    const size_t b = w_.batch, boundary = b * w_.dof, path = boundary * w_.points;
    const size_t records = b * w_.tiles * 64;
    w_.input_knots = allocate(boundary * w_.knots);
    w_.input_start = allocate(boundary);
    w_.input_goal = allocate(boundary);
    w_.input_velocity = allocate(boundary);
    w_.input_acceleration = allocate(boundary);
    w_.input_jerk = allocate(boundary);
    w_.input_dt = allocate(b);
    w_.controls = allocate(boundary * w_.knots);
    w_.start = allocate(4 * boundary);
    w_.goal = allocate(boundary);
    w_.reference_speed = allocate(2 * b);
    w_.input_valid = allocate<bool>(b);
    w_.stationary = allocate<bool>(b);
    w_.path = allocate(4 * path);
    w_.q = allocate(3 * path);
    w_.qd = allocate(3 * path);
    w_.qdd = allocate(3 * path);
    w_.tau = allocate(3 * path);
    w_.coefficients = allocate(3 * path);
    w_.seed_bounds = allocate(b * w_.seed_tiles * 3);
    w_.output_bounds = allocate(b * w_.seed_tiles * 3);
    w_.rows = allocate(16 * records);
    w_.bounds = allocate(2 * records);
    w_.jacobian = allocate(4 * records);
    w_.row_state = allocate(28 * records);
    w_.partials = allocate(b * w_.tiles * 16);
    w_.corrector_terms = allocate(b * 2 * 32);
    w_.step_bounds = allocate(b * w_.tiles * 2);
    w_.candidate_masks = allocate<int>(b * w_.tiles * 4);
    w_.state = allocate(b * 336);
    w_.timing = allocate(b * 32);
    w_.uniform = allocate(b);
    w_.feasible = allocate<bool>(b);
    w_.numerical = allocate<bool>(b);
    w_.output_timing = allocate(b * 32);
    w_.output_uniform = allocate(b);
    w_.output_joints = allocate(4 * boundary * (w_.intervals + 1));
    w_.output_torque = allocate(boundary * (w_.intervals + 1));
    w_.output_time = allocate(b * (w_.intervals + 1));
    w_.duration = allocate(b);
    w_.dt = allocate(b * w_.intervals);
    w_.output_numerical = allocate<bool>(b);
    w_.success = allocate<bool>(b);
    w_.status = allocate<int>(b);
    w_.sample_times = allocate(b * w_.samples);
    w_.sample_joints = allocate(4 * boundary * w_.samples);
    dynamics_cache_ =
        allocate(3 * b * w_.points * rollout_->num_links() * rollout_->cache_floats_per_link());
  }

  void launch(bool sampling, cudaStream_t stream) {
    if (sampling) {
      CRR_CUDA_CHECK(launch_timeopt_sample(w_, stream));
      return;
    }
    CRR_CUDA_CHECK(launch_timeopt_path(w_, stream));
    rollout_->launch_inverse_dynamics(
        reinterpret_cast<CUdeviceptr>(w_.q), reinterpret_cast<CUdeviceptr>(w_.qd),
        reinterpret_cast<CUdeviceptr>(w_.qdd), reinterpret_cast<CUdeviceptr>(w_.tau),
        reinterpret_cast<CUdeviceptr>(dynamics_cache_), w_.batch * 3 * w_.points,
        reinterpret_cast<CUstream>(stream));
    CRR_CUDA_CHECK(launch_timeopt_solve(w_, stream));
    CRR_CUDA_CHECK(cudaMemcpy2DAsync(w_.duration, sizeof(float), w_.output_time + w_.intervals,
                                     (w_.intervals + 1) * sizeof(float), sizeof(float), w_.batch,
                                     cudaMemcpyDeviceToDevice, stream));
  }

  void execute(bool sampling, cudaStream_t stream) {
    cudaStreamCaptureStatus status{};
    if (stream) CRR_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (!config_.use_cuda_graph || status != cudaStreamCaptureStatusNone) {
      launch(sampling, stream);
      return;
    }
    auto &executable = sampling ? sample_graph_ : solve_graph_;
    if (!executable) {
      cudaGraph_t graph = nullptr;
      const auto capture_stream = stream ? stream : capture_stream_;
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      CRR_CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));
      try {
        launch(sampling, capture_stream);
        CRR_CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
        CRR_CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
        CRR_CUDA_CHECK(cudaGraphDestroy(graph));
      } catch (...) {
        cudaStreamCaptureStatus capture{};
        cudaStreamIsCapturing(capture_stream, &capture);
        if (capture != cudaStreamCaptureStatusNone) cudaStreamEndCapture(capture_stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        throw;
      }
    }
    CRR_CUDA_CHECK(cudaGraphLaunch(executable, stream));
  }

  TrajectoryTimeOptConfig config_;
  TimeOptWorkspace w_{};
  std::vector<DeviceBuffer> buffers_;
  std::unique_ptr<RolloutRuntime> owned_rollout_;
  RolloutRuntime* rollout_{nullptr};
  float *dynamics_cache_{nullptr};
  cudaGraphExec_t solve_graph_{nullptr}, sample_graph_{nullptr};
  cudaStream_t capture_stream_{nullptr};
  bool has_result_{false};
};
}  // namespace curobo::robot::detail

namespace curobo::robot {
TrajectoryTimeOpt::TrajectoryTimeOpt(const std::string &directory,
                                     const TrajectoryTimeOptConfig &config)
    : runtime_(std::make_unique<detail::TrajectoryTimeOptRuntime>(directory, config)) {}
TrajectoryTimeOpt::TrajectoryTimeOpt(const std::string& directory,
                                     const TrajectoryTimeOptConfig& config,
                                     detail::RolloutRuntime& rollout)
    : runtime_(std::make_unique<detail::TrajectoryTimeOptRuntime>(directory, config, &rollout)) {}
TrajectoryTimeOpt::~TrajectoryTimeOpt() = default;
TrajectoryTimeOpt::TrajectoryTimeOpt(TrajectoryTimeOpt &&) noexcept = default;
TrajectoryTimeOpt &TrajectoryTimeOpt::operator=(TrajectoryTimeOpt &&) noexcept = default;
const TrajectoryTimeOptConfig &TrajectoryTimeOpt::config() const { return runtime_->config(); }
int TrajectoryTimeOpt::num_dof() const { return runtime_->num_dof(); }
TrajectoryTimeOptResult TrajectoryTimeOpt::timeopt(const TrajectoryTimeOptRequest &request) const {
  return runtime_->timeopt(request);
}
TrajectoryTimeOptSamples TrajectoryTimeOpt::sample(const float *times, cudaStream_t stream) const {
  return runtime_->sample(times, stream);
}
void TrajectoryTimeOpt::update_links_inertial(const LinkInertialUpdates &updates) {
  runtime_->update_links_inertial(updates);
}
}  // namespace curobo::robot
