// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#include <string>

#include "curobo_robot_runtime/inertial.h"

namespace curobo::robot {
namespace detail {
class TrajectoryTimeOptRuntime;
class TrajectoryOptimizerRuntime;
class RolloutRuntime;
}

/** Fixed shape for cubic-path timeopt with 32 timing controls. */
struct TrajectoryTimeOptConfig {
  int device_ordinal{0};
  int num_seeds{4};
  int num_knots{16};
  /** Shared duration/constraint intervals (512 nodes by default), at least one. */
  int num_intervals{511};
  int max_iterations{10};
  int sample_count{1000};
  float limit_scale{0.95F};
  float jerk_limit_scale{0.95F};
  /** Preserve TrajOpt's start span and initialize from a supplied knot_dt.
   * Disable for fitted static RRT paths without reference timing. */
  bool reference_timing{true};
  bool use_cuda_graph{true};
};

/** Contiguous CUDA float32 inputs. Knots: [B,K,D], boundaries: [B,D], dt: [B].
 * Null derivatives mean zero. Reference timing requires knot_dt. Moving starts
 * require reference timing and a feasible reference, as in the Python timeopt. */
struct TrajectoryTimeOptRequest {
  const float *knots{nullptr};
  const float *start_position{nullptr};
  const float *goal_position{nullptr};
  const float *knot_dt{nullptr};
  const float *start_velocity{nullptr};
  const float *start_acceleration{nullptr};
  const float *start_jerk{nullptr};
  cudaStream_t stream{nullptr};
};

/** Borrowed CUDA buffers, overwritten by the next timeopt on this instance.
 * States/torque: [B,num_intervals+1,D], time: [B,num_intervals+1], timing: [B,32].
 * Success is sampled dynamic feasibility; collision validation is separate. */
struct TrajectoryTimeOptResult {
  const float *timing_controls{nullptr};
  const float *time{nullptr};
  const float *duration{nullptr};
  const float *position{nullptr};
  const float *velocity{nullptr};
  const float *acceleration{nullptr};
  const float *jerk{nullptr};
  const float *torque{nullptr};
  const float *uniform_duration{nullptr};
  const uint8_t *success{nullptr};
  const int32_t *status{nullptr};
};

/** Borrowed sampled states [B,sample_count,D], valid until the next sample call. */
struct TrajectoryTimeOptSamples {
  const float *position{nullptr};
  const float *velocity{nullptr};
  const float *acceleration{nullptr};
  const float *jerk{nullptr};
};

/** Native CUDA TimeOpt using the same kernels and tables as the Python solver.
 * Calls are asynchronous and support external CUDA graph capture. One instance
 * must not execute concurrently on multiple streams. Warmup/capture is excluded
 * from steady-state timing. Null streams replay on the CUDA default stream. */
class TrajectoryTimeOpt final {
 public:
  explicit TrajectoryTimeOpt(const std::string &artifact_directory,
                             const TrajectoryTimeOptConfig &config = {});
  ~TrajectoryTimeOpt();
  TrajectoryTimeOpt(const TrajectoryTimeOpt &) = delete;
  TrajectoryTimeOpt &operator=(const TrajectoryTimeOpt &) = delete;
  TrajectoryTimeOpt(TrajectoryTimeOpt &&) noexcept;
  TrajectoryTimeOpt &operator=(TrajectoryTimeOpt &&) noexcept;
  const TrajectoryTimeOptConfig &config() const;
  int num_dof() const;
  TrajectoryTimeOptResult timeopt(const TrajectoryTimeOptRequest &request) const;
  /** Sample seconds [B,sample_count]; out-of-range times hold the endpoint. */
  TrajectoryTimeOptSamples sample(const float *times, cudaStream_t stream = nullptr) const;
  void update_links_inertial(const LinkInertialUpdates &updates);

 private:
  friend class detail::TrajectoryOptimizerRuntime;
  /** Borrow the owning optimizer's dynamics model and kernel modules. */
  TrajectoryTimeOpt(const std::string& artifact_directory,
                    const TrajectoryTimeOptConfig& config,
                    detail::RolloutRuntime& rollout);
  std::unique_ptr<detail::TrajectoryTimeOptRuntime> runtime_;
};
}  // namespace curobo::robot
