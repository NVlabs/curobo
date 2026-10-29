// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "curobo_robot_runtime/api.h"

namespace curobo::robot {

namespace detail {
class TrajectorySeedFitterRuntime;
}

/** Fixed lane count and CUDA device for trajectory seed fitting. */
struct TrajectorySeedFitterConfig {
  int device_ordinal{0};
  int num_seeds{4};
  bool use_cuda_graph{true};
};

/** Fixed B-spline dimensions and production IRLS iteration count. */
struct TrajectorySeedFitterInfo {
  int num_dof{0};
  int num_knots{0};
  int trajectory_steps{0};
  int irls_iterations{0};
};

/**
 * CUDA tensors for fixed-shape time-optimal B-spline seed fitting.
 *
 * Target positions have shape `[num_seeds, trajectory_steps, num_dof]`.
 * Start position, optional start derivatives, and goal position have shape
 * `[num_seeds, num_dof]`. Null derivative pointers mean zero derivatives.
 * Output knots have shape `[num_seeds, num_knots, num_dof]`; required dt has
 * shape `[num_seeds]`.
 */
struct TrajectorySeedFitterRequest {
  const float* target_position{nullptr};
  const float* start_position{nullptr};
  const float* start_velocity{nullptr};
  const float* start_acceleration{nullptr};
  const float* start_jerk{nullptr};
  const float* goal_position{nullptr};
  float* fitted_knots{nullptr};
  float* required_dt{nullptr};
  cudaStream_t stream{nullptr};
};

/**
 * Standalone production-parity time-optimal B-spline seed fitter.
 *
 * The fitter uses the B-spline bases exported from Python's
 * `BSplineTimeOptimalSeedFitter`, including its arc-length target,
 * acceleration-regularized geometric initialization, fixed IRLS iterations,
 * derivative-limit timestep objective, and improvement-only acceptance.
 * Calls are asynchronous and CUDA graph capturable. One instance must not
 * execute concurrent calls.
 */
class TrajectorySeedFitter final {
 public:
  explicit TrajectorySeedFitter(
      const std::string& artifact_directory,
      const TrajectorySeedFitterConfig& config = {});
  ~TrajectorySeedFitter();

  TrajectorySeedFitter(const TrajectorySeedFitter&) = delete;
  TrajectorySeedFitter& operator=(const TrajectorySeedFitter&) = delete;
  TrajectorySeedFitter(TrajectorySeedFitter&& other) noexcept;
  TrajectorySeedFitter& operator=(TrajectorySeedFitter&& other) noexcept;

  CuroboRobotInfo robot_info() const;
  TrajectorySeedFitterInfo info() const;
  const TrajectorySeedFitterConfig& config() const;

  /** Enqueue fixed-shape arc-length fitting and IRLS refinement. */
  void fit(const TrajectorySeedFitterRequest& request) const;

 private:
  std::unique_ptr<detail::TrajectorySeedFitterRuntime> runtime_;
};

}  // namespace curobo::robot
