// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace curobo::robot {
/** Borrowed physical seconds per cardinal span; one float on the value's device. */
struct UniformTiming {
  const float *knot_dt{nullptr};
};
/** Physical squared speed on the clamped cubic 32-control basis on [0,1]. */
struct TimeOptTiming {
  const float *controls{nullptr};  // [32]
  const float *time{nullptr};      // [time_points] at uniform progress nodes
  int time_points{0};
};
/** Owning interpolated device states, including the exact final endpoint. */
struct InterpolatedTrajectory {
  std::shared_ptr<void> storage;
  const float *position{nullptr};
  const float *velocity{nullptr};
  const float *acceleration{nullptr};
  const float *jerk{nullptr};
  int sample_count{0}, num_dof{0};
  float dt{0};  // Zero for a one-sample stationary trajectory.
  std::vector<std::string> joint_names;
};
/** Complete uniform cardinal cubic path with uniform or TimeOpt timing.
 * Coordinates are [num_controls,num_dof], including boundary support. The
 * active path has num_controls-3 spans. Device -1 denotes CPU; nonnegative
 * values denote CUDA ordinals. CPU values support controller handoff.
 * Borrowed inputs must outlive this value; clone() retains all buffers.
 * Checked interpolation/copies may synchronize and belong outside capture.
 */
class SplineTrajectory {
 public:
  using Timing = std::variant<UniformTiming, TimeOptTiming>;
  SplineTrajectory(const float *controls, int count, std::vector<std::string> names, Timing timing,
                   int device, std::shared_ptr<void> owner = {});
  const float *control_points() const { return controls_; }
  int num_controls() const { return count_; }
  int num_dof() const { return static_cast<int>(names_.size()); }
  int degree() const { return 3; }
  int device() const { return device_; }
  const std::vector<std::string> &joint_names() const { return names_; }
  const Timing &timing() const { return timing_; }
  /** Checked physical duration in seconds; reads device metadata as needed. */
  float duration(cudaStream_t stream = nullptr) const;
  SplineTrajectory clone(cudaStream_t stream = nullptr) const;
  /** Same-device transfer may alias; clone() always owns an independent copy. */
  SplineTrajectory to(int device, cudaStream_t stream = nullptr) const;
  /** One CUDA evaluation of the full trajectory. dt is the maximum interval. */
  InterpolatedTrajectory interpolate(float dt, cudaStream_t stream = nullptr) const;

 private:
  SplineTrajectory copy_to(int device, cudaStream_t stream) const;
  const float *controls_;
  int count_, device_;
  std::vector<std::string> names_;
  Timing timing_;
  std::shared_ptr<void> owner_;
};
}  // namespace curobo::robot
