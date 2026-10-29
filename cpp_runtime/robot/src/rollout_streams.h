// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>

namespace curobo::robot::detail {

/** Persistent CUDA streams used by every native robot rollout. */
enum class RolloutStream : std::size_t {
  DYNAMICS,
  KINEMATICS,
  CSPACE,
  POSE,
  WORLD,
  SELF,
  COUNT,
};

/** Dependency events for the parallel rollout dataflow. */
enum class RolloutEvent : std::size_t {
  STATE_READY,
  DYNAMICS_READY,
  KINEMATICS_READY,
  CSPACE_DONE,
  POSE_DONE,
  WORLD_DONE,
  SELF_DONE,
  KINEMATICS_BACKWARD_DONE,
  COUNT,
};

/** Own fixed streams and events so CUDA Graph captures remain reusable. */
class RolloutStreams final {
 public:
  RolloutStreams();
  ~RolloutStreams();

  RolloutStreams(const RolloutStreams&) = delete;
  RolloutStreams& operator=(const RolloutStreams&) = delete;

  cudaStream_t cuda_stream(RolloutStream stream) const;
  CUstream driver_stream(RolloutStream stream) const;
  void record(RolloutEvent event, cudaStream_t stream) const;
  void wait(cudaStream_t stream, RolloutEvent event) const;

 private:
  void destroy() noexcept;

  static constexpr std::size_t STREAM_COUNT =
      static_cast<std::size_t>(RolloutStream::COUNT);
  static constexpr std::size_t EVENT_COUNT =
      static_cast<std::size_t>(RolloutEvent::COUNT);
  std::array<cudaStream_t, STREAM_COUNT> streams_{};
  std::array<cudaEvent_t, EVENT_COUNT> events_{};
};

}  // namespace curobo::robot::detail
