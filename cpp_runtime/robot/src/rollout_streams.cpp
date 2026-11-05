// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rollout_streams.h"

#include "cuda_check.h"

namespace curobo::robot::detail {
namespace {

template <typename EnumType>
std::size_t index_of(EnumType value) {
  return static_cast<std::size_t>(value);
}

}  // namespace

RolloutStreams::RolloutStreams() {
  try {
    for (auto& stream : streams_) {
      CRR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    for (auto& event : events_) {
      CRR_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
  } catch (...) {
    destroy();
    throw;
  }
}

RolloutStreams::~RolloutStreams() { destroy(); }

cudaStream_t RolloutStreams::cuda_stream(RolloutStream stream) const {
  return streams_.at(index_of(stream));
}

CUstream RolloutStreams::driver_stream(RolloutStream stream) const {
  return reinterpret_cast<CUstream>(cuda_stream(stream));
}

void RolloutStreams::record(RolloutEvent event, cudaStream_t stream) const {
  CRR_CUDA_CHECK(cudaEventRecord(events_.at(index_of(event)), stream));
}

void RolloutStreams::wait(cudaStream_t stream, RolloutEvent event) const {
  CRR_CUDA_CHECK(
      cudaStreamWaitEvent(stream, events_.at(index_of(event)), 0));
}

void RolloutStreams::destroy() noexcept {
  for (const auto stream : streams_) {
    if (stream != nullptr) {
      cudaStreamSynchronize(stream);
    }
  }
  for (auto& event : events_) {
    if (event != nullptr) {
      cudaEventDestroy(event);
      event = nullptr;
    }
  }
  for (auto& stream : streams_) {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
}

}  // namespace curobo::robot::detail
