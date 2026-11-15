// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace curobo_robot_runtime {

/** A contiguous FP32 cost buffer with a compile-time per-trial layout. */
template <int Length, int TrialStride>
struct CostTerm {
  static_assert(Length >= 0 && TrialStride >= 0, "cost layout must be nonnegative");
  const float *values;
};

template <int Threads, int Length, int Stride>
__device__ __forceinline__ float sum_cost_term(CostTerm<Length, Stride> term, int trial) {
  float sum = 0.0F;
  constexpr int chunks = (Length + Threads - 1) / Threads;
  if constexpr (chunks <= 8) {
#pragma unroll
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const int index = threadIdx.x + chunk * Threads;
      if (index < Length) sum += term.values[trial * Stride + index];
    }
  } else {
    for (int index = threadIdx.x; index < Length; index += Threads)
      sum += term.values[trial * Stride + index];
  }
  return sum;
}

/** Sum any fixed set of cost terms, using one block per trial. */
template <int Threads, typename... Terms>
__global__ void reduce_costs(float *output, int trials, Terms... terms) {
  static_assert(Threads >= 32 && Threads <= 1024 && Threads % 32 == 0,
                "reduction requires whole warps");
  static_assert(sizeof...(Terms) > 0, "at least one cost term is required");
  const int trial = blockIdx.x;
  if (trial >= trials) return;
  float sum = (0.0F + ... + sum_cost_term<Threads>(terms, trial));
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
  constexpr int warps = Threads / 32;
  const int lane = threadIdx.x % 32;
  if constexpr (warps > 1) {
    __shared__ float partials[warps];
    if (lane == 0) partials[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x >= 32) return;
    sum = lane < warps ? partials[lane] : 0.0F;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) output[trial] = sum;
}

}  // namespace curobo_robot_runtime
