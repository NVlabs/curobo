// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NVRTC also concatenates this file into its cache-key compilation unit.
#ifndef CUROBO_LBFGS_CANDIDATES_CUH
#define CUROBO_LBFGS_CANDIDATES_CUH

namespace curobo {
namespace optimization {

// Each block owns one problem. The direction has already been limited, and
// each thread reads the same variables it wrote in the direction epilogue.
template<typename ScalarType>
__forceinline__ __device__ void write_lbfgs_candidates(
    const ScalarType* position, const ScalarType* direction,
    const ScalarType* scales, const ScalarType* beta, ScalarType* candidates,
    int dimension, int count) {
  const int batch = blockIdx.x;
  const ScalarType multiplier = beta[batch];
  for (int variable = threadIdx.x; variable < dimension; variable += blockDim.x) {
    const ScalarType origin = position[batch * dimension + variable];
    const ScalarType step = direction[batch * dimension + variable];
    for (int candidate = 0; candidate < count; ++candidate) {
      const ScalarType scale = multiplier * scales[candidate];
      candidates[(batch * count + candidate) * dimension + variable] = origin + scale * step;
    }
  }
}

}  // namespace optimization
}  // namespace curobo

#endif
