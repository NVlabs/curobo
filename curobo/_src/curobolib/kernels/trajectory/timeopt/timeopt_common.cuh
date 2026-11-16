// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace curobo::trajectory::timeopt {

constexpr int TIMEOPT_THREADS = 256;
constexpr int TIMEOPT_WARPS = TIMEOPT_THREADS / 32;
constexpr int TIMEOPT_CONTROLS = 32;
constexpr unsigned FULL_WARP = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(FULL_WARP, value, offset);
  return value;
}

__device__ __forceinline__ float warp_min(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value = fminf(value, __shfl_down_sync(FULL_WARP, value, offset));
  return value;
}

// Fixed-size, deterministic two-level reduction; no contended floating atomics.
template <int Values>
__device__ __forceinline__ void reduce_sums(float (&values)[Values], float *out, float *scratch) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int k = 0; k < Values; ++k) {
    const float sum = warp_sum(values[k]);
    if (lane == 0) scratch[k * TIMEOPT_WARPS + warp] = sum;
  }
  __syncthreads();
// Each eight-lane group reduces one output, preserving the original sum tree.
#pragma unroll
  for (int first = 0; first < Values; first += TIMEOPT_THREADS / TIMEOPT_WARPS) {
    const int k = first + threadIdx.x / TIMEOPT_WARPS;
    float sum = k < Values ? scratch[k * TIMEOPT_WARPS + lane % TIMEOPT_WARPS] : 0.0f;
#pragma unroll
    for (int offset = TIMEOPT_WARPS / 2; offset; offset >>= 1)
      sum += __shfl_down_sync(FULL_WARP, sum, offset, TIMEOPT_WARPS);
    if (k < Values && lane % TIMEOPT_WARPS == 0) out[k] = sum;
  }
}

__device__ __forceinline__ float dot4(float4 a, const float *y) {
  return a.x * y[0] + a.y * y[1] + a.z * y[2] + a.w * y[3];
}

constexpr int TIMEOPT_RECORDS = TIMEOPT_THREADS / 4;
constexpr int PARTIALS_PER_TILE = 16;
constexpr int LINE_SEARCH_STEPS = 8;
enum TimeOptRecord { JOINT_LIMITS = 0, POSITIVE_CONTROL = 1, DURATION = 2, PADDING = 3 };

// Float offsets in one path's state; mirrored by _StateOffset in the Python driver.
// Scalars occupy [0,16), vectors reserve 32 entries, and the lower band reserves
// four entries per control.
enum FusedState {
  F_SPEED_SCALE = 0,     // Physical squared speed = scale * spline(controls).
  F_FIXED_CONTROLS = 1,  // Two for a reference time law, zero for a static seed.
  F_BEST_COST = 2,       // Duration before applying the squared-speed scale.
  F_PRIMAL_STEP = 3,
  F_DUAL_STEP = 4,
  F_ACCEPTED = 5,
  F_ITERATION_VALID = 6,
  F_NUMERICAL_OK = 7,
  F_GAP = 9,
  F_CENTERING = 10,  // sigma * gap, not sigma alone.
  F_CONTROLS = 16,
  F_BEST_CONTROLS = 48,
  F_GRADIENT = 80,
  F_DIAGONAL_SCALE = 112,
  F_PREDICTOR = 144,
  F_CORRECTOR = 176,
  F_FACTOR = 208,
  F_SIZE = 336
};

// Each row-state field contains seven constraint planes: squared velocity,
// positive/negative acceleration, positive/negative torque, positive/negative jerk.
// Positive-control records use only the first constraint plane.
enum RowState {
  ROW_SLACK = 0,
  ROW_DUAL = 7,
  ROW_AFFINE_PRODUCT = 14,  // Predictor slack direction * predictor dual direction.
  ROW_DUAL_DIRECTION = 21
};

// The same 16-float tile buffer is reused after the predictor solve. Assembly
// writes four gradient entries and ten packed lower-triangle Hessian entries.
enum AssemblyPartial { ASSEMBLY_GRADIENT = 0, ASSEMBLY_HESSIAN = 4, ASSEMBLY_INVALID = 15 };

// Corrector RHS = -gradient + affine_rhs - centering * centering_rhs.
// The four moments are sums of s*dual, ds*dual, s*ddual, and ds*ddual.
enum AffinePartial {
  AFFINE_RHS = 0,
  AFFINE_CENTERING_RHS = 4,
  AFFINE_PRIMAL_BOUND = 8,
  AFFINE_DUAL_BOUND = 9,
  AFFINE_MOMENTS = 10
};

__device__ __forceinline__ float fused_infinity() { return __int_as_float(0x7f800000); }

__device__ __forceinline__ float fused_block_min(float value, float *scratch) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = warp_min(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = warp_min(lane < TIMEOPT_WARPS ? scratch[lane] : fused_infinity());
  return __shfl_sync(FULL_WARP, value, 0);
}

__device__ __forceinline__ float fused_block_sum(float value, float *scratch) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = warp_sum(lane < TIMEOPT_WARPS ? scratch[lane] : 0.0f);
  return __shfl_sync(FULL_WARP, value, 0);
}

// The block mask is valid in thread zero; callers publish or broadcast it.
__device__ __forceinline__ int fused_block_mask(int mask, int *scratch) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int offset = 16; offset; offset >>= 1) mask |= __shfl_down_sync(FULL_WARP, mask, offset);
  if (lane == 0) scratch[warp] = mask;
  __syncthreads();
  if (warp == 0) {
    mask = lane < TIMEOPT_WARPS ? scratch[lane] : 0;
    for (int offset = 16; offset; offset >>= 1) mask |= __shfl_down_sync(FULL_WARP, mask, offset);
  }
  return mask;
}

__device__ __forceinline__ float fused_boundary(float value, float delta) {
  return delta < 0.0f ? -value / fminf(delta, -1e-30f) : fused_infinity();
}

}  // namespace curobo::trajectory::timeopt
