// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Vector-free L-BFGS step
 * ---------------------
 * This kernel fuses history insertion, inverse-Hessian application, direction
 * limiting, and line-search candidate generation for one optimization problem
 * per block. It evaluates the usual two-loop recursion through cached scalar
 * products, avoiding a dependent vector reduction for every history entry.
 * The full vectors are still stored and used to reconstruct the direction.
 * The vector-free formulation was motivated by OpenRoboKit's L-BFGS design:
 * https://github.com/openrobokit/robokit
 *
 * Execution and storage:
 * - Launch exactly 128 threads (four full warps). Items is 1 for dimensions
 *   1..128 and 2 for 129..256; out-of-range components contribute zero.
 * - History is specialized at compile time for 1..30 entries. Each warp lane
 *   owns one scalar history entry; the second product-reduction group also
 *   needs two lanes for the initial inverse-Hessian scale, hence the limit 30.
 * - Persistent s/y vectors use [physical_slot, batch, variable] layout; rho
 *   uses [physical_slot, batch]. head is the next insertion slot. The local
 *   view indexes valid pairs newest first, without moving the global ring.
 * - cross[batch, slot_s, slot_y] caches s_slot_s^T y_slot_y. Only products
 *   whose s pair is older than the y pair are needed by the recurrences.
 * - S remains in registers. SharedHistory stages Y and the scalar product
 *   matrix in shared memory to reduce register pressure; otherwise both stay
 *   in registers. The shared matrix has a padded 32x33 layout to avoid bank
 *   conflicts. LBFGSVectorFreeLaunchCfg selects storage for the target GPU;
 *   the host dispatch uses the two-loop implementation outside these bounds.
 *
 * Step phases:
 * 1. Form the applied displacement s = position - previous_position and the
 *    gradient difference y = gradient - previous_gradient. Batch s^T y,
 *    s^T g, S^T g, and S^T y into 32-product tiles. reduce_products transposes
 *    the reduction so lane j receives product j, then combines all four warps
 *    through shared scratch. This group takes one or two tiles as History grows.
 * 2. Insert the pair only when allowed and both curvature s^T y and its
 *    reciprocal are finite and positive. Update the new cache column and ring
 *    metadata; a full ring drops its oldest pair. Rejected pairs leave history
 *    intact. Refresh previous_position/previous_gradient on either outcome.
 * 3. For logical history i = 0..count-1 (newest first), compute
 *      alpha_i = rho_i * (s_i^T g - sum_{j<i} alpha_j * s_i^T y_j).
 *    Cached products and warp shuffles replace per-pair vector reductions.
 * 4. Reconstruct q = g - sum_i alpha_i*y_i. In one more 32-product reduction,
 *    compute every y_i^T q and the newest pair's scaling products. The initial
 *    inverse-Hessian scale gamma is s^T y / y^T y, or sqrt(s^T s / y^T y) for
 *    Geometric. Empty history uses epsilon; stable_mode also guards invalid
 *    scales. Apply the reverse scalar recurrence (oldest first), then form
 *      r = gamma*q + sum_i (alpha_i - beta_i)*s_i,
 *    where beta_i is the usual two-loop coefficient, not the beta input buffer.
 * 5. Negate r, optionally apply one common scale to respect per-variable step
 *    limits, and write candidates = position + beta[batch]*scales[k]*direction.
 *
 * A STATUS_BACKTRACK invocation only regenerates candidates from the existing
 * direction; it does not update history or the previous iterate. Cached products
 * must remain consistent with the ring: resets invalidate them, and an MPC
 * history shift must rebuild them before this kernel is used again.
 */
#pragma once
#include "optimization/lbfgs/lbfgs_candidates.cuh"
#include "optimization/line_search/line_search_helpers.cuh"
#include "common/math.cuh"

namespace curobo {
namespace optimization {
namespace vector_free {

constexpr unsigned MASK = 0xffffffff;

// Assign adjacent product pairs to progressively larger lane groups. After
// group g, lane l owns product l % (2*g) within each remaining product group.
// All 128 threads participate; each warp receives the same 32 block sums.
__device__ __forceinline__ float reduce_products(float (&products)[32], float *scratch) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int group = 1; group < 32; group *= 2) {
    const bool second = (lane & group) != 0;
#pragma unroll
    for (int pair = 0; pair < 16 / group; ++pair) {
      const float even = products[2 * pair];
      const float odd = products[2 * pair + 1];
      const float local = second ? odd : even;
      const float outgoing = second ? even : odd;
      const float incoming = __shfl_xor_sync(MASK, outgoing, group);
      products[pair] = local + incoming;
    }
  }
  scratch[threadIdx.x] = products[0];
  __syncthreads();
  const float result =
      (scratch[lane] + scratch[32 + lane]) + (scratch[64 + lane] + scratch[96 + lane]);
  // Protect these reads before the next reduction or step limiter reuses scratch.
  __syncthreads();
  return result;
}

// Each block owns one problem. Cache s^T y products by physical ring slot;
// logical history is indexed from newest to oldest.
// One or two variable components per thread keep reduction and lane ownership fixed.
template <int History, int Items, bool Geometric, bool SharedHistory>
__global__ void kernel_lbfgs_step_vector_free(
    float *direction, float *rho, float *history_y, float *history_s, const float *position,
    float *previous_position, float *previous_gradient, const float *gradient, const float *maximum,
    int batch_size, int dimension, int *heads, int *counts, bool scale_step, const bool *allowed,
    const int *status, const float *scales, const float *beta, float *candidates, float *cross,
    float epsilon, int candidate_count, bool stable_mode) {
  static_assert(History >= 1 && History <= 30, "History must be in [1, 30]");
  static_assert(Items == 1 || Items == 2, "One or two variable components per thread");
  constexpr int THREADS = 128;
  constexpr int INITIAL_TILES = (2 + 2 * History + 31) / 32;
  const int b = blockIdx.x, t = threadIdx.x, lane = t & 31;
  if (status[b] == curobo::optimization::line_search::STATUS_BACKTRACK) {
    curobo::optimization::write_lbfgs_candidates(position, direction, scales, beta, candidates,
                                                 dimension, candidate_count);
    return;
  }
  __shared__ float scratch[128];
  const int old_count = counts[b], head = heads[b];
  float sr[History][Items], yr[SharedHistory ? 1 : History][Items];
  __shared__ float shared_y[SharedHistory ? History : 1][Items][128];
  float g[Items], s[Items], y[Items];
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int variable = t + item * THREADS;
    const int index = b * dimension + variable;
    g[item] = variable < dimension ? gradient[index] : 0;
    s[item] = variable < dimension ? position[index] - previous_position[index] : 0;
    y[item] = variable < dimension ? g[item] - previous_gradient[index] : 0;
#pragma unroll
    for (int i = 0; i < History; ++i) {
      const int slot = (head - 1 - i + 2 * History) % History;
      const int offset = (slot * batch_size + b) * dimension + variable;
      sr[i][item] = variable < dimension && i < old_count ? history_s[offset] : 0;
      if constexpr (SharedHistory)
        shared_y[i][item][t] = variable < dimension && i < old_count ? history_y[offset] : 0;
      else
        yr[i][item] = variable < dimension && i < old_count ? history_y[offset] : 0;
    }
  }
  // Compute curvature and cross products for the candidate pair.
  float totals[INITIAL_TILES];
#pragma unroll
  for (int tile = 0; tile < INITIAL_TILES; ++tile) {
    float dots[32] = {};
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int k = tile * 32 + j;
#pragma unroll
      for (int item = 0; item < Items; ++item) {
        if (k == 0)
          dots[j] += s[item] * y[item];
        else if (k == 1)
          dots[j] += s[item] * g[item];
        else if (k < 2 + History)
          dots[j] += sr[k - 2][item] * g[item];
        else if (k < 2 + 2 * History)
          dots[j] += sr[k - 2 - History][item] * y[item];
      }
    }
    totals[tile] = reduce_products(dots, scratch);
  }
  const float ys = __shfl_sync(MASK, totals[0], 0);
  const float inverse_curvature = 1.0f / ys;
  const bool insert =
      allowed[b] && ys > 0 && isfinite(ys) && inverse_curvature > 0 && isfinite(inverse_curvature);
  const int count = insert ? min(old_count + 1, History) : old_count;
  const int newest = insert ? head : (head - 1 + History) % History;
  float sg = 0, new_cross = 0;
#pragma unroll
  for (int i = 0; i < History; ++i) {
    const int kg = 2 + i, kc = 2 + History + i;
    const float gv = __shfl_sync(MASK, totals[kg / 32], kg % 32);
    const float cv = __shfl_sync(MASK, totals[kc / 32], kc % 32);
    if (lane == i) {
      sg = gv;
      new_cross = cv;
    }
  }
  if (insert) {
    const float old_sg = __shfl_up_sync(MASK, sg, 1);
    const float new_sg = __shfl_sync(MASK, totals[0], 1);
    sg = lane == 0 ? new_sg : old_sg;
    if (t < old_count && t < History - 1) {
      const int slot = (head - 1 - t + 2 * History) % History;
      cross[(b * History + slot) * History + head] = new_cross;
    }
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const int variable = t + item * THREADS;
      if (variable < dimension) {
        history_s[(head * batch_size + b) * dimension + variable] = s[item];
        history_y[(head * batch_size + b) * dimension + variable] = y[item];
      }
#pragma unroll
      for (int j = History - 1; j > 0; --j) {
        sr[j][item] = sr[j - 1][item];
        if constexpr (SharedHistory)
          shared_y[j][item][t] = shared_y[j - 1][item][t];
        else
          yr[j][item] = yr[j - 1][item];
      }
      sr[0][item] = s[item];
      if constexpr (SharedHistory)
        shared_y[0][item][t] = y[item];
      else
        yr[0][item] = y[item];
    }
    if (t == 0) {
      rho[head * batch_size + b] = inverse_curvature;
      heads[b] = (head + 1) % History;
      counts[b] = count;
    }
  }
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int variable = t + item * THREADS;
    const int index = b * dimension + variable;
    if (variable < dimension) {
      previous_position[index] = position[index];
      previous_gradient[index] = g[item];
    }
  }
  __syncthreads();
  const int own_slot = (newest - min(lane, History - 1) + 2 * History) % History;
  const float rh = lane < count ? rho[own_slot * batch_size + b] : 0;
  sg = lane < count ? sg : 0;
  float row[SharedHistory ? 1 : History], column[SharedHistory ? 1 : History];
  __shared__ float shared_cross[SharedHistory ? 32 : 1][SharedHistory ? 33 : 1];
  if constexpr (SharedHistory) {
    // Padding avoids bank conflicts for both row and column reads.
    if (t < 32) {
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const int slot = (newest - j + 2 * History) % History;
        shared_cross[t][j] =
            t < count && j < t ? cross[(b * History + own_slot) * History + slot] : 0;
      }
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int j = 0; j < History; ++j) {
      const int slot = (newest - j + 2 * History) % History;
      row[j] = lane < count && j < lane ? cross[(b * History + own_slot) * History + slot] : 0;
      column[j] = lane < count && j > lane && j < count
                      ? cross[(b * History + slot) * History + own_slot]
                      : 0;
    }
  }
  float alpha[History];
#pragma unroll
  for (int i = 0; i < History; ++i) {
    alpha[i] = __shfl_sync(MASK, rh * sg, i);
    sg -= alpha[i] * (SharedHistory ? shared_cross[lane][i] : row[i]);
  }
  float q[Items];
  float dots[32] = {};
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    q[item] = g[item];
#pragma unroll
    for (int i = 0; i < History; ++i)
      q[item] -= alpha[i] * (SharedHistory ? shared_y[i][item][t] : yr[i][item]);
#pragma unroll
    for (int i = 0; i < History; ++i)
      dots[i] += (SharedHistory ? shared_y[i][item][t] : yr[i][item]) * q[item];
    dots[History] += Geometric ? sr[0][item] * sr[0][item]
                               : sr[0][item] * (SharedHistory ? shared_y[0][item][t] : yr[0][item]);
    const float newest_y = SharedHistory ? shared_y[0][item][t] : yr[0][item];
    dots[History + 1] += newest_y * newest_y;
  }
  const float total = reduce_products(dots, scratch);
  const float ss = __shfl_sync(MASK, total, History);
  const float yy = __shfl_sync(MASK, total, History + 1);
  float gamma = count > 0 ? (Geometric ? sqrtf(ss / yy) : ss / yy) : epsilon;
  if (stable_mode && (!(gamma > 0) || !isfinite(gamma))) gamma = epsilon;
  gamma = curobo::common::relu(gamma);
  float accumulation = 0, coefficient[History];
#pragma unroll
  for (int j = History - 1; j >= 0; --j) {
    coefficient[j] = alpha[j] - __shfl_sync(MASK, rh * (gamma * total + accumulation), j);
    accumulation += coefficient[j] * (SharedHistory ? shared_cross[j][lane] : column[j]);
  }
  float result[Items];
  float limit = 1, ratio = 0;
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    result[item] = gamma * q[item];
#pragma unroll
    for (int i = 0; i < History; ++i) result[item] += coefficient[i] * sr[i][item];
    const int variable = t + item * THREADS;
    if (scale_step && variable < dimension)
      ratio = fmaxf(ratio, fabsf(result[item]) / maximum[variable]);
  }
  if (scale_step) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
      ratio = fmaxf(ratio, __shfl_xor_sync(MASK, ratio, offset));
    if (lane == 0) scratch[t / 32] = ratio;
    __syncthreads();
#pragma unroll
    for (int w = 0; w < 4; ++w) limit = fmaxf(limit, scratch[w]);
  }
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    const int variable = t + item * THREADS;
    if (variable < dimension) {
      const int index = b * dimension + variable;
      const float step = -result[item] / limit;
      direction[index] = step;
#pragma unroll
      for (int k = 0; k < candidate_count; ++k)
        candidates[(b * candidate_count + k) * dimension + variable] =
            position[index] + beta[b] * scales[k] * step;
    }
  }
}

}  // namespace vector_free
}  // namespace optimization
}  // namespace curobo
