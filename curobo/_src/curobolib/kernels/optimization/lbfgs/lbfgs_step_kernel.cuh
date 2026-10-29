/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



#include "third_party/helper_math.h"
#include "common/block_warp_reductions.cuh"
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "optimization/line_search/line_search_helpers.cuh"

namespace curobo{
namespace optimization{

#ifndef CUROBO_LBFGS_SCRATCH_HISTORY
#define CUROBO_LBFGS_SCRATCH_HISTORY 128
#endif

template<typename ScalarType>
__forceinline__ __device__ ScalarType lbfgs_warp_reduce_sum(ScalarType value)
{
  #pragma unroll
  for (int offset = curobo::common::warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(curobo::common::fullMask, value, offset);
  }
  return value;
}


template<typename ScalarType>
__forceinline__ __device__ ScalarType lbfgs_warp_allreduce_sum(ScalarType value)
{
  #pragma unroll
  for (int offset = curobo::common::warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(curobo::common::fullMask, value, offset);
  }
  return value;
}


template<typename ScalarType>
__forceinline__ __device__ ScalarType lbfgs_warp_reduce_max(ScalarType value)
{
  #pragma unroll
  for (int offset = curobo::common::warpSize / 2; offset > 0; offset /= 2) {
    const ScalarType other = __shfl_down_sync(curobo::common::fullMask, value, offset);
    value = value > other ? value : other;
  }
  return value;
}


template<typename ScalarType>
__forceinline__ __device__ ScalarType lbfgs_warp_allreduce_max(ScalarType value)
{
  #pragma unroll
  for (int offset = curobo::common::warpSize / 2; offset > 0; offset /= 2) {
    const ScalarType other = __shfl_xor_sync(curobo::common::fullMask, value, offset);
    value = value > other ? value : other;
  }
  return value;
}


template<typename ScalarType>
__forceinline__ __device__ ScalarType lbfgs_reduce_sum_slot(
  const ScalarType value,
  const int slot,
  ScalarType *partial_sums)
{
  const int lane = threadIdx.x % curobo::common::warpSize;
  const int warp = threadIdx.x / curobo::common::warpSize;
  const int warp_count = blockDim.x / curobo::common::warpSize;
  if (blockDim.x == curobo::common::warpSize) {
    return lbfgs_warp_allreduce_sum(value);
  }

  ScalarType sum = lbfgs_warp_reduce_sum(value);
  if (lane == 0) {
    partial_sums[slot * warp_count + warp] = sum;
  }
  __syncthreads();

  sum = lane < warp_count ?
      partial_sums[slot * warp_count + lane] : ScalarType(0);
  return lbfgs_warp_allreduce_sum(sum);
}


template<typename ScalarType>
__forceinline__ __device__ ScalarType lbfgs_reduce_max_slot(
  const ScalarType value,
  const int slot,
  ScalarType *partial_maxima)
{
  const int lane = threadIdx.x % curobo::common::warpSize;
  const int warp = threadIdx.x / curobo::common::warpSize;
  const int warp_count = blockDim.x / curobo::common::warpSize;
  if (blockDim.x == curobo::common::warpSize) {
    return lbfgs_warp_allreduce_max(value);
  }

  ScalarType maximum = lbfgs_warp_reduce_max(value);
  if (lane == 0) {
    partial_maxima[slot * warp_count + warp] = maximum;
  }
  __syncthreads();

  maximum = lane < warp_count ?
      partial_maxima[slot * warp_count + lane] : ScalarType(0);
  return lbfgs_warp_allreduce_max(maximum);
}


/**
 * @brief L-BFGS update and two-loop recursion using circular global history buffers.
 *
 * One block owns each optimization problem. Threads traverse the variable dimension
 * with a grid-stride loop, so the supported dimension is independent of the CUDA
 * maximum threads per block. History is always stored circularly in global memory
 * and can optionally be staged in dynamic shared memory for the two-loop recursion.
 */
template<
  typename ScalarType,
  bool rolled_ys,
  int FIXED_M = -1,
  bool USE_SHARED_HISTORY = false,
  int ITEMS_PER_THREAD = 0,
  bool USE_GEOMETRIC_SCALING = true>
__global__ __launch_bounds__(
    (ITEMS_PER_THREAD >= 16 || ITEMS_PER_THREAD == 0) ? 512 : 1024)
void kernel_lbfgs_step_circular(
  ScalarType *step_vec,
  ScalarType *rho_buffer,
  ScalarType *y_buffer,
  ScalarType *s_buffer,
  ScalarType *q,
  ScalarType *x_0,
  ScalarType *grad_0,
  const ScalarType *grad_q,
  const ScalarType *optimization_step_max,
  const float epsilon,
  const int batchsize,
  const int history_m,
  const int v_dim,
  const bool stable_mode,
  int32_t *history_head,
  int32_t *history_count,
  const bool scale_step,
  const bool *curvature_update_allowed,
  const int32_t *line_search_status)
{
  // Block-uniform exit preserves the shortened direction and every history
  // and reference buffer while the next captured iteration retries.
  if (line_search_status[blockIdx.x] == line_search::STATUS_BACKTRACK) return;
  constexpr bool is_compile_time = (FIXED_M > 0);
  const int effective_m = is_compile_time ? FIXED_M : history_m;

  __shared__ ScalarType alpha_buffer[CUROBO_LBFGS_SCRATCH_HISTORY];
  __shared__ ScalarType rho_buffer_shared[CUROBO_LBFGS_SCRATCH_HISTORY];
  __shared__ bool curvature_pair_valid;
  extern __shared__ ScalarType shared_workspace[];
  const int warp_count = blockDim.x / curobo::common::warpSize;
  const int reduction_elements = blockDim.x == curobo::common::warpSize ?
      0 : (2 * effective_m + 3) * warp_count;
  ScalarType* reduction_slots = shared_workspace;
  ScalarType* s_buffer_shared = &shared_workspace[reduction_elements];
  ScalarType* y_buffer_shared = &s_buffer_shared[effective_m * v_dim];

  const int batch = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int head = history_head[batch];
  const int previous_count = history_count[batch];
  const int batch_offset = batch * v_dim;
  const int history_batch_offset =
      head * batchsize * v_dim + batch_offset;
  ScalarType register_work[ITEMS_PER_THREAD > 0 ? ITEMS_PER_THREAD : 1];

  ScalarType local_ys = ScalarType(0);
  if constexpr (ITEMS_PER_THREAD > 0) {
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int variable = thread_idx + item * blockDim.x;
      if (variable < v_dim) {
        const int state_index = batch_offset + variable;
        const ScalarType gradient = grad_q[state_index];
        const ScalarType position = q[state_index];
        const ScalarType y = gradient - grad_0[state_index];
        const ScalarType s = position - x_0[state_index];

        local_ys += y * s;
      }
    }
  } else {
    for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
      const int state_index = batch_offset + variable;
      const ScalarType gradient = grad_q[state_index];
      const ScalarType position = q[state_index];
      const ScalarType y = gradient - grad_0[state_index];
      const ScalarType s = position - x_0[state_index];

      local_ys += y * s;
    }
  }

  const ScalarType numerator =
      lbfgs_reduce_sum_slot(local_ys, 0, &reduction_slots[0]);

  if (thread_idx == 0) {
    ScalarType rho = ScalarType(1) / numerator;
    curvature_pair_valid = curvature_update_allowed[batch] &&
        numerator > ScalarType(0) && !isnan(numerator) && !isinf(numerator) &&
        rho > ScalarType(0) && !isnan(rho) && !isinf(rho);
    if (curvature_pair_valid) {
      rho_buffer[head * batchsize + batch] = rho;
    }
  }
  __syncthreads();

  // Commit the candidate only after its block-wide curvature is known. This
  // second pass avoids overwriting the oldest pair when a full ring rejects a
  // candidate. The reference point advances even when the pair is rejected.
  if constexpr (ITEMS_PER_THREAD > 0) {
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int variable = thread_idx + item * blockDim.x;
      if (variable < v_dim) {
        const int state_index = batch_offset + variable;
        const ScalarType gradient = grad_q[state_index];
        const ScalarType position = q[state_index];
        const ScalarType y = gradient - grad_0[state_index];
        const ScalarType s = position - x_0[state_index];

        if (curvature_pair_valid) {
          y_buffer[history_batch_offset + variable] = y;
          s_buffer[history_batch_offset + variable] = s;
        }
        grad_0[state_index] = gradient;
        x_0[state_index] = position;
        register_work[item] = gradient;
      }
    }
  } else {
    for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
      const int state_index = batch_offset + variable;
      const ScalarType gradient = grad_q[state_index];
      const ScalarType position = q[state_index];
      const ScalarType y = gradient - grad_0[state_index];
      const ScalarType s = position - x_0[state_index];

      if (curvature_pair_valid) {
        y_buffer[history_batch_offset + variable] = y;
        s_buffer[history_batch_offset + variable] = s;
      }
      grad_0[state_index] = gradient;
      x_0[state_index] = position;
      step_vec[state_index] = gradient;
    }
  }
  __syncthreads();

  const int active_count = curvature_pair_valid ?
      min(previous_count + 1, effective_m) : previous_count;
  int newest_history = curvature_pair_valid ? head : head - 1;
  if (newest_history < 0) {
    newest_history += effective_m;
  }
  const int first_active_logical = effective_m - active_count;
  int oldest_history = newest_history - active_count + 1;
  if (oldest_history < 0) {
    oldest_history += effective_m;
  }

  if constexpr (USE_SHARED_HISTORY) {
    for (
      int active_offset = thread_idx;
      active_offset < active_count;
      active_offset += blockDim.x)
    {
      const int logical_history = first_active_logical + active_offset;
      int physical_history = oldest_history + active_offset;
      if (physical_history >= effective_m) {
        physical_history -= effective_m;
      }
      rho_buffer_shared[logical_history] =
          rho_buffer[physical_history * batchsize + batch];
    }

    if constexpr (ITEMS_PER_THREAD > 0) {
      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int variable = thread_idx + item * blockDim.x;
        if (variable < v_dim) {
          int physical_history = oldest_history;
          #pragma unroll 8
          for (int active_offset = 0;
               active_offset < active_count;
               ++active_offset) {
            const int logical_history = first_active_logical + active_offset;
            const int global_index =
                physical_history * batchsize * v_dim +
                batch_offset + variable;
            const int shared_index = logical_history * v_dim + variable;
            s_buffer_shared[shared_index] = s_buffer[global_index];
            y_buffer_shared[shared_index] = y_buffer[global_index];
            ++physical_history;
            if (physical_history == effective_m) {
              physical_history = 0;
            }
          }
        }
      }
    } else {
      const int history_elements = active_count * v_dim;
      for (int element = thread_idx; element < history_elements; element += blockDim.x) {
        const int active_offset = element / v_dim;
        const int variable = element - active_offset * v_dim;
        const int logical_history = first_active_logical + active_offset;
        int physical_history = oldest_history + active_offset;
        if (physical_history >= effective_m) {
          physical_history -= effective_m;
        }
        const int global_index =
            physical_history * batchsize * v_dim +
            batch_offset + variable;
        const int shared_index = logical_history * v_dim + variable;
        s_buffer_shared[shared_index] = s_buffer[global_index];
        y_buffer_shared[shared_index] = y_buffer[global_index];
      }
    }
    __syncthreads();
  }

  // Each block owns one problem, so the active-history loop bound is uniform
  // across the block even when other problems reset independently.
  #pragma unroll 8
  for (int history_offset = 0; history_offset < active_count; ++history_offset) {
    const int logical_history = effective_m - 1 - history_offset;
    int physical_history = newest_history - history_offset;
    if (physical_history < 0) {
      physical_history += effective_m;
    }
    const int history_offset_base =
        physical_history * batchsize * v_dim + batch_offset;

    ScalarType local_dot = ScalarType(0);
    if constexpr (ITEMS_PER_THREAD > 0) {
      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int variable = thread_idx + item * blockDim.x;
        if (variable < v_dim) {
          ScalarType s;
          if constexpr (USE_SHARED_HISTORY) {
            s = s_buffer_shared[logical_history * v_dim + variable];
          } else {
            s = s_buffer[history_offset_base + variable];
          }
          local_dot += register_work[item] * s;
        }
      }
    } else {
      for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
        ScalarType s;
        if constexpr (USE_SHARED_HISTORY) {
          s = s_buffer_shared[logical_history * v_dim + variable];
        } else {
          s = s_buffer[history_offset_base + variable];
        }
        local_dot += step_vec[batch_offset + variable] * s;
      }
    }
    const ScalarType history_dot = lbfgs_reduce_sum_slot(
        local_dot, 1 + history_offset, &reduction_slots[0]);

    ScalarType rho;
    if constexpr (USE_SHARED_HISTORY) {
      rho = rho_buffer_shared[logical_history];
    } else {
      rho = rho_buffer[physical_history * batchsize + batch];
    }
    const ScalarType alpha = history_dot * rho;
    if (thread_idx == 0) {
      alpha_buffer[logical_history] = alpha;
    }
    if constexpr (ITEMS_PER_THREAD > 0) {
      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int variable = thread_idx + item * blockDim.x;
        if (variable < v_dim) {
          ScalarType y;
          if constexpr (USE_SHARED_HISTORY) {
            y = y_buffer_shared[logical_history * v_dim + variable];
          } else {
            y = y_buffer[history_offset_base + variable];
          }
          register_work[item] -= alpha * y;
        }
      }
    } else {
      for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
        const int state_index = batch_offset + variable;
        ScalarType y;
        if constexpr (USE_SHARED_HISTORY) {
          y = y_buffer_shared[logical_history * v_dim + variable];
        } else {
          y = y_buffer[history_offset_base + variable];
        }
        step_vec[state_index] -= alpha * y;
      }
    }
  }

  ScalarType ss = ScalarType(0);
  ScalarType yy = ScalarType(0);
  if (active_count > 0) {
    ScalarType local_ss = ScalarType(0);
    ScalarType local_yy = ScalarType(0);
    if constexpr (ITEMS_PER_THREAD > 0) {
      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int variable = thread_idx + item * blockDim.x;
        if (variable < v_dim) {
          ScalarType s;
          ScalarType y;
          if constexpr (USE_SHARED_HISTORY) {
            s = s_buffer_shared[(effective_m - 1) * v_dim + variable];
            y = y_buffer_shared[(effective_m - 1) * v_dim + variable];
          } else {
            const int newest_offset =
                newest_history * batchsize * v_dim + batch_offset;
            s = s_buffer[newest_offset + variable];
            y = y_buffer[newest_offset + variable];
          }
          if constexpr (USE_GEOMETRIC_SCALING) {
            local_ss += s * s;
          }
          local_yy += y * y;
        }
      }
    } else {
      for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
        ScalarType s;
        ScalarType y;
        if constexpr (USE_SHARED_HISTORY) {
          s = s_buffer_shared[(effective_m - 1) * v_dim + variable];
          y = y_buffer_shared[(effective_m - 1) * v_dim + variable];
        } else {
          const int newest_offset =
              newest_history * batchsize * v_dim + batch_offset;
          s = s_buffer[newest_offset + variable];
          y = y_buffer[newest_offset + variable];
        }
        if constexpr (USE_GEOMETRIC_SCALING) {
          local_ss += s * s;
        }
        local_yy += y * y;
      }
    }
    if constexpr (USE_GEOMETRIC_SCALING) {
      ss = lbfgs_reduce_sum_slot(
          local_ss, 1 + effective_m, &reduction_slots[0]);
    }
    yy = lbfgs_reduce_sum_slot(
        local_yy, 2 + effective_m, &reduction_slots[0]);
  }

  ScalarType gamma = epsilon;
  if (active_count > 0) {
    if constexpr (USE_GEOMETRIC_SCALING) {
      gamma = sqrt(ss / yy);
    } else {
      ScalarType newest_rho;
      if constexpr (USE_SHARED_HISTORY) {
        newest_rho = rho_buffer_shared[effective_m - 1];
      } else {
        newest_rho = rho_buffer[newest_history * batchsize + batch];
      }
      gamma = ScalarType(1) / (newest_rho * yy);
    }
  }
  if (stable_mode &&
      (gamma <= ScalarType(0) || isinf(gamma) || isnan(gamma))) {
    gamma = epsilon;
  }
  gamma = curobo::common::relu(gamma);
  if constexpr (ITEMS_PER_THREAD > 0) {
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int variable = thread_idx + item * blockDim.x;
      if (variable < v_dim) {
        register_work[item] *= gamma;
      }
    }
  } else {
    for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
      step_vec[batch_offset + variable] *= gamma;
    }
  }
  // Second loop: oldest pair to newest pair.
  int physical_history = oldest_history;
  #pragma unroll 8
  for (int history_offset = 0; history_offset < active_count; ++history_offset) {
    const int logical_history = first_active_logical + history_offset;
    const int history_offset_base =
        physical_history * batchsize * v_dim + batch_offset;

    ScalarType local_dot = ScalarType(0);
    if constexpr (ITEMS_PER_THREAD > 0) {
      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int variable = thread_idx + item * blockDim.x;
        if (variable < v_dim) {
          ScalarType y;
          if constexpr (USE_SHARED_HISTORY) {
            y = y_buffer_shared[logical_history * v_dim + variable];
          } else {
            y = y_buffer[history_offset_base + variable];
          }
          local_dot += register_work[item] * y;
        }
      }
    } else {
      for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
        ScalarType y;
        if constexpr (USE_SHARED_HISTORY) {
          y = y_buffer_shared[logical_history * v_dim + variable];
        } else {
          y = y_buffer[history_offset_base + variable];
        }
        local_dot += step_vec[batch_offset + variable] * y;
      }
    }
    const ScalarType history_dot = lbfgs_reduce_sum_slot(
        local_dot, 3 + effective_m + history_offset, &reduction_slots[0]);

    ScalarType rho;
    if constexpr (USE_SHARED_HISTORY) {
      rho = rho_buffer_shared[logical_history];
    } else {
      rho = rho_buffer[physical_history * batchsize + batch];
    }
    const ScalarType beta = history_dot * rho;
    const ScalarType coefficient = alpha_buffer[logical_history] - beta;
    if constexpr (ITEMS_PER_THREAD > 0) {
      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int variable = thread_idx + item * blockDim.x;
        if (variable < v_dim) {
          ScalarType s;
          if constexpr (USE_SHARED_HISTORY) {
            s = s_buffer_shared[logical_history * v_dim + variable];
          } else {
            s = s_buffer[history_offset_base + variable];
          }
          register_work[item] += coefficient * s;
        }
      }
    } else {
      for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
        const int state_index = batch_offset + variable;
        ScalarType s;
        if constexpr (USE_SHARED_HISTORY) {
          s = s_buffer_shared[logical_history * v_dim + variable];
        } else {
          s = s_buffer[history_offset_base + variable];
        }
        step_vec[state_index] += coefficient * s;
      }
    }

    ++physical_history;
    if (physical_history == effective_m) {
      physical_history = 0;
    }
  }

  ScalarType direction_scale = ScalarType(1);
  if (scale_step) {
    ScalarType local_scale = ScalarType(0);
    if constexpr (ITEMS_PER_THREAD > 0) {
      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int variable = thread_idx + item * blockDim.x;
        if (variable < v_dim) {
          const ScalarType ratio =
              fabs(register_work[item]) / optimization_step_max[variable];
          local_scale = local_scale > ratio ? local_scale : ratio;
        }
      }
    } else {
      for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
        const ScalarType ratio =
            fabs(step_vec[batch_offset + variable]) / optimization_step_max[variable];
        local_scale = local_scale > ratio ? local_scale : ratio;
      }
    }
    direction_scale = lbfgs_reduce_max_slot(local_scale, 0, &reduction_slots[0]);
    direction_scale = direction_scale > ScalarType(1) ? direction_scale : ScalarType(1);
  }

  if constexpr (ITEMS_PER_THREAD > 0) {
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int variable = thread_idx + item * blockDim.x;
      if (variable < v_dim) {
        step_vec[batch_offset + variable] = -register_work[item] / direction_scale;
      }
    }
  } else {
    for (int variable = thread_idx; variable < v_dim; variable += blockDim.x) {
      step_vec[batch_offset + variable] =
          -step_vec[batch_offset + variable] / direction_scale;
    }
  }

  if (thread_idx == 0) {
    if (curvature_pair_valid) {
      int next_head = head + 1;
      if (next_head == effective_m) {
        next_head = 0;
      }
      history_head[batch] = next_head;
      history_count[batch] = active_count;
    }
  }
}
}
}
