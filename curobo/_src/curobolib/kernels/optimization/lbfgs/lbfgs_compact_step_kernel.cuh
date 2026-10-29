// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mma.h>
#include "optimization/line_search/line_search_helpers.cuh"

namespace curobo {
namespace optimization {

constexpr int OP_THREADS = 256;
constexpr int OP_WARPS = OP_THREADS / 32;

__forceinline__ __device__ float op_warp_sum(float value)
{
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__forceinline__ __device__ float op_warp_max(float value)
{
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, offset);
    value = value > other ? value : other;
  }
  return value;
}

template<int ValueCount>
__forceinline__ __device__ void op_block_reduce(
  float (&values)[ValueCount],
  float* scratch)
{
  static_assert(ValueCount > 0, "ValueCount must be positive");
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  #pragma unroll
  for (int value_index = 0; value_index < ValueCount; ++value_index) {
    values[value_index] = op_warp_sum(values[value_index]);
  }
  if (lane == 0) {
    #pragma unroll
    for (int value_index = 0; value_index < ValueCount; ++value_index) {
      scratch[value_index * OP_WARPS + warp] = values[value_index];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int value_index = 0; value_index < ValueCount; ++value_index) {
    auto value = lane < OP_WARPS ? scratch[value_index * OP_WARPS + lane] : 0.0f;
    value = op_warp_sum(value);
    values[value_index] = __shfl_sync(0xffffffff, value, 0);
  }
}

template<int History>
__forceinline__ __device__ void op_tensor_matrix_product(
  const float* left,
  const float* right,
  float* output)
{
  using namespace nvcuda;
  constexpr int PADDED_HISTORY = ((History + 15) / 16) * 16;
  constexpr int HISTORY_TILES = PADDED_HISTORY / 16;
  const int warp = threadIdx.x / 32;
  for (
    int tile = warp;
    tile < HISTORY_TILES * HISTORY_TILES;
    tile += OP_WARPS)
  {
    const int tile_row = tile / HISTORY_TILES;
    const int tile_column = tile % HISTORY_TILES;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    #pragma unroll
    for (int k = 0; k < PADDED_HISTORY; k += 8) {
      wmma::fragment<
        wmma::matrix_b,
        16,
        16,
        8,
        wmma::precision::tf32,
        wmma::row_major> right_fragment;
      wmma::load_matrix_sync(
          right_fragment,
          &right[k * PADDED_HISTORY + tile_column * 16],
          PADDED_HISTORY);
      auto right_residual_fragment = right_fragment;
      wmma::fragment<
        wmma::matrix_a,
        16,
        16,
        8,
        wmma::precision::tf32,
        wmma::row_major> left_fragment;
      wmma::load_matrix_sync(
          left_fragment,
          &left[tile_row * 16 * PADDED_HISTORY + k],
          PADDED_HISTORY);
      // Three TF32 products retain the high-high and both first-order residual
      // terms. This is substantially closer to FP32 than one TF32 product
      // without giving up tensor-core throughput.
      auto left_residual_fragment = left_fragment;
      #pragma unroll
      for (int element = 0; element < left_fragment.num_elements; ++element) {
        const float left_value = left_fragment.x[element];
        const float left_tf32 = wmma::__float_to_tf32(left_value);
        left_fragment.x[element] = left_tf32;
        left_residual_fragment.x[element] =
            wmma::__float_to_tf32(left_value - left_tf32);
      }
      #pragma unroll
      for (int element = 0; element < right_fragment.num_elements; ++element) {
        const float right_value = right_fragment.x[element];
        const float right_tf32 = wmma::__float_to_tf32(right_value);
        right_fragment.x[element] = right_tf32;
        right_residual_fragment.x[element] =
            wmma::__float_to_tf32(right_value - right_tf32);
      }
      wmma::mma_sync(accumulator, left_fragment, right_fragment, accumulator);
      wmma::mma_sync(
          accumulator,
          left_fragment,
          right_residual_fragment,
          accumulator);
      wmma::mma_sync(
          accumulator,
          left_residual_fragment,
          right_fragment,
          accumulator);
    }
    wmma::store_matrix_sync(
        &output[tile_row * 16 * PADDED_HISTORY + tile_column * 16],
        accumulator,
        PADDED_HISTORY,
        wmma::mem_row_major);
  }
  __syncthreads();
}

template<
  int History,
  int Dimension,
  int SharedMatrixCount,
  bool ScaleStep,
  bool UseGeometricScaling>
__global__ __launch_bounds__(OP_THREADS)
void kernel_lbfgs_step_compact(
  float* step,
  float* rho_buffer,
  float* y_buffer,
  float* s_buffer,
  float* inverse_r_buffer,
  float* matrix_workspace_global,
  const float* position,
  float* reference_position,
  float* reference_gradient,
  const float* gradient,
  const float* optimization_step_max,
  const float epsilon,
  const int batch_size,
  int32_t* history_head,
  int32_t* history_count,
  const bool* curvature_update_allowed,
  const int32_t* line_search_status)
{
  if (line_search_status[blockIdx.x] == line_search::STATUS_BACKTRACK) return;
  using namespace nvcuda;
  static_assert(History > 0, "History must be positive");
  static_assert(
      SharedMatrixCount == 0 || SharedMatrixCount == 1 ||
      SharedMatrixCount == 2 || SharedMatrixCount == 3,
      "SharedMatrixCount must be 0, 1, 2, or 3");
  static_assert(Dimension > 0, "Dimension must be positive");
  constexpr int OP_HISTORY = History;
  constexpr int OP_PADDED_HISTORY = ((History + 15) / 16) * 16;
  constexpr int OP_DIMENSION = Dimension;
  constexpr int OP_PADDED_DIMENSION = ((Dimension + 7) / 8) * 8;
  constexpr int OP_HISTORY_TILES = OP_PADDED_HISTORY / 16;
  constexpr int OP_MATRIX_ELEMENTS = OP_PADDED_HISTORY * OP_PADDED_HISTORY;
  constexpr int GLOBAL_MATRIX_COUNT = 3 - SharedMatrixCount;
  constexpr int OP_GLOBAL_WORKSPACE_ELEMENTS =
      GLOBAL_MATRIX_COUNT * OP_MATRIX_ELEMENTS;

  extern __shared__ float workspace[];
  float* global_workspace =
      matrix_workspace_global + blockIdx.x * OP_GLOBAL_WORKSPACE_ELEMENTS;
  float* panel_s = workspace;
  float* panel_y = panel_s + OP_PADDED_HISTORY * OP_PADDED_DIMENSION;
  float* shared_matrix_workspace =
      panel_y + OP_PADDED_HISTORY * OP_PADDED_DIMENSION;
  float* global_matrix_workspace = global_workspace;
  float* inverse_r;
  float* middle_matrix;
  float* transformed_inverse_r;
  float* vector_workspace;
  if constexpr (SharedMatrixCount == 3) {
    inverse_r = shared_matrix_workspace;
    middle_matrix = inverse_r + OP_MATRIX_ELEMENTS;
    transformed_inverse_r = middle_matrix + OP_MATRIX_ELEMENTS;
    vector_workspace = transformed_inverse_r + OP_MATRIX_ELEMENTS;
  } else if constexpr (SharedMatrixCount == 1) {
    // inverse(R) has the most reuse: history update, the matrix product, and
    // both coefficient vectors. Keep it shared when only one matrix fits.
    inverse_r = shared_matrix_workspace;
    middle_matrix = global_matrix_workspace;
    transformed_inverse_r = middle_matrix + OP_MATRIX_ELEMENTS;
    vector_workspace = inverse_r + OP_MATRIX_ELEMENTS;
  } else if constexpr (SharedMatrixCount == 2) {
    inverse_r = shared_matrix_workspace;
    middle_matrix = global_matrix_workspace;
    transformed_inverse_r = inverse_r + OP_MATRIX_ELEMENTS;
    vector_workspace = transformed_inverse_r + OP_MATRIX_ELEMENTS;
  } else {
    inverse_r = global_matrix_workspace;
    middle_matrix = inverse_r + OP_MATRIX_ELEMENTS;
    transformed_inverse_r = middle_matrix + OP_MATRIX_ELEMENTS;
    vector_workspace = shared_matrix_workspace;
  }
  float* s_dot_g = vector_workspace;
  float* y_dot_g = s_dot_g + OP_PADDED_HISTORY;
  float* s_dot_y_new = y_dot_g + OP_PADDED_HISTORY;
  float* y_coefficient = s_dot_y_new + OP_PADDED_HISTORY;
  float* s_coefficient = y_coefficient + OP_PADDED_HISTORY;
  float* reduction_scratch = s_coefficient + OP_PADDED_HISTORY;

  __shared__ int pair_valid;
  __shared__ int active_pair_count;
  __shared__ int newest_physical;
  __shared__ int oldest_physical;
  __shared__ int use_two_loop_fallback;
  __shared__ float gamma;
  __shared__ int direction_scale_bits;

  const int batch = blockIdx.x;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int vector_offset = batch * OP_DIMENSION;
  const int head = history_head[batch];

  float pair_moments[3] = {0.0f, 0.0f, 0.0f};
  for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
    const int state_index = vector_offset + variable;
    const float candidate_s = position[state_index] - reference_position[state_index];
    const float candidate_y = gradient[state_index] - reference_gradient[state_index];
    pair_moments[0] += candidate_s * candidate_y;
    pair_moments[1] += candidate_s * candidate_s;
    pair_moments[2] += candidate_y * candidate_y;
  }
  op_block_reduce(pair_moments, reduction_scratch);
  if (thread == 0) {
    const float local_ys = pair_moments[0];
    const float rho = 1.0f / local_ys;
    constexpr int LINE_SEARCH_STATUS_FALLBACK = 4;
    use_two_loop_fallback =
        line_search_status[batch] == LINE_SEARCH_STATUS_FALLBACK;
    const int previous_count = history_count[batch];
    constexpr float MAX_CURVATURE_SCALE_DECREASE = 1.0e4f;
    float minimum_stored_curvature = 3.402823466e+38F;
    int current_physical = head - previous_count;
    if (current_physical < 0) {
      current_physical += OP_HISTORY;
    }
    for (int history = 0; history < previous_count; ++history) {
      const float stored_curvature =
          1.0f / rho_buffer[current_physical * batch_size + batch];
      minimum_stored_curvature =
          fminf(minimum_stored_curvature, stored_curvature);
      current_physical =
          current_physical + 1 == OP_HISTORY ? 0 : current_physical + 1;
    }
    // On a failed line search, a pair many orders of magnitude smaller than
    // every retained pair makes even the FP32 recurrence vulnerable to
    // roundoff. Ignore only that combination; otherwise retain the circular
    // path's pair-eligibility rules.
    const bool candidate_scale_valid = previous_count == 0 ||
        !use_two_loop_fallback ||
        local_ys * MAX_CURVATURE_SCALE_DECREASE >= minimum_stored_curvature;
    pair_valid = curvature_update_allowed[batch] &&
        local_ys > 0.0f && isfinite(local_ys) &&
        candidate_scale_valid &&
        rho > 0.0f && isfinite(rho);
    active_pair_count = pair_valid ? min(previous_count + 1, OP_HISTORY) : previous_count;
    newest_physical = pair_valid ? head : head - 1;
    if (newest_physical < 0) {
      newest_physical += OP_HISTORY;
    }
    oldest_physical = pair_valid ? head + 1 : head;
    if (oldest_physical == OP_HISTORY) {
      oldest_physical = 0;
    }
    if (pair_valid) {
      rho_buffer[head * batch_size + batch] = rho;
      if constexpr (UseGeometricScaling) {
        gamma = sqrtf(pair_moments[1] / pair_moments[2]);
      } else {
        gamma = pair_moments[0] / pair_moments[2];
      }
    }
  }
  __syncthreads();

  for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
    const int state_index = vector_offset + variable;
    if (pair_valid) {
      const int history_index = head * batch_size * OP_DIMENSION + state_index;
      s_buffer[history_index] = position[state_index] - reference_position[state_index];
      y_buffer[history_index] = gradient[state_index] - reference_gradient[state_index];
    }
    reference_position[state_index] = position[state_index];
    reference_gradient[state_index] = gradient[state_index];
  }
  __syncthreads();
  if (!pair_valid) {
    if constexpr (UseGeometricScaling) {
      float newest_pair_norms[2] = {0.0f, 0.0f};
      for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
        const int history_index =
            newest_physical * batch_size * OP_DIMENSION + vector_offset + variable;
        const float newest_s = s_buffer[history_index];
        const float newest_y = y_buffer[history_index];
        newest_pair_norms[0] += newest_s * newest_s;
        newest_pair_norms[1] += newest_y * newest_y;
      }
      op_block_reduce(newest_pair_norms, reduction_scratch);
      if (thread == 0) {
        gamma = sqrtf(newest_pair_norms[0] / newest_pair_norms[1]);
      }
    } else {
      float newest_pair_moments[2] = {0.0f, 0.0f};
      for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
        const int history_index =
            newest_physical * batch_size * OP_DIMENSION + vector_offset + variable;
        const float newest_s = s_buffer[history_index];
        const float newest_y = y_buffer[history_index];
        newest_pair_moments[0] += newest_s * newest_y;
        newest_pair_moments[1] += newest_y * newest_y;
      }
      op_block_reduce(newest_pair_moments, reduction_scratch);
      if (thread == 0) {
        gamma = newest_pair_moments[0] / newest_pair_moments[1];
      }
    }
  }
  __syncthreads();
  if (thread == 0 && (!(gamma > 0.0f) || !isfinite(gamma))) {
    gamma = epsilon;
  }
  if (thread == 0 && !use_two_loop_fallback && active_pair_count > 1) {
    constexpr float MAX_CURVATURE_DIAGONAL_DYNAMIC_RANGE = 1.0e10f;
    float maximum_curvature = 0.0f;
    float maximum_inverse_curvature = 0.0f;
    const int first_active = OP_HISTORY - active_pair_count;
    for (int logical_history = first_active;
         logical_history < OP_HISTORY;
         ++logical_history) {
      int physical = oldest_physical + logical_history;
      if (physical >= OP_HISTORY) {
        physical -= OP_HISTORY;
      }
      const float inverse_curvature =
          rho_buffer[physical * batch_size + batch];
      const float curvature = 1.0f / inverse_curvature;
      maximum_curvature = fmaxf(maximum_curvature, curvature);
      maximum_inverse_curvature =
          fmaxf(maximum_inverse_curvature, inverse_curvature);
    }
    // This ratio is invariant to a common rescaling of the objective. Large
    // values make the explicitly updated inverse sensitive to FP32 roundoff,
    // while the two-loop recurrence remains usable.
    const float diagonal_dynamic_range =
        maximum_curvature * maximum_inverse_curvature;
    use_two_loop_fallback =
        !isfinite(diagonal_dynamic_range) ||
        diagonal_dynamic_range > MAX_CURVATURE_DIAGONAL_DYNAMIC_RANGE;
  }
  for (
    int element = thread;
    element < OP_PADDED_HISTORY * OP_PADDED_DIMENSION;
    element += blockDim.x)
  {
    const int logical_history = element / OP_PADDED_DIMENSION;
    const int variable = element - logical_history * OP_PADDED_DIMENSION;
    float value_s = 0.0f;
    float value_y = 0.0f;
    if (logical_history < OP_HISTORY && variable < OP_DIMENSION) {
      int physical = oldest_physical + logical_history;
      if (physical >= OP_HISTORY) {
        physical -= OP_HISTORY;
      }
      const int history_index =
          physical * batch_size * OP_DIMENSION + vector_offset + variable;
      value_s = s_buffer[history_index];
      value_y = y_buffer[history_index];
    }
    panel_s[element] = value_s;
    panel_y[element] = value_y;
  }
  __syncthreads();

  for (
    int element = thread;
    element < OP_PADDED_HISTORY * OP_PADDED_HISTORY;
    element += blockDim.x)
  {
    const int row = element / OP_PADDED_HISTORY;
    const int column = element - row * OP_PADDED_HISTORY;
    float value = 0.0f;
    if (row < OP_HISTORY && column < OP_HISTORY && column >= row) {
      int physical_row = oldest_physical + row;
      int physical_column = oldest_physical + column;
      if (physical_row >= OP_HISTORY) {
        physical_row -= OP_HISTORY;
      }
      if (physical_column >= OP_HISTORY) {
        physical_column -= OP_HISTORY;
      }
      value = inverse_r_buffer[
          batch * OP_HISTORY * OP_HISTORY + physical_row * OP_HISTORY + physical_column];
    }
    inverse_r[element] = value;
  }
  __syncthreads();

  // These matrix-vector projections are cheap at compact-supported shapes
  // and directly feed the inverse-Hessian coefficients, so keep them in FP32.
  if (thread < OP_HISTORY) {
    float accurate_s_dot_g = 0.0f;
    float accurate_y_dot_g = 0.0f;
    float accurate_s_dot_y_new = 0.0f;
    for (int variable = 0; variable < OP_DIMENSION; ++variable) {
      const float history_s = panel_s[thread * OP_PADDED_DIMENSION + variable];
      accurate_s_dot_g += history_s * gradient[vector_offset + variable];
      accurate_y_dot_g +=
          panel_y[thread * OP_PADDED_DIMENSION + variable] *
          gradient[vector_offset + variable];
      if (pair_valid && thread < OP_HISTORY - 1) {
        accurate_s_dot_y_new +=
            history_s * panel_y[
                (OP_HISTORY - 1) * OP_PADDED_DIMENSION + variable];
      }
    }
    s_dot_g[thread] = accurate_s_dot_g;
    y_dot_g[thread] = accurate_y_dot_g;
    if (pair_valid && thread < OP_HISTORY - 1) {
      s_dot_y_new[thread] = accurate_s_dot_y_new;
    }
  }
  __syncthreads();

  if (pair_valid) {
    constexpr int RETAINED_HISTORY = OP_HISTORY - 1;
    if (thread < RETAINED_HISTORY) {
      float inverse_column = 0.0f;
      #pragma unroll
      for (int column = 0; column < RETAINED_HISTORY; ++column) {
        inverse_column +=
            inverse_r[thread * OP_PADDED_HISTORY + column] * s_dot_y_new[column];
      }
      const float newest_rho =
          rho_buffer[newest_physical * batch_size + batch];
      const float newest_value = -inverse_column * newest_rho;
      inverse_r[thread * OP_PADDED_HISTORY + RETAINED_HISTORY] = newest_value;
      int physical_row = oldest_physical + thread;
      if (physical_row >= OP_HISTORY) {
        physical_row -= OP_HISTORY;
      }
      inverse_r_buffer[
          batch * OP_HISTORY * OP_HISTORY +
          physical_row * OP_HISTORY + newest_physical] = newest_value;
    } else if (thread == RETAINED_HISTORY) {
      const float newest_rho =
          rho_buffer[newest_physical * batch_size + batch];
      inverse_r[RETAINED_HISTORY * OP_PADDED_HISTORY + RETAINED_HISTORY] = newest_rho;
      inverse_r_buffer[
          batch * OP_HISTORY * OP_HISTORY +
          newest_physical * OP_HISTORY + newest_physical] = newest_rho;
    }
  }
  __syncthreads();

  // A failed line search is the practical signal that roundoff in the compact
  // inverse has become significant. The curvature-scale guard catches the
  // same issue before it reaches the line search. Reuse the shared
  // chronological panels for the stable two-loop recurrence in either case.
  if (use_two_loop_fallback) {
    const int first_active = OP_HISTORY - active_pair_count;
    for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
      step[vector_offset + variable] = gradient[vector_offset + variable];
    }
    __syncthreads();

    for (int logical_history = OP_HISTORY - 1;
         logical_history >= first_active;
         --logical_history) {
      float dot_values[1] = {0.0f};
      for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
        dot_values[0] +=
            step[vector_offset + variable] *
            panel_s[logical_history * OP_PADDED_DIMENSION + variable];
      }
      op_block_reduce(dot_values, reduction_scratch);
      int physical = oldest_physical + logical_history;
      if (physical >= OP_HISTORY) {
        physical -= OP_HISTORY;
      }
      const float alpha =
          dot_values[0] * rho_buffer[physical * batch_size + batch];
      if (thread == 0) {
        s_dot_g[logical_history] = alpha;
      }
      for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
        step[vector_offset + variable] -=
            alpha * panel_y[logical_history * OP_PADDED_DIMENSION + variable];
      }
      __syncthreads();
    }

    for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
      step[vector_offset + variable] *= gamma;
    }
    __syncthreads();
    for (int logical_history = first_active;
         logical_history < OP_HISTORY;
         ++logical_history) {
      float dot_values[1] = {0.0f};
      for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
        dot_values[0] +=
            step[vector_offset + variable] *
            panel_y[logical_history * OP_PADDED_DIMENSION + variable];
      }
      op_block_reduce(dot_values, reduction_scratch);
      int physical = oldest_physical + logical_history;
      if (physical >= OP_HISTORY) {
        physical -= OP_HISTORY;
      }
      const float beta =
          dot_values[0] * rho_buffer[physical * batch_size + batch];
      const float coefficient = s_dot_g[logical_history] - beta;
      for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
        step[vector_offset + variable] +=
            coefficient * panel_s[
                logical_history * OP_PADDED_DIMENSION + variable];
      }
      __syncthreads();
    }

    float local_scale = 0.0f;
    for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
      const float direction = step[vector_offset + variable];
      if constexpr (ScaleStep) {
        const float variable_scale =
            fabsf(direction) / optimization_step_max[variable];
        local_scale = local_scale > variable_scale ? local_scale : variable_scale;
      }
      step[vector_offset + variable] = -direction;
    }
    if constexpr (ScaleStep) {
      if (thread == 0) {
        direction_scale_bits = __float_as_int(1.0f);
      }
      __syncthreads();
      local_scale = op_warp_max(local_scale);
      if (thread % 32 == 0) {
        atomicMax(&direction_scale_bits, __float_as_int(local_scale));
      }
      __syncthreads();
      const float direction_scale = __int_as_float(direction_scale_bits);
      for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
        step[vector_offset + variable] /= direction_scale;
      }
    }
    if (thread == 0 && pair_valid) {
      history_head[batch] = head + 1 == OP_HISTORY ? 0 : head + 1;
      history_count[batch] = active_pair_count;
    }
    return;
  }

  for (
    int tile = warp;
    tile < OP_HISTORY_TILES * OP_HISTORY_TILES;
    tile += OP_WARPS)
  {
    const int tile_row = tile / OP_HISTORY_TILES;
    const int tile_column = tile % OP_HISTORY_TILES;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int k = 0; k < OP_PADDED_DIMENSION; k += 8) {
      wmma::fragment<
        wmma::matrix_a,
        16,
        16,
        8,
        wmma::precision::tf32,
        wmma::row_major> a_fragment;
      wmma::fragment<
        wmma::matrix_b,
        16,
        16,
        8,
        wmma::precision::tf32,
        wmma::col_major> b_fragment;
      wmma::load_matrix_sync(
          a_fragment,
          &panel_y[tile_row * 16 * OP_PADDED_DIMENSION + k],
          OP_PADDED_DIMENSION);
      wmma::load_matrix_sync(
          b_fragment,
          &panel_y[tile_column * 16 * OP_PADDED_DIMENSION + k],
          OP_PADDED_DIMENSION);
      // Use the same compensated TF32 product for the history Gram matrix.
      auto a_residual_fragment = a_fragment;
      auto b_residual_fragment = b_fragment;
      #pragma unroll
      for (int element = 0; element < a_fragment.num_elements; ++element) {
        const float a_value = a_fragment.x[element];
        const float a_tf32 = wmma::__float_to_tf32(a_value);
        a_fragment.x[element] = a_tf32;
        a_residual_fragment.x[element] =
            wmma::__float_to_tf32(a_value - a_tf32);
      }
      #pragma unroll
      for (int element = 0; element < b_fragment.num_elements; ++element) {
        const float b_value = b_fragment.x[element];
        const float b_tf32 = wmma::__float_to_tf32(b_value);
        b_fragment.x[element] = b_tf32;
        b_residual_fragment.x[element] =
            wmma::__float_to_tf32(b_value - b_tf32);
      }
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
      wmma::mma_sync(
          accumulator,
          a_fragment,
          b_residual_fragment,
          accumulator);
      wmma::mma_sync(
          accumulator,
          a_residual_fragment,
          b_fragment,
          accumulator);
    }
    wmma::store_matrix_sync(
        &middle_matrix[tile_row * 16 * OP_PADDED_HISTORY + tile_column * 16],
        accumulator,
        OP_PADDED_HISTORY,
        wmma::mem_row_major);
  }
  __syncthreads();

  for (
    int element = thread;
    element < OP_PADDED_HISTORY * OP_PADDED_HISTORY;
    element += blockDim.x)
  {
    const int row = element / OP_PADDED_HISTORY;
    const int column = element - row * OP_PADDED_HISTORY;
    float value = 0.0f;
    if (row < OP_HISTORY && column < OP_HISTORY) {
      value = gamma * middle_matrix[element];
      if (row == column) {
        int physical = oldest_physical + row;
        if (physical >= OP_HISTORY) {
          physical -= OP_HISTORY;
        }
        value += 1.0f / rho_buffer[physical * batch_size + batch];
      }
    }
    middle_matrix[element] = value;
  }
  __syncthreads();
  op_tensor_matrix_product<History>(
      middle_matrix,
      inverse_r,
      transformed_inverse_r);

  // The history-update projection is dead here, so reuse it for
  // z = transformed_inverse_r * (S * gradient) - gamma * (Y * gradient).
  float* coefficient_rhs = s_dot_y_new;
  if (thread < OP_PADDED_HISTORY) {
    float y_value = 0.0f;
    float rhs_value = 0.0f;
    if (thread < OP_HISTORY) {
      #pragma unroll
      for (int column = 0; column < OP_HISTORY; ++column) {
        const float u_value = s_dot_g[column];
        y_value -= gamma *
            inverse_r[thread * OP_PADDED_HISTORY + column] * u_value;
        rhs_value += transformed_inverse_r[
            thread * OP_PADDED_HISTORY + column] * u_value;
      }
      rhs_value -= gamma * y_dot_g[thread];
    }
    y_coefficient[thread] = y_value;
    coefficient_rhs[thread] = rhs_value;
  }
  __syncthreads();
  if (thread < OP_PADDED_HISTORY) {
    float s_value = 0.0f;
    if (thread < OP_HISTORY) {
      #pragma unroll
      for (int row = 0; row < OP_HISTORY; ++row) {
        s_value +=
            inverse_r[row * OP_PADDED_HISTORY + thread] * coefficient_rhs[row];
      }
    }
    s_coefficient[thread] = s_value;
  }
  __syncthreads();
  float local_scale = 0.0f;
  for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
    float direction = gamma * gradient[vector_offset + variable];
    #pragma unroll
    for (int history = 0; history < OP_HISTORY; ++history) {
      direction += panel_y[history * OP_PADDED_DIMENSION + variable] *
          y_coefficient[history];
      direction += panel_s[history * OP_PADDED_DIMENSION + variable] *
          s_coefficient[history];
    }
    step[vector_offset + variable] = -direction;
    if constexpr (ScaleStep) {
      const float variable_scale =
          fabsf(direction) / optimization_step_max[variable];
      local_scale = local_scale > variable_scale ? local_scale : variable_scale;
    }
  }
  if constexpr (ScaleStep) {
    if (thread == 0) {
      direction_scale_bits = __float_as_int(1.0f);
    }
    __syncthreads();
    local_scale = op_warp_max(local_scale);
    if (thread % 32 == 0) {
      atomicMax(&direction_scale_bits, __float_as_int(local_scale));
    }
    __syncthreads();
    const float direction_scale = __int_as_float(direction_scale_bits);
    for (int variable = thread; variable < OP_DIMENSION; variable += blockDim.x) {
      step[vector_offset + variable] /= direction_scale;
    }
  }
  if (thread == 0) {
    if (pair_valid) {
      history_head[batch] = head + 1 == OP_HISTORY ? 0 : head + 1;
      history_count[batch] = active_pair_count;
    }
  }
}

}  // namespace optimization
}  // namespace curobo
