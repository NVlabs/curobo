/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



#include "third_party/helper_math.h"
#include "common/block_warp_reductions.cuh"
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "lbfgs_step_helpers.cuh"

namespace curobo{
namespace optimization{


template<typename ScalarType, bool rolled_ys, int FIXED_M = -1>
__global__ void kernel_lbfgs_step(
  ScalarType *step_vec,     // b x 175
  ScalarType *rho_buffer,   // m x b x 1
  ScalarType *y_buffer,     // m x b x 175
  ScalarType *s_buffer,     // m x b x 175
  ScalarType *q,            // b x 175
  ScalarType *x_0,          // b x 175
  ScalarType *grad_0,       // b x 175
  const ScalarType *grad_q, // b x 175
  const float epsilon, const int batchsize, const int m, const int v_dim,
  const bool stable_mode = false)                // s_buffer and y_buffer are not rolled by default
{
  extern __shared__ float alpha_buffer_sh[];

  // Constexpr logic for compile-time vs runtime
  constexpr bool is_compile_time = (FIXED_M > 0);
  const int effective_m = is_compile_time ? FIXED_M : m;

  __shared__ ScalarType data[32];
  // temporary buffer needed for block-wide reduction
  __shared__ ScalarType result;
  // result of the reduction or vector-vector dot product
  int batch = blockIdx.x; // one block per batch

  if (threadIdx.x >= v_dim)
    return;

  // Load state and compute y, s differences; also returns gq to avoid redundant global load
  ScalarType y, s, gq;
  curobo::optimization::lbfgs::load_lbfgs_state_and_compute_differences(
      batch, threadIdx.x, v_dim, grad_q, q, grad_0, x_0, y, s, gq);

  // Compute y^T * s for rho calculation
  curobo::common::block_reduce_sum(y * s, v_dim, &data[0], &result);
  ScalarType numerator = result;

  // Update history buffers
  curobo::optimization::lbfgs::update_lbfgs_history_buffers<ScalarType, rolled_ys, FIXED_M>(
      batch, threadIdx.x, batchsize, v_dim, effective_m, y, s, y_buffer, s_buffer);

  // Update rho buffer
  curobo::optimization::lbfgs::update_rho_buffer(
      batch, threadIdx.x, batchsize, effective_m, numerator, stable_mode, rho_buffer);

  ////////////////////
  // L-BFGS two-loop algorithm
  ////////////////////

  // gq already loaded from load_lbfgs_state_and_compute_differences (no redundant global read)

  // Backward pass (first loop)
  curobo::optimization::lbfgs::lbfgs_backward_pass<ScalarType, FIXED_M>(
      threadIdx.x, batch, batchsize, v_dim, effective_m, gq, s_buffer, y_buffer,
      rho_buffer, alpha_buffer_sh, &data[0], &result);

  // Reload y from history buffer to shorten its live range (cheap L1 hit, saves a register
  // across the entire backward pass)
  ScalarType y_latest = y_buffer[(effective_m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x];

  // Compute L-BFGS scaling factor and apply it
  curobo::optimization::lbfgs::compute_lbfgs_scaling(
      y_latest, numerator, epsilon, stable_mode, gq, v_dim, &data[0], &result);

  // Forward pass (second loop)
  curobo::optimization::lbfgs::lbfgs_forward_pass<ScalarType, FIXED_M>(
      threadIdx.x, batch, batchsize, v_dim, effective_m, gq, s_buffer, y_buffer,
      rho_buffer, alpha_buffer_sh, &data[0], &result);


  // Store final step direction
  step_vec[batch * v_dim + threadIdx.x] = -gq;
}



template<typename ScalarType, bool rolled_ys, int FIXED_M = -1>
__global__ void kernel_lbfgs_step_shared_memory(
  ScalarType *step_vec,     // b x 175
  ScalarType *rho_buffer,   // m x b x 1
  ScalarType *y_buffer,     // m x b x 175
  ScalarType *s_buffer,     // m x b x 175
  ScalarType *q,            // b x 175
  ScalarType *x_0,          // b x 175
  ScalarType *grad_0,       // b x 175
  const ScalarType *grad_q, // b x 175
  const float epsilon, const int batchsize, const int lbfgs_history, const int v_dim,
  const bool stable_mode = false)                // s_buffer and y_buffer are not rolled by default
{
  extern __shared__ float my_smem_rc[];

  // Constexpr logic for compile-time vs runtime optimization
  constexpr bool is_compile_time = (FIXED_M > 0);
  const int effective_history_m = is_compile_time ? FIXED_M : lbfgs_history;

  // Shared memory layout - optimized for compile-time vs runtime
  float* s_buffer_sh;
  float* y_buffer_sh;
  float* alpha_buffer_sh;
  float* rho_buffer_sh;
  ScalarType* data;
  float* result;

  // Dynamic layout with runtime lbfgs_history
  s_buffer_sh = (float*)&my_smem_rc;
  y_buffer_sh = &s_buffer_sh[effective_history_m * v_dim];
  alpha_buffer_sh = &y_buffer_sh[effective_history_m * v_dim];
  rho_buffer_sh = &alpha_buffer_sh[effective_history_m];
  data = (ScalarType*)&rho_buffer_sh[effective_history_m];
  result = (float*)&data[32];



  int batch = blockIdx.x;                                  // one block per batch

  if (threadIdx.x >= v_dim)
    return;

  // Load state and compute y, s differences; also returns gq to avoid redundant global load
  ScalarType y, s, gq;
  curobo::optimization::lbfgs::load_lbfgs_state_and_compute_differences(
      batch, threadIdx.x, v_dim, grad_q, q, grad_0, x_0, y, s, gq);

  // Copy buffers to shared memory for optimized access
  curobo::optimization::lbfgs::copy_buffers_to_shared_memory<ScalarType, FIXED_M>(
      threadIdx.x, batch, batchsize, v_dim, effective_history_m, y, s,
      y_buffer, s_buffer, rho_buffer,
      y_buffer_sh, s_buffer_sh, rho_buffer_sh, rolled_ys);

  // Update global buffers from shared memory (optimized - no redundant global reads)
  curobo::optimization::lbfgs::update_lbfgs_history_buffers_from_shared<ScalarType, FIXED_M>(
      threadIdx.x, batch, batchsize, v_dim, effective_history_m,
      y_buffer_sh, s_buffer_sh, y_buffer, s_buffer);

  // Compute y^T * s for rho calculation
  curobo::common::block_reduce_sum(y * s, v_dim, &data[0], &result[0]);
  ScalarType numerator = result[0];


  // Update rho buffer and shared memory
  curobo::optimization::lbfgs::update_rho_buffer(
      batch, threadIdx.x, batchsize, effective_history_m, numerator, stable_mode, rho_buffer);

  // Update shared memory with new rho value
  if (threadIdx.x < effective_history_m) {
    rho_buffer_sh[threadIdx.x] = rho_buffer[threadIdx.x * batchsize + batch];
  }

  __syncthreads();

  ////////////////////
  // L-BFGS two-loop algorithm using shared memory
  ////////////////////

  // gq already loaded from load_lbfgs_state_and_compute_differences (no redundant global read)

  // Backward pass (first loop) using shared memory
  curobo::optimization::lbfgs::lbfgs_backward_pass_shared<ScalarType, FIXED_M>(
      threadIdx.x, v_dim, effective_history_m,
      gq, s_buffer_sh, y_buffer_sh, rho_buffer_sh,
      alpha_buffer_sh, &data[0], &result[0]);

  // Reload y from shared memory to shorten its live range (saves a register across
  // the entire backward pass; cheap shared memory read)
  const uint32_t y_sh_idx = effective_history_m * threadIdx.x + (effective_history_m - 1);
  ScalarType y_latest = y_buffer_sh[y_sh_idx];

  // Compute L-BFGS scaling factor and apply it
  curobo::optimization::lbfgs::compute_lbfgs_scaling(
      y_latest, numerator, epsilon, stable_mode, gq, v_dim, &data[0], &result[0]);

  // Forward pass (second loop) using shared memory
  curobo::optimization::lbfgs::lbfgs_forward_pass_shared<ScalarType, FIXED_M>(
      threadIdx.x, v_dim, effective_history_m,
      gq, s_buffer_sh, y_buffer_sh, rho_buffer_sh,
      alpha_buffer_sh, &data[0], &result[0]);



  // Store final step direction
  step_vec[batch * v_dim + threadIdx.x] = -gq;
}
}
}
