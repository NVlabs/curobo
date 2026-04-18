/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "common/block_warp_reductions.cuh"
#include "third_party/helper_math.h"

namespace curobo{
namespace optimization{
namespace lbfgs{

    /**
     * @brief Loads current L-BFGS state and computes y and s vectors
     *
     * This function handles the common pattern of loading current gradients and positions,
     * computing the difference vectors y (gradient difference) and s (position difference),
     * and updating the state arrays. Also returns the current gradient value to avoid
     * a redundant global memory load in the caller.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param batch Current batch index
     * @param thread_idx Current thread index within batch
     * @param v_dim Optimization dimension size
     * @param grad_q Current gradient array
     * @param q Current position array
     * @param grad_0 Previous gradient array (input/output)
     * @param x_0 Previous position array (input/output)
     * @param y Output y vector (gradient difference)
     * @param s Output s vector (position difference)
     * @param gq_out Output current gradient value for reuse by caller
     */
    template<typename ScalarType>
    __device__ __forceinline__ void load_lbfgs_state_and_compute_differences(
        const int batch,
        const int thread_idx,
        const int v_dim,
        const ScalarType* grad_q,
        const ScalarType* q,
        ScalarType* grad_0,
        ScalarType* x_0,
        ScalarType& y,
        ScalarType& s,
        ScalarType& gq_out)
    {
        const uint32_t batch_vdim_tidx = batch * v_dim + thread_idx;

        // Load current state
        ScalarType gq = grad_q[batch_vdim_tidx];
        ScalarType q_t = q[batch_vdim_tidx];

        // Load previous state
        ScalarType gq_0 = grad_0[batch_vdim_tidx];
        ScalarType q_0 = x_0[batch_vdim_tidx];

        // Compute differences
        y = gq - gq_0;
        s = q_t - q_0;

        // Update previous state for next iteration
        grad_0[batch_vdim_tidx] = gq;
        x_0[batch_vdim_tidx] = q_t;

        // Return gradient value for reuse (avoids redundant global load)
        gq_out = gq;
    }

    /**
     * @brief Updates L-BFGS history buffers with new y and s vectors
     *
     * This function handles the rolling/shifting of history buffers and updates
     * them with new y and s vectors. Supports both rolled and non-rolled buffer modes.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @tparam rolled_ys Whether buffers are pre-rolled
     * @param batch Current batch index
     * @param thread_idx Current thread index within batch
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m History buffer size
     * @param y New y vector value
     * @param s New s vector value
     * @param y_buffer History buffer for y vectors (input/output)
     * @param s_buffer History buffer for s vectors (input/output)
     */
    template<typename ScalarType, bool rolled_ys, int FIXED_M = -1>
    __device__ __forceinline__ void update_lbfgs_history_buffers(
        const int batch,
        const int thread_idx,
        const int batchsize,
        const int v_dim,
        const int history_m,
        const ScalarType y,
        const ScalarType s,
        ScalarType* y_buffer,
        ScalarType* s_buffer)
    {
        const uint32_t batch_vdim_tidx = batch * v_dim + thread_idx;

        if constexpr (!rolled_ys) {
            constexpr bool is_compile_time = (FIXED_M > 0);
            if constexpr (is_compile_time) {
                for (int i = 1; i < FIXED_M; i++) {
                    const uint32_t src_idx = i * batchsize * v_dim + batch_vdim_tidx;
                    const uint32_t dst_idx = (i - 1) * batchsize * v_dim + batch_vdim_tidx;

                    s_buffer[dst_idx] = s_buffer[src_idx];
                    y_buffer[dst_idx] = y_buffer[src_idx];
                }
            }
            else {


            // Shift old values: move [1, 2, ..., m-1] to [0, 1, ..., m-2]
            for (int i = 1; i < history_m; i++) {
                const uint32_t src_idx = i * batchsize * v_dim + batch_vdim_tidx;
                const uint32_t dst_idx = (i - 1) * batchsize * v_dim + batch_vdim_tidx;

                s_buffer[dst_idx] = s_buffer[src_idx];
                y_buffer[dst_idx] = y_buffer[src_idx];
            }
            }
        }

        // Store new values at the end
        const uint32_t new_idx = (history_m - 1) * batchsize * v_dim + batch_vdim_tidx;
        s_buffer[new_idx] = s;
        y_buffer[new_idx] = y;
    }

    /**
     * @brief Updates L-BFGS history buffers from shared memory data (unified optimized)
     *
     * This unified function avoids redundant global memory reads by using the already-shifted
     * data from shared memory buffers. Uses constexpr to optimize for compile-time
     * known history sizes while supporting runtime sizes.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @tparam HISTORY_M Compile-time history size (use -1 for runtime)
     * @param thread_idx Current thread index within batch
     * @param batch Current batch index
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m Runtime history size (ignored if HISTORY_M > 0)
     * @param y_buffer_sh Shared memory y buffer (already contains shifted data)
     * @param s_buffer_sh Shared memory s buffer (already contains shifted data)
     * @param y_buffer Global y history buffer (output)
     * @param s_buffer Global s history buffer (output)
     */
    template<typename ScalarType, int HISTORY_M = -1>
    __device__ __forceinline__ void update_lbfgs_history_buffers_from_shared(
        const int thread_idx,
        const int batch,
        const int batchsize,
        const int v_dim,
        const int history_m,
        const float* y_buffer_sh,
        const float* s_buffer_sh,
        ScalarType* y_buffer,
        ScalarType* s_buffer)
    {
        const uint32_t batch_vdim_tidx = batch * v_dim + thread_idx;

        // Use compile-time or runtime history size
        constexpr bool is_compile_time = (HISTORY_M > 0);
        uint32_t current_shared_idx = 0;

        // Copy already-shifted data from shared memory to global memory
        // The shared memory already contains the properly shifted history with new values
        if constexpr (is_compile_time) {
            current_shared_idx = HISTORY_M * thread_idx;

            // Compile-time version with loop unrolling
            #pragma unroll
            for (int i = 0; i < HISTORY_M; i++) {
                const uint32_t global_idx = i * batchsize * v_dim + batch_vdim_tidx;

                s_buffer[global_idx] = static_cast<ScalarType>(s_buffer_sh[current_shared_idx + i]);
                y_buffer[global_idx] = static_cast<ScalarType>(y_buffer_sh[current_shared_idx + i]);
            }
        } else {
            // Runtime version
            current_shared_idx = history_m * thread_idx;

            for (int i = 0; i < history_m; i++) {
                const uint32_t global_idx = i * batchsize * v_dim + batch_vdim_tidx;

                s_buffer[global_idx] = static_cast<ScalarType>(s_buffer_sh[current_shared_idx + i]);
                y_buffer[global_idx] = static_cast<ScalarType>(y_buffer_sh[current_shared_idx + i]);
            }
        }
    }

    /**
     * @brief Updates rho buffer with new curvature information
     *
     * This function computes the new rho value (1 / (y^T * s)) and updates
     * the rho buffer, handling both stability checks and buffer rolling.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param batch Current batch index
     * @param thread_idx Current thread index within batch
     * @param batchsize Total batch size
     * @param history_m History buffer size
     * @param denominator Computed y^T * s value
     * @param stable_mode Whether to apply stability checks
     * @param rho_buffer Rho history buffer (input/output)
     */
    template<typename ScalarType>
    __device__ __forceinline__ void update_rho_buffer(
        const int batch,
        const int thread_idx,
        const int batchsize,
        const int history_m,
        const ScalarType denominator,
        const bool stable_mode,
        ScalarType* rho_buffer)
    {
        // Shift old rho values (threads < history_m - 1)
        if (thread_idx < history_m - 1) {
            ScalarType rho = rho_buffer[(thread_idx + 1) * batchsize + batch];
            rho_buffer[thread_idx * batchsize + batch] = rho;
        }

        // Compute and store new rho value (thread == history_m - 1)
        if (thread_idx == history_m - 1) {
            ScalarType rho = 1.0 / denominator;

            // Stability check: avoid division by zero
            if (stable_mode && (denominator <= 0.0)) {
                rho = 0.0;
            }

            rho_buffer[thread_idx * batchsize + batch] = rho;
        }
    }

    /**
     * @brief Performs the backward pass of the L-BFGS two-loop algorithm
     *
     * This function implements the first loop of the L-BFGS two-loop recursion,
     * computing alpha values and updating the search direction.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param thread_idx Current thread index within batch
     * @param batch Current batch index
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m History buffer size
     * @param gq Current search direction (input/output)
     * @param s_buffer History buffer for s vectors
     * @param y_buffer History buffer for y vectors
     * @param rho_buffer History buffer for rho values
     * @param alpha_buffer Output buffer for alpha values
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType, int HISTORY_M = -1>
    __device__ __forceinline__ void lbfgs_backward_pass(
        const int thread_idx,
        const int batch,
        const int batchsize,
        const int v_dim,
        const int history_m,
        ScalarType& gq,
        const ScalarType* s_buffer,
        const ScalarType* y_buffer,
        const ScalarType* rho_buffer,
        ScalarType* alpha_buffer,
        ScalarType* data,
        float* result)
    {
        constexpr bool is_compile_time = (HISTORY_M > 0);

        if constexpr (is_compile_time) {
            for (int i = HISTORY_M - 1; i > -1; i--) {
                float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_rho = rho_buffer[i * batchsize + batch];
                // Compute s^T * gq
                curobo::common::block_reduce_sum(
                    gq * current_s,
                    v_dim, &data[0], result);

                // alpha_i = rho_i * s_i^T * gq
                float current_alpha = result[0] * current_rho;

                // gq = gq - alpha_i * y_i
                gq = gq - current_alpha * current_y;

                if (thread_idx == 0)
                {
                    alpha_buffer[i] = current_alpha;
                }

            }
        }
        else {
            for (int i = history_m - 1; i > -1; i--) {
                float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_rho = rho_buffer[i * batchsize + batch];

                // Compute s^T * gq
                curobo::common::block_reduce_sum(
                    gq * current_s,
                    v_dim, &data[0], result);

                // alpha_i = rho_i * s_i^T * gq
                float current_alpha = result[0] * current_rho;


                // gq = gq - alpha_i * y_i
                gq = gq - current_alpha * current_y;

                if (thread_idx == 0)
                {
                  alpha_buffer[i] = current_alpha;
                }

            }

        }
    }

    /**
     * @brief Computes the L-BFGS scaling factor (gamma)
     *
     * This function computes the Hessian scaling factor gamma = (s^T * y) / (y^T * y)
     * and applies it to the search direction, with stability checks.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param y Latest y vector value
     * @param numerator Precomputed y^T * s value
     * @param epsilon Stability epsilon value
     * @param stable_mode Whether to apply stability checks
     * @param gq Search direction (input/output)
     * @param v_dim Optimization dimension size
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType>
    __device__ __forceinline__ void compute_lbfgs_scaling(
        const ScalarType y,
        const ScalarType numerator,
        const float epsilon,
        const bool stable_mode,
        ScalarType& gq,
        const int v_dim,
        ScalarType* data,
        float* result)
    {
        // Compute y^T * y
        curobo::common::block_reduce_sum(y * y, v_dim, &data[0], result);
        ScalarType denominator = result[0];

        // Compute gamma = (s^T * y) / (y^T * y)
        // Negative gamma (curvature condition violated) is clamped to 0 by relu below,
        // which lets the forward pass reconstruct the step from history alone.
        ScalarType var1 = numerator / denominator;

        // Apply stability checks
        if (stable_mode && (isinf(var1) || isnan(var1))) {
            var1 = epsilon;
        }

        // Apply scaling: gq = gamma * gq
        ScalarType gamma = curobo::common::relu(var1);
        gq = gamma * gq;
    }

    /**
     * @brief Performs the forward pass of the L-BFGS two-loop algorithm
     *
     * This function implements the second loop of the L-BFGS two-loop recursion,
     * computing the final search direction.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param thread_idx Current thread index within batch
     * @param batch Current batch index
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m History buffer size
     * @param gq Search direction (input/output)
     * @param s_buffer History buffer for s vectors
     * @param y_buffer History buffer for y vectors
     * @param rho_buffer History buffer for rho values
     * @param alpha_buffer Alpha values from backward pass
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType, int HISTORY_M = -1>
    __device__ __forceinline__ void lbfgs_forward_pass(
        const int thread_idx,
        const int batch,
        const int batchsize,
        const int v_dim,
        const int history_m,
        ScalarType& gq,
        const ScalarType* s_buffer,
        const ScalarType* y_buffer,
        const ScalarType* rho_buffer,
        const ScalarType* alpha_buffer,
        ScalarType* data,
        float* result)
    {
        constexpr bool is_compile_time = (HISTORY_M > 0);

        if constexpr (is_compile_time) {
            for (int i = 0; i < HISTORY_M; i++) {
            float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
            float current_rho = rho_buffer[i * batchsize + batch];
            //float current_alpha = alpha_buffer[ thread_idx * HISTORY_M + i];
            float current_alpha = alpha_buffer[i];

            float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];

            // Compute y^T * gq
            curobo::common::block_reduce_sum(
                gq * current_y,
                v_dim, &data[0], result);

            // beta = rho_i * y_i^T * gq
            ScalarType beta = result[0] * current_rho;

            // gq = gq + (alpha_i - beta) * s_i
                gq = gq + (current_alpha - beta) *
                         current_s;
            }
        }
        else {
            for (int i = 0; i < history_m; i++) {

                float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_rho = rho_buffer[i * batchsize + batch];

                //float current_alpha = alpha_buffer[ thread_idx * history_m + i];
                float current_alpha = alpha_buffer[i];

                float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];

                // Compute y^T * gq
                curobo::common::block_reduce_sum(
                    gq * current_y,
                    v_dim, &data[0], result);

                // beta = rho_i * y_i^T * gq
                ScalarType beta = result[0] * current_rho;

                // gq = gq + (alpha_i - beta) * s_i
                gq = gq + (current_alpha - beta) * current_s;
            }
        }
    }

    /**
     * @brief Copy buffers to shared memory for optimized access (unified version)
     *
     * This unified function copies relevant portions of the history buffers to shared memory
     * and enables faster access patterns. Uses constexpr to optimize for compile-time
     * known history sizes while supporting runtime sizes.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @tparam HISTORY_M Compile-time history size (use -1 for runtime)
     * @param thread_idx Current thread index within batch
     * @param batch Current batch index
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m Runtime history size (ignored if HISTORY_M > 0)
     * @param y Latest y vector value
     * @param s Latest s vector value
     * @param y_buffer Global y history buffer
     * @param s_buffer Global s history buffer
     * @param rho_buffer Global rho history buffer
     * @param y_buffer_sh Shared memory y buffer (output)
     * @param s_buffer_sh Shared memory s buffer (output)
     * @param rho_buffer_sh Shared memory rho buffer (output)
     * @param rolled_ys Whether buffers are pre-rolled (only used for runtime)
     */
    template<typename ScalarType, int HISTORY_M = -1>
    __device__ __forceinline__ void copy_buffers_to_shared_memory(
        const int thread_idx,
        const int batch,
        const int batchsize,
        const int v_dim,
        const int history_m,
        const ScalarType y,
        const ScalarType s,
        const ScalarType* y_buffer,
        const ScalarType* s_buffer,
        const ScalarType* rho_buffer,
        float* y_buffer_sh,
        float* s_buffer_sh,
        float* rho_buffer_sh,
        const bool rolled_ys = false)
    {
        const uint32_t batch_vdim_tidx = batch * v_dim + thread_idx;

        // Use compile-time or runtime history size
        constexpr bool is_compile_time = (HISTORY_M > 0);
        const int effective_history_m = is_compile_time ? HISTORY_M : history_m;
        const uint32_t current_shared_idx = effective_history_m * thread_idx;

        // Copy existing history to shared memory
        if constexpr (is_compile_time) {
            // Compile-time version with loop unrolling
            #pragma unroll
            for (int i = 1; i < HISTORY_M; i++) {
                float yt = y_buffer[i * batchsize * v_dim + batch_vdim_tidx];
                float st = s_buffer[i * batchsize * v_dim + batch_vdim_tidx];

                y_buffer_sh[current_shared_idx + i - 1] = yt;
                s_buffer_sh[current_shared_idx + i - 1] = st;
            }
        } else {
            // Runtime version with conditional rolling support
            if (!rolled_ys) {
                for (int i = 1; i < effective_history_m; i++) {
                    float yt = y_buffer[i * batchsize * v_dim + batch_vdim_tidx];
                    float st = s_buffer[i * batchsize * v_dim + batch_vdim_tidx];

                    y_buffer_sh[current_shared_idx + i - 1] = yt;
                    s_buffer_sh[current_shared_idx + i - 1] = st;
                }
            }
        }

        // Store new values at the end
        s_buffer_sh[current_shared_idx + effective_history_m - 1] = s;
        y_buffer_sh[current_shared_idx + effective_history_m - 1] = y;

        // Copy rho values to shared memory (limited threads participate)
        if constexpr (is_compile_time) {
            if (thread_idx < HISTORY_M - 1) {
                ScalarType rho = rho_buffer[(thread_idx + 1) * batchsize + batch];
                rho_buffer_sh[thread_idx] = rho;
            }
        } else {
            if (thread_idx < effective_history_m - 1) {
                ScalarType rho = rho_buffer[(thread_idx + 1) * batchsize + batch];
                rho_buffer_sh[thread_idx] = rho;
            }
            if (thread_idx == effective_history_m - 1) {
                ScalarType rho = rho_buffer[thread_idx * batchsize + batch];
                rho_buffer_sh[thread_idx] = rho;
            }
        }
    }

    /**
     * @brief Perform L-BFGS backward pass using shared memory buffers
     *
     * Optimized version of the backward pass that uses shared memory for faster access.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @tparam HISTORY_M Compile-time history size
     * @param thread_idx Current thread index within batch
     * @param v_dim Optimization dimension size
     * @param gq Search direction (input/output)
     * @param s_buffer_sh Shared memory s buffer
     * @param y_buffer_sh Shared memory y buffer
     * @param rho_buffer_sh Shared memory rho buffer
     * @param alpha_buffer_sh Shared memory alpha buffer (output)
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType, int HISTORY_M>
    __device__ __forceinline__ void lbfgs_backward_pass_shared(
        const int thread_idx,
        const int v_dim,
        const int history_m,
        ScalarType& gq,
        const float* s_buffer_sh,
        const float* y_buffer_sh,
        const float* rho_buffer_sh,
        float* alpha_buffer_sh,
        ScalarType* data,
        float* result)
    {
        const uint32_t current_shared_idx = history_m * thread_idx;

        constexpr bool is_compile_time = (HISTORY_M > 0);

        if constexpr (is_compile_time) {

            for (int i = HISTORY_M - 1; i > -1; i--) {
                float current_s = s_buffer_sh[current_shared_idx + i];
                float current_y = y_buffer_sh[current_shared_idx + i];
                float current_rho = rho_buffer_sh[i];


                // Compute s^T * gq
                curobo::common::block_reduce_sum(gq * current_s, v_dim, &data[0], &result[0]);

                // alpha_i = rho_i * s_i^T * gq
                float current_alpha = result[0] * current_rho;

                // gq = gq - alpha_i * y_i
                gq = gq - current_alpha * current_y;
                if (thread_idx == 0)
                {
                    alpha_buffer_sh[i] = current_alpha;
                }

            }
        } else {
            for (int i = history_m - 1; i > -1; i--) {
                float current_s = s_buffer_sh[current_shared_idx + i];
                float current_y = y_buffer_sh[current_shared_idx + i];
                float current_rho = rho_buffer_sh[i];


                // Compute s^T * gq
                curobo::common::block_reduce_sum(gq * current_s, v_dim, &data[0], &result[0]);

                // alpha_i = rho_i * s_i^T * gq
                float current_alpha = result[0] * current_rho;

                // gq = gq - alpha_i * y_i
                gq = gq - current_alpha * current_y;
                if (thread_idx == 0)
                {
                    alpha_buffer_sh[i] = current_alpha;
                }

            }

        }
    }

    /**
     * @brief Perform L-BFGS forward pass using shared memory buffers
     *
     * Optimized version of the forward pass that uses shared memory for faster access.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @tparam HISTORY_M Compile-time history size
     * @param thread_idx Current thread index within batch
     * @param v_dim Optimization dimension size
     * @param gq Search direction (input/output)
     * @param s_buffer_sh Shared memory s buffer
     * @param y_buffer_sh Shared memory y buffer
     * @param rho_buffer_sh Shared memory rho buffer
     * @param alpha_buffer_sh Shared memory alpha buffer
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType, int HISTORY_M>
    __device__ __forceinline__ void lbfgs_forward_pass_shared(
        const int thread_idx,
        const int v_dim,
        const int history_m,

        ScalarType& gq,
        const float* s_buffer_sh,
        const float* y_buffer_sh,
        const float* rho_buffer_sh,
        const float* alpha_buffer_sh,
        ScalarType* data,
        float* result)
    {
        constexpr bool is_compile_time = (HISTORY_M > 0);
        const uint32_t current_shared_idx = history_m * thread_idx;


        if constexpr (is_compile_time) {


            for (int i = 0; i < HISTORY_M; i++) {
                float current_y = y_buffer_sh[current_shared_idx + i];
                float current_s = s_buffer_sh[current_shared_idx + i];
                //float current_alpha = alpha_buffer_sh[thread_idx * HISTORY_M + i];
                float current_alpha = alpha_buffer_sh[i];

                float current_rho = rho_buffer_sh[i];

                // Compute y^T * gq
                curobo::common::block_reduce_sum(gq * current_y, v_dim, &data[0], &result[0]);

                // beta = rho_i * y_i^T * gq
                float beta = result[0] * current_rho;

                // gq = gq + (alpha_i - beta) * s_i
                gq = gq + (current_alpha - beta) * current_s;
            }
        } else {
            for (int i = 0; i < history_m; i++) {
                float current_y = y_buffer_sh[current_shared_idx + i];
                float current_s = s_buffer_sh[current_shared_idx + i];
                //float current_alpha = alpha_buffer_sh[thread_idx * history_m + i];
                float current_alpha = alpha_buffer_sh[i];

                float current_rho = rho_buffer_sh[i];

                // Compute y^T * gq
                curobo::common::block_reduce_sum(gq * current_y, v_dim, &data[0], &result[0]);

                // beta = rho_i * y_i^T * gq
                float beta = result[0] * current_rho;

                // gq = gq + (alpha_i - beta) * s_i
                gq = gq + (current_alpha - beta) * current_s;
            }
        }
    }

} // namespace lbfgs
} // namespace optimization
} // namespace curobo