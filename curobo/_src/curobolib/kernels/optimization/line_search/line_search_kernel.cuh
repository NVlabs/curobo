/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



#include "third_party/helper_math.h"
#include "line_search_helpers.cuh"


namespace curobo{
    namespace optimization{
     namespace line_search{


      /**
       * @brief Unified line search kernel that handles both runtime and compile-time line search counts
       *
       * This kernel unifies the functionality of both specialized kernels by using template
       * parameters to enable compile-time optimizations when possible while maintaining
       * runtime flexibility when needed.
       *
       * @tparam ScalarType Floating point type (float/double)
       * @tparam CompileTimeNLineSearch Compile-time line search count (-1 for runtime)
       */
      template<typename ScalarType, int CompileTimeNLineSearch = -1>
      __global__ void kernel_line_search(

        // inputs for update best operation
        ScalarType *best_cost, // batchsize x 1
        ScalarType *best_action, // batchsize x opt_dim
        int16_t *best_iteration, // batchsize x 1
        int16_t *current_iteration, // batchsize x 1
        uint8_t *converged_global, // batchsize x 1
        const int convergence_iteration, // 1
        const float cost_delta_threshold, // 1
        const float cost_relative_threshold, // 1

        // inputs for line search operation
        ScalarType *exploration_cost,           // batchsize x 1
        ScalarType *exploration_action,           // batchsize x opt_dim
        ScalarType *exploration_gradient,        // batchsize x opt_dim
        int32_t *exploration_idx,          // batchsize x n_linesearch
        ScalarType *selected_cost,         // batchsize x 1
        ScalarType *selected_action,       // batchsize x opt_dim
        ScalarType *selected_gradient,     // batchsize x opt_dim
        int32_t *selected_idx,           // batchsize x n_linesearch
        const ScalarType *search_cost,          // batchsize x n_linesearch x 1
        const ScalarType *search_action,      //  batchsize x n_linesearch x opt_dim
        const ScalarType *search_gradient,        //  batchsize x n_linesearch x opt_dim
        const ScalarType *step_direction,   // batchsize x opt_dim
        const ScalarType *search_magnitudes, // n_linesearch x 1
        const float armijo_threshold_c_1, // 1
        const float curvature_threshold_c_2, // 1
        const bool strong_wolfe, // 1
        const bool approx_wolfe, // 1
        const int n_linesearch,               // 1
        const int opt_dim,               // 1
        const int batchsize)        // 1
      {
        int batch = blockIdx.x;
        int local_thread_idx = threadIdx.x;

        __shared__ ScalarType   data[curobo::common::warpSize];
        __shared__ ScalarType result[curobo::common::warpSize];
        __shared__ int idx_shared_selected;
        __shared__ int idx_shared_exploration;
        __shared__ bool update_best_shared;
        __shared__ bool converged_shared;

        if (local_thread_idx >= opt_dim)
        {
          return;
        }
        if (batch >= batchsize)
        {
          return;
        }

        ScalarType sv_elem = step_direction[batch * opt_dim + local_thread_idx];

        // Use compile-time value if available, otherwise runtime value
        constexpr bool use_compile_time = (CompileTimeNLineSearch > 0);

        // Conditional loop unrolling based on compile-time knowledge
        if constexpr (use_compile_time) {
          #pragma unroll
          for (int i = 0; i < CompileTimeNLineSearch; i++)
          {
            ScalarType g_x_elem = search_gradient[batch * CompileTimeNLineSearch * opt_dim + opt_dim * i + local_thread_idx];
            curobo::common::block_reduce_sum(g_x_elem * sv_elem, opt_dim,
                  &data[0], &result[i]);
          }
        } else {
          for (int i = 0; i < n_linesearch; i++)
          {
            ScalarType g_x_elem = search_gradient[batch * n_linesearch * opt_dim + opt_dim * i + local_thread_idx];
            curobo::common::block_reduce_sum(g_x_elem * sv_elem, opt_dim,
                  &data[0], &result[i]);
          }
        }

        bool wolfe_1   = false;
        bool wolfe     = false;
        bool wolfe_2 = false;
        const bool condition = local_thread_idx < n_linesearch;

        if (condition)
        {
          ScalarType local_alpha = search_magnitudes[local_thread_idx];
          ScalarType local_c_0 = search_cost[batch * n_linesearch];
          ScalarType local_c_val = search_cost[batch * n_linesearch + local_thread_idx];
          ScalarType local_g_step_val = result[local_thread_idx];
          ScalarType local_g_step_0 = result[0];

          evaluate_wolfe_conditions(condition, local_alpha,
          local_c_0,
          local_c_val,
          local_g_step_val, local_g_step_0, armijo_threshold_c_1, curvature_threshold_c_2, strong_wolfe, approx_wolfe, wolfe, wolfe_1, wolfe_2);
        }
        __syncthreads();

        // Compute Wolfe indices using extracted function
        int exploration_id = 0;
        int selected_id = 0;
        compute_wolfe_indices(wolfe_1, wolfe, condition, approx_wolfe, strong_wolfe, exploration_id, selected_id);

        if (local_thread_idx == 0)
        {

          // Update shared indices for later use
          idx_shared_selected = selected_id + batch * n_linesearch;
          idx_shared_exploration = exploration_id + batch * n_linesearch;

          // Update costs and check convergence using extracted function
          update_costs_and_convergence(batch, exploration_id, selected_id, n_linesearch,
                                     convergence_iteration, cost_delta_threshold, cost_relative_threshold,
                                     search_cost, exploration_cost, selected_cost, best_cost,
                                     best_iteration, current_iteration, converged_global,
                                     update_best_shared, converged_shared);
        }

        __syncthreads();

        // Copy action and gradient results using extracted function
        copy_action_gradient_results(batch, local_thread_idx,
                                    idx_shared_exploration,
                                    idx_shared_selected,
                                    opt_dim, n_linesearch, update_best_shared,
                                    search_action, search_gradient,
                                    exploration_action, exploration_gradient,
                                    selected_action, selected_gradient, best_action,
                                    exploration_idx, selected_idx);
      }

}
}
}