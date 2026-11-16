/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common/block_warp_reductions.cuh"
#include "third_party/helper_math.h"
#include "line_search_helpers.cuh"


namespace curobo{
    namespace optimization{
     namespace line_search{


      constexpr int LINE_SEARCH_MODE_WOLFE = 0;
      constexpr int LINE_SEARCH_MODE_STRONG_WOLFE = 1;
      constexpr int LINE_SEARCH_MODE_APPROX_WOLFE = 2;


      /**
       * @brief Fused line-search selection and optimizer-state update kernel.
       *
       * The probe count and line-search mode are specialized through NVRTC.
       *
       * @tparam ScalarType Floating point type (float/double)
       * @tparam CompileTimeNLineSearch Compile-time line-search probe count
       */
      template<
          typename ScalarType,
          int CompileTimeNLineSearch,
          int CompileTimeLineSearchMode,
          bool SingleVariablePerThread,
          bool Backtracking>
      __global__ void kernel_line_search(

        // inputs for update best operation
        ScalarType *best_cost, // batchsize x 1
        ScalarType *best_action, // batchsize x opt_dim
        int32_t *best_iteration, // batchsize x 1
        int32_t *current_iteration, // batchsize x 1
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
        const ScalarType *search_magnitudes, // n_linesearch
        int32_t *line_search_index, // batchsize
        int32_t *line_search_status, // batchsize
        bool *curvature_update_valid, // batchsize
        const float armijo_threshold_c_1, // 1
        const float curvature_threshold_c_2, // 1
        const int opt_dim,               // 1
        const int batchsize,        // 1
        ScalarType *beta)  // batchsize; persistent parallel-probe multiplier
      {
        static_assert(
            CompileTimeNLineSearch > 0 &&
            CompileTimeNLineSearch <= curobo::common::warpSize,
            "CUDA-core line search requires 1 through 32 probes");
        static_assert(
            CompileTimeLineSearchMode >= LINE_SEARCH_MODE_WOLFE &&
            CompileTimeLineSearchMode <= LINE_SEARCH_MODE_APPROX_WOLFE,
            "Unsupported CUDA-core line-search mode");
        constexpr int n_linesearch = CompileTimeNLineSearch;
        constexpr bool strong_wolfe =
            CompileTimeLineSearchMode == LINE_SEARCH_MODE_STRONG_WOLFE;
        constexpr bool approx_wolfe =
            CompileTimeLineSearchMode == LINE_SEARCH_MODE_APPROX_WOLFE;
        const int batch = blockIdx.x;
        const int local_thread_idx = threadIdx.x;

        __shared__ ScalarType data[curobo::common::warpSize];
        __shared__ ScalarType directional_derivative[CompileTimeNLineSearch];
        __shared__ ScalarType baseline_directional_derivative;
        __shared__ int idx_shared_selected;
        __shared__ int idx_shared_exploration;
        __shared__ bool update_best_shared;

        if (batch >= batchsize)
        {
          return;
        }

        using IndexType = int32_t;
        const IndexType batch_step_offset = static_cast<IndexType>(batch) * opt_dim;
        const IndexType batch_search_offset =
            static_cast<IndexType>(batch) * n_linesearch * opt_dim;

        if constexpr (SingleVariablePerThread) {
          const bool has_variable = local_thread_idx < opt_dim;
          const ScalarType direction_value = has_variable ?
              step_direction[batch_step_offset + local_thread_idx] : ScalarType(0);
          const ScalarType baseline_gradient_value = has_variable ?
              exploration_gradient[batch_step_offset + local_thread_idx] : ScalarType(0);
          curobo::common::block_reduce_sum(
              baseline_gradient_value * direction_value, blockDim.x, &data[0],
              &baseline_directional_derivative);

          #pragma unroll
          for (int candidate = 0; candidate < n_linesearch; ++candidate)
          {
            const IndexType candidate_offset =
                batch_search_offset + static_cast<IndexType>(candidate) * opt_dim;
            const ScalarType gradient_value = has_variable ?
                search_gradient[candidate_offset + local_thread_idx] : ScalarType(0);
            curobo::common::block_reduce_sum(
                gradient_value * direction_value, blockDim.x, &data[0],
                &directional_derivative[candidate]);
          }
        } else {
          ScalarType local_baseline_dot = ScalarType(0);
          for (int variable_idx = local_thread_idx; variable_idx < opt_dim;
               variable_idx += blockDim.x)
          {
            const ScalarType gradient_value =
                exploration_gradient[batch_step_offset + variable_idx];
            local_baseline_dot +=
                gradient_value * step_direction[batch_step_offset + variable_idx];
          }
          curobo::common::block_reduce_sum(
              local_baseline_dot, blockDim.x, &data[0], &baseline_directional_derivative);

          #pragma unroll
          for (int candidate = 0; candidate < n_linesearch; ++candidate)
          {
            ScalarType local_dot = ScalarType(0);
            const IndexType candidate_offset =
                batch_search_offset + static_cast<IndexType>(candidate) * opt_dim;
            for (int variable_idx = local_thread_idx; variable_idx < opt_dim;
                 variable_idx += blockDim.x)
            {
              const ScalarType gradient_value =
                  search_gradient[candidate_offset + variable_idx];
              local_dot += gradient_value * step_direction[batch_step_offset + variable_idx];
            }
            curobo::common::block_reduce_sum(
                local_dot, blockDim.x, &data[0], &directional_derivative[candidate]);
          }
        }

        int exploration_id = -1;
        int selected_id = -1;

        const bool is_candidate_thread = local_thread_idx < n_linesearch;
        bool candidate_armijo = false;
        bool candidate_wolfe = false;
        if (is_candidate_thread)
        {
          ScalarType alpha = search_magnitudes[local_thread_idx];
          if constexpr (Backtracking) alpha *= beta[batch];
          const ScalarType candidate_cost =
              search_cost[batch * n_linesearch + local_thread_idx];
          candidate_armijo = candidate_cost <=
              exploration_cost[batch] + armijo_threshold_c_1 * alpha *
                  baseline_directional_derivative;
          if constexpr (Backtracking) {
            // Equal rounded costs are not progress. Nonfinite trials must not
            // enter quasi-Newton history, even if an infinity comparison passes.
            candidate_armijo = candidate_armijo && isfinite(candidate_cost) &&
                isfinite(directional_derivative[local_thread_idx]) &&
                isfinite(baseline_directional_derivative) &&
                baseline_directional_derivative < ScalarType(0) &&
                candidate_cost < exploration_cost[batch];
          }
          if constexpr (strong_wolfe) {
            candidate_wolfe = candidate_armijo &&
                fabs(directional_derivative[local_thread_idx]) <=
                    curvature_threshold_c_2 * fabs(baseline_directional_derivative);
          } else {
            candidate_wolfe = candidate_armijo &&
                directional_derivative[local_thread_idx] >=
                    curvature_threshold_c_2 * baseline_directional_derivative;
          }
        }
        const unsigned active_mask = __activemask();
        const unsigned armijo_mask =
            __ballot_sync(active_mask, is_candidate_thread && candidate_armijo);
        const unsigned wolfe_mask =
            __ballot_sync(active_mask, is_candidate_thread && candidate_wolfe);

        if (local_thread_idx == 0)
        {
          constexpr int STATUS_BASELINE = 0;
          constexpr int STATUS_WOLFE = 1;
          constexpr int STATUS_ARMIJO = 2;
          constexpr int STATUS_FALLBACK = 4;
          const int scale_offset = batch * n_linesearch;
          const ScalarType baseline_cost_value = exploration_cost[batch];

          const bool has_wolfe = wolfe_mask != 0;
          const bool has_armijo = armijo_mask != 0;
          const int wolfe_id = has_wolfe ? 31 - __clz(wolfe_mask) : -1;
          const int armijo_id = has_armijo ? 31 - __clz(armijo_mask) : -1;

          selected_id = has_wolfe ? wolfe_id : -1;
          if constexpr (strong_wolfe) {
            exploration_id = selected_id;
          } else if (has_wolfe) {
            exploration_id = wolfe_id;
          } else if (has_armijo) {
            exploration_id = armijo_id;
          } else if constexpr (approx_wolfe && !Backtracking) {
            exploration_id = n_linesearch > 1 ? 1 : 0;
          }
          // A successful Armijo retry is a usable solution even if it does
          // not satisfy Wolfe curvature. L-BFGS separately validates s^T y.
          if constexpr (Backtracking) selected_id = exploration_id;
          const int status = Backtracking && exploration_id < 0 ? STATUS_BACKTRACK :
              has_wolfe ? STATUS_WOLFE :
              (has_armijo ? STATUS_ARMIJO :
               (exploration_id >= 0 ? STATUS_FALLBACK : STATUS_BASELINE));
          // This buffer aliases the iteration-state input during CUDA Graph
          // replay.  The graph input copy initializes it to false, so every
          // line-search mode must explicitly publish the current result.
          curvature_update_valid[batch] = exploration_id >= 0;
          if constexpr (Backtracking) {
            beta[batch] = exploration_id < 0 ?
                beta[batch] * BACKTRACK_FACTOR : ScalarType(1);
          }

          idx_shared_selected = selected_id + batch * n_linesearch;
          idx_shared_exploration = exploration_id + batch * n_linesearch;
          line_search_index[batch] = exploration_id;
          line_search_status[batch] = status;

          const ScalarType exploration_cost_value = exploration_id >= 0 ?
              search_cost[scale_offset + exploration_id] : baseline_cost_value;
          const ScalarType selected_cost_value = selected_id >= 0 ?
              search_cost[scale_offset + selected_id] : baseline_cost_value;
          exploration_cost[batch] = exploration_cost_value;
          selected_cost[batch] = selected_cost_value;

          int local_current_iteration = current_iteration[batch] + 1;
          int local_best_iteration = best_iteration[batch];
          bool converged = false;
          check_best_convergence(
              best_cost[batch], selected_cost_value, cost_delta_threshold,
              cost_relative_threshold, local_current_iteration,
              convergence_iteration, local_best_iteration,
              update_best_shared, converged);
          current_iteration[batch] = local_current_iteration;
          best_iteration[batch] = local_best_iteration;
          converged_global[batch] = converged;
          if (update_best_shared) {
            best_cost[batch] = selected_cost_value;
          }
        }

        __syncthreads();

        const int local_exploration_id = idx_shared_exploration - batch * n_linesearch;
        const int local_selected_id = idx_shared_selected - batch * n_linesearch;
        for (int variable_idx = local_thread_idx; variable_idx < opt_dim;
             variable_idx += blockDim.x) {
          const IndexType output_offset = batch_step_offset + variable_idx;
          if (local_selected_id >= 0) {
            // A selected Wolfe candidate is also the exploration candidate.
            const IndexType candidate_offset = batch_search_offset +
                static_cast<IndexType>(local_selected_id) * opt_dim + variable_idx;
            const ScalarType action_value = search_action[candidate_offset];
            const ScalarType gradient_value = search_gradient[candidate_offset];
            exploration_action[output_offset] = action_value;
            exploration_gradient[output_offset] = gradient_value;
            selected_action[output_offset] = action_value;
            selected_gradient[output_offset] = gradient_value;
            if (update_best_shared) {
              best_action[output_offset] = action_value;
            }
          } else {
            // The cached baseline must be read before exploration is overwritten.
            const ScalarType baseline_action_value = exploration_action[output_offset];
            const ScalarType baseline_gradient_value = exploration_gradient[output_offset];
            selected_action[output_offset] = baseline_action_value;
            selected_gradient[output_offset] = baseline_gradient_value;
            if (update_best_shared) {
              best_action[output_offset] = baseline_action_value;
            }
            if (local_exploration_id >= 0) {
              const IndexType exploration_offset = batch_search_offset +
                  static_cast<IndexType>(local_exploration_id) * opt_dim + variable_idx;
              exploration_action[output_offset] = search_action[exploration_offset];
              exploration_gradient[output_offset] = search_gradient[exploration_offset];
            }
          }
        }
        if (local_thread_idx < n_linesearch) {
          exploration_idx[batch * n_linesearch + local_thread_idx] = local_exploration_id;
          selected_idx[batch * n_linesearch + local_thread_idx] = local_selected_id;
        }
      }

}
}
}
