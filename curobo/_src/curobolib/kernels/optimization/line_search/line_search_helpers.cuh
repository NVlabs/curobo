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
namespace line_search{


    __device__ __forceinline__ void check_best_convergence(
        const float best_cost,
        const float current_cost,
        const float cost_delta_threshold,
        const float cost_relative_threshold,
        const int current_iteration,
        const int convergence_iteration,
        int &best_iteration,
        bool &update_best,
        bool &converged){

        // check if current cost is better than best cost
        float cost_delta = best_cost - current_cost;
        float cost_relative = cost_delta / (best_cost + curobo::common::fp32Precision);

        update_best = cost_delta > cost_delta_threshold && cost_relative > cost_relative_threshold;

        // if update_best is true, update best_iteration to current_iteration (assume current_iteration was
        // updated before this function call)

        // check if convergence iteration is reached
        best_iteration = update_best ? current_iteration : best_iteration;
        converged = best_iteration + convergence_iteration < current_iteration;



      }

      __device__ __forceinline__ void get_linesearch_idx(int id1, int id, bool approx_wolfe, bool strong_wolfe,
      int &exploration_id, int &selected_id)
      {

        id1 = (id1 == curobo::common::warpSize) ? 0 : id1;

        id = (id == curobo::common::warpSize) ? 0 : id;
        // If strong_wolfe, just return id
        // Otherwise, use id1 if id is 0
        selected_id = strong_wolfe ? id : ((id == 0) ? id1 : id);

        // For approx_wolfe (and not strong_wolfe), ensure id is at least 1
        exploration_id = (approx_wolfe && !strong_wolfe && selected_id == 0) ? 1 : selected_id;


      }

      /**
       * @brief Computes exploration and selected indices from Wolfe condition ballot results
       *
       * This function encapsulates the common pattern of using ballot operations to find
       * the last occurrence of satisfied Wolfe conditions and determining the appropriate
       * exploration and selected indices based on the line search strategy.
       *
       * @param wolfe_1 Armijo condition satisfaction for current thread
       * @param wolfe Combined Wolfe conditions satisfaction for current thread
       * @param condition Whether current thread participates in ballot
       * @param approx_wolfe Whether using approximate Wolfe conditions
       * @param strong_wolfe Whether using strong Wolfe conditions
       * @param exploration_id Output parameter for exploration index
       * @param selected_id Output parameter for selected index
       */
      __device__ __forceinline__ void compute_wolfe_indices(
          const bool wolfe_1,
          const bool wolfe,
          const bool condition,
          const bool approx_wolfe,
          const bool strong_wolfe,
          int& exploration_id,
          int& selected_id)
      {
          // Ballot sync to collect Wolfe condition results across warp
          unsigned msk1 = __ballot_sync(curobo::common::fullMask, wolfe_1 & condition);
          unsigned msk  = __ballot_sync(curobo::common::fullMask, wolfe & condition);

          // Reverse bit order to find last occurrence
          unsigned msk1_brev = __brev(msk1);
          unsigned msk_brev  = __brev(msk);

          // Find position of least significant bit (last occurrence after reversal)
          int id1 = curobo::common::warpSize - __ffs(msk1_brev);
          int id  = curobo::common::warpSize - __ffs(msk_brev);

          // Determine final indices based on line search strategy
          get_linesearch_idx(id1, id, approx_wolfe, strong_wolfe, exploration_id, selected_id);
      }

      /**
       * @brief Updates costs and checks convergence for line search optimization
       *
       * This function handles the common pattern of updating exploration and selected costs,
       * checking convergence criteria, and updating iteration counters. It's designed to be
       * called by thread 0 only.
       *
       * @tparam ScalarType Floating point type (float/double)
       * @param batch Current batch index
       * @param exploration_id Index for exploration step
       * @param selected_id Index for selected step
       * @param n_linesearch Number of line search points
       * @param convergence_iteration Convergence check interval
       * @param cost_delta_threshold Minimum cost improvement threshold
       * @param cost_relative_threshold Minimum relative cost improvement threshold
       * @param search_cost Input cost array
       * @param exploration_cost Output exploration cost array
       * @param selected_cost Output selected cost array
       * @param best_cost Input/output best cost array
       * @param best_iteration Input/output best iteration array
       * @param current_iteration Input/output current iteration array
       * @param converged_global Output convergence status array
       * @param update_best_shared Output shared memory flag for best update
       * @param converged_shared Output shared memory flag for convergence
       */
      template<typename ScalarType>
      __device__ __forceinline__ void update_costs_and_convergence(
          const int batch,
          const int exploration_id,
          const int selected_id,
          const int n_linesearch,
          const int convergence_iteration,
          const float cost_delta_threshold,
          const float cost_relative_threshold,
          const ScalarType* search_cost,
          ScalarType* exploration_cost,
          ScalarType* selected_cost,
          ScalarType* best_cost,
          int16_t* best_iteration,
          int16_t* current_iteration,
          uint8_t* converged_global,
          bool& update_best_shared,
          bool& converged_shared)
      {
          const int idx_exploration = exploration_id + batch * n_linesearch;
          const int idx_selected = selected_id + batch * n_linesearch;

          // Update exploration and selected costs
          exploration_cost[batch] = search_cost[idx_exploration];
          float selected_cost_val = search_cost[idx_selected];
          selected_cost[batch] = selected_cost_val;

          // Get current best cost and iteration counters
          float current_best_cost = best_cost[batch];
          int local_current_iteration = current_iteration[batch];
          local_current_iteration++;
          int local_best_iteration = best_iteration[batch];

          // Check convergence
          bool update_best = false;
          bool converged = false;
          check_best_convergence(current_best_cost, selected_cost_val,
                                cost_delta_threshold, cost_relative_threshold,
                                local_current_iteration, convergence_iteration,
                                local_best_iteration, update_best, converged);

          // Update shared memory flags
          update_best_shared = update_best;
          converged_shared = converged;

          // Update global state
          converged_global[batch] = converged;
          best_iteration[batch] = local_best_iteration;
          current_iteration[batch] = local_current_iteration;

          if (update_best) {
              best_cost[batch] = selected_cost_val;
          }
      }

      /**
       * @brief Copies action and gradient results for exploration and selected steps
       *
       * This function handles the common pattern of copying action and gradient data
       * from search arrays to output arrays, and optionally updating best action.
       *
       * @tparam ScalarType Floating point type (float/double)
       * @param batch Current batch index
       * @param thread_idx Current thread index within batch
       * @param exploration_idx Global exploration index
       * @param selected_idx Global selected index
       * @param opt_dim Optimization dimension size
       * @param n_linesearch Number of line search points
       * @param update_best_shared Whether to update best action
       * @param search_action Input action array
       * @param search_gradient Input gradient array
       * @param exploration_action Output exploration action array
       * @param exploration_gradient Output exploration gradient array
       * @param selected_action Output selected action array
       * @param selected_gradient Output selected gradient array
       * @param best_action Input/output best action array
       * @param exploration_idx_out Output exploration index array
       * @param selected_idx_out Output selected index array
       */
      template<typename ScalarType>
      __device__ __forceinline__ void copy_action_gradient_results(
          const int batch,
          const int thread_idx,
          const int exploration_idx,
          const int selected_idx,
          const int opt_dim,
          const int n_linesearch,
          const bool update_best_shared,
          const ScalarType* search_action,
          const ScalarType* search_gradient,
          ScalarType* exploration_action,
          ScalarType* exploration_gradient,
          ScalarType* selected_action,
          ScalarType* selected_gradient,
          ScalarType* best_action,
          int32_t* exploration_idx_out,
          int32_t* selected_idx_out)
      {
          // Copy exploration data
          exploration_action[batch * opt_dim + thread_idx] =
              search_action[exploration_idx * opt_dim + thread_idx];
          exploration_gradient[batch * opt_dim + thread_idx] =
              search_gradient[exploration_idx * opt_dim + thread_idx];

          // Copy selected data
          float selected_action_val = search_action[selected_idx * opt_dim + thread_idx];
          float selected_gradient_val = search_gradient[selected_idx * opt_dim + thread_idx];
          selected_action[batch * opt_dim + thread_idx] = selected_action_val;
          selected_gradient[batch * opt_dim + thread_idx] = selected_gradient_val;

          // Update best action if needed
          if (update_best_shared) {
              best_action[batch * opt_dim + thread_idx] = selected_action_val;
          }

          // Update index arrays (only for threads < n_linesearch)
          if (thread_idx < n_linesearch) {
            int local_exploration_idx = exploration_idx - batch * n_linesearch;
            int local_selected_idx = selected_idx - batch * n_linesearch;
            exploration_idx_out[batch * n_linesearch + thread_idx] =
                local_exploration_idx;
            selected_idx_out[batch * n_linesearch + thread_idx] =
                local_selected_idx;
          }
      }

      // Launched with l2 threads/block and batchsize blocks

      /**
      * Evaluates Wolfe conditions for line search optimization
      *
      * @param condition Whether this thread should evaluate the conditions
      * @param alpha_list_elem Step size for this thread
      * @param c_0 Function value at initial point
      * @param c_val Function value at current point
      * @param g_step_val Directional derivative at current point
      * @param g_step_0 Directional derivative at initial point
      * @param c_1 Parameter for Armijo condition
      * @param c_2 Parameter for curvature condition
      * @param strong_wolfe Whether to use strong Wolfe conditions
      * @param approx_wolfe Whether to use approximate Wolfe conditions
      * @param wolfe Output parameter for combined Wolfe condition
      * @param wolfe_1 Output parameter for Armijo condition
      * @param wolfe_2 Output parameter for curvature condition
      */
      template<typename ScalarType>
      __device__ __forceinline__
      void evaluate_wolfe_conditions(
          const bool condition,
          const ScalarType alpha_list_elem,
          const ScalarType c_0,
          const ScalarType c_val,
          const ScalarType g_step_val,
          const ScalarType g_step_0,
          const ScalarType c_1,
          const ScalarType c_2,
          const bool strong_wolfe,
          const bool approx_wolfe,
          bool& wolfe,
          bool& wolfe_1,
          bool& wolfe_2
      ) {
          // Initialize return values
          wolfe = false;
          wolfe_1 = false;
          wolfe_2 = false;

          if (condition) {
              // Precompute common values
              const ScalarType c1_alpha_g0 = c_1 * alpha_list_elem * g_step_0;
              const ScalarType c2_g0 = c_2 * g_step_0;
              const ScalarType c2_abs_g0 = c_2 * fabs(g_step_0);

              // Evaluate Armijo condition (sufficient decrease)
              wolfe_1 = c_val <= (c_0 + c1_alpha_g0);

              // Evaluate curvature condition
              const bool strong_wolfe_condition = fabs(g_step_val) <= c2_abs_g0;
              const bool weak_wolfe_condition = g_step_val >= c2_g0;
              wolfe_2 = (strong_wolfe) ? strong_wolfe_condition : weak_wolfe_condition;

              // Combine conditions
              wolfe = wolfe_1 & wolfe_2;
          }
      }


}
}
}