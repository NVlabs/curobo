/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "common/curobo_constants.h"
#include "common/math.cuh"
namespace curobo{
    namespace trajectory{
        namespace bspline{

    template<int Degree>
    __inline__ __host__ __device__ static constexpr int get_spline_support_size()
    {
        return Degree + 1;
    }

    template<int Degree>
    __inline__ __host__ __device__ int get_total_knots(const int n_knots)
    {
        return n_knots + get_spline_support_size<Degree>();
    }

    template<int Degree>
    __host__ __device__ __forceinline__ bool is_start_boundary(int knot_idx) {
        return knot_idx < get_spline_support_size<Degree>();
    }

    template<int Degree>
    __host__ __device__ __forceinline__ bool is_goal_boundary_replicate(int knot_idx, int n_knots) {
        return knot_idx >  n_knots;
    }

    template<int Degree>
    __host__ __device__ __forceinline__ bool is_goal_boundary_implicit(int knot_idx, int n_knots) {
        return knot_idx > n_knots - 1;
    }


    template<int Degree>
    __host__ __device__ __forceinline__ bool needs_boundary_update(int knot_idx, int n_knots, bool use_implicit_goal) {
        return is_start_boundary<Degree>(knot_idx) ||
                (use_implicit_goal ? is_goal_boundary_implicit<Degree>(knot_idx, n_knots) : is_goal_boundary_replicate<Degree>(knot_idx, n_knots));
    }


    struct BSplineBackwardLayout {
        int interpolation_steps;
        int knots_per_warp;
        int warps_for_n_knots;
        int threads_for_n_knots;
        int padded_horizon;
        int n_knots;
        int padded_n_knots;
        int horizon;
        int dof;
    };

    template<int Degree>
    __host__ __device__ static BSplineBackwardLayout compute_bspline_backward_layout(
        int horizon, int dof, int n_knots) {
        // Degree-specific calculations
        BSplineBackwardLayout layout;
        const int total_knots = get_total_knots<Degree>(n_knots);
        layout.interpolation_steps = (total_knots > 0) ? horizon / total_knots : 0;

        layout.knots_per_warp = (layout.interpolation_steps > 0)
          ? curobo::common::warpSize / layout.interpolation_steps
          : curobo::common::warpSize;

        layout.warps_for_n_knots = curobo::common::ceil_div(n_knots, layout.knots_per_warp) ;
        layout.threads_for_n_knots = layout.warps_for_n_knots * curobo::common::warpSize;
        layout.padded_horizon = horizon + 1;

        layout.n_knots = n_knots;
        layout.padded_n_knots = get_total_knots<Degree>(n_knots);
        layout.horizon = horizon;
        layout.dof = dof;

        return layout;
    }

    /**
     * @brief Structure to organize thread indices and common calculations for B-spline gradient computation.
     *
     * This structure encapsulates all the thread-specific indices and computed values needed for
     * B-spline gradient computation in CUDA kernels. It centralizes the complex thread index
     * decomposition and parameter calculations used throughout the gradient computation pipeline.
     *
     * @note This structure is populated by compute_thread_indices() and used throughout
     *       gradient computation functions to maintain consistent indexing and calculations.
     */
     struct BSplineBackwardThreadInfo {
        int tid;                    ///< Global thread ID in the CUDA kernel
        int b_idx;                  ///< Batch index for this thread
        int d_idx;                  ///< DOF (dimension) index for this thread
        int knot_idx;               ///< Control point (knot) index being processed
        int interpolation_idx;      ///< Index within interpolation steps for current knot
        int h_idx;                  ///< Time step index in trajectory horizon
        int dt_offset;              ///< Index into time step configuration array
        bool use_goal;              ///< Whether to apply goal state constraints
        float t_mod;                ///< Normalized time parameter within knot interval [0,1)
        float knot_dt;              ///< Time interval between consecutive knots
        BSplineBackwardLayout layout;
    };
      /**
     * @brief Computes thread indices and setup calculations for B-spline gradient computation.
     *
     * This function performs the complex thread index decomposition required for efficient
     * B-spline gradient computation in CUDA kernels. It maps a global thread ID to specific
     * batch, DOF, and knot indices, while computing derived parameters needed for gradient
     * calculation.
     *
     * The function implements a hierarchical thread organization:
     * 1. Threads are first organized by batch and DOF
     * 2. Within each (batch, DOF) group, threads are distributed across knots
     * 3. Multiple threads per knot handle different interpolation steps for warp-level reduction
     * 4. Thread indices are computed to enable coalesced memory access patterns
     *
     * @tparam Degree B-spline degree, affects interpolation step calculations
     *
     * @param tid Global thread ID from CUDA kernel
     * @param batch_size Number of trajectories being processed
     * @param dof Degrees of freedom (number of joints/dimensions)
     * @param horizon Number of time steps in trajectory
     * @param n_knots Number of control points in B-spline
     * @param dt_idx Array mapping batches to time step configurations
     *               Shape: [batch_size]
     * @param use_implicit_goal_state Array indicating goal constraint usage
     *                                Shape: [num_dt_configs]
     * @param traj_dt Array of time step values
     *                Shape: [num_dt_configs]
     *
     * @return BSplineBackwardThreadInfo structure with all computed indices and parameters
     *
     * @note This function is degree-specific and uses different interpolation divisors
     *       based on the B-spline degree for optimal performance.
     * @note The computed indices enable efficient warp-level parallelism for gradient
     *       accumulation across multiple interpolation points.
     */
     template<int Degree>
    __device__ BSplineBackwardThreadInfo compute_bspline_backward_thread_info(
         int tid, int batch_size, int dof, int horizon, int n_knots,
         const int32_t *dt_idx, const uint8_t *use_implicit_goal_state,
         const float *traj_dt)
     {
         BSplineBackwardThreadInfo thread_info;

         thread_info.layout = compute_bspline_backward_layout<Degree>(horizon, dof, n_knots);


         thread_info.tid = tid;


         thread_info.b_idx = tid / (dof * thread_info.layout.threads_for_n_knots);
         thread_info.d_idx = ((tid - thread_info.b_idx * dof * thread_info.layout.threads_for_n_knots) / thread_info.layout.threads_for_n_knots);

         const int batch_thread_idx = tid - (thread_info.b_idx * dof * thread_info.layout.threads_for_n_knots + thread_info.d_idx * thread_info.layout.threads_for_n_knots);
         const int knots_offset = (batch_thread_idx / curobo::common::warpSize) * thread_info.layout.knots_per_warp;
         const int warp_thread_idx = batch_thread_idx - (knots_offset * thread_info.layout.interpolation_steps);

         thread_info.interpolation_idx = warp_thread_idx / thread_info.layout.knots_per_warp;
         const int local_knot_idx = warp_thread_idx % thread_info.layout.knots_per_warp;
         thread_info.knot_idx = local_knot_idx + knots_offset;

         const int offset_knot_idx = thread_info.knot_idx + Degree;
         thread_info.h_idx = offset_knot_idx * thread_info.layout.interpolation_steps + thread_info.interpolation_idx;

         // avoid out of bounds access by clamping the batch index, this gets filtered out later.
         const int safe_b_idx = min(thread_info.b_idx, batch_size - 1);
         thread_info.dt_offset = dt_idx[safe_b_idx];
         thread_info.use_goal = use_implicit_goal_state[thread_info.dt_offset];
         float interpolated_dt = traj_dt[thread_info.dt_offset];
         thread_info.knot_dt = interpolated_dt * thread_info.layout.interpolation_steps;

         thread_info.t_mod = (float(thread_info.h_idx) / float(thread_info.layout.interpolation_steps)) - int(thread_info.h_idx / thread_info.layout.interpolation_steps);


         return thread_info;
     }

    }
  }
}