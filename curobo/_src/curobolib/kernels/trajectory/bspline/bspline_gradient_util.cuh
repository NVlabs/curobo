/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "bspline_common.cuh"
#include "bspline_boundary_constraint.cuh"
#include "common/curobo_constants.h"

namespace curobo{
    namespace trajectory{
      namespace bspline{


    /**
     * @brief Structure to organize gradient arrays for B-spline derivative computation.
     *
     * This structure holds gradient values for position, velocity, acceleration, and jerk
     * across the support of a B-spline basis function. The support size equals (Degree + 1),
     * representing the number of consecutive time steps that influence a single control point.
     *
     * @tparam Degree B-spline degree, which determines the support size
     *
     * @note Each array element corresponds to a different time step within the B-spline support,
     *       ordered from earliest to latest influence on the current control point.
     */
    template<int Degree>
    struct GradientArrays {
        static constexpr int splineSupportSize = get_spline_support_size<Degree>();  ///< Number of time steps in B-spline support
        float g_pos[splineSupportSize];   ///< Gradient values w.r.t. position
        float g_vel[splineSupportSize];   ///< Gradient values w.r.t. velocity
        float g_acc[splineSupportSize];   ///< Gradient values w.r.t. acceleration
        float g_jerk[splineSupportSize];  ///< Gradient values w.r.t. jerk

        /**
         * @brief Initialize all gradient arrays to zero.
         *
         * This device function initializes all gradient values to 0.0f, preparing
         * the structure for gradient accumulation operations.
         */
        __device__ void initialize() {
            #pragma unroll
            for (int i = 0; i < splineSupportSize; i++) {
                g_pos[i] = 0.0f;
                g_vel[i] = 0.0f;
                g_acc[i] = 0.0f;
                g_jerk[i] = 0.0f;
            }
        }
    };


    /**
    * @brief Performs warp-level reduction across B-spline knots and computes interpolation average.
    *
    * This function implements a specialized warp reduction that operates on B-spline knot values
    * within a CUDA warp. It uses shuffle operations to efficiently sum values across different
    * knots and then computes the average over the specified number of interpolation steps.
    *
    * The reduction follows a tree-like pattern:
    * 1. Starts with an initial shift based on knots_per_warp * interpolation_steps
    * 2. Uses __shfl_down_sync to gather values from other threads in the warp
    * 3. Continues reduction until offset falls below knots_per_warp threshold
    * 4. Divides final sum by interpolation_steps to compute average
    *
    * @param val The input value from the current thread to be reduced
    * @param knots_per_warp Number of B-spline knots processed per warp (must be > 0)
    * @param interpolation_steps Number of interpolation steps for averaging (must be > 0)
    *
    * @return The averaged value after warp-level reduction and interpolation averaging
    *
    * @note This function must be called by all threads in a warp for correct synchronization.
    * @note The function assumes a 32-thread warp size typical of modern NVIDIA GPUs.
    * @note Input parameters must ensure that knots_per_warp * interpolation_steps <= 32.
    *
    * @warning This is a CUDA device function and can only be called from device code.
    * @warning All threads in the warp must participate in the call for proper execution.
    *
    * @see __shfl_down_sync() for details on the underlying CUDA shuffle operation
    * @see __ballot_sync() for warp synchronization mechanism used
    */
    __inline__ __device__ float warp_interpolation_reduce_sum(float val,
        const int knots_per_warp, const int interpolation_steps)
        {


          unsigned int mask = __ballot_sync(0xFFFFFFFF, true);

          int shift = (knots_per_warp * interpolation_steps + 1) / 2; // Ensure correct rounding
          const int lane_idx = threadIdx.x % 32;


          for (int offset = shift; offset >= knots_per_warp; offset = (offset + 1) / 2) {
            // Ensure correct rounding in the loop
            float other_val = __shfl_down_sync(mask, val, offset);
            // only add if it's with the correct lane index:

            bool condition = (lane_idx + offset) < 32; // This will always be true?
            float to_sum_value = condition ? other_val : 0;
            val += to_sum_value;

          }
          return val;
        }



    /**
     * @brief Loads gradient values from global memory into structured arrays for B-spline computation.
     *
     * This function extracts gradient values from the global gradient arrays and organizes them
     * into the GradientArrays structure for efficient B-spline gradient computation. It loads
     * gradients from the B-spline support window (Degree + 1 consecutive time steps) that
     * influence the current control point.
     *
     * The function loads gradients in forward temporal order from past to present to match
     * the forward interpolation knot loading pattern. For each time step in the support window, it loads position,
     * velocity, acceleration, and jerk gradients simultaneously.
     *
     * @tparam Degree B-spline degree, determines support window size (Degree + 1)
     * @tparam ScalarType Floating point type (float or double)
     *
     * @param gradients Output structure to store loaded gradient values
     * @param indices Thread indices structure containing computed addressing information
     * @param grad_position Input gradient array w.r.t. trajectory positions
     *                      Shape: [batch_size, horizon, dof]
     * @param grad_velocity Input gradient array w.r.t. trajectory velocities
     *                      Shape: [batch_size, horizon, dof]
     * @param grad_acceleration Input gradient array w.r.t. trajectory accelerations
     *                          Shape: [batch_size, horizon, dof]
     * @param grad_jerk Input gradient array w.r.t. trajectory jerks
     *                  Shape: [batch_size, horizon, dof]
     *
     * @note The function automatically handles boundary conditions by only loading gradients
     *       for valid time steps within the trajectory horizon.
     * @note Gradient loading is optimized for coalesced memory access when called by
     *       threads processing consecutive DOF indices.
     */
    template<int Degree, typename ScalarType>
    __device__ void load_gradients(
        GradientArrays<Degree> &gradients,
        const BSplineBackwardThreadInfo &thread_info,
        const ScalarType *grad_position,
        const ScalarType *grad_velocity,
        const ScalarType *grad_acceleration,
        const ScalarType *grad_jerk)
    {
        constexpr int splineSupportSize = get_spline_support_size<Degree>();
        gradients.initialize();

        // Load gradients based on support size, including virtual knot timesteps
        const int extended_horizon = (get_total_knots<Degree>(thread_info.layout.n_knots)) * thread_info.layout.interpolation_steps;
        const bool implicit_goal_boundary = thread_info.use_goal && thread_info.knot_idx >= thread_info.layout.n_knots - 1;//is_goal_boundary_implicit<Degree>(thread_info.knot_idx, thread_info.layout.n_knots);
        const bool replicate_at_last_knot = !thread_info.use_goal && thread_info.knot_idx == thread_info.layout.n_knots - 1;
        const int addr_offset = thread_info.b_idx * thread_info.layout.padded_horizon * thread_info.layout.dof + thread_info.d_idx;
        const int h_offset = (thread_info.knot_idx + 1) * thread_info.layout.interpolation_steps + thread_info.interpolation_idx;
        #pragma unroll
        for (int i = 0; i < splineSupportSize; ++i) {
            int offset_h_idx = h_offset + (i * thread_info.layout.interpolation_steps);



            int addr = addr_offset + offset_h_idx * thread_info.layout.dof;

            // Load gradients for both regular and virtual knot timesteps, but skip if should be zero
            if (offset_h_idx < extended_horizon && !implicit_goal_boundary) {
                gradients.g_pos[i] = grad_position[addr];
                gradients.g_vel[i] = grad_velocity[addr];
                gradients.g_acc[i] = grad_acceleration[addr];
                gradients.g_jerk[i] = grad_jerk[addr];
            }
        }
        if (replicate_at_last_knot) {
            // only called when knot_idx = n_knots - 1

            // Add gradients for additional replication (highlighted with *)

            // knot_idx | local_support                                 | loop_size | target_index | source_index
            // 17       | knots[13], knots[14], knots[15], knots[15]*   | 1         | 0            | 1
            // 18       | knots[14], knots[15], knots[15]*, knots[15]*  | 2         | 0, 1         | 2
            // 19       | knots[15], knots[15]*, knots[15]*, knots[15]* | 3         | 0, 1, 2      | 3

            // If interpolation_idx == 0, there is knot_idx = 20

            #pragma unroll
            for (int i = 1; i < splineSupportSize; ++i) {

                #pragma unroll
                for (int x = 0; x < i; ++x) {
                    gradients.g_pos[x] += gradients.g_pos[i];
                    gradients.g_vel[x] += gradients.g_vel[i];
                    gradients.g_acc[x] += gradients.g_acc[i];
                    gradients.g_jerk[x] += gradients.g_jerk[i];
                }

            }

            if (thread_info.interpolation_idx == 0) {
                //int offset_h_idx = (thread_info.layout.n_knots + 1) * thread_info.layout.interpolation_steps; // padded horizon

                int addr = addr_offset + thread_info.layout.horizon * thread_info.layout.dof;

                float temp_grad_pos = grad_position[addr];
                float temp_grad_vel = grad_velocity[addr];
                float temp_grad_acc = grad_acceleration[addr];
                float temp_grad_jerk = grad_jerk[addr];

                #pragma unroll
                for (int x = 0; x < splineSupportSize; ++x) {
                    gradients.g_pos[x] += temp_grad_pos;
                    //gradients.g_vel[x] += temp_grad_vel;
                    //gradients.g_acc[x] += temp_grad_acc;
                    //gradients.g_jerk[x] += temp_grad_jerk;
                }

            }




        }
    }




  }
}
}
