/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bspline_boundary_constraint.cuh"
#include "bspline_context.cuh"
#include "bspline_common.cuh"
namespace curobo{
    namespace trajectory{
        namespace bspline{






    /**
     * @brief Core B-spline trajectory interpolation function that computes position, velocity,
     *        acceleration, and jerk at a single trajectory point.
     *
     * This device function performs the core B-spline interpolation computation for a single
     * (batch, time_step, dof) combination. It handles boundary constraints, applies start/goal
     * state conditions, and computes all derivatives up to jerk using basis functions.
     *
     * The algorithm:
     * 1. Determines which knot interval the current time step falls into
     * 2. Extracts the relevant control points (knots) for the B-spline segment
     * 3. Applies boundary constraints from start/goal states when near trajectory endpoints
     * 4. Uses basis functions to compute interpolated values and derivatives
     * 5. Writes results to output arrays with proper indexing
     *
     * This method uses the start state to compute the first degree knots (degree).
     * To bring the trajectory to rest, the last knot point is repeated degree times.
     * If an implicit goal state is set to True, the last knot point from u_position is ignored and
     * fixed knots are calculated that enforce reaching goal position, velocity, acceleration, and jerk
     * at the end of the trajectory.
     * Given n_knots, we pad it with support_size knots to the left and right.
     * So total_knots = n_knots + 2 * (degree).
     * horizon = total_knots * interpolation_steps.
     * padded_horizon = horizon + 1.
     *
     * @tparam ScalarType Floating point type (float or double)
     * @tparam Degree B-spline degree (typically 3, 5, or 7)
     * @tparam BasisImpl Backend implementation for basis function computation
     *
     * @param out_position_mem Output array for trajectory positions
     *                         Shape: [batch_size, output_horizon, dof]
     * @param out_velocity_mem Output array for trajectory velocities
     *                         Shape: [batch_size, output_horizon, dof]
     * @param out_acceleration_mem Output array for trajectory accelerations
     *                             Shape: [batch_size, output_horizon, dof]
     * @param out_jerk_mem Output array for trajectory jerks
     *                     Shape: [batch_size, output_horizon, dof]
     * @param out_dt Output array for time steps (written once per batch)
     *               Shape: [batch_size]
     * @param u_position Control points (knots) for B-spline interpolation
     *                   Shape: [batch_size, n_knots, dof]
     * @param start_position Initial position states (shared via b_offset indexing)
     *                       Shape: [num_start_states, dof]
     * @param start_velocity Initial velocity states (shared via b_offset indexing)
     *                       Shape: [num_start_states, dof]
     * @param start_acceleration Initial acceleration states (shared via b_offset indexing)
     *                           Shape: [num_start_states, dof]
     * @param start_jerk Initial jerk states (shared via b_offset indexing)
     *                   Shape: [num_start_states, dof]
     * @param goal_position Target position states (shared via goal_offset indexing)
     *                      Shape: [num_goal_states, dof]
     * @param goal_velocity Target velocity states (shared via goal_offset indexing)
     *                      Shape: [num_goal_states, dof]
     * @param goal_acceleration Target acceleration states (shared via goal_offset indexing)
     *                          Shape: [num_goal_states, dof]
     * @param goal_jerk Target jerk states (shared via goal_offset indexing)
     *                  Shape: [num_goal_states, dof]
     * @param interpolated_dt Time step between trajectory points
     * @param use_implicit_goal_state Array indicating whether to apply goal state constraints
     *                                Shape: [num_goal_states]
     * @param batch_size Number of trajectories in the batch
     * @param padded_horizon Total number of interpolation points including padding
     * @param max_out_tsteps Maximum output time steps for proper indexing
     * @param dof Degrees of freedom (number of joints/dimensions)
     * @param b_idx Current batch index being processed
     * @param h_idx Current time step index being processed
     * @param d_idx Current DOF index being processed
     * @param b_offset Index into start state arrays for this batch (from start_idx[b_idx])
     * @param goal_offset Index into goal state arrays for this batch (from goal_idx[b_idx])
     * @param n_knots Number of control points in B-spline representation
     *
     * @note This function is called by CUDA kernels for each thread processing a single
     *       (batch, time_step, dof) combination. The function handles all boundary cases
     *       and applies appropriate constraints based on the trajectory segment being interpolated.
     */
    template<typename ScalarType, int Degree, BasisBackend BasisImpl = BasisBackend::FACTORED>
    __device__ __forceinline__ void interpolate_bspline_trajectory(ScalarType       *out_position_mem,
                                                    ScalarType       *out_velocity_mem,
                                                    ScalarType       *out_acceleration_mem,
                                                    ScalarType       *out_jerk_mem,
                                                    ScalarType       *out_dt,
                                                    const ScalarType *u_position,
                                                    const ScalarType *start_position,
                                                    const ScalarType *start_velocity,
                                                    const ScalarType *start_acceleration,
                                                    const ScalarType *start_jerk,
                                                    const ScalarType *goal_position,
                                                    const ScalarType *goal_velocity,
                                                    const ScalarType *goal_acceleration,
                                                    const ScalarType *goal_jerk,
                                                    const float     interpolated_dt, // dt value directly
                                                    const uint8_t *use_implicit_goal_state,
                                                    const int       batch_size,
                                                    const int       padded_horizon, // total interpolated points.
                                                    const int       max_out_tsteps,  // for output indexing
                                                    const int       dof,
                                                    const int       b_idx,
                                                    const int       h_idx, // interpolation index
                                                    const int       d_idx,
                                                    const int       b_offset,
                                                    const int       goal_offset,
                                                    const int       n_knots)
    {
      constexpr int splineSupportSize = get_spline_support_size<Degree>();
      const int horizon = padded_horizon - 1;
      const int padded_n_knots = get_total_knots<Degree>(n_knots);
      const int interpolation_steps = (horizon) / padded_n_knots;
      const float knot_dt = fmaxf(interpolated_dt, curobo::common::fp32Precision) * interpolation_steps;

      // read start state:
      float out_pos = 0.0, out_vel = 0.0, out_acc = 0.0, out_jerk = 0.0;
      const bool use_implicit_goal = use_implicit_goal_state[goal_offset];
      const int   b_addrs_action = b_idx * (n_knots) * dof;
      float knots[splineSupportSize];

      int knot_idx = (interpolation_steps > 0) ? int((h_idx) / interpolation_steps) : 0;
      int h_idx_local = h_idx;
      if (knot_idx >= padded_n_knots)
      {
        knot_idx = padded_n_knots - 1; // n_knots + 2*Degree - 1
        h_idx_local = -1;
      }

      // first knots are computed based on start state
      float  t_mod = 0.0;
      BoundaryConstraint constraint(0.0, 0.0, 0.0, 0.0);


      #pragma unroll
      for (int i = 0; i < splineSupportSize; i++)
      {
        knots[i] = 0.0;
      }

      // We offset the knot index by Degree as the first Degree knots are computed based on start state.
      const int start_knot_idx = knot_idx - splineSupportSize;

      // For replicate case:
      // n_knots = 16, Degree = 3, padded_n_knots = n_knots + (support_size) = 19
      // knot_idx | start_knot_idx | local_support
      // 0        | -4             | fixed_knots[0], fixed_knots[1], fixed_knots[2], fixed_knots[3]
      // 1        | -3             | fixed_knots[1], fixed_knots[2], fixed_knots[3], knots[0]
      // 2        | -2             | fixed_knots[2], fixed_knots[3], knots[0], knots[1]
      // 3        | -1             | fixed_knots[3], knots[0], knots[1], knots[2]
      // 4        | 0              | knots[0], knots[1], knots[2], knots[3]
      // 5        | 1              | knots[1], knots[2], knots[3], knots[4]
      // 6        | 2              | knots[2], knots[3], knots[4], knots[5]
      // 7        | 3              | knots[3], knots[4], knots[5], knots[6]
      // 8        | 4              | knots[4], knots[5], knots[6], knots[7]
      // 9        | 5              | knots[5], knots[6], knots[7], knots[8]
      // 10       | 6              | knots[6], knots[7], knots[8], knots[9]
      // 11       | 7              | knots[7], knots[8], knots[9], knots[10]
      // 12       | 8              | knots[8], knots[9], knots[10], knots[11]
      // 13       | 9              | knots[9], knots[10], knots[11], knots[12]
      // 14       | 10             | knots[10], knots[11], knots[12], knots[13]
      // 15       | 11             | knots[11], knots[12], knots[13], knots[14]
      // 16       | 12             | knots[12], knots[13], knots[14], knots[15]
      // 17       | 13             | knots[13], knots[14], knots[15], knots[15]
      // 18       | 14             | knots[14], knots[15], knots[15], knots[15]
      // 19       | 15             | knots[15], knots[15], knots[15], knots[15]


      // For replicate, degree = 4, n_knots = 16
      // knot_idx | start_knot_idx | local_support
      // 0        | -5             | fixed_knots[0], fixed_knots[1], fixed_knots[2], fixed_knots[3], fixed_knots[4]
      // 1        | -4             | fixed_knots[1], fixed_knots[2], fixed_knots[3], fixed_knots[4], knots[0]
      // 2        | -3             | fixed_knots[2], fixed_knots[3], fixed_knots[4], knots[0], knots[1]
      // 3        | -2             | fixed_knots[3], fixed_knots[4], knots[0], knots[1], knots[2]
      // 4        | -1             | fixed_knots[4], knots[0], knots[1], knots[2], knots[3]
      // 5        | 0              | knots[0], knots[1], knots[2], knots[3], knots[4]
      // 6        | 1              | knots[1], knots[2], knots[3], knots[4], knots[5]
      // 7        | 2              | knots[2], knots[3], knots[4], knots[5], knots[6]
      // 8        | 3              | knots[3], knots[4], knots[5], knots[6], knots[7]
      // 9        | 4              | knots[4], knots[5], knots[6], knots[7], knots[8]
      // 10       | 5              | knots[5], knots[6], knots[7], knots[8], knots[9]
      // 11       | 6              | knots[6], knots[7], knots[8], knots[9], knots[10]
      // 12       | 7              | knots[7], knots[8], knots[9], knots[10], knots[11]
      // 13       | 8              | knots[8], knots[9], knots[10], knots[11], knots[12]
      // 14       | 9              | knots[9], knots[10], knots[11], knots[12], knots[13]
      // 15       | 10             | knots[10], knots[11], knots[12], knots[13], knots[14]
      // 16       | 11             | knots[11], knots[12], knots[13], knots[14], knots[15]
      // 17       | 12 (n_knots - Degree)              | knots[12], knots[13], knots[14], knots[15], knots[15]
      // 18       | 13 (n_knots - Degree + 1)          | knots[13], knots[14], knots[15], knots[15], knots[15]
      // 19       | 14 (n_knots - Degree + 2)          | knots[14], knots[15], knots[15], knots[15], knots[15]
      // 20       | 15 (n_knots - Degree + 3)          | knots[15], knots[15], knots[15], knots[15], knots[15]

       // For implicit case: We calculate fixed knots for the goal as well.
      // n_knots = 16, Degree = 3, padded_n_knots = n_knots + 2*(support_size/2) = 19
      // knot_idx | start_knot_idx | local_support
      // 0        | -4             | fixed_knots[0], fixed_knots[1], fixed_knots[2], fixed_knots[3]
      // 1        | -3             | fixed_knots[1], fixed_knots[2], fixed_knots[3], knots[0]
      // 2        | -2             | fixed_knots[2], fixed_knots[3], knots[0], knots[1]
      // 3        | -1             | fixed_knots[3], knots[0], knots[1], knots[2]
      // 4        | 0              | knots[0], knots[1], knots[2], knots[3]
      // 5        | 1              | knots[1], knots[2], knots[3], knots[4]
      // 6        | 2              | knots[2], knots[3], knots[4], knots[5]
      // 7        | 3              | knots[3], knots[4], knots[5], knots[6]
      // 8        | 4              | knots[4], knots[5], knots[6], knots[7]
      // 9        | 5              | knots[5], knots[6], knots[7], knots[8]
      // 10       | 6              | knots[6], knots[7], knots[8], knots[9]
      // 11       | 7              | knots[7], knots[8], knots[9], knots[10]
      // 12       | 8              | knots[8], knots[9], knots[10], knots[11]
      // 13       | 9              | knots[9], knots[10], knots[11], knots[12]
      // 14       | 10             | knots[10], knots[11], knots[12], knots[13]
      // 15       | 11             | knots[11], knots[12], knots[13], knots[14]
      // 16       | 12             | knots[12], knots[13], knots[14], fixed_knots[0] // we skip knots[15] to maintain the same dimension as replicate case.
      // 17       | 13 (n_knots - Degree)              | knots[13], knots[14], fixed_knots[0], fixed_knots[1]
      // 18       | 14 (n_knots - Degree + 1)          | knots[14], fixed_knots[0], fixed_knots[1], fixed_knots[2]
      // 19       | 15 (n_knots - Degree + 2)          | fixed_knots[0], fixed_knots[1], fixed_knots[2], fixed_knots[3]



      // Read knot values from u_position
      #pragma unroll
      for (int i = 0; i < splineSupportSize; i++)
      {
        const int source_knot_idx = start_knot_idx + i;  //

        if (source_knot_idx < n_knots && source_knot_idx >= 0)
        {
            knots[i] = u_position[b_addrs_action + source_knot_idx * dof + d_idx];
        }
      }

      const bool require_start_boundary = is_start_boundary<Degree>(knot_idx);
      const bool require_goal_boundary = use_implicit_goal ? is_goal_boundary_implicit<Degree>(knot_idx, n_knots) : is_goal_boundary_replicate<Degree>(knot_idx, n_knots);

      if (require_start_boundary || require_goal_boundary)
      {
        const int constraint_offset = require_start_boundary ? b_offset : goal_offset;
        const int constraint_idx = constraint_offset * dof + d_idx;
        constraint.position = require_start_boundary ? start_position[constraint_idx] : goal_position[constraint_idx];
        constraint.velocity = require_start_boundary ? start_velocity[constraint_idx] : goal_velocity[constraint_idx];
        constraint.acceleration = require_start_boundary ? start_acceleration[constraint_idx] : goal_acceleration[constraint_idx];
        constraint.jerk = require_start_boundary ? start_jerk[constraint_idx] : goal_jerk[constraint_idx];
      }


      t_mod = (interpolation_steps > 0)
        ? (float(h_idx) / float(interpolation_steps)) - int(h_idx / interpolation_steps)
        : 0.0f;

      t_mod = h_idx_local < 0 ? 1.0 : t_mod;

      // Create context with precomputed powers for efficient computation
      auto context = create_context<Degree, BasisImpl>(t_mod, knot_dt);
      //BSplineBasisOptimized
      // Apply boundary constraints using the context
      apply_boundary_constraints<Degree>(knots,
        context.knot_dt,
        context.knot_dt_2,
        context.knot_dt_3,
        constraint,
        knot_idx,
        n_knots,
        interpolation_steps,
        use_implicit_goal);

      // Compute interpolation values using the context
      out_pos = context.compute_position(knots);
      out_vel = context.compute_velocity(knots);
      out_acc = context.compute_acceleration(knots);
      out_jerk = context.compute_jerk(knots);

      // write out (use max_out_tsteps for output indexing if different from padded_horizon):
      const int output_horizon = (max_out_tsteps > 0) ? max_out_tsteps : padded_horizon;
      out_position_mem[b_idx * output_horizon * dof + h_idx * dof + d_idx]     = out_pos;
      out_velocity_mem[b_idx * output_horizon * dof + h_idx * dof + d_idx]     = out_vel;
      out_acceleration_mem[b_idx * output_horizon * dof + h_idx * dof + d_idx] = out_acc;
      out_jerk_mem[b_idx * output_horizon * dof + h_idx * dof + d_idx]         = out_jerk;

      if (h_idx == 0 && d_idx == 0)
      {
        // out dt has shape [batch, 1]
        // write out only once per batch.
        out_dt[b_idx] = interpolated_dt;
      }
    }



    }
  }
}