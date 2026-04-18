/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include "third_party/helper_math.h"
#include "bspline_interpolation.cuh"
#include "bspline_gradient_util.cuh"
#include "bspline_context.cuh"

namespace curobo{
    namespace trajectory{
      namespace bspline{




    /**
     * @brief CUDA kernel for interpolating B-spline trajectories with position, velocity,
     *        acceleration, and jerk outputs.
     *
     * This kernel performs B-spline interpolation for multiple trajectories in parallel,
     * computing derivatives up to jerk for motion planning applications.
     *
     * @tparam ScalarType Floating point type (float or double)
     * @tparam Degree B-spline degree (typically 3, 5, or 7)
     * @tparam BasisImpl Backend implementation for basis function computation
     *
     * @param out_position_mem Output trajectory positions
     *                         Shape: [batch_size, padded_horizon, dof]
     * @param out_velocity_mem Output trajectory velocities
     *                         Shape: [batch_size, padded_horizon, dof]
     * @param out_acceleration_mem Output trajectory accelerations
     *                             Shape: [batch_size, padded_horizon, dof]
     * @param out_jerk_mem Output trajectory jerks
     *                     Shape: [batch_size, padded_horizon, dof]
     * @param out_dt Output time steps for each trajectory point
     *               Shape: [batch_size]
     * @param u_position Control points (knot positions) for B-spline interpolation
     *                   Shape: [batch_size, n_knots, dof]
     * @param start_position Initial position states (can be shared across batches)
     *                       Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param start_velocity Initial velocity states (can be shared across batches)
     *                       Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param start_acceleration Initial acceleration states (can be shared across batches)
     *                           Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param start_jerk Initial jerk states (can be shared across batches)
     *                   Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param goal_position Target position states (can be shared across batches)
     *                      Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param goal_velocity Target velocity states (can be shared across batches)
     *                      Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param goal_acceleration Target acceleration states (can be shared across batches)
     *                          Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param goal_jerk Target jerk states (can be shared across batches)
     *                  Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param start_idx Indirection array mapping each batch to its start state index
     *                  Shape: [batch_size] -> enables sharing start states across batches
     * @param goal_idx Indirection array mapping each batch to its goal state index
     *                 Shape: [batch_size] -> enables sharing goal states across batches
     * @param traj_dt Time step values for trajectory interpolation
     *                Shape: [batch_size]
     * @param use_implicit_goal_state Flag to use implicit goal state constraints
     *                                Shape: [batch_size]
     * @param batch_size Number of trajectories to process simultaneously
     * @param padded_horizon Number of time steps in each trajectory. This is horizon + 1.
     * @param dof Degrees of freedom (number of joints/dimensions)
     * @param n_knots Number of control points in B-spline representation
     *
     * @note **Threading Model**: This kernel launches `batch_size * dof * horizon` threads.
     *       Each thread processes one (batch, dof, time_step) combination:
     *       - Thread ID decomposition:
     *         - `d_idx = tid % dof` (innermost for memory coalescing)
     *         - `h_idx = (tid / dof) % horizon`
     *         - `b_idx = tid / (dof * horizon)` (outermost)
     *       - Memory access pattern optimized for warp-level coalescing across DOF dimension
     */
    template<typename ScalarType, int Degree, BasisBackend BasisImpl = BasisBackend::FACTORED>
    __global__ void interpolate_bspline_kernel(
      ScalarType *out_position_mem, ScalarType *out_velocity_mem,
      ScalarType *out_acceleration_mem, ScalarType *out_jerk_mem,
      ScalarType *out_dt,
      const ScalarType *u_position, const ScalarType *start_position,
      const ScalarType *start_velocity, const ScalarType *start_acceleration,
      const ScalarType *start_jerk,
      const ScalarType *goal_position,
      const ScalarType *goal_velocity,
      const ScalarType *goal_acceleration,
      const ScalarType *goal_jerk,
      const int32_t *start_idx,
      const int32_t *goal_idx,
      const ScalarType *traj_dt,
      const uint8_t *use_implicit_goal_state,
      const int batch_size,
      const int padded_horizon, const int dof, const int n_knots)
    {
      const int tid = blockDim.x * blockIdx.x + threadIdx.x;

      // number of threads = batch_size * dof * horizon;
      // read
      const int b_idx = tid / (dof * padded_horizon);
      //const int h_idx = tid % horizon;
      //const int d_idx = (tid / horizon) % dof;

      // make this coalasced by threads in a warp reading h_idx
      const int d_idx = tid % dof;
      const int h_idx = (tid / dof) % padded_horizon;


      if (b_idx >= batch_size || d_idx >= dof  || h_idx >= padded_horizon)
      {
        return;
      }


      const int b_offset = start_idx[b_idx];
      const int goal_offset = goal_idx[b_idx];
      const float interpolated_dt = traj_dt[goal_offset];

      interpolate_bspline_trajectory<ScalarType, Degree, BasisImpl>(out_position_mem,
                                  out_velocity_mem,
                                  out_acceleration_mem,
                                  out_jerk_mem,
                                  out_dt,
                                  u_position,
                                  start_position,
                                  start_velocity,
                                  start_acceleration,
                                  start_jerk,
                                  goal_position,
                                  goal_velocity,
                                  goal_acceleration,
                                  goal_jerk,
                                  interpolated_dt,
                                  use_implicit_goal_state,
                                  batch_size,
                                  padded_horizon,
                                  padded_horizon,
                                  dof,
                                  b_idx,
                                  h_idx,
                                  d_idx,
                                  b_offset,
                                  goal_offset,
                                  n_knots);


    }


    /**
     * @brief CUDA kernel for B-spline interpolation with uniform time stepping and variable
     *        output horizon lengths.
     *
     * This kernel performs B-spline interpolation where all trajectories use the same time
     * step but can have different horizon lengths. Useful for real-time applications where
     * consistent timing is required across batches.
     *
     * @tparam ScalarType Floating point type (float or double)
     * @tparam Degree B-spline degree (typically 3, 5, or 7)
     * @tparam BasisImpl Backend implementation for basis function computation
     *
     * @param out_position_mem Output trajectory positions
     *                         Shape: [batch_size, max_out_tsteps, dof]
     * @param out_velocity_mem Output trajectory velocities
     *                         Shape: [batch_size, max_out_tsteps, dof]
     * @param out_acceleration_mem Output trajectory accelerations
     *                             Shape: [batch_size, max_out_tsteps, dof]
     * @param out_jerk_mem Output trajectory jerks
     *                     Shape: [batch_size, max_out_tsteps, dof]
     * @param out_dt Output time steps for each trajectory point
     *               Shape: [batch_size, max_out_tsteps]
     * @param u_position Control points (knot positions) for B-spline interpolation
     *                   Shape: [batch_size, n_knots, dof]
     * @param knot_dt_mem Time intervals between knot points
     *                    Shape: [batch_size, n_knots-1]
     * @param start_position Initial position states (can be shared across batches)
     *                       Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param start_velocity Initial velocity states (can be shared across batches)
     *                       Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param start_acceleration Initial acceleration states (can be shared across batches)
     *                           Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param start_jerk Initial jerk states (can be shared across batches)
     *                   Shape: [num_start_states, dof] where num_start_states <= batch_size
     * @param goal_position Target position states (can be shared across batches)
     *                      Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param goal_velocity Target velocity states (can be shared across batches)
     *                      Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param goal_acceleration Target acceleration states (can be shared across batches)
     *                          Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param goal_jerk Target jerk states (can be shared across batches)
     *                  Shape: [num_goal_states, dof] where num_goal_states <= batch_size
     * @param start_idx Indirection array mapping each batch to its start state index
     *                  Shape: [batch_size] -> enables sharing start states across batches
     * @param goal_idx Indirection array mapping each batch to its goal state index
     *                 Shape: [batch_size] -> enables sharing goal states across batches
     * @param interpolation_dt Uniform time step for interpolation
     *                         Shape: [1] (single value used for all trajectories)
     * @param use_implicit_goal_state Flag to use implicit goal state constraints
     *                                Shape: [batch_size]
     * @param interpolation_horizon Variable horizon length for each trajectory
     *                              Shape: [batch_size]
     * @param batch_size Number of trajectories to process simultaneously
     * @param max_out_tsteps Maximum number of output time steps across all trajectories
     * @param dof Degrees of freedom (number of joints/dimensions)
     * @param n_knots Number of control points in B-spline representation
     *
     * @note **Threading Model**: This kernel launches `batch_size * dof * max_out_tsteps` threads.
     *       Each thread processes one (batch, dof, time_step) combination:
     *       - Thread ID decomposition:
     *         - `d_idx = tid % dof` (innermost for memory coalescing)
     *         - `h_idx_loc = (tid / dof) % max_out_tsteps`
     *         - `b_idx = tid / (dof * max_out_tsteps)` (outermost)
     *       - Supports variable horizon lengths per batch via `interpolation_horizon[b_idx]`
     *       - Threads beyond actual horizon length for each batch return early
     */
    template<typename ScalarType, int Degree, BasisBackend BasisImpl = BasisBackend::FACTORED>
    __global__  void interpolate_bspline_single_dt_kernel(
                                                    ScalarType       *out_position_mem,
                                                    ScalarType       *out_velocity_mem,
                                                    ScalarType       *out_acceleration_mem,
                                                    ScalarType       *out_jerk_mem,
                                                    ScalarType       *out_dt,
                                                    const ScalarType *u_position,
                                                    const ScalarType *knot_dt_mem,
                                                    const ScalarType *start_position,
                                                    const ScalarType *start_velocity,
                                                    const ScalarType *start_acceleration,
                                                    const ScalarType *start_jerk,
                                                    const ScalarType *goal_position,
                                                    const ScalarType *goal_velocity,
                                                    const ScalarType *goal_acceleration,
                                                    const ScalarType *goal_jerk,
                                                    const int32_t *start_idx,
                                                    const int32_t *goal_idx,
                                                    const ScalarType *interpolation_dt,
                                                    const uint8_t *use_implicit_goal_state,
                                                    const int32_t *interpolation_horizon,
                                                    const int       batch_size,
                                                    const int       max_out_tsteps, // total interpolated points.
                                                    const int       dof,
                                                    const int       n_knots)
    {

      const int tid = blockDim.x * blockIdx.x + threadIdx.x;

      // number of threads = batch_size * dof * horizon;
      const int b_idx = tid / (dof * max_out_tsteps);

      const int d_idx = tid % dof;
      const int h_idx_loc = (tid / dof) % max_out_tsteps;

      if (b_idx >= batch_size || d_idx >= dof  || h_idx_loc >= max_out_tsteps)
      {
        return;
      }

      const int b_offset = start_idx[b_idx];
      const int goal_offset = goal_idx[b_idx];
      const int new_horizon = min(interpolation_horizon[b_idx], max_out_tsteps - 1);
      const int padded_horizon = new_horizon + 1;
      const float interpolated_dt = interpolation_dt[0]; // Use interpolation_dt[0] instead of traj_dt[goal_offset]

      // Call the unified core function with adapted parameters
      interpolate_bspline_trajectory<ScalarType, Degree, BasisImpl>(
          out_position_mem, out_velocity_mem, out_acceleration_mem, out_jerk_mem, out_dt,
          u_position, start_position, start_velocity, start_acceleration, start_jerk,
          goal_position, goal_velocity, goal_acceleration, goal_jerk,
          interpolated_dt, use_implicit_goal_state, batch_size, padded_horizon, max_out_tsteps,
          dof, b_idx, h_idx_loc, d_idx, b_offset, goal_offset, n_knots);



    }

    /**
     * @brief CUDA kernel for computing gradients with respect to B-spline control points
     *        (backward pass for automatic differentiation).
     *
     * This kernel performs the backward pass for B-spline trajectory optimization, computing
     * gradients of a loss function with respect to the control points (knots). It supports
     * gradients from position, velocity, acceleration, and jerk terms. During forward pass,
     * each knot is used at sequential positions of the Support Window (e.g., for interpolating
     * from timestep 0 -> 1, the knot will be used at index 0, for timestep 1 -> 2,
     * the knot will be used at index 1, etc.). In backward pass each thread will read 1
     * interpolated timestep per index of  the support window for each knot, calculate the gradient
     * for that interpolated timestep  and then use warp reduction to sum across interpolation
     * steps for each knot. We ignore the first interpolation steps of the gradient as they are
     * not affected by the knot values (they are interpolated from the start state). For the last
     * set of interpolation steps, we aggregate the gradients only when implicit_goal is False. If
     * it's True, we zero the gradients for the last set of interpolation steps. In addition, we
     * have a padding of 1 timestep at the end of the horizon, this gets accumulated to the last
     * interpolated timestep when implicit_goal is False.
     *
     * @tparam Degree B-spline degree (typically 3, 4, or 5)
     * @tparam ScalarType Floating point type (float or double)
     * @tparam BasisImpl Backend implementation for basis function computation
     *
     * @param out_grad_knots Output gradients with respect to control points
     *                       Shape: [batch_size, n_knots, dof]
     * @param grad_position Input gradients from loss function w.r.t. positions
     *                      Shape: [batch_size, horizon, dof]
     * @param grad_velocity Input gradients from loss function w.r.t. velocities
     *                      Shape: [batch_size, horizon, dof]
     * @param grad_acceleration Input gradients from loss function w.r.t. accelerations
     *                          Shape: [batch_size, horizon, dof]
     * @param grad_jerk Input gradients from loss function w.r.t. jerks
     *                  Shape: [batch_size, horizon, dof]
     * @param traj_dt Time step values used during forward interpolation
     *                Shape: [batch_size] or [batch_size, horizon-1]
     * @param dt_idx Indirection array mapping each batch to its time step configuration
     *               Shape: [batch_size] -> enables sharing time step configs across batches
     * @param use_implicit_goal_state Flag indicating if implicit goal constraints were used
     *                                Shape: [batch_size]
     * @param batch_size Number of trajectories processed simultaneously
     * @param horizon Number of time steps in each trajectory
     * @param dof Degrees of freedom (number of joints/dimensions)
     * @param n_knots Number of control points in B-spline representation
     *
     * @note **Threading Model**: This kernel uses a complex warp-based threading model.
     *       - Total threads: `batch_size * dof * threads_for_n_knots`
     *       - Where `threads_for_n_knots = warps_for_n_knots * 32`
     *       - And `warps_for_n_knots = (n_knots + knots_per_warp - 1) / knots_per_warp`
     *       - Each thread processes gradients for one (batch, dof, knot) combination
     *       - Uses warp-level parallelism for efficient gradient accumulation across
     *         multiple interpolation points that influence each control point
     *       - Thread indices computed via `compute_thread_indices<Degree>()` helper
     */
    template<int Degree, typename ScalarType, BasisBackend BasisImpl = BasisBackend::FACTORED>
    __global__ void bspline_backward_kernel(
        ScalarType *out_grad_knots, const ScalarType *grad_position,
        const ScalarType *grad_velocity, const ScalarType *grad_acceleration,
        const ScalarType *grad_jerk, const ScalarType *traj_dt,
        const int32_t *dt_idx,
        const uint8_t *use_implicit_goal_state,
        const int batch_size,
        const int horizon, const int dof, const int n_knots)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;

        // Compute thread indices and setup common calculations
        BSplineBackwardThreadInfo thread_info = compute_bspline_backward_thread_info<Degree>(
            tid, batch_size, dof, horizon, n_knots, dt_idx, use_implicit_goal_state, traj_dt);

        // Early exit for invalid threads
        const int warps_for_n_knots = curobo::common::ceil_div(n_knots, thread_info.layout.knots_per_warp);
        const int threads_for_n_knots = warps_for_n_knots * curobo::common::warpSize;

        if (tid >= batch_size * dof * threads_for_n_knots ||
            thread_info.knot_idx >= n_knots ||
            thread_info.interpolation_idx >= thread_info.layout.interpolation_steps ||
            thread_info.knot_idx < 0) {
            return;
        }

        // Load gradients into structured arrays (now handles boundary conditions directly)
        GradientArrays<Degree> gradients;
        load_gradients<Degree>(gradients, thread_info, grad_position, grad_velocity,
                             grad_acceleration, grad_jerk);

        auto context = create_context<Degree, BasisImpl>(thread_info.t_mod, thread_info.knot_dt);


        // Use basis-driven accumulation (choose backend via BasisImpl)
        // Default to matrix basis backend to match forward context
        float out_grad = context.compute_backward_grad_from_basis(gradients);
        // Apply warp reduction
        if (thread_info.layout.interpolation_steps > 1) {
          out_grad = warp_interpolation_reduce_sum(out_grad, thread_info.layout.knots_per_warp, thread_info.layout.interpolation_steps);
        }


        // Write output only from first thread in interpolation group
        if (thread_info.interpolation_idx == 0) {
            out_grad_knots[thread_info.b_idx * n_knots * dof + thread_info.knot_idx * dof + thread_info.d_idx] = out_grad;
        }
    }



    }
  }
}