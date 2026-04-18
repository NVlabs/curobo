/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bspline_context.cuh"
#include "bspline_common.cuh"

namespace curobo{
    namespace trajectory{
        namespace bspline{


        /**
         * @brief Structure to encapsulate boundary condition values for B-spline trajectory constraints.
         *
         * This structure holds the position, velocity, acceleration, and jerk values that define
         * boundary conditions at the start or goal of a B-spline trajectory. These values are used
         * to compute fixed knot values that enforce the desired boundary conditions.
         *
         * @note All members are in trajectory space units (e.g., radians for joint angles,
         *       rad/s for velocities, etc.)
         */
        struct BoundaryConstraint {
            float position;   ///< Position constraint value
            float velocity;   ///< Velocity constraint value
            float acceleration;   ///< Acceleration constraint value
            float jerk;  ///< Jerk constraint value

            /**
             * @brief Constructor to initialize boundary constraint values.
             *
             * @param p Position constraint value
             * @param v Velocity constraint value
             * @param a Acceleration constraint value
             * @param j Jerk constraint value
             */
            __device__ __forceinline__ BoundaryConstraint(float position, float velocity, float acceleration, float jerk)
                : position(position), velocity(velocity), acceleration(acceleration), jerk(jerk) {}
        };




        // Forward declaration for coefficient data storage
        template<int Degree>
        struct FixedKnotCoeffsData;


        // Degree 3 (cubic) coefficient data
        template<>
        struct FixedKnotCoeffsData<3> {
            static __device__ __forceinline__ float get(int row, int col) {
                constexpr float coeffs[4][4] = {
                    {1.0f, 1.0f, 1.0f, 1.0f},                          // POS coefficients
                    {-1.0f, 0.0f, 1.0f, 2.0f},                         // VEL coefficients
                    {1.0f/3.0f, -1.0f/6.0f, 1.0f/3.0f, 11.0f/6.0f},        // ACC coefficients (FIXED)
                    {0.0f, 0.0f, 0.0f, 0.0f}                          // JERK coefficients (cubic can't control jerk)
                };
                return coeffs[row][col];
            }
        };

                // Degree 4 (quartic) coefficient data
        template<>
        struct FixedKnotCoeffsData<4> {
            static __device__ __forceinline__ float get(int row, int col) {
                constexpr float coeffs[4][5] = {
                    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},                                   // POS coefficients
                    {-3.0f/2.0f, -1.0f/2.0f, 1.0f/2.0f, 3.0f/2.0f, 5.0f/2.0f},           // VEL coefficients
                    {11.0f/12.0f, -1.0f/12.0f, -1.0f/12.0f, 11.0f/12.0f, 35.0f/12.0f},     // ACC coefficients
                    {-3.0f/12.0f, 1.0f/12.0f, -1.0f/12.0f, 3.0f/12.0f, 25.0f/12.0f }          // JERK coefficients
                };
                return coeffs[row][col];
            }
        };

        // Degree 5 (quintic) coefficient data
        template<>
        struct FixedKnotCoeffsData<5> {
            static __device__ __forceinline__ float get(int row, int col) {
                constexpr float coeffs[4][6] = {
                    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},               // POS coefficients (FIXED)
                    {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f},                                      // VEL coefficients
                    {1.75f, 0.25f, -0.25f, 0.25f, 1.75f, 4.25f},              // ACC coefficients (FIXED)
                    {-0.833333f, 0.083333f, 0.0f, -0.083333f, 0.833333f, 3.75f}                  // JERK coefficients (FIXED)
                };
                return coeffs[row][col];
            }
        };

        // Clean interface for accessing fixed knot coefficients
        template<int Degree>
        struct FixedKnotCoeffs {
            static __device__ __forceinline__ float get_position(int col) {
                return FixedKnotCoeffsData<Degree>::get(0, col);
            }

            static __device__ __forceinline__ float get_velocity(int col) {
                return FixedKnotCoeffsData<Degree>::get(1, col);
            }

            static __device__ __forceinline__ float get_acceleration(int col) {
                return FixedKnotCoeffsData<Degree>::get(2, col);
            }

            static __device__ __forceinline__ float get_jerk(int col) {
                return FixedKnotCoeffsData<Degree>::get(3, col);
            }
        };


        /**
         * @brief Computes fixed knot values to enforce boundary constraints in B-spline trajectories.
         *
         * This function calculates the control point (knot) values needed to satisfy boundary
         * conditions at trajectory endpoints. It uses precomputed coefficient matrices and
         * time step powers to efficiently compute the required knot values that will produce
         * the desired position, velocity, acceleration, and jerk at the boundary.
         *
         * The computation uses the formula:
         * fixed_knots[i] = C_pos[i] * pos + C_vel[i] * vel * dt + C_acc[i] * acc * dt² + C_jerk[i] * jerk * dt³
         * where C_*[i] are degree-specific coefficient matrices.
         *
         * @tparam Degree B-spline degree, determines coefficient matrix and number of fixed knots
         *
         * @param fixed_knots Output array to store computed fixed knot values
         *                    Size: [Degree] - exactly Degree knots are computed
         * @param constraint Boundary condition values (position, velocity, acceleration, jerk)
         * @param knot_dt Time interval between consecutive knots
         * @param knot_dt_2 Square of knot_dt (precomputed for efficiency)
         * @param knot_dt_3 Cube of knot_dt (precomputed for efficiency)
         *
         * @note The function uses unrolled loops for optimal performance on GPU
         * @note Coefficient matrices are compile-time constants specific to each B-spline degree
         */
        template<int Degree>
        __device__ __forceinline__ void compute_fixed_knots(
            float* fixed_knots,
            const BoundaryConstraint& constraint,
            const float knot_dt,
            const float knot_dt_2,
            const float knot_dt_3)
        {

            #pragma unroll
            for (int i = 0; i < get_spline_support_size<Degree>(); ++i) {
                fixed_knots[i] = FixedKnotCoeffs<Degree>::get_position(i) * constraint.position +
                                FixedKnotCoeffs<Degree>::get_velocity(i) * constraint.velocity * knot_dt +
                                FixedKnotCoeffs<Degree>::get_acceleration(i) * constraint.acceleration * knot_dt_2 +
                                FixedKnotCoeffs<Degree>::get_jerk(i) * constraint.jerk * knot_dt_3;
            }
        }

        // Assignment pattern functions - degree-specific implementations
        template<int Degree>
        struct KnotAssignmentPattern {


            /**
             * @brief Assigns start knots to the knots array.
             *
             * @param knots Array of control points for current B-spline segment
             *              Size: [Degree + 1] - modified in-place to enforce constraints
             * @param fixed_knots Array of fixed knot values to enforce constraints
             * @param knot_idx Current knot index in the trajectory
             */
            static __device__ __forceinline__ void assign_start_pattern(
                float* knots,
                const float* fixed_knots,
                int knot_idx)
            {
                // assume knot_idx < support_size
                // n_knots = 16, Degree = 3
                // knot_idx | start_knot_idx | local_support
                // 0        | -4             | fixed_knots[0], fixed_knots[1], fixed_knots[2], fixed_knots[3]
                // 1        | -3             | fixed_knots[1], fixed_knots[2], fixed_knots[3], knots[0]
                // 2        | -2             | fixed_knots[2], fixed_knots[3], knots[0], knots[1]
                // 3        | -1             | fixed_knots[3], knots[0], knots[1], knots[2]


                const int loop_size = get_spline_support_size<Degree>() - knot_idx;

                for (int i = 0; i < loop_size; ++i) {
                    knots[i] = fixed_knots[knot_idx + i];
                }



            }

            /**
             * @brief Assigns goal knots to the knots array.
             *
             * @param knots Array of control points for current B-spline segment
             *              Size: [Degree + 1] - modified in-place to enforce constraints
             * @param fixed_knots Array of fixed knot values to enforce constraints
             * @param knot_idx Current knot index in the trajectory
             * @param n_knots Total number of control points in the trajectory
             */
            static __device__ __forceinline__ void assign_goal_pattern_implicit(
                float* knots,
                const float* fixed_knots,
                int knot_idx,
                int n_knots)
            {
                // assume knot_idx >= n_knots - 1;
                constexpr int supportSize = get_spline_support_size<Degree>();

                // Degree == 3, n_knots = 16
                // knot_idx | local_support
                // 16       | knots[0], knots[1], knots[2], fixed_knots[0]
                // 17       | knots[1], knots[2], fixed_knots[0], fixed_knots[1]
                // 18       | knots[2], fixed_knots[0], fixed_knots[1], fixed_knots[2]
                // 19       | fixed_knots[0], fixed_knots[1], fixed_knots[2], fixed_knots[3]


                const int loop_size = knot_idx - n_knots + 1;
                const int start_target_index = supportSize - loop_size;

                for (int i = 0; i < loop_size; ++i) {
                    knots[start_target_index + i] = fixed_knots[i];
                }


            }

            /**
             * @brief Replicates last knot value for Degree knots to bring the trajectory to rest.
             *
             * @param knots Array of control points for current B-spline segment
             *              Size: [Degree + 1] - modified in-place to enforce constraints
             * @param knot_idx Current knot index in the trajectory
             * @param n_knots Total number of control points in the trajectory
             * @note The final degree knots are the set as last_knot. So the last supportSize
             *       knots are set to last_knot.
             */
            static __device__ __forceinline__ void assign_goal_pattern_replicate(
                float* knots,
                int knot_idx,
                int n_knots)
            {
                constexpr int supportSize = get_spline_support_size<Degree>();
                // only called when knot_idx > n_knots

                // Degree == 3, n_knots = 16

                // knot_idx | local_support
                // 17       | knots[13], knots[14], knots[15], knots[15]
                // 18       | knots[14], knots[15], knots[15], knots[15]
                // 19       | knots[15], knots[15], knots[15], knots[15]

                // knot_idx | loop_size | source_index | target_indices
                // 17       | 1         | 2           | 3
                // 18       | 2         | 1           | 3, 2
                // 19       | 3         | 0           | 3, 2, 1


                // Generalizing to n_knots:
                // Degree == 3
                // knot_idx             | loop_size | source_index | target_indices
                //
                // n_knots + 1          | 1         | 2            | 3
                // n_knots + 2          | 2         | 1            | 3, 2
                // n_knots + 3          | 3         | 0            | 3, 2, 1

                // Degree == 4
                // knot_idx             | loop_size | source_index | target_indices
                // n_knots + 1          | 1         | 3            | 4
                // n_knots + 2          | 2         | 2            | 4, 3
                // n_knots + 3          | 3         | 1            | 4, 3, 2
                // n_knots + 4          | 4         | 0            | 4, 3, 2, 1

                // Degree == 5
                // knot_idx             | loop_size | source_index | target_indices
                // n_knots + 1          | 1         | 4            | 5
                // n_knots + 2          | 2         | 3            | 5, 4
                // n_knots + 3          | 3         | 2            | 5, 4, 3
                // n_knots + 4          | 4         | 1            | 5, 4, 3, 2
                // n_knots + 5          | 5         | 0            | 5, 4, 3, 2, 1

                const int loop_size = knot_idx - n_knots;

                const int source_index = supportSize - loop_size - 1;

                const float source_value = knots[source_index];



                for (int i=0; i < loop_size; ++i) {
                    knots[supportSize - i - 1] = source_value;
                }


            }
        };


    // Boundary condition helper functions



    /**
     * @brief Applies boundary constraints to B-spline control points (knots) based on trajectory requirements.
     *
     * This function enforces boundary conditions at the start and goal of B-spline trajectories by
     * modifying control point values. It handles three types of boundary constraint scenarios:
     * 1. Start boundary: Constrains initial trajectory state (position, velocity, acceleration, jerk)
     * 2. Goal boundary with implicit constraints: Constrains final trajectory state explicitly
     * 3. Goal boundary with replication: Uses knot replication for natural boundary behavior
     *
     * The function determines the boundary type based on knot_idx position and applies the appropriate
     * constraint pattern. It uses precomputed time step powers for efficient computation.
     *
     * @tparam Degree B-spline degree, affects constraint patterns and fixed knot computation
     *
     * @param knots Array of control points for current B-spline segment
     *              Size: [Degree + 1] - modified in-place to enforce constraints
     * @param knot_dt Time interval between consecutive knots
     * @param knot_dt_2 Square of knot_dt (precomputed for efficiency)
     * @param knot_dt_3 Cube of knot_dt (precomputed for efficiency)
     * @param constraint Boundary condition values (position, velocity, acceleration, jerk)
     * @param knot_idx Current knot index in the trajectory
     * @param n_knots Total number of control points in the trajectory
     * @param interpolation_steps Number of time steps per knot interval
     * @param use_implicit_goal Whether to use implicit goal constraints (true) or replication (false)
     *
     * @note The function performs early exit if no boundary constraints are needed for the current knot
     * @note Boundary constraint patterns are degree-specific and handled by KnotAssignmentPattern templates
     * @note Start boundaries use sliding window patterns, goal boundaries use implicit or replication patterns
     */
    template<int Degree>
    __device__ __forceinline__ void apply_boundary_constraints(
                  float *knots,
                  const float knot_dt,
                  const float knot_dt_2,
                  const float knot_dt_3,
                  const BoundaryConstraint& constraint,
                  const int knot_idx,
                  const int n_knots,
                  const int interpolation_steps,
                  const bool use_implicit_goal
                  )
    {
        // Only update knots at boundary conditions
        if (!needs_boundary_update<Degree>(knot_idx, n_knots, use_implicit_goal)) {
            return;
        }

        // Compute fixed knots using precomputed powers from context
        float fixed_knots[get_spline_support_size<Degree>()];
        compute_fixed_knots<Degree>(fixed_knots, constraint, knot_dt, knot_dt_2, knot_dt_3);

        // Apply appropriate assignment pattern based on boundary type
        if (is_start_boundary<Degree>(knot_idx)) {
            // Start boundary: use sliding window pattern
            KnotAssignmentPattern<Degree>::assign_start_pattern(knots, fixed_knots, knot_idx);
        }
        else if (use_implicit_goal && is_goal_boundary_implicit<Degree>(knot_idx, n_knots)) {
            // Goal boundary with implicit constraints
            KnotAssignmentPattern<Degree>::assign_goal_pattern_implicit(knots, fixed_knots, knot_idx, n_knots);
        }
        else if (!use_implicit_goal && is_goal_boundary_replicate<Degree>(knot_idx, n_knots)) {
            // Goal boundary with replication pattern
            KnotAssignmentPattern<Degree>::assign_goal_pattern_replicate(knots, knot_idx, n_knots);
        }
    }




        // Note: Boundary gradient handling has been moved to load_gradients for efficiency
    // This eliminates the need for separate gradient aggregation functions

}
}
}