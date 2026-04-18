/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/math.cuh"
#include "basis/bspline_basis_polynomial.cuh"
#include "basis/bspline_basis_factored.cuh"
#include "basis/bspline_basis_matrix.cuh"
#include "bspline_gradient_util.cuh"

namespace curobo{

    namespace trajectory{


        namespace bspline{
        /**
         * @brief BSplineContext provides efficient B-spline computations using BSplineBasis.
         *
         * This struct:
         * - Uses BSplineBasis from bspline_basis.cuh for all basis computations (DRY principle)
         * - Allows swapping different basis implementations via template parameter
         * - Provides a single preallocated basis array reused across all computations
         * - Stores knot_dt for scaling derivatives
         *
         * @tparam Degree The degree of the B-spline (3, 4, or 5)
         * @tparam BasisImpl The basis implementation to use (defaults to BSplineBasis)
         */
        template<int Degree, template<int> class BasisImpl = BSplineBasisPolynomial>
        struct BSplineContext {
            // Store t_mod for basis computations
            float t_mod;

            // Store knot_dt for scaling derivatives
            float knot_dt;

            // Store knot_dt_2 for scaling derivatives
            float knot_dt_2;

            // Store knot_dt_3 for scaling derivatives
            float knot_dt_3;

            // Preallocated basis array (reused across all computations)
            static constexpr int splineSupportSize = get_spline_support_size<Degree>();
            float basis[splineSupportSize];

            /**
             * @brief Constructor stores parameters for basis computations
             * @param t_mod_val Normalized time parameter [0, 1]
             * @param knot_dt_val Time step between knot points
             */
            __device__ __forceinline__ BSplineContext(float t_mod_val, float knot_dt_val)
                : t_mod(t_mod_val), knot_dt(knot_dt_val), knot_dt_2(knot_dt_val * knot_dt_val), knot_dt_3(knot_dt_val * knot_dt_val * knot_dt_val) {

                // Initialize basis array to zero
                #pragma unroll
                for (int i = 0; i < splineSupportSize; ++i) {
                    basis[i] = 0.0f;
                }
            }



            /**
             * @brief Compute position using the basis implementation
             * @param knots Control points for the B-spline
             * @return Interpolated position value
             */
            __device__ __forceinline__ float compute_position(const float* knots) {
                BasisImpl<Degree>::compute_position_basis(t_mod, basis);
                return curobo::common::dot_product<splineSupportSize>(knots, basis);
            }

            /**
             * @brief Compute velocity using the basis implementation
             * @param knots Control points for the B-spline
             * @return Interpolated velocity value (scaled by 1/knot_dt)
             */
            __device__ __forceinline__ float compute_velocity(const float* knots) {
                BasisImpl<Degree>::compute_velocity_basis(t_mod, basis);
                return curobo::common::dot_product<splineSupportSize>(knots, basis) / knot_dt;
            }

            /**
             * @brief Compute acceleration using the basis implementation
             * @param knots Control points for the B-spline
             * @return Interpolated acceleration value (scaled by 1/knot_dt²)
             */
            __device__ __forceinline__ float compute_acceleration(const float* knots) {
                BasisImpl<Degree>::compute_acceleration_basis(t_mod, basis);
                return curobo::common::dot_product<splineSupportSize>(knots, basis) / knot_dt_2;
            }

            /**
             * @brief Compute jerk using the basis implementation
             * @param knots Control points for the B-spline
             * @return Interpolated jerk value (scaled by 1/knot_dt³)
             */
            __device__ __forceinline__ float compute_jerk(const float* knots) {
                BasisImpl<Degree>::compute_jerk_basis(t_mod, basis);
                return curobo::common::dot_product<splineSupportSize>(knots, basis) / knot_dt_3;
            }

            /**
             * @brief Compute backward gradient for B-spline control points using basis functions.
             *
             * This function computes the gradient with respect to control points by combining
             * gradients from position, velocity, acceleration, and jerk terms. It performs the
             * backward pass computation for B-spline trajectory optimization by:
             *
             * 1. Computing basis functions for each derivative order (position, velocity, acceleration, jerk)
             * 2. Taking dot products with corresponding gradient arrays
             * 3. Scaling derivatives by appropriate time step powers
             * 4. Summing all contributions to get total gradient w.r.t. control point
             *
             * The computation follows:
             * grad = grad_pos·B₀ + (grad_vel·B₁)/dt + (grad_acc·B₂)/dt² + (grad_jerk·B₃)/dt³
             * where B₀, B₁, B₂, B₃ are basis functions for position, velocity, acceleration, jerk
             *
             * @param gradients Structure containing gradient arrays for position, velocity,
             *                  acceleration, and jerk from the forward pass
             *                  Shape: each array has size [Degree + 1]
             *
             * @return Combined gradient value w.r.t. the control point
             *
             * @note This function reuses the internal basis array for efficiency across computations
             * @note Time scaling ensures correct units for each derivative term
             * @note Used in the backward pass of B-spline trajectory optimization algorithms
             */
            __device__ __forceinline__ float compute_backward_grad_from_basis(
                const GradientArrays<Degree>& gradients)
            {
                float gradient_sum_pos = 0.0f;
                float gradient_sum_vel = 0.0f;
                float gradient_sum_acc = 0.0f;
                float gradient_sum_jerk = 0.0f;

                // For a fixed knot k, the contributing time steps correspond to
                // forward segments with knot_idx_forward = k + i (i in [0, Degree]).
                // In those segments, knot k appears at basis index (Degree - i).
                // Therefore, we must pair gradients[i] with basis[Degree - i].

                // Position
                BasisImpl<Degree>::compute_position_basis(t_mod, basis);
                gradient_sum_pos = curobo::common::dot_product_reverse<splineSupportSize>(gradients.g_pos, basis);
                /*
                #pragma unroll
                for (int i = 0; i < splineSupportSize; ++i) {
                    gradient_sum_pos += gradients.g_pos[i] * basis[splineSupportSize - 1 - i];
                }
                */

                // Velocity
                BasisImpl<Degree>::compute_velocity_basis(t_mod, basis);
                gradient_sum_vel = curobo::common::dot_product_reverse<splineSupportSize>(gradients.g_vel, basis);

                // Acceleration
                BasisImpl<Degree>::compute_acceleration_basis(t_mod, basis);
                gradient_sum_acc = curobo::common::dot_product_reverse<splineSupportSize>(gradients.g_acc, basis);


                // Jerk
                BasisImpl<Degree>::compute_jerk_basis(t_mod, basis);
                gradient_sum_jerk = curobo::common::dot_product_reverse<splineSupportSize>(gradients.g_jerk, basis);


                return gradient_sum_pos + (gradient_sum_vel / knot_dt) +
                       (gradient_sum_acc / knot_dt_2) + (gradient_sum_jerk / knot_dt_3);
            }

        };

        /**
         * @brief Enum class for selecting B-spline basis computation backend
         * All backends produce identical mathematical results; only computation strategy differs.
         */
        enum class BasisBackend : int {
            FACTORED = 0,      // BSplineBasisFactored backend
            POLYNOMIAL = 1,    // BSplineBasisPolynomial backend
            MATRIX = 2         // BSplineBasisMatrix backend
        };

        /**
         * @brief Factory function to create BSplineContext with specified basis computation backend.
         *
         * This function creates a BSplineContext using one of three available basis computation
         * backends. All backends produce mathematically identical results but differ in
         * computational strategy and performance characteristics:
         *
         * - **FACTORED**: Uses factored basis computation with optimized recursive calculations
         * - **POLYNOMIAL**: Uses direct polynomial evaluation for basis functions
         * - **MATRIX**: Uses matrix-based basis computation for maximum numerical stability
         *
         * The choice of backend can affect performance depending on GPU architecture and
         * compiler optimizations, but mathematical results are equivalent.
         *
         * @tparam Degree The degree of the B-spline (3, 4, or 5)
         * @tparam BasisImpl The basis computation backend to use (FACTORED, POLYNOMIAL, or MATRIX)
         *
         * @param t_mod Normalized time parameter within knot interval [0, 1)
         * @param knot_dt Time step between consecutive knot points
         *
         * @return BSplineContext instance configured with the specified basis implementation
         *
         * @note The function uses compile-time `if constexpr` for zero-overhead backend selection
         * @note Default backend is FACTORED for optimal performance on most GPU architectures
         * @note All backends support degrees 3, 4, and 5 with identical mathematical properties
         */
        template<int Degree=3, BasisBackend BasisImpl = BasisBackend::FACTORED>
        __device__ __forceinline__ auto create_context(float t_mod, float knot_dt) {
            if constexpr (BasisImpl == BasisBackend::POLYNOMIAL) {
                return BSplineContext<Degree, BSplineBasisPolynomial>(t_mod, knot_dt);
            } else if constexpr (BasisImpl == BasisBackend::FACTORED) {
                return BSplineContext<Degree, BSplineBasisFactored>(t_mod, knot_dt);
            } else if constexpr (BasisImpl == BasisBackend::MATRIX) {
                return BSplineContext<Degree, BSplineBasisMatrix>(t_mod, knot_dt);
            } else {
                static_assert(static_cast<int>(BasisImpl) <= 2, "Invalid BasisBackend");
            }
        }
    }


    } // namespace trajectory
} // namespace curobo