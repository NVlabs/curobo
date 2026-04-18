/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


namespace curobo{
    namespace trajectory{
        namespace bspline{
        /**
         * @brief B-spline basis implementation using factored computation
         *
         * This implementation uses factored mathematical expressions and precomputed
         * terms to potentially improve numerical stability and reduce operations.
         * Features optimized formulations with common subexpression elimination.
         */
        template<int DEGREE>
        struct BSplineBasisFactored;

        // Template specialization for degree 3 (cubic B-spline)
        template<>
        struct BSplineBasisFactored<3> {
            static __device__ __forceinline__ void compute_position_basis(float t, float* basis) {
                // Factored formulation with potentially better numerical stability
                float one_minus_t = 1.0f - t;
                float one_minus_t_sq = one_minus_t * one_minus_t;
                float t_sq = t * t;

                // Factored form to reduce operations
                constexpr float ONE_SIXTH = 1.0f / 6.0f;

                basis[0] = ONE_SIXTH * one_minus_t_sq * one_minus_t;
                basis[1] = ONE_SIXTH * (4.0f - 6.0f * t_sq + 3.0f * t_sq * t);
                basis[2] = ONE_SIXTH * (1.0f + 3.0f * t + 3.0f * t_sq - 3.0f * t_sq * t);
                basis[3] = ONE_SIXTH * t_sq * t;
            }

            static __device__ __forceinline__ void compute_velocity_basis(float t, float* basis) {
                float one_minus_t = 1.0f - t;
                float t_sq = t * t;

                constexpr float ONE_HALF = 0.5f;

                basis[0] = -ONE_HALF * one_minus_t * one_minus_t;
                basis[1] = ONE_HALF * (3.0f * t_sq - 4.0f * t);
                basis[2] = ONE_HALF * (1.0f + 2.0f * t - 3.0f * t_sq);
                basis[3] = ONE_HALF * t_sq;
            }

            static __device__ __forceinline__ void compute_acceleration_basis(float t, float* basis) {
                basis[0] = 1.0f - t;
                basis[1] = 3.0f * t - 2.0f;
                basis[2] = 1.0f - 3.0f * t;
                basis[3] = t;
            }

            static __device__ __forceinline__ void compute_jerk_basis(float t, float* basis) {
                // Jerk is constant for cubic B-spline
                basis[0] = -1.0f;
                basis[1] = 3.0f;
                basis[2] = -3.0f;
                basis[3] = 1.0f;
            }
        };

        // Template specialization for degree 4 (quartic B-spline)
        template<>
        struct BSplineBasisFactored<4> {
            static __device__ __forceinline__ void compute_position_basis(float t, float* basis) {
                float t2 = t * t;
                float t3 = t2 * t;
                float t4 = t3 * t;
                basis[0] = (1.0f - 4.0f*t + 6.0f*t2 - 4.0f*t3 + t4) / 24.0f;
                basis[1] = (11.0f - 12.0f*t - 6.0f*t2 + 12.0f*t3 - 4.0f*t4) / 24.0f;
                basis[2] = (11.0f + 12.0f*t - 6.0f*t2 - 12.0f*t3 + 6.0f*t4) / 24.0f;
                basis[3] = (1.0f + 4.0f*t + 6.0f*t2 + 4.0f*t3 - 4.0f*t4) / 24.0f;
                basis[4] = t4 / 24.0f;
            }

            static __device__ __forceinline__ void compute_velocity_basis(float t, float* basis) {
                float t2 = t * t;
                float t3 = t2 * t;
                basis[0] = (-4.0f + 12.0f*t - 12.0f*t2 + 4.0f*t3) / 24.0f;
                basis[1] = (-12.0f - 12.0f*t + 36.0f*t2 - 16.0f*t3) / 24.0f;
                basis[2] = (12.0f - 12.0f*t - 36.0f*t2 + 24.0f*t3) / 24.0f;
                basis[3] = (4.0f + 12.0f*t + 12.0f*t2 - 16.0f*t3) / 24.0f;
                basis[4] = 4.0f * t3 / 24.0f;
            }

            static __device__ __forceinline__ void compute_acceleration_basis(float t, float* basis) {
                float t2 = t * t;
                basis[0] = (12.0f - 24.0f*t + 12.0f*t2) / 24.0f;
                basis[1] = (-12.0f + 72.0f*t - 48.0f*t2) / 24.0f;
                basis[2] = (-12.0f - 72.0f*t + 72.0f*t2) / 24.0f;
                basis[3] = (12.0f + 24.0f*t - 48.0f*t2) / 24.0f;
                basis[4] = 12.0f * t2 / 24.0f;
            }

            static __device__ __forceinline__ void compute_jerk_basis(float t, float* basis) {
                basis[0] = (-24.0f + 24.0f*t) / 24.0f;
                basis[1] = (72.0f - 96.0f*t) / 24.0f;
                basis[2] = (-72.0f + 144.0f*t) / 24.0f;
                basis[3] = (24.0f - 96.0f*t) / 24.0f;
                basis[4] = 24.0f * t / 24.0f;
            }
        };

        // Template specialization for degree 5 (quintic B-spline)
        template<>
        struct BSplineBasisFactored<5> {
            static __device__ __forceinline__ void compute_position_basis(float t, float* basis) {
                float t2 = t * t;
                float t3 = t2 * t;
                float t4 = t3 * t;
                float t5 = t4 * t;
                basis[0] = -1.0f/120.0f*t5 + (1.0f/24.0f)*t4 - 1.0f/12.0f*t3 + (1.0f/12.0f)*t2 - 1.0f/24.0f*t + (1.0f/120.0f);
                basis[1] = (1.0f/24.0f)*t5 - 1.0f/6.0f*t4 + (1.0f/6.0f)*t3 + (1.0f/6.0f)*t2 - 5.0f/12.0f*t + (13.0f/60.0f);
                basis[2] = -1.0f/12.0f*t5 + (1.0f/4.0f)*t4 - 1.0f/2.0f*t2 + (11.0f/20.0f);
                basis[3] = (1.0f/12.0f)*t5 - 1.0f/6.0f*t4 - 1.0f/6.0f*t3 + (1.0f/6.0f)*t2 + (5.0f/12.0f)*t + (13.0f/60.0f);
                basis[4] = -1.0f/24.0f*t5 + (1.0f/24.0f)*t4 + (1.0f/12.0f)*t3 + (1.0f/12.0f)*t2 + (1.0f/24.0f)*t + (1.0f/120.0f);
                basis[5] = (1.0f/120.0f)*t5;
            }

            static __device__ __forceinline__ void compute_velocity_basis(float t, float* basis) {
                float t2 = t * t;
                float t3 = t2 * t;
                float t4 = t3 * t;
                basis[0] = (-t4 + 4.0f*t3 - 6.0f*t2 + 4.0f*t - 1.0f) / 24.0f;
                basis[1] = (5.0f*t4 - 16.0f*t3 + 12.0f*t2 + 8.0f*t - 10.0f) / 24.0f;
                basis[2] = (-10.0f*t4 + 24.0f*t3 - 24.0f*t) / 24.0f;
                basis[3] = (10.0f*t4 - 16.0f*t3 - 12.0f*t2 + 8.0f*t + 10.0f) / 24.0f;
                basis[4] = (-5.0f*t4 + 4.0f*t3 + 6.0f*t2 + 4.0f*t + 1.0f) / 24.0f;
                basis[5] = t4 / 24.0f;
            }

            static __device__ __forceinline__ void compute_acceleration_basis(float t, float* basis) {
                float t2 = t * t;
                float t3 = t2 * t;
                basis[0] = (-4.0f*t3 + 12.0f*t2 - 12.0f*t + 4.0f) / 24.0f;
                basis[1] = (20.0f*t3 - 48.0f*t2 + 24.0f*t + 8.0f) / 24.0f;
                basis[2] = (-40.0f*t3 + 72.0f*t2 - 24.0f) / 24.0f;
                basis[3] = (40.0f*t3 - 48.0f*t2 - 24.0f*t + 8.0f) / 24.0f;
                basis[4] = (-20.0f*t3 + 12.0f*t2 + 12.0f*t + 4.0f) / 24.0f;
                basis[5] = 4.0f * t3 / 24.0f;
            }

            static __device__ __forceinline__ void compute_jerk_basis(float t, float* basis) {
                float t2 = t * t;
                basis[0] = (-12.0f*t2 + 24.0f*t - 12.0f) / 24.0f;
                basis[1] = (60.0f*t2 - 96.0f*t + 24.0f) / 24.0f;
                basis[2] = (-120.0f*t2 + 144.0f*t) / 24.0f;
                basis[3] = (120.0f*t2 - 96.0f*t - 24.0f) / 24.0f;
                basis[4] = (-60.0f*t2 + 24.0f*t + 12.0f) / 24.0f;
                basis[5] = 12.0f * t2 / 24.0f;
            }
        };
    } // namespace bspline

    } // namespace trajectory
} // namespace curobo