/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace curobo{
    namespace trajectory{
        namespace bspline{
        /**
         * @brief B-spline basis implementation using direct polynomial evaluation
         *
         * This is the standard implementation using straightforward polynomial
         * computation. It provides clear, readable mathematical expressions
         * for B-spline basis functions and their derivatives.
         */

        // Forward declarations for basis function structs
        template<int Degree>
        struct BSplineBasisPolynomial;

        // Template specialization for degree 3 (cubic B-spline)
        template<>
        struct BSplineBasisPolynomial<3> {
            static __device__ __forceinline__ void compute_position_basis(float t, float* basis) {
                float t2 = t * t;
                float t3 = t2 * t;
                basis[0] = (1.0f - t) * (1.0f - t) * (1.0f - t) / 6.0f;
                basis[1] = (3.0f * t3 - 6.0f * t2 + 4.0f)  / 6.0f;
                basis[2] = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f;
                basis[3] = t3 / 6.0f;
            }

            static __device__ __forceinline__ void compute_velocity_basis(float t, float* basis) {
                float t2 = t * t;
                basis[0] = -3.0f * (1.0f - t) * (1.0f - t) / 6.0f;
                basis[1] = (9.0f * t2 - 12.0f * t) / 6.0f;
                basis[2] = (-9.0f * t2 + 6.0f * t + 3.0f) / 6.0f;
                basis[3] = 3.0f * t2 / 6.0f;
            }

            static __device__ __forceinline__ void compute_acceleration_basis(float t, float* basis) {
                basis[0] = 6.0f * (1.0f - t) / 6.0f;
                basis[1] = (18.0f * t - 12.0f) / 6.0f;
                basis[2] = (-18.0f * t + 6.0f) / 6.0f;
                basis[3] = 6.0f * t / 6.0f;
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
        struct BSplineBasisPolynomial<4> {
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
        struct BSplineBasisPolynomial<5> {
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

    }
    }
}
