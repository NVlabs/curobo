/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/math.cuh"

namespace curobo{
    namespace trajectory{
        namespace bspline{
        /**
         * @brief B-spline basis using precomputed coefficient matrices
         *
         * This implementation stores coefficients in constant memory for potentially
         * better cache performance on some GPU architectures. Uses matrix-vector
         * multiplication to compute basis functions and their derivatives.
         */

        // B-spline coefficient matrices for different degrees
        __device__ constexpr float DEGREE_3_COEFFS[4][4] = {
            {-1.0f/6.0f,  3.0f/6.0f, -3.0f/6.0f,  1.0f/6.0f},
            { 3.0f/6.0f, -6.0f/6.0f,  0.0f,       4.0f/6.0f},
            {-3.0f/6.0f,  3.0f/6.0f,  3.0f/6.0f,  1.0f/6.0f},
            { 1.0f/6.0f,  0.0f,       0.0f,       0.0f}
        };

        __device__ constexpr float DEGREE_4_COEFFS[5][5] = {
            { 1.0f/24.0f, -4.0f/24.0f,  6.0f/24.0f, -4.0f/24.0f,  1.0f/24.0f},
            {-4.0f/24.0f, 12.0f/24.0f, -6.0f/24.0f, -12.0f/24.0f, 11.0f/24.0f},
            { 6.0f/24.0f, -12.0f/24.0f, -6.0f/24.0f, 12.0f/24.0f, 11.0f/24.0f},
            {-4.0f/24.0f,  4.0f/24.0f,  6.0f/24.0f,  4.0f/24.0f,  1.0f/24.0f},
            { 1.0f/24.0f,  0.0f,        0.0f,        0.0f,        0.0f}
        };

        __device__ constexpr float DEGREE_5_COEFFS[6][6] = {
            {-1.0f/120.0f,  5.0f/120.0f, -10.0f/120.0f,  10.0f/120.0f,  -5.0f/120.0f,   1.0f/120.0f},
            { 5.0f/120.0f, -20.0f/120.0f,  20.0f/120.0f,  20.0f/120.0f, -50.0f/120.0f,  26.0f/120.0f},
            {-10.0f/120.0f, 30.0f/120.0f,  -0.0f/120.0f, -60.0f/120.0f,   0.0f/120.0f,  66.0f/120.0f},
            { 10.0f/120.0f, -20.0f/120.0f, -20.0f/120.0f,  20.0f/120.0f,  50.0f/120.0f,  26.0f/120.0f},
            { -5.0f/120.0f,   5.0f/120.0f,  10.0f/120.0f,  10.0f/120.0f,   5.0f/120.0f,   1.0f/120.0f},
            {  1.0f/120.0f,   0.0f,         0.0f,          0.0f,          0.0f,          0.0f}
        };

        template<int DEGREE>
        struct BSplineBasisMatrix;

        // Template specialization for degree 3 (cubic B-spline)
        template<>
        struct BSplineBasisMatrix<3> {

            static __device__ __forceinline__ void compute_position_basis(float t, float* basis) {
                float t_powers[4] = {t*t*t, t*t, t, 1.0f};

                curobo::common::matrix_vector_product<4,4>(DEGREE_3_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_velocity_basis(float t, float* basis) {
                // Derivative of position basis
                float t_powers[3] = {3.0f*t*t, 2.0f*t, 1.0f};
                curobo::common::partial_matrix_vector_product<4,3>(DEGREE_3_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_acceleration_basis(float t, float* basis) {
                // Second derivative of position basis
                float t_powers[2] = {6.0f*t, 2.0f};

                curobo::common::partial_matrix_vector_product<4,2>(DEGREE_3_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_jerk_basis(float t, float* basis) {
                // Third derivative - constant for cubic
                float t_powers[1] = {6.0f};
                curobo::common::partial_matrix_vector_product<4,1>(DEGREE_3_COEFFS, t_powers, basis);

            }
        };

        // Template specialization for degree 4 (quartic B-spline)
        template<>
        struct BSplineBasisMatrix<4> {

            static __device__ __forceinline__ void compute_position_basis(float t, float* basis) {
                float t_powers[5] = {t*t*t*t, t*t*t, t*t, t, 1.0f};

                curobo::common::matrix_vector_product<5,5>(DEGREE_4_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_velocity_basis(float t, float* basis) {
                // Derivative of position basis
                float t_powers[4] = {4.0f*t*t*t, 3.0f*t*t, 2.0f*t, 1.0f};

                curobo::common::partial_matrix_vector_product<5,4>(DEGREE_4_COEFFS, t_powers, basis);


            }

            static __device__ __forceinline__ void compute_acceleration_basis(float t, float* basis) {
                // Second derivative of position basis
                float t_powers[3] = {12.0f*t*t, 6.0f*t, 2.0f};

                curobo::common::partial_matrix_vector_product<5,3>(DEGREE_4_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_jerk_basis(float t, float* basis) {
                // Third derivative of position basis
                float t_powers[2] = {24.0f*t, 6.0f};

                curobo::common::partial_matrix_vector_product<5,2>(DEGREE_4_COEFFS, t_powers, basis);

            }
        };

        // Template specialization for degree 5 (quintic B-spline)
        template<>
        struct BSplineBasisMatrix<5> {

            static __device__ __forceinline__ void compute_position_basis(float t, float* basis) {
                float t_powers[6] = {t*t*t*t*t, t*t*t*t, t*t*t, t*t, t, 1.0f};

                curobo::common::matrix_vector_product<6,6>(DEGREE_5_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_velocity_basis(float t, float* basis) {
                // Derivative of position basis
                float t_powers[5] = {5.0f*t*t*t*t, 4.0f*t*t*t, 3.0f*t*t, 2.0f*t, 1.0f};

                curobo::common::partial_matrix_vector_product<6,5>(DEGREE_5_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_acceleration_basis(float t, float* basis) {
                // Second derivative of position basis
                float t_powers[4] = {20.0f*t*t*t, 12.0f*t*t, 6.0f*t, 2.0f};

                curobo::common::partial_matrix_vector_product<6,4>(DEGREE_5_COEFFS, t_powers, basis);

            }

            static __device__ __forceinline__ void compute_jerk_basis(float t, float* basis) {
                // Third derivative of position basis
                float t_powers[3] = {60.0f*t*t, 24.0f*t, 6.0f};

                curobo::common::partial_matrix_vector_product<6,3>(DEGREE_5_COEFFS, t_powers, basis);

            }
        };
    } // namespace bspline
    } // namespace trajectory
} // namespace curobo