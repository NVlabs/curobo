/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


namespace curobo {
  namespace common{


    template<typename ScalarType>
    __host__ __device__ __forceinline__  ScalarType relu(ScalarType var){
      return var < 0 ? 0 : var;
    }



    __host__ __device__ __forceinline__ float ceil_div(const float &a, const float &b) {
      return (a + b - 1) / b;
    }
    // Helper function for dot product computation
    template<int N>
    __host__ __device__ __forceinline__ float dot_product(const float* knots, const float* basis) {
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            result += knots[i] * basis[i];
        }
        return result;
    }
     // Helper function for dot product computation
     template<int N>
     __host__ __device__ __forceinline__ float dot_product_reverse(const float* knots, const float* basis) {
         float result = 0.0f;
         #pragma unroll
         for (int i = 0; i < N; ++i) {
             result += knots[i] * basis[N - 1 - i];
         }
         return result;
     }

    /**
     * @brief Computes matrix-vector-vector product: left_vec^T * coeff_matrix * right_vec
     * @tparam N Number of rows in coefficient matrix and length of left_vec
     * @tparam M Number of columns in coefficient matrix and length of right_vec
     * @param left_vec Input left vector
     * @param coeff_matrix NxM coefficient matrix
     * @param right_vec Input right vector
     * @return Scalar result of the computation
     */
     template<int N, int M>
     __device__ __forceinline__ float matrix_bilinear_product(
         const float* left_vec,
         const float coeff_matrix[N][M],
         const float* right_vec) {

         float result = 0.0f;
         #pragma unroll
         for (int i = 0; i < N; i++) {
             float basis_value = 0.0f;
             #pragma unroll
             for (int j = 0; j < M; j++) {
                 basis_value += coeff_matrix[i][j] * right_vec[j];
             }
             result += left_vec[i] * basis_value;
         }
         return result;
     }

     template<int N, int M>
     __device__ __forceinline__  void matrix_vector_product(
        const float in_matrix[N][M],
        const float in_vec[M],
        float out_vec[N]
     )
     {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out_vec[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < M; ++j) {
                out_vec[i] += in_matrix[i][j] * in_vec[j];
            }
        }
      }
      template<int N, int M>
     __device__ __forceinline__  void partial_matrix_vector_product(
        const float in_matrix[N][N], // only N x M part of the matrix is used.
        const float in_vec[M],
        float out_vec[N]
     )
     {

        #pragma unroll
        for (int i = 0; i < N; ++i) {
            out_vec[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < M; ++j) {
                out_vec[i] += in_matrix[i][j] * in_vec[j];
            }
        }
      }
    }
  }