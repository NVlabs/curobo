/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "third_party/helper_math.h"

namespace curobo{
  namespace kinematics{


    template<typename ScalarType>
    __device__ __forceinline__ void
    transform_sphere(const float *transform_mat, const ScalarType *sphere, float *C)
    {
      float4 sphere_pos = *(float4 *)&sphere[0];
      int    st_idx     = 0;

#pragma unroll 3

      for (int i = 0; i < 3; i++)
      {
        st_idx = i * 4;

        // do dot product:
        // C[i] = transform_mat[st_idx] * sphere_pos.x + transform_mat[st_idx+1] *
        // sphere_pos.y + transform_mat[st_idx+2] * sphere_pos.z +
        // transform_mat[st_idx + 3];
        float4 tm = *(float4 *)&transform_mat[st_idx];
        C[i] =
          tm.x * sphere_pos.x + tm.y * sphere_pos.y + tm.z * sphere_pos.z + tm.w;
      }
      C[3] = sphere_pos.w;
    }


    __device__ __forceinline__ void
    transform_sphere_float4(const float *transform_mat, const float4& sphere_pos, float4 &C)
    {

      C.x = transform_mat[0] * sphere_pos.x + transform_mat[1] * sphere_pos.y +
            transform_mat[2] * sphere_pos.z + transform_mat[3];
      C.y = transform_mat[4] * sphere_pos.x + transform_mat[5] * sphere_pos.y +
            transform_mat[6] * sphere_pos.z + transform_mat[7];
      C.z = transform_mat[8] * sphere_pos.x + transform_mat[9] * sphere_pos.y +
            transform_mat[10] * sphere_pos.z + transform_mat[11];
      C.w = sphere_pos.w;

    }
    template<typename ScalarType>
    __device__ __forceinline__ void
    transform_sphere_float4(const float *transform_mat, const ScalarType *sphere, float4 &C)
    {
      float4 sphere_pos = *(float4 *)&sphere[0];

      transform_sphere_float4(transform_mat, sphere_pos, C);

    }


    __device__ __forceinline__ void update_axis_direction(
      float& angle,
      const int j_type,
      const float2 &j_offset)
    {
      // Assume that input j_type >= 0 . Check fixed joint outside of this function.
      // sign should be +ve <= 5 and -ve >5
      // j_type range is [0, 11].
      // cuda code treats -1.0 * 0.0 as negative. Hence we subtract 6. If in future, -1.0 * 0.0 =
      // +ve,
      // then this code should be j_type - 5.
      angle = j_offset.x * angle + j_offset.y;
    }


    __forceinline__ __device__ void read_cumul_to_shared_mem(
      float *shared_cumul_mat, // size is num_batches_per_block * nlinks * 12
      const float *global_cumul_mat, // size is batch_size * nlinks * 12
      const int start_batch_index,
      const int thread_idx,
      const int nlinks,
      const int batch_size,
      const int num_batches_per_block,
      const int threads_per_batch
    )
    {
      // Calculate the actual number of valid batches we can process
      const int remaining_batches = batch_size - start_batch_index;
      const int actual_batches_per_block = min(num_batches_per_block, remaining_batches);

      // Calculate active threads based on actual batches (4 threads per batch)
      const int active_threads = actual_batches_per_block * threads_per_batch;

      // Only process valid elements
      const int total_valid_elements = actual_batches_per_block * nlinks;

      // Calculate exact number of iterations for this thread
      const int start_element = thread_idx;
      const int end_element = total_valid_elements;

      #pragma unroll
      for (int element_idx = start_element; element_idx < end_element; element_idx += active_threads) {
        const int batch_in_block = element_idx / nlinks;
        const int link_idx = element_idx % nlinks;
        const int global_batch_idx = start_batch_index + batch_in_block;

        const int shared_offset = batch_in_block * nlinks * 12 + link_idx * 12;
        const int global_offset = global_batch_idx * nlinks * 12 + link_idx * 12;

        *(float4 *)&shared_cumul_mat[shared_offset] = *(float4 *)&global_cumul_mat[global_offset];
        *(float4 *)&shared_cumul_mat[shared_offset + 4] = *(float4 *)&global_cumul_mat[global_offset + 4];
        *(float4 *)&shared_cumul_mat[shared_offset + 8] = *(float4 *)&global_cumul_mat[global_offset + 8];
      }

      __syncthreads();
    }

    template <int CumulMatSize = 12>
    __forceinline__ __device__ void write_cumul_to_global_mem(
      float *global_cumul_mat, // size is batch_size * nlinks * 12
      const float *cumul_mat, // size is num_batches_per_block * nlinks * 12
      const int start_batch_index,
      const int thread_idx,
      const int nlinks,
      const int batch_size,
      const int num_batches_per_block,
      const int threads_per_batch
    )
    {
      // Calculate the actual number of valid batches we can process
      const int remaining_batches = batch_size - start_batch_index;
      const int actual_batches_per_block = min(num_batches_per_block, remaining_batches);

      // Calculate active threads based on actual batches (4 threads per batch)
      const int active_threads = actual_batches_per_block * threads_per_batch;

      // Only process valid elements
      const int total_valid_elements = actual_batches_per_block * nlinks;

      // Calculate exact number of iterations for this thread
      const int start_element = thread_idx;
      const int end_element = total_valid_elements;

      #pragma unroll
      for (int element_idx = start_element; element_idx < end_element; element_idx += active_threads) {
        const int batch_in_block = element_idx / nlinks;
        const int link_idx = element_idx % nlinks;
        const int global_batch_idx = start_batch_index + batch_in_block;

        // Skip if we're beyond the valid batch range
        if (global_batch_idx >= batch_size) {
          break;
        }

        const int shared_offset = batch_in_block * nlinks * CumulMatSize + link_idx * CumulMatSize;
        const int global_offset = global_batch_idx * nlinks * 12 + link_idx * 12;

        // Write 12 values (3x4 matrix) using vectorized operations for efficiency
        *(float4 *)&global_cumul_mat[global_offset] = *(float4 *)&cumul_mat[shared_offset];
        *(float4 *)&global_cumul_mat[global_offset + 4] = *(float4 *)&cumul_mat[shared_offset + 4];
        *(float4 *)&global_cumul_mat[global_offset + 8] = *(float4 *)&cumul_mat[shared_offset + 8];
      }

      // __syncthreads();
    }

  }
}