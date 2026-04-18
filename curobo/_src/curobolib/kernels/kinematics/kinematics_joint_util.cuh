/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "third_party/helper_math.h"
namespace curobo{
  namespace kinematics{

    template<typename ScalarType>
    __device__ __forceinline__ void fixed_joint_fn(const ScalarType *fixedTransform,
                                                   const int col_idx,
                                                   float          *JM)
    {
      JM[0] = fixedTransform[0];
      JM[1] = fixedTransform[4];
      JM[2] = fixedTransform[8];
      JM[3] = col_idx == 3 ? 1.0f : 0.0f;
    }

    // prism_fn withOUT control flow
    template<typename ScalarType>
    __device__ __forceinline__ void prism_fn(const ScalarType *fixedTransform,
                                             const float angle, const int col_idx,
                                             float *JM, const int xyz)
    {
      //        int _and = (col_idx & (col_idx>>1)) & 0x1; // 1 for thread 3, 0 for all
      // other threads (0,1,2)
      //
      //        float f1 = (1-_and) + _and * angle; // 1 for threads 0,1,2; angle for
      // thread 3       int addr_offset = (1-_and) * col_idx + _and * xyz; // col_idx
      // for threads 0,1,2; xyz for thread 3
      //
      //        JM[0] = fixedTransform[0 + addr_offset] * f1 + _and *
      // fixedTransform[3];//FT_0[1];   JM[1] = fixedTransform[M + addr_offset]
      // * f1 + _and * fixedTransform[M + 3];//FT_1[1];         JM[2] = fixedTransform[M
      // + M + addr_offset] * f1 + _and * fixedTransform[M + M + 3];//FT_2[1];
      // JM[3] = fixedTransform[M + M + M + addr_offset] * (1-_and) + _and * 1; //
      // first three threads will get fixedTransform[3M+col_idx], the last thread
      // will get 1

      if (col_idx <= 2)
      {
        fixed_joint_fn(&fixedTransform[col_idx], col_idx, &JM[0]);
      }
      else
      {
        JM[0] = fixedTransform[0 + xyz] * angle + fixedTransform[3];     // FT_0[1];
        JM[1] = fixedTransform[4 + xyz] * angle + fixedTransform[7]; // FT_1[1];
        JM[2] = fixedTransform[8 + xyz] * angle +
                fixedTransform[11];                               // FT_2[1];
        JM[3] = 1;
      }
    }
    template<typename ScalarType>
    __device__ __forceinline__ void xyz_rot_fn(const ScalarType *fixedTransform,
                                               const float angle, const int col_idx,
                                               float *JM, const int xyz)
    {
      float cos   = 0.0f;//__cosf(angle);
      float sin   = 0.0f;//__sinf(angle);
      sincosf(angle, &sin, &cos);

      float n_sin = -1 * sin;

      // Precompute bit operations for all axes
      int bit1 = col_idx & 0x1;
      int bit2 = (col_idx & 0x2) >> 1;
      int _xor = bit1 ^ bit2;
      int col_idx_by_2 = col_idx / 2;
      int col_idx_per_2 = col_idx % 2;

      // Compute initial f1 and f2 values based on axis
      // X: f1 = (1-div)*cos + div*n_sin, f2 = (1-div)*sin + div*cos
      // Y: f1 = (1-div)*cos + div*sin,   f2 = (1-div)*n_sin + div*cos
      // Z: f1 = (1-mod)*cos + mod*n_sin, f2 = (1-mod)*sin + mod*cos

      float f1_x = (1 - col_idx_by_2) * cos + col_idx_by_2 * n_sin;
      float f2_x = (1 - col_idx_by_2) * sin + col_idx_by_2 * cos;

      float f1_y = (1 - col_idx_by_2) * cos + col_idx_by_2 * sin;
      float f2_y = (1 - col_idx_by_2) * n_sin + col_idx_by_2 * cos;

      float f1_z = (1 - col_idx_per_2) * cos + col_idx_per_2 * n_sin;
      float f2_z = (1 - col_idx_per_2) * sin + col_idx_per_2 * cos;

      // Select base values using arithmetic (no branches)
      int is_x = (xyz == 0);
      int is_y = (xyz == 1);
      int is_z = (xyz == 2);

      float f1_base = is_x * f1_x + is_y * f1_y + is_z * f1_z;
      float f2_base = is_x * f2_x + is_y * f2_y + is_z * f2_z;

      // Apply masking based on axis
      // X: f1 = _xor*f1 + (1-_xor)*1, f2 = _xor*f2
      // Y: f1 = (1-mod)*f1 + mod*1,   f2 = (1-mod)*f2
      // Z: f1 = (1-div)*f1 + div*1,   f2 = (1-div)*f2

      float f1 = is_x * (_xor * f1_base + (1 - _xor) * 1) +
                 is_y * ((1 - col_idx_per_2) * f1_base + col_idx_per_2 * 1) +
                 is_z * ((1 - col_idx_by_2) * f1_base + col_idx_by_2 * 1);

      float f2 = is_x * (_xor * f2_base) +
                 is_y * ((1 - col_idx_per_2) * f2_base) +
                 is_z * ((1 - col_idx_by_2) * f2_base);

      // Compute f3 for JM[3]
      // X: f3 = 1 - _xor, Y: f3 = col_idx_per_2, Z: f3 = col_idx_by_2
      float f3 = is_x * (1 - _xor) + is_y * col_idx_per_2 + is_z * col_idx_by_2;

      // Compute address offset
      // X: addr = _xor + (1-_xor)*col_idx, Y: addr = mod*col_idx, Z: addr = div*col_idx
      int addr_offset = is_x * (_xor + (1 - _xor) * col_idx) +
                        is_y * (col_idx_per_2 * col_idx) +
                        is_z * (col_idx_by_2 * col_idx);

      // Cross product column indices
      // X,Y: use column 2 (indices 2, 6, 10), Z: use column 1 (indices 1, 5, 9)
      int cross_col_0 = is_z * 1 + (is_x + is_y) * 2;
      int cross_col_1 = is_z * 5 + (is_x + is_y) * 6;
      int cross_col_2 = is_z * 9 + (is_x + is_y) * 10;

      const float last_row = addr_offset == 3 ? f3 : 0.0f;
      // Final computation
      JM[0] = fixedTransform[0 + addr_offset] * f1 + f2 * fixedTransform[cross_col_0];
      JM[1] = fixedTransform[4 + addr_offset] * f1 + f2 * fixedTransform[cross_col_1];
      JM[2] = fixedTransform[8 + addr_offset] * f1 + f2 * fixedTransform[cross_col_2];
      JM[3] = last_row;
    }

    template<typename AccumulatorType>
    __device__ __forceinline__ void
    rot_backward_translation(const float3& vec, const float *cumul_mat, const float3& l_pos,
                             const float3& loc_grad, AccumulatorType& grad_q, const float axis_sign = 1)
    {
      float3 e_pos, j_pos;

      e_pos.x = cumul_mat[3];
      e_pos.y = cumul_mat[7];
      e_pos.z = cumul_mat[11];

      // compute position gradient:
      j_pos = l_pos - e_pos;

      float3 scale_grad = axis_sign * loc_grad;
      grad_q += dot(scale_grad, cross(vec, j_pos));
    }


    __device__ __forceinline__ void
    xyz_rot_backward_translation(const float *cumul_mat, const float3& l_pos, const float3& loc_grad,
                                 float& grad_q, const int xyz,  const float axis_sign = 1)
    {
      // get rotation vector:
      rot_backward_translation(
        make_float3(cumul_mat[0 + xyz], cumul_mat[4 + xyz], cumul_mat[8 + xyz]),
        &cumul_mat[0], l_pos, loc_grad, grad_q, axis_sign);
    }


    template<typename AccumulatorType>
    __device__ __forceinline__ void
    xyz_rot_backward(const float *link_cumul_mat, const float3& l_pos,
                    const float3& loc_grad_position,
                   const float3& loc_grad_orientation, AccumulatorType& grad_q,  const int xyz, const float axis_sign = 1)
    {
      // Directly compute the rotation vector based on xyz without branching
      const float3 vec = make_float3(link_cumul_mat[0 + xyz], link_cumul_mat[4 + xyz], link_cumul_mat[8 + xyz]);

      // Apply rotation backward translation and rotation
      rot_backward_translation(vec, &link_cumul_mat[0], l_pos, loc_grad_position, grad_q, axis_sign);

      grad_q += axis_sign * dot(vec, loc_grad_orientation);
    }

    template<typename AccumulatorType>
    __device__ __forceinline__ void
    xyz_prism_backward(const float *cumul_mat, const float3& loc_grad,
                      AccumulatorType& grad_q, const int xyz,  const float axis_sign = 1)
    {
      const float3 vec = make_float3(cumul_mat[0 + xyz], cumul_mat[4 + xyz], cumul_mat[8 + xyz]);

      grad_q += axis_sign * dot(vec, loc_grad);

    }




  }
}
