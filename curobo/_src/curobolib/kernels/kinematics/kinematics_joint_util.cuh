/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "third_party/helper_math.h"
namespace curobo{
  namespace kinematics{

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
