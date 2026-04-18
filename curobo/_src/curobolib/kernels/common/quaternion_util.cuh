/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "third_party/helper_math.h"
namespace curobo {
namespace common {

    /**
     * @brief Multiply two quaternions
     *
     * @param q1 First quaternion (x y z w format)
     * @param q2 Second quaternion (x y z w format)
     * @return Product quaternion q1 * q2
     */
    __host__ __device__ __forceinline__ float4
    quaternion_multiply(const float4& q1, const float4& q2)
    {
        // q1 = [x1, y1, z1, w1], q2 = [x2, y2, z2, w2]
        // q1 * q2 = [w1*x2 + x1*w2 + y1*z2 - z1*y2,
        //            w1*y2 - x1*z2 + y1*w2 + z1*x2,
        //            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        //            w1*w2 - x1*x2 - y1*y2 - z1*z2]

        const float x1 = q1.x, y1 = q1.y, z1 = q1.z, w1 = q1.w;
        const float x2 = q2.x, y2 = q2.y, z2 = q2.z, w2 = q2.w;

        return make_float4(
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  // x component
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  // y component
            w1*z2 + x1*y2 - y1*x2 + z1*w2,  // z component
            w1*w2 - x1*x2 - y1*y2 - z1*z2   // w component
        );
    }

    /**
     * @brief Compute quaternion conjugate (inverse for unit quaternions)
     *
     * @param q Input quaternion in x y z w format
     * @return Conjugate quaternion [-x, -y, -z, w]
     */
    __host__ __device__ __forceinline__ float4
    quaternion_conjugate(const float4& q)
    {
        return make_float4(-q.x, -q.y, -q.z, q.w);
    }

    __host__ __device__ __forceinline__ float4
    quaternion_normalize(const float4& input_quaternion)
    {
        float inv_length = rsqrtf(dot(input_quaternion, input_quaternion));
        inv_length = input_quaternion.w < 0.0 ? -inv_length : inv_length;
        float4 normalized_quaternion = input_quaternion * inv_length;
        return normalized_quaternion;
    }

    __host__ __device__ __forceinline__ float3
    quaternion_transform_vector(const float4& quaternion, const float3& vector)
    {
        // Extract quaternion components (x, y, z, w format)
        float4 normalized_quaternion = quaternion_normalize(quaternion);


        // Rotate point by quaternion using efficient formula:
        // p' = p + 2*q_w*(q_xyz × p) + 2*(q_xyz × (q_xyz × p))
        const float3 q_xyz = make_float3(normalized_quaternion);

        // q_xyz × point
        const float3 cross1 = cross(q_xyz, vector);

        // q_xyz × (q_xyz × point)
        const float3 cross2 = cross(q_xyz, cross1);

        // Apply rotation: p' = p + 2*qw*(q_xyz × p) + 2*(q_xyz × (q_xyz × p))
        const float3 rotated_vector = make_float3(
            vector.x + 2.0f * normalized_quaternion.w * cross1.x + 2.0f * cross2.x,
            vector.y + 2.0f * normalized_quaternion.w * cross1.y + 2.0f * cross2.y,
            vector.z + 2.0f * normalized_quaternion.w * cross1.z + 2.0f * cross2.z
        );
        return rotated_vector;
    }

    __host__ __device__ __forceinline__ void quaternion_gradient_to_angular_velocity(
        float4 quat, // xyzw format
        float *quat_grad, // wxyz format
        float3 &omega // xyz
      )
    {
      const float dqw = quat_grad[0];
      const float dqx = quat_grad[1];
      const float dqy = quat_grad[2];
      const float dqz = quat_grad[3];

      // Compute the angular velocity gradient using the Jacobian transpose
      omega.x = 0.5 * (-quat.x * dqw + quat.w * dqx + quat.z * dqy - quat.y * dqz);
      omega.y = 0.5 * (-quat.y * dqw - quat.z * dqx + quat.w * dqy + quat.x * dqz);
      omega.z = 0.5 * (-quat.z * dqw + quat.y * dqx - quat.x * dqy + quat.w * dqz);

    }

     /**
     * @brief get quaternion from transformation matrix
     *
     * @param t # rotation matrix 3x3
     * @param q quaternion in xyzw format
     */
     __device__ __forceinline__ void quaternion_from_rotation_matrix(float *t, float4 &q)
     {
       float n;
       float n_sqrt;

       if (t[8] < 0.0)
       {
         if (t[0] > t[4])
         {
           n      = 1 + t[0] - t[4] - t[8];
           n_sqrt = 0.5 * rsqrtf(n);
           q.x   = n * n_sqrt;
           q.y   = (t[1] + t[3]) * n_sqrt;
           q.z   = (t[6] + t[2]) * n_sqrt;
           q.w   = -1 * (t[5] - t[7]) * n_sqrt; // * -1 ; // this is the wrong one?
         }
         else
         {
           n      = 1 - t[0] + t[4] - t[8];
           n_sqrt = 0.5 * rsqrtf(n);
           q.x   = (t[1] + t[3]) * n_sqrt;
           q.y   = n * n_sqrt;
           q.z   = (t[5] + t[7]) * n_sqrt;
           q.w   = -1 * (t[6] - t[2]) * n_sqrt;
         }
       }
       else
       {
         if (t[0] < -1 * t[4])
         {
           n      = 1 - t[0] - t[4] + t[8];
           n_sqrt = 0.5 * rsqrtf(n);
           q.x   = (t[6] + t[2]) * n_sqrt;
           q.y   = (t[5] + t[7]) * n_sqrt;
           q.z   = n * n_sqrt;
           q.w   = -1 * (t[1] - t[3]) * n_sqrt;
         }
         else
         {
           n      = 1 + t[0] + t[4] + t[8];
           n_sqrt = 0.5 * rsqrtf(n);
           q.x   = (t[5] - t[7]) * n_sqrt;
           q.y   = (t[6] - t[2]) * n_sqrt;
           q.z   = (t[1] - t[3]) * n_sqrt;
           q.w   = -1 * n * n_sqrt;
         }
       }
       q = quaternion_normalize(q);
     }

    /**
     * @brief get quaternion from transformation matrix
     *
     * @param t transformation matrix 3x4. Assumption is that this array has 16 byte alignment.
     * @param q quaternion in wxyz format
     */
     __device__ __forceinline__ void quaternion_from_transform_matrix(const float *t, float4 &q)
     {
       float rot_mat[9] = {0.0f};

       // t is a 3x4 matrix. We only read the first 3 columns
       #pragma unroll
       for (int i = 0; i < 3; i++)
       {
         *(float3 *)&rot_mat[i*3] = *(float3 *)&t[i*4];
       }
       quaternion_from_rotation_matrix(rot_mat, q);
     }

} // namespace common
} // namespace curobo
