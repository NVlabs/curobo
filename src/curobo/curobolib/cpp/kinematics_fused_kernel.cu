/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "helper_math.h"
#include "check_cuda.h"
#include <vector>

#define M 4

#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5

#define MAX_BATCH_PER_BLOCK 32    // tunable parameter for improving occupancy
#define MAX_BW_BATCH_PER_BLOCK 16 // tunable parameter for improving occupancy

#define MAX_TOTAL_LINKS \
  750                             // limited by shared memory size. We need to fit 16 * float32 per
                                  // link

namespace Curobo
{
  namespace Kinematics
  {


    template<typename psum_t>
    __device__ __forceinline__ void scale_cross_sum(float3 a, float3 b,
                                                    float3 scale, psum_t& sum_out)
    {
      sum_out += scale.x * (a.y * b.z - a.z * b.y) +
                 scale.y * (a.z * b.x - a.x * b.z) +
                 scale.z * (a.x * b.y - a.y * b.x);
    }

    __device__ __forceinline__ void normalize_quaternion(float *q)
    {
      // get length:
      float length = 1.0 / norm4df(q[0], q[1], q[2], q[3]);

      if (q[0] < 0.0)
      {
        length = -1.0 * length;
      }

      q[1] = length * q[1];
      q[2] = length * q[2];
      q[3] = length * q[3];
      q[0] = length * q[0];
    }


    __device__ __forceinline__ void normalize_quaternion(float4 &q)
    {
      // get length:
      float inv_length = 1.0 / length(q);

      if (q.w < 0.0)
      {
        inv_length = -1.0 * inv_length;
      }
      q = inv_length * q;

    }

    /**
     * @brief get quaternion from transformation matrix
     *
     * @param t transformation matrix 4x4
     * @param q quaternion in wxyz format
     */
    __device__ __forceinline__ void mat_to_quat(float *t, float *q)
    {
      float n;
      float n_sqrt;

      if (t[10] < 0.0)
      {
        if (t[0] > t[5])
        {
          n      = 1 + t[0] - t[5] - t[10];
          n_sqrt = 0.5 * rsqrtf(n);
          q[1]   = n * n_sqrt;
          q[2]   = (t[1] + t[4]) * n_sqrt;
          q[3]   = (t[8] + t[2]) * n_sqrt;
          q[0]   = -1 * (t[6] - t[9]) * n_sqrt; // * -1 ; // this is the wrong one?
        }
        else
        {
          n      = 1 - t[0] + t[5] - t[10];
          n_sqrt = 0.5 * rsqrtf(n);
          q[1]   = (t[1] + t[4]) * n_sqrt;
          q[2]   = n * n_sqrt;
          q[3]   = (t[6] + t[9]) * n_sqrt;
          q[0]   = -1 * (t[8] - t[2]) * n_sqrt;
        }
      }
      else
      {
        if (t[0] < -1 * t[5])
        {
          n      = 1 - t[0] - t[5] + t[10];
          n_sqrt = 0.5 * rsqrtf(n);
          q[1]   = (t[8] + t[2]) * n_sqrt;
          q[2]   = (t[6] + t[9]) * n_sqrt;
          q[3]   = n * n_sqrt;
          q[0]   = -1 * (t[1] - t[4]) * n_sqrt;
        }
        else
        {
          n      = 1 + t[0] + t[5] + t[10];
          n_sqrt = 0.5 * rsqrtf(n);
          q[1]   = (t[6] - t[9]) * n_sqrt;
          q[2]   = (t[8] - t[2]) * n_sqrt;
          q[3]   = (t[1] - t[4]) * n_sqrt;
          q[0]   = -1 * n * n_sqrt;
        }
      }
      normalize_quaternion(q);
    }

    /**
     * @brief get quaternion from transformation matrix
     *
     * @param t # rotation matrix 3x3
     * @param q quaternion in wxyz format
     */
    __device__ __forceinline__ void rot_to_quat(float *t, float4 &q)
    {
      // q.x = w, q.y = x, q.z = y,  q.w = z,
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
      normalize_quaternion(q);
    }

    __device__ __forceinline__ void rot_mul(float *r1, float *r2, float *r_out)
    {
      for (int i = 0; i < 9; i++)
      {
        r_out[i] = 0.0;
      }
#pragma unroll

      for (int k = 0; k < 3; k++)
      {
#pragma unroll

        for (int j = 0; j < 3; j++)
        {
#pragma unroll

          for (int i = 0; i < 3; i++)
          {
            r_out[i * 3 + j] += r1[i * 3 + k] * r2[k * 3 + j];
          }
        }
      }
    }

    __device__ __forceinline__ void rot_inverse_rot_mul(float *r1, float *r2,
                                                        float *r_out)
    {
      // multiply two matrices:
      r_out[0] = r1[0] * r2[0] + r1[4] * r2[4] + r1[8] * r2[8];
      r_out[1] = r1[0] * r2[1] + r1[4] * r2[5] + r1[8] * r2[9];
      r_out[2] = r1[0] * r2[2] + r1[4] * r2[6] + r1[8] * r2[10];

      r_out[3] = r1[1] * r2[0] + r1[5] * r2[4] + r1[9] * r2[8];
      r_out[4] = r1[1] * r2[1] + r1[5] * r2[5] + r1[9] * r2[9];
      r_out[5] = r1[1] * r2[2] + r1[5] * r2[6] + r1[9] * r2[10];

      r_out[6] = r1[2] * r2[0] + r1[6] * r2[4] + r1[10] * r2[8];
      r_out[7] = r1[2] * r2[1] + r1[6] * r2[5] + r1[10] * r2[9];
      r_out[8] = r1[2] * r2[2] + r1[6] * r2[6] + r1[10] * r2[10];
    }

    template<typename scalar_t>
    __device__ __forceinline__ void
    transform_sphere(const float *transform_mat, const scalar_t *sphere, float *C)
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

    template<typename scalar_t>
    __device__ __forceinline__ void
    transform_sphere_float4(const float *transform_mat, const scalar_t *sphere, float4 &C)
    {
      float4 sphere_pos = *(float4 *)&sphere[0];
      int    st_idx     = 0;

      C.x = transform_mat[0] * sphere_pos.x + transform_mat[1] * sphere_pos.y +
            transform_mat[2] * sphere_pos.z + transform_mat[3];
      C.y = transform_mat[4] * sphere_pos.x + transform_mat[5] * sphere_pos.y +
            transform_mat[6] * sphere_pos.z + transform_mat[7];
      C.z = transform_mat[8] * sphere_pos.x + transform_mat[9] * sphere_pos.y +
            transform_mat[10] * sphere_pos.z + transform_mat[11];
      C.w = sphere_pos.w;

    }

    template<typename scalar_t>
    __device__ __forceinline__ void fixed_joint_fn(const scalar_t *fixedTransform,
                                                   float          *JM)
    {
      JM[0] = fixedTransform[0];
      JM[1] = fixedTransform[M];
      JM[2] = fixedTransform[M * 2];
      JM[3] = fixedTransform[M * 3];
    }

    // prism_fn withOUT control flow
    template<typename scalar_t>
    __device__ __forceinline__ void prism_fn(const scalar_t *fixedTransform,
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
        fixed_joint_fn(&fixedTransform[col_idx], &JM[0]);
      }
      else
      {
        JM[0] = fixedTransform[0 + xyz] * angle + fixedTransform[3];     // FT_0[1];
        JM[1] = fixedTransform[M + xyz] * angle + fixedTransform[M + 3]; // FT_1[1];
        JM[2] = fixedTransform[M + M + xyz] * angle +
                fixedTransform[M + M + 3];                               // FT_2[1];
        JM[3] = 1;
      }
    }


    __device__ __forceinline__ void update_axis_direction(
      float& angle,
      int  & j_type,
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

    // In the following versions of rot_fn, some non-nan values may become nan as we
    // add multiple values instead of using if-else/switch-case.

    // version with no control flow
    template<typename scalar_t>
    __device__ __forceinline__ void xrot_fn(const scalar_t *fixedTransform,
                                            const float angle, const int col_idx,
                                            float *JM)
    {
      // we found no change in convergence between fast approximate and IEEE sin,
      // cos functions using fast approximate method saves 5 registers per thread.
      float cos   = __cosf(angle);
      float sin   = __sinf(angle);
      float n_sin = -1 * sin;

      int bit1         = col_idx & 0x1;
      int bit2         = (col_idx & 0x2) >> 1;
      int _xor         = bit1 ^ bit2;  // 0 for threads 0 and 3, 1 for threads 1 and 2
      int col_idx_by_2 =
        col_idx / 2;                   // 0 for threads 0 and 1, 1 for threads 2 and 3

      float f1 = (1 - col_idx_by_2) * cos +
                 col_idx_by_2 * n_sin; // thread 1 get cos , thread 2 gets n_sin
      float f2 = (1 - col_idx_by_2) * sin +
                 col_idx_by_2 * cos;   // thread 1 get sin, thread 2 gets cos

      f1 = _xor * f1 + (1 - _xor) * 1; // threads 1 and 2 will get f1; the other
                                       // two threads will get 1
      f2 = _xor *
           f2;                         // threads 1 and 2 will get f2, the other two threads will
                                       // get 0.0
      float f3 = 1 - _xor;

      int addr_offset =
        _xor + (1 - _xor) *
        col_idx; // 1 for threads 1 and 2, col_idx for threads 0 and 3

      JM[0] = fixedTransform[0 + addr_offset] * f1 + f2 * fixedTransform[2];
      JM[1] = fixedTransform[M + addr_offset] * f1 + f2 * fixedTransform[M + 2];
      JM[2] =
        fixedTransform[M + M + addr_offset] * f1 + f2 * fixedTransform[M + M + 2];
      JM[3] = fixedTransform[M + M + M + addr_offset] *
              f3; // threads 1 and 2 get 0.0, remaining two get fixedTransform[3M];
    }

    // version with no control flow
    template<typename scalar_t>
    __device__ __forceinline__ void yrot_fn(const scalar_t *fixedTransform,
                                            const float angle, const int col_idx,
                                            float *JM)
    {
      float cos   = __cosf(angle);
      float sin   = __sinf(angle);
      float n_sin = -1 * sin;

      int col_idx_per_2 =
        col_idx % 2;                 // threads 0 and 2 will be 0 and threads 1 and 3 will be 1.
      int col_idx_by_2 =
        col_idx / 2;                 // threads 0 and 1 will be 0 and threads 2 and 3 will be 1.

      float f1 = (1 - col_idx_by_2) * cos +
                 col_idx_by_2 * sin; // thread 0 get cos , thread 2 gets sin
      float f2 = (1 - col_idx_by_2) * n_sin +
                 col_idx_by_2 * cos; // thread 0 get n_sin, thread 2 gets cos

      f1 = (1 - col_idx_per_2) * f1 +
           col_idx_per_2 * 1;        // threads 0 and 2 will get f1; the other two
                                     // threads will get 1
      f2 = (1 - col_idx_per_2) *
           f2;                       // threads 0 and 2 will get f2, the other two threads will get
                                     // 0.0
      float f3 =
        col_idx_per_2;               // threads 0 and 2 will be 0 and threads 1 and 3 will be 1.

      int addr_offset =
        col_idx_per_2 *
        col_idx; // threads 0 and 2 will get 0, the other two will get col_idx.

      JM[0] = fixedTransform[0 + addr_offset] * f1 + f2 * fixedTransform[2];
      JM[1] = fixedTransform[M + addr_offset] * f1 + f2 * fixedTransform[M + 2];
      JM[2] =
        fixedTransform[M + M + addr_offset] * f1 + f2 * fixedTransform[M + M + 2];
      JM[3] = fixedTransform[M + M + M + addr_offset] *
              f3; // threads 0 and 2 threads get 0.0, remaining two get
                  // fixedTransform[3M];
    }

    // version with no control flow
    template<typename scalar_t>
    __device__ __forceinline__ void zrot_fn(const scalar_t *fixedTransform,
                                            const float angle, const int col_idx,
                                            float *JM)
    {
      float cos   = __cosf(angle);
      float sin   = __sinf(angle);
      float n_sin = -1 * sin;

      int col_idx_by_2 =
        col_idx / 2;                    // first two threads will be 0 and the next two will be 1.
      int col_idx_per_2 =
        col_idx % 2;                    // first thread will be 0 and the second thread will be 1.
      float f1 = (1 - col_idx_per_2) * cos +
                 col_idx_per_2 * n_sin; // thread 0 get cos , thread 1 gets n_sin
      float f2 = (1 - col_idx_per_2) * sin +
                 col_idx_per_2 * cos;   // thread 0 get sin, thread 1 gets cos

      f1 = (1 - col_idx_by_2) * f1 +
           col_idx_by_2 * 1;            // first two threads get f1, other two threads get 1
      f2 = (1 - col_idx_by_2) *
           f2;                          // first two threads get f2, other two threads get 0.0

      int addr_offset =
        col_idx_by_2 *
        col_idx; // first 2 threads will get 0, the other two will get col_idx.

      JM[0] = fixedTransform[0 + addr_offset] * f1 + f2 * fixedTransform[1];
      JM[1] = fixedTransform[M + addr_offset] * f1 + f2 * fixedTransform[M + 1];
      JM[2] =
        fixedTransform[M + M + addr_offset] * f1 + f2 * fixedTransform[M + M + 1];
      JM[3] = fixedTransform[M + M + M + addr_offset] *
              col_idx_by_2; // first two threads get 0.0, remaining two get
                            // fixedTransform[3M];
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    rot_backward_translation(const float3& vec, float *cumul_mat, float *l_pos,
                             const float3& loc_grad, psum_t& grad_q, const float axis_sign = 1)
    {
      float3 e_pos, j_pos;

      e_pos.x = cumul_mat[3];
      e_pos.y = cumul_mat[4 + 3];
      e_pos.z = cumul_mat[4 + 4 + 3];

      // compute position gradient:
      j_pos = make_float3(l_pos[0], l_pos[1], l_pos[2]) - e_pos;            // - e_pos;
      float3 scale_grad = axis_sign * loc_grad;
      scale_cross_sum(vec, j_pos, scale_grad, grad_q); // cross product
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    rot_backward_rotation(const float3 vec,
                          const float3 grad_vec,
                          psum_t     & grad_q,
                          const float axis_sign = 1)
    {
      grad_q += axis_sign * dot(vec, grad_vec);
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    prism_backward_translation(const float3 vec, const float3 grad_vec,
                               psum_t& grad_q, const float axis_sign = 1)
    {
      grad_q += axis_sign * dot(vec, grad_vec);
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    z_rot_backward(float *link_cumul_mat, float *l_pos, float3& loc_grad_position,
                   float3& loc_grad_orientation, psum_t& grad_q, const float axis_sign = 1)
    {
      float3 vec =
        make_float3(link_cumul_mat[2], link_cumul_mat[6], link_cumul_mat[10]);

      // get rotation vector:
      rot_backward_translation(vec, &link_cumul_mat[0], &l_pos[0],
                               loc_grad_position, grad_q, axis_sign);

      rot_backward_rotation(vec, loc_grad_orientation, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    x_rot_backward(float *link_cumul_mat, float *l_pos, float3& loc_grad_position,
                   float3& loc_grad_orientation, psum_t& grad_q,  const float axis_sign = 1)
    {
      float3 vec =
        make_float3(link_cumul_mat[0], link_cumul_mat[4], link_cumul_mat[8]);

      // get rotation vector:
      rot_backward_translation(vec, &link_cumul_mat[0], &l_pos[0],
                               loc_grad_position, grad_q, axis_sign);

      rot_backward_rotation(vec, loc_grad_orientation, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    y_rot_backward(float *link_cumul_mat, float *l_pos, float3& loc_grad_position,
                   float3& loc_grad_orientation, psum_t& grad_q,  const float axis_sign = 1)
    {
      float3 vec =
        make_float3(link_cumul_mat[1], link_cumul_mat[5], link_cumul_mat[9]);

      // get rotation vector:
      rot_backward_translation(vec, &link_cumul_mat[0], &l_pos[0],
                               loc_grad_position, grad_q, axis_sign);

      rot_backward_rotation(vec, loc_grad_orientation, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    xyz_prism_backward_translation(float *cumul_mat, float3& loc_grad,
                                   psum_t& grad_q, int xyz,  const float axis_sign = 1)
    {
      prism_backward_translation(
        make_float3(cumul_mat[0 + xyz], cumul_mat[4 + xyz], cumul_mat[8 + xyz]),
        loc_grad, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void x_prism_backward_translation(float       *cumul_mat,
                                                                 float3     & loc_grad,
                                                                 psum_t     & grad_q,
                                                                 const float axis_sign = 1)
    {
      // get rotation vector:
      prism_backward_translation(
        make_float3(cumul_mat[0], cumul_mat[4], cumul_mat[8]), loc_grad, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void y_prism_backward_translation(float       *cumul_mat,
                                                                 float3     & loc_grad,
                                                                 psum_t     & grad_q,
                                                                 const float axis_sign = 1)
    {
      // get rotation vector:
      prism_backward_translation(
        make_float3(cumul_mat[1], cumul_mat[5], cumul_mat[9]), loc_grad, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void z_prism_backward_translation(float       *cumul_mat,
                                                                 float3     & loc_grad,
                                                                 psum_t     & grad_q,
                                                                 const float axis_sign = 1)
    {
      // get rotation vector:
      prism_backward_translation(
        make_float3(cumul_mat[2], cumul_mat[6], cumul_mat[10]), loc_grad, grad_q, axis_sign);
    }
    __device__ __forceinline__ void
    xyz_rot_backward_translation(float *cumul_mat, float *l_pos, float3& loc_grad,
                                 float& grad_q, int xyz,  const float axis_sign = 1)
    {
      // get rotation vector:
      rot_backward_translation(
        make_float3(cumul_mat[0 + xyz], cumul_mat[4 + xyz], cumul_mat[8 + xyz]),
        &cumul_mat[0], &l_pos[0], loc_grad, grad_q, axis_sign);
    }


    template<typename psum_t>
    __device__ __forceinline__ void
    x_rot_backward_translation(float *cumul_mat, float *l_pos, float3& loc_grad,
                               psum_t& grad_q,  const float axis_sign = 1)
    {
      // get rotation vector:
      rot_backward_translation(
        make_float3(cumul_mat[0], cumul_mat[4], cumul_mat[8]), &cumul_mat[0],
        &l_pos[0], loc_grad, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    y_rot_backward_translation(float *cumul_mat, float *l_pos, float3& loc_grad,
                               psum_t& grad_q,  const float axis_sign = 1)
    {
      // get rotation vector:
      rot_backward_translation(
        make_float3(cumul_mat[1], cumul_mat[5], cumul_mat[9]), &cumul_mat[0],
        &l_pos[0], loc_grad, grad_q, axis_sign);
    }

    template<typename psum_t>
    __device__ __forceinline__ void
    z_rot_backward_translation(float *cumul_mat, float *l_pos, float3& loc_grad,
                               psum_t& grad_q,  const float axis_sign = 1)
    {
      // get rotation vector:
      rot_backward_translation(
        make_float3(cumul_mat[2], cumul_mat[6], cumul_mat[10]), &cumul_mat[0],
        &l_pos[0], loc_grad, grad_q, axis_sign);
    }

    // An optimized version of kin_fused_warp_kernel.
    // This one should be about 10% faster.
    template<typename scalar_t, bool use_global_cumul>
    __global__ void
    kin_fused_warp_kernel2(float *link_pos,             // batchSize xz store_n_links x M x M
                           float *link_quat,            // batchSize x store_n_links x M x M
                           scalar_t *b_robot_spheres,      // batchSize x nspheres x M
                           float *global_cumul_mat, // batchSize x nlinks x M x M
                           const float *q,              // batchSize x njoints
                           const float *fixedTransform, // nlinks x M x M
                           const float *robot_spheres,  // nspheres x M
                           const int8_t *jointMapType,     // nlinks
                           const int16_t *jointMap,        // nlinks
                           const int16_t *linkMap,         // nlinks
                           const int16_t *storeLinkMap,    // store_n_links
                           const int16_t *linkSphereMap,   // nspheres
                           const float *jointOffset, // nlinks
                           const int batchSize, const int nspheres,
                           const int nlinks, const int njoints,
                           const int store_n_links)
    {
      extern __shared__ float cumul_mat[];

      int t           = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / 4;

      if (batch >= batchSize)
        return;

      int col_idx           = threadIdx.x % 4;
      const int local_batch = threadIdx.x / 4;
      const int matAddrBase = local_batch * nlinks * M * M;

      // read all fixed transforms to local cache:

      // copy base link transform:
      *(float4 *)&cumul_mat[matAddrBase + col_idx * M] =
        *(float4 *)&fixedTransform[col_idx * M];

      if (use_global_cumul)
      {
        *(float4 *)&global_cumul_mat[batch * nlinks * 16 + col_idx * M] =
          *(float4 *)&cumul_mat[matAddrBase + col_idx * M];
      }

      for (int8_t l = 1; l < nlinks; l++) //
      {
        // get one row of fixedTransform
        int ftAddrStart  = l * M * M;
        int inAddrStart  = matAddrBase + linkMap[l] * M * M;
        int outAddrStart = matAddrBase + l * M * M;

        // row index:
        // check joint type and use one of the helper functions:
        float JM[M];
        int   j_type = jointMapType[l];

        if (j_type == FIXED)
        {
          fixed_joint_fn(&fixedTransform[ftAddrStart + col_idx], &JM[0]);
        }
        else
        {
          float angle = q[batch * njoints + jointMap[l]];
          float2 angle_offset = *(float2 *)&jointOffset[l*2];
          update_axis_direction(angle, j_type, angle_offset);

          if (j_type <= Z_PRISM)
          {
            prism_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0], j_type);
          }
          else if (j_type == X_ROT)
          {
            xrot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0]);
          }
          else if (j_type == Y_ROT)
          {
            yrot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0]);
          }
          else if (j_type == Z_ROT)
          {
            zrot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0]);
          }
          else
          {
            assert(j_type >= FIXED && j_type <= Z_ROT);
          }
        }

#pragma unroll 4

        for (int i = 0; i < M; i++)
        {
          cumul_mat[outAddrStart + (i * M) + col_idx] =
            dot(*(float4 *)&cumul_mat[inAddrStart + (i * M)], make_float4(JM[0], JM[1], JM[2], JM[3]));
        }

        if (use_global_cumul)
        {
          *(float4 *)&global_cumul_mat[batch * nlinks * 16 + l * 16 + col_idx * M] =
            *(float4 *)&cumul_mat[outAddrStart + col_idx * M];
        }
      }

      // write out link:

      // do robot_spheres

      const int batchAddrs = batch * nspheres * 4;

      // read cumul mat index to run for this thread:
      int16_t read_cumul_idx    = -1;
      int16_t spheres_perthread = (nspheres + 3) / 4;

      for (int16_t i = 0; i < spheres_perthread; i++)
      {
        // const int16_t sph_idx = col_idx * spheres_perthread + i;
        const int16_t sph_idx = col_idx + i * 4;

        // const int8_t sph_idx =
        //     i * 4 + col_idx; // different order such that adjacent
        //  spheres are in neighboring threads
        if (sph_idx >= nspheres)
        {
          break;
        }

        // read cumul idx:
        read_cumul_idx = linkSphereMap[sph_idx];
        float4 spheres_mem = make_float4(0.0, 0.0, 0.0, 0.0);
        const int16_t sphAddrs = sph_idx * 4;

        transform_sphere_float4(&cumul_mat[matAddrBase + (read_cumul_idx * 16)],
                         &robot_spheres[sphAddrs], spheres_mem);

        //b_robot_spheres[batchAddrs + sphAddrs] = spheres_mem[0];
        //b_robot_spheres[batchAddrs + sphAddrs + 1] = spheres_mem[1];
        //b_robot_spheres[batchAddrs + sphAddrs + 2] = spheres_mem[2];
        //b_robot_spheres[batchAddrs + sphAddrs + 3] = spheres_mem[3];

        //float4 test_sphere =  *(float4 *)&spheres_mem[0];// make_float4(spheres_mem[0],spheres_mem[1],spheres_mem[2],spheres_mem[3]);

        *(float4 *)&b_robot_spheres[batchAddrs + sphAddrs] = spheres_mem;

      }

      // write position and rotation, we convert rotation matrix to a quaternion and
      // write it out
      for (int16_t i = 0; i < store_n_links; i++)
      {
        int16_t l_map          = storeLinkMap[i];
        int     l_outAddrStart =
          (batch * store_n_links); // * 7) + i * 7;// + (t % M) * M;
        int outAddrStart = matAddrBase + l_map * M * M;

        float quat[4];

        // TODO: spread the work to different threads. For now all the threads will
        // do the same work.
        mat_to_quat(
          &cumul_mat[outAddrStart],
          &quat[0]);     // get quaternion, all the 4 threads will do the same work
        link_quat[l_outAddrStart * 4 + i * 4 + col_idx] =
          quat[col_idx]; // one thread will write one element to memory

        if (col_idx < 3)
        {
          // threads 0,1,2 will execute the following store
          link_pos[l_outAddrStart * 3 + i * 3 + col_idx] =
            cumul_mat[outAddrStart + 3 + (col_idx) * 4];
        }
      }
    }

    // kin_fused_backward_kernel3 uses 16 threads per batch, instead of 4 per batch
    // as in kin_fused_backward_kernel2.
    template<typename scalar_t, typename psum_t, bool use_global_cumul,
             bool enable_sparsity_opt, int16_t MAX_JOINTS, bool PARALLEL_WRITE>
    __global__ void kin_fused_backward_kernel3(
      float *grad_out_link_q,       // batchSize * njoints
      const float *grad_nlinks_pos, // batchSize * store_n_links * 16
      const float *grad_nlinks_quat,
      const scalar_t *grad_spheres,    // batchSize * nspheres * 4
      const float *global_cumul_mat,
      const float *q,               // batchSize * njoints
      const float *fixedTransform,  // nlinks * 16
      const float *robotSpheres,    // batchSize * nspheres * 4
      const int8_t *jointMapType,      // nlinks
      const int16_t *jointMap,         // nlinks
      const int16_t *linkMap,          // nlinks
      const int16_t *storeLinkMap,     // store_n_links
      const int16_t *linkSphereMap,    // nspheres
      const int16_t *linkChainMap,     // nlinks*nlinks
      const float *jointOffset,        // nlinks*2
      const int batchSize, const int nspheres, const int nlinks,
      const int njoints, const int store_n_links)
    {
      extern __shared__ float cumul_mat[];

      int t           = blockDim.x * blockIdx.x + threadIdx.x;
      const int batch = t / 16;
      unsigned  mask  = __ballot_sync(0xffffffff, batch < batchSize);

      if (batch >= batchSize)
        return;
        // Each thread computes one element of the cumul_mat.
      // first 4 threads compute a row of the output;
      const int elem_idx    = threadIdx.x % 16;
      const int col_idx     = elem_idx % 4;
      const int local_batch = threadIdx.x / 16;
      const int matAddrBase = local_batch * nlinks * M * M;

      if (use_global_cumul)
      {
        for (int l = 0; l < nlinks; l++)
        {
          int outAddrStart = matAddrBase + l * M * M; // + (t % M) * M;

          cumul_mat[outAddrStart + elem_idx] =
            global_cumul_mat[batch * nlinks * M * M + l * M * M + elem_idx];
        }
      }
      else
      {
        cumul_mat[matAddrBase + elem_idx] = fixedTransform[elem_idx];

        for (int l = 1; l < nlinks; l++)                // TODO: add base link transform
        {
          float JM[M];                                  // store one row locally for mat-mul
          int   ftAddrStart  = l * M * M;               // + (t % M) * M;
          int   inAddrStart  = matAddrBase + linkMap[l] * M * M;
          int   outAddrStart = matAddrBase + l * M * M; // + (t % M) * M;

          int j_type = jointMapType[l];


          if (j_type == FIXED)
          {
            fixed_joint_fn(&fixedTransform[ftAddrStart + col_idx], &JM[0]);
          }
          else
          {
            float angle = q[batch * njoints + jointMap[l]];
            float2 angle_offset = *(float2 *)&jointOffset[l*2];
            update_axis_direction(angle, j_type, angle_offset);

            if (j_type <= Z_PRISM)
            {
              prism_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0], j_type);
            }
            else if (j_type == X_ROT)
            {
              xrot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0]);
            }
            else if (j_type == Y_ROT)
            {
              yrot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0]);
            }
            else if (j_type == Z_ROT)
            {
              zrot_fn(&fixedTransform[ftAddrStart], angle, col_idx, &JM[0]);
            }
            else
            {
              assert(j_type >= FIXED && j_type <= Z_ROT);
            }
          }

          // fetch one row of cumul_mat, multiply with a column, which is in JM
          cumul_mat[outAddrStart + elem_idx] =
            dot(*(float4 *)&cumul_mat[inAddrStart + ((elem_idx / 4) * M)],
                make_float4(JM[0], JM[1], JM[2], JM[3]));
        }
      }

      // thread-local partial sum accumulators
      // We would like to keep these partial sums in register file and avoid memory
      // accesses
      psum_t psum_grad[MAX_JOINTS]; // MAX_JOINTS
      // we are allocating a lot larger array. So, we will be initilizing just the
      // portion we need.
#pragma unroll

      for (int i = 0; i < njoints; i++)
      {
        psum_grad[i] = 0.0;
      }

      // read cumul mat index to run for this thread:
      int read_cumul_idx = -1;

      const int spheres_perthread = (nspheres + 15) / 16;

      for (int i = 0; i < spheres_perthread; i++)
      {
        // const int sph_idx = elem_idx * spheres_perthread + i;
        const int sph_idx = elem_idx + i * 16;

        if (sph_idx >= nspheres)
        {
          break;
        }
        const int sphAddrs          = sph_idx * 4;
        const int batchAddrs        = batch * nspheres * 4;
        float4    loc_grad_sphere_t = *(float4 *)&grad_spheres[batchAddrs + sphAddrs];

        // Sparsity-based optimization: Skip zero computation
        if (enable_sparsity_opt)
        {
          if ((loc_grad_sphere_t.x == 0) && (loc_grad_sphere_t.y == 0) &&
              (loc_grad_sphere_t.z == 0))
          {
            continue;
          }
        }
        float3 loc_grad_sphere = make_float3(
          loc_grad_sphere_t.x, loc_grad_sphere_t.y, loc_grad_sphere_t.z);

        // read cumul idx:
        read_cumul_idx = linkSphereMap[sph_idx];
        float spheres_mem[4] = {0.0,0.0,0.0,0.0};
        transform_sphere(&cumul_mat[matAddrBase + read_cumul_idx * 16],
                         &robotSpheres[sphAddrs], &spheres_mem[0]);

        // assuming this sphere only depends on links lower than this index
        // This could be relaxed by making read_cumul_idx = number of links.
        // const int16_t loop_max = read_cumul_idx;
        const int16_t loop_max = nlinks - 1;

        for (int j = loop_max; j > -1; j--)
        {
          if (linkChainMap[read_cumul_idx * nlinks + j] == 0.0)
          {
            continue;
          }
          float axis_sign = jointOffset[j*2];

          int j_type = jointMapType[j];


          if (j_type == Z_ROT)
          {
            float result = 0.0;
            z_rot_backward_translation(&cumul_mat[matAddrBase + j * 16],
                                       &spheres_mem[0], loc_grad_sphere, result, axis_sign);
            psum_grad[jointMap[j]] += (psum_t)result;
          }
          else if ((j_type >= X_PRISM) && (j_type <= Z_PRISM))
          {
            float result = 0.0;
            xyz_prism_backward_translation(&cumul_mat[matAddrBase + j * 16],
                                           loc_grad_sphere, result, j_type, axis_sign);
            psum_grad[jointMap[j]] += (psum_t)result;
          }
          else if (j_type == X_ROT)
          {
            float result = 0.0;
            x_rot_backward_translation(&cumul_mat[matAddrBase + j * 16],
                                       &spheres_mem[0], loc_grad_sphere, result, axis_sign);
            psum_grad[jointMap[j]] += (psum_t)result;
          }
          else if (j_type == Y_ROT)
          {
            float result = 0.0;
            y_rot_backward_translation(&cumul_mat[matAddrBase + j * 16],
                                       &spheres_mem[0], loc_grad_sphere, result, axis_sign);
            psum_grad[jointMap[j]] += (psum_t)result;
          }
        }
      }


      // Instead of accumulating the sphere_grad and link_grad separately, we will
      // accumulate them together once below.
      //
      // // accumulate across 4 threads using shuffle operation
      // for(int j=0; j<njoints; j++)
      // {
      //   psum_t sum = psum_grad[j];
      //   sum += __shfl_down_sync(mask, sum, 1);
      //   sum += __shfl_down_sync(mask, sum, 2);

      //   // thread 0 (col_idx==0) has the accumulated psum_grad[j] in sum
      //   if (col_idx == 0) {
      //     loc_grad_q[gqAddrBase + j] = (psum_t)sum;
      //   }
      // }

      // compute jacobian for all links in linkmat:

      // write out link:
      //
      for (int16_t i = 0; i < store_n_links; i++)
      {
        const int batchAddrs      = batch * store_n_links;
        float3    g_position      = *(float3 *)&grad_nlinks_pos[batchAddrs * 3 + i * 3];
        float4    g_orientation_t =
          *(float4 *)&grad_nlinks_quat[batchAddrs * 4 + i * 4];

        // sparisty check here:
        if (enable_sparsity_opt)
        {
          if ((g_position.x == 0) && (g_position.y == 0) && (g_position.z == 0) &&
              (g_orientation_t.y == 0) && (g_orientation_t.z == 0) &&
              (g_orientation_t.w == 0))
          {
            continue;
          }
        }
        float3 g_orientation =
          make_float3(g_orientation_t.y, g_orientation_t.z, g_orientation_t.w);

        const int16_t l_map = storeLinkMap[i];


        float l_pos[3];
        const int outAddrStart = matAddrBase + l_map * M * M;
        l_pos[0] = cumul_mat[outAddrStart + 3]; // read position of stored link
        l_pos[1] = cumul_mat[outAddrStart + M + 3];
        l_pos[2] = cumul_mat[outAddrStart + M * 2 + 3];

        const int16_t max_lmap = nlinks - 1;

        const int16_t joints_per_thread = (max_lmap + 15) / 16;

        // for (int16_t k = joints_per_thread; k >= 0; k--)
        for (int16_t k = 0; k < joints_per_thread; k++)
        {
          int16_t j = elem_idx * joints_per_thread + k;
          //int16_t j = elem_idx + k * 16;
          // int16_t j = elem_idx + k * 16; // (threadidx.x % 16) + k * 16 (0 to 16)

          // int16_t j = k * M + elem_idx;
          if ((j > max_lmap))
            break;

          // This can be spread across threads as they are not sequential?
          if (linkChainMap[l_map * nlinks + j] == 0.0)
          {
            continue;
          }
          int16_t j_idx  = jointMap[j];
          int     j_type = jointMapType[j];

          float axis_sign = jointOffset[j*2];


          // get rotation vector:
          if (j_type == Z_ROT)
          {
            z_rot_backward(&cumul_mat[matAddrBase + (j) * M * M], &l_pos[0],
                           g_position, g_orientation, psum_grad[j_idx], axis_sign);
          }
          else if (j_type >= X_PRISM & j_type <= Z_PRISM)
          {
            xyz_prism_backward_translation(&cumul_mat[matAddrBase + j * 16],
                                           g_position, psum_grad[j_idx], j_type, axis_sign);
          }
          else if (j_type == X_ROT)
          {
            x_rot_backward(&cumul_mat[matAddrBase + (j) * M * M], &l_pos[0],
                           g_position, g_orientation, psum_grad[j_idx], axis_sign);
          }
          else if (j_type == Y_ROT)
          {
            y_rot_backward(&cumul_mat[matAddrBase + (j) * M * M], &l_pos[0],
                           g_position, g_orientation, psum_grad[j_idx], axis_sign);
          }
        }

      }
      __syncthreads();
      if (PARALLEL_WRITE)
      {
        // accumulate the partial sums across the 16 threads
#pragma unroll

        for (int16_t j = 0; j < njoints; j++)
        {
          psum_grad[j] += __shfl_xor_sync(mask, psum_grad[j], 1);
          psum_grad[j] += __shfl_xor_sync(mask, psum_grad[j], 2);
          psum_grad[j] += __shfl_xor_sync(mask, psum_grad[j], 4);
          psum_grad[j] += __shfl_xor_sync(mask, psum_grad[j], 8);

          // thread 0: psum_grad[j] will have the sum across 16 threads
          // write out using only thread 0
        }

        const int16_t joints_per_thread = (njoints + 15) / 16;

#pragma unroll

        for (int16_t j = 0; j < joints_per_thread; j++)
        {
          const int16_t j_idx = elem_idx * joints_per_thread + j;
          //const int16_t j_idx = elem_idx + j * 16;

          if (j_idx >= njoints)
          {
            break;
          }
          grad_out_link_q[batch * njoints + j_idx] =
            psum_grad[j_idx]; // write the sum to memory
        }
      }
      else
      {
#pragma unroll

        for (int16_t j = 0; j < njoints; j++)
        {
          psum_grad[j] += __shfl_down_sync(mask, psum_grad[j], 1);
          psum_grad[j] += __shfl_down_sync(mask, psum_grad[j], 2);
          psum_grad[j] += __shfl_down_sync(mask, psum_grad[j], 4);
          psum_grad[j] += __shfl_down_sync(mask, psum_grad[j], 8);

          // thread 0: psum_grad[j] will have the sum across 16 threads
          // write out using only thread 0
        }

        if (elem_idx > 0)
        {
          return;
        }

#pragma unroll

        for (int16_t j = 0; j < njoints; j++)
        {
          {
            grad_out_link_q[batch * njoints + j] =
              (float) psum_grad[j]; // write the sum to memory
          }
        }
      }

      // accumulate the partial sums across the 16 threads
    }

    template<typename scalar_t>
    __global__ void mat_to_quat_kernel(scalar_t       *out_quat,
                                       const scalar_t *in_rot_mat,
                                       const int       batch_size)
    {
      // Only works for float32
      const int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

      if (batch_idx >= batch_size)
      {
        return;
      }
      //float q[4] = { 0.0 }; // initialize array
      float4 q = make_float4(0,0,0,0);
      float rot[9];

      // read rot
      #pragma unroll 9
      for (int k = 0; k<9; k++)
      {
        rot[k] = in_rot_mat[batch_idx * 9 + k];
      }

      // *(float3 *)&rot[0] = *(float3 *)&in_rot_mat[batch_idx * 9];
      // *(float3 *)&rot[3] = *(float3 *)&in_rot_mat[batch_idx * 9 + 3];
      // *(float3 *)&rot[6] = *(float3 *)&in_rot_mat[batch_idx * 9 + 6];

      rot_to_quat(&rot[0], q);

      // write quaternion:

      *(float4 *)&out_quat[batch_idx * 4] = q;
    }
  } // namespace Kinematics
}   // namespace Curobo

std::vector<torch::Tensor>kin_fused_forward(
  torch::Tensor link_pos, torch::Tensor link_quat,
  torch::Tensor batch_robot_spheres, torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec, const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres, const torch::Tensor link_map,
  const torch::Tensor joint_map, const torch::Tensor joint_map_type,
  const torch::Tensor store_link_map, const torch::Tensor link_sphere_map,
  const torch::Tensor joint_offset_map,
  const int batch_size,
  const int n_joints,const int n_spheres,
  const bool use_global_cumul = false)
{
  using namespace Curobo::Kinematics;
  CHECK_INPUT_GUARD(joint_vec);
  CHECK_INPUT(link_pos);
  CHECK_INPUT(link_quat);
  CHECK_INPUT(global_cumul_mat);
  CHECK_INPUT(batch_robot_spheres);
  CHECK_INPUT(fixed_transform);
  CHECK_INPUT(robot_spheres);
  CHECK_INPUT(link_map);
  CHECK_INPUT(joint_map);
  CHECK_INPUT(joint_map_type);
  CHECK_INPUT(store_link_map);
  CHECK_INPUT(link_sphere_map);
  CHECK_INPUT(joint_offset_map);

  const int n_links       = link_map.size(0);
  const int store_n_links = link_pos.size(1);
  assert(joint_map.dtype() == torch::kInt16);
  assert(joint_map_type.dtype() == torch::kInt8);
  assert(store_link_map.dtype() == torch::kInt16);
  assert(link_sphere_map.dtype() == torch::kInt16);
  assert(link_map.dtype() == torch::kInt16);
  assert(batch_size > 0);
  assert(n_links < MAX_TOTAL_LINKS);

  int batches_per_block = (int)((MAX_TOTAL_LINKS / n_links));

  if (batches_per_block == 0)
  {
    batches_per_block = 1;
  }

  if (batches_per_block > MAX_BATCH_PER_BLOCK)
  {
    batches_per_block = MAX_BATCH_PER_BLOCK;
  }

  if (batches_per_block * M > 1024)
  {
    batches_per_block = 1024 / (M);
  }

  // batches_per_block = 1;
  const int threadsPerBlock = batches_per_block * M;
  const int blocksPerGrid   =
    (batch_size * M + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMemSize = batches_per_block * n_links * M * M * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_global_cumul)
  {
    AT_DISPATCH_FLOATING_TYPES(
      batch_robot_spheres.scalar_type(), "kin_fused_forward", ([&] {
      kin_fused_warp_kernel2<scalar_t, true>
        << < blocksPerGrid, threadsPerBlock, sharedMemSize, stream >> > (
        link_pos.data_ptr<float>(), link_quat.data_ptr<float>(),
        batch_robot_spheres.data_ptr<scalar_t>(),
        global_cumul_mat.data_ptr<float>(),
        joint_vec.data_ptr<float>(),
        fixed_transform.data_ptr<float>(),
        robot_spheres.data_ptr<float>(),
        joint_map_type.data_ptr<int8_t>(),
        joint_map.data_ptr<int16_t>(), link_map.data_ptr<int16_t>(),
        store_link_map.data_ptr<int16_t>(),
        link_sphere_map.data_ptr<int16_t>(),
        joint_offset_map.data_ptr<float>(),
        batch_size, n_spheres,
        n_links, n_joints, store_n_links);
    }));
  }
  else
  {
    AT_DISPATCH_FLOATING_TYPES(
      batch_robot_spheres.scalar_type(), "kin_fused_forward", ([&] {
      kin_fused_warp_kernel2<scalar_t, false>
        << < blocksPerGrid, threadsPerBlock, sharedMemSize, stream >> > (
        link_pos.data_ptr<float>(), link_quat.data_ptr<float>(),
        batch_robot_spheres.data_ptr<scalar_t>(),
        global_cumul_mat.data_ptr<float>(),
        joint_vec.data_ptr<float>(),
        fixed_transform.data_ptr<float>(),
        robot_spheres.data_ptr<float>(),
        joint_map_type.data_ptr<int8_t>(),
        joint_map.data_ptr<int16_t>(), link_map.data_ptr<int16_t>(),
        store_link_map.data_ptr<int16_t>(),
        link_sphere_map.data_ptr<int16_t>(),
        joint_offset_map.data_ptr<float>(),
        batch_size, n_spheres,
        n_links, n_joints, store_n_links);
    }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { link_pos, link_quat, batch_robot_spheres, global_cumul_mat };
}

/////////////////////////////////////////////
// Backward kinematics
// Uses 16 threads per batch.
// This version is 30-100% faster compared to
// kin_fused_backward_4t.
/////////////////////////////////////////////
std::vector<torch::Tensor>kin_fused_backward_16t(
  torch::Tensor grad_out, const torch::Tensor grad_nlinks_pos,
  const torch::Tensor grad_nlinks_quat, const torch::Tensor grad_spheres,
  const torch::Tensor global_cumul_mat, const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform, const torch::Tensor robot_spheres,
  const torch::Tensor link_map, const torch::Tensor joint_map,
  const torch::Tensor joint_map_type, const torch::Tensor store_link_map,
  const torch::Tensor link_sphere_map, const torch::Tensor link_chain_map,
  const torch::Tensor joint_offset_map,
  const int batch_size,  const int n_joints, const int n_spheres, const bool sparsity_opt = true,
  const bool use_global_cumul = false)
{
  using namespace Curobo::Kinematics;
  CHECK_INPUT_GUARD(joint_vec);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(grad_nlinks_pos);
  CHECK_INPUT(grad_nlinks_quat);
  CHECK_INPUT(global_cumul_mat);
  CHECK_INPUT(fixed_transform);
  CHECK_INPUT(robot_spheres);
  CHECK_INPUT(link_map);
  CHECK_INPUT(joint_map);
  CHECK_INPUT(joint_map_type);
  CHECK_INPUT(store_link_map);
  CHECK_INPUT(link_sphere_map);
  CHECK_INPUT(link_chain_map);
  CHECK_INPUT(joint_offset_map);

  const int n_links       = link_map.size(0);
  const int store_n_links = store_link_map.size(0);

  // assert(n_links < 128);
  assert(n_joints < 128); // for larger num. of joints, change kernel3's
                          // MAX_JOINTS template value.
  assert(n_links < MAX_TOTAL_LINKS);

  // We need 16 threads per batch
  // Find the maximum number of batches we can use per block:
  //
  int batches_per_block = (int)((MAX_TOTAL_LINKS / n_links));

  if (batches_per_block == 0)
  {
    batches_per_block = 1;
  }

  // To optimize for better occupancy, we might limit to MAX_BATCH_PER_BLOCK
  if (batches_per_block > MAX_BW_BATCH_PER_BLOCK)
  {
    batches_per_block = MAX_BW_BATCH_PER_BLOCK;
  }

  // we cannot have more than 1024 threads:
  if (batches_per_block * M * M > 1024)
  {
    batches_per_block = 1024 / (M * M);
  }

  const int threadsPerBlock = batches_per_block * M * M;

  const int blocksPerGrid =
    (batch_size * M * M + threadsPerBlock - 1) / threadsPerBlock;

  // assert to make sure n_joints, n_links < 128 to avoid overflow
  // printf("threadsPerBlock: %d, blocksPerGRid: %d\n", threadsPerBlock,
  // blocksPerGrid);

  const int sharedMemSize = batches_per_block * n_links * M * M * sizeof(float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  assert(sparsity_opt);
  const bool parallel_write = true;
  if (use_global_cumul)
  {
    if (n_joints < 16)
    {
      AT_DISPATCH_FLOATING_TYPES(
        grad_spheres.scalar_type(), "kin_fused_backward_16t", ([&] {
        kin_fused_backward_kernel3<scalar_t, double, true, true, 16, parallel_write>
          << < blocksPerGrid, threadsPerBlock, sharedMemSize, stream >> > (
          grad_out.data_ptr<float>(),
          grad_nlinks_pos.data_ptr<float>(),
          grad_nlinks_quat.data_ptr<float>(),
          grad_spheres.data_ptr<scalar_t>(),
          global_cumul_mat.data_ptr<float>(),
          joint_vec.data_ptr<float>(),
          fixed_transform.data_ptr<float>(),
          robot_spheres.data_ptr<float>(),
          joint_map_type.data_ptr<int8_t>(),
          joint_map.data_ptr<int16_t>(), link_map.data_ptr<int16_t>(),
          store_link_map.data_ptr<int16_t>(),
          link_sphere_map.data_ptr<int16_t>(),
          link_chain_map.data_ptr<int16_t>(),
          joint_offset_map.data_ptr<float>(),
          batch_size, n_spheres,
          n_links, n_joints, store_n_links);
      }));
    }
    else if (n_joints < 64)
    {
      AT_DISPATCH_FLOATING_TYPES(
        grad_spheres.scalar_type(), "kin_fused_backward_16t", ([&] {
        kin_fused_backward_kernel3<scalar_t, double, true, true, 64, parallel_write>
          << < blocksPerGrid, threadsPerBlock, sharedMemSize, stream >> > (
          grad_out.data_ptr<float>(),
          grad_nlinks_pos.data_ptr<float>(),
          grad_nlinks_quat.data_ptr<float>(),
          grad_spheres.data_ptr<scalar_t>(),
          global_cumul_mat.data_ptr<float>(),
          joint_vec.data_ptr<float>(),
          fixed_transform.data_ptr<float>(),
          robot_spheres.data_ptr<float>(),
          joint_map_type.data_ptr<int8_t>(),
          joint_map.data_ptr<int16_t>(), link_map.data_ptr<int16_t>(),
          store_link_map.data_ptr<int16_t>(),
          link_sphere_map.data_ptr<int16_t>(),
          link_chain_map.data_ptr<int16_t>(),
          joint_offset_map.data_ptr<float>(),
          batch_size, n_spheres,
          n_links, n_joints, store_n_links);
      }));
    }
    else
    {
      AT_DISPATCH_FLOATING_TYPES(
        grad_spheres.scalar_type(), "kin_fused_backward_16t", ([&] {
        kin_fused_backward_kernel3<scalar_t, double, true, true, 128, parallel_write>
          << < blocksPerGrid, threadsPerBlock, sharedMemSize, stream >> > (
          grad_out.data_ptr<float>(),
          grad_nlinks_pos.data_ptr<float>(),
          grad_nlinks_quat.data_ptr<float>(),
          grad_spheres.data_ptr<scalar_t>(),
          global_cumul_mat.data_ptr<float>(),
          joint_vec.data_ptr<float>(),
          fixed_transform.data_ptr<float>(),
          robot_spheres.data_ptr<float>(),
          joint_map_type.data_ptr<int8_t>(),
          joint_map.data_ptr<int16_t>(), link_map.data_ptr<int16_t>(),
          store_link_map.data_ptr<int16_t>(),
          link_sphere_map.data_ptr<int16_t>(),
          link_chain_map.data_ptr<int16_t>(),
          joint_offset_map.data_ptr<float>(),
          batch_size, n_spheres,
          n_links, n_joints, store_n_links);
      }));
    }

    //
  }
  else
  {
    //
    AT_DISPATCH_FLOATING_TYPES(
      grad_spheres.scalar_type(), "kin_fused_backward_16t", ([&] {
      kin_fused_backward_kernel3<scalar_t, double, false, true, 128, parallel_write>
        << < blocksPerGrid, threadsPerBlock, sharedMemSize, stream >> > (
        grad_out.data_ptr<float>(),
        grad_nlinks_pos.data_ptr<float>(),
        grad_nlinks_quat.data_ptr<float>(),
        grad_spheres.data_ptr<scalar_t>(),
        global_cumul_mat.data_ptr<float>(),
        joint_vec.data_ptr<float>(),
        fixed_transform.data_ptr<float>(),
        robot_spheres.data_ptr<float>(),
        joint_map_type.data_ptr<int8_t>(),
        joint_map.data_ptr<int16_t>(),
        link_map.data_ptr<int16_t>(),
        store_link_map.data_ptr<int16_t>(),
        link_sphere_map.data_ptr<int16_t>(),
        link_chain_map.data_ptr<int16_t>(),
        joint_offset_map.data_ptr<float>(),
        batch_size, n_spheres,
        n_links, n_joints, store_n_links);
    }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { grad_out };
}

std::vector<torch::Tensor>
matrix_to_quaternion(torch::Tensor       out_quat,
                     const torch::Tensor in_rot // batch_size, 3
                     )
{
  using namespace Curobo::Kinematics;
  CHECK_INPUT(out_quat);
  CHECK_INPUT_GUARD(in_rot);
  // we compute the warp threads based on number of boxes:

  // TODO: verify this math
  const int batch_size = in_rot.size(0);

  int threadsPerBlock = batch_size;

  if (batch_size > 512)
  {
    threadsPerBlock = 512;
  }

  // we fit warp thread spheres in a threadsPerBlock

  int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    in_rot.scalar_type(), "matrix_to_quaternion", ([&] {
    mat_to_quat_kernel<scalar_t>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
      out_quat.data_ptr<scalar_t>(), in_rot.data_ptr<scalar_t>(),
      batch_size);
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return { out_quat };
}
