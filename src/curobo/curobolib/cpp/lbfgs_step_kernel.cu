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

#include <cuda.h>
#include <torch/extension.h>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

// #include "helper_cuda.h"
#include "helper_math.h"

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <math.h>

// #include <stdio.h>
//
// For the CUDA runtime routines (prefixed with "cuda_")
// #include <cuda_runtime.h>

// #include <cuda_fp16.h>
// #include <helper_cuda.h>
#define M_MAX 512
#define HALF_MAX 65504.0
#define M 15
#define VDIM 175 // 25 * 7,

#define FULL_MASK 0xffffffff

namespace Curobo
{
  namespace Optimization
  {
    template<typename scalar_t>
    __device__ __forceinline__ void
    scalar_vec_product(const scalar_t a, const scalar_t *b, scalar_t *out,
                       const int v_dim)
    {
      for (int i = 0; i < v_dim; i++)
      {
        out[i] = a * b[i];
      }
    }

    template<typename scalar_t>
    __device__ __forceinline__ void
    m_scalar_vec_product(const scalar_t *a, const scalar_t *b, scalar_t *out,
                         const int v_dim, const int m)
    {
      for (int j = 0; j < m; j++)
      {
        for (int i = 0; i < v_dim; i++)
        {
          out[j * v_dim + i] = a[j] * b[j * v_dim + i];
        }
      }
    }

    template<typename scalar_t>
    __device__ __forceinline__ void vec_vec_dot(const scalar_t *a,
                                                const scalar_t *b, scalar_t& out,
                                                const int v_dim)
    {
      for (int i = 0; i < v_dim; i++)
      {
        out += a[i] * b[i];
      }
    }

    template<typename scalar_t>
    __device__ __forceinline__ void update_r(const scalar_t *rho_y,
                                             const scalar_t *s_buffer, scalar_t *r,
                                             scalar_t& alpha, const int v_dim)
    {
      // dot product: and subtract with alpha
      for (int i = 0; i < v_dim; i++)
      {
        alpha -= rho_y[i] * r[i];
      }

      // scalar vector product:
      for (int i = 0; i < v_dim; i++)
      {
        r[i] = r[i] + alpha * s_buffer[i];
      }
    }

    template<typename scalar_t>
    __device__ __forceinline__ void update_q(const scalar_t *y_buffer, scalar_t *gq,
                                             const scalar_t alpha,
                                             const int v_dim)
    {
      //
      for (int i = 0; i < v_dim; i++)
      {
        gq[i] = gq[i] - (alpha * y_buffer[i]);
      }
    }

    template<typename scalar_t>
    __global__ void
    lbfgs_step_kernel_old(scalar_t *step_vec, scalar_t *rho_buffer,
                          const scalar_t *y_buffer, const scalar_t *s_buffer,
                          const scalar_t *grad_q, const float epsilon,
                          const int batchSize, const int m, const int v_dim)
    {
      // each thread writes one sphere of some link
      const int t     = blockDim.x * blockIdx.x + threadIdx.x; // batch
      const int b_idx = t;

      if (t >= (batchSize))
      {
        return;
      }

      // get thread start address:
      const int b_start_scalar_adrs = b_idx * m;
      const int b_start_vec_adrs    = b_idx * m * v_dim;
      const int b_step_start_adrs   = b_idx * v_dim;

      scalar_t rho_s[M * VDIM];

      // copy floats to local buffer?
      // y_buffer, s_buffer, rho_buffer
      // compute rho_s
      scalar_t loc_ybuf[M * VDIM];
      scalar_t loc_sbuf[M * VDIM];
      scalar_t loc_rho[M];
      scalar_t gq[VDIM]; // , l_q[VDIM];
      scalar_t alpha_buffer[M];
      scalar_t t_1, t_2;

      for (int i = 0; i < m * v_dim; i++)
      {
        loc_ybuf[i] = y_buffer[b_start_vec_adrs + i];
        loc_sbuf[i] = s_buffer[b_start_vec_adrs + i];
      }

      for (int i = 0; i < v_dim; i++)
      {
        gq[i] = grad_q[b_step_start_adrs + i];
      }

      for (int i = 0; i < m; i++)
      {
        loc_rho[i] = rho_buffer[b_start_scalar_adrs + i];
      }

      m_scalar_vec_product(&loc_rho[0], &loc_sbuf[0], &rho_s[0], v_dim, m);

      // for loop over m
      for (int i = m - 1; i > m - 2; i--)
      {
        // l_start_vec_adrs = i * v_dim;
        // scalar_vec_product(loc_rho[i], &loc_sbuf[i*v_dim], &rho_s[i*v_dim],
        // v_dim);
        vec_vec_dot(&rho_s[i * v_dim], &gq[0], alpha_buffer[i], v_dim);
        update_q(&loc_ybuf[(i * v_dim)], &gq[0], alpha_buffer[i], v_dim);
      }

      // compute initial hessian:
      vec_vec_dot(&loc_sbuf[(m - 1) * v_dim], &loc_ybuf[(m - 1) * v_dim], t_1,
                  v_dim);
      vec_vec_dot(&loc_ybuf[(m - 1) * v_dim], &loc_ybuf[(m - 1) * v_dim], t_2,
                  v_dim);
      t_1 = t_1 / t_2;

      if (t_1 < 0)
      {
        t_1 = 0;
      }

      t_1 += epsilon;
      scalar_vec_product(t_1, &gq[0], &gq[0], v_dim);

      m_scalar_vec_product(&loc_rho[0], &loc_ybuf[0], &rho_s[0], v_dim, m);

      for (int i = 0; i < m; i++)
      {
        // scalar_vec_product(loc_rho[i], &loc_ybuf[i*v_dim], &rho_s[i*v_dim],
        // v_dim);
        update_r(&rho_s[i * v_dim], &loc_sbuf[i * v_dim], &gq[0], alpha_buffer[i],
                 v_dim);
      }

      // write gq to out grad:

      for (int i = 0; i < v_dim; i++)
      {
        step_vec[b_step_start_adrs + i] = -1.0 * gq[i];
      }
    }

    template<typename psum_t>
    __forceinline__ __device__ psum_t warpReduce(psum_t v, int elems,
                                                 unsigned mask)
    {
      psum_t val   = v;
      int    shift = 1;

      for (int i = elems; i > 1; i /= 2)
      {
        val   += __shfl_down_sync(mask, val, shift);
        shift *= 2;
      }

      // val += __shfl_down_sync(mask, val, 1); // i=32
      // val += __shfl_down_sync(mask, val, 2); // i=16
      // val += __shfl_down_sync(mask, val, 4); // i=8
      // val += __shfl_down_sync(mask, val, 8); // i=4
      // val += __shfl_down_sync(mask, val, 16); // i=2
      return val;
    }

    template<typename scalar_t, typename psum_t>
    __forceinline__ __device__ void reduce(scalar_t v, int m, psum_t *data,
                                           scalar_t *result)
    {
      psum_t   val  = v;
      unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < m);

      val += __shfl_down_sync(mask, val, 1);
      val += __shfl_down_sync(mask, val, 2);
      val += __shfl_down_sync(mask, val, 4);
      val += __shfl_down_sync(mask, val, 8);
      val += __shfl_down_sync(mask, val, 16);

      // int leader = __ffs(mask) – 1;    // select a leader lane
      int leader = 0;

      if (threadIdx.x % 32 == leader)
      {
        if (m < 32)
        {
          result[0] = (scalar_t)val;
        }
        else
        {
          data[(threadIdx.x + 1) / 32] = val;
        }
      }

      if (m >= 32)
      {
        __syncthreads();

        int elems      = (m + 31) / 32;
        unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);

        if (threadIdx.x / 32 == 0) // only the first warp will do this work
        {
          psum_t val2  = data[threadIdx.x % 32];
          int    shift = 1;

          for (int i = elems - 1; i > 0; i /= 2)
          {
            val2  += __shfl_down_sync(mask2, val2, shift);
            shift *= 2;
          }

          // int leader = __ffs(mask2) – 1;    // select a leader lane
          int leader = 0;

          if (threadIdx.x % 32 == leader)
          {
            result[0] = (scalar_t)val2;
          }
        }
      }
      __syncthreads();
    }

    // blockReduce
    template<typename scalar_t, typename psum_t>
    __forceinline__ __device__ void reduce_v1(scalar_t v, int m, psum_t *data,
                                              scalar_t *result)
    {
      unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < m);
      psum_t   val  = warpReduce(v, 32, mask);

      // int leader = __ffs(mask) – 1;    // select a leader lane
      int leader = 0;

      if (threadIdx.x % 32 == leader)
      {
        data[(threadIdx.x + 1) / 32] = val;
      }

      if (m >= 32)
      {
        __syncthreads();

        int elems      = (m + 31) / 32;
        unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);

        if (threadIdx.x / 32 == 0) // only the first warp will do this work
        {
          psum_t val2 = warpReduce(data[threadIdx.x % 32], elems, mask2);

          // // int leader = __ffs(mask2) – 1;    // select a leader lane
          if (threadIdx.x == leader)
          {
            result[0] = (scalar_t)val2;
          }
        }
      }
      else
      {
        if (threadIdx.x == leader)
        {
          result[0] = (scalar_t)val;
        }
      }
      __syncthreads();
    }

    template<typename scalar_t, typename psum_t>
    __inline__ __device__ void dot(const scalar_t *mat1, const scalar_t *mat2,
                                   const int m, psum_t *data, scalar_t *result)
    {
      scalar_t val = mat1[threadIdx.x] * mat2[threadIdx.x];

      reduce(val, m, data, result);
    }

    template<typename scalar_t>__inline__ __device__ scalar_t relu(scalar_t var)
    {
      if (var < 0)
        return 0;
      else
        return var;
    }

    //////////////////////////////////////////////////////////
    // one block per batch
    // v_dim threads per block
    //////////////////////////////////////////////////////////
    template<typename scalar_t, typename psum_t>
    __global__ void lbfgs_step_kernel(scalar_t *step_vec,       // b x 175
                                      scalar_t *rho_buffer,     // m x b x 1
                                      const scalar_t *y_buffer, // m x b x 175
                                      const scalar_t *s_buffer, // m x b x 175
                                      const scalar_t *grad_q,   // b x 175
                                      const float epsilon, const int batchsize,
                                      const int m, const int v_dim)
    {
      __shared__ psum_t
        data[32];                  // temporary buffer needed for block-wide reduction
      __shared__ scalar_t
        result;                    // result of the reduction or vector-vector dot product
      __shared__ scalar_t gq[175]; /// gq  = batch * v_dim

      assert(v_dim < 176);
      int batch = blockIdx.x;      // one block per batch

      if (threadIdx.x >= v_dim)
        return;

      gq[threadIdx.x] = grad_q[batch * v_dim + threadIdx.x]; // copy grad_q to gq

      scalar_t alpha_buffer[16];

      // assert(m<16); // allocating a buffer assuming m < 16

      for (int i = m - 1; i > -1; i--)
      {
        dot(&gq[0], &s_buffer[i * batchsize * v_dim + batch * v_dim], v_dim,
            &data[0], &result);
        alpha_buffer[i] = result * rho_buffer[i * batchsize + batch];
        gq[threadIdx.x] =
          gq[threadIdx.x] -
          alpha_buffer[i] *
          y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      // compute var1
      scalar_t val1 =
        y_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x];
      scalar_t val2 =
        s_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x];
      reduce(val1 * val1, v_dim, data, &result);
      scalar_t denominator = result;
      reduce(val1 * val2, v_dim, data, &result);
      scalar_t numerator = result;
      scalar_t var1      = numerator / denominator;

      scalar_t gamma = relu(var1) + epsilon; // epsilon

      gq[threadIdx.x] = gamma * gq[threadIdx.x];

      for (int i = 0; i < m; i++)
      {
        dot(&gq[0], &y_buffer[i * batchsize * v_dim + batch * v_dim], v_dim,
            &data[0], &result);
        gq[threadIdx.x] =
          gq[threadIdx.x] +
          (alpha_buffer[i] - result * rho_buffer[i * batchsize + batch]) *
          s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      step_vec[batch * v_dim + threadIdx.x] =
        -1 * gq[threadIdx.x]; // copy from shared memory to global memory
    }

    template<typename scalar_t, typename psum_t>
    __global__ void lbfgs_update_buffer_kernel(scalar_t *rho_buffer,   // m x b x 1
                                               scalar_t *y_buffer,     // m x b x 175
                                               scalar_t *s_buffer,     // m x b x 175
                                               scalar_t *q,            // b x 175
                                               scalar_t *x_0,          // b x 175
                                               scalar_t *grad_0,       // b x 175
                                               const scalar_t *grad_q, // b x 175
                                               const int batchsize, const int m,
                                               const int v_dim)
    {
      __shared__ psum_t
        data[32]; // temporary buffer needed for block-wide reduction
      __shared__ scalar_t
        result;   // result of the reduction or vector-vector dot product

      // __shared__ scalar_t y[175]; // temporary shared memory storage
      // __shared__ scalar_t s[175]; // temporary shared memory storage
      assert(v_dim <= VDIM);
      int batch = blockIdx.x; // one block per batch

      if (threadIdx.x >= v_dim)
        return;

      scalar_t y =
        grad_q[batch * v_dim + threadIdx.x] - grad_0[batch * v_dim + threadIdx.x];
      scalar_t s =
        q[batch * v_dim + threadIdx.x] - x_0[batch * v_dim + threadIdx.x];
      reduce(y * s, v_dim, &data[0], &result);

      for (int i = 1; i < m; i++)
      {
        s_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] =
          s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
        y_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] =
          y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      // __syncthreads();

      s_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = s;
      y_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = y;
      grad_0[batch * v_dim +
             threadIdx.x]                                                 =
        grad_q[batch * v_dim + threadIdx.x];
      x_0[batch * v_dim +
          threadIdx.x] =
        q[batch * v_dim + threadIdx.x];

      if (threadIdx.x == 0)
      {
        scalar_t rho = 1 / result;

        for (int i = 1; i < m; i++)
        {
          rho_buffer[(i - 1) * batchsize + batch] =
            rho_buffer[i * batchsize + batch];
        }
        rho_buffer[(m - 1) * batchsize + batch] = rho;
      }
    }

    template<typename scalar_t, typename psum_t>
    __global__ void reduce_kernel(
      scalar_t *vec1,         // b x 175
      scalar_t *vec2,         // b x 175
      scalar_t *rho_buffer,   // m x b x 1

      scalar_t *sum_out,      // m x b x 1

      const int batchsize, const int m,
      const int v_dim)        // s_buffer and y_buffer are not rolled by default
    {
      __shared__ psum_t
        data[32];             // temporary buffer needed for block-wide reduction
      __shared__ scalar_t
          result;             // result of the reduction or vector-vector dot product
      int batch = blockIdx.x; // one block per batch

      if (threadIdx.x >= v_dim)
        return;

      ////////////////////
      // update_buffer
      ////////////////////
      scalar_t y = vec1[batch * v_dim + threadIdx.x];
      scalar_t s = vec2[batch * v_dim + threadIdx.x];

      reduce(y * s, v_dim, &data[0], &result);
      scalar_t numerator = result;

      if (threadIdx.x == 0)
      {
        sum_out[batch] = 1 / numerator;
      }

      // return;
      if (threadIdx.x < m - 1)
      {
        // m thread participate to shif the values
        // this is safe as m<32  and this happens in lockstep
        rho_buffer[threadIdx.x * batchsize + batch] =
          rho_buffer[(threadIdx.x + 1) * batchsize + batch];
      }
      else if (threadIdx.x == m - 1)
      {
        scalar_t rho = (1 / numerator);
        rho_buffer[threadIdx.x * batchsize + batch] = rho;
      }
    }

    template<typename scalar_t, typename psum_t, bool rolled_ys>
    __global__ void lbfgs_update_buffer_and_step_v1(
      scalar_t *step_vec,     // b x 175
      scalar_t *rho_buffer,   // m x b x 1
      scalar_t *y_buffer,     // m x b x 175
      scalar_t *s_buffer,     // m x b x 175
      scalar_t *q,            // b x 175
      scalar_t *x_0,          // b x 175
      scalar_t *grad_0,       // b x 175
      const scalar_t *grad_q, // b x 175
      const float epsilon, const int batchsize, const int m, const int v_dim,
      const bool stable_mode =
        false)                // s_buffer and y_buffer are not rolled by default
    {
      // extern __shared__ scalar_t alpha_buffer_sh[];
      extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
      scalar_t *my_smem_rc      = reinterpret_cast<scalar_t *>(my_smem);
      scalar_t *alpha_buffer_sh = &my_smem_rc[0];              // m*blockDim.x
      scalar_t *rho_buffer_sh   = &my_smem_rc[m * blockDim.x]; // batchsize*m
      scalar_t *s_buffer_sh     =
        &my_smem_rc[m * blockDim.x + m * batchsize];           // m*blockDim.x
      scalar_t *y_buffer_sh =
        &my_smem_rc[2 * m * blockDim.x + m * batchsize];       // m*blockDim.x

      __shared__ psum_t
        data[32];                                              // temporary buffer needed for
                                                               // block-wide reduction
      __shared__ scalar_t
          result;                                              // result of the reduction or
                                                               // vector-vector dot product
      int batch = blockIdx.x;                                  // one block per batch

      if (threadIdx.x >= v_dim)
        return;

      scalar_t gq;
      gq = grad_q[batch * v_dim + threadIdx.x]; // copy grad_q to gq

      ////////////////////
      // update_buffer
      ////////////////////
      scalar_t y = gq - grad_0[batch * v_dim + threadIdx.x];

      // if y is close to zero
      scalar_t s =
        q[batch * v_dim + threadIdx.x] - x_0[batch * v_dim + threadIdx.x];
      reduce_v1(y * s, v_dim, &data[0], &result);

      // reduce(y * s, v_dim, &data[0], &result);
      scalar_t numerator = result;

      // scalar_t rho = 1.0/numerator;

      if (!rolled_ys)
      {
#pragma unroll

        for (int i = 1; i < m; i++)
        {
          scalar_t st =
            s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          scalar_t yt =
            y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          s_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = st;
          y_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = yt;
          s_buffer_sh[m * threadIdx.x + i - 1]                                = st;
          y_buffer_sh[m * threadIdx.x + i - 1]                                = yt;
        }
      }

      s_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = s;
      y_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = y;
      s_buffer_sh[m * threadIdx.x + m - 1]                                = s;
      y_buffer_sh[m * threadIdx.x + m - 1]                                = y;
      grad_0[batch * v_dim + threadIdx.x]                                 = gq;
      x_0[batch * v_dim +
          threadIdx.x]                                                    =
        q[batch * v_dim + threadIdx.x];

      if (threadIdx.x < m - 1)
      {
        // m thread participate to shif the values
        // this is safe as m<32  and this happens in lockstep
        scalar_t rho = rho_buffer[(threadIdx.x + 1) * batchsize + batch];
        rho_buffer[threadIdx.x * batchsize + batch]    = rho;
        rho_buffer_sh[threadIdx.x * batchsize + batch] = rho;
      }

      if (threadIdx.x == m - 1)
      {
        scalar_t rho = 1.0 / numerator;

        // if this is nan, make it zero:
        if (stable_mode && (numerator == 0.0))
        {
          rho = 0.0;
        }
        rho_buffer[threadIdx.x * batchsize + batch]    = rho;
        rho_buffer_sh[threadIdx.x * batchsize + batch] = rho;
      }

      // return;
      __syncthreads();

      ////////////////////
      // step
      ////////////////////
      // scalar_t alpha_buffer[16];
      // assert(m<16); // allocating a buffer assuming m < 16

#pragma unroll

      for (int i = m - 1; i > -1; i--)
      {
        // reduce(gq * s_buffer[i*batchsize*v_dim + batch*v_dim + threadIdx.x],
        // v_dim, &data[0], &result);
        reduce_v1(gq * s_buffer_sh[m * threadIdx.x + i], v_dim, &data[0], &result);
        alpha_buffer_sh[threadIdx.x * m + i] =
          result * rho_buffer_sh[i * batchsize + batch];

        // gq = gq - alpha_buffer_sh[threadIdx.x*m+i]*y_buffer[i*batchsize*v_dim +
        // batch*v_dim + threadIdx.x];
        gq = gq - alpha_buffer_sh[threadIdx.x * m + i] *
             y_buffer_sh[m * threadIdx.x + i];
      }

      // compute var1
      reduce_v1(y * y, v_dim, data, &result);
      scalar_t denominator = result;

      // reduce(s*y, v_dim, data, &result); // redundant - already computed it above
      // scalar_t numerator = result;
      scalar_t var1 = numerator / denominator;

      // To improve stability, uncomment below line: [this however leads to poor
      // convergence]

      if (stable_mode && (denominator == 0.0))
      {
        var1 = epsilon;
      }

      scalar_t gamma = relu(var1);
      gq = gamma * gq;

#pragma unroll

      for (int i = 0; i < m; i++)
      {
        // reduce(gq * y_buffer[i*batchsize*v_dim + batch*v_dim + threadIdx.x],
        // v_dim, &data[0], &result); gq = gq + (alpha_buffer_sh[threadIdx.x*m+i] -
        // result * rho_buffer_sh[i*batchsize+batch]) * s_buffer[i*batchsize*v_dim +
        // batch*v_dim + threadIdx.x];
        reduce_v1(gq * y_buffer_sh[m * threadIdx.x + i], v_dim, &data[0], &result);
        gq = gq + (alpha_buffer_sh[threadIdx.x * m + i] -
                   result * rho_buffer_sh[i * batchsize + batch]) *
             s_buffer_sh[m * threadIdx.x + i];
      }

      step_vec[batch * v_dim + threadIdx.x] =
        -1.0 * gq; // copy from shared memory to global memory
    }

    // (32/M) rolls per warp
    // Threads in a warp in a GPU execute in a lock-step. We leverage that to do
    // the roll without using temporary storage or explicit synchronization.
    template<typename scalar_t>
    __global__ void lbfgs_roll(scalar_t *a, // m x b x 175
                               scalar_t *b, // m x b x 175
                               const int m_t, const int batchsize, const int m,
                               const int v_dim)
    {
      assert(m_t <= 32);
      int t = blockDim.x * blockIdx.x + threadIdx.x;

      if (t >= m_t * v_dim * batchsize)
        return;

      int _m     = t % m_t;
      int _v_dim = (t / m_t) % v_dim;
      int batch  = t / (m * v_dim); // this line could be wrong?

      if (_m < m - 1)
      {
        a[_m * batchsize * v_dim + batch * v_dim + _v_dim] =
          a[(_m + 1) * batchsize * v_dim + batch * v_dim + _v_dim];
        b[_m * batchsize * v_dim + batch * v_dim + _v_dim] =
          b[(_m + 1) * batchsize * v_dim + batch * v_dim + _v_dim];
      }
    }

    template<typename scalar_t, typename psum_t, bool rolled_ys>
    __global__ void lbfgs_update_buffer_and_step(
      scalar_t *step_vec,     // b x 175
      scalar_t *rho_buffer,   // m x b x 1
      scalar_t *y_buffer,     // m x b x 175
      scalar_t *s_buffer,     // m x b x 175
      scalar_t *q,            // b x 175
      scalar_t *x_0,          // b x 175
      scalar_t *grad_0,       // b x 175
      const scalar_t *grad_q, // b x 175
      const float epsilon, const int batchsize, const int m, const int v_dim,
      const bool stable_mode =
        false)                // s_buffer and y_buffer are not rolled by default
    {
      // extern __shared__ scalar_t alpha_buffer_sh[];
      extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
      scalar_t *alpha_buffer_sh = reinterpret_cast<scalar_t *>(my_smem);

      __shared__ psum_t
        data[32];             // temporary buffer needed for block-wide reduction
      __shared__ scalar_t
          result;             // result of the reduction or vector-vector dot product
      int batch = blockIdx.x; // one block per batch

      if (threadIdx.x >= v_dim)
        return;

      scalar_t gq;
      gq = grad_q[batch * v_dim + threadIdx.x]; // copy grad_q to gq

      ////////////////////
      // update_buffer
      ////////////////////
      scalar_t y = gq - grad_0[batch * v_dim + threadIdx.x];

      // if y is close to zero
      scalar_t s =
        q[batch * v_dim + threadIdx.x] - x_0[batch * v_dim + threadIdx.x];
      reduce(y * s, v_dim, &data[0], &result);
      scalar_t numerator = result;

      // scalar_t rho = 1.0/numerator;

      if (!rolled_ys)
      {
        for (int i = 1; i < m; i++)
        {
          s_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] =
            s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          y_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] =
            y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
        }
      }

      s_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = s;
      y_buffer[(m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = y;
      grad_0[batch * v_dim + threadIdx.x]                                 = gq;
      x_0[batch * v_dim +
          threadIdx.x]                                                    =
        q[batch * v_dim + threadIdx.x];

      if (threadIdx.x < m - 1)
      {
        // m thread participate to shif the values
        // this is safe as m<32  and this happens in lockstep
        rho_buffer[threadIdx.x * batchsize + batch] =
          rho_buffer[(threadIdx.x + 1) * batchsize + batch];
      }

      if (threadIdx.x == m - 1)
      {
        scalar_t rho = 1.0 / numerator;

        // if this is nan, make it zero:
        if (stable_mode && (numerator == 0.0))
        {
          rho = 0.0;
        }
        rho_buffer[threadIdx.x * batchsize + batch] = rho;
      }

      // return;
      // __syncthreads();
      ////////////////////
      // step
      ////////////////////
      // scalar_t alpha_buffer[16];
      // assert(m<16); // allocating a buffer assuming m < 16

#pragma unroll

      for (int i = m - 1; i > -1; i--)
      {
        reduce(gq * s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x],
               v_dim, &data[0], &result);
        alpha_buffer_sh[threadIdx.x * m + i] =
          result * rho_buffer[i * batchsize + batch];
        gq = gq - alpha_buffer_sh[threadIdx.x * m + i] *
             y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      // compute var1
      reduce(y * y, v_dim, data, &result);
      scalar_t denominator = result;

      // reduce(s*y, v_dim, data, &result); // redundant - already computed it above
      // scalar_t numerator = result;
      scalar_t var1 = numerator / denominator;

      // To improve stability, uncomment below line: [this however leads to poor
      // convergence]

      if (stable_mode && (denominator == 0.0))
      {
        var1 = epsilon;
      }

      scalar_t gamma = relu(var1);

      gq = gamma * gq;

#pragma unroll

      for (int i = 0; i < m; i++)
      {
        reduce(gq * y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x],
               v_dim, &data[0], &result);
        gq = gq + (alpha_buffer_sh[threadIdx.x * m + i] -
                   result * rho_buffer[i * batchsize + batch]) *
             s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      step_vec[batch * v_dim + threadIdx.x] =
        -1.0 * gq; // copy from shared memory to global memory
    }
  } // namespace Optimization
} // namespace Curobo
std::vector<torch::Tensor>
lbfgs_step_cuda(torch::Tensor step_vec, torch::Tensor rho_buffer,
                torch::Tensor y_buffer, torch::Tensor s_buffer,
                torch::Tensor grad_q, const float epsilon, const int batch_size,
                const int m, const int v_dim)
{
  using namespace Curobo::Optimization;
  const int threadsPerBlock = 128;
  const int blocksPerGrid   =
    ((batch_size) + threadsPerBlock - 1) / threadsPerBlock;

  // launch threads per batch:
  // int threadsPerBlock = pow(2,((int)log2(v_dim))+1);

  // const int blocksPerGrid = batch_size; //((batch_size) + threadsPerBlock -
  // 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    step_vec.scalar_type(), "lbfgs_step_cu", ([&] {
    lbfgs_step_kernel_old<scalar_t>
      << < blocksPerGrid, threadsPerBlock,
      v_dim * sizeof(step_vec.scalar_type()), stream >> > (
      step_vec.data_ptr<scalar_t>(), rho_buffer.data_ptr<scalar_t>(),
      y_buffer.data_ptr<scalar_t>(), s_buffer.data_ptr<scalar_t>(),
      grad_q.data_ptr<scalar_t>(), epsilon, batch_size, m, v_dim);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { step_vec };
}

std::vector<torch::Tensor>
lbfgs_update_cuda(torch::Tensor rho_buffer, torch::Tensor y_buffer,
                  torch::Tensor s_buffer, torch::Tensor q, torch::Tensor grad_q,
                  torch::Tensor x_0, torch::Tensor grad_0, const int batch_size,
                  const int m, const int v_dim)
{
  using namespace Curobo::Optimization;

  // const int threadsPerBlock = 128;
  //  launch threads per batch:
  // int threadsPerBlock = pow(2,((int)log2(v_dim))+1);
  int threadsPerBlock = v_dim;

  const int blocksPerGrid =
    batch_size; // ((batch_size) + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    y_buffer.scalar_type(), "lbfgs_update_cu", ([&] {
    lbfgs_update_buffer_kernel<scalar_t, scalar_t>
      << < blocksPerGrid, threadsPerBlock,
      v_dim * sizeof(y_buffer.scalar_type()), stream >> > (
      rho_buffer.data_ptr<scalar_t>(), y_buffer.data_ptr<scalar_t>(),
      s_buffer.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(),
      x_0.data_ptr<scalar_t>(), grad_0.data_ptr<scalar_t>(),
      grad_q.data_ptr<scalar_t>(), batch_size, m, v_dim);
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { rho_buffer, y_buffer, s_buffer, x_0, grad_0 };
}

std::vector<torch::Tensor>
lbfgs_cuda_fuse(torch::Tensor step_vec, torch::Tensor rho_buffer,
                torch::Tensor y_buffer, torch::Tensor s_buffer, torch::Tensor q,
                torch::Tensor grad_q, torch::Tensor x_0, torch::Tensor grad_0,
                const float epsilon, const int batch_size, const int m,
                const int v_dim, const bool stable_mode)
{
  using namespace Curobo::Optimization;

  // call first kernel:

  int threadsPerBlock = v_dim;
  assert(threadsPerBlock < 1024);
  assert(m < M_MAX);
  int blocksPerGrid =
    batch_size; // ((batch_size) + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int smemsize        = 0;

  if (true)
  {
    AT_DISPATCH_FLOATING_TYPES(
      y_buffer.scalar_type(), "lbfgs_cuda_fuse_kernel", [&] {
      smemsize = m * threadsPerBlock * sizeof(scalar_t);
      lbfgs_update_buffer_and_step<scalar_t, scalar_t, false>
        << < blocksPerGrid, threadsPerBlock, smemsize, stream >> > (
        step_vec.data_ptr<scalar_t>(),
        rho_buffer.data_ptr<scalar_t>(),
        y_buffer.data_ptr<scalar_t>(), s_buffer.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(), x_0.data_ptr<scalar_t>(),
        grad_0.data_ptr<scalar_t>(), grad_q.data_ptr<scalar_t>(),
        epsilon, batch_size, m, v_dim, stable_mode);
    });
  }
  else
  {
    // v1 does not work
    AT_DISPATCH_FLOATING_TYPES(
      y_buffer.scalar_type(), "lbfgs_cuda_fuse_kernel_v1", [&] {
      smemsize = 3 * m * threadsPerBlock * sizeof(scalar_t) +
                 m * batch_size * sizeof(scalar_t);
      lbfgs_update_buffer_and_step_v1<scalar_t, scalar_t, false>
        << < blocksPerGrid, threadsPerBlock, smemsize, stream >> > (
        step_vec.data_ptr<scalar_t>(),
        rho_buffer.data_ptr<scalar_t>(),
        y_buffer.data_ptr<scalar_t>(), s_buffer.data_ptr<scalar_t>(),
        q.data_ptr<scalar_t>(), x_0.data_ptr<scalar_t>(),
        grad_0.data_ptr<scalar_t>(), grad_q.data_ptr<scalar_t>(),
        epsilon, batch_size, m, v_dim, stable_mode);
    });
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0 };
}

std::vector<torch::Tensor>reduce_cuda(torch::Tensor vec, torch::Tensor vec2,
                                      torch::Tensor rho_buffer,
                                      torch::Tensor sum, const int batch_size,
                                      const int m, const int v_dim)
{
  using namespace Curobo::Optimization;

  int threadsPerBlock = pow(2, ((int)log2(v_dim)) + 1);
  int blocksPerGrid   =
    batch_size; // ((batch_size) + threadsPerBlock - 1) / threadsPerBlock;
                // printf("threadsPerBlock:%d, blocksPerGrid: %d\n",
                // threadsPerBlock, blocksPerGrid);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    vec.scalar_type(), "reduce_cu", ([&] {
    reduce_kernel<scalar_t, scalar_t>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
      vec.data_ptr<scalar_t>(), vec2.data_ptr<scalar_t>(),
      rho_buffer.data_ptr<scalar_t>(), sum.data_ptr<scalar_t>(),
      batch_size, m, v_dim);
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { sum, rho_buffer };
}
