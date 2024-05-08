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
#include <cuda_runtime.h>
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
//#define M_MAX 512
//#define HALF_MAX 65504.0
//#define M 15
//#define VDIM 175 // 25 * 7,
#define FULL_MASK 0xffffffff
#define VOLTA_PLUS true
namespace Curobo
{
  namespace Optimization
  {
    
     
    template<typename scalar_t, typename psum_t>
    __forceinline__ __device__ void reduce_v0(scalar_t v, int m, psum_t *data,
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
      const int leader = 0;
      

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

          if (threadIdx.x % 32 == leader)
          {
            result[0] = (scalar_t)val2;
          }
        }
      }
      __syncthreads();
      
    }

   

    template<typename scalar_t>__inline__ __device__ scalar_t relu(scalar_t var)
    {
      if (var < 0)
        return 0;
      else
        return var;
    }

    template<typename psum_t>
    __forceinline__ __device__ psum_t warpReduce(psum_t v,
    const int elems,
                                                 unsigned mask)
    {
      psum_t val   = v;
      int    shift = 1;

      #pragma unroll 
      for (int i = elems; i > 1; i /= 2)
      {
        val   += __shfl_down_sync(mask, val, shift);
        shift *= 2;
      }
      return val;
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
        if (m < 32)
        {
          result[0] = (scalar_t)val;
        }
        else
        {
          data[(threadIdx.x + 1) / 32] = val;
        }
      }
      /*
      if (threadIdx.x % 32 == leader)
      {
        data[(threadIdx.x + 1) / 32] = val;
      }
      */
      if (m >= 32)
      {
        __syncthreads();

        int elems      = (m + 31) / 32;
        unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);

        if (threadIdx.x / 32 == 0) // only the first warp will do this work
        {
          psum_t val2  = data[threadIdx.x % 32];
          int    shift = 1;

          #pragma unroll
          for (int i = elems - 1; i > 0; i /= 2)
          {
            val2  += __shfl_down_sync(mask2, val2, shift);
            shift *= 2;
          }

          //psum_t val2 = warpReduce(data[threadIdx.x % 32], elems - 1, mask2);

          // // int leader = __ffs(mask2) – 1;    // select a leader lane
          if (threadIdx.x % 32 == leader)
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
      const float epsilon, const int batchsize, const int lbfgs_history, const int v_dim,
      const bool stable_mode = false)                // s_buffer and y_buffer are not rolled by default
    {
      extern __shared__ float my_smem_rc[];
      //__shared__ float my_smem_rc[21 * (3 * (32 * 7) + 1)];
      int history_m = lbfgs_history;
      
      // align the external shared memory by 4 bytes
      float* s_buffer_sh = (float *) &my_smem_rc;           // m*blockDim.x

      float* y_buffer_sh = (float *) &s_buffer_sh[history_m * v_dim];       // m*blockDim.x
      float* alpha_buffer_sh = (float *) &y_buffer_sh[history_m * v_dim];       // m*blockDim.x
      float* rho_buffer_sh = (float *) &alpha_buffer_sh[history_m * v_dim];       // m*blockDim.x

      psum_t* data = (psum_t *)&rho_buffer_sh[history_m];
      float* result = (float *)&data[32];


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

      //reduce_v1<scalar_t, psum_t>(y * s, v_dim, &data[0], &result);
      reduce_v1<scalar_t, psum_t>(y * s, v_dim, &data[0], result);
      

      scalar_t numerator = result[0];

      if (!rolled_ys)
      {
        #pragma unroll
        for (int i = 1; i < history_m; i++)
        {
          scalar_t st =
            s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          scalar_t yt =
            y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          s_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = st;
          y_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = yt;
          s_buffer_sh[history_m * threadIdx.x + i - 1]                                = st;
          y_buffer_sh[history_m * threadIdx.x + i - 1]                                = yt;
        }
      }

      s_buffer[(history_m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = s;
      y_buffer[(history_m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = y;
      s_buffer_sh[history_m * threadIdx.x + history_m - 1]                                = s;
      y_buffer_sh[history_m * threadIdx.x + history_m - 1]                                = y;
      grad_0[batch * v_dim + threadIdx.x]                                 = gq;
      x_0[batch * v_dim +
          threadIdx.x]                                                    =
        q[batch * v_dim + threadIdx.x];

      if (threadIdx.x < history_m - 1)
      {
        // m thread participate to shif the values
        // this is safe as m<32  and this happens in lockstep
        scalar_t rho = rho_buffer[(threadIdx.x + 1) * batchsize + batch];
        rho_buffer[threadIdx.x * batchsize + batch]    = rho;
        rho_buffer_sh[threadIdx.x] = rho;
      }

      if (threadIdx.x == history_m - 1)
      {
        scalar_t rho = 1.0 / numerator;

        // if this is nan, make it zero:
        if (stable_mode && (numerator == 0.0))
        {
          rho = 0.0;
        }
        rho_buffer[threadIdx.x * batchsize + batch]    = rho;
        rho_buffer_sh[threadIdx.x] = rho;
      }

      __syncthreads();

      ////////////////////
      // step
      ////////////////////
      // scalar_t alpha_buffer[16];
      // assert(m<16); // allocating a buffer assuming m < 16

      #pragma unroll
      for (int i = history_m - 1; i > -1; i--)
      {
        // reduce(gq * s_buffer[i*batchsize*v_dim + batch*v_dim + threadIdx.x],
        // v_dim, &data[0], &result);
        //reduce_v1<scalar_t, psum_t>(gq * s_buffer_sh[m * threadIdx.x + i], v_dim, &data[0], &result);
        reduce_v1<scalar_t, psum_t>(gq * s_buffer_sh[history_m * threadIdx.x + i], v_dim, &data[0], result);

        alpha_buffer_sh[threadIdx.x * history_m + i] =
          result[0] * rho_buffer_sh[i];

        // gq = gq - alpha_buffer_sh[threadIdx.x*m+i]*y_buffer[i*batchsize*v_dim +
        // batch*v_dim + threadIdx.x];
        gq = gq - alpha_buffer_sh[threadIdx.x * history_m + i] *
             y_buffer_sh[history_m * threadIdx.x + i];
      }
      //return;

      // compute var1
      //reduce_v1<scalar_t, psum_t>(y * y, v_dim, &data[0], &result);
      reduce_v1<scalar_t, psum_t>(y * y, v_dim, &data[0], result);

      scalar_t denominator = result[0];

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
      for (int i = 0; i < history_m; i++)
      {
        // reduce(gq * y_buffer[i*batchsize*v_dim + batch*v_dim + threadIdx.x],
        // v_dim, &data[0], &result); gq = gq + (alpha_buffer_sh[threadIdx.x*m+i] -
        // result * rho_buffer_sh[i*batchsize+batch]) * s_buffer[i*batchsize*v_dim +
        // batch*v_dim + threadIdx.x];
        //reduce_v1<scalar_t, psum_t>(gq * y_buffer_sh[m * threadIdx.x + i], v_dim, &data[0], &result);
        reduce_v1<scalar_t, psum_t>(gq * y_buffer_sh[history_m * threadIdx.x + i], v_dim, &data[0], result);

        gq = gq + (alpha_buffer_sh[threadIdx.x * history_m + i] -
                   result[0] * rho_buffer_sh[i]) *
             s_buffer_sh[history_m * threadIdx.x + i];
      }

      step_vec[batch * v_dim + threadIdx.x] =
        -1.0 * gq; // copy from shared memory to global memory
    }

    template<typename scalar_t, typename psum_t, bool rolled_ys, int FIXED_M>
    __global__ void lbfgs_update_buffer_and_step_v1_compile_m(
      scalar_t *step_vec,     // b x 175
      scalar_t *rho_buffer,   // m x b x 1
      scalar_t *y_buffer,     // m x b x 175
      scalar_t *s_buffer,     // m x b x 175
      scalar_t *q,            // b x 175
      scalar_t *x_0,          // b x 175
      scalar_t *grad_0,       // b x 175
      const scalar_t *grad_q, // b x 175
      const float epsilon, const int batchsize, const int lbfgs_history, const int v_dim,
      const bool stable_mode = false)                // s_buffer and y_buffer are not rolled by default
    {
      extern __shared__ float my_smem_rc[];
      //__shared__ float my_smem_rc[21 * (3 * (32 * 7) + 1)];
      
      // align the external shared memory by 4 bytes
      float* s_buffer_sh = (float *) &my_smem_rc;           // m*blockDim.x

      float* y_buffer_sh = (float *) &s_buffer_sh[FIXED_M  * v_dim];       // m*blockDim.x
      float* alpha_buffer_sh = (float *) &y_buffer_sh[FIXED_M  * v_dim];       // m*blockDim.x
      float* rho_buffer_sh = (float *) &alpha_buffer_sh[FIXED_M  * v_dim];       // m*blockDim.x

      psum_t* data = (psum_t *)&rho_buffer_sh[FIXED_M];
      float* result = (float *)&data[32];


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

      //reduce_v1<scalar_t, psum_t>(y * s, v_dim, &data[0], &result);
      reduce_v1<scalar_t, psum_t>(y * s, v_dim, &data[0], result);
      

      scalar_t numerator = result[0];

      if (!rolled_ys)
      {
        scalar_t st = 0;
        scalar_t yt = 0;
        #pragma unroll
        for (int i = 1; i < FIXED_M ; i++)
        {
          st =
            s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          yt =
            y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          s_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = st;
          y_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = yt;
          s_buffer_sh[FIXED_M  * threadIdx.x + i - 1]                                = st;
          y_buffer_sh[FIXED_M  * threadIdx.x + i - 1]                                = yt;
        }
      }

      s_buffer[(FIXED_M  - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = s;
      y_buffer[(FIXED_M  - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = y;
      s_buffer_sh[FIXED_M  * threadIdx.x + FIXED_M  - 1]                                = s;
      y_buffer_sh[FIXED_M  * threadIdx.x + FIXED_M  - 1]                                = y;
      grad_0[batch * v_dim + threadIdx.x]                                 = gq;
      x_0[batch * v_dim +
          threadIdx.x]                                                    =
        q[batch * v_dim + threadIdx.x];

      if (threadIdx.x < FIXED_M  - 1)
      {
        // m thread participate to shif the values
        // this is safe as m<32  and this happens in lockstep
        scalar_t rho = rho_buffer[(threadIdx.x + 1) * batchsize + batch];
        rho_buffer[threadIdx.x * batchsize + batch]    = rho;
        rho_buffer_sh[threadIdx.x] = rho;
      }

      if (threadIdx.x == FIXED_M  - 1)
      {
        scalar_t rho = 1.0 / numerator;

        // if this is nan, make it zero:
        if (stable_mode && (numerator == 0.0))
        {
          rho = 0.0;
        }
        rho_buffer[threadIdx.x * batchsize + batch]    = rho;
        rho_buffer_sh[threadIdx.x] = rho;
      }

      __syncthreads();

      ////////////////////
      // step
      ////////////////////
      // scalar_t alpha_buffer[16];
      // assert(m<16); // allocating a buffer assuming m < 16

      #pragma unroll
      for (int i = FIXED_M  - 1; i > -1; i--)
      {
        // reduce(gq * s_buffer[i*batchsize*v_dim + batch*v_dim + threadIdx.x],
        // v_dim, &data[0], &result);
        //reduce_v1<scalar_t, psum_t>(gq * s_buffer_sh[m * threadIdx.x + i], v_dim, &data[0], &result);
        reduce_v1<scalar_t, psum_t>(gq * s_buffer_sh[FIXED_M  * threadIdx.x + i], v_dim, &data[0], result);

        alpha_buffer_sh[threadIdx.x * FIXED_M  + i] =
          result[0] * rho_buffer_sh[i];

        // gq = gq - alpha_buffer_sh[threadIdx.x*m+i]*y_buffer[i*batchsize*v_dim +
        // batch*v_dim + threadIdx.x];
        gq = gq - alpha_buffer_sh[threadIdx.x * FIXED_M  + i] *
             y_buffer_sh[FIXED_M  * threadIdx.x + i];
      }
      //return;

      // compute var1
      //reduce_v1<scalar_t, psum_t>(y * y, v_dim, &data[0], &result);
      reduce_v1<scalar_t, psum_t>(y * y, v_dim, &data[0], result);

      scalar_t denominator = result[0];

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
      for (int i = 0; i < FIXED_M ; i++)
      {
        // reduce(gq * y_buffer[i*batchsize*v_dim + batch*v_dim + threadIdx.x],
        // v_dim, &data[0], &result); gq = gq + (alpha_buffer_sh[threadIdx.x*m+i] -
        // result * rho_buffer_sh[i*batchsize+batch]) * s_buffer[i*batchsize*v_dim +
        // batch*v_dim + threadIdx.x];
        //reduce_v1<scalar_t, psum_t>(gq * y_buffer_sh[m * threadIdx.x + i], v_dim, &data[0], &result);
        reduce_v1<scalar_t, psum_t>(gq * y_buffer_sh[FIXED_M  * threadIdx.x + i], v_dim, &data[0], result);

        gq = gq + (alpha_buffer_sh[threadIdx.x * FIXED_M  + i] -
                   result[0] * rho_buffer_sh[i]) *
             s_buffer_sh[FIXED_M * threadIdx.x + i];
      }

      step_vec[batch * v_dim + threadIdx.x] =
        -1.0 * gq; // copy from shared memory to global memory
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
      extern __shared__ float alpha_buffer_sh[];
      
      //extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
      //scalar_t *alpha_buffer_sh = reinterpret_cast<scalar_t *>(my_smem);
      
      //extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
      //float *alpha_buffer_sh = reinterpret_cast<float *>(my_smem);
      
        __shared__ psum_t
        data[32];    
              // temporary buffer needed for block-wide reduction
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
      reduce_v1<scalar_t, psum_t>(y * s, v_dim, &data[0], &result);
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
        reduce_v1<scalar_t, psum_t>(gq * s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x],
               v_dim, &data[0], &result);
        alpha_buffer_sh[threadIdx.x * m + i] =
          result * rho_buffer[i * batchsize + batch];
        gq = gq - alpha_buffer_sh[threadIdx.x * m + i] *
             y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      // compute var1
      reduce_v1<scalar_t, psum_t>(y * y, v_dim, &data[0], &result);
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
        reduce_v1<scalar_t, psum_t>(gq * y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x],
               v_dim, &data[0], &result);
        gq = gq + (alpha_buffer_sh[threadIdx.x * m + i] -
                   result * rho_buffer[i * batchsize + batch]) *
             s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      step_vec[batch * v_dim + threadIdx.x] =
        -1.0 * gq; // copy from shared memory to global memory
    }

  
template<typename scalar_t, typename psum_t, bool rolled_ys, int FIXED_M=1>
    __global__ void lbfgs_update_buffer_and_step_compile_m(
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
      extern __shared__ float alpha_buffer_sh[];
      
      //extern __shared__ __align__(sizeof(scalar_t)) unsigned char my_smem[];
      //scalar_t *alpha_buffer_sh = reinterpret_cast<scalar_t *>(my_smem);
      
      //extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
      //float *alpha_buffer_sh = reinterpret_cast<float *>(my_smem);
      
      __shared__ psum_t data[32];    
      // temporary buffer needed for block-wide reduction
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
      reduce_v1<scalar_t, psum_t>(y * s, v_dim, &data[0], &result);
      scalar_t numerator = result;

      // scalar_t rho = 1.0/numerator;

      if (!rolled_ys)
      {

        # pragma unroll
        for (int i = 1; i < FIXED_M; i++)
        {
          s_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] =
            s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
          y_buffer[(i - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] =
            y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
        }
      }

      s_buffer[(FIXED_M - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = s;
      y_buffer[(FIXED_M - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x] = y;
      
      grad_0[batch * v_dim + threadIdx.x]                                 = gq;
      x_0[batch * v_dim +
          threadIdx.x]                                                    =
        q[batch * v_dim + threadIdx.x];

      if (threadIdx.x < FIXED_M - 1)
      {
        // m thread participate to shif the values
        // this is safe as m<32  and this happens in lockstep
        rho_buffer[threadIdx.x * batchsize + batch] =
          rho_buffer[(threadIdx.x + 1) * batchsize + batch];
      }

      if (threadIdx.x == FIXED_M - 1)
      {
        scalar_t rho = 1.0 / numerator;

        // if this is nan, make it zero:
        if (stable_mode && (numerator == 0.0))
        {
          rho = 0.0;
        }
        rho_buffer[threadIdx.x * batchsize + batch] = rho;
      }

     
      #pragma unroll
      for (int i = FIXED_M - 1; i > -1; i--)
      {
        reduce_v1<scalar_t, psum_t>(gq * s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x],
               v_dim, &data[0], &result);
        alpha_buffer_sh[threadIdx.x * FIXED_M + i] =
          result * rho_buffer[i * batchsize + batch];
        gq = gq - alpha_buffer_sh[threadIdx.x * FIXED_M + i] *
             y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      // compute var1
      reduce_v1<scalar_t, psum_t>(y * y, v_dim, &data[0], &result);
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
      for (int i = 0; i < FIXED_M; i++)
      {
        reduce_v1<scalar_t, psum_t>(gq * y_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x],
               v_dim, &data[0], &result);
        gq = gq + (alpha_buffer_sh[threadIdx.x * FIXED_M + i] -
                   result * rho_buffer[i * batchsize + batch]) *
             s_buffer[i * batchsize * v_dim + batch * v_dim + threadIdx.x];
      }

      step_vec[batch * v_dim + threadIdx.x] =
        -1.0 * gq; // copy from shared memory to global memory
    }
  } // namespace Optimization
} // namespace Curobo

std::vector<torch::Tensor>
lbfgs_cuda_fuse(torch::Tensor step_vec, torch::Tensor rho_buffer,
                torch::Tensor y_buffer, torch::Tensor s_buffer, torch::Tensor q,
                torch::Tensor grad_q, torch::Tensor x_0, torch::Tensor grad_0,
                const float epsilon, const int batch_size, const int history_m,
                const int v_dim, const bool stable_mode, const bool use_shared_buffers)
{
  using namespace Curobo::Optimization;


  // call first kernel:
  //const bool use_experimental = true;
  const bool use_fixed_m = true;

  int threadsPerBlock = v_dim;
  assert(threadsPerBlock < 1024);
  assert(history_m < 32);
  int blocksPerGrid = batch_size; 

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int smemsize = history_m * v_dim * sizeof(float);
  const int shared_buffer_smemsize = (((3 * v_dim)  + 1)  * history_m + 32) * sizeof(float);
  const int max_shared_increase = shared_buffer_smemsize;
  const int max_shared_base = 48000;
  const int max_shared_allowed = 65536; // Turing limit, others support 98304 
  int max_shared = max_shared_base;

  auto kernel_5 = Curobo::Optimization::lbfgs_update_buffer_and_step_v1_compile_m<float, float,false, 5>;
  auto kernel_6 = Curobo::Optimization::lbfgs_update_buffer_and_step_v1_compile_m<float, float,false, 6>;
  auto kernel_7 = Curobo::Optimization::lbfgs_update_buffer_and_step_v1_compile_m<float, float,false, 7>;

  auto kernel_15 = Curobo::Optimization::lbfgs_update_buffer_and_step_v1_compile_m<float, float,false, 15>;
  auto kernel_27 = Curobo::Optimization::lbfgs_update_buffer_and_step_v1_compile_m<float, float,false, 27>;

  auto kernel_31 = Curobo::Optimization::lbfgs_update_buffer_and_step_v1_compile_m<float, float,false, 31>;
  auto kernel_n = Curobo::Optimization::lbfgs_update_buffer_and_step_v1<float, float,false>;
  

  auto stable_kernel_5 = Curobo::Optimization::lbfgs_update_buffer_and_step_compile_m<float, float,false, 5>;
  auto stable_kernel_6 = Curobo::Optimization::lbfgs_update_buffer_and_step_compile_m<float, float,false, 6>;
  auto stable_kernel_7 = Curobo::Optimization::lbfgs_update_buffer_and_step_compile_m<float, float,false, 7>;

  auto stable_kernel_15 = Curobo::Optimization::lbfgs_update_buffer_and_step_compile_m<float, float,false, 15>;
  auto stable_kernel_27 = Curobo::Optimization::lbfgs_update_buffer_and_step_compile_m<float, float,false, 27>;

  auto stable_kernel_31 = Curobo::Optimization::lbfgs_update_buffer_and_step_compile_m<float, float,false, 31>;
  auto stable_kernel_n = Curobo::Optimization::lbfgs_update_buffer_and_step<float, float,false>;
  

  auto selected_kernel = kernel_n;
  auto stable_selected_kernel = stable_kernel_n;


  switch (history_m) 
  {
  
  case 5:
    selected_kernel = kernel_5;
    stable_selected_kernel = stable_kernel_5;
    break;
  case 6:
    selected_kernel = kernel_6;
    stable_selected_kernel = stable_kernel_6;
    break;
  case 7:
    selected_kernel = kernel_7;
    stable_selected_kernel = stable_kernel_7;
    break;
  
  case 15:
    selected_kernel = kernel_15;
    stable_selected_kernel = stable_kernel_15;

    break;
  case 27:
    selected_kernel = kernel_27;
    stable_selected_kernel = stable_kernel_27;

    break;
  case 31:
    selected_kernel = kernel_31;
    stable_selected_kernel = stable_kernel_31;

    break;
  }
  if (!use_fixed_m)
  {
    stable_selected_kernel = stable_kernel_n;
    selected_kernel = kernel_n;
  }
  
  // try to increase shared memory:
  // Note that this feature is only available from volta+ (cuda 7.0+)
  #if (VOLTA_PLUS)
  {
    
  
  if (use_shared_buffers && max_shared_increase > max_shared_base && max_shared_increase <= max_shared_allowed)
  { 
      max_shared = max_shared_increase;
      cudaError_t result;
      result = cudaFuncSetAttribute(selected_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_increase);
      if (result != cudaSuccess)
      {
       max_shared = max_shared_base;
      }
      
  }
  }
  #endif

  if (use_shared_buffers && shared_buffer_smemsize <= max_shared)
  {

    selected_kernel
      << < blocksPerGrid, threadsPerBlock, shared_buffer_smemsize, stream >> > (
        step_vec.data_ptr<float>(),
        rho_buffer.data_ptr<float>(),
        y_buffer.data_ptr<float>(), s_buffer.data_ptr<float>(),
        q.data_ptr<float>(), x_0.data_ptr<float>(),
        grad_0.data_ptr<float>(), grad_q.data_ptr<float>(),
        epsilon, batch_size, history_m, v_dim, stable_mode);
      
      
  }
  else 
  {

    stable_selected_kernel
  << < blocksPerGrid, threadsPerBlock, smemsize, stream >> > (
        step_vec.data_ptr<float>(),
        rho_buffer.data_ptr<float>(),
        y_buffer.data_ptr<float>(), s_buffer.data_ptr<float>(),
        q.data_ptr<float>(), x_0.data_ptr<float>(),
        grad_0.data_ptr<float>(), grad_q.data_ptr<float>(),
        epsilon, batch_size, history_m, v_dim, stable_mode);
  
  }


  
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0 };
}

