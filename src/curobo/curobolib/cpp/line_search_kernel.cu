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
#include <cub/cub.cuh>
#include <math.h>

// #include <stdio.h>
//
// For the CUDA runtime routines (prefixed with "cuda_")
// #include <cuda_runtime.h>

// #include <cuda_fp16.h>
// #include <helper_cuda.h>

#define FULL_MASK 0xffffffff

namespace Curobo
{
  namespace Optimization
  {
    template<typename scalar_t, typename psum_t>
    __inline__ __device__ void reduce(scalar_t v, int m, unsigned mask,
                                      psum_t *data, scalar_t *result)
    {
      psum_t val = v;

      val += __shfl_down_sync(mask, val, 1);
      val += __shfl_down_sync(mask, val, 2);
      val += __shfl_down_sync(mask, val, 4);
      val += __shfl_down_sync(mask, val, 8);
      val += __shfl_down_sync(mask, val, 16);

      // int leader = __ffs(mask) – 1;    // select a leader lane
      int leader = 0;

      if (threadIdx.x % 32 == leader)
      {
        if (m <= 32)
        {
          result[0] = (scalar_t)val;
        }
        else
        {
          data[(threadIdx.x + 1) / 32] = val;
        }
      }

      if (m > 32)
      {
        __syncthreads();

        int elems = (m + 31) / 32;
        assert(elems <= 32);
        unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);

        if (threadIdx.x < elems) // only the first warp will do this work
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

    // Launched with l2 threads/block and batchsize blocks
    template<typename scalar_t, typename psum_t>
    __global__ void line_search_kernel(

      // int64_t *m_idx, // 4x1x1
      scalar_t *best_x,           // 4x280
      scalar_t *best_c,           // 4x1
      scalar_t *best_grad,        // 4x280
      const scalar_t *g_x,        //  4x6x280
      const scalar_t *x_set,      //  4x6x280
      const scalar_t *step_vec,   // 4x280x1
      const scalar_t *c,          // 4x6x1
      const scalar_t *alpha_list, // 4x6x1
      const int64_t *c_idx,       // 4x1x1
      const float c_1, const float c_2, const bool strong_wolfe,
      const bool approx_wolfe,
      const int l1,               // 6
      const int l2,               // 280
      const int batchsize)        // 4
    {
      int batch = blockIdx.x;
      __shared__ psum_t   data[32];
      __shared__ scalar_t result[32];

      assert(l1 <= 32);
      unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < l2);

      if (threadIdx.x >= l2)
      {
        return;
      }

      scalar_t sv_elem = step_vec[batch * l2 + threadIdx.x];

      // g_step = g0 @ step_vec_T
      // g_x @ step_vec_T
      for (int i = 0; i < l1; i++)
      {
        reduce(g_x[batch * l1 * l2 + l2 * i + threadIdx.x] * sv_elem, l2, mask,
               &data[0], &result[i]);
      }

      __shared__ scalar_t step_success[32];
      __shared__ scalar_t step_success_w1[32];
      assert(blockDim.x >= l1);
      bool wolfe_1   = false;
      bool wolfe     = false;
      bool condition = threadIdx.x < l1;

      if (condition)
      {
        // scalar_t alpha_list_elem = alpha_list[batch*l1 + threadIdx.x];
        scalar_t alpha_list_elem = alpha_list[threadIdx.x];

        // condition 1:
        wolfe_1 = c[batch * l1 + threadIdx.x] <=
                  (c[batch * l1] + c_1 * alpha_list_elem * result[0]);

        // condition 2:
        bool wolfe_2;

        if (strong_wolfe)
        {
          wolfe_2 = abs(result[threadIdx.x]) <= c_2 *abs(result[0]);
        }
        else
        {
          wolfe_2 = result[threadIdx.x] >= c_2 * result[0];
        }

        wolfe = wolfe_1 & wolfe_2;

        step_success[threadIdx.x]    = wolfe * (alpha_list_elem + 0.1);
        step_success_w1[threadIdx.x] = wolfe_1 * (alpha_list_elem + 0.1);
      }

      __syncthreads();

      __shared__ int idx_shared;

      if (threadIdx.x == 0)
      {
        int m_id      = 0;
        int m1_id     = 0;
        scalar_t max1 = step_success[0];
        scalar_t max2 = step_success_w1[0];

        for (int i = 1; i < l1; i++)
        {
          if (max1 < step_success[i])
          {
            max1 = step_success[i];
            m_id = i;
          }

          if (max2 < step_success_w1[i])
          {
            max2  = step_success_w1[i];
            m1_id = i;
          }
        }

        if (!approx_wolfe)
        {
          //   m_idx = torch.where(m_idx == 0, m1_idx, m_idx)
          if (m_id == 0)
          {
            m_id = m1_id;
          }

          // m_idx[m_idx == 0] = 1
          if (m_id == 0)
          {
            m_id = 1;
          }
        }
        idx_shared = m_id + c_idx[batch];
      }

      ////////////////////////////////////
      // write outputs using the computed index.
      // one index per batch is computed
      ////////////////////////////////////
      // l2 is d_opt, l1 is line_search n.
      // idx_shared contains index in l1
      //
      __syncthreads();

      if (threadIdx.x < l2)
      {
        if (threadIdx.x == 0)
        {
          // printf("block: %d, idx_shared: %d\n", batch, idx_shared);
        }
        best_x[batch * l2 + threadIdx.x]    = x_set[idx_shared * l2 + threadIdx.x];
        best_grad[batch * l2 + threadIdx.x] = g_x[idx_shared * l2 + threadIdx.x];
      }

      if (threadIdx.x == 0)
      {
        best_c[batch] = c[idx_shared];
      }
    }

    // Launched with l2 threads/block and #blocks = batchsize
    template<typename scalar_t, typename psum_t>
    __global__ void line_search_kernel_mask(

      // int64_t *m_idx, // 4x1x1
      scalar_t *best_x,           // 4x280
      scalar_t *best_c,           // 4x1
      scalar_t *best_grad,        // 4x280
      const scalar_t *g_x,        //  4x6x280
      const scalar_t *x_set,      //  4x6x280
      const scalar_t *step_vec,   // 4x280x1
      const scalar_t *c,          // 4x6x1
      const scalar_t *alpha_list, // 4x6x1
      const int64_t *c_idx,       // 4x1x1
      const float c_1, const float c_2, const bool strong_wolfe,
      const bool approx_wolfe,
      const int l1,               // 6
      const int l2,               // 280
      const int batchsize)        // 4
    {
      int batch = blockIdx.x;
      __shared__ psum_t   data[32];
      __shared__ scalar_t result[32];

      assert(l1 <= 32);
      unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < l2);

      if (threadIdx.x >= l2)
      {
        return;
      }

      scalar_t sv_elem = step_vec[batch * l2 + threadIdx.x];

      // g_step = g0 @ step_vec_T
      // g_x @ step_vec_T
      for (int i = 0; i < l1; i++)
      {
        reduce(g_x[batch * l1 * l2 + l2 * i + threadIdx.x] * sv_elem, l2, mask,
               &data[0], &result[i]);
      }

      // __shared__ scalar_t step_success[32];
      // __shared__ scalar_t step_success_w1[32];
      assert(blockDim.x >= l1);
      bool wolfe_1   = false;
      bool wolfe     = false;
      bool condition = threadIdx.x < l1;

      if (condition)
      {
        scalar_t alpha_list_elem = alpha_list[threadIdx.x];

        // scalar_t alpha_list_elem = alpha_list[batch*l1 + threadIdx.x];

        // condition 1:
        wolfe_1 = c[batch * l1 + threadIdx.x] <=
                  (c[batch * l1] + c_1 * alpha_list_elem * result[0]);

        // condition 2:
        bool wolfe_2;

        if (strong_wolfe)
        {
          wolfe_2 = abs(result[threadIdx.x]) <= c_2 *abs(result[0]);
        }
        else
        {
          wolfe_2 = result[threadIdx.x] >= c_2 * result[0];
        }

        // wolfe = torch.logical_and(wolfe_1, wolfe_2)
        wolfe = wolfe_1 & wolfe_2;

        // // step_success = wolfe * (self.alpha_list[:, :, 0:1] + 0.1)
        // // step_success_w1 = wolfe_1 * (self.alpha_list[:, :, 0:1] + 0.1)
        // step_success[threadIdx.x]    = wolfe   * (alpha_list_elem + 0.1);
        // step_success_w1[threadIdx.x] = wolfe_1 * (alpha_list_elem + 0.1);
      }
      unsigned msk1 = __ballot_sync(FULL_MASK, wolfe_1 & condition);
      unsigned msk  = __ballot_sync(FULL_MASK, wolfe & condition);

      // get the index of the last occurance of true
      unsigned msk1_brev = __brev(msk1);
      unsigned msk_brev  = __brev(msk);

      int id1 = 32 - __ffs(msk1_brev); // position of least signficant bit set to 1
      int id  = 32 - __ffs(msk_brev);  // position of least signficant bit set to 1

      __syncthreads();

      __shared__ int idx_shared;

      if (threadIdx.x == 0)
      {
        if (!approx_wolfe)
        {
          if (id == 32) // msk is zero
          {
            id = id1;
          }

          if (id == 0) // bit 0 is set
          {
            id = id1;
          }

          if (id == 32) // msk is zero
          {
            id = 1;
          }

          if (id == 0)
          {
            id = 1;
          }
        }
        else
        {
          if (id == 32) // msk is zero
          {
            id = 0;
          }
        }

        // //  _, m_idx = torch.max(step_success, dim=-2)
        // //  _, m1_idx = torch.max(step_success_w1, dim=-2)
        // int m_id = 0;
        // int m1_id = 0;
        // scalar_t max1 = step_success[0];
        // scalar_t max2 = step_success_w1[0];
        // for (int i=1; i<l1; i++) {
        //   if (max1<step_success[i]) {
        //     max1 = step_success[i];
        //     m_id = i;
        //   }
        //   if (max2<step_success_w1[i]) {
        //     max2 = step_success_w1[i];
        //     m1_id = i;
        //   }
        // }

        // //   m_idx = torch.where(m_idx == 0, m1_idx, m_idx)
        // if (m_id == 0) {
        //     m_id = m1_id;
        // }

        // // m_idx[m_idx == 0] = 1
        // if (m_id == 0) {
        //     m_id = 1;
        // }

        // if (id != m_id) {
        //   printf("id=%d, m_id=%d\n", id, m_id);
        //   printf("msk1=%x, msk=%x, raw id1=%d, raw id=%d\n", msk1, msk,
        //   32-__ffs(msk1_brev), 32-__ffs(msk_brev));
        // }

        // m_idx[batch] = m_id;
        //  m_idx[batch] = id;
        idx_shared = id + c_idx[batch];
      }

      ////////////////////////////////////
      // write outputs using the computed index.
      // one index per batch is computed
      ////////////////////////////////////
      __syncthreads();

      if (threadIdx.x < l2)
      {
        if (threadIdx.x == 0)
        {
          // printf("block: %d, idx_shared: %d\n", batch, idx_shared);
        }
        best_x[batch * l2 + threadIdx.x]    = x_set[idx_shared * l2 + threadIdx.x];
        best_grad[batch * l2 + threadIdx.x] = g_x[idx_shared * l2 + threadIdx.x];
      }

      if (threadIdx.x == 0)
      {
        best_c[batch] = c[idx_shared];
      }
    }
  } // namespace Optimization
}   // namespace Curobo
std::vector<torch::Tensor>line_search_cuda(

  // torch::Tensor m_idx,
  torch::Tensor best_x, torch::Tensor best_c, torch::Tensor best_grad,
  const torch::Tensor g_x, const torch::Tensor x_set,
  const torch::Tensor step_vec, const torch::Tensor c_0,
  const torch::Tensor alpha_list, const torch::Tensor c_idx, const float c_1,
  const float c_2, const bool strong_wolfe, const bool approx_wolfe,
  const int l1, const int l2, const int batchsize)
{
  using namespace Curobo::Optimization;
  assert(l2 <= 1024);

  // multiple of 32
  const int threadsPerBlock = 32 * ((l2 + 31) / 32); //  l2;
  const int blocksPerGrid   = batchsize;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    g_x.scalar_type(), "line_search_cu", ([&] {
    line_search_kernel_mask<scalar_t, scalar_t>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (

      // m_idx.data_ptr<int>(),
      best_x.data_ptr<scalar_t>(), best_c.data_ptr<scalar_t>(),
      best_grad.data_ptr<scalar_t>(), g_x.data_ptr<scalar_t>(),
      x_set.data_ptr<scalar_t>(), step_vec.data_ptr<scalar_t>(),
      c_0.data_ptr<scalar_t>(), alpha_list.data_ptr<scalar_t>(),
      c_idx.data_ptr<int64_t>(), c_1, c_2, strong_wolfe, approx_wolfe,
      l1, l2, batchsize);
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { best_x, best_c, best_grad };
}
