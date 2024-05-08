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

namespace Curobo
{
  namespace Optimization
  {
    // We launch with d_opt*cost_s1 threads.
    // We assume that cost_s2 is always 1.
    template<typename scalar_t>
    __global__ void update_best_kernel(scalar_t       *best_cost,         // 200x1
                                       scalar_t       *best_q,            // 200x7
                                       int16_t        *best_iteration,    // 200 x 1
                                       int16_t        *current_iteration, // 1
                                       const scalar_t *cost,              // 200x1
                                       const scalar_t *q,                 // 200x7
                                       const int       d_opt,             // 7
                                       const int       cost_s1,           // 200
                                       const int       cost_s2,
                                       const int       iteration,
                                       const float     delta_threshold,
                                       const float     relative_threshold) // 1
    {
      int tid  = blockIdx.x * blockDim.x + threadIdx.x;
      int size = cost_s1 * d_opt;                                          // size of best_q

      if (tid >= size)
      {
        return;
      }

      const int   cost_idx     = tid / d_opt;
      const float cost_new     = cost[cost_idx];
      const float best_cost_in = best_cost[cost_idx];
      const bool  change       = (best_cost_in - cost_new) > delta_threshold &&
                                 cost_new < best_cost_in * relative_threshold;

      if (change)
      {
        best_q[tid] = q[tid]; // update best_q

        if (tid % d_opt == 0)
        {
          best_cost[cost_idx] = cost_new; // update best_cost
          // best_iteration[cost_idx] = curr_iter + iteration; //
          // this tensor keeps track of whether the cost reduced by at least
          // delta_threshold.
          // here iteration is the last_best parameter.
        }
      }

      if (tid % d_opt == 0)
      {
        if (change)
        {
          best_iteration[cost_idx] = 0;
        }
        else
        {
          best_iteration[cost_idx] -= 1;
        }
      }

      // .if (tid == 0)
      // {
      //  curr_iter += 1;
      //  current_iteration[0] = curr_iter;
      // }
    }
  } // namespace Optimization
}   // namespace Curobo

std::vector<torch::Tensor>
update_best_cuda(torch::Tensor best_cost, torch::Tensor best_q,
                 torch::Tensor best_iteration,
                 torch::Tensor current_iteration,
                 const torch::Tensor cost,
                 const torch::Tensor q, const int d_opt, const int cost_s1,
                 const int cost_s2, const int iteration,
                 const float delta_threshold,
                 const float relative_threshold = 0.999)
{
  using namespace Curobo::Optimization;
  const int threadsPerBlock = 128;
  const int cost_size       = cost_s1 * d_opt;
  assert(cost_s2 == 1); // assumption
  const int blocksPerGrid = (cost_size + threadsPerBlock - 1) / threadsPerBlock;

  // printf("cost_s1=%d, d_opt=%d, blocksPerGrid=%d\n", cost_s1, d_opt,
  // blocksPerGrid);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    cost.scalar_type(), "update_best_cu", ([&] {
    update_best_kernel<scalar_t>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
      best_cost.data_ptr<scalar_t>(), best_q.data_ptr<scalar_t>(),
      best_iteration.data_ptr<int16_t>(),
      current_iteration.data_ptr<int16_t>(),
      cost.data_ptr<scalar_t>(),
      q.data_ptr<scalar_t>(), d_opt, cost_s1, cost_s2, iteration,
      delta_threshold, relative_threshold);
  }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { best_cost, best_q, best_iteration };
}
