/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
 #include <cuda.h>
 #include <torch/extension.h>
 #include <vector>

 #include <c10/cuda/CUDAStream.h>

 // #include "helper_cuda.h"

 #include <assert.h>
 #include <cstdio>
 #include <cstdlib>
 #include <ctime>
 #include <math.h>
 #include "common/torch_cuda_utils.h"
 #include "line_search_kernel.cuh"




 namespace curobo{
   namespace optimization{
    namespace line_search{




 void launch_line_search(
   torch::Tensor best_cost,
   torch::Tensor best_action,
   torch::Tensor best_iteration,
   torch::Tensor current_iteration,
   torch::Tensor converged_global,
   const int convergence_iteration,
   const float cost_delta_threshold,
   const float cost_relative_threshold,

   torch::Tensor exploration_cost,
   torch::Tensor exploration_action,
   torch::Tensor exploration_gradient,
   torch::Tensor exploration_idx,
   torch::Tensor selected_cost,
   torch::Tensor selected_action,
   torch::Tensor selected_gradient,
   torch::Tensor selected_idx,

   const torch::Tensor search_cost,
   const torch::Tensor search_action,
   const torch::Tensor search_gradient,
   const torch::Tensor step_direction,
   const torch::Tensor search_magnitudes,
   const float armijo_threshold_c_1,
   const float curvature_threshold_c_2,
   const bool strong_wolfe,
   const bool approx_wolfe,
   const int n_linesearch,
   const int opt_dim,
   const int batchsize)
 {
   assert(opt_dim <= 1024);
   assert(n_linesearch <= curobo::common::warpSize);
   assert(n_linesearch <= opt_dim);
   curobo::common::validate_cuda_input(best_cost, "best_cost");
   curobo::common::validate_cuda_input(best_action, "best_action");
   curobo::common::validate_cuda_input(best_iteration, "best_iteration");
   curobo::common::validate_cuda_input(current_iteration, "current_iteration");
   curobo::common::validate_cuda_input(converged_global, "converged_global");
   curobo::common::validate_cuda_input(exploration_cost, "exploration_cost");
   curobo::common::validate_cuda_input(exploration_action, "exploration_action");
   curobo::common::validate_cuda_input(exploration_gradient, "exploration_gradient");
   curobo::common::validate_cuda_input(exploration_idx, "exploration_idx");
   curobo::common::validate_cuda_input(selected_cost, "selected_cost");
   curobo::common::validate_cuda_input(selected_action, "selected_action");
   curobo::common::validate_cuda_input(selected_gradient, "selected_gradient");
   curobo::common::validate_cuda_input(selected_idx, "selected_idx");
   curobo::common::validate_cuda_input(search_cost, "search_cost");
   curobo::common::validate_cuda_input(search_action, "search_action");
   curobo::common::validate_cuda_input(search_gradient, "search_gradient");
   curobo::common::validate_cuda_input(step_direction, "step_direction");
   curobo::common::validate_cuda_input(search_magnitudes, "search_magnitudes");


   int threadsPerBlock = opt_dim;



   const int blocksPerGrid   = batchsize;

   cudaStream_t stream = at::cuda::getCurrentCUDAStream();
   auto selected_kernel = n_linesearch == 4 ? kernel_line_search<float, 4> : kernel_line_search<float, -1>;

   selected_kernel
     <<< blocksPerGrid, threadsPerBlock, 0, stream >>> (
       best_cost.data_ptr<float>(),
       best_action.data_ptr<float>(),
       best_iteration.data_ptr<int16_t>(),
       current_iteration.data_ptr<int16_t>(),
       converged_global.data_ptr<uint8_t>(),
       convergence_iteration,
       cost_delta_threshold,
       cost_relative_threshold,
       exploration_cost.data_ptr<float>(),
       exploration_action.data_ptr<float>(),
       exploration_gradient.data_ptr<float>(),
       exploration_idx.data_ptr<int32_t>(),
       selected_cost.data_ptr<float>(),
       selected_action.data_ptr<float>(),
       selected_gradient.data_ptr<float>(),
       selected_idx.data_ptr<int32_t>(),
       search_cost.data_ptr<float>(),
       search_action.data_ptr<float>(),
       search_gradient.data_ptr<float>(),
       step_direction.data_ptr<float>(),
       search_magnitudes.data_ptr<float>(),
       armijo_threshold_c_1,
       curvature_threshold_c_2,
       strong_wolfe,
       approx_wolfe,
       n_linesearch,
       opt_dim,
       batchsize);



   C10_CUDA_KERNEL_LAUNCH_CHECK();
   return;
 }

   }
 }
}