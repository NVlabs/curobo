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

#include <torch/extension.h>

#include <vector>

#include <c10/cuda/CUDAGuard.h>

// CUDA forward declarations

std::vector<torch::Tensor>
update_best_cuda(torch::Tensor       best_cost,
                 torch::Tensor       best_q,
                 torch::Tensor       best_iteration,
                 torch::Tensor       current_iteration,
                 const torch::Tensor cost,
                 const torch::Tensor q,
                 const int           d_opt,
                 const int           cost_s1,
                 const int           cost_s2,
                 const int           iteration,
                 const float         delta_threshold,
                 const float         relative_threshold = 0.999);

std::vector<torch::Tensor>line_search_cuda(

  // torch::Tensor m,
  torch::Tensor       best_x,
  torch::Tensor       best_c,
  torch::Tensor       best_grad,
  const torch::Tensor g_x,
  const torch::Tensor x_set,
  const torch::Tensor step_vec,
  const torch::Tensor c_0,
  const torch::Tensor alpha_list,
  const torch::Tensor c_idx,
  const float         c_1,
  const float         c_2,
  const bool          strong_wolfe,
  const bool          approx_wolfe,
  const int           l1,
  const int           l2,
  const int           batchsize);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), # x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), # x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor>line_search_call(

  // torch::Tensor m,
  torch::Tensor best_x, torch::Tensor best_c, torch::Tensor best_grad,
  const torch::Tensor g_x, const torch::Tensor x_set,
  const torch::Tensor step_vec, const torch::Tensor c_0,
  const torch::Tensor alpha_list, const torch::Tensor c_idx, const float c_1,
  const float c_2, const bool strong_wolfe, const bool approx_wolfe,
  const int l1, const int l2, const int batchsize)
{
  CHECK_INPUT(g_x);
  CHECK_INPUT(x_set);
  CHECK_INPUT(step_vec);
  CHECK_INPUT(c_0);
  CHECK_INPUT(alpha_list);
  CHECK_INPUT(c_idx);

  // CHECK_INPUT(m);
  CHECK_INPUT(best_x);
  CHECK_INPUT(best_c);
  CHECK_INPUT(best_grad);
  const at::cuda::OptionalCUDAGuard guard(best_x.device());

  // return line_search_cuda(m, g_x, step_vec, c_0, alpha_list, c_1, c_2,
  // strong_wolfe, l1, l2, batchsize);
  return line_search_cuda(best_x, best_c, best_grad, g_x, x_set, step_vec, c_0,
                          alpha_list, c_idx, c_1, c_2, strong_wolfe,
                          approx_wolfe, l1, l2, batchsize);
}

std::vector<torch::Tensor>
update_best_call(torch::Tensor best_cost, torch::Tensor best_q,
                 torch::Tensor best_iteration,
                 torch::Tensor current_iteration,
                 const torch::Tensor cost,
                 const torch::Tensor q, const int d_opt, const int cost_s1,
                 const int cost_s2, const int iteration,
                 const float delta_threshold,
                 const float relative_threshold = 0.999)
{
  CHECK_INPUT(best_cost);
  CHECK_INPUT(best_q);
  CHECK_INPUT(cost);
  CHECK_INPUT(q);
  CHECK_INPUT(current_iteration);
  CHECK_INPUT(best_iteration);
  const at::cuda::OptionalCUDAGuard guard(cost.device());

  return update_best_cuda(best_cost, best_q, best_iteration, current_iteration, cost, q, d_opt,
                          cost_s1, cost_s2, iteration, delta_threshold, relative_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("update_best", &update_best_call, "Update Best (CUDA)");
  m.def("line_search", &line_search_call, "Line search (CUDA)");
}
