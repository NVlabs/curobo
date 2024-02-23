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
std::vector<torch::Tensor>reduce_cuda(torch::Tensor vec,
                                      torch::Tensor vec2,
                                      torch::Tensor rho_buffer,
                                      torch::Tensor sum,
                                      const int     batch_size,
                                      const int     v_dim,
                                      const int     m);

std::vector<torch::Tensor>
lbfgs_step_cuda(torch::Tensor step_vec,
                torch::Tensor rho_buffer,
                torch::Tensor y_buffer,
                torch::Tensor s_buffer,
                torch::Tensor grad_q,
                const float   epsilon,
                const int     batch_size,
                const int     m,
                const int     v_dim);

std::vector<torch::Tensor>
lbfgs_update_cuda(torch::Tensor rho_buffer,
                  torch::Tensor y_buffer,
                  torch::Tensor s_buffer,
                  torch::Tensor q,
                  torch::Tensor grad_q,
                  torch::Tensor x_0,
                  torch::Tensor grad_0,
                  const int     batch_size,
                  const int     m,
                  const int     v_dim);

std::vector<torch::Tensor>
lbfgs_cuda_fuse(torch::Tensor step_vec,
                torch::Tensor rho_buffer,
                torch::Tensor y_buffer,
                torch::Tensor s_buffer,
                torch::Tensor q,
                torch::Tensor grad_q,
                torch::Tensor x_0,
                torch::Tensor grad_0,
                const float   epsilon,
                const int     batch_size,
                const int     m,
                const int     v_dim,
                const bool    stable_mode);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), # x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), # x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
lbfgs_step_call(torch::Tensor step_vec, torch::Tensor rho_buffer,
                torch::Tensor y_buffer, torch::Tensor s_buffer,
                torch::Tensor grad_q, const float epsilon, const int batch_size,
                const int m, const int v_dim)
{
  CHECK_INPUT(step_vec);
  CHECK_INPUT(rho_buffer);
  CHECK_INPUT(y_buffer);
  CHECK_INPUT(s_buffer);
  CHECK_INPUT(grad_q);
  const at::cuda::OptionalCUDAGuard guard(grad_q.device());

  return lbfgs_step_cuda(step_vec, rho_buffer, y_buffer, s_buffer, grad_q,
                         epsilon, batch_size, m, v_dim);
}

std::vector<torch::Tensor>
lbfgs_update_call(torch::Tensor rho_buffer, torch::Tensor y_buffer,
                  torch::Tensor s_buffer, torch::Tensor q, torch::Tensor grad_q,
                  torch::Tensor x_0, torch::Tensor grad_0, const int batch_size,
                  const int m, const int v_dim)
{
  CHECK_INPUT(rho_buffer);
  CHECK_INPUT(y_buffer);
  CHECK_INPUT(s_buffer);
  CHECK_INPUT(grad_q);
  CHECK_INPUT(x_0);
  CHECK_INPUT(grad_0);
  CHECK_INPUT(q);
  const at::cuda::OptionalCUDAGuard guard(grad_q.device());

  return lbfgs_update_cuda(rho_buffer, y_buffer, s_buffer, q, grad_q, x_0,
                           grad_0, batch_size, m, v_dim);
}

std::vector<torch::Tensor>
reduce_cuda_call(torch::Tensor vec, torch::Tensor vec2,
                 torch::Tensor rho_buffer, torch::Tensor sum,
                 const int batch_size, const int v_dim, const int m)
{
  const at::cuda::OptionalCUDAGuard guard(sum.device());

  return reduce_cuda(vec, vec2, rho_buffer, sum, batch_size, v_dim, m);
}

std::vector<torch::Tensor>
lbfgs_call(torch::Tensor step_vec, torch::Tensor rho_buffer,
           torch::Tensor y_buffer, torch::Tensor s_buffer, torch::Tensor q,
           torch::Tensor grad_q, torch::Tensor x_0, torch::Tensor grad_0,
           const float epsilon, const int batch_size, const int m,
           const int v_dim, const bool stable_mode)
{
  CHECK_INPUT(step_vec);
  CHECK_INPUT(rho_buffer);
  CHECK_INPUT(y_buffer);
  CHECK_INPUT(s_buffer);
  CHECK_INPUT(grad_q);
  CHECK_INPUT(x_0);
  CHECK_INPUT(grad_0);
  CHECK_INPUT(q);
  const at::cuda::OptionalCUDAGuard guard(grad_q.device());

  return lbfgs_cuda_fuse(step_vec, rho_buffer, y_buffer, s_buffer, q, grad_q,
                         x_0, grad_0, epsilon, batch_size, m, v_dim,
                         stable_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("step",         &lbfgs_step_call,   "L-BFGS step (CUDA)");
  m.def("update",       &lbfgs_update_call, "L-BFGS Update (CUDA)");
  m.def("forward",      &lbfgs_call,        "L-BFGS Update + Step (CUDA)");
  m.def("debug_reduce", &reduce_cuda_call,  "L-BFGS Debug");
}
