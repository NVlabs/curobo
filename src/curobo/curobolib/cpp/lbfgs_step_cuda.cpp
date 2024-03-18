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
                const bool    stable_mode,
                const bool use_shared_buffers);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), # x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), # x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
lbfgs_call(torch::Tensor step_vec, torch::Tensor rho_buffer,
           torch::Tensor y_buffer, torch::Tensor s_buffer, torch::Tensor q,
           torch::Tensor grad_q, torch::Tensor x_0, torch::Tensor grad_0,
           const float epsilon, const int batch_size, const int m,
           const int v_dim, const bool stable_mode, const bool use_shared_buffers)
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
                         stable_mode, use_shared_buffers);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
 
  m.def("forward",      &lbfgs_call,        "L-BFGS Update + Step (CUDA)");
}
