/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <torch/extension.h>

#include <vector>



// CUDA forward declarations
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
  const int batchsize
  );
}


  std::vector<torch::Tensor>
  launch_lbfgs_step(torch::Tensor step_vec,
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

  }
}

// C++ interface

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("launch_line_search", &curobo::optimization::line_search::launch_line_search);
  m.def("launch_lbfgs_step",      &curobo::optimization::launch_lbfgs_step);

}
