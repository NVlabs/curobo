/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <torch/extension.h>

namespace curobo {

namespace trajectory{

namespace legacy{
void launch_differentiation_position_forward_kernel(
  torch::Tensor       out_position,
  torch::Tensor       out_velocity,
  torch::Tensor       out_acceleration,
  torch::Tensor       out_jerk,
  torch::Tensor       out_dt,
  const torch::Tensor u_position,
  const torch::Tensor start_position,
  const torch::Tensor start_velocity,
  const torch::Tensor start_acceleration,
  const torch::Tensor goal_position,
  const torch::Tensor goal_velocity,
  const torch::Tensor goal_acceleration,
  const torch::Tensor start_idx,
  const torch::Tensor goal_idx,
  const torch::Tensor traj_dt,
  const torch::Tensor use_implicit_goal_state,
  const int           batch_size,
  const int           horizon,
  const int           dof);

void launch_differentiation_position_backward_kernel(
  torch::Tensor       out_grad_position,
  const torch::Tensor grad_position,
  const torch::Tensor grad_velocity,
  const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk,
  const torch::Tensor traj_dt,
  const torch::Tensor dt_idx,
  const torch::Tensor use_implicit_goal_state,
  const int           batch_size,
  const int           horizon,
  const int           dof);


void launch_integration_acceleration_kernel(
  torch::Tensor       out_position,
  torch::Tensor       out_velocity,
  torch::Tensor       out_acceleration,
  torch::Tensor       out_jerk,
  const torch::Tensor u_acc,
  const torch::Tensor start_position,
  const torch::Tensor start_velocity,
  const torch::Tensor start_acceleration,
  const torch::Tensor start_idx,
  const torch::Tensor traj_dt,
  const int           batch_size,
  const int           horizon,
  const int           dof,
  const bool          use_rk2 = true);

}

namespace bspline{

void launch_bspline_interpolation_forward_kernel(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  torch::Tensor out_dt,
  const torch::Tensor u_position,
  const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor start_jerk,
  const torch::Tensor goal_position,
  const torch::Tensor goal_velocity, const torch::Tensor goal_acceleration,
  const torch::Tensor goal_jerk,
  const torch::Tensor start_idx,
  const torch::Tensor goal_idx,
  const torch::Tensor traj_dt,
  const torch::Tensor use_implicit_goal_state,
  const int batch_size, const int horizon, const int dof,
  const int n_knots,
  const int bspline_degree);

void launch_bspline_interpolation_backward_kernel(
  torch::Tensor out_grad_position, const torch::Tensor grad_position,
  const torch::Tensor grad_velocity, const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk, const torch::Tensor traj_dt,
  const torch::Tensor dt_idx,
  const torch::Tensor use_implicit_goal_state,
  const int batch_size, const int padded_horizon, const int dof,
  const int n_knots,
  const int bspline_degree,
  const bool use_direct_polynomial);


void launch_bspline_interpolation_single_dt_kernel(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  torch::Tensor out_dt,
  const torch::Tensor knots,
  const torch::Tensor knot_dt,
  const torch::Tensor start_position,
  const torch::Tensor start_velocity,
  const torch::Tensor start_acceleration,
  const torch::Tensor start_jerk,
  const torch::Tensor goal_position,
  const torch::Tensor goal_velocity,
  const torch::Tensor goal_acceleration,
  const torch::Tensor goal_jerk,
  const torch::Tensor start_idx,
  const torch::Tensor goal_idx,
  const torch::Tensor interpolation_dt,
  const torch::Tensor use_implicit_goal_state,
  const torch::Tensor interpolation_horizon,
  const int batch_size,
  const int max_out_tsteps,
  const int dof,
  const int n_knots,
  const int bspline_degree);

}


}

}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

  m.def("launch_differentiation_position_forward_kernel",      &curobo::trajectory::legacy::launch_differentiation_position_forward_kernel);
  m.def("launch_differentiation_position_backward_kernel", &curobo::trajectory::legacy::launch_differentiation_position_backward_kernel);
  m.def("launch_integration_acceleration_kernel",   &curobo::trajectory::legacy::launch_integration_acceleration_kernel);
  m.def("launch_bspline_interpolation_forward_kernel",         &curobo::trajectory::bspline::launch_bspline_interpolation_forward_kernel);
  m.def("launch_bspline_interpolation_backward_kernel",        &curobo::trajectory::bspline::launch_bspline_interpolation_backward_kernel);
  m.def("launch_bspline_interpolation_single_dt_kernel",        &curobo::trajectory::bspline::launch_bspline_interpolation_single_dt_kernel);

}
