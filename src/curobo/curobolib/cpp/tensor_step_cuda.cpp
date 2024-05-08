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

#include <c10/cuda/CUDAGuard.h>
#include <vector>

std::vector<torch::Tensor>step_position_clique(
  torch::Tensor       out_position,
  torch::Tensor       out_velocity,
  torch::Tensor       out_acceleration,
  torch::Tensor       out_jerk,
  const torch::Tensor u_position,
  const torch::Tensor start_position,
  const torch::Tensor start_velocity,
  const torch::Tensor start_acceleration,
  const torch::Tensor traj_dt,
  const int           batch_size,
  const int           horizon,
  const int           dof);
std::vector<torch::Tensor>step_position_clique2(
  torch::Tensor       out_position,
  torch::Tensor       out_velocity,
  torch::Tensor       out_acceleration,
  torch::Tensor       out_jerk,
  const torch::Tensor u_position,
  const torch::Tensor start_position,
  const torch::Tensor start_velocity,
  const torch::Tensor start_acceleration,
  const torch::Tensor traj_dt,
  const int           batch_size,
  const int           horizon,
  const int           dof,
  const int           mode);
std::vector<torch::Tensor>step_position_clique2_idx(
  torch::Tensor       out_position,
  torch::Tensor       out_velocity,
  torch::Tensor       out_acceleration,
  torch::Tensor       out_jerk,
  const torch::Tensor u_position,
  const torch::Tensor start_position,
  const torch::Tensor start_velocity,
  const torch::Tensor start_acceleration,
  const torch::Tensor start_idx,
  const torch::Tensor traj_dt,
  const int           batch_size,
  const int           horizon,
  const int           dof,
  const int           mode);

std::vector<torch::Tensor>backward_step_position_clique(
  torch::Tensor       out_grad_position,
  const torch::Tensor grad_position,
  const torch::Tensor grad_velocity,
  const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk,
  const torch::Tensor traj_dt,
  const int           batch_size,
  const int           horizon,
  const int           dof);
std::vector<torch::Tensor>backward_step_position_clique2(
  torch::Tensor       out_grad_position,
  const torch::Tensor grad_position,
  const torch::Tensor grad_velocity,
  const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk,
  const torch::Tensor traj_dt,
  const int           batch_size,
  const int           horizon,
  const int           dof,
  const int           mode);

std::vector<torch::Tensor>
step_acceleration(torch::Tensor       out_position,
                  torch::Tensor       out_velocity,
                  torch::Tensor       out_acceleration,
                  torch::Tensor       out_jerk,
                  const torch::Tensor u_acc,
                  const torch::Tensor start_position,
                  const torch::Tensor start_velocity,
                  const torch::Tensor start_acceleration,
                  const torch::Tensor traj_dt,
                  const int           batch_size,
                  const int           horizon,
                  const int           dof,
                  const bool          use_rk2 = true);

std::vector<torch::Tensor>step_acceleration_idx(
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

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), # x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), # x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>step_position_clique_wrapper(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_position, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor traj_dt, const int batch_size, const int horizon,
  const int dof)
{
  const at::cuda::OptionalCUDAGuard guard(u_position.device());

  assert(false); // not supported
  CHECK_INPUT(u_position);
  CHECK_INPUT(out_position);
  CHECK_INPUT(out_velocity);
  CHECK_INPUT(out_acceleration);
  CHECK_INPUT(out_jerk);
  CHECK_INPUT(start_position);
  CHECK_INPUT(start_velocity);
  CHECK_INPUT(start_acceleration);
  CHECK_INPUT(traj_dt);

  return step_position_clique(out_position, out_velocity, out_acceleration,
                              out_jerk, u_position, start_position,
                              start_velocity, start_acceleration, traj_dt,
                              batch_size, horizon, dof);
}

std::vector<torch::Tensor>step_position_clique2_wrapper(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_position, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor traj_dt, const int batch_size, const int horizon,
  const int dof,
  const int mode)
{
  const at::cuda::OptionalCUDAGuard guard(u_position.device());

  CHECK_INPUT(u_position);
  CHECK_INPUT(out_position);
  CHECK_INPUT(out_velocity);
  CHECK_INPUT(out_acceleration);
  CHECK_INPUT(out_jerk);
  CHECK_INPUT(start_position);
  CHECK_INPUT(start_velocity);
  CHECK_INPUT(start_acceleration);
  CHECK_INPUT(traj_dt);

  return step_position_clique2(out_position, out_velocity, out_acceleration,
                               out_jerk, u_position, start_position,
                               start_velocity, start_acceleration, traj_dt,
                               batch_size, horizon, dof, mode);
}

std::vector<torch::Tensor>step_position_clique2_idx_wrapper(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_position, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor start_idx, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof,
  const int mode)
{
  const at::cuda::OptionalCUDAGuard guard(u_position.device());

  CHECK_INPUT(u_position);
  CHECK_INPUT(out_position);
  CHECK_INPUT(out_velocity);
  CHECK_INPUT(out_acceleration);
  CHECK_INPUT(out_jerk);
  CHECK_INPUT(start_position);
  CHECK_INPUT(start_velocity);
  CHECK_INPUT(start_acceleration);
  CHECK_INPUT(traj_dt);
  CHECK_INPUT(start_idx);

  return step_position_clique2_idx(
    out_position, out_velocity, out_acceleration, out_jerk, u_position,
    start_position, start_velocity, start_acceleration, start_idx, traj_dt,
    batch_size, horizon, dof, mode);
}

std::vector<torch::Tensor>backward_step_position_clique_wrapper(
  torch::Tensor out_grad_position, const torch::Tensor grad_position,
  const torch::Tensor grad_velocity, const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof)
{
  const at::cuda::OptionalCUDAGuard guard(grad_position.device());

  assert(false); // not supported
  CHECK_INPUT(out_grad_position);
  CHECK_INPUT(grad_position);
  CHECK_INPUT(grad_velocity);
  CHECK_INPUT(grad_acceleration);
  CHECK_INPUT(grad_jerk);
  CHECK_INPUT(traj_dt);

  return backward_step_position_clique(
    out_grad_position, grad_position, grad_velocity, grad_acceleration,
    grad_jerk, traj_dt, batch_size, horizon, dof);
}

std::vector<torch::Tensor>backward_step_position_clique2_wrapper(
  torch::Tensor out_grad_position, const torch::Tensor grad_position,
  const torch::Tensor grad_velocity, const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof,
  const int mode)
{
  const at::cuda::OptionalCUDAGuard guard(grad_position.device());

  CHECK_INPUT(out_grad_position);
  CHECK_INPUT(grad_position);
  CHECK_INPUT(grad_velocity);
  CHECK_INPUT(grad_acceleration);
  CHECK_INPUT(grad_jerk);
  CHECK_INPUT(traj_dt);

  return backward_step_position_clique2(
    out_grad_position, grad_position, grad_velocity, grad_acceleration,
    grad_jerk, traj_dt, batch_size, horizon, dof, mode);
}

std::vector<torch::Tensor>step_acceleration_wrapper(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_acc, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor traj_dt, const int batch_size, const int horizon,
  const int dof, const bool use_rk2 = true)
{
  const at::cuda::OptionalCUDAGuard guard(u_acc.device());

  CHECK_INPUT(u_acc);
  CHECK_INPUT(out_position);
  CHECK_INPUT(out_velocity);
  CHECK_INPUT(out_acceleration);
  CHECK_INPUT(out_jerk);
  CHECK_INPUT(start_position);
  CHECK_INPUT(start_velocity);
  CHECK_INPUT(start_acceleration);
  CHECK_INPUT(traj_dt);

  return step_acceleration(out_position, out_velocity, out_acceleration,
                           out_jerk, u_acc, start_position, start_velocity,
                           start_acceleration, traj_dt, batch_size, horizon,
                           dof, use_rk2);
}

std::vector<torch::Tensor>step_acceleration_idx_wrapper(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_acc, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor start_idx, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof,
  const bool use_rk2 = true)
{
  const at::cuda::OptionalCUDAGuard guard(u_acc.device());

  CHECK_INPUT(u_acc);
  CHECK_INPUT(out_position);
  CHECK_INPUT(out_velocity);
  CHECK_INPUT(out_acceleration);
  CHECK_INPUT(out_jerk);
  CHECK_INPUT(start_position);
  CHECK_INPUT(start_velocity);
  CHECK_INPUT(start_acceleration);
  CHECK_INPUT(start_idx);
  CHECK_INPUT(traj_dt);

  return step_acceleration_idx(out_position, out_velocity, out_acceleration,
                               out_jerk, u_acc, start_position, start_velocity,
                               start_acceleration, start_idx, traj_dt,
                               batch_size, horizon, dof, use_rk2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("step_position",           &step_position_clique_wrapper,
        "Tensor Step Position (curobolib)");
  m.def("step_position2",          &step_position_clique2_wrapper,
        "Tensor Step Position (curobolib)");
  m.def("step_idx_position2",      &step_position_clique2_idx_wrapper,
        "Tensor Step Position (curobolib)");
  m.def("step_position_backward",  &backward_step_position_clique_wrapper,
        "Tensor Step Position (curobolib)");
  m.def("step_position_backward2", &backward_step_position_clique2_wrapper,
        "Tensor Step Position (curobolib)");
  m.def("step_acceleration",       &step_acceleration_wrapper,
        "Tensor Step Acceleration (curobolib)");
  m.def("step_acceleration_idx",   &step_acceleration_idx_wrapper,
        "Tensor Step Acceleration (curobolib)");
}
