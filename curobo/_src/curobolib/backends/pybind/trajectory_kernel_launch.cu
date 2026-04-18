/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "bspline/bspline_common.cuh"
#include "common/torch_cuda_utils.h"

#define MAX_H 100
#define BWD_DIFF -1
#define CENTRAL_DIFF 0
#define USE_STENCIL true

#include "legacy/differentiation_position_kernel.cuh"
#include "legacy/integration_acceleration_kernel.cuh"
#include "bspline/bspline_kernel.cuh"

namespace curobo {

namespace trajectory{

namespace legacy{
void launch_differentiation_position_forward_kernel(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  torch::Tensor out_dt,
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

  const int batch_size, const int horizon, const int dof)
{
  if (horizon <= 5)
  {
    throw std::runtime_error("horizon must be greater than 5");
  }
  curobo::common::validate_cuda_input(out_position, "out_position");
  curobo::common::validate_cuda_input(out_velocity, "out_velocity");
  curobo::common::validate_cuda_input(out_acceleration, "out_acceleration");
  curobo::common::validate_cuda_input(out_jerk, "out_jerk");
  curobo::common::validate_cuda_input(out_dt, "out_dt");
  curobo::common::validate_cuda_input(u_position, "u_position");
  curobo::common::validate_cuda_input(start_position, "start_position");
  curobo::common::validate_cuda_input(start_velocity, "start_velocity");
  curobo::common::validate_cuda_input(start_acceleration, "start_acceleration");
  curobo::common::validate_cuda_input(goal_position, "goal_position");
  curobo::common::validate_cuda_input(goal_velocity, "goal_velocity");
  curobo::common::validate_cuda_input(goal_acceleration, "goal_acceleration");
  curobo::common::validate_cuda_input(start_idx, "start_idx");
  curobo::common::validate_cuda_input(goal_idx, "goal_idx");
  curobo::common::validate_cuda_input(traj_dt, "traj_dt");
  curobo::common::validate_cuda_input(use_implicit_goal_state, "use_implicit_goal_state");




  const int k_size    = batch_size * dof * horizon;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }

  int blocksPerGrid   = curobo::common::ceil_div(k_size, threadsPerBlock);
  cudaStream_t stream = curobo::common::get_cuda_stream();

  if (u_position.sizes()[u_position.sizes().size() - 2] != horizon - 4)
  {
    throw std::runtime_error("u_position.sizes()[u_position.sizes().size() - 2] != horizon - 4");
  }

  position_clique_loop_idx_fwd_kernel<float, USE_STENCIL>
    << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
    out_position.data_ptr<float>(),
    out_velocity.data_ptr<float>(),
    out_acceleration.data_ptr<float>(),
    out_jerk.data_ptr<float>(),
    out_dt.data_ptr<float>(),
    u_position.data_ptr<float>(),
    start_position.data_ptr<float>(),
    start_velocity.data_ptr<float>(),
    start_acceleration.data_ptr<float>(),
    goal_position.data_ptr<float>(),
    goal_velocity.data_ptr<float>(),
    goal_acceleration.data_ptr<float>(),
    start_idx.data_ptr<int32_t>(),
    goal_idx.data_ptr<int32_t>(),
    traj_dt.data_ptr<float>(),
    use_implicit_goal_state.data_ptr<uint8_t>(),
    batch_size, horizon, dof);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return;
}


void launch_differentiation_position_backward_kernel(
  torch::Tensor out_grad_position,
  const torch::Tensor grad_position,
  const torch::Tensor grad_velocity,
  const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk,
  const torch::Tensor traj_dt,
  const torch::Tensor dt_idx,
  const torch::Tensor use_implicit_goal_state,
  const int batch_size,
  const int horizon,
  const int dof)
{
  if (horizon <= 5)
  {
    throw std::runtime_error("horizon must be greater than 5");
  }
  curobo::common::validate_cuda_input(out_grad_position, "out_grad_position");
  curobo::common::validate_cuda_input(grad_position, "grad_position");
  curobo::common::validate_cuda_input(grad_velocity, "grad_velocity");
  curobo::common::validate_cuda_input(grad_acceleration, "grad_acceleration");
  curobo::common::validate_cuda_input(grad_jerk, "grad_jerk");
  curobo::common::validate_cuda_input(traj_dt, "traj_dt");
  curobo::common::validate_cuda_input(dt_idx, "dt_idx");
  curobo::common::validate_cuda_input(use_implicit_goal_state, "use_implicit_goal_state");



  // This kernel is currently called for backward
  // assert(horizon < MAX_H - 4);




  // const int k_size = batch_size * dof;
  const int k_size    = batch_size * dof * (horizon - 4);
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }

  int blocksPerGrid   = curobo::common::ceil_div(k_size, threadsPerBlock);
  cudaStream_t stream = curobo::common::get_cuda_stream();


  if (out_grad_position.sizes()[1] != horizon - 4)
  {
    throw std::runtime_error("out_grad_position.sizes()[1] != horizon - 4");
  }
    position_clique_loop_idx_bwd_kernel<float, USE_STENCIL>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
      out_grad_position.data_ptr<float>(),
      grad_position.data_ptr<float>(),
      grad_velocity.data_ptr<float>(),
      grad_acceleration.data_ptr<float>(),
      grad_jerk.data_ptr<float>(),
      traj_dt.data_ptr<float>(),
      dt_idx.data_ptr<int32_t>(),
      use_implicit_goal_state.data_ptr<uint8_t>(),

      batch_size, horizon, dof);


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return;
}


void launch_integration_acceleration_kernel(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_acc, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor start_idx, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof,
  const bool use_rk2 = true)
{
  if (horizon >= MAX_H)
  {
    throw std::runtime_error("horizon must be less than MAX_H");
  }
  curobo::common::validate_cuda_input(out_position, "out_position");
  curobo::common::validate_cuda_input(out_velocity, "out_velocity");
  curobo::common::validate_cuda_input(out_acceleration, "out_acceleration");
  curobo::common::validate_cuda_input(out_jerk, "out_jerk");
  curobo::common::validate_cuda_input(u_acc, "u_acc");
  curobo::common::validate_cuda_input(start_position, "start_position");
  curobo::common::validate_cuda_input(start_velocity, "start_velocity");
  curobo::common::validate_cuda_input(start_acceleration, "start_acceleration");
  curobo::common::validate_cuda_input(start_idx, "start_idx");
  curobo::common::validate_cuda_input(traj_dt, "traj_dt");





  const int k_size    = batch_size * dof;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 512)
  {
    threadsPerBlock = 512;
  }

  int blocksPerGrid   = curobo::common::ceil_div(k_size, threadsPerBlock);
  cudaStream_t stream = curobo::common::get_cuda_stream();

  if (use_rk2)
  {
      acceleration_loop_idx_rk2_kernel<float, MAX_H>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(), u_acc.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(), traj_dt.data_ptr<float>(),
        batch_size, horizon, dof);

  }
  else
  {

      acceleration_loop_idx_kernel<float, MAX_H>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(), u_acc.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(), traj_dt.data_ptr<float>(),
        batch_size, horizon, dof);

  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return; // { out_position, out_velocity, out_acceleration, out_jerk };
}
}

namespace bspline{

constexpr BasisBackend defaultBasisImpl = BasisBackend::MATRIX;



void launch_bspline_interpolation_forward_kernel(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  torch::Tensor out_dt,
  const torch::Tensor u_position,
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
  const torch::Tensor traj_dt,
  const torch::Tensor use_implicit_goal_state,
  const int batch_size,
  const int horizon,
  const int dof,
  const int n_knots,
  const int bspline_degree)
{

  curobo::common::validate_cuda_input(out_position, "out_position");
  curobo::common::validate_cuda_input(out_velocity, "out_velocity");
  curobo::common::validate_cuda_input(out_acceleration, "out_acceleration");
  curobo::common::validate_cuda_input(out_jerk, "out_jerk");
  curobo::common::validate_cuda_input(u_position, "u_position");
  curobo::common::validate_cuda_input(start_position, "start_position");
  curobo::common::validate_cuda_input(start_velocity, "start_velocity");
  curobo::common::validate_cuda_input(start_acceleration, "start_acceleration");
  curobo::common::validate_cuda_input(start_jerk, "start_jerk");
  curobo::common::validate_cuda_input(traj_dt, "traj_dt");
  curobo::common::validate_cuda_input(start_idx, "start_idx");
  curobo::common::validate_cuda_input(goal_idx, "goal_idx");
  curobo::common::validate_cuda_input(goal_position, "goal_position");
  curobo::common::validate_cuda_input(goal_velocity, "goal_velocity");
  curobo::common::validate_cuda_input(goal_acceleration, "goal_acceleration");
  curobo::common::validate_cuda_input(goal_jerk, "goal_jerk");
  curobo::common::validate_cuda_input(use_implicit_goal_state, "use_implicit_goal_state");
  curobo::common::validate_cuda_input(out_dt, "out_dt");
  if (horizon < 5)
  {
    throw std::runtime_error("horizon must be greater than 5");
  }


  const int k_size    = batch_size * dof * horizon;
  const int max_threads = 128;

  int threadsPerBlock = k_size;

  if (threadsPerBlock > max_threads)
  {
    threadsPerBlock = max_threads;
  }
  int blocksPerGrid   = curobo::common::ceil_div(k_size, threadsPerBlock);
  cudaStream_t stream = curobo::common::get_cuda_stream();

  if (bspline_degree == 4)
  {


      interpolate_bspline_kernel<float, 4, defaultBasisImpl>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(),
        out_dt.data_ptr<float>(),
        u_position.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_jerk.data_ptr<float>(),
        goal_position.data_ptr<float>(),
        goal_velocity.data_ptr<float>(),
        goal_acceleration.data_ptr<float>(),
        goal_jerk.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(),
        goal_idx.data_ptr<int32_t>(),
        traj_dt.data_ptr<float>(),
        use_implicit_goal_state.data_ptr<uint8_t>(),
        batch_size, horizon, dof, n_knots);
  }
  else if (bspline_degree == 3)
  {
      interpolate_bspline_kernel<float, 3, defaultBasisImpl>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(),
        out_dt.data_ptr<float>(),
        u_position.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_jerk.data_ptr<float>(),
        goal_position.data_ptr<float>(),
        goal_velocity.data_ptr<float>(),
        goal_acceleration.data_ptr<float>(),
        goal_jerk.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(),
        goal_idx.data_ptr<int32_t>(),
        traj_dt.data_ptr<float>(),
        use_implicit_goal_state.data_ptr<uint8_t>(),
        batch_size, horizon, dof, n_knots);

  }
  else if (bspline_degree == 5)
  {
      interpolate_bspline_kernel<float, 5, defaultBasisImpl>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(),
        out_dt.data_ptr<float>(),
        u_position.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_jerk.data_ptr<float>(),
        goal_position.data_ptr<float>(),
        goal_velocity.data_ptr<float>(),
        goal_acceleration.data_ptr<float>(),
        goal_jerk.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(),
        goal_idx.data_ptr<int32_t>(),
        traj_dt.data_ptr<float>(),
        use_implicit_goal_state.data_ptr<uint8_t>(),
        batch_size, horizon, dof, n_knots);

  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return ;
}


void launch_bspline_interpolation_single_dt_kernel(
  torch::Tensor out_position,
  torch::Tensor out_velocity,
  torch::Tensor out_acceleration,
  torch::Tensor out_jerk,
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
  const int bspline_degree)
{
  curobo::common::validate_cuda_input(out_position, "out_position");
  curobo::common::validate_cuda_input(out_velocity, "out_velocity");
  curobo::common::validate_cuda_input(out_acceleration, "out_acceleration");
  curobo::common::validate_cuda_input(out_jerk, "out_jerk");
  curobo::common::validate_cuda_input(knots, "knots");
  curobo::common::validate_cuda_input(knot_dt, "knot_dt");
  curobo::common::validate_cuda_input(start_position, "start_position");
  curobo::common::validate_cuda_input(start_velocity, "start_velocity");
  curobo::common::validate_cuda_input(start_acceleration, "start_acceleration");
  curobo::common::validate_cuda_input(start_jerk, "start_jerk");
  curobo::common::validate_cuda_input(start_idx, "start_idx");
  curobo::common::validate_cuda_input(goal_idx, "goal_idx");
  curobo::common::validate_cuda_input(goal_position, "goal_position");
  curobo::common::validate_cuda_input(goal_velocity, "goal_velocity");
  curobo::common::validate_cuda_input(goal_acceleration, "goal_acceleration");
  curobo::common::validate_cuda_input(goal_jerk, "goal_jerk");
  curobo::common::validate_cuda_input(use_implicit_goal_state, "use_implicit_goal_state");
  curobo::common::validate_cuda_input(out_dt, "out_dt");

  if (n_knots < 5)
  {
    throw std::runtime_error("n_knots must be greater than 5");
  }


  const int k_size    = batch_size * dof * max_out_tsteps;
  const int max_threads = 256;

  int threadsPerBlock = k_size;

  if (threadsPerBlock > max_threads)
  {
    threadsPerBlock = max_threads;
  }

  int blocksPerGrid   = curobo::common::ceil_div(k_size, threadsPerBlock);
  cudaStream_t stream = curobo::common::get_cuda_stream();
  if(bspline_degree==3)
  {
    interpolate_bspline_single_dt_kernel<float, 3, defaultBasisImpl>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(),
        out_dt.data_ptr<float>(),
        knots.data_ptr<float>(),
        knot_dt.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_jerk.data_ptr<float>(),
        goal_position.data_ptr<float>(),
        goal_velocity.data_ptr<float>(),
        goal_acceleration.data_ptr<float>(),
        goal_jerk.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(),
        goal_idx.data_ptr<int32_t>(),
        interpolation_dt.data_ptr<float>(),
        use_implicit_goal_state.data_ptr<uint8_t>(),
        interpolation_horizon.data_ptr<int32_t>(),
        batch_size,
        max_out_tsteps,
        dof,
        n_knots
        );
  }
  else if(bspline_degree==4)
  {
    interpolate_bspline_single_dt_kernel<float, 4, defaultBasisImpl>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(),
        out_dt.data_ptr<float>(),
        knots.data_ptr<float>(),
        knot_dt.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_jerk.data_ptr<float>(),
        goal_position.data_ptr<float>(),
        goal_velocity.data_ptr<float>(),
        goal_acceleration.data_ptr<float>(),
        goal_jerk.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(),
        goal_idx.data_ptr<int32_t>(),
        interpolation_dt.data_ptr<float>(),
        use_implicit_goal_state.data_ptr<uint8_t>(),
        interpolation_horizon.data_ptr<int32_t>(),
        batch_size,
        max_out_tsteps,
        dof,
        n_knots
        );
  }
  else if(bspline_degree==5)
  {
    interpolate_bspline_single_dt_kernel<float, 5, defaultBasisImpl>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<float>(),
        out_velocity.data_ptr<float>(),
        out_acceleration.data_ptr<float>(),
        out_jerk.data_ptr<float>(),
        out_dt.data_ptr<float>(),
        knots.data_ptr<float>(),
        knot_dt.data_ptr<float>(),
        start_position.data_ptr<float>(),
        start_velocity.data_ptr<float>(),
        start_acceleration.data_ptr<float>(),
        start_jerk.data_ptr<float>(),
        goal_position.data_ptr<float>(),
        goal_velocity.data_ptr<float>(),
        goal_acceleration.data_ptr<float>(),
        goal_jerk.data_ptr<float>(),
        start_idx.data_ptr<int32_t>(),
        goal_idx.data_ptr<int32_t>(),
        interpolation_dt.data_ptr<float>(),
        use_implicit_goal_state.data_ptr<uint8_t>(),
        interpolation_horizon.data_ptr<int32_t>(),
        batch_size,
        max_out_tsteps,
        dof,
        n_knots
        );
  }
  else
  {
    throw std::runtime_error("Unsupported B-spline degree: " + std::to_string(bspline_degree));
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

}



void launch_bspline_interpolation_backward_kernel(
  torch::Tensor out_grad_position, const torch::Tensor grad_position,
  const torch::Tensor grad_velocity, const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk, const torch::Tensor traj_dt,
  const torch::Tensor dt_idx,
  const torch::Tensor use_implicit_goal_state,
  const int batch_size, const int padded_horizon, const int dof,
  const int n_knots,
  const int bspline_degree,
  const bool use_direct_polynomial)
{
  // This kernel is currently called for backward
  // assert(horizon < MAX_H - 4);
  curobo::common::validate_cuda_input(out_grad_position, "out_grad_position");
  curobo::common::validate_cuda_input(grad_position, "grad_position");
  curobo::common::validate_cuda_input(grad_velocity, "grad_velocity");
  curobo::common::validate_cuda_input(grad_acceleration, "grad_acceleration");
  curobo::common::validate_cuda_input(grad_jerk, "grad_jerk");
  curobo::common::validate_cuda_input(traj_dt, "traj_dt");
  curobo::common::validate_cuda_input(dt_idx, "dt_idx");
  curobo::common::validate_cuda_input(use_implicit_goal_state, "use_implicit_goal_state");
  int horizon = padded_horizon - 1;

  if (horizon < 5)
  {
    throw std::runtime_error("horizon must be greater than 5");
  }
  BSplineBackwardLayout layout;




  switch (bspline_degree) {
    case 3:
      layout = compute_bspline_backward_layout<3>(horizon, dof, n_knots);
      break;
    case 4:
      layout = compute_bspline_backward_layout<4>(horizon, dof, n_knots);
      break;
    case 5:
      layout = compute_bspline_backward_layout<5>(horizon, dof, n_knots);
      break;
    default:
      throw std::invalid_argument("Unsupported B-spline degree: " + std::to_string(bspline_degree));
  }

  if (layout.interpolation_steps <= 0){
    throw std::runtime_error("interpolation_steps is 0: horizon (" + std::to_string(horizon) +
      ") too small for n_knots (" + std::to_string(n_knots) +
      ") and degree (" + std::to_string(bspline_degree) + ")");
  }

  if (layout.interpolation_steps > curobo::common::warpSize){
    throw std::runtime_error("interpolation_steps > " + std::to_string(curobo::common::warpSize) + " is not supported");
  }

  cudaStream_t stream = curobo::common::get_cuda_stream();

  int k_size    = batch_size * dof * n_knots;
  const int max_threads = 128;

  int threadsPerBlock = k_size;

  if (threadsPerBlock > max_threads)
  {
    threadsPerBlock = max_threads;
  }
  int blocksPerGrid   = curobo::common::ceil_div(k_size, threadsPerBlock);

  k_size = batch_size * dof * layout.threads_for_n_knots;
  threadsPerBlock = k_size;

  if (threadsPerBlock > max_threads)
  {
    threadsPerBlock = max_threads;
  }
  blocksPerGrid   = curobo::common::ceil_div(k_size, threadsPerBlock);



  // Use basis-driven backward kernel; the USE_DIRECT_POLYNOMIAL flag no longer changes behavior.
  decltype(bspline_backward_kernel<3, float>) *selected_kernel;
  switch (bspline_degree) {
    case 3:
      selected_kernel = bspline_backward_kernel<3, float, defaultBasisImpl>;
      break;
    case 4:
      selected_kernel = bspline_backward_kernel<4, float, defaultBasisImpl>;
      break;
    case 5:
      selected_kernel = bspline_backward_kernel<5, float, defaultBasisImpl>;
      break;
    default:
      throw std::invalid_argument("Unsupported B-spline degree: " + std::to_string(bspline_degree));
  }

  selected_kernel
    << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
    out_grad_position.data_ptr<float>(),
    grad_position.data_ptr<float>(),
    grad_velocity.data_ptr<float>(),
    grad_acceleration.data_ptr<float>(),
    grad_jerk.data_ptr<float>(),
    traj_dt.data_ptr<float>(),
    dt_idx.data_ptr<int32_t>(),
    use_implicit_goal_state.data_ptr<uint8_t>(),
    batch_size, horizon, dof, n_knots);


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return;
}
} // namespace bspline

} // namespace trajectory
} // namespace curobo