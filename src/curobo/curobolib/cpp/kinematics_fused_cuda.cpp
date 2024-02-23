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

// CUDA forward declarations

std::vector<torch::Tensor>
matrix_to_quaternion(torch::Tensor       out_quat,
                     const torch::Tensor in_rot // batch_size, 3
                     );

std::vector<torch::Tensor>kin_fused_forward(
  torch::Tensor       link_pos,
  torch::Tensor       link_quat,
  torch::Tensor       batch_robot_spheres,
  torch::Tensor       global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_map,
  const torch::Tensor joint_map,
  const torch::Tensor joint_map_type,
  const torch::Tensor store_link_map,
  const torch::Tensor link_sphere_map,
  const int           batch_size,
  const int           n_spheres,
  const bool          use_global_cumul = false);

std::vector<torch::Tensor>kin_fused_backward_16t(
  torch::Tensor       grad_out,
  const torch::Tensor grad_nlinks_pos,
  const torch::Tensor grad_nlinks_quat,
  const torch::Tensor grad_spheres,
  const torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_map,
  const torch::Tensor joint_map,
  const torch::Tensor joint_map_type,
  const torch::Tensor store_link_map,
  const torch::Tensor link_sphere_map,
  const torch::Tensor link_chain_map,
  const int           batch_size,
  const int           n_spheres,
  const bool          sparsity_opt     = true,
  const bool          use_global_cumul = false);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), # x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), # x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>kin_forward_wrapper(
  torch::Tensor link_pos, torch::Tensor link_quat,
  torch::Tensor batch_robot_spheres, torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec, const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres, const torch::Tensor link_map,
  const torch::Tensor joint_map, const torch::Tensor joint_map_type,
  const torch::Tensor store_link_map, const torch::Tensor link_sphere_map,
  const int batch_size, const int n_spheres,
  const bool use_global_cumul = false)
{
  const at::cuda::OptionalCUDAGuard guard(joint_vec.device());

  // TODO: add check input
  return kin_fused_forward(
    link_pos, link_quat, batch_robot_spheres, global_cumul_mat, joint_vec,
    fixed_transform, robot_spheres, link_map, joint_map, joint_map_type,
    store_link_map, link_sphere_map, batch_size, n_spheres, use_global_cumul);
}

std::vector<torch::Tensor>kin_backward_wrapper(
  torch::Tensor grad_out, const torch::Tensor grad_nlinks_pos,
  const torch::Tensor grad_nlinks_quat, const torch::Tensor grad_spheres,
  const torch::Tensor global_cumul_mat, const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform, const torch::Tensor robot_spheres,
  const torch::Tensor link_map, const torch::Tensor joint_map,
  const torch::Tensor joint_map_type, const torch::Tensor store_link_map,
  const torch::Tensor link_sphere_map, const torch::Tensor link_chain_map,
  const int batch_size, const int n_spheres, const bool sparsity_opt = true,
  const bool use_global_cumul = false)
{
  const at::cuda::OptionalCUDAGuard guard(joint_vec.device());

  return kin_fused_backward_16t(
    grad_out, grad_nlinks_pos, grad_nlinks_quat, grad_spheres,
    global_cumul_mat, joint_vec, fixed_transform, robot_spheres, link_map,
    joint_map, joint_map_type, store_link_map, link_sphere_map,
    link_chain_map, batch_size, n_spheres, sparsity_opt, use_global_cumul);
}

std::vector<torch::Tensor>
matrix_to_quaternion_wrapper(torch::Tensor       out_quat,
                             const torch::Tensor in_rot // batch_size, 3
                             )
{
  const at::cuda::OptionalCUDAGuard guard(in_rot.device());

  CHECK_INPUT(in_rot);
  CHECK_INPUT(out_quat);
  return matrix_to_quaternion(out_quat, in_rot);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward",              &kin_forward_wrapper,  "Kinematics fused forward (CUDA)");
  m.def("backward",             &kin_backward_wrapper, "Kinematics fused backward (CUDA)");
  m.def("matrix_to_quaternion", &matrix_to_quaternion_wrapper,
        "Rotation Matrix to Quaternion (CUDA)");
}
