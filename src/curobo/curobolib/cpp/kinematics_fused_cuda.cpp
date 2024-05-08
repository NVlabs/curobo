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

#include "check_cuda.h"

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
  const torch::Tensor joint_offset_map,
  const int           batch_size,
  const int           n_joints,
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
  const torch::Tensor joint_offset_map,
  const int           batch_size,
  const int           n_joints,
  const int           n_spheres,
  const bool          sparsity_opt     = true,
  const bool          use_global_cumul = false);

// C++ interface


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward",              &kin_fused_forward,  "Kinematics fused forward (CUDA)");
  m.def("backward",             &kin_fused_backward_16t, "Kinematics fused backward (CUDA)");
  m.def("matrix_to_quaternion", &matrix_to_quaternion,
        "Rotation Matrix to Quaternion (CUDA)");
}
