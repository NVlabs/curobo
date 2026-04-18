/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <vector>


// CUDA forward declarations
namespace curobo{
  namespace kinematics{

void launch_kinematics_forward(
  torch::Tensor       link_pos, // [batch_size, n_tool_frames, 3]
  torch::Tensor       link_quat, // [batch_size, n_tool_frames, 4]
  torch::Tensor       batch_robot_spheres, // [batch_size, n_spheres, 4]
  torch::Tensor       batch_center_of_mass, // [batch_size * 4] - xyz=global CoM, w=total mass
  torch::Tensor       batch_jacobian, // [batch_size, n_tool_frames, 6, n_joints]
  torch::Tensor       global_cumul_mat, // [batch_size, n_links, 12]
  const torch::Tensor joint_vec, // [batch_size, n_joints]
  const torch::Tensor fixed_transform, // [n_links, 12]
  const torch::Tensor robot_spheres, // [n_spheres, 4]
  const torch::Tensor link_masses_com, // [n_links * 4] - xyz=local CoM, w=mass
  const torch::Tensor link_map, // [n_links]
  const torch::Tensor joint_map, // [n_joints]
  const torch::Tensor joint_map_type, // [n_joints]
  const torch::Tensor tool_frame_map, // [n_tool_frames]
  const torch::Tensor link_sphere_map, // [n_spheres]
  const torch::Tensor link_chain_data, // [chain_data_size]
  const torch::Tensor link_chain_offsets, // [n_links + 1]
  const torch::Tensor joint_links_data, // [joint_links_data_size]
  const torch::Tensor joint_links_offsets, // [n_joints + 1]
  const torch::Tensor joint_affects_endeffector, // [n_joints * n_tool_frames] - NEW
  const torch::Tensor joint_offset_map, // [n_joints]
  const torch::Tensor env_query_idx,
  const int64_t           num_envs,
  const int64_t           batch_size,
  const int64_t           horizon,
  const int64_t           n_joints,
  const int64_t           n_spheres,
  const bool compute_jacobian,
  const bool compute_com);

void launch_kinematics_backward(
  torch::Tensor       grad_out,
  const torch::Tensor grad_nlinks_pos,
  const torch::Tensor grad_nlinks_quat,
  const torch::Tensor grad_spheres,
  const torch::Tensor grad_center_of_mass,  // [batch_size, 4] - xyz=pos grad, w=mass grad (ignored)
  const torch::Tensor batch_center_of_mass, // [batch_size, 4] - xyz=global CoM, w=total mass (INPUT)
  const torch::Tensor global_cumul_mat,
  const torch::Tensor joint_vec,
  const torch::Tensor fixed_transform,
  const torch::Tensor robot_spheres,
  const torch::Tensor link_masses_com,      // [n_links, 4] - xyz=local CoM, w=mass
  const torch::Tensor link_map,
  const torch::Tensor joint_map,
  const torch::Tensor joint_map_type,
  const torch::Tensor tool_frame_map,
  const torch::Tensor link_sphere_map,
  const torch::Tensor link_chain_data,      // NEW
  const torch::Tensor link_chain_offsets,   // NEW
  const torch::Tensor joint_offset_map,
  const torch::Tensor env_query_idx,
  const int64_t           num_envs,
  const int64_t           batch_size,
  const int64_t           horizon,
  const int64_t           n_joints,
  const int64_t           n_spheres,
  const bool compute_com);


void launch_kinematics_jacobian_backward(
  torch::Tensor       grad_joint, // [batch_size, n_joints]
  const torch::Tensor grad_jacobian, // [batch_size, n_tool_frames, 6, n_joints]
  const torch::Tensor global_cumul_mat, // [batch_size, n_links, 12]
  const torch::Tensor joint_map_type, // [n_joints]
  const torch::Tensor joint_map, // [n_joints]
  const torch::Tensor link_map, // [n_links]
  const torch::Tensor link_chain_data, // [chain_data_size]
  const torch::Tensor link_chain_offsets, // [n_links + 1]
  const torch::Tensor joint_links_data, // [joint_links_data_size]
  const torch::Tensor joint_links_offsets, // [n_joints + 1]
  const torch::Tensor joint_affects_endeffector, // [n_joints * n_tool_frames] - NEW
  const torch::Tensor tool_frame_map, // [n_tool_frames]
  const torch::Tensor joint_offset_map, // [n_joints]
  const int64_t           batch_size,
  const int64_t           n_joints,
  const int64_t           n_tool_frames);

}
}
// C++ interface

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("launch_kinematics_forward", &curobo::kinematics::launch_kinematics_forward, "Kinematics forward (CUDA)",
        py::arg("link_pos"),
        py::arg("link_quat"),
        py::arg("batch_robot_spheres"),
        py::arg("batch_center_of_mass"),
        py::arg("batch_jacobian"),
        py::arg("global_cumul_mat"),
        py::arg("joint_vec"),
        py::arg("fixed_transform"),
        py::arg("robot_spheres"),
        py::arg("link_masses_com"),
        py::arg("link_map"),
        py::arg("joint_map"),
        py::arg("joint_map_type"),
        py::arg("tool_frame_map"),
        py::arg("link_sphere_map"),
        py::arg("link_chain_data"),
        py::arg("link_chain_offsets"),
        py::arg("joint_links_data"),
        py::arg("joint_links_offsets"),
        py::arg("joint_affects_endeffector"),
        py::arg("joint_offset_map"),
        py::arg("env_query_idx"),
        py::arg("num_envs"),
        py::arg("batch_size"),
        py::arg("horizon"),
        py::arg("n_joints"),
        py::arg("n_spheres"),
        py::arg("compute_jacobian"),
        py::arg("compute_com"));

  m.def("launch_kinematics_backward", &curobo::kinematics::launch_kinematics_backward, "Kinematics backward (CUDA)",
        py::arg("grad_out"),
        py::arg("grad_nlinks_pos"),
        py::arg("grad_nlinks_quat"),
        py::arg("grad_spheres"),
        py::arg("grad_center_of_mass"),
        py::arg("batch_center_of_mass"),
        py::arg("global_cumul_mat"),
        py::arg("joint_vec"),
        py::arg("fixed_transform"),
        py::arg("robot_spheres"),
        py::arg("link_masses_com"),
        py::arg("link_map"),
        py::arg("joint_map"),
        py::arg("joint_map_type"),
        py::arg("tool_frame_map"),
        py::arg("link_sphere_map"),
        py::arg("link_chain_data"),
        py::arg("link_chain_offsets"),
        py::arg("joint_offset_map"),
        py::arg("env_query_idx"),
        py::arg("num_envs"),
        py::arg("batch_size"),
        py::arg("horizon"),
        py::arg("n_joints"),
        py::arg("n_spheres"),
        py::arg("compute_com"));

  m.def("launch_kinematics_jacobian_backward", &curobo::kinematics::launch_kinematics_jacobian_backward, "Kinematics jacobian backward (CUDA)",
        py::arg("grad_joint"),
        py::arg("grad_jacobian"),
        py::arg("global_cumul_mat"),
        py::arg("joint_map_type"),
        py::arg("joint_map"),
        py::arg("link_map"),
        py::arg("link_chain_data"),
        py::arg("link_chain_offsets"),
        py::arg("joint_links_data"),
        py::arg("joint_links_offsets"),
        py::arg("joint_affects_endeffector"),
        py::arg("tool_frame_map"),
        py::arg("joint_offset_map"),
        py::arg("batch_size"),
        py::arg("n_joints"),
        py::arg("n_tool_frames"));
}
