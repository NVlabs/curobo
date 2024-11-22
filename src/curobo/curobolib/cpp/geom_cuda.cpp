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
#include <map>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "check_cuda.h"
// CUDA forward declarations

std::vector<torch::Tensor>self_collision_distance(
  torch::Tensor       out_distance,
  torch::Tensor       out_vec,
  torch::Tensor       sparse_index,
  const torch::Tensor robot_spheres,    // batch_size x n_spheres x 4
  const torch::Tensor collision_offset, // n_spheres x n_spheres
  const torch::Tensor weight,
  const torch::Tensor collision_matrix, // n_spheres x n_spheres
  const torch::Tensor thread_locations,
  const int           locations_size,
  const int           batch_size,
  const int           nspheres,
  const bool          compute_grad = false,
  const int           ndpt         = 8, // Does this need to match template?
  const bool          debug        = false);

std::vector<torch::Tensor>swept_sphere_obb_clpt(
  const torch::Tensor sphere_position, // batch_size, 3
  torch::Tensor       distance,        // batch_size, 1
  torch::Tensor
  closest_point,                       // batch size, 4 -> written out as x,y,z,0 for gradient
  torch::Tensor       sparsity_idx,
  const torch::Tensor weight,
  const torch::Tensor activation_distance,
  const torch::Tensor speed_dt,
  const torch::Tensor obb_accel,     // n_boxes, 4, 4
  const torch::Tensor obb_bounds,    // n_boxes, 3
  const torch::Tensor obb_pose,      // n_boxes, 4, 4
  const torch::Tensor obb_enable,    // n_boxes, 4,
  const torch::Tensor n_env_obb,     // n_boxes, 4, 4
  const torch::Tensor env_query_idx, // n_boxes, 4, 4
  const int           max_nobs,
  const int           batch_size,
  const int           horizon,
  const int           n_spheres,
  const int           sweep_steps,
  const bool          enable_speed_metric,
  const bool          transform_back,
  const bool          compute_distance,
  const bool          use_batch_env,
  const bool          sum_collisions);

std::vector<torch::Tensor>
sphere_obb_clpt(const torch::Tensor sphere_position, // batch_size, 4
                torch::Tensor       distance,
                torch::Tensor       closest_point,   // batch size, 3
                torch::Tensor       sparsity_idx,
                const torch::Tensor weight,
                const torch::Tensor activation_distance,
                  const torch::Tensor max_distance,
                const torch::Tensor obb_accel,     // n_boxes, 4, 4
                const torch::Tensor obb_bounds,    // n_boxes, 3
                const torch::Tensor obb_pose,      // n_boxes, 4, 4
                const torch::Tensor obb_enable,    // n_boxes, 4, 4
                const torch::Tensor n_env_obb,     // n_boxes, 4, 4
                const torch::Tensor env_query_idx, // n_boxes, 4, 4
                const int           max_nobs,
                const int           batch_size,
                const int           horizon,
                const int           n_spheres,
                const bool          transform_back,
                const bool          compute_distance,
                const bool          use_batch_env,
                const bool          sum_collisions,
                const bool          compute_esdf);

std::vector<torch::Tensor>
sphere_voxel_clpt(const torch::Tensor sphere_position, // batch_size, 3
                torch::Tensor distance,
                torch::Tensor closest_point,         // batch size, 3
                torch::Tensor sparsity_idx, const torch::Tensor weight,
                const torch::Tensor activation_distance,
                const torch::Tensor max_distance,
                const torch::Tensor grid_features,       // n_boxes, 4, 4
                const torch::Tensor grid_params,      // n_boxes, 3
                const torch::Tensor grid_pose,        // n_boxes, 4, 4
                const torch::Tensor grid_enable,      // n_boxes, 4, 4
                const torch::Tensor n_env_grid,
                const torch::Tensor env_query_idx,   // n_boxes, 4, 4
                const int max_nobs, const int batch_size, const int horizon,
                const int n_spheres, const bool transform_back,
                const bool compute_distance, const bool use_batch_env,
                const bool sum_collisions,
                const bool compute_esdf);

std::vector<torch::Tensor>
swept_sphere_voxel_clpt(const torch::Tensor sphere_position, // batch_size, 3
                torch::Tensor distance,
                torch::Tensor closest_point,         // batch size, 3
                torch::Tensor sparsity_idx, const torch::Tensor weight,
                const torch::Tensor activation_distance,
                const torch::Tensor max_distance,
                const torch::Tensor speed_dt,
                const torch::Tensor grid_features,       // n_boxes, 4, 4
                const torch::Tensor grid_params,      // n_boxes, 3
                const torch::Tensor grid_pose,        // n_boxes, 4, 4
                const torch::Tensor grid_enable,      // n_boxes, 4, 4
                const torch::Tensor n_env_grid,
                const torch::Tensor env_query_idx,   // n_boxes, 4, 4
                const int max_nobs,
                const int batch_size,
                const int horizon,
                const int n_spheres,
                const int sweep_steps,
                const bool enable_speed_metric,
                const bool transform_back,
                const bool compute_distance,
                const bool use_batch_env,
                const bool sum_collisions);

std::vector<torch::Tensor>pose_distance(
  torch::Tensor       out_distance,
  torch::Tensor       out_position_distance,
  torch::Tensor       out_rotation_distance,
  torch::Tensor       distance_p_vector, // batch size, 3
  torch::Tensor       distance_q_vector, // batch size, 4
  torch::Tensor       out_gidx,
  const torch::Tensor current_position,  // batch_size, 3
  const torch::Tensor goal_position,     // n_boxes, 3
  const torch::Tensor current_quat,
  const torch::Tensor goal_quat,
  const torch::Tensor vec_weight,        // n_boxes, 4, 4
  const torch::Tensor weight,            // n_boxes, 4, 4
  const torch::Tensor vec_convergence,
  const torch::Tensor run_weight,
  const torch::Tensor run_vec_weight,
  const torch::Tensor offset_waypoint,
  const torch::Tensor offset_tstep_fraction,
  const torch::Tensor batch_pose_idx,
  const torch::Tensor project_distance,
  const int           batch_size,
  const int           horizon,
  const int           mode,
  const int           num_goals        = 1,
  const bool          compute_grad     = false,
  const bool          write_distance   = true,
  const bool          use_metric       = false
  );

std::vector<torch::Tensor>
backward_pose_distance(torch::Tensor       out_grad_p,
                       torch::Tensor       out_grad_q,
                       const torch::Tensor grad_distance,   // batch_size, 3
                       const torch::Tensor grad_p_distance, // n_boxes, 3
                       const torch::Tensor grad_q_distance,
                       const torch::Tensor pose_weight,
                       const torch::Tensor grad_p_vec,      // n_boxes, 4, 4
                       const torch::Tensor grad_q_vec,
                       const int           batch_size,
                       const bool          use_distance = false);

// C++ interface


std::vector<torch::Tensor>self_collision_distance_wrapper(
  torch::Tensor out_distance, torch::Tensor out_vec,
  torch::Tensor sparse_index,
  const torch::Tensor robot_spheres,    // batch_size x n_spheres x 4
  const torch::Tensor collision_offset, // n_spheres
  const torch::Tensor weight,
  const torch::Tensor collision_matrix, // n_spheres
  const torch::Tensor thread_locations, const int thread_locations_size,
  const int batch_size, const int nspheres, const bool compute_grad = false,
  const int ndpt = 8, const bool debug = false)
{
  CHECK_INPUT(out_distance);
  CHECK_INPUT(out_vec);
  CHECK_INPUT(robot_spheres);
  CHECK_INPUT(collision_offset);
  CHECK_INPUT(sparse_index);
  CHECK_INPUT(weight);
  CHECK_INPUT(thread_locations);
  CHECK_INPUT(collision_matrix);
  const at::cuda::OptionalCUDAGuard guard(robot_spheres.device());

  return self_collision_distance(
    out_distance, out_vec, sparse_index, robot_spheres,
    collision_offset, weight, collision_matrix, thread_locations,
    thread_locations_size, batch_size, nspheres, compute_grad, ndpt, debug);
}

std::vector<torch::Tensor>sphere_obb_clpt_wrapper(
  const torch::Tensor sphere_position, // batch_size, 4
  torch::Tensor distance,
  torch::Tensor closest_point,         // batch size, 3
  torch::Tensor sparsity_idx, const torch::Tensor weight,
  const torch::Tensor activation_distance,
  const torch::Tensor max_distance,
  const torch::Tensor obb_accel,       // n_boxes, 4, 4
  const torch::Tensor obb_bounds,      // n_boxes, 3
  const torch::Tensor obb_pose,        // n_boxes, 4, 4
  const torch::Tensor obb_enable,      // n_boxes, 4, 4
  const torch::Tensor n_env_obb,       // n_boxes, 4, 4
  const torch::Tensor env_query_idx,   // n_boxes, 4, 4
  const int max_nobs, const int batch_size, const int horizon,
  const int n_spheres,
  const bool transform_back, const bool compute_distance,
  const bool use_batch_env, const bool sum_collisions = true,
  const bool compute_esdf = false)
{
  const at::cuda::OptionalCUDAGuard guard(sphere_position.device());

  CHECK_INPUT(distance);
  CHECK_INPUT(closest_point);
  CHECK_INPUT(sphere_position);
  CHECK_INPUT(sparsity_idx);
  CHECK_INPUT(weight);
  CHECK_INPUT(activation_distance);
  CHECK_INPUT(obb_accel);
  return sphere_obb_clpt(
    sphere_position, distance, closest_point, sparsity_idx, weight,
    activation_distance, max_distance, obb_accel, obb_bounds, obb_pose, obb_enable,
    n_env_obb, env_query_idx, max_nobs, batch_size, horizon, n_spheres,
    transform_back, compute_distance, use_batch_env, sum_collisions, compute_esdf);
}

std::vector<torch::Tensor>swept_sphere_obb_clpt_wrapper(
  const torch::Tensor sphere_position, // batch_size, 4
  torch::Tensor distance,
  torch::Tensor closest_point,         // batch size, 3
  torch::Tensor sparsity_idx, const torch::Tensor weight,
  const torch::Tensor activation_distance, const torch::Tensor speed_dt,
  const torch::Tensor obb_accel,       // n_boxes, 4, 4
  const torch::Tensor obb_bounds,      // n_boxes, 3
  const torch::Tensor obb_pose,        // n_boxes, 4, 4
  const torch::Tensor obb_enable,      // n_boxes, 4, 4
  const torch::Tensor n_env_obb,       // n_boxes, 4, 4
  const torch::Tensor env_query_idx,   // n_boxes, 4, 4
  const int max_nobs, const int batch_size, const int horizon,
  const int n_spheres, const int sweep_steps, const bool enable_speed_metric,
  const bool transform_back, const bool compute_distance,
  const bool use_batch_env, const bool sum_collisions = true)
{
  const at::cuda::OptionalCUDAGuard guard(sphere_position.device());

  CHECK_INPUT(distance);
  CHECK_INPUT(closest_point);
  CHECK_INPUT(sphere_position);

  return swept_sphere_obb_clpt(
    sphere_position,
    distance, closest_point, sparsity_idx, weight, activation_distance,
    speed_dt, obb_accel, obb_bounds, obb_pose, obb_enable, n_env_obb,
    env_query_idx, max_nobs, batch_size, horizon, n_spheres, sweep_steps,
    enable_speed_metric, transform_back, compute_distance, use_batch_env, sum_collisions);
}

std::vector<torch::Tensor>
sphere_voxel_clpt_wrapper(const torch::Tensor sphere_position, // batch_size, 3
                torch::Tensor distance,
                torch::Tensor closest_point,         // batch size, 3
                torch::Tensor sparsity_idx, const torch::Tensor weight,
                const torch::Tensor activation_distance,
                const torch::Tensor max_distance,
                const torch::Tensor grid_features,       // n_boxes, 4, 4
                const torch::Tensor grid_params,      // n_boxes, 3
                const torch::Tensor grid_pose,        // n_boxes, 4, 4
                const torch::Tensor grid_enable,      // n_boxes, 4, 4
                const torch::Tensor n_env_grid,
                const torch::Tensor env_query_idx,   // n_boxes, 4, 4
                const int max_ngrid, const int batch_size, const int horizon,
                const int n_spheres, const bool transform_back,
                const bool compute_distance, const bool use_batch_env,
                const bool sum_collisions,
                const bool compute_esdf)
{
    const at::cuda::OptionalCUDAGuard guard(sphere_position.device());

  CHECK_INPUT(distance);
  CHECK_INPUT(closest_point);
  CHECK_INPUT(sphere_position);
  return sphere_voxel_clpt(sphere_position, distance, closest_point, sparsity_idx, weight,
  activation_distance, max_distance, grid_features, grid_params,
  grid_pose, grid_enable, n_env_grid, env_query_idx, max_ngrid, batch_size, horizon, n_spheres,
  transform_back, compute_distance, use_batch_env, sum_collisions, compute_esdf);
}

std::vector<torch::Tensor>pose_distance_wrapper(
  torch::Tensor out_distance, torch::Tensor out_position_distance,
  torch::Tensor out_rotation_distance,
  torch::Tensor distance_p_vector,      // batch size, 3
  torch::Tensor distance_q_vector,      // batch size, 4
  torch::Tensor out_gidx,
  const torch::Tensor current_position, // batch_size, 3
  const torch::Tensor goal_position,    // n_boxes, 3
  const torch::Tensor current_quat, const torch::Tensor goal_quat,
  const torch::Tensor vec_weight,       // n_boxes, 4, 4
  const torch::Tensor weight, const torch::Tensor vec_convergence,
  const torch::Tensor run_weight, const torch::Tensor run_vec_weight,
  const torch::Tensor offset_waypoint, const torch::Tensor offset_tstep_fraction,
  const torch::Tensor batch_pose_idx,
  const torch::Tensor project_distance,
  const int batch_size, const int horizon,
  const int mode, const int num_goals = 1, const bool compute_grad = false,
  const bool write_distance = false, const bool use_metric = false)
{
  // at::cuda::DeviceGuard guard(angle.device());
  CHECK_INPUT(out_distance);
  CHECK_INPUT(out_position_distance);
  CHECK_INPUT(out_rotation_distance);
  CHECK_INPUT(distance_p_vector);
  CHECK_INPUT(distance_q_vector);
  CHECK_INPUT(current_position);
  CHECK_INPUT(goal_position);
  CHECK_INPUT(current_quat);
  CHECK_INPUT(goal_quat);
  CHECK_INPUT(batch_pose_idx);
  CHECK_INPUT(offset_waypoint);
  CHECK_INPUT(offset_tstep_fraction);
  CHECK_INPUT(project_distance);
  const at::cuda::OptionalCUDAGuard guard(current_position.device());

  return pose_distance(
    out_distance, out_position_distance, out_rotation_distance,
    distance_p_vector, distance_q_vector, out_gidx, current_position,
    goal_position, current_quat, goal_quat, vec_weight, weight,
    vec_convergence, run_weight, run_vec_weight,
    offset_waypoint,
    offset_tstep_fraction,
    batch_pose_idx,
    project_distance,
    batch_size,
    horizon, mode, num_goals, compute_grad, write_distance, use_metric);
}

std::vector<torch::Tensor>backward_pose_distance_wrapper(
  torch::Tensor out_grad_p, torch::Tensor out_grad_q,
  const torch::Tensor grad_distance,   // batch_size, 3
  const torch::Tensor grad_p_distance, // n_boxes, 3
  const torch::Tensor grad_q_distance, const torch::Tensor pose_weight,
  const torch::Tensor grad_p_vec,      // n_boxes, 4, 4
  const torch::Tensor grad_q_vec, const int batch_size,
  const bool use_distance)
{
  CHECK_INPUT(out_grad_p);
  CHECK_INPUT(out_grad_q);
  CHECK_INPUT(grad_distance);
  CHECK_INPUT(grad_p_distance);
  CHECK_INPUT(grad_q_distance);

  const at::cuda::OptionalCUDAGuard guard(grad_distance.device());

  return backward_pose_distance(
    out_grad_p, out_grad_q, grad_distance, grad_p_distance, grad_q_distance,
    pose_weight, grad_p_vec, grad_q_vec, batch_size, use_distance);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("pose_distance",           &pose_distance_wrapper, "Pose Distance (curobolib)");
  m.def("pose_distance_backward",  &backward_pose_distance_wrapper,
        "Pose Distance Backward (curobolib)");

  m.def("closest_point",           &sphere_obb_clpt_wrapper,
        "Closest Point OBB(curobolib)");
  m.def("swept_closest_point",     &swept_sphere_obb_clpt_wrapper,
        "Swept Closest Point OBB(curobolib)");
  m.def("closest_point_voxel",           &sphere_voxel_clpt_wrapper,
        "Closest Point Voxel(curobolib)");
  m.def("swept_closest_point_voxel",           &swept_sphere_voxel_clpt,
        "Swpet Closest Point Voxel(curobolib)");



  m.def("self_collision_distance", &self_collision_distance_wrapper,
        "Self Collision Distance (curobolib)");
}
