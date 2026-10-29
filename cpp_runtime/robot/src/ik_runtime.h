// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

#include "curobo_robot_runtime/inverse_kinematics.h"
#include "robot_runtime.h"

namespace curobo::robot::detail {

class SceneRuntime;
class LinkSphereAttachment;
class RolloutRuntime;
class RolloutStreams;
struct IkModelParameters;
struct RobotCollisionModel;

bool has_valid_solution_for_each_batch(const std::vector<uint8_t>& success,
                                       int batch_size, int num_solutions);

class IkRuntime final {
 public:
  IkRuntime(std::string artifact_directory, const InverseKinematicsConfig& config,
            const Scene* scene);
  IkRuntime(const IkRuntime& shared_robot, const InverseKinematicsConfig& config);
  ~IkRuntime();

  IkRuntime(const IkRuntime&) = delete;
  IkRuntime& operator=(const IkRuntime&) = delete;

  CuroboRobotInfo info() const;
  const InverseKinematicsConfig& config() const { return config_; }
  int attachment_sphere_capacity(const std::string& link_name) const;
  void attach_object(const SphereAttachment& attachment);
  void detach_object(const std::string& link_name);
  std::shared_ptr<RobotCollisionModel> collision_model() const;
  void set_link_collision_mask(const std::string& link, uint32_t layers);
  void set_link_collision_enabled(const std::string& link_name, bool enabled);
  void update_link_inertial(const LinkInertialUpdate& update);
  void update_links_inertial(const LinkInertialUpdates& updates);
  bool solve(const IkSolveRequest& request, uint8_t* host_success = nullptr,
             std::size_t host_success_count = 0) const;

 private:
  IkRuntime(std::string artifact_directory, const InverseKinematicsConfig& config,
            const SceneRuntime* scene, std::shared_ptr<IkModelParameters> model);
  void validate_config() const;
  void validate_scene() const;
  void load_constants();
  void load_kernels();
  void allocate_workspace();
  bool solve_eager(const IkSolveRequest& request, CUstream stream,
                   uint8_t* host_success,
                   std::size_t host_success_count) const;
  void solve_lm(const IkSolveRequest& request, CUstream stream) const;
  void solve_lbfgs(const IkSolveRequest& request, CUstream stream) const;
  void finalize_solution(CUdeviceptr joint_position,
                         const IkSolveRequest& request, int problems,
                         CUstream stream) const;
  bool lm_result_can_exit_early(const IkSolveRequest& request,
                                cudaStream_t stream, uint8_t* host_success,
                                std::size_t host_success_count) const;
  static bool same_graph_request(const IkSolveRequest& left,
                                 const IkSolveRequest& right);
  struct GraphCache {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t executable{nullptr};
    IkSolveRequest request{};
    bool initialized{false};
    cudaStream_t last_stream{nullptr};
  };
  void launch_graph_stage(GraphCache* cache, const IkSolveRequest& request,
                          bool lm_stage) const;
  static void destroy_graph(GraphCache* cache) noexcept;
  void launch_kinematics(CUdeviceptr joint_position, int problems,
                         CUstream stream) const;
  void launch_error(CUdeviceptr joint_position, CUdeviceptr error,
                    CUdeviceptr position_error, CUdeviceptr orientation_error,
                    const IkSolveRequest& request, int problems,
                    bool include_joint_limits, CUstream stream) const;
  void launch_collision(int problems, bool optimizer_cost,
                        CUstream stream) const;
  void launch_self_collision(int problems, bool optimizer_cost,
                             CUstream stream) const;
  void launch_kinematics_backward(int problems, CUstream stream) const;
  void launch_cspace(CUdeviceptr joint_position, int problems,
                     CUstream stream) const;
  void evaluate_lbfgs(CUdeviceptr joint_position, int problems,
                      const IkSolveRequest& request, CUstream stream) const;
  void prepare_lbfgs(int problems, CUstream stream) const;
  void prepare_search_actions(int problems, CUstream stream) const;
  void line_search(int problems, CUstream stream) const;
  void lbfgs_step(int problems, CUstream stream) const;

  std::string artifact_directory_;
  InverseKinematicsConfig config_;
  const SceneRuntime* scene_{nullptr};
  std::shared_ptr<IkModelParameters> model_;
  const curobo_robot_runtime::IkManifest& manifest_;
  const curobo_robot_runtime::IkKernelManifest& prepare_search_actions_spec_;
  const curobo_robot_runtime::IkKernelManifest& line_search_spec_;
  mutable curobo_robot_runtime::DeviceBuffer seed_offset_;
  curobo_robot_runtime::DeviceBuffer collision_weight_;
  curobo_robot_runtime::DeviceBuffer collision_activation_distance_;
  curobo_robot_runtime::DeviceBuffer inactive_effort_bounds_;
  curobo_robot_runtime::DeviceBuffer cspace_weight_;
  curobo_robot_runtime::DeviceBuffer cspace_activation_distance_;
  curobo_robot_runtime::DeviceBuffer cspace_target_weight_;
  curobo_robot_runtime::DeviceBuffer cspace_target_dof_weight_;
  curobo_robot_runtime::DeviceBuffer cspace_regularization_weight_;
  curobo_robot_runtime::DeviceBuffer start_cspace_weight_;
  curobo_robot_runtime::DeviceBuffer start_cspace_dof_weight_;
  curobo_robot_runtime::DeviceBuffer optimizer_world_collision_weight_;
  curobo_robot_runtime::DeviceBuffer optimizer_self_collision_weight_;
  curobo_robot_runtime::DeviceBuffer optimizer_collision_activation_distance_;
  curobo_robot_runtime::DeviceBuffer line_search_scale_;
  curobo_robot_runtime::DeviceBuffer optimization_step_max_;
  curobo_robot_runtime::DeviceBuffer zero_state_;
  curobo_robot_runtime::DeviceBuffer zero_indices_;
  curobo_robot_runtime::DeviceBuffer scene_indices_;

  mutable curobo_robot_runtime::DeviceBuffer joint_position_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_position_;
  mutable curobo_robot_runtime::DeviceBuffer link_position_;
  mutable curobo_robot_runtime::DeviceBuffer link_quaternion_;
  mutable curobo_robot_runtime::DeviceBuffer link_jacobian_;
  mutable curobo_robot_runtime::DeviceBuffer global_cumulative_;
  mutable curobo_robot_runtime::DeviceBuffer robot_spheres_;
  mutable curobo_robot_runtime::DeviceBuffer zero_link_position_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer zero_link_quaternion_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer scaled_jacobian_;
  mutable curobo_robot_runtime::DeviceBuffer jt_error_;
  mutable curobo_robot_runtime::DeviceBuffer current_error_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_error_;
  mutable curobo_robot_runtime::DeviceBuffer current_position_error_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_position_error_;
  mutable curobo_robot_runtime::DeviceBuffer current_orientation_error_;
  mutable curobo_robot_runtime::DeviceBuffer rank_score_;
  mutable curobo_robot_runtime::DeviceBuffer rank_success_;
  mutable curobo_robot_runtime::DeviceBuffer candidate_orientation_error_;
  mutable curobo_robot_runtime::DeviceBuffer damping_;
  mutable curobo_robot_runtime::DeviceBuffer predicted_reduction_;
  mutable curobo_robot_runtime::DeviceBuffer collision_distance_;
  mutable curobo_robot_runtime::DeviceBuffer collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_distance_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_pair_distance_;
  mutable curobo_robot_runtime::DeviceBuffer self_collision_sparse_index_;
  mutable curobo_robot_runtime::DeviceBuffer collision_joint_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer cspace_cost_;
  mutable curobo_robot_runtime::DeviceBuffer cspace_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer zero_effort_;
  mutable curobo_robot_runtime::DeviceBuffer zero_effort_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer zero_derivatives_;
  mutable curobo_robot_runtime::DeviceBuffer effort_position_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer effort_velocity_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer effort_acceleration_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer rnea_forward_cache_;
  mutable curobo_robot_runtime::DeviceBuffer goalset_index_;
  mutable curobo_robot_runtime::DeviceBuffer output_goalset_index_;

  mutable curobo_robot_runtime::DeviceBuffer action_;
  mutable curobo_robot_runtime::DeviceBuffer gradient_;
  mutable curobo_robot_runtime::DeviceBuffer cost_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_action_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_cost_;
  mutable curobo_robot_runtime::DeviceBuffer best_action_;
  mutable curobo_robot_runtime::DeviceBuffer best_cost_;
  mutable curobo_robot_runtime::DeviceBuffer best_iteration_;
  mutable curobo_robot_runtime::DeviceBuffer current_iteration_;
  mutable curobo_robot_runtime::DeviceBuffer converged_;
  mutable curobo_robot_runtime::DeviceBuffer step_direction_;
  mutable curobo_robot_runtime::DeviceBuffer reference_action_;
  mutable curobo_robot_runtime::DeviceBuffer reference_gradient_;
  mutable curobo_robot_runtime::DeviceBuffer history_rho_;
  mutable curobo_robot_runtime::DeviceBuffer history_y_;
  mutable curobo_robot_runtime::DeviceBuffer history_s_;
  mutable curobo_robot_runtime::DeviceBuffer history_head_;
  mutable curobo_robot_runtime::DeviceBuffer history_count_;
  mutable curobo_robot_runtime::DeviceBuffer curvature_update_valid_;
  mutable curobo_robot_runtime::DeviceBuffer search_actions_;
  mutable curobo_robot_runtime::DeviceBuffer exploration_indices_;
  mutable curobo_robot_runtime::DeviceBuffer selected_indices_;
  mutable curobo_robot_runtime::DeviceBuffer line_search_index_;
  mutable curobo_robot_runtime::DeviceBuffer line_search_status_;
  curobo_robot_runtime::DeviceBuffer line_search_beta_;

  mutable GraphCache lm_graph_{};
  mutable GraphCache lbfgs_graph_{};
  cudaStream_t graph_capture_stream_{nullptr};

  curobo_robot_runtime::KernelModule kinematics_kernel_;
  curobo_robot_runtime::KernelModule kinematics_backward_kernel_;
  curobo_robot_runtime::KernelModule initialize_kernel_;
  curobo_robot_runtime::KernelModule error_kernel_;
  curobo_robot_runtime::KernelModule update_kernel_;
  curobo_robot_runtime::KernelModule finalize_kernel_;
  curobo_robot_runtime::KernelModule rank_kernel_;
  curobo_robot_runtime::KernelModule lm_kernel_;
  curobo_robot_runtime::KernelModule cspace_kernel_;
  curobo_robot_runtime::KernelModule prepare_initial_state_kernel_;
  curobo_robot_runtime::KernelModule prepare_search_actions_kernel_;
  curobo_robot_runtime::KernelModule combine_lbfgs_rollout_kernel_;
  curobo_robot_runtime::KernelModule line_search_kernel_;
  curobo_robot_runtime::KernelModule lbfgs_kernel_;
  curobo_robot_runtime::KernelModule collision_cuboid_kernel_;
  curobo_robot_runtime::KernelModule collision_capsule_kernel_;
  curobo_robot_runtime::KernelModule collision_cylinder_kernel_;
  curobo_robot_runtime::KernelModule collision_mesh_kernel_;
  curobo_robot_runtime::KernelModule collision_voxel_kernel_;
  curobo_robot_runtime::KernelModule self_collision_kernel_;
  std::unique_ptr<RolloutStreams> rollout_streams_;
};

}  // namespace curobo::robot::detail
