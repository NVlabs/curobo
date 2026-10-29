// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "motion_planner_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_check.h"
#include "grasp_collision_runtime.h"
#include "ik_runtime.h"
#include "motion_planner_cuda.h"
#include "rrt_runtime.h"
#include "trajectory_optimizer_runtime.h"
#include "warp_abi.h"

namespace curobo::robot::detail {

TrajectoryOptimizationCandidates MotionPlannerRuntime::candidates() const {
  return trajectory_optimizer_->candidates();
}

TimeOptTrajectory MotionPlannerRuntime::timeopt_trajectory() const {
  return trajectory_optimizer_->timeopt_trajectory();
}
void MotionPlannerRuntime::set_execution_speed_scale(float scale) {
  trajectory_optimizer_->set_execution_speed_scale(scale);
  config_.trajectory_optimization.execution_speed_scale = scale;
}

const int32_t *MotionPlannerRuntime::validate(const TrajectoryValidationRequest &request) const {
  return trajectory_optimizer_->validate(request);
}
namespace {

size_t checked_product(size_t left, size_t right, const char* description) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    throw std::invalid_argument(std::string(description) + " size overflow");
  }
  return left * right;
}

unsigned int grid_for(size_t count, int block_size) {
  if (count == 0 || block_size <= 0) {
    throw std::invalid_argument("kernel launch dimensions must be positive");
  }
  const auto blocks =
      (count + static_cast<size_t>(block_size) - 1) /
      static_cast<size_t>(block_size);
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument("kernel grid exceeds the CUDA launch range");
  }
  return static_cast<unsigned int>(blocks);
}

WarpArray array_1d(CUdeviceptr pointer, int count, int element_bytes) {
  return warp_array_1d(static_cast<uint64_t>(pointer), count, element_bytes);
}

WarpArray array_2d(CUdeviceptr pointer, int rows, int columns,
                   int element_bytes) {
  return warp_array_2d(static_cast<uint64_t>(pointer), rows, columns,
                       element_bytes);
}

WarpArray array_3d(CUdeviceptr pointer, int first, int second, int third,
                   int element_bytes) {
  return warp_array_3d(static_cast<uint64_t>(pointer), first, second, third,
                       element_bytes);
}

std::array<float, 3> axis_vector(CartesianAxis axis) {
  switch (axis) {
    case CartesianAxis::X:
      return {1.0F, 0.0F, 0.0F};
    case CartesianAxis::Y:
      return {0.0F, 1.0F, 0.0F};
    case CartesianAxis::Z:
      return {0.0F, 0.0F, 1.0F};
  }
  throw std::invalid_argument("grasp axis is invalid");
}

void validate_phase_output(const GraspPhaseOutput& output,
                           const char* phase_name) {
  if (output.optimized_knots == nullptr ||
      output.trajectory_position == nullptr ||
      output.trajectory_velocity == nullptr ||
      output.trajectory_acceleration == nullptr ||
      output.trajectory_jerk == nullptr || output.success == nullptr ||
      output.cost == nullptr || output.collision_cost == nullptr ||
      output.trajectory_dt == nullptr || output.used_rrt == nullptr ||
      output.rrt_iterations == nullptr) {
    throw std::invalid_argument(std::string(phase_name) +
                                " grasp phase outputs must be non-null");
  }
}

PoseMotionPlanningRequest make_pose_request(
    const float* goal_positions, const float* goal_quaternions,
    const float* current_position, const float* current_velocity,
    const float* current_acceleration, const GraspPhaseOutput& output,
    cudaStream_t stream) {
  PoseMotionPlanningRequest request;
  request.goal_positions = goal_positions;
  request.goal_quaternions = goal_quaternions;
  request.current_position = current_position;
  request.current_velocity = current_velocity;
  request.current_acceleration = current_acceleration;
  request.optimized_knots = output.optimized_knots;
  request.trajectory_position = output.trajectory_position;
  request.trajectory_velocity = output.trajectory_velocity;
  request.trajectory_acceleration = output.trajectory_acceleration;
  request.trajectory_jerk = output.trajectory_jerk;
  request.success = output.success;
  request.cost = output.cost;
  request.collision_cost = output.collision_cost;
  request.trajectory_dt = output.trajectory_dt;
  request.used_rrt = output.used_rrt;
  request.rrt_iterations = output.rrt_iterations;
  request.selected_seed = output.selected_seed;
  request.planning_attempts = output.planning_attempts;
  request.stream = stream;
  return request;
}

}  // namespace

MotionPlannerRuntime::MotionPlannerRuntime(
    std::string artifact_directory, const MotionPlanningConfig& config,
    const Scene* scene)
    : artifact_directory_(std::move(artifact_directory)),
      scene_handle_(scene),
      config_(config),
      manifest_(curobo_robot_runtime::load_motion_planning_manifest(
          artifact_directory_)) {
  validate_config();
  CRR_CUDA_CHECK(
      cudaSetDevice(config_.trajectory_optimization.device_ordinal));
  TrajectorySeedFitterConfig seed_fitter_config;
  seed_fitter_config.device_ordinal =
      config_.trajectory_optimization.device_ordinal;
  seed_fitter_config.num_seeds = config_.trajectory_optimization.num_seeds;
  seed_fitter_config.use_cuda_graph = config_.use_cuda_graph;
  seed_fitter_ = std::make_unique<TrajectorySeedFitter>(
      artifact_directory_, seed_fitter_config);
  auto inverse_kinematics_config = config_.inverse_kinematics;
  inverse_kinematics_config.use_cuda_graph =
      inverse_kinematics_config.use_cuda_graph && config_.use_cuda_graph;
  inverse_kinematics_ = std::make_unique<InverseKinematics>(
      artifact_directory_, inverse_kinematics_config, scene);
  auto collision = inverse_kinematics_->runtime_->collision_model();
  grasp_collision_workspace_ = std::make_unique<GraspCollisionWorkspace>(*collision);
  rrt_ = std::make_unique<RrtRuntime>(artifact_directory_, config_.rrt, scene, collision);
  trajectory_optimizer_ = std::make_unique<TrajectoryOptimizerRuntime>(
      artifact_directory_, config_.trajectory_optimization, scene, collision);
  // Separate graph/workspace ownership; link-local geometry is shared before FK.
  grasp_inverse_kinematics_ = std::make_unique<InverseKinematics>(
      *inverse_kinematics_, inverse_kinematics_config);
  validate_config();
  load_fit_artifact();
  allocate_workspace();
  if (config_.use_cuda_graph) {
    CRR_CUDA_CHECK(
        cudaStreamCreateWithFlags(&internal_stream_, cudaStreamNonBlocking));
  }
}

MotionPlannerRuntime::~MotionPlannerRuntime() {
  cudaSetDevice(config_.trajectory_optimization.device_ordinal);
  for (auto* graph_cache : {
           &default_trajopt_graphs_,
           &goalset_trajopt_graphs_,
           &grasp_approach_trajopt_graphs_,
           &grasp_insert_trajopt_graphs_,
           &grasp_lift_trajopt_graphs_,
       }) {
    destroy_graph_cache(graph_cache);
  }
  grasp_inverse_kinematics_.reset();
  inverse_kinematics_.reset();
  trajectory_optimizer_.reset();
  seed_fitter_.reset();
  rrt_.reset();
  if (internal_stream_ != nullptr) {
    cudaStreamDestroy(internal_stream_);
    internal_stream_ = nullptr;
  }
}

CuroboRobotInfo MotionPlannerRuntime::robot_info() const {
  return trajectory_optimizer_->robot_info();
}

MotionPlanningInfo MotionPlannerRuntime::info() const {
  return MotionPlanningInfo{
      manifest_.num_dof,
      manifest_.num_knots,
      manifest_.trajectory_steps,
      manifest_.max_rrt_path_nodes,
  };
}

int MotionPlannerRuntime::attachment_sphere_capacity(
    const std::string& link_name) const {
  return inverse_kinematics_->attachment_sphere_capacity(link_name);
}

std::unique_ptr<InverseKinematics> MotionPlannerRuntime::create_inverse_kinematics(
    const InverseKinematicsConfig& config) const {
  return std::make_unique<InverseKinematics>(*inverse_kinematics_, config);
}

void MotionPlannerRuntime::attach_object(
    const SphereAttachment& attachment) {
  inverse_kinematics_->attach_object(attachment);
}

void MotionPlannerRuntime::set_link_collision_mask(const std::string& link_name, uint32_t layers) {
  inverse_kinematics_->set_link_collision_mask(link_name, layers);
}

void MotionPlannerRuntime::detach_object(const std::string& link_name) {
  inverse_kinematics_->detach_object(link_name);
}

void MotionPlannerRuntime::set_collision_mask(uint32_t layers) {
  const auto model = inverse_kinematics_->runtime_->collision_model();
  CRR_CUDA_CHECK(cudaSetDevice(model->device_ordinal));
  CRR_CUDA_CHECK(launch_set_sphere_collision_mask(
      reinterpret_cast<uint32_t*>(model->sphere_collision_mask.ptr()), nullptr,
      static_cast<int>(model->sphere_collision_mask.bytes() / sizeof(uint32_t)), layers, nullptr));
  CRR_CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void MotionPlannerRuntime::update_link_inertial(
    const LinkInertialUpdate& update) {
  update_links_inertial({update});
}

void MotionPlannerRuntime::update_links_inertial(
    const LinkInertialUpdates& updates) const {
  inverse_kinematics_->update_links_inertial(updates);
  rrt_->update_links_inertial(updates);
  trajectory_optimizer_->update_links_inertial(updates);
}

void MotionPlannerRuntime::set_collision_links(
    const std::vector<std::string>& links, bool enabled) const {
  for (const auto& link : links) {
    attachment_sphere_capacity(link);
  }
  for (const auto& link : links) {
    inverse_kinematics_->set_link_collision_enabled(link, enabled);
  }
}

bool MotionPlannerRuntime::run_goalset_selection(
    const GraspPlanningRequest& request, cudaStream_t stream) const {
  const int seeds = config_.trajectory_optimization.num_seeds;
  const int solutions = request.use_trajopt_for_goalset_selection ? seeds : 1;
  const int tool_frames = trajectory_optimizer_->info().num_tool_frames;
  IkSolveRequest ik_request;
  ik_request.goal_positions = request.grasp_positions;
  ik_request.goal_quaternions = request.grasp_quaternions;
  ik_request.current_positions = request.current_position;
  ik_request.solution = reinterpret_cast<float*>(ik_solutions_.ptr());
  ik_request.success = reinterpret_cast<uint8_t*>(ik_success_.ptr());
  ik_request.position_error =
      reinterpret_cast<float*>(ik_position_error_.ptr());
  ik_request.orientation_error =
      reinterpret_cast<float*>(ik_orientation_error_.ptr());
  ik_request.collision_cost =
      reinterpret_cast<float*>(ik_collision_cost_.ptr());
  ik_request.goalset_index =
      reinterpret_cast<int32_t*>(ik_goalset_index_.ptr());
  ik_request.batch_size = 1;
  ik_request.num_goalset = request.num_grasps;
  ik_request.num_solutions = solutions;
  ik_request.stream = stream;
  std::vector<uint8_t> success(static_cast<size_t>(solutions), 0U);
  const bool success_is_ready = grasp_inverse_kinematics_->solve_with_host_success(
      ik_request, success.data(), success.size());
  if (!success_is_ready) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        success.data(), reinterpret_cast<const void*>(ik_success_.ptr()),
        success.size(), cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  const auto first_valid = std::find_if(
      success.begin(), success.end(), [](uint8_t value) { return value != 0U; });
  if (first_valid == success.end()) {
    return false;
  }
  const size_t first_valid_index =
      static_cast<size_t>(std::distance(success.begin(), first_valid));
  const size_t dof_bytes =
      static_cast<size_t>(manifest_.num_dof) * sizeof(float);

  int chosen_seed = static_cast<int>(first_valid_index);
  if (request.use_trajopt_for_goalset_selection) {
    for (size_t seed = 0; seed < success.size(); ++seed) {
      if (success.at(seed) != 0U) {
        continue;
      }
      CRR_CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<void*>(ik_solutions_.ptr() + seed * dof_bytes),
          reinterpret_cast<const void*>(ik_solutions_.ptr() +
                                        first_valid_index * dof_bytes),
          dof_bytes, cudaMemcpyDeviceToDevice, stream));
      for (auto* metric : {&ik_position_error_, &ik_orientation_error_,
                           &ik_collision_cost_}) {
        CRR_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(metric->ptr() + seed * sizeof(float)),
            reinterpret_cast<const void*>(metric->ptr() +
                                          first_valid_index * sizeof(float)),
            sizeof(float), cudaMemcpyDeviceToDevice, stream));
      }
      CRR_CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<void*>(ik_goalset_index_.ptr() +
                                  seed * tool_frames * sizeof(int32_t)),
          reinterpret_cast<const void*>(ik_goalset_index_.ptr() +
                                        first_valid_index * tool_frames *
                                            sizeof(int32_t)),
          static_cast<size_t>(tool_frames) * sizeof(int32_t),
          cudaMemcpyDeviceToDevice, stream));
    }

    CRR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(pose_joint_goal_.ptr()),
        reinterpret_cast<const void*>(ik_solutions_.ptr() +
                                      first_valid_index * dof_bytes),
        dof_bytes, cudaMemcpyDeviceToDevice, stream));

    MotionPlanningRequest motion_request;
    motion_request.start_position = request.current_position;
    motion_request.start_velocity = request.current_velocity;
    motion_request.start_acceleration = request.current_acceleration;
    motion_request.goal_position =
        reinterpret_cast<const float*>(pose_joint_goal_.ptr());
    motion_request.seed_goal_positions =
        reinterpret_cast<const float*>(ik_solutions_.ptr());
    motion_request.optimized_knots =
        reinterpret_cast<float*>(selection_knots_.ptr());
    motion_request.trajectory_position =
        reinterpret_cast<float*>(selection_position_.ptr());
    motion_request.trajectory_velocity =
        reinterpret_cast<float*>(selection_velocity_.ptr());
    motion_request.trajectory_acceleration =
        reinterpret_cast<float*>(selection_acceleration_.ptr());
    motion_request.trajectory_jerk =
        reinterpret_cast<float*>(selection_jerk_.ptr());
    motion_request.success =
        reinterpret_cast<uint8_t*>(selection_success_.ptr());
    motion_request.cost = reinterpret_cast<float*>(selection_cost_.ptr());
    motion_request.collision_cost =
        reinterpret_cast<float*>(selection_collision_cost_.ptr());
    motion_request.trajectory_dt =
        reinterpret_cast<float*>(selection_dt_.ptr());
    motion_request.used_rrt =
        reinterpret_cast<uint8_t*>(selection_used_rrt_.ptr());
    motion_request.rrt_iterations =
        reinterpret_cast<int32_t*>(selection_rrt_iterations_.ptr());
    motion_request.seed_valid =
        reinterpret_cast<const uint8_t*>(ik_success_.ptr());
    motion_request.selected_seed =
        reinterpret_cast<int32_t*>(pose_selected_seed_.ptr());
    motion_request.stream = stream;
    motion_request.finetune_attempts = 1;
    motion_request.finetune_dt_scale = config_.trajectory_optimization.finetune_dt_scale;
    const bool planned =
        plan_impl(motion_request, true, &goalset_trajopt_graphs_);

    int32_t planned_seed = -1;
    if (!planned) {
      return false;
    }
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        &planned_seed, reinterpret_cast<const void*>(pose_selected_seed_.ptr()),
        sizeof(planned_seed), cudaMemcpyDeviceToHost, stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (planned_seed < 0 || planned_seed >= seeds) {
      return false;
    }
    chosen_seed = planned_seed;
  }

  GraspPosePreparation preparation{
      request.grasp_positions, request.grasp_quaternions,
      reinterpret_cast<const int32_t*>(ik_goalset_index_.ptr()),
      chosen_seed, tool_frames, request.num_grasps,
      static_cast<int>(request.grasp_approach_axis),
      static_cast<int>(request.grasp_lift_axis),
      request.grasp_approach_offset, request.grasp_lift_offset,
      request.grasp_approach_in_tool_frame, request.grasp_lift_in_tool_frame,
      reinterpret_cast<float*>(selected_grasp_position_.ptr()),
      reinterpret_cast<float*>(selected_grasp_quaternion_.ptr()),
      reinterpret_cast<float*>(approach_goal_position_.ptr()),
      reinterpret_cast<float*>(approach_goal_quaternion_.ptr()),
      reinterpret_cast<float*>(lift_goal_position_.ptr()),
      reinterpret_cast<float*>(lift_goal_quaternion_.ptr()),
      request.goalset_index,
      reinterpret_cast<uint8_t*>(selection_success_.ptr()),
  };
  CRR_CUDA_CHECK(launch_prepare_grasp_poses(preparation, stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(selected_grasp_configuration_.ptr()),
      reinterpret_cast<const void*>(ik_solutions_.ptr() +
                                    static_cast<size_t>(chosen_seed) *
                                        dof_bytes),
      dof_bytes, cudaMemcpyDeviceToDevice, stream));
  if (request.selected_grasp_configuration != nullptr) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        request.selected_grasp_configuration,
        reinterpret_cast<const void*>(selected_grasp_configuration_.ptr()),
        dof_bytes, cudaMemcpyDeviceToDevice, stream));
  }
  const auto metric_offset =
      static_cast<size_t>(chosen_seed) * sizeof(float);
  for (const auto& output : {
           std::pair<float*, curobo_robot_runtime::DeviceBuffer*>(
               request.goalset_position_error, &ik_position_error_),
           {request.goalset_orientation_error, &ik_orientation_error_},
           {request.goalset_collision_cost, &ik_collision_cost_},
       }) {
    if (output.first != nullptr) {
      CRR_CUDA_CHECK(cudaMemcpyAsync(
          output.first,
          reinterpret_cast<const void*>(output.second->ptr() + metric_offset),
          sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
  }
  if (request.goalset_success != nullptr) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        request.goalset_success,
        reinterpret_cast<const void*>(selection_success_.ptr()),
        sizeof(uint8_t), cudaMemcpyDeviceToDevice, stream));
  }
  uint8_t valid_pose = 0U;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      &valid_pose, reinterpret_cast<const void*>(selection_success_.ptr()),
      sizeof(valid_pose), cudaMemcpyDeviceToHost, stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return valid_pose != 0U;
}

GraspPhaseOutput MotionPlannerRuntime::grasp_phase_workspace() const {
  GraspPhaseOutput output;
  output.optimized_knots = reinterpret_cast<float*>(selection_knots_.ptr());
  output.trajectory_position =
      reinterpret_cast<float*>(selection_position_.ptr());
  output.trajectory_velocity =
      reinterpret_cast<float*>(selection_velocity_.ptr());
  output.trajectory_acceleration =
      reinterpret_cast<float*>(selection_acceleration_.ptr());
  output.trajectory_jerk = reinterpret_cast<float*>(selection_jerk_.ptr());
  output.success = reinterpret_cast<uint8_t*>(selection_success_.ptr());
  output.cost = reinterpret_cast<float*>(selection_cost_.ptr());
  output.collision_cost =
      reinterpret_cast<float*>(selection_collision_cost_.ptr());
  output.trajectory_dt = reinterpret_cast<float*>(selection_dt_.ptr());
  output.used_rrt = reinterpret_cast<uint8_t*>(selection_used_rrt_.ptr());
  output.rrt_iterations =
      reinterpret_cast<int32_t*>(selection_rrt_iterations_.ptr());
  output.selected_seed = reinterpret_cast<int32_t*>(pose_selected_seed_.ptr());
  output.planning_attempts =
      reinterpret_cast<int32_t*>(selection_planning_attempts_.ptr());
  return output;
}

void MotionPlannerRuntime::copy_grasp_phase_output(
    const GraspPhaseOutput& output, cudaStream_t stream) const {
  const auto copy = [stream](void* destination,
                             const curobo_robot_runtime::DeviceBuffer& source) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(destination,
                                   reinterpret_cast<const void*>(source.ptr()),
                                   source.bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
  };
  copy(output.optimized_knots, selection_knots_);
  copy(output.trajectory_position, selection_position_);
  copy(output.trajectory_velocity, selection_velocity_);
  copy(output.trajectory_acceleration, selection_acceleration_);
  copy(output.trajectory_jerk, selection_jerk_);
  copy(output.success, selection_success_);
  copy(output.cost, selection_cost_);
  copy(output.collision_cost, selection_collision_cost_);
  copy(output.trajectory_dt, selection_dt_);
  copy(output.used_rrt, selection_used_rrt_);
  copy(output.rrt_iterations, selection_rrt_iterations_);
  if (output.selected_seed != nullptr) {
    copy(output.selected_seed, pose_selected_seed_);
  }
  if (output.planning_attempts != nullptr) {
    copy(output.planning_attempts, selection_planning_attempts_);
  }
  const auto executable = timeopt_trajectory();
  const auto copy_controls = [stream](float* destination, const float* source, size_t count) {
    if (destination == nullptr) return;
    if (source != nullptr) {
      CRR_CUDA_CHECK(cudaMemcpyAsync(destination, source, count * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    } else {
      CRR_CUDA_CHECK(cudaMemsetAsync(destination, 0, count * sizeof(float), stream));
    }
  };
  copy_controls(output.spatial_controls, executable.spatial_controls,
                (manifest_.num_knots + 6) * manifest_.num_dof);
  copy_controls(output.timing_controls, executable.timing_controls, 32);
  copy_controls(output.timeopt_duration, executable.duration, 1);
}

bool MotionPlannerRuntime::plan_linear_phase(
    const float* goal_positions, const float* goal_quaternions,
    const float* start_position, const float* start_velocity,
    const float* start_acceleration,
    const std::array<float, 3>& direction, bool direction_in_goal_frame,
    int finetune_attempts,
    const GraspPhaseOutput& output, cudaStream_t stream,
    TrajoptGraphCache* graph_cache,
    const float* known_goal_configuration) const {
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_phase_goal_position_.ptr()), goal_positions,
      grasp_phase_goal_position_.bytes(), cudaMemcpyDeviceToDevice, stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_phase_goal_quaternion_.ptr()),
      goal_quaternions, grasp_phase_goal_quaternion_.bytes(),
      cudaMemcpyDeviceToDevice, stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_phase_start_.ptr()), start_position,
      grasp_phase_start_.bytes(), cudaMemcpyDeviceToDevice, stream));
  if (start_velocity == nullptr) {
    grasp_phase_start_velocity_.clear(stream);
  } else {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(grasp_phase_start_velocity_.ptr()),
        start_velocity, grasp_phase_start_velocity_.bytes(),
        cudaMemcpyDeviceToDevice, stream));
  }
  if (start_acceleration == nullptr) {
    grasp_phase_start_acceleration_.clear(stream);
  } else {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(grasp_phase_start_acceleration_.ptr()),
        start_acceleration, grasp_phase_start_acceleration_.bytes(),
        cudaMemcpyDeviceToDevice, stream));
  }
  const auto workspace = grasp_phase_workspace();
  auto pose_request = make_pose_request(
      reinterpret_cast<const float*>(grasp_phase_goal_position_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_goal_quaternion_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_start_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_start_velocity_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_start_acceleration_.ptr()),
      workspace, stream);
  pose_request.constrain_tool_path = true;
  pose_request.free_position_direction = direction;
  pose_request.direction_in_goal_frame = direction_in_goal_frame;
  pose_request.path_position_tolerance =
      config_.inverse_kinematics.position_tolerance;
  pose_request.path_orientation_tolerance =
      config_.inverse_kinematics.orientation_tolerance;
  pose_request.finetune_attempts = finetune_attempts;
  pose_request.finetune_dt_scale = config_.trajectory_optimization.finetune_dt_scale;
  plan_pose_impl(pose_request, graph_cache, known_goal_configuration);
  copy_grasp_phase_output(output, stream);
  uint8_t success = 0U;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      &success, reinterpret_cast<const void*>(selection_success_.ptr()),
      sizeof(success), cudaMemcpyDeviceToHost, stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return success != 0U;
}

void MotionPlannerRuntime::validate_config() const {
  if (config_.trajectory_optimization.device_ordinal !=
          config_.rrt.device_ordinal ||
      config_.trajectory_optimization.device_ordinal !=
          config_.inverse_kinematics.device_ordinal) {
    throw std::invalid_argument(
        "motion-planning IK, TrajOpt, and RRT must use the same CUDA device");
  }
  if (config_.trajectory_optimization.num_seeds <= 0 ||
      config_.trajectory_optimization.num_seeds >
          manifest_.max_trajopt_seeds) {
    throw std::invalid_argument(
        "motion-planning TrajOpt seed count exceeds the artifact capacity");
  }
  if (config_.rrt.max_plans < config_.trajectory_optimization.num_seeds) {
    throw std::invalid_argument(
        "motion-planning RRT max_plans must cover every TrajOpt seed");
  }
  if (config_.inverse_kinematics.num_seeds <
      config_.trajectory_optimization.num_seeds) {
    throw std::invalid_argument(
        "motion-planning IK must produce every configured TrajOpt seed");
  }
  if (rrt_ == nullptr || trajectory_optimizer_ == nullptr ||
      seed_fitter_ == nullptr || inverse_kinematics_ == nullptr) {
    return;
  }
  const auto rrt_info = rrt_->info();
  const auto trajopt_info = trajectory_optimizer_->info();
  const auto seed_fitter_info = seed_fitter_->info();
  const auto ik_info = inverse_kinematics_->info();
  const auto rrt_manifest =
      curobo_robot_runtime::load_rrt_manifest(artifact_directory_);
  const auto trajopt_manifest =
      curobo_robot_runtime::load_trajectory_optimization_manifest(
          artifact_directory_);
  if (manifest_.arch != rrt_manifest.arch ||
      manifest_.arch != trajopt_manifest.arch ||
      rrt_info.num_dof != manifest_.num_dof ||
      rrt_info.max_path_nodes != manifest_.max_rrt_path_nodes ||
      trajopt_info.num_dof != manifest_.num_dof ||
      trajopt_info.num_knots != manifest_.num_knots ||
      trajopt_info.trajectory_steps != manifest_.trajectory_steps ||
      seed_fitter_info.num_dof != manifest_.num_dof ||
      seed_fitter_info.num_knots != manifest_.num_knots ||
      seed_fitter_info.trajectory_steps != manifest_.trajectory_steps ||
      ik_info.num_dof != manifest_.num_dof ||
      ik_info.num_tool_frames != trajopt_manifest.num_tool_frames) {
    throw std::runtime_error(
        "motion-planning artifact does not match its RRT and TrajOpt artifacts");
  }
}

void MotionPlannerRuntime::load_fit_artifact() {
  const auto load = [&](const curobo_robot_runtime::TensorManifest& tensor,
                        const char* name,
                        curobo_robot_runtime::DeviceBuffer* destination) {
    destination->upload(curobo_robot_runtime::load_tensor_data(
        artifact_directory_, tensor, name));
  };
  load(manifest_.arc_progress, "motion arc_progress", &arc_progress_);
  resample_path_kernel_.load(
      (std::filesystem::path(artifact_directory_) /
       manifest_.resample_path.cubin)
          .string(),
      manifest_.resample_path.symbol);
  prepare_seed_paths_kernel_.load(
      (std::filesystem::path(artifact_directory_) /
       manifest_.prepare_seed_paths.cubin)
          .string(),
      manifest_.prepare_seed_paths.symbol);
}

void MotionPlannerRuntime::allocate_workspace() {
  const auto dof = static_cast<size_t>(manifest_.num_dof);
  const auto knots = static_cast<size_t>(manifest_.num_knots);
  const auto steps = static_cast<size_t>(manifest_.trajectory_steps);
  const auto path_nodes = static_cast<size_t>(manifest_.max_rrt_path_nodes);
  const auto seeds =
      static_cast<size_t>(config_.trajectory_optimization.num_seeds);
  const auto tool_frames =
      static_cast<size_t>(trajectory_optimizer_->info().num_tool_frames);
  rrt_path_.allocate(
      checked_product(checked_product(seeds, path_nodes, "motion RRT paths"),
                      dof * sizeof(float), "motion RRT path values"));
  rrt_path_count_.allocate(seeds * sizeof(int32_t));
  rrt_success_.allocate(seeds * sizeof(uint8_t));
  rrt_iterations_.allocate(seeds * sizeof(int32_t));
  dense_path_.allocate(
      checked_product(checked_product(seeds, steps, "motion dense paths"),
                      dof * sizeof(float), "motion dense path values"));
  fitted_seed_knots_.allocate(checked_product(
      checked_product(seeds, knots, "motion fitted seeds"),
      dof * sizeof(float), "motion fitted seed values"));
  fitted_seed_starts_.allocate(
      checked_product(seeds, dof * sizeof(float), "motion fitted starts"));
  fitted_seed_start_velocities_.allocate(checked_product(
      seeds, dof * sizeof(float), "motion fitted start velocities"));
  fitted_seed_start_accelerations_.allocate(checked_product(
      seeds, dof * sizeof(float), "motion fitted start accelerations"));
  fitted_seed_goals_.allocate(
      checked_product(seeds, dof * sizeof(float), "motion fitted goals"));
  fitted_seed_required_dt_.allocate(seeds * sizeof(float));
  trajopt_selected_seed_.allocate(sizeof(int32_t));
  ik_solutions_.allocate(checked_product(
      seeds, dof * sizeof(float), "motion pose IK solutions"));
  ik_success_.allocate(seeds * sizeof(uint8_t));
  ik_position_error_.allocate(seeds * sizeof(float));
  ik_orientation_error_.allocate(seeds * sizeof(float));
  ik_collision_cost_.allocate(seeds * sizeof(float));
  ik_goalset_index_.allocate(seeds * tool_frames * sizeof(int32_t));
  pose_selected_seed_.allocate(sizeof(int32_t));
  staged_start_position_.allocate(dof * sizeof(float));
  staged_start_velocity_.allocate(dof * sizeof(float));
  staged_start_acceleration_.allocate(dof * sizeof(float));
  staged_goal_position_.allocate(dof * sizeof(float));
  staged_seed_start_positions_.allocate(seeds * dof * sizeof(float));
  staged_seed_goal_positions_.allocate(seeds * dof * sizeof(float));
  staged_seed_valid_.allocate(seeds * sizeof(uint8_t));
  staged_tool_goal_positions_.allocate(tool_frames * 3 * sizeof(float));
  staged_tool_goal_quaternions_.allocate(tool_frames * 4 * sizeof(float));
  staged_ik_seed_positions_.allocate(
      static_cast<size_t>(config_.inverse_kinematics.num_seeds) * dof *
      sizeof(float));
  staged_selected_goal_position_.allocate(dof * sizeof(float));
  staged_position_error_.allocate(sizeof(float));
  staged_orientation_error_.allocate(sizeof(float));
  staged_selected_seed_.allocate(sizeof(int32_t));
  selected_grasp_position_.allocate(tool_frames * 3 * sizeof(float));
  selected_grasp_quaternion_.allocate(tool_frames * 4 * sizeof(float));
  selected_grasp_configuration_.allocate(dof * sizeof(float));
  const auto max_goalset =
      static_cast<size_t>(config_.inverse_kinematics.max_goalset);
  grasp_goalset_position_.allocate(checked_product(
      checked_product(tool_frames, max_goalset, "grasp goalset poses"),
      3 * sizeof(float), "grasp goalset positions"));
  grasp_goalset_quaternion_.allocate(checked_product(
      checked_product(tool_frames, max_goalset, "grasp goalset poses"),
      4 * sizeof(float), "grasp goalset quaternions"));
  grasp_phase_start_.allocate(dof * sizeof(float));
  grasp_phase_start_velocity_.allocate(dof * sizeof(float));
  grasp_phase_start_acceleration_.allocate(dof * sizeof(float));
  grasp_phase_goal_position_.allocate(tool_frames * 3 * sizeof(float));
  grasp_phase_goal_quaternion_.allocate(tool_frames * 4 * sizeof(float));
  pose_joint_goal_.allocate(dof * sizeof(float));
  approach_goal_position_.allocate(tool_frames * 3 * sizeof(float));
  approach_goal_quaternion_.allocate(tool_frames * 4 * sizeof(float));
  lift_goal_position_.allocate(tool_frames * 3 * sizeof(float));
  lift_goal_quaternion_.allocate(tool_frames * 4 * sizeof(float));
  selection_knots_.allocate(knots * dof * sizeof(float));
  const auto trajectory_values = steps * dof;
  selection_position_.allocate(trajectory_values * sizeof(float));
  selection_velocity_.allocate(trajectory_values * sizeof(float));
  selection_acceleration_.allocate(trajectory_values * sizeof(float));
  selection_jerk_.allocate(trajectory_values * sizeof(float));
  selection_success_.allocate(sizeof(uint8_t));
  selection_cost_.allocate(sizeof(float));
  selection_collision_cost_.allocate(sizeof(float));
  selection_dt_.allocate(sizeof(float));
  selection_used_rrt_.allocate(sizeof(uint8_t));
  selection_rrt_iterations_.allocate(sizeof(int32_t));
  selection_planning_attempts_.allocate(sizeof(int32_t));
}

void MotionPlannerRuntime::fit_rrt_path(
    const MotionPlanningRequest& request, CUstream stream) const {
  const int dof = manifest_.num_dof;
  const int path_nodes = manifest_.max_rrt_path_nodes;
  const int steps = manifest_.trajectory_steps;
  const int seeds = config_.trajectory_optimization.num_seeds;

  const auto resample_bounds = warp_bounds_1d(seeds * steps);
  auto path =
      array_2d(rrt_path_.ptr(), seeds * path_nodes, dof, sizeof(float));
  auto path_count =
      array_1d(rrt_path_count_.ptr(), seeds, sizeof(int32_t));
  auto arc_progress = array_1d(arc_progress_.ptr(), steps, sizeof(float));
  auto dense_path =
      array_2d(dense_path_.ptr(), seeds * steps, dof, sizeof(float));
  std::vector<void*> resample_arguments{
      const_cast<WarpLaunchBounds1*>(&resample_bounds),
      &path,
      &path_count,
      &arc_progress,
      &dense_path,
  };
  resample_path_kernel_.launch(
      grid_for(static_cast<size_t>(seeds) * steps,
               manifest_.resample_path.block_dim),
      static_cast<unsigned int>(manifest_.resample_path.block_dim),
      static_cast<unsigned int>(manifest_.resample_path.shared_memory), stream,
      resample_arguments);

  fit_seed_knots(request, false, stream);
}

void MotionPlannerRuntime::fit_seed_knots(
    const MotionPlanningRequest& request, bool use_linear_path,
    CUstream stream) const {
  const int dof = manifest_.num_dof;
  const int steps = manifest_.trajectory_steps;
  const int seeds = config_.trajectory_optimization.num_seeds;
  const int path_variables = steps * dof;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (use_linear_path) {
    CRR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(rrt_success_.ptr()), 1,
        static_cast<size_t>(seeds) * sizeof(uint8_t), cuda_stream));
  }

  const auto fit_bounds = warp_bounds_2d(seeds, path_variables);
  const bool use_seed_starts = request.seed_start_positions != nullptr;
  auto start = warp_array_1d(
      reinterpret_cast<uint64_t>(request.start_position), dof, sizeof(float));
  auto start_velocity = warp_array_1d(
      reinterpret_cast<uint64_t>(request.start_velocity), dof, sizeof(float));
  auto start_acceleration = warp_array_1d(
      reinterpret_cast<uint64_t>(request.start_acceleration), dof,
      sizeof(float));
  const auto seed_start_pointer =
      use_seed_starts ? request.seed_start_positions : request.start_position;
  auto seed_start = warp_array_2d(
      reinterpret_cast<uint64_t>(seed_start_pointer),
      use_seed_starts ? seeds : 1, dof, sizeof(float));
  const bool use_seed_goals = request.seed_goal_positions != nullptr;
  auto goal = warp_array_1d(
      reinterpret_cast<uint64_t>(request.goal_position), dof, sizeof(float));
  const auto seed_goal_pointer =
      use_seed_goals ? request.seed_goal_positions : request.goal_position;
  auto seed_goal = warp_array_2d(
      reinterpret_cast<uint64_t>(seed_goal_pointer), use_seed_goals ? seeds : 1,
      dof, sizeof(float));
  const auto seed_valid_pointer =
      request.seed_valid == nullptr
          ? reinterpret_cast<const uint8_t*>(rrt_success_.ptr())
          : request.seed_valid;
  auto seed_valid = warp_array_1d(
      reinterpret_cast<uint64_t>(seed_valid_pointer), seeds, sizeof(uint8_t));
  auto rrt_success =
      array_1d(rrt_success_.ptr(), seeds, sizeof(uint8_t));
  auto dense_path =
      array_3d(dense_path_.ptr(), seeds, steps, dof, sizeof(float));
  auto resolved_starts =
      array_2d(fitted_seed_starts_.ptr(), seeds, dof, sizeof(float));
  auto resolved_start_velocities = array_2d(
      fitted_seed_start_velocities_.ptr(), seeds, dof, sizeof(float));
  auto resolved_start_accelerations = array_2d(
      fitted_seed_start_accelerations_.ptr(), seeds, dof, sizeof(float));
  auto resolved_goals =
      array_2d(fitted_seed_goals_.ptr(), seeds, dof, sizeof(float));
  uint8_t linear_path = use_linear_path ? 1U : 0U;
  uint8_t seed_starts = use_seed_starts ? 1U : 0U;
  uint8_t seed_goals = use_seed_goals ? 1U : 0U;
  uint8_t use_seed_valid = request.seed_valid == nullptr ? 0U : 1U;
  std::vector<void*> fit_arguments{
      const_cast<WarpLaunchBounds2*>(&fit_bounds),
      &dense_path,
      &start,
      &start_velocity,
      &start_acceleration,
      &goal,
      &seed_start,
      &seed_goal,
      &seed_valid,
      &rrt_success,
      &resolved_starts,
      &resolved_start_velocities,
      &resolved_start_accelerations,
      &resolved_goals,
      &linear_path,
      &seed_starts,
      &seed_goals,
      &use_seed_valid,
  };
  prepare_seed_paths_kernel_.launch(
      grid_for(static_cast<size_t>(seeds) * path_variables,
               manifest_.prepare_seed_paths.block_dim),
      static_cast<unsigned int>(manifest_.prepare_seed_paths.block_dim),
      static_cast<unsigned int>(manifest_.prepare_seed_paths.shared_memory),
      stream, fit_arguments);

  TrajectorySeedFitterRequest fitter_request;
  fitter_request.target_position =
      reinterpret_cast<const float*>(dense_path_.ptr());
  fitter_request.start_position =
      reinterpret_cast<const float*>(fitted_seed_starts_.ptr());
  fitter_request.start_velocity =
      reinterpret_cast<const float*>(fitted_seed_start_velocities_.ptr());
  fitter_request.start_acceleration =
      reinterpret_cast<const float*>(fitted_seed_start_accelerations_.ptr());
  fitter_request.goal_position =
      reinterpret_cast<const float*>(fitted_seed_goals_.ptr());
  fitter_request.fitted_knots =
      reinterpret_cast<float*>(fitted_seed_knots_.ptr());
  fitter_request.required_dt =
      reinterpret_cast<float*>(fitted_seed_required_dt_.ptr());
  fitter_request.stream = reinterpret_cast<cudaStream_t>(stream);
  seed_fitter_->fit(fitter_request);
}

void MotionPlannerRuntime::set_rrt_outputs(
    const MotionPlanningRequest& request, bool rrt_stage,
    CUstream stream) const {
  CRR_CUDA_CHECK(launch_motion_result_metadata(
      reinterpret_cast<const int32_t*>(trajopt_selected_seed_.ptr()),
      reinterpret_cast<const uint8_t*>(rrt_success_.ptr()),
      reinterpret_cast<const int32_t*>(rrt_iterations_.ptr()),
      config_.trajectory_optimization.num_seeds, rrt_stage,
      request.selected_seed, request.used_rrt, request.rrt_iterations,
      reinterpret_cast<cudaStream_t>(stream)));
}

void MotionPlannerRuntime::finish_rrt_failure(
    const MotionPlanningRequest& request, CUstream stream) const {
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const float infinity = std::numeric_limits<float>::infinity();
  CRR_CUDA_CHECK(
      cudaMemsetAsync(request.success, 0, sizeof(uint8_t), cuda_stream));
  CRR_CUDA_CHECK(
      cudaMemsetAsync(request.used_rrt, 0, sizeof(uint8_t), cuda_stream));
  CRR_CUDA_CHECK(cudaMemsetAsync(request.rrt_iterations, 0, sizeof(int32_t),
                                 cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(request.cost, &infinity, sizeof(infinity),
                                 cudaMemcpyHostToDevice, cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(request.collision_cost, &infinity,
                                 sizeof(infinity), cudaMemcpyHostToDevice,
                                 cuda_stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(request.trajectory_dt, &infinity,
                                 sizeof(infinity), cudaMemcpyHostToDevice,
                                 cuda_stream));
  const int32_t no_seed = -1;
  if (request.selected_seed != nullptr) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(request.selected_seed, &no_seed,
                                   sizeof(no_seed), cudaMemcpyHostToDevice,
                                   cuda_stream));
  }
  CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void MotionPlannerRuntime::finish_pose_failure(
    const PoseMotionPlanningRequest& request, cudaStream_t stream,
    int attempts) const {
  const float infinity = std::numeric_limits<float>::infinity();
  const int32_t no_seed = -1;
  const int32_t attempt_count = attempts;
  CRR_CUDA_CHECK(cudaMemsetAsync(request.success, 0, sizeof(uint8_t), stream));
  CRR_CUDA_CHECK(
      cudaMemsetAsync(request.used_rrt, 0, sizeof(uint8_t), stream));
  CRR_CUDA_CHECK(
      cudaMemsetAsync(request.rrt_iterations, 0, sizeof(int32_t), stream));
  for (float* output : {request.cost, request.collision_cost,
                        request.trajectory_dt, request.position_error,
                        request.orientation_error}) {
    if (output != nullptr) {
      CRR_CUDA_CHECK(cudaMemcpyAsync(output, &infinity, sizeof(infinity),
                                     cudaMemcpyHostToDevice, stream));
    }
  }
  if (request.selected_seed != nullptr) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(request.selected_seed, &no_seed,
                                   sizeof(no_seed), cudaMemcpyHostToDevice,
                                   stream));
  }
  if (request.planning_attempts != nullptr) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(request.planning_attempts, &attempt_count,
                                   sizeof(attempt_count),
                                   cudaMemcpyHostToDevice, stream));
  }
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void MotionPlannerRuntime::copy_pose_outputs(
    const PoseMotionPlanningRequest& request, int attempts,
    cudaStream_t stream) const {
  CRR_CUDA_CHECK(launch_pose_result_metadata(
      reinterpret_cast<const int32_t*>(pose_selected_seed_.ptr()),
      reinterpret_cast<const float*>(ik_solutions_.ptr()),
      reinterpret_cast<const float*>(ik_position_error_.ptr()),
      reinterpret_cast<const float*>(ik_orientation_error_.ptr()),
      config_.trajectory_optimization.num_seeds, manifest_.num_dof, attempts,
      request.selected_goal_position, request.position_error,
      request.orientation_error, request.selected_seed,
      request.planning_attempts, request.success, stream));
}

bool MotionPlannerRuntime::same_graph_request(
    const TrajectoryOptimizationRequest& left,
    const TrajectoryOptimizationRequest& right) {
  return left.start_position == right.start_position &&
         left.start_velocity == right.start_velocity &&
         left.start_acceleration == right.start_acceleration &&
         left.goal_position == right.goal_position &&
         left.seed_knots == right.seed_knots &&
         left.seed_goal_positions == right.seed_goal_positions &&
         left.seed_start_positions == right.seed_start_positions &&
         left.seed_valid == right.seed_valid &&
         left.optimized_knots == right.optimized_knots &&
         left.trajectory_position == right.trajectory_position &&
         left.trajectory_velocity == right.trajectory_velocity &&
         left.trajectory_acceleration == right.trajectory_acceleration &&
         left.trajectory_jerk == right.trajectory_jerk &&
         left.success == right.success && left.cost == right.cost &&
         left.collision_cost == right.collision_cost &&
         left.trajectory_dt == right.trajectory_dt &&
         left.selected_seed == right.selected_seed &&
         left.finetune_dt_scale == right.finetune_dt_scale &&
         left.goal_tool_positions == right.goal_tool_positions &&
         left.goal_tool_quaternions == right.goal_tool_quaternions &&
         left.constrain_tool_path == right.constrain_tool_path &&
         left.free_position_direction == right.free_position_direction &&
         left.direction_in_goal_frame == right.direction_in_goal_frame &&
         left.path_position_tolerance == right.path_position_tolerance &&
         left.path_orientation_tolerance == right.path_orientation_tolerance;
}

void MotionPlannerRuntime::destroy_graph(TrajoptGraph* graph) noexcept {
  for (auto* stage : {
           &graph->setup,
           &graph->pass,
           &graph->refinement_pass,
           &graph->base_accumulate,
           &graph->finetune_prepare,
           &graph->finetune_accumulate,
           &graph->finalize,
       }) {
    if (stage->executable != nullptr) {
      cudaGraphExecDestroy(stage->executable);
    }
    if (stage->graph != nullptr) {
      cudaGraphDestroy(stage->graph);
    }
  }
  *graph = {};
}

void MotionPlannerRuntime::destroy_graph_cache(
    TrajoptGraphCache* graph_cache) noexcept {
  destroy_graph(&graph_cache->direct);
  destroy_graph(&graph_cache->rrt);
}

void MotionPlannerRuntime::optimize_trajectory(
    const TrajectoryOptimizationRequest& request, TrajoptGraph* graph) const {
  trajectory_optimizer_->validate_request(request);
  const int finetune_attempts =
      trajectory_optimizer_->resolved_finetune_attempts(request);
  const auto run_stages = [&](const auto& launch) {
    launch(graph->setup);
    launch(graph->pass);
    launch(graph->base_accumulate);
    if (finetune_attempts > 0 &&
        trajectory_optimizer_->base_pass_succeeded(request.stream)) {
      for (int attempt = 0; attempt < finetune_attempts; ++attempt) {
        launch(graph->finetune_prepare);
        launch(graph->refinement_pass);
        launch(graph->finetune_accumulate);
      }
    }
    launch(graph->finalize);
  };
  if (!config_.use_cuda_graph) {
    trajectory_optimizer_->begin_optimization(request);
    trajectory_optimizer_->run_optimization_pass(request);
    trajectory_optimizer_->accumulate_optimization_pass(true, request);
    if (finetune_attempts > 0 &&
        trajectory_optimizer_->base_pass_succeeded(request.stream)) {
      for (int attempt = 0; attempt < finetune_attempts; ++attempt) {
        trajectory_optimizer_->prepare_finetune_pass(request);
        trajectory_optimizer_->run_optimization_pass(request, true);
        trajectory_optimizer_->accumulate_optimization_pass(false, request);
      }
    }
    trajectory_optimizer_->finish_optimization(request);
    return;
  }
  if (graph->initialized &&
      graph->has_finetune_stages == (finetune_attempts > 0) &&
      same_graph_request(graph->request, request)) {
    run_stages([&](const CudaGraphStage& stage) {
      CRR_CUDA_CHECK(cudaGraphLaunch(stage.executable, request.stream));
    });
    return;
  }

  destroy_graph(graph);
  CRR_CUDA_CHECK(cudaStreamSynchronize(request.stream));
  try {
    const auto capture_stage = [&](CudaGraphStage* stage,
                                   const auto& enqueue) {
      CRR_CUDA_CHECK(cudaStreamBeginCapture(
          request.stream, cudaStreamCaptureModeThreadLocal));
      bool capture_active = true;
      try {
        enqueue();
        CRR_CUDA_CHECK(cudaStreamEndCapture(request.stream, &stage->graph));
        capture_active = false;
        CRR_CUDA_CHECK(
            cudaGraphInstantiate(&stage->executable, stage->graph, 0));
      } catch (...) {
        if (capture_active) {
          cudaGraph_t abandoned_graph = nullptr;
          cudaStreamEndCapture(request.stream, &abandoned_graph);
          if (abandoned_graph != nullptr) {
            cudaGraphDestroy(abandoned_graph);
          }
        }
        throw;
      }
    };
    capture_stage(&graph->setup, [&]() {
      trajectory_optimizer_->begin_optimization(request);
    });
    capture_stage(&graph->pass, [&]() {
      trajectory_optimizer_->run_optimization_pass(request);
    });
    capture_stage(&graph->base_accumulate, [&]() {
      trajectory_optimizer_->accumulate_optimization_pass(true, request);
    });
    if (finetune_attempts > 0) {
      capture_stage(&graph->refinement_pass, [&]() {
        trajectory_optimizer_->run_optimization_pass(request, true);
      });
      capture_stage(&graph->finetune_prepare, [&]() {
        trajectory_optimizer_->prepare_finetune_pass(request);
      });
      capture_stage(&graph->finetune_accumulate, [&]() {
        trajectory_optimizer_->accumulate_optimization_pass(false, request);
      });
    }
    capture_stage(&graph->finalize, [&]() {
      trajectory_optimizer_->finish_optimization(request);
    });
  } catch (...) {
    destroy_graph(graph);
    throw;
  }
  graph->request = request;
  graph->has_finetune_stages = finetune_attempts > 0;
  graph->initialized = true;
  run_stages([&](const CudaGraphStage& stage) {
    CRR_CUDA_CHECK(cudaGraphLaunch(stage.executable, request.stream));
  });
}

void MotionPlannerRuntime::plan(const MotionPlanningRequest& request) const {
  last_ik_time_seconds_ = 0.0;
  last_rrt_time_seconds_ = 0.0;
  last_rrt_calls_ = 0;
  rrt_->reset_seed();
  if (request.start_position == nullptr || request.goal_position == nullptr ||
      request.optimized_knots == nullptr ||
      request.trajectory_position == nullptr ||
      request.trajectory_velocity == nullptr ||
      request.trajectory_acceleration == nullptr ||
      request.trajectory_jerk == nullptr || request.success == nullptr ||
      request.cost == nullptr || request.collision_cost == nullptr ||
      request.used_rrt == nullptr || request.rrt_iterations == nullptr ||
      request.trajectory_dt == nullptr) {
    throw std::invalid_argument(
        "motion-planning input and output CUDA addresses must be non-null");
  }
  CRR_CUDA_CHECK(
      cudaSetDevice(config_.trajectory_optimization.device_ordinal));
  cudaStream_t stream = request.stream;
  if (stream == nullptr && config_.use_cuda_graph) {
    stream = internal_stream_;
  }
  const auto dof = static_cast<size_t>(manifest_.num_dof);
  const auto seeds =
      static_cast<size_t>(config_.trajectory_optimization.num_seeds);
  const auto tool_frames =
      static_cast<size_t>(trajectory_optimizer_->info().num_tool_frames);
  const auto copy_to_staging = [&](curobo_robot_runtime::DeviceBuffer* output,
                                   const void* input, size_t bytes) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(output->ptr()), input, bytes,
        cudaMemcpyDeviceToDevice, stream));
  };
  copy_to_staging(&staged_start_position_, request.start_position,
                  dof * sizeof(float));
  if (request.start_velocity == nullptr) {
    staged_start_velocity_.clear(stream);
  } else {
    copy_to_staging(&staged_start_velocity_, request.start_velocity,
                    dof * sizeof(float));
  }
  if (request.start_acceleration == nullptr) {
    staged_start_acceleration_.clear(stream);
  } else {
    copy_to_staging(&staged_start_acceleration_, request.start_acceleration,
                    dof * sizeof(float));
  }
  copy_to_staging(&staged_goal_position_, request.goal_position,
                  dof * sizeof(float));

  auto stable_request = request;
  stable_request.start_position =
      reinterpret_cast<const float*>(staged_start_position_.ptr());
  stable_request.start_velocity =
      reinterpret_cast<const float*>(staged_start_velocity_.ptr());
  stable_request.start_acceleration =
      reinterpret_cast<const float*>(staged_start_acceleration_.ptr());
  stable_request.goal_position =
      reinterpret_cast<const float*>(staged_goal_position_.ptr());
  if (request.seed_start_positions != nullptr) {
    copy_to_staging(&staged_seed_start_positions_,
                    request.seed_start_positions, seeds * dof * sizeof(float));
    stable_request.seed_start_positions = reinterpret_cast<const float*>(
        staged_seed_start_positions_.ptr());
  }
  if (request.seed_goal_positions != nullptr) {
    copy_to_staging(&staged_seed_goal_positions_, request.seed_goal_positions,
                    seeds * dof * sizeof(float));
    stable_request.seed_goal_positions = reinterpret_cast<const float*>(
        staged_seed_goal_positions_.ptr());
  }
  if (request.seed_valid != nullptr) {
    copy_to_staging(&staged_seed_valid_, request.seed_valid,
                    seeds * sizeof(uint8_t));
    stable_request.seed_valid =
        reinterpret_cast<const uint8_t*>(staged_seed_valid_.ptr());
  }
  if (request.goal_tool_positions != nullptr) {
    copy_to_staging(&staged_tool_goal_positions_,
                    request.goal_tool_positions,
                    tool_frames * 3 * sizeof(float));
    stable_request.goal_tool_positions = reinterpret_cast<const float*>(
        staged_tool_goal_positions_.ptr());
  }
  if (request.goal_tool_quaternions != nullptr) {
    copy_to_staging(&staged_tool_goal_quaternions_,
                    request.goal_tool_quaternions,
                    tool_frames * 4 * sizeof(float));
    stable_request.goal_tool_quaternions = reinterpret_cast<const float*>(
        staged_tool_goal_quaternions_.ptr());
  }
  stable_request.optimized_knots =
      reinterpret_cast<float*>(selection_knots_.ptr());
  stable_request.trajectory_position =
      reinterpret_cast<float*>(selection_position_.ptr());
  stable_request.trajectory_velocity =
      reinterpret_cast<float*>(selection_velocity_.ptr());
  stable_request.trajectory_acceleration =
      reinterpret_cast<float*>(selection_acceleration_.ptr());
  stable_request.trajectory_jerk =
      reinterpret_cast<float*>(selection_jerk_.ptr());
  stable_request.success = reinterpret_cast<uint8_t*>(selection_success_.ptr());
  stable_request.cost = reinterpret_cast<float*>(selection_cost_.ptr());
  stable_request.collision_cost =
      reinterpret_cast<float*>(selection_collision_cost_.ptr());
  stable_request.trajectory_dt = reinterpret_cast<float*>(selection_dt_.ptr());
  stable_request.used_rrt =
      reinterpret_cast<uint8_t*>(selection_used_rrt_.ptr());
  stable_request.rrt_iterations =
      reinterpret_cast<int32_t*>(selection_rrt_iterations_.ptr());
  stable_request.selected_seed =
      reinterpret_cast<int32_t*>(staged_selected_seed_.ptr());
  stable_request.stream = stream;
  plan_impl(stable_request, false, &default_trajopt_graphs_);

  const auto trajectory_values =
      static_cast<size_t>(manifest_.trajectory_steps) * dof;
  const auto knot_values = static_cast<size_t>(manifest_.num_knots) * dof;
  const auto copy_from_staging = [&](void* output,
                                     const curobo_robot_runtime::DeviceBuffer& input,
                                     size_t bytes) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        output, reinterpret_cast<const void*>(input.ptr()), bytes,
        cudaMemcpyDeviceToDevice, stream));
  };
  copy_from_staging(request.optimized_knots, selection_knots_,
                    knot_values * sizeof(float));
  copy_from_staging(request.trajectory_position, selection_position_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.trajectory_velocity, selection_velocity_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.trajectory_acceleration, selection_acceleration_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.trajectory_jerk, selection_jerk_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.success, selection_success_, sizeof(uint8_t));
  copy_from_staging(request.cost, selection_cost_, sizeof(float));
  copy_from_staging(request.collision_cost, selection_collision_cost_,
                    sizeof(float));
  copy_from_staging(request.trajectory_dt, selection_dt_, sizeof(float));
  copy_from_staging(request.used_rrt, selection_used_rrt_, sizeof(uint8_t));
  copy_from_staging(request.rrt_iterations, selection_rrt_iterations_,
                    sizeof(int32_t));
  if (request.selected_seed != nullptr) {
    copy_from_staging(request.selected_seed, staged_selected_seed_,
                      sizeof(int32_t));
  }
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool MotionPlannerRuntime::plan_impl(
    const MotionPlanningRequest& request,
    bool activate_repaired_seeds,
    TrajoptGraphCache* graph_cache) const {
  CRR_CUDA_CHECK(
      cudaSetDevice(config_.trajectory_optimization.device_ordinal));
  if (request.start_position == nullptr || request.goal_position == nullptr ||
      request.optimized_knots == nullptr ||
      request.trajectory_position == nullptr ||
      request.trajectory_velocity == nullptr ||
      request.trajectory_acceleration == nullptr ||
      request.trajectory_jerk == nullptr || request.success == nullptr ||
      request.cost == nullptr || request.collision_cost == nullptr ||
      request.used_rrt == nullptr || request.rrt_iterations == nullptr ||
      request.trajectory_dt == nullptr) {
    throw std::invalid_argument(
        "motion-planning input and output CUDA addresses must be non-null");
  }
  auto active_request = request;
  if (active_request.stream == nullptr && config_.use_cuda_graph) {
    active_request.stream = internal_stream_;
  }
  const auto stream = reinterpret_cast<CUstream>(active_request.stream);
  const auto cuda_stream = active_request.stream;

  const auto optimize = [&](TrajoptGraph* graph) {
    TrajectoryOptimizationRequest trajopt_request;
    trajopt_request.start_position = active_request.start_position;
    trajopt_request.start_velocity = active_request.start_velocity;
    trajopt_request.start_acceleration = active_request.start_acceleration;
    trajopt_request.goal_position = active_request.goal_position;
    trajopt_request.seed_knots =
        reinterpret_cast<const float*>(fitted_seed_knots_.ptr());
    trajopt_request.seed_goal_positions =
        reinterpret_cast<const float*>(fitted_seed_goals_.ptr());
    trajopt_request.optimized_knots = active_request.optimized_knots;
    trajopt_request.trajectory_position = active_request.trajectory_position;
    trajopt_request.trajectory_velocity = active_request.trajectory_velocity;
    trajopt_request.trajectory_acceleration =
        active_request.trajectory_acceleration;
    trajopt_request.trajectory_jerk = active_request.trajectory_jerk;
    trajopt_request.success = active_request.success;
    trajopt_request.cost = active_request.cost;
    trajopt_request.collision_cost = active_request.collision_cost;
    trajopt_request.trajectory_dt = active_request.trajectory_dt;
    trajopt_request.selected_seed =
        reinterpret_cast<int32_t*>(trajopt_selected_seed_.ptr());
    trajopt_request.stream = active_request.stream;
    trajopt_request.seed_start_positions =
        reinterpret_cast<const float*>(fitted_seed_starts_.ptr());
    trajopt_request.seed_valid =
        activate_repaired_seeds ? nullptr : active_request.seed_valid;
    trajopt_request.finetune_attempts = active_request.finetune_attempts;
    trajopt_request.finetune_dt_scale = active_request.finetune_dt_scale;
    trajopt_request.constrain_tool_path = active_request.constrain_tool_path;
    if (active_request.constrain_tool_path) {
      trajopt_request.goal_tool_positions = active_request.goal_tool_positions;
      trajopt_request.goal_tool_quaternions =
          active_request.goal_tool_quaternions;
      trajopt_request.free_position_direction =
          active_request.free_position_direction;
      trajopt_request.direction_in_goal_frame =
          active_request.direction_in_goal_frame;
      trajopt_request.path_position_tolerance =
          active_request.path_position_tolerance;
      trajopt_request.path_orientation_tolerance =
          active_request.path_orientation_tolerance;
    }
    optimize_trajectory(trajopt_request, graph);
  };

  fit_seed_knots(active_request, true, stream);
  if (config_.attempt_direct_trajectory_optimization &&
      active_request.attempt_direct_trajectory_optimization) {
    optimize(&graph_cache->direct);
    uint8_t direct_success = 0U;
    CRR_CUDA_CHECK(cudaMemcpyAsync(&direct_success, active_request.success,
                                   sizeof(direct_success),
                                   cudaMemcpyDeviceToHost, cuda_stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    if (direct_success != 0U) {
      set_rrt_outputs(active_request, false, stream);
      return true;
    }
  }
  if (!active_request.allow_rrt) {
    finish_rrt_failure(active_request, stream);
    return false;
  }

  RrtRequest rrt_request;
  rrt_request.start_position =
      reinterpret_cast<const float*>(fitted_seed_starts_.ptr());
  rrt_request.goal_position =
      reinterpret_cast<const float*>(fitted_seed_goals_.ptr());
  rrt_request.path = reinterpret_cast<float*>(rrt_path_.ptr());
  rrt_request.path_count = reinterpret_cast<int32_t*>(rrt_path_count_.ptr());
  rrt_request.success = reinterpret_cast<uint8_t*>(rrt_success_.ptr());
  rrt_request.iterations = reinterpret_cast<int32_t*>(rrt_iterations_.ptr());
  rrt_request.stream = active_request.stream;
  rrt_request.num_plans = config_.trajectory_optimization.num_seeds;
  rrt_request.plan_valid = active_request.seed_valid;
  rrt_path_.clear(cuda_stream);
  const auto rrt_start = std::chrono::steady_clock::now();
  rrt_->plan(rrt_request);
  const auto rrt_end = std::chrono::steady_clock::now();
  last_rrt_time_seconds_ +=
      std::chrono::duration<double>(rrt_end - rrt_start).count();
  ++last_rrt_calls_;
  std::vector<uint8_t> rrt_success(
      static_cast<size_t>(config_.trajectory_optimization.num_seeds), 0U);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      rrt_success.data(), reinterpret_cast<void*>(rrt_success_.ptr()),
      rrt_success.size(), cudaMemcpyDeviceToHost, cuda_stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  if (std::none_of(rrt_success.begin(), rrt_success.end(),
                   [](uint8_t value) { return value != 0U; })) {
    finish_rrt_failure(active_request, stream);
    return false;
  }

  fit_rrt_path(active_request, stream);
  optimize(&graph_cache->rrt);
  set_rrt_outputs(active_request, true, stream);
  uint8_t success = 0U;
  CRR_CUDA_CHECK(cudaMemcpyAsync(&success, active_request.success,
                                 sizeof(success), cudaMemcpyDeviceToHost,
                                 cuda_stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  return success != 0U;
}

void MotionPlannerRuntime::plan_pose(
    const PoseMotionPlanningRequest& request) const {
  last_ik_time_seconds_ = 0.0;
  last_rrt_time_seconds_ = 0.0;
  last_rrt_calls_ = 0;
  rrt_->reset_seed();
  if (request.goal_positions == nullptr ||
      request.goal_quaternions == nullptr ||
      request.current_position == nullptr || request.optimized_knots == nullptr ||
      request.trajectory_position == nullptr ||
      request.trajectory_velocity == nullptr ||
      request.trajectory_acceleration == nullptr ||
      request.trajectory_jerk == nullptr || request.success == nullptr ||
      request.cost == nullptr || request.collision_cost == nullptr ||
      request.trajectory_dt == nullptr || request.used_rrt == nullptr ||
      request.rrt_iterations == nullptr) {
    throw std::invalid_argument(
        "pose motion-planning input and output CUDA addresses must be non-null");
  }
  CRR_CUDA_CHECK(
      cudaSetDevice(config_.trajectory_optimization.device_ordinal));
  cudaStream_t stream = request.stream;
  if (stream == nullptr && config_.use_cuda_graph) {
    stream = internal_stream_;
  }
  const auto dof = static_cast<size_t>(manifest_.num_dof);
  const auto tool_frames =
      static_cast<size_t>(trajectory_optimizer_->info().num_tool_frames);
  const auto copy_to_staging = [&](curobo_robot_runtime::DeviceBuffer* output,
                                   const void* input, size_t bytes) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(output->ptr()), input, bytes,
        cudaMemcpyDeviceToDevice, stream));
  };
  copy_to_staging(&staged_tool_goal_positions_, request.goal_positions,
                  tool_frames * 3 * sizeof(float));
  copy_to_staging(&staged_tool_goal_quaternions_, request.goal_quaternions,
                  tool_frames * 4 * sizeof(float));
  copy_to_staging(&staged_start_position_, request.current_position,
                  dof * sizeof(float));
  if (request.current_velocity == nullptr) {
    staged_start_velocity_.clear(stream);
  } else {
    copy_to_staging(&staged_start_velocity_, request.current_velocity,
                    dof * sizeof(float));
  }
  if (request.current_acceleration == nullptr) {
    staged_start_acceleration_.clear(stream);
  } else {
    copy_to_staging(&staged_start_acceleration_, request.current_acceleration,
                    dof * sizeof(float));
  }

  auto stable_request = request;
  stable_request.goal_positions =
      reinterpret_cast<const float*>(staged_tool_goal_positions_.ptr());
  stable_request.goal_quaternions =
      reinterpret_cast<const float*>(staged_tool_goal_quaternions_.ptr());
  stable_request.current_position =
      reinterpret_cast<const float*>(staged_start_position_.ptr());
  stable_request.current_velocity =
      reinterpret_cast<const float*>(staged_start_velocity_.ptr());
  stable_request.current_acceleration =
      reinterpret_cast<const float*>(staged_start_acceleration_.ptr());
  if (request.ik_seed_positions != nullptr) {
    const auto ik_seed_values =
        static_cast<size_t>(config_.inverse_kinematics.num_seeds) * dof;
    copy_to_staging(&staged_ik_seed_positions_, request.ik_seed_positions,
                    ik_seed_values * sizeof(float));
    stable_request.ik_seed_positions =
        reinterpret_cast<const float*>(staged_ik_seed_positions_.ptr());
  }
  stable_request.selected_goal_position =
      reinterpret_cast<float*>(staged_selected_goal_position_.ptr());
  stable_request.optimized_knots =
      reinterpret_cast<float*>(selection_knots_.ptr());
  stable_request.trajectory_position =
      reinterpret_cast<float*>(selection_position_.ptr());
  stable_request.trajectory_velocity =
      reinterpret_cast<float*>(selection_velocity_.ptr());
  stable_request.trajectory_acceleration =
      reinterpret_cast<float*>(selection_acceleration_.ptr());
  stable_request.trajectory_jerk =
      reinterpret_cast<float*>(selection_jerk_.ptr());
  stable_request.success = reinterpret_cast<uint8_t*>(selection_success_.ptr());
  stable_request.cost = reinterpret_cast<float*>(selection_cost_.ptr());
  stable_request.collision_cost =
      reinterpret_cast<float*>(selection_collision_cost_.ptr());
  stable_request.trajectory_dt = reinterpret_cast<float*>(selection_dt_.ptr());
  stable_request.position_error =
      reinterpret_cast<float*>(staged_position_error_.ptr());
  stable_request.orientation_error =
      reinterpret_cast<float*>(staged_orientation_error_.ptr());
  stable_request.used_rrt =
      reinterpret_cast<uint8_t*>(selection_used_rrt_.ptr());
  stable_request.rrt_iterations =
      reinterpret_cast<int32_t*>(selection_rrt_iterations_.ptr());
  stable_request.selected_seed =
      reinterpret_cast<int32_t*>(staged_selected_seed_.ptr());
  stable_request.planning_attempts =
      reinterpret_cast<int32_t*>(selection_planning_attempts_.ptr());
  stable_request.stream = stream;
  plan_pose_impl(stable_request, &default_trajopt_graphs_);

  const auto trajectory_values =
      static_cast<size_t>(manifest_.trajectory_steps) * dof;
  const auto knot_values = static_cast<size_t>(manifest_.num_knots) * dof;
  const auto copy_from_staging = [&](void* output,
                                     const curobo_robot_runtime::DeviceBuffer& input,
                                     size_t bytes) {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        output, reinterpret_cast<const void*>(input.ptr()), bytes,
        cudaMemcpyDeviceToDevice, stream));
  };
  copy_from_staging(request.optimized_knots, selection_knots_,
                    knot_values * sizeof(float));
  copy_from_staging(request.trajectory_position, selection_position_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.trajectory_velocity, selection_velocity_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.trajectory_acceleration, selection_acceleration_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.trajectory_jerk, selection_jerk_,
                    trajectory_values * sizeof(float));
  copy_from_staging(request.success, selection_success_, sizeof(uint8_t));
  copy_from_staging(request.cost, selection_cost_, sizeof(float));
  copy_from_staging(request.collision_cost, selection_collision_cost_,
                    sizeof(float));
  copy_from_staging(request.trajectory_dt, selection_dt_, sizeof(float));
  copy_from_staging(request.used_rrt, selection_used_rrt_, sizeof(uint8_t));
  copy_from_staging(request.rrt_iterations, selection_rrt_iterations_,
                    sizeof(int32_t));
  if (request.selected_goal_position != nullptr) {
    copy_from_staging(request.selected_goal_position,
                      staged_selected_goal_position_, dof * sizeof(float));
  }
  if (request.position_error != nullptr) {
    copy_from_staging(request.position_error, staged_position_error_,
                      sizeof(float));
  }
  if (request.orientation_error != nullptr) {
    copy_from_staging(request.orientation_error, staged_orientation_error_,
                      sizeof(float));
  }
  if (request.selected_seed != nullptr) {
    copy_from_staging(request.selected_seed, staged_selected_seed_,
                      sizeof(int32_t));
  }
  if (request.planning_attempts != nullptr) {
    copy_from_staging(request.planning_attempts,
                      selection_planning_attempts_, sizeof(int32_t));
  }
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void MotionPlannerRuntime::plan_pose_impl(
    const PoseMotionPlanningRequest& request,
    TrajoptGraphCache* graph_cache,
    const float* known_goal_configuration) const {
  if (request.goal_positions == nullptr ||
      request.goal_quaternions == nullptr ||
      request.current_position == nullptr || request.optimized_knots == nullptr ||
      request.trajectory_position == nullptr ||
      request.trajectory_velocity == nullptr ||
      request.trajectory_acceleration == nullptr ||
      request.trajectory_jerk == nullptr || request.success == nullptr ||
      request.cost == nullptr || request.collision_cost == nullptr ||
      request.trajectory_dt == nullptr || request.used_rrt == nullptr ||
      request.rrt_iterations == nullptr) {
    throw std::invalid_argument(
        "pose motion-planning input and output CUDA addresses must be non-null");
  }
  if (request.max_attempts <= 0 || request.enable_graph_attempt < 0 ||
      request.finetune_attempts < 0 ||
      !std::isfinite(request.finetune_dt_scale) ||
      request.finetune_dt_scale <= 0.0F ||
      request.finetune_dt_scale > 1.0F) {
    throw std::invalid_argument(
        "pose motion-planning attempt or finetuning policy is invalid");
  }
  CRR_CUDA_CHECK(
      cudaSetDevice(config_.trajectory_optimization.device_ordinal));
  cudaStream_t active_stream = request.stream;
  if (active_stream == nullptr && config_.use_cuda_graph) {
    active_stream = internal_stream_;
  }
  const int seeds = config_.trajectory_optimization.num_seeds;
  int completed_attempts = 0;
  int ik_solve_index = 0;
  std::vector<uint8_t> ik_success(static_cast<size_t>(seeds), 0U);
  const size_t dof_bytes =
      static_cast<size_t>(manifest_.num_dof) * sizeof(float);
  size_t first_valid_index = 0;
  bool has_reusable_ik_solution = false;
  for (int attempt = 0; attempt < request.max_attempts; ++attempt) {
    completed_attempts = attempt + 1;
    const bool reuse_ik_solution =
        attempt == request.enable_graph_attempt && has_reusable_ik_solution;
    has_reusable_ik_solution = false;
    if (known_goal_configuration == nullptr && !reuse_ik_solution) {
      IkSolveRequest ik_request;
      ik_request.goal_positions = request.goal_positions;
      ik_request.goal_quaternions = request.goal_quaternions;
      ik_request.seed_positions = request.ik_seed_positions;
      ik_request.current_positions = request.current_position;
      ik_request.solution = reinterpret_cast<float*>(ik_solutions_.ptr());
      ik_request.success = reinterpret_cast<uint8_t*>(ik_success_.ptr());
      ik_request.position_error =
          reinterpret_cast<float*>(ik_position_error_.ptr());
      ik_request.orientation_error =
          reinterpret_cast<float*>(ik_orientation_error_.ptr());
      ik_request.collision_cost =
          reinterpret_cast<float*>(ik_collision_cost_.ptr());
      ik_request.batch_size = 1;
      ik_request.num_goalset = 1;
      ik_request.num_solutions = seeds;
      ik_request.stream = active_stream;
      // Match Python's SeedIK progression by advancing the Halton window only
      // when IK actually runs, not when its result is reused for RRT.
      ik_request.seed_offset =
          request.ik_seed_positions == nullptr
              ? ik_solve_index * (config_.inverse_kinematics.num_seeds - 1)
              : 0;
      const auto ik_begin = std::chrono::steady_clock::now();
      const bool ik_success_is_ready =
          inverse_kinematics_->solve_with_host_success(
              ik_request, ik_success.data(), ik_success.size());
      ++ik_solve_index;

      if (!ik_success_is_ready) {
        CRR_CUDA_CHECK(cudaMemcpyAsync(
            ik_success.data(), reinterpret_cast<const void*>(ik_success_.ptr()),
            ik_success.size(), cudaMemcpyDeviceToHost, active_stream));
        CRR_CUDA_CHECK(cudaStreamSynchronize(active_stream));
      }
      last_ik_time_seconds_ +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - ik_begin).count();
      const auto first_valid =
          std::find_if(ik_success.begin(), ik_success.end(),
                       [](uint8_t value) { return value != 0U; });
      if (first_valid == ik_success.end()) {
        continue;
      }
      first_valid_index =
          static_cast<size_t>(std::distance(ik_success.begin(), first_valid));
      for (size_t seed = 0; seed < ik_success.size(); ++seed) {
        if (ik_success.at(seed) != 0U) {
          continue;
        }
        CRR_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(ik_solutions_.ptr() + seed * dof_bytes),
            reinterpret_cast<const void*>(ik_solutions_.ptr() +
                                          first_valid_index * dof_bytes),
            dof_bytes, cudaMemcpyDeviceToDevice, active_stream));
        for (auto* metric : {&ik_position_error_, &ik_orientation_error_,
                             &ik_collision_cost_}) {
          CRR_CUDA_CHECK(cudaMemcpyAsync(
              reinterpret_cast<void*>(metric->ptr() + seed * sizeof(float)),
              reinterpret_cast<const void*>(metric->ptr() +
                                            first_valid_index * sizeof(float)),
              sizeof(float), cudaMemcpyDeviceToDevice, active_stream));
        }
      }
      has_reusable_ik_solution =
          attempt + 1 == request.enable_graph_attempt;
    }
    if (known_goal_configuration == nullptr) {
      CRR_CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<void*>(pose_joint_goal_.ptr()),
          reinterpret_cast<const void*>(ik_solutions_.ptr() +
                                        first_valid_index * dof_bytes),
          dof_bytes, cudaMemcpyDeviceToDevice, active_stream));
    }
    const auto first_goal = known_goal_configuration != nullptr
        ? known_goal_configuration
        : reinterpret_cast<const float*>(pose_joint_goal_.ptr());

    MotionPlanningRequest motion_request;
    motion_request.start_position = request.current_position;
    motion_request.start_velocity = request.current_velocity;
    motion_request.start_acceleration = request.current_acceleration;
    motion_request.goal_position = first_goal;
    motion_request.seed_goal_positions =
        known_goal_configuration == nullptr
            ? reinterpret_cast<const float*>(ik_solutions_.ptr()) : nullptr;
    motion_request.optimized_knots = request.optimized_knots;
    motion_request.trajectory_position = request.trajectory_position;
    motion_request.trajectory_velocity = request.trajectory_velocity;
    motion_request.trajectory_acceleration = request.trajectory_acceleration;
    motion_request.trajectory_jerk = request.trajectory_jerk;
    motion_request.success = request.success;
    motion_request.cost = request.cost;
    motion_request.collision_cost = request.collision_cost;
    motion_request.trajectory_dt = request.trajectory_dt;
    motion_request.used_rrt = request.used_rrt;
    motion_request.rrt_iterations = request.rrt_iterations;
    motion_request.stream = active_stream;
    motion_request.seed_valid =
        known_goal_configuration == nullptr
            ? reinterpret_cast<const uint8_t*>(ik_success_.ptr()) : nullptr;
    motion_request.selected_seed =
        reinterpret_cast<int32_t*>(pose_selected_seed_.ptr());
    motion_request.attempt_direct_trajectory_optimization =
        attempt < request.enable_graph_attempt;
    motion_request.allow_rrt = attempt >= request.enable_graph_attempt;
    motion_request.finetune_attempts = request.finetune_attempts;
    motion_request.finetune_dt_scale = request.finetune_dt_scale;
    motion_request.goal_tool_positions = request.goal_positions;
    motion_request.goal_tool_quaternions = request.goal_quaternions;
    motion_request.constrain_tool_path = request.constrain_tool_path;
    motion_request.free_position_direction = request.free_position_direction;
    motion_request.direction_in_goal_frame = request.direction_in_goal_frame;
    motion_request.path_position_tolerance =
        request.path_position_tolerance;
    motion_request.path_orientation_tolerance =
        request.path_orientation_tolerance;
    if (plan_impl(motion_request, true, graph_cache)) {
      copy_pose_outputs(request, completed_attempts, active_stream);
      return;
    }
  }
  finish_pose_failure(request, active_stream, completed_attempts);
}

GraspPlanningStatus MotionPlannerRuntime::plan_grasp(
    const GraspPlanningRequest& request) const {
  last_ik_time_seconds_ = 0.0;
  last_rrt_time_seconds_ = 0.0;
  last_rrt_calls_ = 0;
  rrt_->reset_seed();
  if (request.grasp_positions == nullptr ||
      request.grasp_quaternions == nullptr ||
      request.current_position == nullptr || request.success == nullptr ||
      request.goalset_index == nullptr || request.approach.success == nullptr ||
      request.grasp.success == nullptr || request.lift.success == nullptr) {
    throw std::invalid_argument(
        "grasp inputs and overall phase-status outputs must be non-null");
  }
  if (request.num_grasps <= 0 ||
      request.num_grasps > config_.inverse_kinematics.max_goalset ||
      !std::isfinite(request.grasp_approach_offset) ||
      !std::isfinite(request.grasp_lift_offset)) {
    throw std::invalid_argument(
        "grasp goal count or Cartesian offset is invalid");
  }
  validate_phase_output(request.approach, "approach");
  if (request.plan_approach_to_grasp) {
    validate_phase_output(request.grasp, "insertion");
  }
  if (request.plan_approach_to_grasp && request.plan_grasp_to_lift) {
    validate_phase_output(request.lift, "lift");
  }

  CRR_CUDA_CHECK(
      cudaSetDevice(config_.trajectory_optimization.device_ordinal));
  cudaStream_t stream = request.stream;
  if (stream == nullptr && config_.use_cuda_graph) {
    stream = internal_stream_;
  }
  const int tool_frames = trajectory_optimizer_->info().num_tool_frames;
  const size_t goalset_position_bytes =
      static_cast<size_t>(tool_frames) * request.num_grasps * 3 * sizeof(float);
  const size_t goalset_quaternion_bytes =
      static_cast<size_t>(tool_frames) * request.num_grasps * 4 * sizeof(float);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_goalset_position_.ptr()),
      request.grasp_positions, goalset_position_bytes, cudaMemcpyDeviceToDevice,
      stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_goalset_quaternion_.ptr()),
      request.grasp_quaternions, goalset_quaternion_bytes,
      cudaMemcpyDeviceToDevice, stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_phase_start_.ptr()),
      request.current_position, grasp_phase_start_.bytes(),
      cudaMemcpyDeviceToDevice, stream));
  if (request.current_velocity == nullptr) {
    grasp_phase_start_velocity_.clear(stream);
  } else {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(grasp_phase_start_velocity_.ptr()),
        request.current_velocity, grasp_phase_start_velocity_.bytes(),
        cudaMemcpyDeviceToDevice, stream));
  }
  if (request.current_acceleration == nullptr) {
    grasp_phase_start_acceleration_.clear(stream);
  } else {
    CRR_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(grasp_phase_start_acceleration_.ptr()),
        request.current_acceleration, grasp_phase_start_acceleration_.bytes(),
        cudaMemcpyDeviceToDevice, stream));
  }
  auto stable_request = request;
  stable_request.grasp_positions =
      reinterpret_cast<const float*>(grasp_goalset_position_.ptr());
  stable_request.grasp_quaternions =
      reinterpret_cast<const float*>(grasp_goalset_quaternion_.ptr());
  stable_request.current_position =
      reinterpret_cast<const float*>(grasp_phase_start_.ptr());
  stable_request.current_velocity =
      reinterpret_cast<const float*>(grasp_phase_start_velocity_.ptr());
  stable_request.current_acceleration =
      reinterpret_cast<const float*>(grasp_phase_start_acceleration_.ptr());
  const auto approach_axis = axis_vector(request.grasp_approach_axis);
  const auto lift_axis = axis_vector(request.grasp_lift_axis);
  for (const auto& phase : {request.approach, request.grasp, request.lift}) {
    if (phase.spatial_controls != nullptr) {
      CRR_CUDA_CHECK(cudaMemsetAsync(phase.spatial_controls, 0,
          (manifest_.num_knots + 6) * manifest_.num_dof * sizeof(float), stream));
    }
    if (phase.timing_controls != nullptr) {
      CRR_CUDA_CHECK(cudaMemsetAsync(phase.timing_controls, 0, 32 * sizeof(float), stream));
    }
    if (phase.timeopt_duration != nullptr) {
      CRR_CUDA_CHECK(cudaMemsetAsync(phase.timeopt_duration, 0, sizeof(float), stream));
    }
  }
  for (uint8_t* output : {request.success, request.goalset_success,
                          request.approach.success, request.grasp.success,
                          request.lift.success}) {
    if (output != nullptr) {
      CRR_CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(uint8_t), stream));
    }
  }
  CRR_CUDA_CHECK(cudaMemsetAsync(request.goalset_index, 0xff,
                                 sizeof(int32_t), stream));

  std::vector<std::string> collision_links =
      request.disable_collision_links;
  if (collision_links.empty() && request.use_default_collision_links) {
    collision_links = manifest_.grasp_contact_link_names;
  }
  std::vector<std::string> unique_links;
  for (const auto& link : collision_links) {
    if (link.empty()) {
      throw std::invalid_argument("grasp collision-link names cannot be empty");
    }
    if (std::find(unique_links.begin(), unique_links.end(), link) ==
        unique_links.end()) {
      unique_links.push_back(link);
    }
  }

  if (request.collision_policy.enabled) {
    if (tool_frames != 1 || request.use_trajopt_for_goalset_selection) {
      throw std::invalid_argument(
          "layered grasp planning requires one tool and initial IK selection");
    }
    unique_links.clear();  // Never disable world/self collision for contact links.
  }
  GraspCollisionScope collision_scope(*grasp_collision_workspace_, request.collision_policy,
                                      stream);
  auto finish = [&](GraspPlanningStatus status) {
    collision_scope.restore();
    if (config_.trajectory_optimization.enable_timeopt) {
      trajectory_optimizer_->set_execution_speed_scale(config_.trajectory_optimization.execution_speed_scale);
    }
    return status;
  };
  for (float scale : request.phase_execution_speed_scales) {
    if (!std::isfinite(scale) || scale < 0.0F || scale > 1.0F) {
      throw std::invalid_argument("Grasp phase execution speed must be zero or in (0,1]");
    }
  }
  if (config_.trajectory_optimization.enable_timeopt) {
    trajectory_optimizer_->set_execution_speed_scale(
        request.phase_execution_speed_scales[0] > 0.0F ? request.phase_execution_speed_scales[0]
            : config_.trajectory_optimization.execution_speed_scale);
  }
  collision_scope.set_phase(GraspCollisionPhase::Grasp);
  if (!request.phase_inertial_updates[0].empty()) {
    update_links_inertial(request.phase_inertial_updates[0]);
  }
  set_collision_links(unique_links, false);
  try {
    const bool selected = run_goalset_selection(stable_request, stream);
    set_collision_links(unique_links, true);
    if (!selected) {
      return finish(GraspPlanningStatus::GOALSET_UNREACHABLE);
    }
  } catch (...) {
    set_collision_links(unique_links, true);
    throw;
  }
  collision_scope.set_phase(GraspCollisionPhase::Approach);
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_phase_goal_position_.ptr()),
      reinterpret_cast<const void*>(approach_goal_position_.ptr()),
      grasp_phase_goal_position_.bytes(), cudaMemcpyDeviceToDevice, stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void*>(grasp_phase_goal_quaternion_.ptr()),
      reinterpret_cast<const void*>(approach_goal_quaternion_.ptr()),
      grasp_phase_goal_quaternion_.bytes(), cudaMemcpyDeviceToDevice, stream));
  const auto phase_workspace = grasp_phase_workspace();
  auto approach_request = make_pose_request(
      reinterpret_cast<const float*>(grasp_phase_goal_position_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_goal_quaternion_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_start_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_start_velocity_.ptr()),
      reinterpret_cast<const float*>(grasp_phase_start_acceleration_.ptr()),
      phase_workspace, stream);
  plan_pose_impl(approach_request, &grasp_approach_trajopt_graphs_);
  copy_grasp_phase_output(request.approach, stream);
  uint8_t phase_success = 0U;
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      &phase_success, reinterpret_cast<const void*>(selection_success_.ptr()),
      sizeof(phase_success), cudaMemcpyDeviceToHost, stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (phase_success == 0U) {
    return finish(GraspPlanningStatus::APPROACH_FAILED);
  }
  if (!request.plan_approach_to_grasp) {
    CRR_CUDA_CHECK(cudaMemsetAsync(request.success, 1, sizeof(uint8_t), stream));
    CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return finish(GraspPlanningStatus::SUCCESS);
  }

  const size_t endpoint_offset =
      static_cast<size_t>(manifest_.trajectory_steps - 1) *
      manifest_.num_dof;
  const auto* approach_end =
      reinterpret_cast<const float*>(selection_position_.ptr()) +
      endpoint_offset;
  const auto* approach_end_velocity =
      reinterpret_cast<const float*>(selection_velocity_.ptr()) +
      endpoint_offset;
  const auto* approach_end_acceleration =
      reinterpret_cast<const float*>(selection_acceleration_.ptr()) +
      endpoint_offset;
  const int constrained_finetune_attempts =
      request.use_double_pass_trajopt_for_constrained_phases ? 1 : 0;
  collision_scope.set_phase(GraspCollisionPhase::Grasp);
  set_collision_links(unique_links, false);
  try {
    if (!request.phase_inertial_updates[1].empty()) {
      update_links_inertial(request.phase_inertial_updates[1]);
    }
    if (config_.trajectory_optimization.enable_timeopt) {
      trajectory_optimizer_->set_execution_speed_scale(
          request.phase_execution_speed_scales[1] > 0.0F ? request.phase_execution_speed_scales[1]
              : config_.trajectory_optimization.execution_speed_scale);
    }
    const bool grasp_success = plan_linear_phase(
        reinterpret_cast<const float*>(selected_grasp_position_.ptr()),
        reinterpret_cast<const float*>(selected_grasp_quaternion_.ptr()),
        approach_end, approach_end_velocity, approach_end_acceleration,
        approach_axis, request.grasp_approach_in_tool_frame,
        constrained_finetune_attempts, request.grasp, stream,
        &grasp_insert_trajopt_graphs_,
        reinterpret_cast<const float*>(selected_grasp_configuration_.ptr()));
    if (!grasp_success) {
      set_collision_links(unique_links, true);
      return finish(GraspPlanningStatus::GRASP_FAILED);
    }
    if (!request.plan_grasp_to_lift) {
      set_collision_links(unique_links, true);
      CRR_CUDA_CHECK(cudaMemsetAsync(request.success, 1, sizeof(uint8_t), stream));
      CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
      return finish(GraspPlanningStatus::SUCCESS);
    }

    const auto* grasp_end =
        reinterpret_cast<const float*>(selection_position_.ptr()) +
        endpoint_offset;
    const auto* grasp_end_velocity =
        reinterpret_cast<const float*>(selection_velocity_.ptr()) +
        endpoint_offset;
    const auto* grasp_end_acceleration =
        reinterpret_cast<const float*>(selection_acceleration_.ptr()) +
        endpoint_offset;
    // Lift uses the target-subtracted scene. Inertial payload updates are
    // independent of attaching observed collision geometry after the lift.
    collision_scope.set_phase(GraspCollisionPhase::Lift);
    if (!request.phase_inertial_updates[2].empty()) {
      update_links_inertial(request.phase_inertial_updates[2]);
    }
    if (config_.trajectory_optimization.enable_timeopt) {
      trajectory_optimizer_->set_execution_speed_scale(
          request.phase_execution_speed_scales[2] > 0.0F ? request.phase_execution_speed_scales[2]
              : config_.trajectory_optimization.execution_speed_scale);
    }
    const bool lift_success = plan_linear_phase(
        reinterpret_cast<const float*>(lift_goal_position_.ptr()),
        reinterpret_cast<const float*>(lift_goal_quaternion_.ptr()), grasp_end,
        grasp_end_velocity, grasp_end_acceleration, lift_axis,
        request.grasp_lift_in_tool_frame,
        constrained_finetune_attempts, request.lift, stream,
        &grasp_lift_trajopt_graphs_, nullptr);
    set_collision_links(unique_links, true);
    if (!lift_success) {
      return finish(GraspPlanningStatus::LIFT_FAILED);
    }
  } catch (...) {
    set_collision_links(unique_links, true);
    throw;
  }

  CRR_CUDA_CHECK(cudaMemsetAsync(request.success, 1, sizeof(uint8_t), stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return finish(GraspPlanningStatus::SUCCESS);
}

}  // namespace curobo::robot::detail
