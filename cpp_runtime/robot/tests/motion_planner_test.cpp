// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/motion_planner.h"
#include "curobo_robot_runtime/scene.h"

namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename ValueType>
class DeviceValue final {
 public:
  explicit DeviceValue(size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          count * sizeof(ValueType)));
  }

  ~DeviceValue() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceValue(const DeviceValue&) = delete;
  DeviceValue& operator=(const DeviceValue&) = delete;

  ValueType* get() { return pointer_; }

  void upload(const ValueType* values) {
    check_cuda(cudaMemcpy(pointer_, values, count_ * sizeof(ValueType),
                          cudaMemcpyHostToDevice));
  }

  void download(ValueType* values) const {
    check_cuda(cudaMemcpy(values, pointer_, count_ * sizeof(ValueType),
                          cudaMemcpyDeviceToHost));
  }

 private:
  ValueType* pointer_{nullptr};
  size_t count_{0};
};

class GraspPhaseStorage final {
 public:
  GraspPhaseStorage(size_t knot_values, size_t trajectory_values)
      : knots(knot_values),
        position(trajectory_values),
        velocity(trajectory_values),
        acceleration(trajectory_values),
        jerk(trajectory_values),
        success(1),
        cost(1),
        collision_cost(1),
        trajectory_dt(1),
        used_rrt(1),
        rrt_iterations(1),
        selected_seed(1),
        planning_attempts(1) {
    output.optimized_knots = knots.get();
    output.trajectory_position = position.get();
    output.trajectory_velocity = velocity.get();
    output.trajectory_acceleration = acceleration.get();
    output.trajectory_jerk = jerk.get();
    output.success = success.get();
    output.cost = cost.get();
    output.collision_cost = collision_cost.get();
    output.trajectory_dt = trajectory_dt.get();
    output.used_rrt = used_rrt.get();
    output.rrt_iterations = rrt_iterations.get();
    output.selected_seed = selected_seed.get();
    output.planning_attempts = planning_attempts.get();
  }

  DeviceValue<float> knots;
  DeviceValue<float> position;
  DeviceValue<float> velocity;
  DeviceValue<float> acceleration;
  DeviceValue<float> jerk;
  DeviceValue<uint8_t> success;
  DeviceValue<float> cost;
  DeviceValue<float> collision_cost;
  DeviceValue<float> trajectory_dt;
  DeviceValue<uint8_t> used_rrt;
  DeviceValue<int32_t> rrt_iterations;
  DeviceValue<int32_t> selected_seed;
  DeviceValue<int32_t> planning_attempts;
  curobo::robot::GraspPhaseOutput output;
};

void require_near(float actual, float expected, float tolerance,
                  const char* description) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(description) + " mismatch: " +
                             std::to_string(actual) + " vs " +
                             std::to_string(expected));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && (argc != 3 || std::string(argv[2]) != "--layered-grasp")) {
      throw std::invalid_argument(
          "usage: curobo_robot_motion_test ARTIFACT_DIRECTORY [--layered-grasp]");
    }
    const bool layered_grasp = argc == 3;
    curobo::robot::MotionPlanningConfig config;
    if (config.trajectory_optimization.max_iterations != 25 ||
        config.trajectory_optimization.refinement_iterations != 25 ||
        config.trajectory_optimization.finetune_attempts != 2 ||
        curobo::robot::PoseMotionPlanningRequest{}.finetune_attempts != 2 ||
        config.trajectory_optimization.initial_dt_scale != 1.25F ||
        config.trajectory_optimization.finetune_dt_scale != 0.6F) {
      throw std::runtime_error("motion-planner defaults must use fitted 25+2*25");
    }
    if (config.inverse_kinematics.cspace_rank_weight != 5.0F) {
      throw std::runtime_error(
          "motion-planner IK c-space rank weight must match Python");
    }
    if (config.trajectory_optimization.constraint_tolerance != 0.0F) {
      throw std::runtime_error(
          "motion-planner constraint tolerance must match Python");
    }
    config.trajectory_optimization.num_seeds = 4;
    config.trajectory_optimization.max_iterations = 50;
    config.trajectory_optimization.check_effort_limits = true;
    config.inverse_kinematics.check_effort_limits = true;
    config.rrt.check_effort_limits = true;
    config.rrt.max_plans = config.trajectory_optimization.num_seeds;
    curobo::robot::MotionPlanner direct_planner(argv[1], config);
    const auto info = direct_planner.info();
    if (info.num_dof != 7 || info.num_knots != 16 ||
        info.trajectory_steps != 81 || info.max_rrt_path_nodes < 2) {
      throw std::runtime_error(
          "motion-planner test expects the default Franka artifact");
    }
    if (direct_planner.attachment_sphere_capacity("attached_object") != (layered_grasp ? 32 : 4)) {
      throw std::runtime_error("Franka attachment sphere capacity mismatch");
    }
    curobo::robot::SphereAttachment attachment;
    attachment.spheres = {{{0.0F, 0.0F, 0.1F}, 0.01F}};
    direct_planner.attach_object(attachment);
    curobo::robot::LinkInertialUpdate payload;
    payload.link_name = "attached_object";
    payload.mass = 0.1F;
    payload.center_of_mass = std::array<float, 3>{0.0F, 0.0F, 0.1F};
    payload.inertia =
        std::array<float, 6>{0.001F, 0.001F, 0.001F, 0.0F, 0.0F, 0.0F};
    direct_planner.update_links_inertial({payload});

    const std::array<float, 7> start{
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F};
    const std::array<float, 7> goal{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    DeviceValue<float> start_device(start.size());
    DeviceValue<float> goal_device(goal.size());
    start_device.upload(start.data());
    goal_device.upload(goal.data());

    const size_t knot_values =
        static_cast<size_t>(info.num_knots * info.num_dof);
    const size_t trajectory_values =
        static_cast<size_t>(info.trajectory_steps * info.num_dof);
    DeviceValue<float> knots(knot_values);
    DeviceValue<float> position(trajectory_values);
    DeviceValue<float> velocity(trajectory_values);
    DeviceValue<float> acceleration(trajectory_values);
    DeviceValue<float> jerk(trajectory_values);
    DeviceValue<uint8_t> success(1);
    DeviceValue<float> cost(1);
    DeviceValue<float> collision_cost(1);
    DeviceValue<uint8_t> used_rrt(1);
    DeviceValue<int32_t> rrt_iterations(1);
    DeviceValue<float> trajectory_dt(1);
    DeviceValue<int32_t> selected_seed(1);
    curobo::robot::MotionPlanningRequest request{
        start_device.get(), goal_device.get(), nullptr, knots.get(), position.get(),
        velocity.get(), acceleration.get(), jerk.get(), success.get(),
        cost.get(), collision_cost.get(), trajectory_dt.get(), used_rrt.get(),
        rrt_iterations.get(), nullptr};
    request.selected_seed = selected_seed.get();

    direct_planner.plan(request);
    uint8_t success_value = 0;
    uint8_t used_rrt_value = 1;
    int32_t rrt_iteration_value = -1;
    float collision_cost_value = -1.0F;
    float trajectory_dt_value = 0.0F;
    int32_t selected_seed_value = -1;
    success.download(&success_value);
    used_rrt.download(&used_rrt_value);
    rrt_iterations.download(&rrt_iteration_value);
    collision_cost.download(&collision_cost_value);
    trajectory_dt.download(&trajectory_dt_value);
    if (success_value != 1 || used_rrt_value != 0 ||
        rrt_iteration_value != 0 || collision_cost_value != 0.0F ||
        !(trajectory_dt_value > 0.0F) ||
        trajectory_dt_value > config.trajectory_optimization.trajectory_dt ||
        direct_planner.last_rrt_calls() != 0 ||
        direct_planner.last_rrt_time() != 0.0) {
      throw std::runtime_error("direct motion-planning attempt failed");
    }

    // The composed planner must forward the selected seed's timing controls.
    auto timeopt_config = config;
    timeopt_config.trajectory_optimization.enable_timeopt = true;
    curobo::robot::MotionPlanner timeopt_planner(argv[1], timeopt_config);
    timeopt_planner.update_links_inertial({payload});
    timeopt_planner.plan(request);
    check_cuda(cudaDeviceSynchronize());
    success.download(&success_value);
    const auto timeopt = timeopt_planner.timeopt_trajectory();
    if (success_value != 1 || timeopt.timing_controls == nullptr ||
        timeopt.num_timing_controls != 32) {
      throw std::runtime_error("planner did not expose selected timing controls");
    }
    std::vector<float> timing(timeopt.num_timing_controls);
    check_cuda(cudaMemcpy(timing.data(), timeopt.timing_controls,
                          timing.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (const float value : timing) {
      if (!std::isfinite(value) || value <= 0.0F) {
        throw std::runtime_error("planner timing controls must be finite and positive");
      }
    }

    const std::array<float, 7> start_velocity{
        0.01F, -0.01F, 0.005F, -0.01F, 0.005F, 0.01F, -0.005F};
    const std::array<float, 7> start_acceleration{
        0.02F, -0.02F, 0.01F, -0.02F, 0.01F, 0.02F, -0.01F};
    DeviceValue<float> start_velocity_device(start_velocity.size());
    DeviceValue<float> start_acceleration_device(start_acceleration.size());
    start_velocity_device.upload(start_velocity.data());
    start_acceleration_device.upload(start_acceleration.data());
    request.start_velocity = start_velocity_device.get();
    request.start_acceleration = start_acceleration_device.get();
    direct_planner.plan(request);
    success.download(&success_value);
    if (success_value != 1) {
      throw std::runtime_error(
          "motion planner rejected a non-static start boundary");
    }
    std::vector<float> velocity_host(trajectory_values);
    std::vector<float> acceleration_host(trajectory_values);
    velocity.download(velocity_host.data());
    acceleration.download(acceleration_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(velocity_host.at(static_cast<size_t>(joint)),
                   start_velocity.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "motion trajectory start velocity");
      require_near(acceleration_host.at(static_cast<size_t>(joint)),
                   start_acceleration.at(static_cast<size_t>(joint)), 1.0E-4F,
                   "motion trajectory start acceleration");
    }
    request.start_velocity = nullptr;
    request.start_acceleration = nullptr;

    direct_planner.plan(request);
    success.download(&success_value);
    used_rrt.download(&used_rrt_value);
    if (success_value != 1 || used_rrt_value != 0) {
      throw std::runtime_error("direct motion-planning graph replay failed");
    }
    velocity.download(velocity_host.data());
    acceleration.download(acceleration_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(velocity_host.at(static_cast<size_t>(joint)), 0.0F,
                   1.0E-5F, "default motion start velocity");
      require_near(acceleration_host.at(static_cast<size_t>(joint)), 0.0F,
                   1.0E-4F, "default motion start acceleration");
    }
    direct_planner.detach_object();
    direct_planner.plan(request);
    success.download(&success_value);
    used_rrt.download(&used_rrt_value);
    if (success_value != 1 || used_rrt_value != 0) {
      throw std::runtime_error("post-detachment motion graph replay failed");
    }

    const auto robot_info = direct_planner.robot_info();
    const size_t pose_position_values =
        static_cast<size_t>(robot_info.num_tool_frames * 3);
    const size_t pose_quaternion_values =
        static_cast<size_t>(robot_info.num_tool_frames * 4);
    const size_t pose_jacobian_values = static_cast<size_t>(
        robot_info.num_tool_frames * 6 * robot_info.num_dof);
    DeviceValue<float> goal_positions(pose_position_values);
    DeviceValue<float> goal_quaternions(pose_quaternion_values);
    DeviceValue<float> pose_jacobians(pose_jacobian_values);
    curobo::robot::Kinematics kinematics(argv[1]);
    kinematics.forward(goal_device.get(), goal_positions.get(),
                       goal_quaternions.get(), pose_jacobians.get(), 1, 1);
    check_cuda(cudaDeviceSynchronize());

    DeviceValue<float> selected_goal(robot_info.num_dof);
    DeviceValue<float> position_error(1);
    DeviceValue<float> orientation_error(1);
    DeviceValue<int32_t> planning_attempts(1);
    curobo::robot::PoseMotionPlanningRequest pose_request;
    pose_request.goal_positions = goal_positions.get();
    pose_request.goal_quaternions = goal_quaternions.get();
    pose_request.current_position = start_device.get();
    pose_request.current_velocity = start_velocity_device.get();
    pose_request.current_acceleration = start_acceleration_device.get();
    pose_request.selected_goal_position = selected_goal.get();
    pose_request.optimized_knots = knots.get();
    pose_request.trajectory_position = position.get();
    pose_request.trajectory_velocity = velocity.get();
    pose_request.trajectory_acceleration = acceleration.get();
    pose_request.trajectory_jerk = jerk.get();
    pose_request.success = success.get();
    pose_request.cost = cost.get();
    pose_request.collision_cost = collision_cost.get();
    pose_request.trajectory_dt = trajectory_dt.get();
    pose_request.position_error = position_error.get();
    pose_request.orientation_error = orientation_error.get();
    pose_request.used_rrt = used_rrt.get();
    pose_request.rrt_iterations = rrt_iterations.get();
    pose_request.selected_seed = selected_seed.get();
    pose_request.planning_attempts = planning_attempts.get();
    direct_planner.plan_pose(pose_request);

    float position_error_value = 0.0F;
    float orientation_error_value = 0.0F;
    int32_t planning_attempts_value = 0;
    success.download(&success_value);
    used_rrt.download(&used_rrt_value);
    selected_seed.download(&selected_seed_value);
    position_error.download(&position_error_value);
    orientation_error.download(&orientation_error_value);
    planning_attempts.download(&planning_attempts_value);
    if (success_value != 1 || used_rrt_value != 0 ||
        selected_seed_value < 0 ||
        selected_seed_value >= config.trajectory_optimization.num_seeds ||
        planning_attempts_value != 1 ||
        position_error_value > config.inverse_kinematics.position_tolerance ||
        orientation_error_value >
            config.inverse_kinematics.orientation_tolerance) {
      throw std::runtime_error(
          "single-pose motion planning did not match the IK-to-TrajOpt "
          "contract");
    }
    velocity.download(velocity_host.data());
    acceleration.download(acceleration_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(velocity_host.at(static_cast<size_t>(joint)),
                   start_velocity.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "pose trajectory start velocity");
      require_near(acceleration_host.at(static_cast<size_t>(joint)),
                   start_acceleration.at(static_cast<size_t>(joint)), 1.0E-4F,
                   "pose trajectory start acceleration");
    }

    DeviceValue<float> endpoint_positions(pose_position_values);
    DeviceValue<float> endpoint_quaternions(pose_quaternion_values);
    DeviceValue<float> endpoint_jacobians(pose_jacobian_values);
    const float* endpoint =
        position.get() + static_cast<size_t>(info.trajectory_steps - 1) *
                             static_cast<size_t>(info.num_dof);
    kinematics.forward(endpoint, endpoint_positions.get(),
                       endpoint_quaternions.get(), endpoint_jacobians.get(), 1,
                       1);
    check_cuda(cudaDeviceSynchronize());
    std::vector<float> goal_position_host(pose_position_values);
    std::vector<float> endpoint_position_host(pose_position_values);
    std::vector<float> goal_quaternion_host(pose_quaternion_values);
    std::vector<float> endpoint_quaternion_host(pose_quaternion_values);
    goal_positions.download(goal_position_host.data());
    endpoint_positions.download(endpoint_position_host.data());
    goal_quaternions.download(goal_quaternion_host.data());
    endpoint_quaternions.download(endpoint_quaternion_host.data());
    for (int tool = 0; tool < robot_info.num_tool_frames; ++tool) {
      float squared_position_error = 0.0F;
      float quaternion_dot = 0.0F;
      for (int axis = 0; axis < 3; ++axis) {
        const auto index = static_cast<size_t>(tool * 3 + axis);
        const float difference =
            endpoint_position_host.at(index) - goal_position_host.at(index);
        squared_position_error += difference * difference;
      }
      for (int component = 0; component < 4; ++component) {
        const auto index = static_cast<size_t>(tool * 4 + component);
        quaternion_dot += endpoint_quaternion_host.at(index) *
                          goal_quaternion_host.at(index);
      }
      if (std::sqrt(squared_position_error) >
              config.inverse_kinematics.position_tolerance ||
          2.0F * std::acos(std::min(std::abs(quaternion_dot), 1.0F)) >
              config.inverse_kinematics.orientation_tolerance) {
        throw std::runtime_error(
            "single-pose trajectory endpoint does not reach the goal pose");
      }
    }

    GraspPhaseStorage approach_phase(knot_values, trajectory_values);
    GraspPhaseStorage grasp_phase(knot_values, trajectory_values);
    GraspPhaseStorage lift_phase(knot_values, trajectory_values);
    DeviceValue<uint8_t> grasp_success(1);
    DeviceValue<uint8_t> goalset_success(1);
    DeviceValue<int32_t> goalset_index(1);
    DeviceValue<float> selected_grasp(robot_info.num_dof);
    DeviceValue<float> goalset_position_error(1);
    DeviceValue<float> goalset_orientation_error(1);
    DeviceValue<float> goalset_collision_cost(1);
    std::vector<float> grasp_goalset_positions(
        static_cast<size_t>(robot_info.num_tool_frames) * 2 * 3);
    std::vector<float> grasp_goalset_quaternions(
        static_cast<size_t>(robot_info.num_tool_frames) * 2 * 4);
    for (int tool = 0; tool < robot_info.num_tool_frames; ++tool) {
      for (int axis = 0; axis < 3; ++axis) {
        grasp_goalset_positions.at(
            (static_cast<size_t>(tool) * 2 + 0) * 3 + axis) = 10.0F;
        grasp_goalset_positions.at(
            (static_cast<size_t>(tool) * 2 + 1) * 3 + axis) =
            goal_position_host.at(static_cast<size_t>(tool) * 3 + axis);
      }
      for (int component = 0; component < 4; ++component) {
        const float value = goal_quaternion_host.at(
            static_cast<size_t>(tool) * 4 + component);
        grasp_goalset_quaternions.at(
            (static_cast<size_t>(tool) * 2 + 0) * 4 + component) = value;
        grasp_goalset_quaternions.at(
            (static_cast<size_t>(tool) * 2 + 1) * 4 + component) = value;
      }
    }
    DeviceValue<float> grasp_goalset_positions_device(
        grasp_goalset_positions.size());
    DeviceValue<float> grasp_goalset_quaternions_device(
        grasp_goalset_quaternions.size());
    grasp_goalset_positions_device.upload(grasp_goalset_positions.data());
    grasp_goalset_quaternions_device.upload(grasp_goalset_quaternions.data());
    curobo::robot::GraspPlanningRequest grasp_request;
    grasp_request.grasp_positions = grasp_goalset_positions_device.get();
    grasp_request.grasp_quaternions = grasp_goalset_quaternions_device.get();
    grasp_request.current_position = start_device.get();
    grasp_request.current_velocity = start_velocity_device.get();
    grasp_request.current_acceleration = start_acceleration_device.get();
    grasp_request.num_grasps = 2;
    grasp_request.approach = approach_phase.output;
    grasp_request.grasp = grasp_phase.output;
    grasp_request.lift = lift_phase.output;
    grasp_request.success = grasp_success.get();
    grasp_request.goalset_success = goalset_success.get();
    grasp_request.goalset_index = goalset_index.get();
    grasp_request.selected_grasp_configuration = selected_grasp.get();
    grasp_request.goalset_position_error = goalset_position_error.get();
    grasp_request.goalset_orientation_error = goalset_orientation_error.get();
    grasp_request.goalset_collision_cost = goalset_collision_cost.get();
    grasp_request.grasp_approach_offset = -0.02F;
    grasp_request.grasp_lift_offset = 0.02F;
    DeviceValue<float> target_point(3);
    if (layered_grasp) {
      const auto& q = goal_quaternion_host;
      const float point[3]{goal_position_host[0] + .2F * (q[1] * q[3] + q[0] * q[2]),
                           goal_position_host[1] + .2F * (q[2] * q[3] - q[0] * q[1]),
                           goal_position_host[2] + .1F * (1 - 2 * (q[1] * q[1] + q[2] * q[2]))};
      target_point.upload(point);
      grasp_request.collision_policy.enabled = true;
      grasp_request.collision_policy.contact_links = {"panda_leftfinger", "panda_rightfinger"};
      grasp_request.collision_policy.target_points = target_point.get();
      grasp_request.collision_policy.target_point_count = 1;
      grasp_request.collision_policy.target_voxel_radius = .005F;
    }
    grasp_request.num_grasps = 1;
    if (direct_planner.plan_grasp(grasp_request) !=
        curobo::robot::GraspPlanningStatus::GOALSET_UNREACHABLE) {
      throw std::runtime_error("unreachable grasp goalset was not rejected");
    }
    grasp_request.num_grasps = 2;
    const auto grasp_status = direct_planner.plan_grasp(grasp_request);

    uint8_t goalset_success_value = 0U;
    uint8_t approach_success_value = 0U;
    uint8_t insertion_success_value = 0U;
    uint8_t lift_success_value = 0U;
    int32_t goalset_index_value = -1;
    grasp_success.download(&success_value);
    goalset_success.download(&goalset_success_value);
    approach_phase.success.download(&approach_success_value);
    grasp_phase.success.download(&insertion_success_value);
    lift_phase.success.download(&lift_success_value);
    goalset_index.download(&goalset_index_value);
    if (grasp_status != curobo::robot::GraspPlanningStatus::SUCCESS ||
        success_value != 1U || goalset_success_value != 1U ||
        approach_success_value != 1U || insertion_success_value != 1U ||
        lift_success_value != 1U || goalset_index_value != 1) {
      throw std::runtime_error(
          "grasp planning did not complete goalset, approach, insertion, and lift");
    }
    std::vector<float> selected_grasp_host(static_cast<size_t>(info.num_dof));
    std::vector<float> insertion_position_host(trajectory_values);
    selected_grasp.download(selected_grasp_host.data());
    grasp_phase.position.download(insertion_position_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(
          insertion_position_host.at(static_cast<size_t>(info.trajectory_steps - 1) *
                               info.num_dof + joint),
          selected_grasp_host.at(static_cast<size_t>(joint)), 1.0E-5F,
          "insertion must reuse the selected grasp joint configuration");
    }
    approach_phase.velocity.download(velocity_host.data());
    approach_phase.acceleration.download(acceleration_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(velocity_host.at(static_cast<size_t>(joint)),
                   start_velocity.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "grasp approach start velocity");
      require_near(acceleration_host.at(static_cast<size_t>(joint)),
                   start_acceleration.at(static_cast<size_t>(joint)), 1.0E-4F,
                   "grasp approach start acceleration");
    }

    const size_t phase_pose_position_values =
        static_cast<size_t>(info.trajectory_steps) *
        robot_info.num_tool_frames * 3;
    const size_t phase_pose_quaternion_values =
        static_cast<size_t>(info.trajectory_steps) *
        robot_info.num_tool_frames * 4;
    const size_t phase_pose_jacobian_values =
        static_cast<size_t>(info.trajectory_steps) *
        robot_info.num_tool_frames * 6 * robot_info.num_dof;
    DeviceValue<float> phase_positions(phase_pose_position_values);
    DeviceValue<float> phase_quaternions(phase_pose_quaternion_values);
    DeviceValue<float> phase_jacobians(phase_pose_jacobian_values);
    std::vector<float> phase_position_host(phase_pose_position_values);
    std::vector<float> phase_quaternion_host(phase_pose_quaternion_values);
    const auto validate_linear_phase = [&](GraspPhaseStorage& phase,
                                           float goal_axis_offset,
                                           bool check_whole_path) {
      kinematics.forward(phase.position.get(), phase_positions.get(),
                         phase_quaternions.get(), phase_jacobians.get(), 1,
                         info.trajectory_steps);
      check_cuda(cudaDeviceSynchronize());
      phase_positions.download(phase_position_host.data());
      phase_quaternions.download(phase_quaternion_host.data());
      for (int tool = 0; tool < robot_info.num_tool_frames; ++tool) {
        const size_t goal_p = static_cast<size_t>(tool) * 3;
        const size_t goal_q = static_cast<size_t>(tool) * 4;
        const float w = goal_quaternion_host.at(goal_q);
        const float x = goal_quaternion_host.at(goal_q + 1);
        const float y = goal_quaternion_host.at(goal_q + 2);
        const float z = goal_quaternion_host.at(goal_q + 3);
        const float inverse_norm =
            1.0F / std::sqrt(w * w + x * x + y * y + z * z);
        const float nw = w * inverse_norm;
        const float nx = x * inverse_norm;
        const float ny = y * inverse_norm;
        const float nz = z * inverse_norm;
        const std::array<float, 3> direction{
            2.0F * (nx * nz + nw * ny),
            2.0F * (ny * nz - nw * nx),
            1.0F - 2.0F * (nx * nx + ny * ny),
        };
        const int first_state = check_whole_path ? 0 : info.trajectory_steps - 1;
        for (int state = first_state; state < info.trajectory_steps; ++state) {
          const size_t state_p =
              (static_cast<size_t>(state) * robot_info.num_tool_frames + tool) *
              3;
          const size_t state_q =
              (static_cast<size_t>(state) * robot_info.num_tool_frames + tool) *
              4;
          std::array<float, 3> difference{};
          float along = 0.0F;
          for (int axis = 0; axis < 3; ++axis) {
            const float target = goal_position_host.at(goal_p + axis) +
                                 goal_axis_offset * direction.at(axis);
            difference.at(axis) = phase_position_host.at(state_p + axis) - target;
            along += difference.at(axis) * direction.at(axis);
          }
          float perpendicular_squared = 0.0F;
          for (int axis = 0; axis < 3; ++axis) {
            const float perpendicular =
                difference.at(axis) - along * direction.at(axis);
            perpendicular_squared += perpendicular * perpendicular;
          }
          float quaternion_dot = 0.0F;
          for (int component = 0; component < 4; ++component) {
            quaternion_dot += phase_quaternion_host.at(state_q + component) *
                              goal_quaternion_host.at(goal_q + component);
          }
          const float orientation_distance =
              2.0F * std::acos(std::min(std::abs(quaternion_dot), 1.0F));
          if (std::sqrt(perpendicular_squared) >
                  config.trajectory_optimization.position_tolerance + 1.0E-3F ||
              orientation_distance >
                  config.trajectory_optimization.orientation_tolerance + 1.0E-3F) {
            throw std::runtime_error(
                "grasp phase violated its Cartesian line constraint");
          }
          if (state == info.trajectory_steps - 1 &&
              std::abs(along) >
                  config.trajectory_optimization.position_tolerance + 1.0E-3F) {
            throw std::runtime_error("grasp phase did not reach its terminal pose");
          }
        }
      }
    };
    validate_linear_phase(approach_phase, -0.02F, false);
    validate_linear_phase(grasp_phase, 0.0F, true);
    validate_linear_phase(lift_phase, 0.02F, true);

    grasp_request.use_trajopt_for_goalset_selection = true;
    grasp_request.collision_policy.enabled = false;  // Separate, non-layered selection contract.
    grasp_request.use_double_pass_trajopt_for_constrained_phases = true;
    if (direct_planner.plan_grasp(grasp_request) !=
        curobo::robot::GraspPlanningStatus::SUCCESS) {
      throw std::runtime_error(
          "TrajOpt goalset selection or double-pass grasp planning failed");
    }
    grasp_request.use_trajopt_for_goalset_selection = false;
    grasp_request.collision_policy.enabled = layered_grasp;
    grasp_request.use_double_pass_trajopt_for_constrained_phases = false;

    DeviceValue<float> alternate_grasp_positions(
        grasp_goalset_positions.size());
    DeviceValue<float> alternate_grasp_quaternions(
        grasp_goalset_quaternions.size());
    DeviceValue<float> alternate_start(start.size());
    alternate_grasp_positions.upload(grasp_goalset_positions.data());
    alternate_grasp_quaternions.upload(grasp_goalset_quaternions.data());
    alternate_start.upload(start.data());
    GraspPhaseStorage alternate_approach(knot_values, trajectory_values);
    GraspPhaseStorage alternate_grasp(knot_values, trajectory_values);
    GraspPhaseStorage alternate_lift(knot_values, trajectory_values);
    auto alternate_request = grasp_request;
    alternate_request.grasp_positions = alternate_grasp_positions.get();
    alternate_request.grasp_quaternions = alternate_grasp_quaternions.get();
    alternate_request.current_position = alternate_start.get();
    alternate_request.approach = alternate_approach.output;
    alternate_request.grasp = alternate_grasp.output;
    alternate_request.lift = alternate_lift.output;
    if (direct_planner.plan_grasp(alternate_request) !=
        curobo::robot::GraspPlanningStatus::SUCCESS) {
      throw std::runtime_error(
          "stable grasp workspace rejected alternate caller buffers");
    }
    alternate_approach.success.download(&approach_success_value);
    alternate_grasp.success.download(&insertion_success_value);
    alternate_lift.success.download(&lift_success_value);
    if (approach_success_value != 1U || insertion_success_value != 1U ||
        lift_success_value != 1U) {
      throw std::runtime_error(
          "stable grasp workspace did not copy alternate phase outputs");
    }

    pose_request.enable_graph_attempt = 0;
    direct_planner.plan_pose(pose_request);
    success.download(&success_value);
    used_rrt.download(&used_rrt_value);
    planning_attempts.download(&planning_attempts_value);
    if (success_value != 1 || used_rrt_value != 1 ||
        planning_attempts_value != 1 ||
        direct_planner.last_rrt_calls() != 1 ||
        !(direct_planner.last_rrt_time() > 0.0)) {
      throw std::runtime_error(
          "pose graph-attempt policy did not select the RRT pipeline");
    }

    const std::array<float, 28> seed_starts{
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    const std::array<float, 28> seed_goals{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F,
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F,
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F,
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    const std::array<uint8_t, 4> seed_valid{0U, 0U, 0U, 1U};
    DeviceValue<float> seed_starts_device(seed_starts.size());
    DeviceValue<float> seed_goals_device(seed_goals.size());
    DeviceValue<uint8_t> seed_valid_device(seed_valid.size());
    seed_starts_device.upload(seed_starts.data());
    seed_goals_device.upload(seed_goals.data());
    seed_valid_device.upload(seed_valid.data());
    request.seed_start_positions = seed_starts_device.get();
    request.seed_goal_positions = seed_goals_device.get();
    request.seed_valid = seed_valid_device.get();
    direct_planner.plan(request);
    success.download(&success_value);
    selected_seed.download(&selected_seed_value);
    if (success_value != 1 || selected_seed_value != 3) {
      throw std::runtime_error(
          "motion planner did not select the valid endpoint pair");
    }
    std::vector<float> paired_position_host(trajectory_values);
    position.download(paired_position_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      const auto end_index = static_cast<size_t>(
          (info.trajectory_steps - 1) * info.num_dof + joint);
      require_near(paired_position_host.at(static_cast<size_t>(joint)),
                   goal.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "paired motion start");
      require_near(paired_position_host.at(end_index),
                   goal.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "paired motion goal");
    }
    const std::array<float, 28> graph_seed_starts{
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F,
        0.0F, -1.3F, 0.0F, -2.5F, 0.0F, 1.5F, 0.8F};
    seed_starts_device.upload(graph_seed_starts.data());

    curobo::robot::Scene scene;
    curobo::robot::Cuboid obstacle;
    obstacle.name = "mid_path_obstacle";
    obstacle.pose.position = {0.218F, 0.126F, 0.723F};
    obstacle.dimensions = {0.02F, 0.02F, 0.02F};
    scene.add_cuboid(obstacle);
    config.attempt_direct_trajectory_optimization = false;
    curobo::robot::MotionPlanner graph_planner(argv[1], config, &scene);
    graph_planner.plan(request);
    success.download(&success_value);
    used_rrt.download(&used_rrt_value);
    rrt_iterations.download(&rrt_iteration_value);
    selected_seed.download(&selected_seed_value);
    collision_cost.download(&collision_cost_value);
    if (success_value != 1 || used_rrt_value != 1 ||
        rrt_iteration_value <= 0 || selected_seed_value != 3 ||
        collision_cost_value != 0.0F) {
      throw std::runtime_error(
          "RRT-seeded motion planning failed: success=" +
          std::to_string(success_value) + " used_rrt=" +
          std::to_string(used_rrt_value) + " iterations=" +
          std::to_string(rrt_iteration_value) + " selected_seed=" +
          std::to_string(selected_seed_value) + " collision_cost=" +
          std::to_string(collision_cost_value));
    }

    std::vector<float> position_host(trajectory_values);
    position.download(position_host.data());
    for (int joint = 0; joint < info.num_dof; ++joint) {
      require_near(position_host.at(static_cast<size_t>(joint)),
                   start.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "motion trajectory start");
      const auto goal_index = static_cast<size_t>(
          (info.trajectory_steps - 1) * info.num_dof + joint);
      require_near(position_host.at(goal_index),
                   goal.at(static_cast<size_t>(joint)), 1.0E-5F,
                   "motion trajectory goal");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
