// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "curobo_robot_runtime/inverse_kinematics.h"
#include "curobo_robot_runtime/kinematics.h"
#include "curobo_robot_runtime/scene.h"
#include "ik_runtime.h"

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
  const ValueType* get() const { return pointer_; }

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

struct SolveBuffers {
  explicit SolveBuffers(int dof, int batch_size = 1, int num_solutions = 1)
      : batch_size(batch_size),
        num_solutions(num_solutions),
        solution(static_cast<size_t>(dof * batch_size * num_solutions)),
        success(static_cast<size_t>(batch_size * num_solutions)),
        position_error(static_cast<size_t>(batch_size * num_solutions)),
        orientation_error(static_cast<size_t>(batch_size * num_solutions)),
        collision_cost(static_cast<size_t>(batch_size * num_solutions)) {}

  curobo::robot::IkSolveRequest request(const float* goal_position,
                                        const float* goal_quaternion,
                                        cudaStream_t stream = nullptr) {
    return curobo::robot::IkSolveRequest{
        goal_position,
        goal_quaternion,
        nullptr,
        nullptr,
        solution.get(),
        success.get(),
        position_error.get(),
        orientation_error.get(),
        collision_cost.get(),
        nullptr,
        batch_size,
        1,
        num_solutions,
        stream,
    };
  }

  int batch_size{1};
  int num_solutions{1};
  DeviceValue<float> solution;
  DeviceValue<uint8_t> success;
  DeviceValue<float> position_error;
  DeviceValue<float> orientation_error;
  DeviceValue<float> collision_cost;
};

void require_success(const SolveBuffers& buffers, float maximum_position_error,
                     float maximum_orientation_error,
                     const std::string& context) {
  uint8_t success = 0;
  float position_error = 0.0F;
  float orientation_error = 0.0F;
  float collision_cost = 0.0F;
  buffers.success.download(&success);
  buffers.position_error.download(&position_error);
  buffers.orientation_error.download(&orientation_error);
  buffers.collision_cost.download(&collision_cost);
  if (success != 1 || position_error >= maximum_position_error ||
      orientation_error >= maximum_orientation_error || collision_cost != 0.0F) {
    throw std::runtime_error(
        context + " failed its IK convergence contract: success=" +
        std::to_string(success) + " position_error=" +
        std::to_string(position_error) + " orientation_error=" +
        std::to_string(orientation_error) + " collision_cost=" +
        std::to_string(collision_cost));
  }
}

void require_failure(const SolveBuffers& buffers, const std::string& context) {
  uint8_t success = 1U;
  buffers.success.download(&success);
  if (success != 0U) {
    throw std::runtime_error(context + " unexpectedly succeeded");
  }
}

void test_lm_exit_criterion() {
  using curobo::robot::detail::has_valid_solution_for_each_batch;

  if (!has_valid_solution_for_each_batch({1U, 0U, 0U, 0U}, 1, 4)) {
    throw std::runtime_error(
        "LM exit must accept one valid solution in a single-pose request");
  }
  if (!has_valid_solution_for_each_batch({0U, 1U, 0U, 0U, 0U, 1U}, 2, 3)) {
    throw std::runtime_error(
        "LM exit must accept one valid solution for every batch item");
  }
  if (has_valid_solution_for_each_batch({1U, 0U, 0U, 0U, 0U, 0U}, 2, 3)) {
    throw std::runtime_error(
        "LM exit must reject a batch item without a valid solution");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    test_lm_exit_criterion();
    if (argc != 2) {
      throw std::invalid_argument("usage: curobo_robot_ik_test ARTIFACT_DIRECTORY");
    }
    const std::string artifact_directory = argv[1];
    curobo::robot::Kinematics kinematics(artifact_directory);
    const auto info = kinematics.info();
    if (info.num_dof != 7 || info.num_tool_frames != 1) {
      throw std::runtime_error("IK test expects the exported Franka artifact");
    }

    const std::array<float, 7> target_joint_position{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F};
    DeviceValue<float> target_joint_device(target_joint_position.size());
    target_joint_device.upload(target_joint_position.data());
    DeviceValue<float> goal_position(3);
    DeviceValue<float> goal_quaternion(4);
    DeviceValue<float> goal_jacobian(static_cast<size_t>(6 * info.num_dof));
    kinematics.forward(target_joint_device.get(), goal_position.get(),
                       goal_quaternion.get(), goal_jacobian.get(), 1, 1);
    check_cuda(cudaDeviceSynchronize());

    curobo::robot::InverseKinematicsConfig config;
    config.num_seeds = 32;
    config.lbfgs_iterations = 0;
    config.check_effort_limits = true;
    curobo::robot::InverseKinematics inverse_kinematics(artifact_directory, config);
    if (inverse_kinematics.attachment_sphere_capacity("attached_object") != 4) {
      throw std::runtime_error("Franka attachment sphere capacity mismatch");
    }
    curobo::robot::SphereAttachment attachment;
    attachment.spheres = {{{0.0F, 0.0F, 0.1F}, 0.01F}};
    inverse_kinematics.attach_object(attachment);
    curobo::robot::LinkInertialUpdate payload;
    payload.link_name = "attached_object";
    payload.mass = 0.1F;
    payload.center_of_mass = std::array<float, 3>{0.0F, 0.0F, 0.1F};
    payload.inertia =
        std::array<float, 6>{0.001F, 0.001F, 0.001F, 0.0F, 0.0F, 0.0F};
    inverse_kinematics.update_link_inertial(payload);
    SolveBuffers solve_buffers(info.num_dof);
    auto request = solve_buffers.request(goal_position.get(), goal_quaternion.get());
    inverse_kinematics.solve(request);
    check_cuda(cudaDeviceSynchronize());
    require_success(solve_buffers, config.position_tolerance,
                    config.orientation_tolerance, "initial LM plus L-BFGS solve");
    inverse_kinematics.solve(request);
    check_cuda(cudaDeviceSynchronize());
    require_success(solve_buffers, config.position_tolerance,
                    config.orientation_tolerance, "LM plus L-BFGS graph replay");
    payload.mass = 1000.0F;
    payload.center_of_mass = std::array<float, 3>{0.5F, 0.0F, 0.1F};
    inverse_kinematics.update_link_inertial(payload);
    inverse_kinematics.solve(request);
    check_cuda(cudaDeviceSynchronize());
    require_failure(solve_buffers, "over-limit fixed-link payload");
    payload.mass = 0.1F;
    payload.center_of_mass = std::array<float, 3>{0.0F, 0.0F, 0.1F};
    inverse_kinematics.update_link_inertial(payload);
    inverse_kinematics.solve(request);
    check_cuda(cudaDeviceSynchronize());
    require_success(solve_buffers, config.position_tolerance,
                    config.orientation_tolerance,
                    "payload update with stable CUDA graphs");
    inverse_kinematics.detach_object();
    inverse_kinematics.solve(request);
    check_cuda(cudaDeviceSynchronize());
    require_success(solve_buffers, config.position_tolerance,
                    config.orientation_tolerance,
                    "post-detachment IK graph replay");

    SolveBuffers ranked_buffers(info.num_dof, 1, 2);
    auto ranked_request = ranked_buffers.request(
        goal_position.get(), goal_quaternion.get());
    ranked_request.current_positions = target_joint_device.get();
    inverse_kinematics.solve(ranked_request);
    check_cuda(cudaDeviceSynchronize());
    std::array<uint8_t, 2> ranked_success{};
    std::array<float, 14> ranked_solutions{};
    ranked_buffers.success.download(ranked_success.data());
    ranked_buffers.solution.download(ranked_solutions.data());
    if (ranked_success.at(0) != 1 || ranked_success.at(1) != 1) {
      throw std::runtime_error("IK did not return two feasible ranked solutions");
    }
    bool solutions_differ = false;
    for (int joint = 0; joint < info.num_dof; ++joint) {
      solutions_differ =
          solutions_differ ||
          std::abs(ranked_solutions.at(static_cast<size_t>(joint)) -
                   ranked_solutions.at(
                       static_cast<size_t>(info.num_dof + joint))) >
              1.0E-4F;
    }
    if (!solutions_differ) {
      throw std::runtime_error("IK returned the same candidate twice");
    }

    curobo::robot::InverseKinematicsConfig request_seed_config = config;
    request_seed_config.num_seeds = 1;
    request_seed_config.lbfgs_iterations = 1;
    request_seed_config.exit_early = false;
    request_seed_config.check_effort_limits = false;
    curobo::robot::InverseKinematics request_seed_ik(
        artifact_directory, request_seed_config);
    SolveBuffers request_seed_buffers(info.num_dof);
    auto request_seed_request = request_seed_buffers.request(
        goal_position.get(), goal_quaternion.get());
    request_seed_request.seed_positions = target_joint_device.get();
    request_seed_ik.solve(request_seed_request);
    check_cuda(cudaDeviceSynchronize());
    require_success(request_seed_buffers, request_seed_config.position_tolerance,
                    request_seed_config.orientation_tolerance,
                    "request-seeded LM plus L-BFGS solve");

    curobo::robot::InverseKinematics current_seed_ik(
        artifact_directory, request_seed_config);
    SolveBuffers current_seed_buffers(info.num_dof);
    auto current_seed_request = current_seed_buffers.request(
        goal_position.get(), goal_quaternion.get());
    current_seed_request.current_positions = target_joint_device.get();
    current_seed_ik.solve(current_seed_request);
    check_cuda(cudaDeviceSynchronize());
    require_success(current_seed_buffers,
                    request_seed_config.position_tolerance,
                    request_seed_config.orientation_tolerance,
                    "current-position-seeded LM plus L-BFGS solve");

    std::array<float, 3> reachable_position{};
    std::array<float, 4> reachable_quaternion{};
    goal_position.download(reachable_position.data());
    goal_quaternion.download(reachable_quaternion.data());
    const std::array<float, 6> goalset_positions{
        reachable_position.at(0) + 10.0F,
        reachable_position.at(1),
        reachable_position.at(2),
        reachable_position.at(0),
        reachable_position.at(1),
        reachable_position.at(2),
    };
    const std::array<float, 8> goalset_quaternions{
        reachable_quaternion.at(0), reachable_quaternion.at(1),
        reachable_quaternion.at(2), reachable_quaternion.at(3),
        reachable_quaternion.at(0), reachable_quaternion.at(1),
        reachable_quaternion.at(2), reachable_quaternion.at(3),
    };
    DeviceValue<float> goalset_position_device(goalset_positions.size());
    DeviceValue<float> goalset_quaternion_device(goalset_quaternions.size());
    DeviceValue<int32_t> goalset_index(1);
    goalset_position_device.upload(goalset_positions.data());
    goalset_quaternion_device.upload(goalset_quaternions.data());
    SolveBuffers goalset_buffers(info.num_dof);
    auto goalset_request = goalset_buffers.request(
        goalset_position_device.get(), goalset_quaternion_device.get());
    goalset_request.seed_positions = target_joint_device.get();
    goalset_request.goalset_index = goalset_index.get();
    goalset_request.num_goalset = 2;
    request_seed_ik.solve(goalset_request);
    check_cuda(cudaDeviceSynchronize());
    require_success(goalset_buffers, request_seed_config.position_tolerance,
                    request_seed_config.orientation_tolerance,
                    "goalset LM plus L-BFGS solve");
    int32_t selected_goalset_index = -1;
    goalset_index.download(&selected_goalset_index);
    if (selected_goalset_index != 1) {
      throw std::runtime_error("IK selected the wrong goalset entry");
    }

    const std::array<float, 14> batched_target_joint_position{
        0.4F, -0.5F, 0.3F, -1.7F, 0.2F, 1.4F, 0.6F,
        -0.3F, 0.25F, -0.4F, -2.1F, 0.35F, 1.8F, -0.2F};
    DeviceValue<float> batched_target_joint_device(
        batched_target_joint_position.size());
    batched_target_joint_device.upload(batched_target_joint_position.data());
    DeviceValue<float> batched_goal_position(6);
    DeviceValue<float> batched_goal_quaternion(8);
    DeviceValue<float> batched_goal_jacobian(
        static_cast<size_t>(2 * 6 * info.num_dof));
    kinematics.forward(
        batched_target_joint_device.get(), batched_goal_position.get(),
        batched_goal_quaternion.get(), batched_goal_jacobian.get(), 2, 1);
    curobo::robot::InverseKinematicsConfig batched_config = config;
    batched_config.max_batch_size = 2;
    curobo::robot::InverseKinematics batched_ik(
        artifact_directory, batched_config);
    SolveBuffers batched_buffers(info.num_dof, 2);
    auto batched_request = batched_buffers.request(
        batched_goal_position.get(), batched_goal_quaternion.get());
    batched_ik.solve(batched_request);
    check_cuda(cudaDeviceSynchronize());
    std::array<uint8_t, 2> batched_success{};
    std::array<float, 2> batched_position_error{};
    std::array<float, 2> batched_orientation_error{};
    batched_buffers.success.download(batched_success.data());
    batched_buffers.position_error.download(batched_position_error.data());
    batched_buffers.orientation_error.download(batched_orientation_error.data());
    for (size_t index = 0; index < batched_success.size(); ++index) {
      if (batched_success.at(index) != 1 ||
          batched_position_error.at(index) >= batched_config.position_tolerance ||
          batched_orientation_error.at(index) >=
              batched_config.orientation_tolerance) {
        throw std::runtime_error(
            "batched IK failed convergence: index=" + std::to_string(index) +
            " success=" + std::to_string(batched_success.at(index)) +
            " position_error=" +
            std::to_string(batched_position_error.at(index)) +
            " orientation_error=" +
            std::to_string(batched_orientation_error.at(index)));
      }
    }

    cudaStream_t graph_stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    auto external_graph_config = config;
    external_graph_config.use_cuda_graph = false;
    external_graph_config.exit_early = false;
    curobo::robot::InverseKinematics external_graph_ik(
        artifact_directory, external_graph_config);
    check_cuda(cudaStreamCreate(&graph_stream));
    request.stream = graph_stream;
    check_cuda(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal));
    external_graph_ik.solve(request);
    check_cuda(cudaStreamEndCapture(graph_stream, &graph));
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, 0));
    check_cuda(cudaGraphLaunch(graph_exec, graph_stream));
    check_cuda(cudaStreamSynchronize(graph_stream));
    require_success(solve_buffers, config.position_tolerance,
                    config.orientation_tolerance,
                    "external graph LM plus L-BFGS solve");
    check_cuda(cudaGraphExecDestroy(graph_exec));
    check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(graph_stream));

    curobo::robot::SceneConfig scene_config;
    scene_config.max_cuboids = 1;
    scene_config.max_capsules = 0;
    scene_config.max_cylinders = 0;
    scene_config.max_meshes = 0;
    curobo::robot::Scene scene(scene_config);
    curobo::robot::Cuboid blocker;
    blocker.name = "tool_goal_blocker";
    blocker.pose.position = reachable_position;
    blocker.dimensions = {4.0F, 4.0F, 4.0F};
    scene.add_cuboid(blocker);
    curobo::robot::InverseKinematics collision_ik(artifact_directory, config, &scene);
    SolveBuffers collision_buffers(info.num_dof);
    auto collision_request =
        collision_buffers.request(goal_position.get(), goal_quaternion.get());
    collision_ik.solve(collision_request);
    check_cuda(cudaDeviceSynchronize());
    uint8_t collision_success = 1;
    float collision_cost = 0.0F;
    collision_buffers.success.download(&collision_success);
    collision_buffers.collision_cost.download(&collision_cost);
    if (collision_success != 0 || !(collision_cost > 0.0F) ||
        !std::isfinite(collision_cost)) {
      throw std::runtime_error("IK did not reject the scene-colliding solution");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
