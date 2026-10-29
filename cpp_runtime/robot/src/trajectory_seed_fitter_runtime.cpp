// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "trajectory_seed_fitter_runtime.h"

#include "cuda_check.h"
#include "trajectory_seed_fitter_cuda.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace curobo::robot::detail {
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

void load_kernel(const std::string& artifact_directory,
                 const curobo_robot_runtime::IkKernelManifest& manifest,
                 curobo_robot_runtime::KernelModule* kernel) {
  kernel->load(
      (std::filesystem::path(artifact_directory) / manifest.cubin).string(),
      manifest.symbol);
}

}  // namespace

TrajectorySeedFitterRuntime::TrajectorySeedFitterRuntime(
    std::string artifact_directory, const TrajectorySeedFitterConfig& config)
    : artifact_directory_(std::move(artifact_directory)),
      config_(config),
      manifest_(curobo_robot_runtime::load_motion_planning_manifest(
          artifact_directory_)),
      trajopt_manifest_(
          curobo_robot_runtime::load_trajectory_optimization_manifest(
              artifact_directory_)) {
  validate_artifacts();
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  load_constants();
  load_kernels();
  allocate_workspace();
}

TrajectorySeedFitterRuntime::~TrajectorySeedFitterRuntime() {
  cudaSetDevice(config_.device_ordinal);
  destroy_graph();
}

CuroboRobotInfo TrajectorySeedFitterRuntime::robot_info() const {
  return CuroboRobotInfo{trajopt_manifest_.num_links,
                         trajopt_manifest_.num_dof,
                         trajopt_manifest_.num_tool_frames};
}

TrajectorySeedFitterInfo TrajectorySeedFitterRuntime::info() const {
  return TrajectorySeedFitterInfo{manifest_.num_dof, manifest_.num_knots,
                                  manifest_.trajectory_steps,
                                  manifest_.irls_iterations};
}

void TrajectorySeedFitterRuntime::validate_artifacts() const {
  if (config_.num_seeds <= 0 ||
      config_.num_seeds > manifest_.max_trajopt_seeds) {
    throw std::invalid_argument(
        "seed-fitter num_seeds exceeds the artifact capacity");
  }
  if (manifest_.arch != trajopt_manifest_.arch ||
      manifest_.num_dof != trajopt_manifest_.num_dof ||
      manifest_.num_knots != trajopt_manifest_.num_knots ||
      manifest_.trajectory_steps != trajopt_manifest_.trajectory_steps ||
      config_.num_seeds > trajopt_manifest_.max_seeds) {
    throw std::runtime_error(
        "seed-fitter and TrajOpt artifact metadata do not match");
  }
}

void TrajectorySeedFitterRuntime::load_constants() {
  const auto load = [&](const curobo_robot_runtime::TensorManifest& tensor,
                        const char* name,
                        curobo_robot_runtime::DeviceBuffer* destination) {
    destination->upload(curobo_robot_runtime::load_tensor_data(
        artifact_directory_, tensor, name));
  };
  load(manifest_.arc_progress, "seed fitter arc_progress", &arc_progress_);
  load(manifest_.fit_matrix, "seed fitter fit_matrix", &fit_matrix_);
  load(manifest_.start_coefficient, "seed fitter start coefficient",
       &start_position_coefficient_);
  load(manifest_.start_velocity_coefficient,
       "seed fitter start velocity coefficient",
       &start_velocity_coefficient_);
  load(manifest_.start_acceleration_coefficient,
       "seed fitter start acceleration coefficient",
       &start_acceleration_coefficient_);
  load(manifest_.start_jerk_coefficient,
       "seed fitter start jerk coefficient", &start_jerk_coefficient_);
  load(manifest_.goal_coefficient, "seed fitter goal coefficient",
       &goal_position_coefficient_);
  load(manifest_.basis_position, "seed fitter position basis",
       &basis_position_);
  load(manifest_.basis_velocity, "seed fitter velocity basis",
       &basis_velocity_);
  load(manifest_.basis_acceleration, "seed fitter acceleration basis",
       &basis_acceleration_);
  load(manifest_.basis_jerk, "seed fitter jerk basis", &basis_jerk_);
  load(manifest_.path_normal, "seed fitter path normal", &path_normal_);
  load(manifest_.max_velocity, "seed fitter velocity limit", &max_velocity_);
  load(manifest_.max_acceleration, "seed fitter acceleration limit",
       &max_acceleration_);
  load(manifest_.max_jerk, "seed fitter jerk limit", &max_jerk_);
}

void TrajectorySeedFitterRuntime::load_kernels() {
  load_kernel(artifact_directory_, trajopt_manifest_.bspline_forward,
              &bspline_forward_kernel_);
}

void TrajectorySeedFitterRuntime::allocate_workspace() {
  const auto seeds = static_cast<size_t>(config_.num_seeds);
  const auto dof = static_cast<size_t>(manifest_.num_dof);
  const auto knots = static_cast<size_t>(manifest_.num_knots);
  const auto steps = static_cast<size_t>(manifest_.trajectory_steps);
  const auto state_values = checked_product(
      checked_product(seeds, steps, "seed fitter states"), dof,
      "seed fitter state values");
  const auto knot_values = checked_product(
      checked_product(seeds, knots, "seed fitter knots"), dof,
      "seed fitter knot values");
  const auto state_bytes = checked_product(state_values, sizeof(float),
                                           "seed fitter state bytes");
  const auto knot_bytes = checked_product(knot_values, sizeof(float),
                                          "seed fitter knot bytes");
  zero_controls_.allocate(knot_bytes);
  zero_controls_.clear();
  zero_state_.allocate(seeds * dof * sizeof(float));
  zero_state_.clear();
  std::vector<int32_t> lane_indices(seeds);
  for (size_t index = 0; index < seeds; ++index) {
    lane_indices.at(index) = static_cast<int32_t>(index);
  }
  lane_indices_.upload(std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(lane_indices.data()),
      reinterpret_cast<const uint8_t*>(lane_indices.data()) +
          lane_indices.size() * sizeof(int32_t)));
  implicit_goal_.upload(std::vector<uint8_t>(seeds, 1U));
  const std::vector<float> trajectory_dt(seeds, manifest_.trajectory_dt);
  trajectory_dt_.upload(std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(trajectory_dt.data()),
      reinterpret_cast<const uint8_t*>(trajectory_dt.data()) +
          trajectory_dt.size() * sizeof(float)));
  dense_dt_.allocate(seeds * sizeof(float));
  arc_target_.allocate(state_bytes);
  position_residual_.allocate(state_bytes);
  boundary_position_.allocate(state_bytes);
  boundary_velocity_.allocate(state_bytes);
  boundary_acceleration_.allocate(state_bytes);
  boundary_jerk_.allocate(state_bytes);
  current_knots_.allocate(knot_bytes);
  current_position_.allocate(state_bytes);
  current_velocity_.allocate(state_bytes);
  current_acceleration_.allocate(state_bytes);
  current_jerk_.allocate(state_bytes);
  current_required_dt_.allocate(seeds * sizeof(float));
  candidate_knots_.allocate(knot_bytes);
  candidate_position_.allocate(state_bytes);
  candidate_velocity_.allocate(state_bytes);
  candidate_acceleration_.allocate(state_bytes);
  candidate_jerk_.allocate(state_bytes);
  candidate_required_dt_.allocate(seeds * sizeof(float));
  derivative_scale_.allocate(seeds * sizeof(float));
  velocity_trace_workspace_.allocate(state_bytes);
  acceleration_trace_workspace_.allocate(state_bytes);
  jerk_trace_workspace_.allocate(state_bytes);
}

void TrajectorySeedFitterRuntime::launch_bspline(
    CUdeviceptr knots, CUdeviceptr start_position, CUdeviceptr start_velocity,
    CUdeviceptr start_acceleration, CUdeviceptr start_jerk,
    CUdeviceptr goal_position, CUdeviceptr out_position,
    CUdeviceptr out_velocity, CUdeviceptr out_acceleration,
    CUdeviceptr out_jerk, CUstream stream) const {
  CUdeviceptr output_dt = dense_dt_.ptr();
  CUdeviceptr zero = zero_state_.ptr();
  CUdeviceptr indices = lane_indices_.ptr();
  CUdeviceptr trajectory_dt = trajectory_dt_.ptr();
  CUdeviceptr implicit_goal = implicit_goal_.ptr();
  int batch_size = config_.num_seeds;
  int horizon = manifest_.trajectory_steps;
  int dof = manifest_.num_dof;
  int n_knots = manifest_.num_knots;
  std::vector<void*> arguments{
      &out_position,       &out_velocity, &out_acceleration, &out_jerk,
      &output_dt,          &knots,        &start_position,   &start_velocity,
      &start_acceleration, &start_jerk,   &goal_position,    &zero,
      &zero,               &zero,         &indices,          &indices,
      &trajectory_dt,      &implicit_goal, &batch_size,       &horizon,
      &dof,                &n_knots,
  };
  const size_t elements = static_cast<size_t>(batch_size) *
                          static_cast<size_t>(horizon) *
                          static_cast<size_t>(dof);
  bspline_forward_kernel_.launch(
      grid_for(elements, trajopt_manifest_.bspline_forward.block_dim),
      static_cast<unsigned int>(trajopt_manifest_.bspline_forward.block_dim),
      static_cast<unsigned int>(
          trajopt_manifest_.bspline_forward.shared_memory),
      stream, arguments);
}

void TrajectorySeedFitterRuntime::fit_impl(
    const TrajectorySeedFitterRequest& request) const {
  if (request.target_position == nullptr || request.start_position == nullptr ||
      request.goal_position == nullptr || request.fitted_knots == nullptr ||
      request.required_dt == nullptr) {
    throw std::invalid_argument(
        "seed-fitter input and output CUDA addresses must be non-null");
  }
  const auto stream = reinterpret_cast<CUstream>(request.stream);
  const int seeds = config_.num_seeds;
  const int dof = manifest_.num_dof;
  const int knots = manifest_.num_knots;
  const int steps = manifest_.trajectory_steps;
  const int knot_variables = knots * dof;
  const CUdeviceptr target =
      reinterpret_cast<CUdeviceptr>(request.target_position);
  const CUdeviceptr start =
      reinterpret_cast<CUdeviceptr>(request.start_position);
  const CUdeviceptr goal =
      reinterpret_cast<CUdeviceptr>(request.goal_position);
  const CUdeviceptr zero = zero_state_.ptr();
  const CUdeviceptr start_velocity =
      request.start_velocity == nullptr
          ? zero
          : reinterpret_cast<CUdeviceptr>(request.start_velocity);
  const CUdeviceptr start_acceleration =
      request.start_acceleration == nullptr
          ? zero
          : reinterpret_cast<CUdeviceptr>(request.start_acceleration);
  const CUdeviceptr start_jerk =
      request.start_jerk == nullptr
          ? zero
          : reinterpret_cast<CUdeviceptr>(request.start_jerk);

  CRR_CUDA_CHECK(launch_seed_fitter_resample_dense(
      reinterpret_cast<const float*>(target),
      reinterpret_cast<const float*>(arc_progress_.ptr()),
      reinterpret_cast<float*>(arc_target_.ptr()), seeds, steps, dof,
      request.stream));

  launch_bspline(zero_controls_.ptr(), start, start_velocity,
                 start_acceleration, start_jerk, goal,
                 boundary_position_.ptr(), boundary_velocity_.ptr(),
                 boundary_acceleration_.ptr(), boundary_jerk_.ptr(), stream);

  CRR_CUDA_CHECK(launch_seed_fitter_initialize_fit(
      reinterpret_cast<const float*>(arc_target_.ptr()),
      reinterpret_cast<const float*>(boundary_position_.ptr()),
      reinterpret_cast<const float*>(start),
      reinterpret_cast<const float*>(start_velocity),
      reinterpret_cast<const float*>(start_acceleration),
      reinterpret_cast<const float*>(start_jerk),
      reinterpret_cast<const float*>(goal),
      reinterpret_cast<const float*>(fit_matrix_.ptr()),
      reinterpret_cast<const float*>(start_position_coefficient_.ptr()),
      reinterpret_cast<const float*>(start_velocity_coefficient_.ptr()),
      reinterpret_cast<const float*>(start_acceleration_coefficient_.ptr()),
      reinterpret_cast<const float*>(start_jerk_coefficient_.ptr()),
      reinterpret_cast<const float*>(goal_position_coefficient_.ptr()),
      reinterpret_cast<float*>(position_residual_.ptr()),
      reinterpret_cast<float*>(current_knots_.ptr()), seeds, steps, dof,
      knots, request.stream));

  launch_bspline(current_knots_.ptr(), start, start_velocity,
                 start_acceleration, start_jerk, goal,
                 current_position_.ptr(), current_velocity_.ptr(),
                 current_acceleration_.ptr(), current_jerk_.ptr(), stream);

  const auto launch_required_dt = [&](CUdeviceptr velocity,
                                      CUdeviceptr acceleration,
                                      CUdeviceptr jerk,
                                      CUdeviceptr output) {
    CRR_CUDA_CHECK(launch_seed_fitter_required_dt(
        reinterpret_cast<const float*>(velocity),
        reinterpret_cast<const float*>(acceleration),
        reinterpret_cast<const float*>(jerk),
        reinterpret_cast<const float*>(max_velocity_.ptr()),
        reinterpret_cast<const float*>(max_acceleration_.ptr()),
        reinterpret_cast<const float*>(max_jerk_.ptr()), seeds, steps, dof,
        manifest_.trajectory_dt, manifest_.dt_epsilon,
        reinterpret_cast<float*>(output), request.stream));
  };
  launch_required_dt(current_velocity_.ptr(), current_acceleration_.ptr(),
                     current_jerk_.ptr(), current_required_dt_.ptr());

  for (int iteration = 0; iteration < manifest_.irls_iterations; ++iteration) {
    CRR_CUDA_CHECK(launch_seed_fitter_derivative_scale(
        reinterpret_cast<const float*>(current_velocity_.ptr()),
        reinterpret_cast<const float*>(current_acceleration_.ptr()),
        reinterpret_cast<const float*>(max_velocity_.ptr()),
        reinterpret_cast<const float*>(max_acceleration_.ptr()),
        reinterpret_cast<const float*>(max_jerk_.ptr()),
        reinterpret_cast<const float*>(basis_velocity_.ptr()),
        reinterpret_cast<const float*>(basis_acceleration_.ptr()),
        reinterpret_cast<const float*>(basis_jerk_.ptr()), seeds, steps, dof,
        knots, manifest_.path_normal_trace, manifest_.timing_regularization,
        manifest_.irls_epsilon,
        reinterpret_cast<float*>(velocity_trace_workspace_.ptr()),
        reinterpret_cast<float*>(acceleration_trace_workspace_.ptr()),
        reinterpret_cast<float*>(jerk_trace_workspace_.ptr()),
        reinterpret_cast<float*>(derivative_scale_.ptr()), request.stream));

    CRR_CUDA_CHECK(launch_seed_fitter_solve_irls(
        reinterpret_cast<const float*>(position_residual_.ptr()),
        reinterpret_cast<const float*>(boundary_velocity_.ptr()),
        reinterpret_cast<const float*>(boundary_acceleration_.ptr()),
        reinterpret_cast<const float*>(boundary_jerk_.ptr()),
        reinterpret_cast<const float*>(current_velocity_.ptr()),
        reinterpret_cast<const float*>(current_acceleration_.ptr()),
        reinterpret_cast<const float*>(current_knots_.ptr()),
        reinterpret_cast<const float*>(goal),
        reinterpret_cast<const float*>(max_velocity_.ptr()),
        reinterpret_cast<const float*>(max_acceleration_.ptr()),
        reinterpret_cast<const float*>(max_jerk_.ptr()),
        reinterpret_cast<const float*>(basis_position_.ptr()),
        reinterpret_cast<const float*>(basis_velocity_.ptr()),
        reinterpret_cast<const float*>(basis_acceleration_.ptr()),
        reinterpret_cast<const float*>(basis_jerk_.ptr()),
        reinterpret_cast<const float*>(path_normal_.ptr()),
        reinterpret_cast<const float*>(derivative_scale_.ptr()), seeds, steps,
        dof, knots, manifest_.regularization,
        manifest_.trust_region_regularization, manifest_.path_diagonal_scale,
        manifest_.irls_epsilon,
        reinterpret_cast<float*>(candidate_knots_.ptr()), request.stream));

    launch_bspline(candidate_knots_.ptr(), start, start_velocity,
                   start_acceleration, start_jerk, goal,
                   candidate_position_.ptr(), candidate_velocity_.ptr(),
                   candidate_acceleration_.ptr(), candidate_jerk_.ptr(),
                   stream);
    launch_required_dt(candidate_velocity_.ptr(),
                       candidate_acceleration_.ptr(), candidate_jerk_.ptr(),
                       candidate_required_dt_.ptr());

    CRR_CUDA_CHECK(launch_seed_fitter_accept_candidate(
        reinterpret_cast<const float*>(candidate_knots_.ptr()),
        reinterpret_cast<const float*>(candidate_velocity_.ptr()),
        reinterpret_cast<const float*>(candidate_acceleration_.ptr()),
        reinterpret_cast<const float*>(candidate_jerk_.ptr()),
        reinterpret_cast<const float*>(candidate_required_dt_.ptr()),
        reinterpret_cast<float*>(current_knots_.ptr()),
        reinterpret_cast<float*>(current_velocity_.ptr()),
        reinterpret_cast<float*>(current_acceleration_.ptr()),
        reinterpret_cast<float*>(current_jerk_.ptr()),
        reinterpret_cast<float*>(current_required_dt_.ptr()), seeds, steps,
        dof, knots, request.stream));
  }

  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.fitted_knots,
      reinterpret_cast<const void*>(current_knots_.ptr()),
      static_cast<size_t>(seeds) * knot_variables * sizeof(float),
      cudaMemcpyDeviceToDevice, request.stream));
  CRR_CUDA_CHECK(cudaMemcpyAsync(
      request.required_dt,
      reinterpret_cast<const void*>(current_required_dt_.ptr()),
      static_cast<size_t>(seeds) * sizeof(float), cudaMemcpyDeviceToDevice,
      request.stream));
}

bool TrajectorySeedFitterRuntime::same_graph_request(
    const TrajectorySeedFitterRequest& left,
    const TrajectorySeedFitterRequest& right) {
  return left.target_position == right.target_position &&
         left.start_position == right.start_position &&
         left.start_velocity == right.start_velocity &&
         left.start_acceleration == right.start_acceleration &&
         left.start_jerk == right.start_jerk &&
         left.goal_position == right.goal_position &&
         left.fitted_knots == right.fitted_knots &&
         left.required_dt == right.required_dt && left.stream == right.stream;
}

void TrajectorySeedFitterRuntime::destroy_graph() noexcept {
  if (graph_executable_ != nullptr) {
    cudaGraphExecDestroy(graph_executable_);
  }
  if (graph_ != nullptr) {
    cudaGraphDestroy(graph_);
  }
  graph_ = nullptr;
  graph_executable_ = nullptr;
  graph_request_ = {};
}

void TrajectorySeedFitterRuntime::fit(
    const TrajectorySeedFitterRequest& request) const {
  CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal));
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (request.stream != nullptr) {
    CRR_CUDA_CHECK(cudaStreamIsCapturing(request.stream, &capture));
  }
  if (!config_.use_cuda_graph || request.stream == nullptr ||
      capture != cudaStreamCaptureStatusNone) {
    fit_impl(request);
    return;
  }
  if (graph_executable_ != nullptr &&
      same_graph_request(graph_request_, request)) {
    CRR_CUDA_CHECK(cudaGraphLaunch(graph_executable_, request.stream));
    return;
  }

  const_cast<TrajectorySeedFitterRuntime*>(this)->destroy_graph();
  CRR_CUDA_CHECK(cudaStreamSynchronize(request.stream));
  CRR_CUDA_CHECK(cudaStreamBeginCapture(request.stream,
                                        cudaStreamCaptureModeThreadLocal));
  try {
    fit_impl(request);
    CRR_CUDA_CHECK(cudaStreamEndCapture(request.stream, &graph_));
    CRR_CUDA_CHECK(cudaGraphInstantiate(&graph_executable_, graph_, 0));
  } catch (...) {
    cudaGraph_t abandoned_graph = nullptr;
    cudaStreamEndCapture(request.stream, &abandoned_graph);
    if (abandoned_graph != nullptr) {
      cudaGraphDestroy(abandoned_graph);
    }
    const_cast<TrajectorySeedFitterRuntime*>(this)->destroy_graph();
    throw;
  }
  graph_request_ = request;
  CRR_CUDA_CHECK(cudaGraphLaunch(graph_executable_, request.stream));
}

}  // namespace curobo::robot::detail
