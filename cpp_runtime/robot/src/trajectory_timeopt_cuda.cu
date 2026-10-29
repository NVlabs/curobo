// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include "trajectory/timeopt/timeopt_common.cuh"
#include "trajectory/timeopt/timeopt_path_kernel.cuh"

// TimeOpt I/O requires the path helpers above. Preserve this include group.
#include "trajectory/timeopt/timeopt_io_kernel.cuh"
#include "trajectory/timeopt/timeopt_solver_kernel.cuh"
#include "trajectory_timeopt_cuda.h"

namespace curobo::robot::detail {
namespace rt = curobo::trajectory::timeopt;
static_assert(TIMEOPT_CONTROL_COUNT == rt::TIMEOPT_CONTROLS);

namespace {
__global__ void scale_states(TrajectoryTimeOptSamples samples, const float* scale,
                            size_t values, float* output) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= values) return;
  const float alpha = *scale;
  output[i] = samples.velocity[i] * alpha;
  output[values+i] = samples.acceleration[i] * alpha * alpha;
  output[2*values+i] = samples.jerk[i] * alpha * alpha * alpha;
}
__global__ void gather_spatial(
    const float* controls, const float* boundaries, const float* knot_dt,
    const int32_t* selected_seed, const uint8_t* success, int batch, int knots,
    int dof, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (knots + 6) * dof) return;
  const int seed = *selected_seed;
  if (!*success || seed < 0 || seed >= batch) {
    output[index] = 0.0F;
    return;
  }
  const int control = index / dof, joint = index % dof, plane = batch * dof;
  if (control < 4) {
    float fixed[4];
    const curobo::trajectory::bspline::BoundaryConstraint boundary(
        boundaries[seed * dof + joint], boundaries[2 * plane + seed * dof + joint],
        boundaries[3 * plane + seed * dof + joint], 0.0F);
    const float dt = knot_dt[seed];
    curobo::trajectory::bspline::compute_fixed_knots<3>(fixed, boundary, dt, dt * dt, dt * dt * dt);
    output[index] = fixed[control];
  } else if (control >= knots + 3) {
    output[index] = boundaries[plane + seed * dof + joint];
  } else {
    output[index] = controls[(seed * knots + control - 4) * dof + joint];
  }
}
__global__ void prepare_boundaries(
    const float* position, const float* dt, const float* velocity,
    const float* acceleration, int batch, int dof, int steps, int interpolation_steps,
    float* boundaries, float* knot_dt) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch * dof) return;
  const int seed = index / dof, joint = index % dof, plane = batch * dof;
  boundaries[index] = position[seed * steps * dof + joint];
  boundaries[plane + index] = position[(seed * steps + steps - 1) * dof + joint];
  boundaries[2 * plane + index] = velocity ? velocity[joint] : 0.0F;
  boundaries[3 * plane + index] = acceleration ? acceleration[joint] : 0.0F;
  if (joint == 0) knot_dt[seed] = dt[seed] * interpolation_steps;
}

__global__ void prepare_times(const float* duration, int batch, int capacity,
                              float interpolation_dt, float* times, float* dt,
                              int32_t* counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch * capacity) return;
  const int seed = index / capacity, sample = index % capacity;
  const float value = duration[seed];
  const float intervals = fmaxf(ceilf(value / interpolation_dt), 1.0F);
  const bool fits = isfinite(value) && value >= 0.0F && intervals < capacity;
  const int safe_intervals = fits ? static_cast<int>(intervals) : capacity - 1;
  const float sample_dt = isfinite(value) ? value / safe_intervals : 0.0F;
  times[index] = sample * sample_dt;
  if (sample == 0) {
    dt[seed] = sample_dt;
    counts[seed] = fits ? safe_intervals + 1 : 0;
  }
}

__global__ void rank_timeopt(
    const uint8_t* timeopt_success, const int32_t* violation, const int32_t* counts,
    const uint8_t* seed_valid, const uint8_t* pose_valid, const float* duration,
    int batch, uint8_t* success, float* rank) {
  const int seed = blockIdx.x * blockDim.x + threadIdx.x;
  if (seed >= batch) return;
  const bool valid = timeopt_success[seed] && violation[seed] == 0 && counts[seed] > 0 &&
                     seed_valid[seed] && pose_valid[seed] && isfinite(duration[seed]);
  success[seed] = valid;
  rank[seed] = valid ? duration[seed] : CUDART_INF_F;
}

__global__ void gather_timeopt(
    TrajectoryTimeOptSamples samples, const float* timing_controls,
    const float* dt, const float* duration,
    const int32_t* counts, const int32_t* selected_seed, const uint8_t* success,
    int capacity, int dof, float* states, float* metadata, int32_t* count,
    float* selected_timing_controls) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int values = capacity * dof;
  if (index >= values && index >= TIMEOPT_CONTROL_COUNT) return;
  const int seed = *selected_seed;
  const bool valid = *success && seed >= 0;
  if (index < TIMEOPT_CONTROL_COUNT) {
    selected_timing_controls[index] = valid
        ? timing_controls[seed * TIMEOPT_CONTROL_COUNT + index] : 0.0F;
  }
  if (index >= values) return;
  if (!valid) {
    states[index] = states[values + index] = 0.0F;
    states[2 * values + index] = states[3 * values + index] = 0.0F;
    if (index == 0) {
      metadata[0] = metadata[1] = 0.0F;
      *count = 0;
    }
    return;
  }
  const int source = seed * values + index;
  states[index] = samples.position[source];
  states[values + index] = samples.velocity[source];
  states[2 * values + index] = samples.acceleration[source];
  states[3 * values + index] = samples.jerk[source];
  if (index == 0) {
    metadata[0] = dt[seed];
    metadata[1] = duration[seed];
    *count = *success ? counts[seed] : 0;
  }
}
}  // namespace

cudaError_t launch_timeopt_boundaries(
    const float* position, const float* dt, const float* start_velocity,
    const float* start_acceleration, int batch, int dof, int steps,
    int interpolation_steps, float* boundaries, float* knot_dt, cudaStream_t stream) {
  prepare_boundaries<<<(batch * dof + 255) / 256, 256, 0, stream>>>(
      position, dt, start_velocity, start_acceleration, batch, dof, steps,
      interpolation_steps, boundaries, knot_dt);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_scale(TrajectoryTimeOptSamples samples,
    const float* scale, size_t values, float* output, cudaStream_t stream) {
  scale_states<<<(values+255)/256,256,0,stream>>>(samples,scale,values,output);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_spatial_output(
    const float* controls, const float* boundaries, const float* knot_dt,
    const int32_t* selected_seed, const uint8_t* success, int batch, int knots,
    int dof, float* output, cudaStream_t stream) {
  gather_spatial<<<((knots + 6) * dof + 255) / 256, 256, 0, stream>>>(
      controls, boundaries, knot_dt, selected_seed, success, batch, knots, dof, output);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_times(const float* duration, int batch, int capacity,
                                 float interpolation_dt, float* times, float* dt,
                                 int32_t* counts, cudaStream_t stream) {
  prepare_times<<<(batch * capacity + 255) / 256, 256, 0, stream>>>(
      duration, batch, capacity, interpolation_dt, times, dt, counts);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_rank(
    const uint8_t* timeopt_success, const int32_t* violation,
    const int32_t* counts, const uint8_t* seed_valid, const uint8_t* pose_valid,
    const float* duration, int batch, uint8_t* success, float* rank, cudaStream_t stream) {
  rank_timeopt<<<(batch + 255) / 256, 256, 0, stream>>>(
      timeopt_success, violation, counts, seed_valid, pose_valid, duration, batch, success, rank);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_output(
    TrajectoryTimeOptSamples samples, const float* timing_controls,
    const float* dt, const float* duration,
    const int32_t* counts, const int32_t* selected_seed, const uint8_t* success,
    int capacity, int dof, float* states, float* metadata, int32_t* count,
    float* selected_timing_controls, cudaStream_t stream) {
  const int values = std::max(capacity * dof, TIMEOPT_CONTROL_COUNT);
  gather_timeopt<<<(values + 255) / 256, 256, 0, stream>>>(
      samples, timing_controls, dt, duration, counts, selected_seed, success, capacity, dof,
      states, metadata, count, selected_timing_controls);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_path(const TimeOptWorkspace &w, cudaStream_t stream) {
  rt::fused_inputs_and_path<<<dim3(w.seed_tiles, w.batch), 256, 0, stream>>>(
      w.input_knots, w.input_start, w.input_goal, w.input_velocity, w.input_acceleration,
      w.input_jerk, w.input_dt, reinterpret_cast<float2 *>(w.reference_speed), w.controls, w.start,
      w.goal, w.input_valid, w.stationary, w.segments, w.local, w.path, w.q, w.qd, w.qdd, w.knots,
      w.dof, w.points, w.batch, w.start_offset);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_solve(const TimeOptWorkspace &w, cudaStream_t stream) {
  const auto *meta = reinterpret_cast<const int2 *>(w.meta);
  const auto *basis = reinterpret_cast<const float4 *>(w.basis);
  auto *rows = reinterpret_cast<float4 *>(w.rows);
  auto *jacobian = reinterpret_cast<float4 *>(w.jacobian);
  const dim3 grid(w.tiles, w.batch);
  rt::fused_coefficients_and_seed_bounds<<<dim3(w.seed_tiles, w.batch), 256, 0, stream>>>(
      w.path, w.tau, w.coefficients, w.vlim, w.alim, w.jlim, w.lower, w.upper, w.seed_bounds,
      w.points, w.dof, w.batch, w.seed_tiles);
  rt::fused_prepare<<<grid, 256, 0, stream>>>(
      meta, w.offsets, basis, w.path, w.coefficients, w.vlim, w.alim, w.jlim,
      w.lower, w.upper, w.seed_bounds, reinterpret_cast<const float2 *>(w.reference_speed), w.state,
      w.feasible, w.numerical, w.uniform, rows, w.bounds, w.points, w.dof, w.batch, w.tiles,
      w.seed_tiles);
  for (int iteration = 0; iteration < w.iterations; ++iteration) {
    rt::fused_assemble<<<grid, 256, 0, stream>>>(
        meta, w.offsets, basis, rows, w.bounds, w.state, w.row_state, jacobian, w.partials,
        w.points, w.intervals + 1, w.dof, w.batch, w.tiles, static_cast<int>(iteration == 0));
    rt::fused_factor_reduce<<<dim3(21, w.batch), 256, 0, stream>>>(w.partials, w.offsets, w.starts,
                                                                   w.state, w.tiles);
    rt::fused_factor<<<w.batch, 32, 0, stream>>>(w.state, w.numerical);
    rt::fused_affine<<<grid, 256, 0, stream>>>(meta, w.offsets, rows, jacobian, w.state,
                                               w.row_state, w.partials, w.batch, w.tiles);
    rt::fused_correct_reduce<<<dim3(5, w.batch), 256, 0, stream>>>(
        w.partials, w.offsets, w.starts, w.corrector_terms, w.state, w.tiles,
        7 * w.points * w.dof + 32);
    rt::fused_correct<<<w.batch, 32, 0, stream>>>(w.corrector_terms, w.state, w.numerical);
    rt::fused_bounds<<<grid, 256, 0, stream>>>(meta, w.offsets, rows, jacobian, w.state,
                                               w.row_state, w.step_bounds, w.batch, w.tiles);
    rt::fused_search_bounds<<<w.batch, 256, 0, stream>>>(w.step_bounds, w.state, w.tiles);
    rt::fused_search<2><<<dim3(4 * w.tiles, w.batch), 256, 0, stream>>>(
        meta, w.offsets, basis, rows, w.bounds, w.state, w.candidate_masks, w.points, w.dof,
        w.batch, w.tiles);
    rt::fused_accept<<<w.batch, 256, 0, stream>>>(w.candidate_masks, basis, w.point_first,
                                                  w.state, w.timing, 4 * w.tiles,
                                                  w.intervals + 1);
  }
  rt::fused_output<<<dim3(w.seed_tiles, w.batch), 256, 0, stream>>>(
      w.timing, basis, w.point_first, w.path, w.coefficients, w.vlim, w.alim, w.jlim, w.lower,
      w.upper, reinterpret_cast<const float4 *>(w.time_basis), w.time_first, w.time_weights,
      w.output_bounds, w.dt, w.output_joints, w.output_torque, w.points, w.dof, w.batch,
      w.intervals, w.seed_tiles);
  rt::fused_finish<<<w.batch, 256, 0, stream>>>(
      w.output_bounds, w.feasible, w.uniform, w.numerical, w.input_valid, w.stationary,
      w.output_uniform, w.output_numerical, w.success, w.status, w.timing, w.dt, w.output_timing,
      w.output_joints, w.output_torque, w.output_time, w.intervals, w.dof, w.batch, w.seed_tiles);
  return cudaGetLastError();
}

cudaError_t launch_timeopt_sample(const TimeOptWorkspace &w, cudaStream_t stream) {
  rt::sample_kernel<<<(w.batch * w.samples + 255) / 256, 256, 0, stream>>>(
      w.controls, w.start, w.goal, w.output_timing, w.polynomial, w.nodes, w.time_weights,
      w.output_time, w.sample_times, w.success, w.stationary, w.sample_joints, w.batch, w.samples,
      w.dof, w.knots, w.intervals, w.start_offset);
  return cudaGetLastError();
}
}  // namespace curobo::robot::detail
