// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include "cuda_check.h"
#include "curobo_robot_runtime/spline_trajectory.h"
#include "spline_trajectory_internal.h"
#include "trajectory/timeopt/trajectory_spline_kernel.cuh"

namespace curobo::robot {
namespace {
__global__ void validate_spline(const float *controls, int count, int dof, const float *timing,
                                const float *time, int time_points, float duration, int *status) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  bool invalid = false;
  if (i < count * dof) {
    invalid = !isfinite(controls[i]) || (duration == 0 && controls[i] != controls[i % dof]);
  }
  if (timing && i < 32) invalid |= !isfinite(timing[i]) || timing[i] < 0;
  if (timing && i == 0 && duration > 0) {
    bool positive = false;
    for (int k = 0; k < 32; ++k) positive |= timing[k] > 0;
    invalid |= !positive;
  }
  if (time && i < time_points)
    invalid |= !isfinite(time[i]) || (i == 0 ? time[i] != 0 : time[i] < time[i - 1]);
  if (invalid) atomicOr(status, 1);
}
struct DeviceScope {
  int previous{-1};
  explicit DeviceScope(int device) {
    if (device < 0) return;
    CRR_CUDA_CHECK(cudaGetDevice(&previous));
    if (device >= 0) CRR_CUDA_CHECK(cudaSetDevice(device));
  }
  ~DeviceScope() {
    if (previous >= 0) cudaSetDevice(previous);
  }
};
void checked_stream(cudaStream_t stream) {
  cudaStreamCaptureStatus capture;
  CRR_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::invalid_argument("Checked spline operations must run outside capture");
}
std::shared_ptr<void> allocate(size_t count, int device) {
  if (device < 0)
    return std::shared_ptr<void>(new float[count],
                                 [](void *p) { delete[] static_cast<float *>(p); });
  DeviceScope scope(device);
  void *p;
  CRR_CUDA_CHECK(cudaMalloc(&p, count * sizeof(float)));
  return std::shared_ptr<void>(p, [device](void *pointer) {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaFree(pointer);
    cudaSetDevice(previous);
  });
}
void copy(float *dst, int target, const float *src, int source, size_t count, cudaStream_t stream) {
  if (source < 0 && target < 0) {
    std::copy(src, src + count, dst);
    return;
  }
  CRR_CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDefault, stream));
}
float scalar(const float *p, int device, cudaStream_t stream) {
  float result;
  copy(&result, -1, p, device, 1, stream);
  if (device >= 0) CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return result;
}
struct Tables {
  std::shared_ptr<void> storage;
  const float *polynomial, *nodes, *weights;
};
Tables tables(int device) {
  static std::mutex mutex;
  static std::map<int, Tables> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto found = cache.find(device);
  if (found != cache.end()) return found->second;
  std::vector<float> values(29 * 4 * 32 + 12, 0.0F);
  std::array<double, 36> knots{};
  for (int i = 0; i < 36; ++i) knots[i] = std::clamp((i - 3) / 29.0, 0.0, 1.0);
  // Expand Cox-de Boor in the local coordinate of each clamped timing span.
  for (int span = 0; span < 29; ++span) {
    double basis[35][4]{};
    basis[span + 3][0] = 1;
    const double left = span / 29.0;
    for (int degree = 1; degree <= 3; ++degree) {
      double next[35][4]{};
      for (int i = 0; i < 35 - degree; ++i) {
        const double a = knots[i + degree] - knots[i];
        const double b = knots[i + degree + 1] - knots[i + 1];
        for (int power = 0; power <= degree; ++power) {
          if (a > 0)
            next[i][power] +=
                (left - knots[i]) / a * basis[i][power] + (power ? basis[i][power - 1] / a : 0);
          if (b > 0)
            next[i][power] += (knots[i + degree + 1] - left) / b * basis[i + 1][power] -
                              (power ? basis[i + 1][power - 1] / b : 0);
        }
      }
      std::copy(&next[0][0], &next[0][0] + 35 * 4, &basis[0][0]);
    }
    for (int power = 0; power < 4; ++power)
      for (int i = 0; i < 32; ++i)
        values[(span * 4 + power) * 32 + i] = static_cast<float>(basis[i][3 - power]);
  }
  constexpr float nodes[6] = {.0337652429F, .1693953068F, .380690407F,
                              .619309593F,  .830604693F,  .966234757F};
  constexpr float weights[6] = {.0856622462F, .180380787F, .233956967F,
                                .233956967F,  .180380787F, .0856622462F};
  const int n = 29 * 4 * 32;
  std::copy(nodes, nodes + 6, values.begin() + n);
  std::copy(weights, weights + 6, values.begin() + n + 6);
  auto storage = allocate(values.size(), device);
  CRR_CUDA_CHECK(cudaMemcpy(storage.get(), values.data(), values.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  const auto *base = static_cast<float *>(storage.get());
  Tables t{storage, base, base + n, base + n + 6};
  cache.emplace(device, t);
  return t;
}
}  // namespace
SplineTrajectory::SplineTrajectory(const float *controls, int count, std::vector<std::string> names,
                                   Timing timing, int device, std::shared_ptr<void> owner)
    : controls_(controls),
      count_(count),
      device_(device),
      names_(std::move(names)),
      timing_(timing),
      owner_(std::move(owner)) {
  if (!controls || count < 4 || names_.empty() || device < -1 ||
      std::set<std::string>(names_.begin(), names_.end()).size() != names_.size() ||
      std::any_of(
          names_.begin(), names_.end(), [](const auto &name) { return name.empty(); }) ||
      static_cast<size_t>(count) * names_.size() > std::numeric_limits<int>::max())
    throw std::invalid_argument("Invalid complete cubic spline layout");
  if (const auto *uniform = std::get_if<UniformTiming>(&timing_)) {
    if (!uniform->knot_dt) throw std::invalid_argument("Uniform timing requires knot_dt");
  } else {
    const auto &t = std::get<TimeOptTiming>(timing_);
    if (!t.controls || !t.time || t.time_points < 2)
      throw std::invalid_argument("TimeOpt requires 32 controls and a time table");
  }
  if (device >= 0) {
    DeviceScope scope(device);
    const auto check = [device](const float *pointer) {
      cudaPointerAttributes attr{};
      CRR_CUDA_CHECK(cudaPointerGetAttributes(&attr, pointer));
      if (attr.type != cudaMemoryTypeDevice || attr.device != device)
        throw std::invalid_argument("Spline buffers must share the declared CUDA device");
    };
    check(controls_);
    if (const auto *uniform = std::get_if<UniformTiming>(&timing_))
      check(uniform->knot_dt);
    else {
      const auto &timing = std::get<TimeOptTiming>(timing_);
      check(timing.controls);
      check(timing.time);
    }
  }
}
float SplineTrajectory::duration(cudaStream_t stream) const {
  DeviceScope scope(device_);
  if (device_ >= 0) checked_stream(stream);
  if (const auto *t = std::get_if<UniformTiming>(&timing_)) {
    const float dt = scalar(t->knot_dt, device_, stream);
    if (!std::isfinite(dt) || dt <= 0) throw std::invalid_argument("knot_dt must be positive");
    return dt * (count_ - 3);
  }
  const auto &t = std::get<TimeOptTiming>(timing_);
  return scalar(t.time + t.time_points - 1, device_, stream);
}
SplineTrajectory SplineTrajectory::copy_to(int device, cudaStream_t stream) const {
  DeviceScope scope(device_ >= 0 ? device_ : device);
  if (device_ >= 0 || device >= 0) checked_stream(stream);
  if (device < -1) throw std::invalid_argument("Invalid target device");
  const auto *timing = std::get_if<TimeOptTiming>(&timing_);
  const size_t points = static_cast<size_t>(count_) * num_dof();
  auto storage = allocate(points + (timing ? 32 + timing->time_points : 1), device);
  auto *target = static_cast<float *>(storage.get());
  copy(target, device, controls_, device_, points, stream);
  Timing out;
  if (timing) {
    copy(target + points, device, timing->controls, device_, 32, stream);
    copy(target + points + 32, device, timing->time, device_, timing->time_points, stream);
    out = TimeOptTiming{target + points, target + points + 32, timing->time_points};
  } else {
    copy(target + points, device, std::get<UniformTiming>(timing_).knot_dt, device_, 1, stream);
    out = UniformTiming{target + points};
  }
  if (device_ >= 0 || device >= 0) CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return SplineTrajectory(target, count_, names_, out, device, storage);
}
SplineTrajectory SplineTrajectory::clone(cudaStream_t stream) const {
  return copy_to(device_, stream);
}
SplineTrajectory SplineTrajectory::to(int device, cudaStream_t stream) const {
  return device == device_ ? *this : copy_to(device, stream);
}
InterpolatedTrajectory SplineTrajectory::interpolate(float dt, cudaStream_t stream) const {
  if (!std::isfinite(dt) || dt <= 0) throw std::invalid_argument("dt must be finite and positive");
  if (device_ < 0) throw std::invalid_argument("Interpolation requires CUDA");
  DeviceScope scope(device_);
  checked_stream(stream);
  const float end = duration(stream);
  if (!std::isfinite(end) || end < 0) throw std::invalid_argument("Invalid duration");
  const double intervals = end > 0 ? std::ceil(static_cast<double>(end) / dt) : 0;
  if ((intervals + 1) * 4 * num_dof() > std::numeric_limits<int>::max())
    throw std::invalid_argument("Too many samples");
  const int count = static_cast<int>(intervals) + 1;
  const float actual = intervals ? static_cast<float>(end / intervals) : 0;
  const auto plane = static_cast<size_t>(count) * num_dof();
  auto storage = allocate(4 * plane, device_);
  auto *out = static_cast<float *>(storage.get());
  const auto table = tables(device_);
  const auto *timing = std::get_if<TimeOptTiming>(&timing_);
  auto *status = reinterpret_cast<int *>(out);
  CRR_CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(int), stream));
  const int checked_values =
      std::max(count_ * num_dof(), timing ? std::max(32, timing->time_points) : 0);
  validate_spline<<<(checked_values + 255) / 256, 256, 0, stream>>>(
      controls_, count_, num_dof(), timing ? timing->controls : nullptr,
      timing ? timing->time : nullptr, timing ? timing->time_points : 0, end, status);
  CRR_CUDA_CHECK(cudaGetLastError());
  int invalid = 0;
  CRR_CUDA_CHECK(cudaMemcpyAsync(&invalid, status, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (invalid) throw std::invalid_argument("Invalid spatial controls or physical time law");
  trajectory::timeopt::interpolate_spline_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      controls_, timing ? nullptr : std::get<UniformTiming>(timing_).knot_dt,
      timing ? timing->controls : nullptr, timing ? timing->time : nullptr, table.polynomial,
      table.nodes, table.weights, out, count_, num_dof(), count, timing ? timing->time_points : 0,
      actual);
  CRR_CUDA_CHECK(cudaGetLastError());
  return {storage, out,       out + plane, out + 2 * plane, out + 3 * plane,
          count,   num_dof(), actual,      names_};
}

namespace detail {
namespace {
__global__ void pack_native(const float *actions, const float *p, const float *v, const float *a,
                            const float *j, const float *dt, int knots, int steps, int dof,
                            float *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (knots + 7) * dof) return;
  const int control = i / dof, joint = i % dof;
  const float h = *dt * (steps - 1) / (knots + 4);
  if (i == 0) out[(knots + 7) * dof] = h;
  if (control >= 4 && control < knots + 3) {
    out[i] = actions[(control - 4) * dof + joint];
    return;
  }
  const bool end = control >= knots + 3;
  const int offset = end ? (steps - 1) * dof + joint : joint;
  float fixed[4];
  const curobo::trajectory::bspline::BoundaryConstraint boundary(
      p[offset], end ? -v[offset] : v[offset], a[offset], end ? -j[offset] : j[offset]);
  curobo::trajectory::bspline::compute_fixed_knots<3>(fixed, boundary, h, h * h, h * h * h);
  out[i] = fixed[end ? 3 - (control - knots - 3) : control];
}
__global__ void pack_timeopt(const float *actions, const float *start, const float *goal, int knots,
                             int dof, int offset, float *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = knots + 6 - offset;
  if (i >= count * dof) return;
  const int c = i / dof + offset, joint = i % dof;
  out[i] = c < 4 ? start[c * dof + joint]
                 : (c >= knots + 3 ? goal[joint] : actions[(c - 4) * dof + joint]);
}
}  // namespace
SplineTrajectory native_spline(const float *actions, const float *p, const float *v, const float *a,
                               const float *j, const float *dt, int knots, int steps,
                               std::vector<std::string> names, int device, cudaStream_t stream) {
  DeviceScope scope(device);
  checked_stream(stream);
  const int dof = names.size(), count = knots + 7, values = count * dof;
  auto storage = allocate(values + 1, device);
  auto *out = static_cast<float *>(storage.get());
  pack_native<<<(values + 255) / 256, 256, 0, stream>>>(actions, p, v, a, j, dt, knots, steps, dof,
                                                        out);
  CRR_CUDA_CHECK(cudaGetLastError());
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return SplineTrajectory(out, count, std::move(names), UniformTiming{out + values}, device,
                          storage);
}
SplineTrajectory timeopt_spline(const float *actions, const float *start, const float *goal,
                                const float *timing, const float *time, int knots, int points,
                                bool reference, std::vector<std::string> names, int device,
                                cudaStream_t stream) {
  DeviceScope scope(device);
  checked_stream(stream);
  const int offset = reference ? 0 : 1, count = knots + 6 - offset;
  const int dof = names.size(), values = count * dof;
  auto storage = allocate(values + 32 + points, device);
  auto *out = static_cast<float *>(storage.get());
  pack_timeopt<<<(values + 255) / 256, 256, 0, stream>>>(actions, start, goal, knots, dof, offset,
                                                         out);
  CRR_CUDA_CHECK(cudaGetLastError());
  copy(out + values, device, timing, device, 32, stream);
  copy(out + values + 32, device, time, device, points, stream);
  CRR_CUDA_CHECK(cudaStreamSynchronize(stream));
  return SplineTrajectory(out, count, std::move(names),
                          TimeOptTiming{out + values, out + values + 32, points}, device, storage);
}
}  // namespace detail
}  // namespace curobo::robot
