// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "trajectory_waypoints_cuda.h"

#include <cuda_runtime.h>

namespace curobo::robot::detail {
namespace {

constexpr int BLOCK_SIZE = 128;

__global__ void moving_reference_margin(float *dt, const float *velocity, const float *acceleration,
                                        int seeds, int dof, float factor, float maximum_dt) {
  const int seed = blockIdx.x * blockDim.x + threadIdx.x;
  if (seed >= seeds) return;
  for (int joint = 0; joint < dof; ++joint) {
    const int i = seed * dof + joint;
    if (velocity[i] != 0.0F || acceleration[i] != 0.0F) {
      dt[seed] = fminf(maximum_dt, dt[seed] * factor);
      return;
    }
  }
}

__global__ void iteration_count(int32_t *value, int increment) { *value += increment; }

__global__ void optimization_condition(cudaGraphConditionalHandle handle, const int32_t *native,
                                       const uint8_t *result, OptimizationCondition condition) {
  bool enabled = false;
  switch (condition) {
    case OptimizationCondition::INITIAL_FAILED:
      enabled = native[0] == 0;
      break;
    case OptimizationCondition::INITIAL_SUCCEEDED:
      enabled = native[0] != 0;
      break;
    case OptimizationCondition::REFINEMENT_FAILED:
      enabled = native[0] != 0 && result[0] == 0;
      break;
  }
  cudaGraphSetConditional(handle, enabled ? 1U : 0U);
}

__device__ __forceinline__ float4 normalized_quaternion(const float *q) {
  const float norm = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return make_float4(q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);
}

__device__ __forceinline__ float4 slerp(float4 a, float4 b, float t) {
  float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  if (dot < 0.0F) {
    b = make_float4(-b.x, -b.y, -b.z, -b.w);
    dot = -dot;
  }
  float u = 1.0F - t, v = t;
  if (dot < 0.9999999F) {
    const float angle = acosf(fminf(1.0F, dot));
    u = sinf((1.0F - t) * angle) / sinf(angle);
    v = sinf(t * angle) / sinf(angle);
  }
  float4 q =
      make_float4(u * a.x + v * b.x, u * a.y + v * b.y, u * a.z + v * b.z, u * a.w + v * b.w);
  const float norm = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  return make_float4(q.x / norm, q.y / norm, q.z / norm, q.w / norm);
}

__global__ void path_settings(float *output, ToolPathMode mode, bool orientation, float p, float r,
                              float fp, float fr, float x, float y, float z, bool goal_frame) {
  output[0] = static_cast<float>(mode);
  output[1] = orientation;
  output[2] = p;
  output[3] = r;
  output[4] = fp;
  output[5] = fr;
  output[6] = x;
  output[7] = y;
  output[8] = z;
  output[9] = goal_frame;
}

__global__ void pose_reference(const float *position, const float *quaternion, int count,
                               const float *start_position, const float *start_quaternion,
                               int tools, SplineLayout layout, float *output_position,
                               float *output_quaternion) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= layout.native_horizon * tools) return;
  const int tool = idx % tools, step = idx / tools;
  int segment = 0;
  float fraction = 0.0F;
  if (count > 0) {
    const int active = layout.reference_intervals;
    const int query = max(0, min(active, step - layout.padding));
    while (segment < count - 1 && __float2int_rn(float(active * (segment + 1)) / count) < query)
      ++segment;
    const int left = __float2int_rn(float(active * segment) / count);
    const int right = __float2int_rn(float(active * (segment + 1)) / count);
    fraction = float(query - left) / float(right - left);
  }
  const int first = (max(0, segment - 1) * tools + tool);
  const int last = segment * tools + tool;
  const float *p = count == 0 ? position + tool * 3
                              : (segment == 0 ? start_position + tool * 3 : position + first * 3);
  const float *q = count == 0
                       ? quaternion + tool * 4
                       : (segment == 0 ? start_quaternion + tool * 4 : quaternion + first * 4);
  const float *end_p = count == 0 ? p : position + last * 3;
  const float *end_q = count == 0 ? q : quaternion + last * 4;
  for (int axis = 0; axis < 3; ++axis)
    output_position[idx * 3 + axis] = (1.0F - fraction) * p[axis] + fraction * end_p[axis];
  const float4 value = slerp(normalized_quaternion(q), normalized_quaternion(end_q), fraction);
  output_quaternion[idx * 4] = value.x;
  output_quaternion[idx * 4 + 1] = value.y;
  output_quaternion[idx * 4 + 2] = value.z;
  output_quaternion[idx * 4 + 3] = value.w;
}

__global__ void reference_progress(const int32_t *counts, int seeds, int samples,
                                   SplineLayout layout, float *progress) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= seeds * samples) return;
  const float native_index =
      float(idx % samples) * (layout.native_horizon - 1) / max(1, counts[idx / samples] - 1);
  progress[idx] =
      fminf(1.0F, fmaxf(0.0F, (native_index - layout.padding) / layout.reference_intervals));
}

__global__ void validate_tool_path(const float *positions, const float *quaternions,
                                   const float *reference_position,
                                   const float *reference_quaternion, const float *progress,
                                   const int32_t *counts, const float *settings, int seeds,
                                   int samples, int tools, SplineLayout layout,
                                   int32_t *violation) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= seeds * samples * tools || settings[0] == static_cast<float>(ToolPathMode::DISABLED))
    return;
  const int sample = (idx / tools) % samples, seed = idx / (samples * tools), tool = idx % tools;
  if (counts[seed] < 1 || counts[seed] > samples) {
    atomicMax(violation + seed, 1);
    return;
  }
  if (sample >= counts[seed]) return;
  const float s = progress[seed * samples + sample];
  if (!isfinite(s) || s < 0.0F || s > 1.0F) {
    atomicMax(violation + seed, 1);
    return;
  }
  // Line constraints use one terminal pose, independent of spline horizon.
  const bool temporal = settings[0] == static_cast<float>(ToolPathMode::WAYPOINTS);
  const float coordinate = temporal ? s * layout.reference_intervals : 0.0F;
  const int interval = temporal ? min(int(coordinate), layout.reference_intervals - 1) : 0;
  const float t = coordinate - interval;
  const int first = temporal ? (interval + layout.padding) * tools + tool : tool;
  const int last = temporal ? first + tools : first;
  float3 error = make_float3(0.0F, 0.0F, 0.0F);
  float *e = reinterpret_cast<float *>(&error);
  for (int axis = 0; axis < 3; ++axis)
    e[axis] = positions[idx * 3 + axis] - (1 - t) * reference_position[first * 3 + axis] -
              t * reference_position[last * 3 + axis];
  const float4 target = slerp(normalized_quaternion(reference_quaternion + first * 4),
                              normalized_quaternion(reference_quaternion + last * 4), t);
  const float4 current = normalized_quaternion(quaternions + idx * 4);
  const bool terminal = sample == counts[seed] - 1;
  if (settings[0] == static_cast<float>(ToolPathMode::LINE) && !terminal) {
    float3 direction = make_float3(settings[6], settings[7], settings[8]);
    const float norm =
        sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction.x /= norm;
    direction.y /= norm;
    direction.z /= norm;
    if (settings[9] != 0.0F) {
      const float3 u = make_float3(target.y, target.z, target.w);
      const float3 c = make_float3(2 * (u.y * direction.z - u.z * direction.y),
                                   2 * (u.z * direction.x - u.x * direction.z),
                                   2 * (u.x * direction.y - u.y * direction.x));
      direction = make_float3(direction.x + target.x * c.x + u.y * c.z - u.z * c.y,
                              direction.y + target.x * c.y + u.z * c.x - u.x * c.z,
                              direction.z + target.x * c.z + u.x * c.y - u.y * c.x);
    }
    const float along = error.x * direction.x + error.y * direction.y + error.z * direction.z;
    error.x -= along * direction.x;
    error.y -= along * direction.y;
    error.z -= along * direction.z;
  }
  const float p = sqrtf(error.x * error.x + error.y * error.y + error.z * error.z);
  const float dot = fabsf(current.x * target.x + current.y * target.y + current.z * target.z +
                          current.w * target.w);
  const float3 relative = make_float3(
      target.x * current.y - target.y * current.x - target.z * current.w + target.w * current.z,
      target.x * current.z + target.y * current.w - target.z * current.x - target.w * current.y,
      target.x * current.w - target.y * current.z + target.z * current.y - target.w * current.x);
  const float r =
      2 * atan2f(sqrtf(relative.x * relative.x + relative.y * relative.y + relative.z * relative.z),
                 dot);
  if (!isfinite(p) || !isfinite(r) || p > settings[terminal ? 4 : 2] ||
      (settings[1] != 0.0F && r > settings[terminal ? 5 : 3]))
    atomicMax(violation + seed, 1);
}
}  // namespace

cudaError_t launch_tool_path_settings(float *output, ToolPathMode mode, bool orientation, float p,
                                      float r, float fp, float fr, float x, float y, float z,
                                      bool frame, cudaStream_t stream) {
  path_settings<<<1, 1, 0, stream>>>(output, mode, orientation, p, r, fp, fr, x, y, z, frame);
  return cudaGetLastError();
}
cudaError_t launch_pose_waypoints(const float *p, const float *q, int count, const float *sp,
                                  const float *sq, int tools, SplineLayout layout, float *op,
                                  float *oq, cudaStream_t stream) {
  pose_reference<<<(layout.native_horizon * tools + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                   stream>>>(p, q, count, sp, sq, tools, layout, op, oq);
  return cudaGetLastError();
}
cudaError_t launch_reference_progress(const int32_t *count, int seeds, int samples,
                                      SplineLayout layout, float *progress, cudaStream_t stream) {
  reference_progress<<<(seeds * samples + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      count, seeds, samples, layout, progress);
  return cudaGetLastError();
}
cudaError_t launch_validate_tool_path(const float *p, const float *q, const float *rp,
                                      const float *rq, const float *progress, const int32_t *counts,
                                      const float *settings, int seeds, int samples, int tools,
                                      SplineLayout layout, int32_t *violation,
                                      cudaStream_t stream) {
  validate_tool_path<<<(seeds * samples * tools + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                       stream>>>(p, q, rp, rq, progress, counts, settings, seeds, samples, tools,
                                 layout, violation);
  return cudaGetLastError();
}
cudaError_t launch_moving_reference_margin(float *dt, const float *velocity,
                                           const float *acceleration, int seeds, int dof,
                                           float factor, float maximum_dt, cudaStream_t stream) {
  moving_reference_margin<<<(seeds + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      dt, velocity, acceleration, seeds, dof, factor, maximum_dt);
  return cudaGetLastError();
}

cudaError_t launch_initial_iteration_count(int32_t *count, int increment, cudaStream_t stream) {
  iteration_count<<<1, 1, 0, stream>>>(count, increment);
  return cudaGetLastError();
}
cudaError_t launch_optimization_condition(cudaGraphConditionalHandle handle, const int32_t *native,
                                          const uint8_t *result, OptimizationCondition condition,
                                          cudaStream_t stream) {
  optimization_condition<<<1, 1, 0, stream>>>(handle, native, result, condition);
  return cudaGetLastError();
}
}  // namespace curobo::robot::detail
