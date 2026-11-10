// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <math_constants.h>
#include "mpc_cuda.h"

namespace curobo::robot::detail {
namespace {
constexpr int BLOCK_SIZE = 128;
__global__ void pose_acceptance_kernel(float *position, float *rotation,
                                       const float *tool_position, const float *goal_position,
                                       const float *goal_quaternion, const float *direction,
                                       const int32_t *error_frame, const uint8_t *constrain_path,
                                       int steps, int tools) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= steps * tools) return;
  if (!constrain_path[0]) {
    position[index] = 0.F;
    rotation[index] = 0.F;
    return;
  }
  const int tool = index % tools;
  float dx = direction[0], dy = direction[1], dz = direction[2];
  if (error_frame[tool]) {
    const float *q = goal_quaternion + 4 * tool;
    const float tx = 2.F * (q[2] * dz - q[3] * dy);
    const float ty = 2.F * (q[3] * dx - q[1] * dz);
    const float tz = 2.F * (q[1] * dy - q[2] * dx);
    dx += q[0] * tx + q[2] * tz - q[3] * ty;
    dy += q[0] * ty + q[3] * tx - q[1] * tz;
    dz += q[0] * tz + q[1] * ty - q[2] * tx;
  }
  const float ex = tool_position[3 * index] - goal_position[3 * tool];
  const float ey = tool_position[3 * index + 1] - goal_position[3 * tool + 1];
  const float ez = tool_position[3 * index + 2] - goal_position[3 * tool + 2];
  // ||error x direction|| is perpendicular distance without subtracting the
  // weighted along-axis cost, which would couple acceptance to progress weight.
  const float cx = ey * dz - ez * dy;
  const float cy = ez * dx - ex * dz;
  const float cz = ex * dy - ey * dx;
  const float distance = sqrtf(cx * cx + cy * cy + cz * cz);
  position[index] = isfinite(distance) ? distance : CUDART_INF_F;
  if (!isfinite(rotation[index])) rotation[index] = CUDART_INF_F;
}

__global__ void tail_kernel(const float *seed, float *goal, const uint8_t *implicit_goal,
                            int knots, int dof) {
  const int joint = blockIdx.x * blockDim.x + threadIdx.x;
  if (joint < dof && !implicit_goal[0]) goal[joint] = seed[(knots - 1) * dof + joint];
}

__global__ void shift_kernel(const float *input, float *output, const float *goal,
                             const int *shift, int rows, int knots, int dof) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int dimension = knots * dof;
  if (index >= rows * dimension) return;
  const int joint = index % dof;
  const int row = index / dimension;
  const int knot = (index % dimension) / dof;
  const int source = knot + min(max(shift[0], 0), knots);
  const float tail = goal == nullptr ? 0.0F : goal[joint];
  output[index] = source < knots ? input[row * dimension + source * dof + joint] : tail;
}

__global__ void broadcast_kernel(const float *input, float *output, int values, int rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < values * rows) output[index] = input[index % values];
}

__global__ void rank_kernel(uint8_t *success, const float *cost, const float *dt,
                             float *rank, int candidates) {
  const int index = threadIdx.x;
  if (index >= candidates) return;
  if (!isfinite(cost[index]) || !isfinite(dt[index])) success[index] = 0;
  rank[index] = success[index] ? dt[index] : CUDART_INF_F;
}

__global__ void validate_kernel(const float *q, const float *v, const float *a, const float *j,
                                const float *q_bounds, const float *v_bounds,
                                const float *a_bounds, const float *j_bounds, const float *world,
                                const float *self, const float *spheres, int num_spheres,
                                float max_sphere_step, int *violation, int steps, int dof) {
  const int local_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_index >= steps * dof) return;
  const int batch = blockIdx.y;
  const int index = batch * steps * dof + local_index;
  const int joint = local_index % dof, step = local_index / dof;
  const int state = batch * steps + step;
  const float values[4]{q[index], v[index], a[index], j[index]};
  const float *bounds[4]{q_bounds, v_bounds, a_bounds, j_bounds};
  int flags = 0;
  for (int term = 0; term < 4; ++term)
    if (!isfinite(values[term]) || values[term] < bounds[term][joint] ||
        values[term] > bounds[term][dof + joint])
      flags |= 1 << term;
  if (!isfinite(world[state]) || world[state] > 0.0F) flags |= 16;
  if (!isfinite(self[state]) || self[state] > 0.0F) flags |= 32;
  if (joint == 0 && step > 0) {
    for (int sphere = 0; sphere < num_spheres; ++sphere) {
      const int i = (state * num_spheres + sphere) * 4;
      if (spheres[i + 3] <= 0.0F) continue;
      float distance_squared = 0.0F;
      for (int axis = 0; axis < 3; ++axis) {
        const float delta = spheres[i + axis] - spheres[i - 4 * num_spheres + axis];
        distance_squared += delta * delta;
      }
      if (!isfinite(distance_squared) || distance_squared > max_sphere_step * max_sphere_step)
        flags |= 64;
    }
  }
  if (flags != 0) atomicOr(violation + batch, flags);
}
}  // namespace

cudaError_t launch_mpc_shift(const float *input, float *output, const float *goal,
                             const int *shift, int rows, int knots, int dof, cudaStream_t stream) {
  shift_kernel<<<(rows * knots * dof + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      input, output, goal, shift, rows, knots, dof);
  return cudaGetLastError();
}

cudaError_t launch_mpc_tail(const float *seed, float *goal, const uint8_t *implicit_goal,
                            int knots, int dof, cudaStream_t stream) {
  tail_kernel<<<(dof + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      seed, goal, implicit_goal, knots, dof);
  return cudaGetLastError();
}
cudaError_t launch_mpc_pose_acceptance(float *position, float *rotation,
                                       const float *tool_position, const float *goal_position,
                                       const float *goal_quaternion, const float *direction,
                                       const int32_t *error_frame, const uint8_t *constrain_path,
                                       int steps, int tools, cudaStream_t stream) {
  pose_acceptance_kernel<<<(steps * tools + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      position, rotation, tool_position, goal_position, goal_quaternion, direction, error_frame,
      constrain_path, steps, tools);
  return cudaGetLastError();
}

cudaError_t launch_mpc_broadcast(const float *input, float *output, int values, int rows,
                                 cudaStream_t stream) {
  broadcast_kernel<<<(values * rows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      input, output, values, rows);
  return cudaGetLastError();
}

cudaError_t launch_mpc_rank(uint8_t *success, const float *cost, const float *dt,
                            float *rank, int candidates, cudaStream_t stream) {
  rank_kernel<<<1, BLOCK_SIZE, 0, stream>>>(success, cost, dt, rank, candidates);
  return cudaGetLastError();
}

cudaError_t launch_mpc_validate(const float *q, const float *v, const float *a, const float *j,
                                const float *q_bounds, const float *v_bounds,
                                const float *a_bounds, const float *j_bounds, const float *world,
                                const float *self, const float *spheres, int num_spheres,
                                float max_sphere_step, int *violation, int steps, int dof,
                                cudaStream_t stream, int batches) {
  validate_kernel<<<dim3((steps * dof + BLOCK_SIZE - 1) / BLOCK_SIZE, batches),
                    BLOCK_SIZE, 0, stream>>>(
      q, v, a, j, q_bounds, v_bounds, a_bounds, j_bounds, world, self, spheres, num_spheres,
      max_sphere_step, violation, steps, dof);
  return cudaGetLastError();
}
}  // namespace curobo::robot::detail
