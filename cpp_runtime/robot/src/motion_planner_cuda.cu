// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <math_constants.h>

#include "motion_planner_cuda.h"

namespace curobo::robot::detail {
namespace {

__device__ float3 rotated_axis(const float* quaternion, int axis) {
  const float inverse_norm = rsqrtf(
      quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
      quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  const float w = quaternion[0] * inverse_norm;
  const float x = quaternion[1] * inverse_norm;
  const float y = quaternion[2] * inverse_norm;
  const float z = quaternion[3] * inverse_norm;
  const float3 v = make_float3(axis == 0, axis == 1, axis == 2);
  const float3 t = make_float3(2 * (y * v.z - z * v.y),
                             2 * (z * v.x - x * v.z),
                             2 * (x * v.y - y * v.x));
  return make_float3(v.x + w * t.x + y * t.z - z * t.y,
                    v.y + w * t.y + z * t.x - x * t.z,
                    v.z + w * t.z + x * t.y - y * t.x);
}

__global__ void prepare_grasp_poses(GraspPosePreparation request) {
  const int goal = request.goal_indices[request.selected_seed * request.num_tools];
  bool valid = goal >= 0 && goal < request.num_grasps;
  for (int tool = threadIdx.x; tool < request.num_tools && valid;
       tool += blockDim.x) {
    const int pose = tool * request.num_grasps + goal;
    const float* p = request.positions + pose * 3;
    const float* q = request.quaternions + pose * 4;
    const float norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    valid = isfinite(p[0]) && isfinite(p[1]) && isfinite(p[2]) &&
            isfinite(norm) && norm > 0;
    if (!valid) {
      break;
    }
    const float3 approach = request.approach_in_tool_frame
        ? rotated_axis(q, request.approach_axis)
        : make_float3(request.approach_axis == 0, request.approach_axis == 1,
                      request.approach_axis == 2);
    const float3 lift = request.lift_in_tool_frame
        ? rotated_axis(q, request.lift_axis)
        : make_float3(request.lift_axis == 0, request.lift_axis == 1,
                      request.lift_axis == 2);
    const float da[3]{approach.x, approach.y, approach.z};
    const float dl[3]{lift.x, lift.y, lift.z};
    for (int component = 0; component < 3; ++component) {
      request.selected_positions[tool * 3 + component] = p[component];
      request.approach_positions[tool * 3 + component] =
          p[component] + request.approach_offset * da[component];
      request.lift_positions[tool * 3 + component] =
          p[component] + request.lift_offset * dl[component];
    }
    for (int component = 0; component < 4; ++component) {
      request.selected_quaternions[tool * 4 + component] = q[component];
      request.approach_quaternions[tool * 4 + component] = q[component];
      request.lift_quaternions[tool * 4 + component] = q[component];
    }
  }
  const int all_valid = __syncthreads_and(valid);
  if (threadIdx.x == 0) {
    *request.valid = all_valid ? 1U : 0U;
    *request.selected_index = all_valid ? goal : -1;
  }
}

__global__ void set_sphere_collision_enabled(
    float* spheres, const float* nominal_spheres, const int32_t* slots,
    int num_slots, bool enabled) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_slots) {
    const int radius = slots[index] * 4 + 3;
    spheres[radius] = enabled ? nominal_spheres[radius] : -100.0F;
  }
}

__global__ void set_sphere_collision_mask(uint32_t* masks, const int32_t* slots, int num_slots,
                                          uint32_t layers) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_slots) {
    masks[slots == nullptr ? index : slots[index]] = layers;
  }
}

// A conservative 4x4x2 AABB cover in the selected tool frame. All selected
// voxel centers contribute; no CPU point readback or live attachment update.
__global__ void fit_grasp_attachment(const float* points, int count, const float* position,
                                     const float* quaternion, float voxel_radius,
                                     const int32_t* slots, float* spheres, uint8_t* valid) {
  __shared__ float bounds[6][256];
  const int thread = threadIdx.x;
  for (int axis = 0; axis < 3; ++axis) {
    bounds[axis][thread] = CUDART_INF_F;
    bounds[axis + 3][thread] = -CUDART_INF_F;
  }
  const float3 axes[3] = {rotated_axis(quaternion, 0), rotated_axis(quaternion, 1),
                          rotated_axis(quaternion, 2)};
  const float norm = quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                     quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3];
  bool finite = isfinite(norm) && norm > 0;
  for (int i = thread; i < count; i += blockDim.x) {
    const float3 delta = make_float3(points[i * 3] - position[0], points[i * 3 + 1] - position[1],
                                     points[i * 3 + 2] - position[2]);
    for (int axis = 0; axis < 3; ++axis) {
      const float value = delta.x * axes[axis].x + delta.y * axes[axis].y + delta.z * axes[axis].z;
      finite = finite && isfinite(value);
      bounds[axis][thread] = fminf(bounds[axis][thread], value);
      bounds[axis + 3][thread] = fmaxf(bounds[axis + 3][thread], value);
    }
  }
  const bool all_finite = __syncthreads_and(finite);
  if (thread == 0) *valid = all_finite;
  if (!all_finite) return;
  for (int stride = 128; stride > 0; stride /= 2) {
    if (thread < stride) {
      for (int axis = 0; axis < 3; ++axis) {
        bounds[axis][thread] = fminf(bounds[axis][thread], bounds[axis][thread + stride]);
        bounds[axis + 3][thread] =
            fmaxf(bounds[axis + 3][thread], bounds[axis + 3][thread + stride]);
      }
    }
    __syncthreads();
  }
  if (thread < 32) {
    const int cells[3] = {4, 4, 2};
    const int index[3] = {thread / 8, (thread / 2) % 4, thread % 2};
    float radius_squared = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const float pitch = fmaxf(bounds[axis + 3][0] - bounds[axis][0], 1.e-6F) / cells[axis];
      spheres[slots[thread] * 4 + axis] = bounds[axis][0] + (index[axis] + .5F) * pitch;
      radius_squared += pitch * pitch;
    }
    spheres[slots[thread] * 4 + 3] = .5F * sqrtf(radius_squared) + voxel_radius;
  }
}

__global__ void write_motion_result_metadata(
    const int32_t* trajopt_selected_seed, const uint8_t* rrt_success,
    const int32_t* rrt_iterations, int num_seeds, bool rrt_stage,
    int32_t* selected_seed, uint8_t* used_rrt,
    int32_t* selected_rrt_iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int32_t seed = *trajopt_selected_seed;
  const bool valid_seed = seed >= 0 && seed < num_seeds;
  if (selected_seed != nullptr) {
    *selected_seed = seed;
  }
  if (rrt_stage && valid_seed) {
    *used_rrt = rrt_success[seed];
    *selected_rrt_iterations = rrt_iterations[seed];
    return;
  }
  *used_rrt = 0U;
  *selected_rrt_iterations = 0;
}

__global__ void write_pose_result_metadata(
    const int32_t* selected_seed, const float* ik_solutions,
    const float* ik_position_error, const float* ik_orientation_error,
    int num_seeds, int num_dof, int planning_attempts,
    float* selected_goal_position, float* position_error,
    float* orientation_error, int32_t* output_selected_seed,
    int32_t* output_planning_attempts, uint8_t* success) {
  __shared__ int32_t seed;
  if (threadIdx.x == 0) {
    seed = *selected_seed;
    const bool valid_seed = seed >= 0 && seed < num_seeds;
    if (!valid_seed) {
      *success = 0U;
    }
    if (output_selected_seed != nullptr) {
      *output_selected_seed = valid_seed ? seed : -1;
    }
    if (output_planning_attempts != nullptr) {
      *output_planning_attempts = planning_attempts;
    }
    if (valid_seed && position_error != nullptr) {
      *position_error = ik_position_error[seed];
    }
    if (valid_seed && orientation_error != nullptr) {
      *orientation_error = ik_orientation_error[seed];
    }
  }
  __syncthreads();
  if (selected_goal_position != nullptr && seed >= 0 && seed < num_seeds) {
    for (int joint = threadIdx.x; joint < num_dof; joint += blockDim.x) {
      selected_goal_position[joint] = ik_solutions[seed * num_dof + joint];
    }
  }
}

}  // namespace

cudaError_t launch_prepare_grasp_poses(const GraspPosePreparation& request,
                                     cudaStream_t stream) {
  prepare_grasp_poses<<<1, 32, 0, stream>>>(request);
  return cudaGetLastError();
}

cudaError_t launch_set_sphere_collision_enabled(
    float* spheres, const float* nominal_spheres, const int32_t* slots,
    int num_slots, bool enabled, cudaStream_t stream) {
  if (num_slots > 0) {
    set_sphere_collision_enabled<<<(num_slots + 127) / 128, 128, 0, stream>>>(
        spheres, nominal_spheres, slots, num_slots, enabled);
  }
  return cudaGetLastError();
}

cudaError_t launch_set_sphere_collision_mask(uint32_t* masks, const int32_t* slots, int num_slots,
                                             uint32_t layers, cudaStream_t stream) {
  if (num_slots > 0) {
    set_sphere_collision_mask<<<(num_slots + 127) / 128, 128, 0, stream>>>(masks, slots, num_slots,
                                                                           layers);
  }
  return cudaGetLastError();
}

cudaError_t launch_fit_grasp_attachment(const float* points, int count, const float* position,
                                        const float* quaternion, float voxel_radius,
                                        const int32_t* slots, float* spheres, uint8_t* valid,
                                        cudaStream_t stream) {
  fit_grasp_attachment<<<1, 256, 0, stream>>>(points, count, position, quaternion, voxel_radius,
                                              slots, spheres, valid);
  return cudaGetLastError();
}

cudaError_t launch_motion_result_metadata(
    const int32_t* trajopt_selected_seed, const uint8_t* rrt_success,
    const int32_t* rrt_iterations, int num_seeds, bool rrt_stage,
    int32_t* selected_seed, uint8_t* used_rrt,
    int32_t* selected_rrt_iterations, cudaStream_t stream) {
  write_motion_result_metadata<<<1, 1, 0, stream>>>(
      trajopt_selected_seed, rrt_success, rrt_iterations, num_seeds, rrt_stage,
      selected_seed, used_rrt, selected_rrt_iterations);
  return cudaGetLastError();
}

cudaError_t launch_pose_result_metadata(
    const int32_t* selected_seed, const float* ik_solutions,
    const float* ik_position_error, const float* ik_orientation_error,
    int num_seeds, int num_dof, int planning_attempts,
    float* selected_goal_position, float* position_error,
    float* orientation_error, int32_t* output_selected_seed,
    int32_t* output_planning_attempts, uint8_t* success,
    cudaStream_t stream) {
  constexpr int THREADS_PER_BLOCK = 32;
  write_pose_result_metadata<<<1, THREADS_PER_BLOCK, 0, stream>>>(
      selected_seed, ik_solutions, ik_position_error, ik_orientation_error,
      num_seeds, num_dof, planning_attempts, selected_goal_position,
      position_error, orientation_error, output_selected_seed,
      output_planning_attempts, success);
  return cudaGetLastError();
}

}  // namespace curobo::robot::detail
