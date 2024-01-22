/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "helper_math.h"
#include <cmath>
#include <cuda_fp16.h>
#include <vector>

#define SINGLE_GOAL 0
#define BATCH_GOAL 1
#define GOALSET 2
#define BATCH_GOALSET 3

namespace Curobo 
{
namespace Pose
{
__device__ __forceinline__ void
transform_vec_quat(const float4 q, // x,y,z, qw, qx,qy,qz
                   const float *d_vec_weight, 
                   float *C) {
  // do dot product:
  // new_p = q * p * q_inv + obs_p

  if (false || q.x != 0 || q.y != 0 || q.z != 0) {

    C[0] = q.w * q.w * d_vec_weight[0] + 2 * q.y * q.w * d_vec_weight[2] -
          2 * q.z * q.w * d_vec_weight[1] + q.x * q.x * d_vec_weight[0] +
          2 * q.y * q.x * d_vec_weight[1] + 2 * q.z * q.x * d_vec_weight[2] -
          q.z * q.z * d_vec_weight[0] - q.y * q.y * d_vec_weight[0];
    C[1] = 2 * q.x * q.y * d_vec_weight[0] + q.y * q.y * d_vec_weight[1] +
          2 * q.z * q.y * d_vec_weight[2] + 2 * q.w * q.z * d_vec_weight[0] -
          q.z * q.z * d_vec_weight[1] + q.w * q.w * d_vec_weight[1] -
          2 * q.x * q.w * d_vec_weight[2] - q.x * q.x * d_vec_weight[1];
    C[2] = 2 * q.x * q.z * d_vec_weight[0] + 2 * q.y * q.z * d_vec_weight[1] +
          q.z * q.z * d_vec_weight[2] - 2 * q.w * q.y * d_vec_weight[0] -
          q.y * q.y * d_vec_weight[2] + 2 * q.w * q.x * d_vec_weight[1] -
          q.x * q.x * d_vec_weight[2] + q.w * q.w * d_vec_weight[2];

    C[3] = q.w * q.w * d_vec_weight[3] + 2 * q.y * q.w * d_vec_weight[5] -
          2 * q.z * q.w * d_vec_weight[4] + q.x * q.x * d_vec_weight[3] +
          2 * q.y * q.x * d_vec_weight[4] + 2 * q.z * q.x * d_vec_weight[5] -
          q.z * q.z * d_vec_weight[3] - q.y * q.y * d_vec_weight[3];
    C[4] = 2 * q.x * q.y * d_vec_weight[3] + q.y * q.y * d_vec_weight[4] + 
          2 * q.z * q.y * d_vec_weight[5] + 2 * q.w * q.z * d_vec_weight[3] -
          q.z * q.z * d_vec_weight[4] + q.w * q.w * d_vec_weight[4] -
          2 * q.x * q.w * d_vec_weight[5] - q.x * q.x * d_vec_weight[4];
    C[5] = 2 * q.x * q.z * d_vec_weight[3] + 2 * q.y * q.z * d_vec_weight[4] +
          q.z * q.z * d_vec_weight[5] - 2 * q.w * q.y * d_vec_weight[3] -
          q.y * q.y * d_vec_weight[5] + 2 * q.w * q.x * d_vec_weight[4] -
          q.x * q.x * d_vec_weight[5] + q.w * q.w * d_vec_weight[5];
  }
  {
    C[0] = d_vec_weight[0];
    C[1] = d_vec_weight[1];
    C[2] = d_vec_weight[2];
    C[3] = d_vec_weight[3];
    C[4] = d_vec_weight[4];
    C[5] = d_vec_weight[5];
    

  }
}

__device__ __forceinline__ void
inv_transform_vec_quat(const float4 q_in, // x,y,z, qw, qx,qy,qz
                   const float *d_vec_weight, 
                   float *C) {
  // do dot product:
  // new_p = q * p * q_inv + obs_p

  if (q_in.x != 0 || q_in.y != 0 || q_in.z != 0) {
    float4 q = make_float4(-q_in.x, -q_in.y, -q_in.z, q_in.w);

    C[0] = q.w * q.w * d_vec_weight[0] + 2 * q.y * q.w * d_vec_weight[2] -
          2 * q.z * q.w * d_vec_weight[1] + q.x * q.x * d_vec_weight[0] +
          2 * q.y * q.x * d_vec_weight[1] + 2 * q.z * q.x * d_vec_weight[2] -
          q.z * q.z * d_vec_weight[0] - q.y * q.y * d_vec_weight[0];
    C[1] = 2 * q.x * q.y * d_vec_weight[0] + q.y * q.y * d_vec_weight[1] +
          2 * q.z * q.y * d_vec_weight[2] + 2 * q.w * q.z * d_vec_weight[0] -
          q.z * q.z * d_vec_weight[1] + q.w * q.w * d_vec_weight[1] -
          2 * q.x * q.w * d_vec_weight[2] - q.x * q.x * d_vec_weight[1];
    C[2] = 2 * q.x * q.z * d_vec_weight[0] + 2 * q.y * q.z * d_vec_weight[1] +
          q.z * q.z * d_vec_weight[2] - 2 * q.w * q.y * d_vec_weight[0] -
          q.y * q.y * d_vec_weight[2] + 2 * q.w * q.x * d_vec_weight[1] -
          q.x * q.x * d_vec_weight[2] + q.w * q.w * d_vec_weight[2];

    C[3] = q.w * q.w * d_vec_weight[3] + 2 * q.y * q.w * d_vec_weight[5] -
          2 * q.z * q.w * d_vec_weight[4] + q.x * q.x * d_vec_weight[3] +
          2 * q.y * q.x * d_vec_weight[4] + 2 * q.z * q.x * d_vec_weight[5] -
          q.z * q.z * d_vec_weight[3] - q.y * q.y * d_vec_weight[3];
    C[4] = 2 * q.x * q.y * d_vec_weight[3] + q.y * q.y * d_vec_weight[4] + 
          2 * q.z * q.y * d_vec_weight[5] + 2 * q.w * q.z * d_vec_weight[3] -
          q.z * q.z * d_vec_weight[4] + q.w * q.w * d_vec_weight[4] -
          2 * q.x * q.w * d_vec_weight[5] - q.x * q.x * d_vec_weight[4];
    C[5] = 2 * q.x * q.z * d_vec_weight[3] + 2 * q.y * q.z * d_vec_weight[4] +
          q.z * q.z * d_vec_weight[5] - 2 * q.w * q.y * d_vec_weight[3] -
          q.y * q.y * d_vec_weight[5] + 2 * q.w * q.x * d_vec_weight[4] -
          q.x * q.x * d_vec_weight[5] + q.w * q.w * d_vec_weight[5];
  }
  {
    C[0] = d_vec_weight[0];
    C[1] = d_vec_weight[1];
    C[2] = d_vec_weight[2];
    C[3] = d_vec_weight[3];
    C[4] = d_vec_weight[4];
    C[5] = d_vec_weight[5];
    

  }
}
__device__ __forceinline__ void
compute_quat_distance(float *result, const float4 a, const float4 b) {
  // We compute distance with the conjugate of b
  float r_w = (a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z);
  if (r_w < 0.0) {
    r_w = 1.0;
  } else {
    r_w = -1.0;
  }

  result[0] = r_w * (-1 * a.w * b.x + b.w * a.x - a.y * b.z + b.y * a.z);
  result[1] = r_w * (-1 * a.w * b.y + b.w * a.y - a.z * b.x + b.z * a.x);
  result[2] = r_w * (-1 * a.w * b.z + b.w * a.z - a.x * b.y + b.x * a.y);
}

__device__ __forceinline__ void
compute_distance(float *distance_vec, float &distance, float &position_distance,
                 float &rotation_distance, const float3 current_position,
                 const float3 goal_position, const float4 current_quat,
                 const float4 goal_quat, const float *vec_weight,
                 const float *vec_convergence, const float position_weight,
                 const float rotation_weight) {
  compute_quat_distance(&distance_vec[3], goal_quat, current_quat);
  // compute position distance
  *(float3 *)&distance_vec[0] = current_position - goal_position;
  // distance_vec[0] = goal_position
  position_distance = 0;
  rotation_distance = 0;
// scale by vec weight and position weight:
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    distance_vec[i] = vec_weight[i + 3] * distance_vec[i];
    position_distance += distance_vec[i] * distance_vec[i];
  }
#pragma unroll 3
  for (int i = 3; i < 6; i++) {
    distance_vec[i] = vec_weight[i - 3] * distance_vec[i];
    rotation_distance += distance_vec[i] * distance_vec[i];
  }

  distance = 0;
  if (rotation_distance > vec_convergence[0] * vec_convergence[0]) {
    rotation_distance = sqrtf(rotation_distance);
    //rotation_distance -= vec_convergence[0];
  
    distance += rotation_weight * rotation_distance;
  }
  if (position_distance > vec_convergence[1] * vec_convergence[1]) {
    position_distance = sqrtf(position_distance);
    //position_distance -= vec_convergence[1];
  
    distance += position_weight * position_distance;
  }
}

__device__ __forceinline__ void compute_metric_distance(
    float *distance_vec, float &distance, float &position_distance,
    float &rotation_distance, const float3 current_position,
    const float3 goal_position, const float4 current_quat,
    const float4 goal_quat, const float *vec_weight,
    const float *vec_convergence, const float position_weight,
    const float p_alpha, const float rotation_weight, const float r_alpha) {
  compute_quat_distance(&distance_vec[3], goal_quat, current_quat);
  // compute position distance
  *(float3 *)&distance_vec[0] = current_position - goal_position;
  // distance_vec[0] = goal_position
  position_distance = 0;
  rotation_distance = 0;
// scale by vec weight and position weight:
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    distance_vec[i] = vec_weight[i + 3] * distance_vec[i];
    // position_distance += powf(distance_vec[i],2);
    position_distance += distance_vec[i] * distance_vec[i];
  }
#pragma unroll 3
  for (int i = 3; i < 6; i++) {
    distance_vec[i] = vec_weight[i - 3] * distance_vec[i];
    // rotation_distance += powf(distance_vec[i],2);
    rotation_distance += distance_vec[i] * distance_vec[i];
  }

  distance = 0;
  if (rotation_distance > vec_convergence[0] * vec_convergence[0]) {
    rotation_distance = sqrtf(rotation_distance);
    //rotation_distance -= vec_convergence[0];
  
    distance += rotation_weight * log2f(coshf(r_alpha * rotation_distance));
  }
  if (position_distance > vec_convergence[1] * vec_convergence[1]) {
    //position_distance -= vec_convergence[1];
    position_distance = sqrtf(position_distance);
    distance += position_weight * log2f(coshf(p_alpha * position_distance));
  }
}

template <typename scalar_t>
__global__ void
backward_pose_distance_kernel(scalar_t *out_grad_p,            // [b,3]
                              scalar_t *out_grad_q,            // [b,4]
                              const scalar_t *grad_distance,   // [b,1]
                              const scalar_t *grad_p_distance, // [b,1]
                              const scalar_t *grad_q_distance, // [b,1]
                              const scalar_t *pose_weight,     // [2]
                              const scalar_t *grad_p_vec,      // [b,3]
                              const scalar_t *grad_q_vec,      // [b,4]
                              const int batch_size) {
  const int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  // read data
  const float g_distance = grad_distance[batch_idx];
  const float2 p_weight = *(float2 *)&pose_weight[0];
  const float3 g_p_v = *(float3 *)&grad_p_vec[batch_idx * 3];
  const float3 g_q_v = *(float3 *)&grad_q_vec[batch_idx * 4 + 1];
  const float g_p_distance = grad_p_distance[batch_idx];
  const float g_q_distance = grad_q_distance[batch_idx];

  // compute position gradient
  float3 g_p =
      (g_p_v) * ((g_p_distance + g_distance * p_weight.y)); // scalar * float3
  float3 g_q =
      (g_q_v) * ((g_q_distance + g_distance * p_weight.x)); // scalar * float3

  // write out
  *(float3 *)&out_grad_p[batch_idx * 3] = g_p;
  *(float3 *)&out_grad_q[batch_idx * 4 + 1] = g_q;
}

template <typename scalar_t>
__global__ void backward_pose_kernel(scalar_t *out_grad_p,          // [b,3]
                                     scalar_t *out_grad_q,          // [b,4]
                                     const scalar_t *grad_distance, // [b,1]
                                     const scalar_t *pose_weight,   // [2]
                                     const scalar_t *grad_p_vec,    // [b,3]
                                     const scalar_t *grad_q_vec,    // [b,4]
                                     const int batch_size) {
  const int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  // read data
  const float g_distance = grad_distance[batch_idx];
  const float2 p_weight = *(float2 *)&pose_weight[0];
  const float3 g_p_v = *(float3 *)&grad_p_vec[batch_idx * 3];
  const float3 g_q_v = *(float3 *)&grad_q_vec[batch_idx * 4 + 1];

  // compute position gradient
  float3 g_p = (g_p_v) * ((g_distance * p_weight.y)); // scalar * float3
  float3 g_q = (g_q_v) * ((g_distance * p_weight.x)); // scalar * float3

  // write out
  *(float3 *)&out_grad_p[batch_idx * 3] = g_p;
  *(float3 *)&out_grad_q[batch_idx * 4 + 1] = g_q;
}

template <typename scalar_t, bool write_distance>
__global__ void goalset_pose_distance_kernel(
    scalar_t *out_distance, scalar_t *out_position_distance,
    scalar_t *out_rotation_distance, scalar_t *out_p_vec, scalar_t *out_q_vec,
    int32_t *out_gidx, const scalar_t *current_position,
    const scalar_t *goal_position, const scalar_t *current_quat,
    const scalar_t *goal_quat, const scalar_t *vec_weight,
    const scalar_t *weight, const scalar_t *vec_convergence,
    const scalar_t *run_weight, const scalar_t *run_vec_weight,
    const int32_t *batch_pose_idx, const int mode, const int num_goals,
    const int batch_size, const int horizon, const bool write_grad = false) {
  const int t_idx = (blockDim.x * blockIdx.x + threadIdx.x);
  const int batch_idx = t_idx / horizon;
  const int h_idx = t_idx - (batch_idx * horizon);
  if (batch_idx >= batch_size || h_idx >= horizon) {
    return;
  }

  // read current pose:
  float3 position =
      *(float3 *)&current_position[batch_idx * horizon * 3 + h_idx * 3];
  float4 quat_4 = *(float4 *)&current_quat[batch_idx * horizon * 4 + h_idx * 4];
  float4 quat = make_float4(quat_4.y, quat_4.z, quat_4.w, quat_4.x);

  // read weights:

  float position_weight = weight[1];
  float rotation_weight = weight[0];
  if (!write_distance) {
    position_weight *= run_weight[h_idx];
    rotation_weight *= run_weight[h_idx];
    if (position_weight == 0.0 && rotation_weight == 0.0) {
      return;
    }
  }

  float3 l_goal_position;
  float4 l_goal_quat;
  float distance_vec[6]; //  = {0.0};
  float pose_distance = 0.0;
  float position_distance = 0.0;
  float rotation_distance = 0.0;
  float best_distance = INFINITY;
  float best_position_distance = 0.0;
  float best_rotation_distance = 0.0;
  float best_distance_vec[6] = {0.0};
  float best_des_vec_weight[6] = {0.0};
  float d_vec_convergence[2];

  *(float2 *)&d_vec_convergence[0] = *(float2 *)&vec_convergence[0];

  int best_idx = -1;
  float d_vec_weight[6];
  float des_vec_weight[6] = {0.0};
  *(float3 *)&d_vec_weight[0] = *(float3 *)&vec_weight[0];
  *(float3 *)&d_vec_weight[3] = *(float3 *)&vec_weight[3];
  if (h_idx < horizon - 1) {
    *(float3 *)&d_vec_weight[0] *= *(float3 *)&run_vec_weight[0];
    *(float3 *)&d_vec_weight[3] *= *(float3 *)&run_vec_weight[3];
  }
 

  // read offset
  int offset = batch_pose_idx[batch_idx];
  if (mode == BATCH_GOALSET || mode == BATCH_GOAL) {
    offset = (offset)*num_goals;
  }

  for (int k = 0; k < num_goals; k++) {

    l_goal_position = *(float3 *)&goal_position[(offset + k) * 3];
    float4 gq4 = *(float4 *)&goal_quat[(offset + k) * 4];
    l_goal_quat = make_float4(gq4.y, gq4.z, gq4.w, gq4.x);
      transform_vec_quat(l_goal_quat, &d_vec_weight[0], &des_vec_weight[0]);

    compute_distance(&distance_vec[0], pose_distance, position_distance,
                     rotation_distance, position, l_goal_position, quat,
                     l_goal_quat,
                     &des_vec_weight[0],      //&l_vec_weight[0],
                     &d_vec_convergence[0], //&l_vec_convergence[0],
                     position_weight, rotation_weight);
    if (pose_distance <= best_distance) {
      best_idx = k;
      best_distance = pose_distance;
      best_position_distance = position_distance;
      best_rotation_distance = rotation_distance;
      if (write_grad) {
        //inv_transform_vec_quat(l_goal_quat, &d_vec_weight[0], &best_des_vec_weight[0]);
#pragma unroll 6
        for (int i = 0; i < 6; i++) {
          best_distance_vec[i] = distance_vec[i];
          best_des_vec_weight[i] = des_vec_weight[i];
        }
      }
    }
  }
  // write out:

  // write out pose distance:
  out_distance[batch_idx * horizon + h_idx] = best_distance;
  if (write_distance) {
    if(position_weight == 0.0)
    {
      best_position_distance = 0.0;
    }
    if (rotation_weight == 0.0)
    {
      best_rotation_distance = 0.0;
    }
    out_position_distance[batch_idx * horizon + h_idx] = best_position_distance;
    out_rotation_distance[batch_idx * horizon + h_idx] = best_rotation_distance;
  }
  out_gidx[batch_idx * horizon + h_idx] = best_idx;

  if (write_grad) {
    if (write_distance) {
      position_weight = 1;
      rotation_weight = 1;
    }
    if (best_position_distance > 0) {
      best_position_distance = (position_weight / best_position_distance);

      out_p_vec[batch_idx * horizon * 3 + h_idx * 3] =
          best_des_vec_weight[3] * best_distance_vec[0] * best_position_distance;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 1] =
          best_des_vec_weight[4] * best_distance_vec[1] * best_position_distance;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 2] =
          best_des_vec_weight[5] * best_distance_vec[2] * best_position_distance;

    } else {
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3] = 0.0;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 1] = 0.0;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 2] = 0.0;
    }

    if (best_rotation_distance > 0) {
      best_rotation_distance = rotation_weight / best_rotation_distance;

      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 1] =
          best_des_vec_weight[0] * best_distance_vec[3] * best_rotation_distance;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 2] =
          best_des_vec_weight[1] * best_distance_vec[4] * best_rotation_distance;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 3] =
          best_des_vec_weight[2] * best_distance_vec[5] * best_rotation_distance;
    } else {
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 1] = 0.0;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 2] = 0.0;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 3] = 0.0;
    }
  }
}

template <typename scalar_t, bool write_distance>
__global__ void goalset_pose_distance_metric_kernel(
    scalar_t *out_distance, scalar_t *out_position_distance,
    scalar_t *out_rotation_distance, scalar_t *out_p_vec, scalar_t *out_q_vec,
    int32_t *out_gidx, const scalar_t *current_position,
    const scalar_t *goal_position, const scalar_t *current_quat,
    const scalar_t *goal_quat, const scalar_t *vec_weight,
    const scalar_t *weight, const scalar_t *vec_convergence,
    const scalar_t *run_weight, const scalar_t *run_vec_weight,
    const int32_t *batch_pose_idx, const int mode, const int num_goals,
    const int batch_size, const int horizon, const bool write_grad = false) {
  const int t_idx = (blockDim.x * blockIdx.x + threadIdx.x);
  const int batch_idx = t_idx / horizon;
  const int h_idx = t_idx - (batch_idx * horizon);
  if (batch_idx >= batch_size || h_idx >= horizon) {
    return;
  }

  // read current pose:
  float3 position =
      *(float3 *)&current_position[batch_idx * horizon * 3 + h_idx * 3];
  float4 quat_4 = *(float4 *)&current_quat[batch_idx * horizon * 4 + h_idx * 4];
  float4 quat = make_float4(quat_4.y, quat_4.z, quat_4.w, quat_4.x);

  // read weights:

  float position_weight = weight[1];
  float p_w_alpha = weight[3];
  float r_w_alpha = weight[2];
  float rotation_weight = weight[0];
  if (!write_distance) {
    position_weight *= run_weight[h_idx];
    rotation_weight *= run_weight[h_idx];
    p_w_alpha *= run_weight[h_idx];
    r_w_alpha *= run_weight[h_idx];
    if (position_weight == 0.0 && rotation_weight == 0.0) {
      return;
    }
  }

  float d_vec_convergence[2];
  *(float2 *)&d_vec_convergence[0] = *(float2 *)&vec_convergence[0];

  float d_vec_weight[6];
  *(float3 *)&d_vec_weight[0] = *(float3 *)&vec_weight[0];
  *(float3 *)&d_vec_weight[3] = *(float3 *)&vec_weight[3];
  if (h_idx < horizon - 1) {
    *(float3 *)&d_vec_weight[0] *= *(float3 *)&run_vec_weight[0];
    *(float3 *)&d_vec_weight[3] *= *(float3 *)&run_vec_weight[3];
  }

  float des_vec_weight[6] = {0.0};
  float3 l_goal_position;
  float4 l_goal_quat;
  float distance_vec[6]; //  = {0.0};
  float pose_distance = 0.0;
  float position_distance = 0.0;
  float rotation_distance = 0.0;
  float best_distance = INFINITY;
  float best_position_distance = 0.0;
  float best_rotation_distance = 0.0;
  float best_distance_vec[6] = {0.0};
  float best_des_vec_weight[6] = {0.0};
  int best_idx = -1.0;
  int offset = batch_pose_idx[batch_idx];
  if (mode == BATCH_GOALSET || mode == BATCH_GOAL) {
    offset = (offset)*num_goals;
  }

  for (int k = 0; k < num_goals; k++) {

    l_goal_position = *(float3 *)&goal_position[(offset + k) * 3];
    float4 gq4 =
        *(float4 *)&goal_quat[(offset + k) * 4]; // reading qw, qx, qy, qz
    l_goal_quat = make_float4(gq4.y, gq4.z, gq4.w, gq4.x);

    transform_vec_quat(l_goal_quat, &d_vec_weight[0], &des_vec_weight[0]);
    compute_metric_distance(
        &distance_vec[0], pose_distance, position_distance, rotation_distance,
        position, l_goal_position, quat, l_goal_quat,
        &des_vec_weight[0],      //&l_vec_weight[0],
        &d_vec_convergence[0], //&l_vec_convergence[0],
        position_weight, p_w_alpha, rotation_weight, r_w_alpha);
    if (pose_distance <= best_distance) {
      best_idx = k;
      best_distance = pose_distance;
      best_position_distance = position_distance;
      best_rotation_distance = rotation_distance;
      if (write_grad) {
         // inv_transform_vec_quat(l_goal_quat, &d_vec_weight[0], &best_des_vec_weight[0]);

#pragma unroll 6
        for (int i = 0; i < 6; i++) {
          best_distance_vec[i] = distance_vec[i];
         best_des_vec_weight[i] = des_vec_weight[i];
        }
      }
    }
  }

  // add scaling metric:
  // best_distance = log2f(acoshf(best_distance));
  // write out:

  // write out pose distance:
  out_distance[batch_idx * horizon + h_idx] = best_distance;
  if (write_distance) {
    if(position_weight == 0.0)
    {
      best_position_distance = 0.0;
    }
    if (rotation_weight == 0.0)
    {
      best_rotation_distance = 0.0;
    }
    out_position_distance[batch_idx * horizon + h_idx] = best_position_distance;
    out_rotation_distance[batch_idx * horizon + h_idx] = best_rotation_distance;
  }
  out_gidx[batch_idx * horizon + h_idx] = best_idx;

  if (write_grad) {
    // write gradient
    // compute scalar term:
    // -w * (d_vec)/ (length * length(negative +1) * acos(length)
    // -w * (d_vec) * sinh(length) / (length * cosh(length))
    // compute extra term:

    if (write_distance) {
      position_weight = 1.0;
      rotation_weight = 1.0;
    }

    if (best_position_distance > 0) {
      best_position_distance =
          (p_w_alpha * position_weight *
           sinhf(p_w_alpha * best_position_distance)) /
          (best_position_distance * coshf(p_w_alpha * best_position_distance));

      // best_position_distance = (position_weight/best_position_distance);
      //  comput scalar gradient

      out_p_vec[batch_idx * horizon * 3 + h_idx * 3] =
          best_des_vec_weight[3] * best_distance_vec[0] * best_position_distance;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 1] =
          best_des_vec_weight[4] * best_distance_vec[1] * best_position_distance;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 2] =
          best_des_vec_weight[5] * best_distance_vec[2] * best_position_distance;

    } else {
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3] = 0.0;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 1] = 0.0;
      out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 2] = 0.0;
    }

    if (best_rotation_distance > 0) {
      best_rotation_distance =
          (r_w_alpha * rotation_weight *
           sinhf(r_w_alpha * best_rotation_distance)) /
          (best_rotation_distance * coshf(r_w_alpha * best_rotation_distance));

      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 1] =
          best_des_vec_weight[0] * best_distance_vec[3] * best_rotation_distance;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 2] =
          best_des_vec_weight[1] * best_distance_vec[4] * best_rotation_distance;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 3] =
          best_des_vec_weight[2] * best_distance_vec[5] * best_rotation_distance;
    } else {
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 1] = 0.0;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 2] = 0.0;
      out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 3] = 0.0;
    }
  }
}

} // namespace
}

std::vector<torch::Tensor>
pose_distance(torch::Tensor out_distance, torch::Tensor out_position_distance,
              torch::Tensor out_rotation_distance,
              torch::Tensor distance_p_vector, // batch size, 3
              torch::Tensor distance_q_vector, // batch size, 4
              torch::Tensor out_gidx,
              const torch::Tensor current_position, // batch_size, 3
              const torch::Tensor goal_position,    // n_boxes, 3
              const torch::Tensor current_quat, const torch::Tensor goal_quat,
              const torch::Tensor vec_weight, // n_boxes, 4, 4
              const torch::Tensor weight, const torch::Tensor vec_convergence,
              const torch::Tensor run_weight,
              const torch::Tensor run_vec_weight,
              const torch::Tensor batch_pose_idx, // batch_size, 1
              const int batch_size, const int horizon, const int mode,
              const int num_goals = 1, const bool compute_grad = false,
              const bool write_distance = true, const bool use_metric = false) {
  using namespace Curobo::Pose;
  // we compute the warp threads based on number of boxes:
  assert(batch_pose_idx.size(0) == batch_size);
  // TODO: verify this math
  // const int batch_size = out_distance.size(0);
  assert(run_weight.size(-1) == horizon);
  const int bh = batch_size * horizon;
  int threadsPerBlock = bh;
  if (bh > 128) {
    threadsPerBlock = 128;
  }

  // we fit warp thread spheres in a threadsPerBlock

  int blocksPerGrid = (bh + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (use_metric) {
    if (write_distance)
    {
       AT_DISPATCH_FLOATING_TYPES(
        current_position.scalar_type(), "batch_pose_distance", ([&] {
          goalset_pose_distance_metric_kernel
              <scalar_t, true><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                  out_distance.data_ptr<scalar_t>(),
                  out_position_distance.data_ptr<scalar_t>(),
                  out_rotation_distance.data_ptr<scalar_t>(),
                  distance_p_vector.data_ptr<scalar_t>(),
                  distance_q_vector.data_ptr<scalar_t>(),
                  out_gidx.data_ptr<int32_t>(),
                  current_position.data_ptr<scalar_t>(),
                  goal_position.data_ptr<scalar_t>(),
                  current_quat.data_ptr<scalar_t>(),
                  goal_quat.data_ptr<scalar_t>(),
                  vec_weight.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                  vec_convergence.data_ptr<scalar_t>(),
                  run_weight.data_ptr<scalar_t>(),
                  run_vec_weight.data_ptr<scalar_t>(),
                  batch_pose_idx.data_ptr<int32_t>(), mode, num_goals,
                  batch_size, horizon, compute_grad);
        }));
    
    }
    else
    {

    
    AT_DISPATCH_FLOATING_TYPES(
        current_position.scalar_type(), "batch_pose_distance", ([&] {
          goalset_pose_distance_metric_kernel
              // goalset_pose_distance_kernel
              <scalar_t, false><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                  out_distance.data_ptr<scalar_t>(),
                  out_position_distance.data_ptr<scalar_t>(),
                  out_rotation_distance.data_ptr<scalar_t>(),
                  distance_p_vector.data_ptr<scalar_t>(),
                  distance_q_vector.data_ptr<scalar_t>(),
                  out_gidx.data_ptr<int32_t>(),
                  current_position.data_ptr<scalar_t>(),
                  goal_position.data_ptr<scalar_t>(),
                  current_quat.data_ptr<scalar_t>(),
                  goal_quat.data_ptr<scalar_t>(),
                  vec_weight.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                  vec_convergence.data_ptr<scalar_t>(),
                  run_weight.data_ptr<scalar_t>(),
                  run_vec_weight.data_ptr<scalar_t>(),
                  batch_pose_idx.data_ptr<int32_t>(), mode, num_goals,
                  batch_size, horizon, compute_grad);
        }));
    }
  } else {
    if(write_distance)
    {
       AT_DISPATCH_FLOATING_TYPES(
        current_position.scalar_type(), "batch_pose_distance", ([&] {
          // goalset_pose_distance_metric_kernel
          goalset_pose_distance_kernel<scalar_t, true>
              <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                  out_distance.data_ptr<scalar_t>(),
                  out_position_distance.data_ptr<scalar_t>(),
                  out_rotation_distance.data_ptr<scalar_t>(),
                  distance_p_vector.data_ptr<scalar_t>(),
                  distance_q_vector.data_ptr<scalar_t>(),
                  out_gidx.data_ptr<int32_t>(),
                  current_position.data_ptr<scalar_t>(),
                  goal_position.data_ptr<scalar_t>(),
                  current_quat.data_ptr<scalar_t>(),
                  goal_quat.data_ptr<scalar_t>(),
                  vec_weight.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                  vec_convergence.data_ptr<scalar_t>(),
                  run_weight.data_ptr<scalar_t>(),
                  run_vec_weight.data_ptr<scalar_t>(),
                  batch_pose_idx.data_ptr<int32_t>(), mode, num_goals,
                  batch_size, horizon, compute_grad);
                          }));
    }
    else
    {
    AT_DISPATCH_FLOATING_TYPES(
        current_position.scalar_type(), "batch_pose_distance", ([&] {
          // goalset_pose_distance_metric_kernel
          goalset_pose_distance_kernel<scalar_t, false>
              <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                  out_distance.data_ptr<scalar_t>(),
                  out_position_distance.data_ptr<scalar_t>(),
                  out_rotation_distance.data_ptr<scalar_t>(),
                  distance_p_vector.data_ptr<scalar_t>(),
                  distance_q_vector.data_ptr<scalar_t>(),
                  out_gidx.data_ptr<int32_t>(),
                  current_position.data_ptr<scalar_t>(),
                  goal_position.data_ptr<scalar_t>(),
                  current_quat.data_ptr<scalar_t>(),
                  goal_quat.data_ptr<scalar_t>(),
                  vec_weight.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                  vec_convergence.data_ptr<scalar_t>(),
                  run_weight.data_ptr<scalar_t>(),
                  run_vec_weight.data_ptr<scalar_t>(),
                  batch_pose_idx.data_ptr<int32_t>(), mode, num_goals,
                  batch_size, horizon, compute_grad);
        }));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out_distance,      out_position_distance, out_rotation_distance,
          distance_p_vector, distance_q_vector,     out_gidx};
}

std::vector<torch::Tensor>
backward_pose_distance(torch::Tensor out_grad_p, torch::Tensor out_grad_q,
                       const torch::Tensor grad_distance,   // batch_size, 3
                       const torch::Tensor grad_p_distance, // n_boxes, 3
                       const torch::Tensor grad_q_distance,
                       const torch::Tensor pose_weight,
                       const torch::Tensor grad_p_vec, // n_boxes, 4, 4
                       const torch::Tensor grad_q_vec, const int batch_size,
                       const bool use_distance = false) {

  // we compute the warp threads based on number of boxes:

  // TODO: verify this math
  // const int batch_size = grad_distance.size(0);
  using namespace Curobo::Pose;

  int threadsPerBlock = batch_size;
  if (batch_size > 128) {
    threadsPerBlock = 128;
  }

  // we fit warp thread spheres in a threadsPerBlock

  int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_distance) {

    AT_DISPATCH_FLOATING_TYPES(
        grad_distance.scalar_type(), "backward_pose_distance", ([&] {
          backward_pose_distance_kernel<scalar_t>
              <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                  out_grad_p.data_ptr<scalar_t>(),
                  out_grad_q.data_ptr<scalar_t>(),
                  grad_distance.data_ptr<scalar_t>(),
                  grad_p_distance.data_ptr<scalar_t>(),
                  grad_q_distance.data_ptr<scalar_t>(),
                  pose_weight.data_ptr<scalar_t>(),
                  grad_p_vec.data_ptr<scalar_t>(),
                  grad_q_vec.data_ptr<scalar_t>(), batch_size);
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        grad_distance.scalar_type(), "backward_pose", ([&] {
          backward_pose_kernel<scalar_t>
              <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                  out_grad_p.data_ptr<scalar_t>(),
                  out_grad_q.data_ptr<scalar_t>(),
                  grad_distance.data_ptr<scalar_t>(),
                  pose_weight.data_ptr<scalar_t>(),
                  grad_p_vec.data_ptr<scalar_t>(),
                  grad_q_vec.data_ptr<scalar_t>(), batch_size);
        }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out_grad_p, out_grad_q};
}