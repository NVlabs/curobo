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
    transform_error_quat(const float4 q, // x,y,z, qw, qx,qy,qz
                         const float3 error,
                         float       *result)
    {
      // do dot product:
      if ((q.x != 0) || (q.y != 0) || (q.z != 0))
      {
        result[0] = q.w * q.w * error.x + 2 * q.y * q.w * error.z -
                    2 * q.z * q.w * error.y + q.x * q.x * error.x +
                    2 * q.y * q.x * error.y + 2 * q.z * q.x * error.z -
                    q.z * q.z * error.x - q.y * q.y * error.x;
        result[1] = 2 * q.x * q.y * error.x + q.y * q.y * error.y +
                    2 * q.z * q.y * error.z + 2 * q.w * q.z * error.x -
                    q.z * q.z * error.y + q.w * q.w * error.y -
                    2 * q.x * q.w * error.z - q.x * q.x * error.y;
        result[2] = 2 * q.x * q.z * error.x + 2 * q.y * q.z * error.y +
                    q.z * q.z * error.z - 2 * q.w * q.y * error.x -
                    q.y * q.y * error.z + 2 * q.w * q.x * error.y -
                    q.x * q.x * error.z + q.w * q.w * error.z;
      }
      else
      {
        result[0] = error.x;
        result[1] = error.y;
        result[2] = error.z;
      }
    }

    __device__ __forceinline__ void transform_point(const float3  frame_pos,
                                                    const float4  frame_quat,
                                                    const float3& point,
                                                    float3      & transformed_point)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      const float4 q = frame_quat;
      const float3 p = frame_pos;

      if ((q.x != 0) || (q.y != 0) || (q.z != 0))
      {
        transformed_point.x = p.x + q.w * q.w * point.x + 2 * q.y * q.w * point.z -
                              2 * q.z * q.w * point.y + q.x * q.x * point.x +
                              2 * q.y * q.x * point.y + 2 * q.z * q.x * point.z -
                              q.z * q.z * point.x - q.y * q.y * point.x;
        transformed_point.y = p.y + 2 * q.x * q.y * point.x + q.y * q.y * point.y +
                              2 * q.z * q.y * point.z + 2 * q.w * q.z * point.x -
                              q.z * q.z * point.y + q.w * q.w * point.y - 2 * q.x * q.w * point.z -
                              q.x * q.x * point.y;
        transformed_point.z = p.z + 2 * q.x * q.z * point.x + 2 * q.y * q.z * point.y +
                              q.z * q.z * point.z - 2 * q.w * q.y * point.x - q.y * q.y * point.z +
                              2 * q.w * q.x * point.y - q.x * q.x * point.z + q.w * q.w * point.z;
      }
      else
      {
        transformed_point.x = p.x + point.x;
        transformed_point.y = p.y + point.y;
        transformed_point.z = p.z + point.z;
      }
    }

    __device__ __forceinline__ void inv_transform_point(const float3  frame_pos,
                                                        const float4  frame_quat,
                                                        const float3& point,
                                                        float3      & transformed_point)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      float4 q = make_float4(-1 * frame_quat.x, -1 * frame_quat.y, -1 * frame_quat.z, frame_quat.w);
      float3 p = make_float3(0, 0, 0);

      transform_point(make_float3(0, 0, 0), q, frame_pos, p);
      p = -1.0 * p;

      if ((q.x != 0) || (q.y != 0) || (q.z != 0))
      {
        transformed_point.x = p.x + q.w * q.w * point.x + 2 * q.y * q.w * point.z -
                              2 * q.z * q.w * point.y + q.x * q.x * point.x +
                              2 * q.y * q.x * point.y + 2 * q.z * q.x * point.z -
                              q.z * q.z * point.x - q.y * q.y * point.x;
        transformed_point.y = p.y + 2 * q.x * q.y * point.x + q.y * q.y * point.y +
                              2 * q.z * q.y * point.z + 2 * q.w * q.z * point.x -
                              q.z * q.z * point.y + q.w * q.w * point.y - 2 * q.x * q.w * point.z -
                              q.x * q.x * point.y;
        transformed_point.z = p.z + 2 * q.x * q.z * point.x + 2 * q.y * q.z * point.y +
                              q.z * q.z * point.z - 2 * q.w * q.y * point.x - q.y * q.y * point.z +
                              2 * q.w * q.x * point.y - q.x * q.x * point.z + q.w * q.w * point.z;
      }
      else
      {
        transformed_point.x = p.x + point.x;
        transformed_point.y = p.y + point.y;
        transformed_point.z = p.z + point.z;
      }
    }

    __device__ __forceinline__ void inv_transform_quat(
      const float4  frame_quat,
      const float4& quat,
      float4      & transformed_quat)
    {
      // do dot product:
      // new_p = q * p * q_inv + obs_p
      float4 q = make_float4(-1 * frame_quat.x, -1 * frame_quat.y, -1 * frame_quat.z, frame_quat.w);

      if ((q.x != 0) || (q.y != 0) || (q.z != 0))
      {
        // multiply quats together new_q = q * quat;
        transformed_quat.w = q.w * quat.w - q.x * quat.x - q.y * quat.y - q.z * quat.z;
        transformed_quat.x = q.w * quat.x + quat.w * q.x + q.y * quat.z - quat.y * q.z;
        transformed_quat.y = q.w * quat.y + quat.w * q.y + q.z * quat.x - quat.z * q.x;
        transformed_quat.z = q.w * quat.z + quat.w * q.z + q.x * quat.y - quat.x * q.y;
      }
      else
      {
        transformed_quat = quat;
      }
    }

    __device__ __forceinline__ void
    compute_pose_distance_vector(float       *result_vec,
                                 const float3 goal_position,
                                 const float4 goal_quat,
                                 const float3 current_position,
                                 const float4 current_quat,
                                 const float *vec_weight,
                                 const float3 offset_position,
                                 const float3 offset_rotation,
                                 const bool   reach_offset,
                                 const bool project_distance)
    {
      // project current position to goal frame:
      float3 error_position = make_float3(0, 0, 0);


      float3 error_quat = make_float3(0, 0, 0);

      if (project_distance)
      {
        float4 projected_quat = make_float4(0, 0, 0, 0);

        inv_transform_point(goal_position, goal_quat, current_position, error_position);

        // project current quat to goal frame:
        inv_transform_quat(goal_quat, current_quat, projected_quat);


        float r_w = projected_quat.w;

        if (r_w < 0.0)
        {
          r_w = -1.0;
        }
        else
        {
          r_w = 1.0;
        }

        error_quat.x = r_w * (projected_quat.x);
        error_quat.y = r_w * (projected_quat.y);
        error_quat.z = r_w * (projected_quat.z);
      }
      else
      {
        error_position = current_position - goal_position;


        float r_w =
          (goal_quat.w * current_quat.w + goal_quat.x * current_quat.x + goal_quat.y *
           current_quat.y + goal_quat.z * current_quat.z);

        if (r_w < 0.0)
        {
          r_w = 1.0;
        }
        else
        {
          r_w = -1.0;
        }

        error_quat.x =  r_w *
                       (-1 * goal_quat.w * current_quat.x + current_quat.w * goal_quat.x -
                        goal_quat.y *
                        current_quat.z + current_quat.y * goal_quat.z);
        error_quat.y =  r_w *
                       (-1 * goal_quat.w * current_quat.y + current_quat.w * goal_quat.y -
                        goal_quat.z *
                        current_quat.x + current_quat.z * goal_quat.x);
        error_quat.z =  r_w *
                       (-1 * goal_quat.w * current_quat.z + current_quat.w * goal_quat.z -
                        goal_quat.x *
                        current_quat.y + current_quat.x * goal_quat.y);
      }


      if (reach_offset)
      {
        error_position = error_position + offset_position;
        error_quat     = error_quat + offset_rotation;
      }


      error_position = (*(float3 *)&vec_weight[3]) * error_position;

      error_quat = (*(float3 *)&vec_weight[0]) * error_quat;


      // compute rotation distance:


      if (project_distance)
      {
        // project this error back:

        transform_error_quat(goal_quat, error_quat,     &result_vec[3]);

        // projected distance back to world frame:
        transform_error_quat(goal_quat, error_position, &result_vec[0]);
      }
      else
      {
        *(float3 *)&result_vec[0] = error_position;
        *(float3 *)&result_vec[3] = error_quat;
      }
    }

    template<bool use_metric>
    __device__ __forceinline__ void
    compute_pose_distance(float *distance_vec, float& distance, float& position_distance,
                          float& rotation_distance, const float3 current_position,
                          const float3 goal_position, const float4 current_quat,
                          const float4 goal_quat, const float *vec_weight,
                          const float *vec_convergence, const float position_weight,
                          const float rotation_weight,
                          const float p_alpha,
                          const float r_alpha,
                          const float3 offset_position,
                          const float3 offset_rotation,
                          const bool reach_offset,
                          const bool project_distance)
    {
      compute_pose_distance_vector(&distance_vec[0],
                                    goal_position,
                                    goal_quat,
                                    current_position,
                                    current_quat,
                                    &vec_weight[0],
                                    offset_position,
                                    offset_rotation,
                                    reach_offset,
                                    project_distance);

      position_distance = 0;
      rotation_distance = 0;

      // scale by vec weight and position weight:
#pragma unroll 3

      for (int i = 0; i < 3; i++)
      {
        position_distance += distance_vec[i] * distance_vec[i];
      }
#pragma unroll 3

      for (int i = 3; i < 6; i++)
      {
        rotation_distance += distance_vec[i] * distance_vec[i];
      }

      distance = 0;

      if (rotation_distance > vec_convergence[0] * vec_convergence[0])
      {
        rotation_distance = sqrtf(rotation_distance);

        if (use_metric)
        {
          distance += rotation_weight * log2f(coshf(r_alpha * rotation_distance));
        }
        else
        {
          distance += rotation_weight * rotation_distance;
        }
      }

      if (position_distance > vec_convergence[1] * vec_convergence[1])
      {
        position_distance = sqrtf(position_distance);

        if (use_metric)
        {
          distance += position_weight * log2f(coshf(p_alpha * position_distance));
        }
        else
        {
          distance += position_weight * position_distance;
        }
      }
    }

    template<typename scalar_t>
    __global__ void
    backward_pose_distance_kernel(scalar_t       *out_grad_p,      // [b,3]
                                  scalar_t       *out_grad_q,      // [b,4]
                                  const scalar_t *grad_distance,   // [b,1]
                                  const scalar_t *grad_p_distance, // [b,1]
                                  const scalar_t *grad_q_distance, // [b,1]
                                  const scalar_t *pose_weight,     // [2]
                                  const scalar_t *grad_p_vec,      // [b,3]
                                  const scalar_t *grad_q_vec,      // [b,4]
                                  const int       batch_size)
    {
      const int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

      if (batch_idx >= batch_size)
      {
        return;
      }

      // read data
      const float  g_distance   = grad_distance[batch_idx];
      const float2 p_weight     = *(float2 *)&pose_weight[0];
      const float3 g_p_v        = *(float3 *)&grad_p_vec[batch_idx * 3];
      const float3 g_q_v        = *(float3 *)&grad_q_vec[batch_idx * 4 + 1];
      const float  g_p_distance = grad_p_distance[batch_idx];
      const float  g_q_distance = grad_q_distance[batch_idx];

      // compute position gradient
      float3 g_p =
        (g_p_v) * ((g_p_distance + g_distance * p_weight.y)); // scalar * float3
      float3 g_q =
        (g_q_v) * ((g_q_distance + g_distance * p_weight.x)); // scalar * float3

      // write out
      *(float3 *)&out_grad_p[batch_idx * 3]     = g_p;
      *(float3 *)&out_grad_q[batch_idx * 4 + 1] = g_q;
    }

    template<typename scalar_t>
    __global__ void backward_pose_kernel(scalar_t       *out_grad_p,    // [b,3]
                                         scalar_t       *out_grad_q,    // [b,4]
                                         const scalar_t *grad_distance, // [b,1]
                                         const scalar_t *pose_weight,   // [2]
                                         const scalar_t *grad_p_vec,    // [b,3]
                                         const scalar_t *grad_q_vec,    // [b,4]
                                         const int       batch_size)
    {
      const int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

      if (batch_idx >= batch_size)
      {
        return;
      }

      // read data
      const float  g_distance = grad_distance[batch_idx];
      const float2 p_weight   = *(float2 *)&pose_weight[0];
      const float3 g_p_v      = *(float3 *)&grad_p_vec[batch_idx * 3];
      const float3 g_q_v      = *(float3 *)&grad_q_vec[batch_idx * 4 + 1];

      // compute position gradient
      float3 g_p = (g_p_v) * ((g_distance * p_weight.y)); // scalar * float3
      float3 g_q = (g_q_v) * ((g_distance * p_weight.x)); // scalar * float3

      // write out
      *(float3 *)&out_grad_p[batch_idx * 3]     = g_p;
      *(float3 *)&out_grad_q[batch_idx * 4 + 1] = g_q;
    }

    template<typename scalar_t, bool write_distance, bool use_metric>
    __global__ void goalset_pose_distance_kernel(
      scalar_t *out_distance, scalar_t *out_position_distance,
      scalar_t *out_rotation_distance, scalar_t *out_p_vec, scalar_t *out_q_vec,
      int32_t *out_gidx, const scalar_t *current_position,
      const scalar_t *goal_position, const scalar_t *current_quat,
      const scalar_t *goal_quat, const scalar_t *vec_weight,
      const scalar_t *weight, const scalar_t *vec_convergence,
      const scalar_t *run_weight, const scalar_t *run_vec_weight,
      const scalar_t *offset_waypoint,
      const scalar_t *offset_tstep_fraction,
      const int32_t *batch_pose_idx,
      const uint8_t *project_distance_tensor,
      const int mode, const int num_goals,
      const int batch_size, const int horizon, const bool write_grad = false)
    {
      const int t_idx     = (blockDim.x * blockIdx.x + threadIdx.x);
      const int batch_idx = t_idx / horizon;
      const int h_idx     = t_idx - (batch_idx * horizon);

      if ((batch_idx >= batch_size) || (h_idx >= horizon))
      {
        return;
      }
      const bool project_distance = project_distance_tensor[0];
      // read current pose:
      float3 position =
        *(float3 *)&current_position[batch_idx * horizon * 3 + h_idx * 3];
      float4 quat_4 = *(float4 *)&current_quat[batch_idx * horizon * 4 + h_idx * 4];
      float4 quat   = make_float4(quat_4.y, quat_4.z, quat_4.w, quat_4.x);

      // read weights:
      float rotation_weight          = weight[0];
      float position_weight          = weight[1];
      float r_w_alpha                = weight[2];
      float p_w_alpha                = weight[3];
      bool  reach_offset             = false;
      const float offset_tstep_ratio = offset_tstep_fraction[0];
      int offset_tstep               = floorf(offset_tstep_ratio * horizon); // if offset_tstep
                                                                                // is ? horizon, not
                                                                                // in this mode
      float d_vec_weight[6] = { 0.0 };
      #pragma unroll 6
      for (int k = 0; k < 6; k++)
      {
        d_vec_weight[k] = vec_weight[k];
      }
      //*(float3 *)&d_vec_weight[0] = *(float3 *)&vec_weight[0]; // TODO
      //*(float3 *)&d_vec_weight[3] = *(float3 *)&vec_weight[3];
      float3 offset_rotation = *(float3 *)&offset_waypoint[0];
      float3 offset_position = *(float3 *)&offset_waypoint[3];

      if ((h_idx < horizon - 1) && (h_idx != horizon - offset_tstep))
      {
        #pragma unroll 6
        for (int k = 0; k < 6; k++)
        {
          d_vec_weight[k] *= run_vec_weight[k];
        }
        //*(float3 *)&d_vec_weight[0] *= *(float3 *)&run_vec_weight[0];
        //*(float3 *)&d_vec_weight[3] *= *(float3 *)&run_vec_weight[3];
      }

      if (!write_distance)
      {
        position_weight *= run_weight[h_idx];
        rotation_weight *= run_weight[h_idx];
        float sum_weight = 0;

        #pragma unroll 6
        for (int i = 0; i < 6; i++)
        {
          sum_weight += d_vec_weight[i];
        }

        if (((position_weight == 0.0) && (rotation_weight == 0.0)) || (sum_weight == 0.0))
        {
          return;
        }
      }

      if ((horizon > 1) && (offset_tstep >= 0 && (h_idx == horizon - offset_tstep)))
      {
        reach_offset = true;
      }

      float3 l_goal_position;
      float4 l_goal_quat;
      float  distance_vec[6]; //  = {0.0};
      float  pose_distance          = 0.0;
      float  position_distance      = 0.0;
      float  rotation_distance      = 0.0;
      float  best_distance          = INFINITY;
      float  best_position_distance = 0.0;
      float  best_rotation_distance = 0.0;
      float  best_distance_vec[6]   = { 0.0 };
      float  d_vec_convergence[2];

      //*(float2 *)&d_vec_convergence[0] = *(float2 *)&vec_convergence[0]; // TODO
      d_vec_convergence[0] = vec_convergence[0];
      d_vec_convergence[1] = vec_convergence[1];

      int best_idx = -1;


      // read offset
      int offset = batch_pose_idx[batch_idx];

      if ((mode == BATCH_GOALSET) || (mode == BATCH_GOAL))
      {
        offset = (offset) * num_goals;
      }

      for (int k = 0; k < num_goals; k++)
      {
        l_goal_position = *(float3 *)&goal_position[(offset + k) * 3];
        float4 gq4 = *(float4 *)&goal_quat[(offset + k) * 4];
        l_goal_quat = make_float4(gq4.y, gq4.z, gq4.w, gq4.x);

        compute_pose_distance<use_metric>(&distance_vec[0],
                                                            pose_distance,
                                                            position_distance,
                                                            rotation_distance,
                                                            position,
                                                            l_goal_position,
                                                            quat,
                                                            l_goal_quat,
                                                            &d_vec_weight[0],

// &l_vec_weight[0],
                                                            &d_vec_convergence[0],

// &l_vec_convergence[0],
                                                            position_weight,
                                                            rotation_weight,
                                                            p_w_alpha,
                                                            r_w_alpha,
                                                            offset_position,
                                                            offset_rotation,
                                                            reach_offset,
                                                            project_distance);

        if (pose_distance <= best_distance)
        {
          best_idx               = k;
          best_distance          = pose_distance;
          best_position_distance = position_distance;
          best_rotation_distance = rotation_distance;

          if (write_grad)
          {
        #pragma unroll 6

            for (int i = 0; i < 6; i++)
            {
              best_distance_vec[i] = distance_vec[i];
            }
          }
        }
      }

      // write out:

      // write out pose distance:
      out_distance[batch_idx * horizon + h_idx] = best_distance;

      if (write_distance)
      {
        if (position_weight == 0.0)
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

      if (write_grad)
      {
        if (write_distance)
        {
          position_weight = 1;
          rotation_weight = 1;
        }

        if (best_position_distance > 0)
        {
          if (use_metric)
          {
            best_position_distance =
              (p_w_alpha * position_weight *
               sinhf(p_w_alpha * best_position_distance)) /
              (best_position_distance * coshf(p_w_alpha * best_position_distance));
          }
          else
          {
            best_position_distance = (position_weight / best_position_distance);
          }

          out_p_vec[batch_idx * horizon * 3 + h_idx * 3] =
            best_distance_vec[0] * best_position_distance;
          out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 1] =
            best_distance_vec[1] * best_position_distance;
          out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 2] =
            best_distance_vec[2] * best_position_distance;
        }
        else
        {
          out_p_vec[batch_idx * horizon * 3 + h_idx * 3]     = 0.0;
          out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 1] = 0.0;
          out_p_vec[batch_idx * horizon * 3 + h_idx * 3 + 2] = 0.0;
        }

        if (best_rotation_distance > 0)
        {
          if (use_metric)
          {
            best_rotation_distance =
              (r_w_alpha * rotation_weight *
               sinhf(r_w_alpha * best_rotation_distance)) /
              (best_rotation_distance * coshf(r_w_alpha * best_rotation_distance));
          }
          else
          {
            best_rotation_distance = rotation_weight / best_rotation_distance;
          }

          out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 1] =
            best_distance_vec[3] * best_rotation_distance;
          out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 2] =
            best_distance_vec[4] * best_rotation_distance;
          out_q_vec[batch_idx * horizon * 4 + h_idx * 4 + 3] =
            best_distance_vec[5] * best_rotation_distance;
        }
        else
        {
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
              torch::Tensor distance_p_vector,      // batch size, 3
              torch::Tensor distance_q_vector,      // batch size, 4
              torch::Tensor out_gidx,
              const torch::Tensor current_position, // batch_size, 3
              const torch::Tensor goal_position,    // n_boxes, 3
              const torch::Tensor current_quat, const torch::Tensor goal_quat,
              const torch::Tensor vec_weight,       // n_boxes, 4, 4
              const torch::Tensor weight, const torch::Tensor vec_convergence,
              const torch::Tensor run_weight,
              const torch::Tensor run_vec_weight,
              const torch::Tensor offset_waypoint,
              const torch::Tensor offset_tstep_fraction,
              const torch::Tensor batch_pose_idx, // batch_size, 1
              const torch::Tensor project_distance,
              const int batch_size, const int horizon, const int mode,
              const int num_goals = 1, const bool compute_grad = false,
              const bool write_distance = true, const bool use_metric = false)
{
  using namespace Curobo::Pose;

  // we compute the warp threads based on number of boxes:
  assert(batch_pose_idx.size(0) == batch_size);

  // TODO: verify this math
  // const int batch_size = out_distance.size(0);
  assert(run_weight.size(-1) == horizon);
  const int bh        = batch_size * horizon;
  int threadsPerBlock = bh;

  if (bh > 128)
  {
    threadsPerBlock = 128;
  }

  // we fit warp thread spheres in a threadsPerBlock

  int blocksPerGrid = (bh + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();


    if (use_metric)
    {
      if (write_distance)
      {
        AT_DISPATCH_FLOATING_TYPES(
          current_position.scalar_type(), "batch_pose_distance", ([&] {
          goalset_pose_distance_kernel
          <scalar_t, true, true><< < blocksPerGrid, threadsPerBlock, 0,
            stream >> > (
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
            offset_waypoint.data_ptr<scalar_t>(),
            offset_tstep_fraction.data_ptr<scalar_t>(),
            batch_pose_idx.data_ptr<int32_t>(),
            project_distance.data_ptr<uint8_t>(),
            mode, num_goals,
            batch_size, horizon, compute_grad);
        }));
      }
      else
      {
        AT_DISPATCH_FLOATING_TYPES(
          current_position.scalar_type(), "batch_pose_distance", ([&] {
          goalset_pose_distance_kernel
          <scalar_t, false, true><< < blocksPerGrid, threadsPerBlock, 0,
            stream >> > (
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
            offset_waypoint.data_ptr<scalar_t>(),
            offset_tstep_fraction.data_ptr<scalar_t>(),
            batch_pose_idx.data_ptr<int32_t>(),
            project_distance.data_ptr<uint8_t>(),
            mode, num_goals,
            batch_size, horizon, compute_grad);
        }));
      }
    }
    else
    {
      if (write_distance)
      {
        AT_DISPATCH_FLOATING_TYPES(
          current_position.scalar_type(), "batch_pose_distance", ([&] {
          goalset_pose_distance_kernel<scalar_t, true, false>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
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
            offset_waypoint.data_ptr<scalar_t>(),
            offset_tstep_fraction.data_ptr<scalar_t>(),
            batch_pose_idx.data_ptr<int32_t>(),
            project_distance.data_ptr<uint8_t>(),
            mode, num_goals,
            batch_size, horizon, compute_grad);
        }));
      }
      else
      {
        AT_DISPATCH_FLOATING_TYPES(
          current_position.scalar_type(), "batch_pose_distance", ([&] {
          goalset_pose_distance_kernel<scalar_t, false, false>
            << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
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
            offset_waypoint.data_ptr<scalar_t>(),
            offset_tstep_fraction.data_ptr<scalar_t>(),
            batch_pose_idx.data_ptr<int32_t>(),
            project_distance.data_ptr<uint8_t>(),
            mode, num_goals,
            batch_size, horizon, compute_grad);
        }));
      }
    }


  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_distance,      out_position_distance, out_rotation_distance,
           distance_p_vector, distance_q_vector,     out_gidx };
}

std::vector<torch::Tensor>
backward_pose_distance(torch::Tensor out_grad_p, torch::Tensor out_grad_q,
                       const torch::Tensor grad_distance,   // batch_size, 3
                       const torch::Tensor grad_p_distance, // n_boxes, 3
                       const torch::Tensor grad_q_distance,
                       const torch::Tensor pose_weight,
                       const torch::Tensor grad_p_vec,      // n_boxes, 4, 4
                       const torch::Tensor grad_q_vec, const int batch_size,
                       const bool use_distance = false)
{
  // we compute the warp threads based on number of boxes:

  // TODO: verify this math
  // const int batch_size = grad_distance.size(0);
  using namespace Curobo::Pose;

  int threadsPerBlock = batch_size;

  if (batch_size > 128)
  {
    threadsPerBlock = 128;
  }

  // we fit warp thread spheres in a threadsPerBlock

  int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_distance)
  {
    AT_DISPATCH_FLOATING_TYPES(
      grad_distance.scalar_type(), "backward_pose_distance", ([&] {
      backward_pose_distance_kernel<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_grad_p.data_ptr<scalar_t>(),
        out_grad_q.data_ptr<scalar_t>(),
        grad_distance.data_ptr<scalar_t>(),
        grad_p_distance.data_ptr<scalar_t>(),
        grad_q_distance.data_ptr<scalar_t>(),
        pose_weight.data_ptr<scalar_t>(),
        grad_p_vec.data_ptr<scalar_t>(),
        grad_q_vec.data_ptr<scalar_t>(), batch_size);
    }));
  }
  else
  {
    AT_DISPATCH_FLOATING_TYPES(
      grad_distance.scalar_type(), "backward_pose", ([&] {
      backward_pose_kernel<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_grad_p.data_ptr<scalar_t>(),
        out_grad_q.data_ptr<scalar_t>(),
        grad_distance.data_ptr<scalar_t>(),
        pose_weight.data_ptr<scalar_t>(),
        grad_p_vec.data_ptr<scalar_t>(),
        grad_q_vec.data_ptr<scalar_t>(), batch_size);
    }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_grad_p, out_grad_q };
}
