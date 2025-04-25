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
#define MAX_H 100
#define BWD_DIFF -1
#define CENTRAL_DIFF 0

namespace Curobo
{
  namespace TensorStep
  {
    template<typename scalar_t>
    __global__ void position_clique_loop_kernel(
      scalar_t *out_position_mem, scalar_t *out_velocity_mem,
      scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
      const scalar_t *u_position, const scalar_t *start_position,
      const scalar_t *start_velocity, const scalar_t *start_acceleration,
      const scalar_t *traj_dt, const int batch_size, const int horizon,
      const int dof)
    {
      // there are batch * horizon * dof threads:
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }

      const float dt =
        traj_dt[0]; // assume same dt across traj TODO: Implement variable dt

      // read start state:
      float u_arr[MAX_H];
      float out_pos[MAX_H], out_vel[MAX_H], out_acc[MAX_H], out_jerk[MAX_H];
      const int b_addrs = b_idx * horizon * dof;

#pragma unroll

      for (int i = 0; i < horizon; i++)
      {
        u_arr[i]    = u_position[b_addrs + i * dof + d_idx];
        out_pos[i]  = 0;
        out_vel[i]  = 0;
        out_acc[i]  = 0;
        out_jerk[i] = 0;
      }

      out_pos[0]  = start_position[b_idx * dof + d_idx];
      out_vel[0]  = start_velocity[b_idx * dof + d_idx];
      out_acc[0]  = start_acceleration[b_idx * dof + d_idx];
      out_jerk[0] = 0.0;

      for (int h_idx = 1; h_idx < horizon; h_idx++)
      {
        // read actions: batch, horizon
        out_pos[h_idx] = u_arr[h_idx - 1];
        out_vel[h_idx] = (out_pos[h_idx] - out_pos[h_idx - 1]) * dt;  // 1 - 0
        out_acc[h_idx] =
          (out_vel[h_idx] - out_vel[h_idx - 1]) * dt;                 // 2 - 2.0 * 1 + 1
        out_jerk[h_idx] = (out_acc[h_idx] - out_acc[h_idx - 1]) * dt; // -1 3 -3 1
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon; h_idx++)
      {
        out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_pos[h_idx]; // new_position;
        out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_vel[h_idx];
        out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_acc[h_idx];
        out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_jerk[h_idx];
      }
    }

    template<typename scalar_t>
    __device__ __forceinline__ void compute_backward_difference(scalar_t       *out_position_mem,
                                                                scalar_t       *out_velocity_mem,
                                                                scalar_t       *out_acceleration_mem,
                                                                scalar_t       *out_jerk_mem,
                                                                const scalar_t *u_position,
                                                                const scalar_t *start_position,
                                                                const scalar_t *start_velocity,
                                                                const scalar_t *start_acceleration,
                                                                const scalar_t *traj_dt,
                                                                const int       batch_size,
                                                                const int       horizon,
                                                                const int       dof,
                                                                const int       b_idx,
                                                                const int       h_idx,
                                                                const int       d_idx,
                                                                const int       b_offset)
    {
      const float dt =
        traj_dt[0]; // assume same dt across traj TODO: Implement variable dt

      // read start state:
      float in_pos[4];
      float st_pos = 0.0, st_vel = 0.0, st_acc = 0.0;

  #pragma unroll 4

      for (int i = 0; i < 4; i++)
      {
        in_pos[i] = 0.0;
      }

      float out_pos = 0.0, out_vel = 0.0, out_acc = 0.0, out_jerk = 0.0;
      const int b_addrs = b_idx * horizon * dof;

      if (h_idx < 4)
      {
        st_pos = start_position[b_offset * dof + d_idx];
        st_vel = start_velocity[b_offset * dof + d_idx];
        st_acc = start_acceleration[b_offset * dof + d_idx];
      }

      if (h_idx == 0)
      {
        out_pos = st_pos;
        out_vel = st_vel;
        out_acc = st_acc; //
      }
      else
      {
        if (h_idx == 1)
        {
          for (int i = 3; i < 4; i++)
          {
            in_pos[i] = u_position[b_addrs + (h_idx - 1 - 3 + i) * dof + d_idx];
          }


          in_pos[0] = st_pos - 2.0 * (st_vel - 0.5 * st_acc * dt) * dt;

          in_pos[1] = st_pos - (st_vel - 0.5 * st_acc * dt) * dt;

          in_pos[2] = st_pos;
        }
        else if (h_idx == 2)
        {
          for (int i = 2; i < 4; i++)
          {
            in_pos[i] = u_position[b_addrs + (h_idx - 1 - 3 + i) * dof + d_idx];
          }

          in_pos[0] = st_pos - (st_vel - 0.5 * st_acc * dt) * dt;

          in_pos[1] = st_pos;
        }
        else if (h_idx == 3)
        {
          for (int i = 1; i < 4; i++)
          {
            in_pos[i] = u_position[b_addrs + (h_idx - 1 - 3 + i) * dof + d_idx];
          }
          in_pos[0] = st_pos;
        }
        else // h_idx >= 4

        {
          for (int i = 0; i < 4; i++)
          {
            in_pos[i] = u_position[b_addrs + (h_idx - 1 - 3 + i) * dof + d_idx];
          }
        }

        out_pos  = in_pos[3];
        out_vel  = (-in_pos[2] + in_pos[3]) * dt;
        out_acc  = (in_pos[1] - 2 * in_pos[2] + in_pos[3]) * dt * dt;
        out_jerk = (-in_pos[0] + 3 * in_pos[1] - 3 * in_pos[2] + in_pos[3]) * dt * dt * dt;
      }


      // write out:
      out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
        out_pos; // new_position;
      out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx]     = out_vel;
      out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_acc;
      out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx]         = out_jerk;
    }

    template<typename scalar_t>
    __device__ __forceinline__ void compute_central_difference_v0(scalar_t       *out_position_mem,
                                                                  scalar_t       *out_velocity_mem,
                                                                  scalar_t       *out_acceleration_mem,
                                                                  scalar_t       *out_jerk_mem,
                                                                  const scalar_t *u_position,
                                                                  const scalar_t *start_position,
                                                                  const scalar_t *start_velocity,
                                                                  const scalar_t *start_acceleration,
                                                                  const scalar_t *traj_dt,
                                                                  const int       batch_size,
                                                                  const int       horizon,
                                                                  const int       dof,
                                                                  const int       b_idx,
                                                                  const int       h_idx,
                                                                  const int       d_idx,
                                                                  const int       b_offset)
    {
      const float dt = traj_dt[0]; // assume same dt across traj TODO: Implement variable dt
      // dt here is actually 1/dt;

      // read start state:
      float out_pos = 0.0, out_vel = 0.0, out_acc = 0.0, out_jerk = 0.0;
      float st_pos = 0.0, st_vel = 0.0, st_acc = 0.0;

      const int b_addrs = b_idx * horizon * dof;
      float     in_pos[5]; // create a 5 value scalar

  #pragma unroll 5

      for (int i = 0; i < 5; i++)
      {
        in_pos[i] = 0.0;
      }

      if (h_idx < 4)
      {
        st_pos = start_position[b_offset * dof + d_idx];
        st_vel = start_velocity[b_offset * dof + d_idx];
        st_acc = start_acceleration[b_offset * dof + d_idx];
      }

      if (h_idx == 0)
      {
        out_pos = st_pos;
        out_vel = st_vel;
        out_acc = st_acc;
      }
      else if (h_idx < horizon - 2)
      {
        if (h_idx == 1)
        {
          in_pos[0] = st_pos - dt * (st_vel - (0.5 * st_acc * dt)); // start -1, start, u0, u1
          in_pos[1] = st_pos;
          in_pos[2] = u_position[b_addrs + (h_idx - 1) * dof + d_idx];
          in_pos[3] = u_position[b_addrs + (h_idx - 1 + 1) * dof + d_idx];
          in_pos[4] = u_position[b_addrs + (h_idx - 1 + 2) * dof + d_idx];
        }
        else if (h_idx == 2)
        {
          in_pos[0] = start_position[b_offset * dof + d_idx];
          in_pos[1] = u_position[b_addrs + (h_idx - 1 - 1) * dof + d_idx];
          in_pos[2] = u_position[b_addrs + (h_idx - 1) * dof + d_idx];
          in_pos[3] = u_position[b_addrs + (h_idx - 1 + 1) * dof + d_idx];
          in_pos[4] = u_position[b_addrs + (h_idx - 1 + 2) * dof + d_idx]; \

        }


        else if (h_idx > 2)
        {
          in_pos[0] = u_position[b_addrs + (h_idx - 1 - 2) * dof + d_idx];
          in_pos[1] = u_position[b_addrs + (h_idx - 1 - 1) * dof + d_idx];
          in_pos[2] = u_position[b_addrs + (h_idx - 1) * dof + d_idx];
          in_pos[3] = u_position[b_addrs + (h_idx - 1 + 1) * dof + d_idx];
          in_pos[4] = u_position[b_addrs + (h_idx - 1 + 2) * dof + d_idx];
        }
        out_pos  = in_pos[2];
        out_vel  = (0.5 * in_pos[3] - 0.5 * in_pos[1]) * dt;
        out_acc  = (in_pos[3] + in_pos[1] - 2 * in_pos[2]) * dt * dt;
        out_jerk = ((-0.5) * in_pos[0] + in_pos[1]  - in_pos[3] + (0.5) * in_pos[4]) *
                   (dt * dt * dt);
      }
      else if (h_idx == horizon - 2)
      {
        // use backward difference for jerk

        in_pos[0] = u_position[b_addrs + (h_idx - 1 - 3) * dof + d_idx];
        in_pos[1] = u_position[b_addrs + (h_idx - 1 - 2) * dof + d_idx];
        in_pos[2] = u_position[b_addrs + (h_idx - 1 - 1) * dof + d_idx];
        in_pos[3] = u_position[b_addrs + (h_idx - 1) * dof + d_idx];
        in_pos[4] = u_position[b_addrs + (h_idx - 1 + 1) * dof + d_idx];

        out_pos  = in_pos[3];
        out_vel  = (0.5 * in_pos[4] - 0.5 * in_pos[2]) * dt;
        out_acc  = (in_pos[4] + in_pos[2] - 2 * in_pos[3]) * dt * dt;
        out_jerk = (-1 * in_pos[0] + 3 * in_pos[1] - 3 * in_pos[2] + in_pos[3]) * dt * dt * dt;
      }
      else if (h_idx == horizon - 1)
      { // use backward difference for vel, acc, jerk
        for (int i = 0; i < 4; i++)
        {
          in_pos[i] = u_position[b_addrs + (h_idx - 1 - 3 + i) * dof + d_idx];
        }
        out_pos  = in_pos[3];
        out_vel  = (-in_pos[2] + in_pos[3]) * dt;
        out_acc  = (in_pos[1] - 2 * in_pos[2] + in_pos[3]) * dt * dt;
        out_jerk = (-in_pos[0] + 3 * in_pos[1] - 3 * in_pos[2] + in_pos[3]) * dt * dt * dt;
      }

      // write out:
      out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx]     = out_pos;
      out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx]     = out_vel;
      out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_acc;
      out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx]         = out_jerk;
    }

    template<typename scalar_t>
    __device__ __forceinline__ void compute_central_difference(scalar_t       *out_position_mem,
                                                               scalar_t       *out_velocity_mem,
                                                               scalar_t       *out_acceleration_mem,
                                                               scalar_t       *out_jerk_mem,
                                                               const scalar_t *u_position,
                                                               const scalar_t *start_position,
                                                               const scalar_t *start_velocity,
                                                               const scalar_t *start_acceleration,
                                                               const scalar_t *traj_dt,
                                                               const int       batch_size,
                                                               const int       horizon,
                                                               const int       dof,
                                                               const int       b_idx,
                                                               const int       h_idx,
                                                               const int       d_idx,
                                                               const int       b_offset)
    {
      const float dt = traj_dt[0]; // assume same dt across traj TODO: Implement variable dt
      // dt here is actually 1/dt;
      const float dt_inv  = 1.0 / dt;
      const float st_jerk = 0.0;   // Note: start jerk can also be passed from global memory
      // read start state:
      float out_pos = 0.0, out_vel = 0.0, out_acc = 0.0, out_jerk = 0.0;
      float st_pos = 0.0, st_vel = 0.0, st_acc = 0.0;

      const int   b_addrs_action = b_idx * (horizon - 4) * dof;
      float       in_pos[5]; // create a 5 value scalar

  #pragma unroll 5

      for (int i = 0; i < 5; i++)
      {
        in_pos[i] = 0.0;
      }

      if (h_idx < 5)
      {
        st_pos = start_position[b_offset * dof + d_idx];
        st_vel = start_velocity[b_offset * dof + d_idx];
        st_acc = start_acceleration[b_offset * dof + d_idx];
      }

      if ((h_idx > 3) && (h_idx < horizon - 4))
      {
        in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
        in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
        in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
        in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
        in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
      }


      else if (h_idx == 0)
      {
        in_pos[0] = (3.0f / 2) *
                    (-1 * st_acc * (dt_inv * dt_inv) - (dt_inv * dt_inv * dt_inv) * st_jerk) -
                    3.0f * dt_inv *
                    st_vel + st_pos;
        in_pos[1] = -2.0f * st_acc * dt_inv * dt_inv - (4.0f / 3) * dt_inv * dt_inv * dt_inv *
                    st_jerk - 2.0 * dt_inv * st_vel + st_pos;
        in_pos[2] = -(3.0f / 2) * st_acc * dt_inv * dt_inv - (7.0f / 6) * dt_inv * dt_inv * dt_inv *
                    st_jerk - dt_inv * st_vel + st_pos;
        in_pos[3] = st_pos;
        in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
      }

      else if (h_idx == 1)
      {
        in_pos[0] = -2.0f * st_acc * dt_inv * dt_inv - (4.0f / 3) * dt_inv * dt_inv * dt_inv *
                    st_jerk - 2.0 * dt_inv * st_vel + st_pos;
        in_pos[1] = -(3.0f / 2) * st_acc * dt_inv * dt_inv - (7.0f / 6) * dt_inv * dt_inv * dt_inv *
                    st_jerk - dt_inv * st_vel + st_pos;


        in_pos[2] = st_pos;
        in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
        in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
      }

      else if (h_idx == 2)
      {
        in_pos[0] = -(3.0f / 2) * st_acc * dt_inv * dt_inv - (7.0f / 6) * dt_inv * dt_inv * dt_inv *
                    st_jerk - dt_inv * st_vel + st_pos;
        in_pos[1] = st_pos;
        in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
        in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
        in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
      }
      else if (h_idx == 3)
      {
        in_pos[0] = st_pos;
        in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
        in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
        in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
        in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
      }

      else if (h_idx == horizon - 4)
      {
        in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
        in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
        in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
        in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
        in_pos[4] = in_pos[3]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +
                               // d_idx];
      }

      else if (h_idx == horizon - 3)
      {
        in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
        in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
        in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
        in_pos[3] = in_pos[2]; // u_position[b_addrs_action + (h_idx - 1 + 1) * dof + d_idx];
        in_pos[4] = in_pos[2]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +
                               // d_idx];
      }
      else if (h_idx == horizon - 2)
      {
        in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
        in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
        in_pos[2] = in_pos[1];
        in_pos[3] = in_pos[1]; // u_position[b_addrs_action + (h_idx - 1 + 1) * dof + d_idx];
        in_pos[4] = in_pos[1]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +
                               // d_idx];
      }

      else if (h_idx == horizon - 1)
      {
        in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
        in_pos[1] = in_pos[0];
        in_pos[2] = in_pos[0]; // u_position[b_addrs_action + (h_idx - 1 ) * dof + d_idx];
        in_pos[3] = in_pos[0]; // u_position[b_addrs_action + (h_idx - 1 + 1) * dof + d_idx];
        in_pos[4] = in_pos[0]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +
                               // d_idx];
      }
      out_pos = in_pos[2];

      // out_vel = (0.5 * in_pos[3] - 0.5 * in_pos[1]) * dt;
      out_vel =
        ((0.083333333f) * in_pos[0] - (0.666666667f) * in_pos[1] + (0.666666667f) * in_pos[3] +
         (-0.083333333f) * in_pos[4]) * dt;

      // out_acc = (in_pos[3] + in_pos[1] - 2.0 * in_pos[2]) * dt * dt;
      out_acc =
        ((-0.083333333f) * in_pos[0] + (1.333333333f) * in_pos[1] + (-2.5f) * in_pos[2] +
         (1.333333333f) * in_pos[3] + (-0.083333333f) * in_pos[4]) * dt * dt;
      out_jerk = ((-0.5f) * in_pos[0] + in_pos[1]  - in_pos[3] + (0.5f) * in_pos[4]) *
                 (dt * dt * dt);

      // write out:
      out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx]     = out_pos;
      out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx]     = out_vel;
      out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_acc;
      out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx]         = out_jerk;
    }

    template<typename scalar_t, int8_t mode>
    __global__ void position_clique_loop_kernel2(
      scalar_t *out_position_mem, scalar_t *out_velocity_mem,
      scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
      const scalar_t *u_position, const scalar_t *start_position,
      const scalar_t *start_velocity, const scalar_t *start_acceleration,
      const scalar_t *traj_dt, const int batch_size, const int horizon,
      const int dof)
    {
      const int tid = blockDim.x * blockIdx.x + threadIdx.x;

      // number of threads = batch_size * dof * horizon;
      const int h_idx = tid % horizon;
      const int d_idx = (tid / horizon) % dof;
      const int b_idx = tid / (dof * horizon);

      if (tid >= batch_size * dof * horizon)
      {
        return;
      }


      const int b_offset = b_idx;

      if (mode == BWD_DIFF)
      {
        compute_backward_difference(out_position_mem,
                                    out_velocity_mem,
                                    out_acceleration_mem,
                                    out_jerk_mem,
                                    u_position,
                                    start_position,
                                    start_velocity,
                                    start_acceleration,
                                    traj_dt,
                                    batch_size,
                                    horizon,
                                    dof,
                                    b_idx,
                                    h_idx,
                                    d_idx,
                                    b_offset);
      }
      else if (mode == CENTRAL_DIFF)
      {
        compute_central_difference(out_position_mem,
                                   out_velocity_mem,
                                   out_acceleration_mem,
                                   out_jerk_mem,
                                   u_position,
                                   start_position,
                                   start_velocity,
                                   start_acceleration,
                                   traj_dt,
                                   batch_size,
                                   horizon,
                                   dof,
                                   b_idx,
                                   h_idx,
                                   d_idx,
                                   b_offset);
      }
      else
      {
        assert(false);
      }
    }

    template<typename scalar_t, int8_t mode>
    __global__ void position_clique_loop_idx_kernel2(
      scalar_t *out_position_mem, scalar_t *out_velocity_mem,
      scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
      const scalar_t *u_position, const scalar_t *start_position,
      const scalar_t *start_velocity, const scalar_t *start_acceleration,
      const int32_t *start_idx, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      const int tid = blockDim.x * blockIdx.x + threadIdx.x;

      // number of threads = batch_size * dof * horizon;
      const int h_idx = tid % horizon;
      const int d_idx = (tid / horizon) % dof;
      const int b_idx = tid / (dof * horizon);

      if (tid >= batch_size * dof * horizon)
      {
        return;
      }


      const int b_offset = start_idx[b_idx];

      if (mode == BWD_DIFF)
      {
        compute_backward_difference(out_position_mem,
                                    out_velocity_mem,
                                    out_acceleration_mem,
                                    out_jerk_mem,
                                    u_position,
                                    start_position,
                                    start_velocity,
                                    start_acceleration,
                                    traj_dt,
                                    batch_size,
                                    horizon,
                                    dof,
                                    b_idx,
                                    h_idx,
                                    d_idx,
                                    b_offset);
      }
      else if (mode == CENTRAL_DIFF)
      {
        compute_central_difference(out_position_mem,
                                   out_velocity_mem,
                                   out_acceleration_mem,
                                   out_jerk_mem,
                                   u_position,
                                   start_position,
                                   start_velocity,
                                   start_acceleration,
                                   traj_dt,
                                   batch_size,
                                   horizon,
                                   dof,
                                   b_idx,
                                   h_idx,
                                   d_idx,
                                   b_offset);
      }
      else
      {
        assert(false);
      }
    }

    template<typename scalar_t>
    __global__ void backward_position_clique_loop_kernel(
      scalar_t *out_grad_position, const scalar_t *grad_position,
      const scalar_t *grad_velocity, const scalar_t *grad_acceleration,
      const scalar_t *grad_jerk, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }
      const int b_addrs = b_idx * horizon * dof;

      // read gradients:
      float g_pos[MAX_H];
      float g_vel[MAX_H];
      float g_acc[MAX_H];
      float g_jerk[MAX_H];
      const float dt   = traj_dt[0];
      const float dt_2 = dt * dt;      // dt * dt;
      const float dt_3 = dt * dt * dt; // dt * dt * dt;

      // not used index == 0
      g_pos[0]  = 0.0;
      g_vel[0]  = 0.0;
      g_acc[0]  = 0.0;
      g_jerk[0] = 0.0;
#pragma unroll

      for (int h_idx = 1; h_idx < horizon; h_idx++)
      {
        g_pos[h_idx]  = grad_position[b_addrs + (h_idx) * dof + d_idx];
        g_vel[h_idx]  = grad_velocity[b_addrs + (h_idx) * dof + d_idx];
        g_acc[h_idx]  = grad_acceleration[b_addrs + (h_idx) * dof + d_idx];
        g_jerk[h_idx] = grad_jerk[b_addrs + (h_idx) * dof + d_idx];
      }
#pragma unroll

      for (int i = 0; i < 4; i++)
      {
        g_vel[horizon + i]  = 0.0;
        g_acc[horizon + i]  = 0.0;
        g_jerk[horizon + i] = 0.0;
      }

      // compute gradient and sum
      for (int h_idx = 0; h_idx < horizon - 1; h_idx++)
      {
        g_pos[h_idx + 1] +=
          ((g_vel[h_idx + 1] - g_vel[h_idx + 2]) * dt +
           (g_acc[h_idx + 1] - 2 * g_acc[h_idx + 2] + g_acc[h_idx + 3]) * dt_2 +
           (g_jerk[h_idx + 1] - 3 * g_jerk[h_idx + 2] + 3 * g_jerk[h_idx + 3] -
            g_jerk[h_idx + 4]) *
           dt_3);
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon - 1; h_idx++)
      {
        out_grad_position[b_addrs + (h_idx) * dof + d_idx] = g_pos[h_idx + 1];
      }
      out_grad_position[b_addrs + (horizon - 1) * dof + d_idx] = 0.0;
    }

    template<typename scalar_t>
    __global__ void backward_position_clique_loop_backward_difference_kernel2(
      scalar_t *out_grad_position, const scalar_t *grad_position,
      const scalar_t *grad_velocity, const scalar_t *grad_acceleration,
      const scalar_t *grad_jerk, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      const int tid = blockDim.x * blockIdx.x + threadIdx.x;

      // number of threads = batch_size * dof * horizon;
      const int h_idx = tid % horizon;
      const int d_idx = (tid / horizon) % dof;
      const int b_idx = tid / (dof * horizon);

      if (tid >= batch_size * dof * horizon)
      {
        return;
      }
      const int b_addrs = b_idx * horizon * dof;

      if (h_idx == 0)
      {
        return;
      }


      const float dt = traj_dt[0];

      // read gradients:
      float g_pos    = 0.0;
      float out_grad = 0.0;
      float g_vel[4];
      float g_acc[4];
      float g_jerk[4];

  #pragma unroll 4

      for (int i = 0; i < 4; i++)
      {
        g_vel[i]  = 0.0;
        g_acc[i]  = 0.0;
        g_jerk[i] = 0.0;
      }

      int hid = h_idx; // + 1;

      g_pos = grad_position[b_addrs + (hid) * dof + d_idx];


      if (hid < horizon - 3)
      {
        for (int i = 0; i < 4; i++)
        {
          g_vel[i]  = grad_velocity[b_addrs + (hid + i) * dof + d_idx];
          g_acc[i]  = grad_acceleration[b_addrs + (hid + i) * dof + d_idx];
          g_jerk[i] = grad_jerk[b_addrs + (hid + i) * dof + d_idx];
        }
      }
      else if (hid == horizon - 3)
      {
        for (int i = 0; i < 3; i++)
        {
          g_vel[i]  = grad_velocity[b_addrs + (hid + i) * dof + d_idx];
          g_acc[i]  = grad_acceleration[b_addrs + (hid + i) * dof + d_idx];
          g_jerk[i] = grad_jerk[b_addrs + (hid + i) * dof + d_idx];
        }
      }


      else if (hid == horizon - 2)
      {
        for (int i = 0; i < 2; i++)
        {
          g_vel[i]  = grad_velocity[b_addrs + (hid + i) * dof + d_idx];
          g_acc[i]  = grad_acceleration[b_addrs + (hid + i) * dof + d_idx];
          g_jerk[i] = grad_jerk[b_addrs + (hid + i) * dof + d_idx];
        }
      }
      else if (hid == horizon - 1)
      {
        for (int i = 0; i < 1; i++)
        {
          g_vel[i]  = grad_velocity[b_addrs + (hid + i) * dof + d_idx];
          g_acc[i]  = grad_acceleration[b_addrs + (hid + i) * dof + d_idx];
          g_jerk[i] = grad_jerk[b_addrs + (hid + i) * dof + d_idx];
        }
      }


      out_grad = (g_pos +
                  (g_vel[0] -  g_vel[1]) * dt +
                  (g_acc[0] - 2 * g_acc[1] + g_acc[2]) * dt * dt +
                  (g_jerk[0] - 3 * g_jerk[1] + 3 * g_jerk[2] - g_jerk[3]) * dt * dt * dt);


      // write out:
      out_grad_position[b_addrs + (h_idx - 1) * dof + d_idx] = out_grad;
    }

    template<typename scalar_t>
    __global__ void backward_position_clique_loop_central_difference_kernel2(
      scalar_t *out_grad_position, const scalar_t *grad_position,
      const scalar_t *grad_velocity, const scalar_t *grad_acceleration,
      const scalar_t *grad_jerk, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      const int tid = blockDim.x * blockIdx.x + threadIdx.x;

      // number of threads = batch_size * dof * horizon;
      const int h_idx = tid % horizon;
      const int d_idx = (tid / horizon) % dof;
      const int b_idx = tid / (dof * horizon);

      if (tid >= batch_size * dof * horizon)
      {
        return;
      }
      const int b_addrs        = b_idx * horizon * dof;
      const int b_addrs_action = b_idx * (horizon - 4) * dof;

      if ((h_idx < 2) || (h_idx >= horizon - 2))
      {
        return;
      }

      const float dt = traj_dt[0];

      // read gradients:
      // float g_pos= 0.0;
      float out_grad = 0.0;
      float g_pos[3];
      float g_vel[5];
      float g_acc[5];
      float g_jerk[5];

  #pragma unroll 5

      for (int i = 0; i < 5; i++)
      {
        g_vel[i]  = 0.0;
        g_acc[i]  = 0.0;
        g_jerk[i] = 0.0;
      }

      const int hid = h_idx;

      g_pos[0] = grad_position[b_addrs + (hid) * dof + d_idx];
      g_pos[1] = 0.0;
      g_pos[2] = 0.0;


      if ((hid > 1) && (h_idx < horizon - 3))
      {
    #pragma unroll

        for (int i = 0; i < 5; i++)
        {
          g_vel[i]  = grad_velocity[b_addrs + ((hid - 2) + i) * dof + d_idx];
          g_acc[i]  = grad_acceleration[b_addrs + ((hid - 2) + i) * dof + d_idx];
          g_jerk[i] = grad_jerk[b_addrs + ((hid - 2) + i) * dof + d_idx];
        }
        out_grad = (g_pos[0] +

// ((-0.5) * g_vel[3] + (0.5) * g_vel[1]) * dt +

                    ((-0.083333333f) * g_vel[0] + (0.666666667f) * g_vel[1] + (-0.666666667f) *
                     g_vel[3] + (0.083333333f) * g_vel[4]) * dt +


                    ((-0.083333333f) * g_acc[0] + (1.333333333f) * g_acc[1] + (-2.5f) * g_acc[2] +
                     (1.333333333f) * g_acc[3] + (-0.083333333f) * g_acc[4]) * dt * dt +

// ( g_acc[3] + g_acc[1] - (2.0) * g_acc[2]) * dt * dt +
                    //(-0.5f * g_jerk[0] + g_jerk[1] - g_jerk[3] + 0.5f * g_jerk[4]) * dt * dt * dt);
                    (0.5f * g_jerk[0]  - g_jerk[1] + g_jerk[3] - 0.5f * g_jerk[4]) * dt * dt * dt);

                    //(0.500000000000000 * g_jerk[0] - 1 * g_jerk[1] + 0 * g_jerk[2] + 1 * g_jerk[3] - 0.500000000000000 * g_jerk[4]) * dt_inv_3;


      }
      else if (hid == horizon - 3)
      {
        // The below can cause oscilatory gradient steps.

        /*
         #pragma unroll
           for (int i=0; i< 5; i++)
           {
           g_vel[i] = grad_velocity[b_addrs + ((hid - 2) + i)*dof + d_idx];
           g_acc[i] = grad_acceleration[b_addrs + ((hid -2) + i)*dof + d_idx];
           g_jerk[i] = grad_jerk[b_addrs + ((hid -2) + i)*dof + d_idx];
           }
         */
        g_pos[1] = grad_position[b_addrs + (hid + 1) * dof + d_idx];
        g_pos[2] = grad_position[b_addrs + (hid + 2) * dof + d_idx];

        out_grad = (g_pos[0] + g_pos[1] + g_pos[2]);

        /* +
           //((0.5) * g_vel[1] + (0.5) * g_vel[2]) * dt +
           ((-0.083333333f) * g_vel[0] + (0.583333333f) * g_vel[1] + (0.583333333f) * g_vel[2] +
              (-0.083333333f) * g_vel[3]) * dt  +
           ((-0.083333333f) * g_acc[0] + (1.25f) * g_acc[1] + (-1.25f) * g_acc[2] + (0.083333333f) *
              g_acc[3]) * dt * dt +
           //( g_acc[1] - g_acc[2]) * dt * dt +
           (0.5f * g_jerk[0] - 0.5f * g_jerk[1] -0.5f * g_jerk[2] + 0.5f * g_jerk[3]) * dt * dt *
              dt);
         */
      }


      // write out:
      out_grad_position[b_addrs_action + (h_idx - 2) * dof + d_idx] = out_grad;
    }

    // for MPPI:

    template<typename scalar_t>
    __global__ void
    acceleration_loop_kernel(scalar_t *out_position_mem, scalar_t *out_velocity_mem,
                             scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
                             const scalar_t *u_acc, const scalar_t *start_position,
                             const scalar_t *start_velocity,
                             const scalar_t *start_acceleration,
                             const scalar_t *traj_dt, const int batch_size,
                             const int horizon, const int dof)
    {
      // there are batch * horizon * dof threads:
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }

      // read start state:
      float u_arr[MAX_H], dt[MAX_H];
      float out_pos[MAX_H], out_vel[MAX_H], out_acc[MAX_H], out_jerk[MAX_H];
      const int b_addrs = b_idx * horizon * dof;

#pragma unroll

      for (int i = 0; i < horizon; i++)
      {
        u_arr[i]    = u_acc[b_addrs + i * dof + d_idx];
        dt[i]       = traj_dt[i];
        out_pos[i]  = 0;
        out_vel[i]  = 0;
        out_acc[i]  = 0;
        out_jerk[i] = 0;
      }

      out_pos[0]  = start_position[b_idx * dof + d_idx];
      out_vel[0]  = start_velocity[b_idx * dof + d_idx];
      out_acc[0]  = start_acceleration[b_idx * dof + d_idx];
      out_jerk[0] = 0.0;

      for (int h_idx = 1; h_idx < horizon; h_idx++)
      {
        // do semi implicit euler integration:
        out_acc[h_idx]  = u_arr[h_idx - 1];
        out_vel[h_idx]  = out_vel[h_idx - 1] + out_acc[h_idx] * dt[h_idx];
        out_pos[h_idx]  = out_pos[h_idx - 1] + out_vel[h_idx] * dt[h_idx];
        out_jerk[h_idx] = (out_acc[h_idx] - out_acc[h_idx - 1]) / dt[h_idx];
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon; h_idx++)
      {
        out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_pos[h_idx]; // new_position;
        out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_vel[h_idx];
        out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_acc[h_idx];
        out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_jerk[h_idx];
      }
    }

    template<typename scalar_t>
    __global__ void acceleration_loop_rk2_kernel(
      scalar_t *out_position_mem, scalar_t *out_velocity_mem,
      scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
      const scalar_t *u_acc, const scalar_t *start_position,
      const scalar_t *start_velocity, const scalar_t *start_acceleration,
      const scalar_t *traj_dt, const int batch_size, const int horizon,
      const int dof)
    {
      // there are batch * horizon * dof threads:
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }

      // read start state:
      float u_arr[MAX_H], dt[MAX_H];
      float out_pos[MAX_H], out_vel[MAX_H], out_acc[MAX_H], out_jerk[MAX_H];
      const int b_addrs = b_idx * horizon * dof;

#pragma unroll

      for (int i = 0; i < horizon; i++)
      {
        u_arr[i]    = u_acc[b_addrs + i * dof + d_idx];
        dt[i]       = traj_dt[i];
        out_pos[i]  = 0;
        out_vel[i]  = 0;
        out_acc[i]  = 0;
        out_jerk[i] = 0;
      }

      out_pos[0]  = start_position[b_idx * dof + d_idx];
      out_vel[0]  = start_velocity[b_idx * dof + d_idx];
      out_acc[0]  = start_acceleration[b_idx * dof + d_idx];
      out_jerk[0] = 0.0;

      for (int h_idx = 1; h_idx < horizon; h_idx++)
      {
        // do rk2 integration:

        out_acc[h_idx]  = u_arr[h_idx - 1];
        out_jerk[h_idx] = (out_acc[h_idx] - out_acc[h_idx - 1]) / dt[h_idx];
        out_pos[h_idx]  = out_pos[h_idx - 1] + out_vel[h_idx - 1] * dt[h_idx] +
                          0.5 * dt[h_idx] * dt[h_idx] * out_acc[h_idx];
        out_vel[h_idx] = out_vel[h_idx - 1] + 0.5 * dt[h_idx] * out_acc[h_idx];
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon; h_idx++)
      {
        out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_pos[h_idx]; // new_position;
        out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_vel[h_idx];
        out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_acc[h_idx];
        out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_jerk[h_idx];
      }
    }

    template<typename scalar_t>
    __global__ void acceleration_loop_idx_kernel(
      scalar_t *out_position_mem, scalar_t *out_velocity_mem,
      scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
      const scalar_t *u_acc, const scalar_t *start_position,
      const scalar_t *start_velocity, const scalar_t *start_acceleration,
      const int32_t *start_idx, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      // there are batch * horizon * dof threads:
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }

      // read start state:
      float u_arr[MAX_H], dt[MAX_H];
      float out_pos[MAX_H], out_vel[MAX_H], out_acc[MAX_H], out_jerk[MAX_H];
      const int b_addrs  = b_idx * horizon * dof;
      const int b_offset = start_idx[b_idx];

#pragma unroll

      for (int i = 0; i < horizon; i++)
      {
        u_arr[i]    = u_acc[b_addrs + i * dof + d_idx];
        dt[i]       = traj_dt[i];
        out_pos[i]  = 0;
        out_vel[i]  = 0;
        out_acc[i]  = 0;
        out_jerk[i] = 0;
      }

      out_pos[0]  = start_position[b_offset * dof + d_idx];
      out_vel[0]  = start_velocity[b_offset * dof + d_idx];
      out_acc[0]  = start_acceleration[b_offset * dof + d_idx];
      out_jerk[0] = 0.0;

      for (int h_idx = 1; h_idx < horizon; h_idx++)
      {
        // do semi implicit euler integration:
        out_acc[h_idx]  = u_arr[h_idx - 1];
        out_vel[h_idx]  = out_vel[h_idx - 1] + out_acc[h_idx] * dt[h_idx];
        out_pos[h_idx]  = out_pos[h_idx - 1] + out_vel[h_idx] * dt[h_idx];
        out_jerk[h_idx] = (out_acc[h_idx] - out_acc[h_idx - 1]) / dt[h_idx];
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon; h_idx++)
      {
        out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_pos[h_idx]; // new_position;
        out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_vel[h_idx];
        out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_acc[h_idx];
        out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_jerk[h_idx];
      }
    }

    template<typename scalar_t>
    __global__ void acceleration_loop_idx_rk2_kernel(
      scalar_t *out_position_mem, scalar_t *out_velocity_mem,
      scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
      const scalar_t *u_acc, const scalar_t *start_position,
      const scalar_t *start_velocity, const scalar_t *start_acceleration,
      const int32_t *start_idx, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      // there are batch * horizon * dof threads:
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }

      // read start state:
      float u_arr[MAX_H], dt[MAX_H];
      float out_pos[MAX_H], out_vel[MAX_H], out_acc[MAX_H], out_jerk[MAX_H];
      const int b_addrs  = b_idx * horizon * dof;
      const int b_offset = start_idx[b_idx];

#pragma unroll

      for (int i = 0; i < horizon; i++)
      {
        u_arr[i]    = u_acc[b_addrs + i * dof + d_idx];
        dt[i]       = traj_dt[i];
        out_pos[i]  = 0;
        out_vel[i]  = 0;
        out_acc[i]  = 0;
        out_jerk[i] = 0;
      }

      out_pos[0]  = start_position[b_offset * dof + d_idx];
      out_vel[0]  = start_velocity[b_offset * dof + d_idx];
      out_acc[0]  = start_acceleration[b_offset * dof + d_idx];
      out_jerk[0] = 0.0;

      for (int h_idx = 1; h_idx < horizon; h_idx++)
      {
        // do semi implicit euler integration:

        out_acc[h_idx]  = u_arr[h_idx - 1];
        out_vel[h_idx]  = out_vel[h_idx - 1] + out_acc[h_idx] * dt[h_idx];
        out_pos[h_idx]  = out_pos[h_idx - 1] + out_vel[h_idx] * dt[h_idx];
        out_jerk[h_idx] = (out_acc[h_idx] - out_acc[h_idx - 1]) / dt[h_idx];
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon; h_idx++)
      {
        out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_pos[h_idx]; // new_position;
        out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_vel[h_idx];
        out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] =
          out_acc[h_idx];
        out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_jerk[h_idx];
      }
    }

    // Not used

    template<typename scalar_t>
    __global__ void position_clique_kernel(
      scalar_t *out_position, scalar_t *out_velocity, scalar_t *out_acceleration,
      scalar_t *out_jerk, const scalar_t *u_position,
      const scalar_t *start_position, const scalar_t *start_velocity,
      const scalar_t *start_acceleration, const scalar_t *traj_dt,
      const int batch_size, const int horizon, const int dof)
    {
      // there are batch * horizon * dof threads:
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (horizon * dof);
      const int h_idx = (tid - b_idx * (horizon * dof)) / dof;
      const int d_idx = (tid - b_idx * horizon * dof - h_idx * dof);

      if ((b_idx >= batch_size) || (h_idx >= horizon) || (d_idx >= dof))
      {
        return;
      }

      float new_position, new_velocity, new_acceleration, new_jerk;
      const float dt = traj_dt[h_idx];

      // read actions: batch, horizon
      if (h_idx == 0)
      {
        new_position     = start_position[b_idx * dof + d_idx];
        new_velocity     = start_velocity[b_idx * dof + d_idx];
        new_acceleration = start_acceleration[b_idx * dof + d_idx];
        new_jerk         = 0.0;
      }
      else if (h_idx == 1)
      {
        float2 u_clique = make_float2(
          start_position[b_idx * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 1) * dof + d_idx]);
        new_position     = u_clique.y;
        new_velocity     = (u_clique.y - u_clique.x) * dt;    // 1 - 0
        new_acceleration =
          (u_clique.y - u_clique.x) * dt * dt;                // 2 - 2.0 * 1 + 1
        new_jerk = (u_clique.y - u_clique.x) *  dt * dt * dt; // -1 3 -3 1
      }
      else if (h_idx == 2)
      {
        float3 u_clique = make_float3(
          start_position[b_idx * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 2) * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 1) * dof + d_idx]);

        new_position     = u_clique.z;
        new_velocity     = (u_clique.z - u_clique.y) * dt; // 1 - 0
        new_acceleration = (u_clique.x - 2 * u_clique.y + u_clique.z) *
                           dt * dt;                        // 2 - 2.0 * 1 + 1
        new_jerk = (2 * u_clique.x - 3 * u_clique.y + u_clique.z) *
                   dt * dt * dt;                           // -1 3 -3 1
      }
      else if (h_idx == 3)
      {
        float4 u_clique = make_float4(
          start_position[b_idx * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 3) * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 2) * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 1) * dof + d_idx]);
        new_position     = u_clique.w;
        new_velocity     = (u_clique.w - u_clique.z) * dt; // 1 - 0
        new_acceleration = (u_clique.y - 2 * u_clique.z + u_clique.w) *
                           dt * dt;                        // 2 - 2.0 * 1 + 1
        new_jerk =
          (-1.0 * u_clique.x + 3 * u_clique.y - 3 * u_clique.z + u_clique.w) *
          dt * dt * dt;
      }
      else
      {
        float4 u_clique = make_float4(
          u_position[b_idx * horizon * dof + (h_idx - 4) * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 3) * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 2) * dof + d_idx],
          u_position[b_idx * horizon * dof + (h_idx - 1) * dof + d_idx]);
        new_position     = u_clique.w;
        new_velocity     = (u_clique.w - u_clique.z) * dt; // 1 - 0
        new_acceleration = (u_clique.y - 2 * u_clique.z + u_clique.w) *
                           dt * dt;                        // 2 - 2.0 * 1 + 1
        new_jerk =
          (-1.0 * u_clique.x + 3 * u_clique.y - 3 * u_clique.z + u_clique.w) *
          dt * dt * dt;
      }

      // h_idx = h_idx + 1;
      out_position[b_idx * horizon * dof + h_idx * dof + d_idx] =
        new_position; // new_position;
      out_velocity[b_idx * horizon * dof + h_idx * dof + d_idx]     = new_velocity;
      out_acceleration[b_idx * horizon * dof + h_idx * dof + d_idx] =
        new_acceleration;
      out_jerk[b_idx * horizon * dof + h_idx * dof + d_idx] = new_jerk;
    }

    // Not used
    template<typename scalar_t>
    __global__ void position_clique_loop_coalesce_kernel(
      scalar_t *out_position_mem, scalar_t *out_velocity_mem,
      scalar_t *out_acceleration_mem, scalar_t *out_jerk_mem,
      const scalar_t *u_position, const scalar_t *start_position,
      const scalar_t *start_velocity, const scalar_t *start_acceleration,
      const scalar_t *traj_dt, const int batch_size, const int horizon,
      const int dof)
    {
      // data is stored as batch, dof, horizon
      // there are batch * horizon * dof threads:
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }

      const float dt =
        traj_dt[0]; // assume same dt across traj TODO: Implement variable dt

      // read start state:
      float u_arr[MAX_H];
      float out_pos[MAX_H], out_vel[MAX_H], out_acc[MAX_H], out_jerk[MAX_H];
      const int b_addrs = b_idx * horizon * dof;

#pragma unroll

      for (int i = 0; i < horizon; i++)
      {
        u_arr[i]    = u_position[b_addrs + d_idx * horizon + i];
        out_pos[i]  = 0;
        out_vel[i]  = 0;
        out_acc[i]  = 0;
        out_jerk[i] = 0;
      }

      out_pos[0]  = start_position[b_idx * dof + d_idx];
      out_vel[0]  = start_velocity[b_idx * dof + d_idx];
      out_acc[0]  = start_acceleration[b_idx * dof + d_idx];
      out_jerk[0] = 0.0;

      for (int h_idx = 1; h_idx < horizon; h_idx++)
      {
        // read actions: batch, horizon

        out_pos[h_idx] = u_arr[h_idx - 1];

        out_vel[h_idx] = (out_pos[h_idx] - out_pos[h_idx - 1]) * dt;  // 1 - 0
        out_acc[h_idx] =
          (out_vel[h_idx] - out_vel[h_idx - 1]) * dt;                 // 2 - 2.0 * 1 + 1
        out_jerk[h_idx] = (out_acc[h_idx] - out_acc[h_idx - 1]) * dt; // -1 3 -3 1
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon; h_idx++)
      {
        out_position_mem[b_idx * horizon * dof + d_idx * horizon + h_idx] =
          out_pos[h_idx]; // new_position;
        out_velocity_mem[b_idx * horizon * dof + d_idx * horizon + h_idx] =
          out_vel[h_idx];
        out_acceleration_mem[b_idx * horizon * dof + d_idx * horizon + h_idx] =
          out_acc[h_idx];
        out_jerk_mem[b_idx * horizon * dof + d_idx * horizon + h_idx] =
          out_jerk[h_idx];
      }
    }

    // Not used
    template<typename scalar_t>
    __global__ void backward_position_clique_loop_coalesce_kernel(
      scalar_t *out_grad_position, const scalar_t *grad_position,
      const scalar_t *grad_velocity, const scalar_t *grad_acceleration,
      const scalar_t *grad_jerk, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (dof);
      const int d_idx = (tid - b_idx * dof);

      if ((b_idx >= batch_size) || (d_idx >= dof))
      {
        return;
      }
      const int b_addrs = b_idx * horizon * dof;

      // read gradients:
      float g_pos[MAX_H];
      float g_vel[MAX_H];
      float g_acc[MAX_H];
      float g_jerk[MAX_H];
      const float dt   = traj_dt[0];
      const float dt_2 = dt * dt;
      const float dt_3 = dt * dt * dt;
#pragma unroll

      for (int h_idx = 0; h_idx < horizon; h_idx++)
      {
        g_pos[h_idx]  = grad_position[b_addrs + d_idx * horizon + h_idx];
        g_vel[h_idx]  = grad_velocity[b_addrs + d_idx * horizon + h_idx];
        g_acc[h_idx]  = grad_acceleration[b_addrs + d_idx * horizon + h_idx];
        g_jerk[h_idx] = grad_jerk[b_addrs + d_idx * horizon + h_idx];
      }
#pragma unroll

      for (int i = 0; i < 4; i++)
      {
        g_vel[horizon + i]  = 0.0;
        g_acc[horizon + i]  = 0.0;
        g_jerk[horizon + i] = 0.0;
      }

      // compute gradient and sum
      for (int h_idx = 0; h_idx < horizon - 1; h_idx++)
      {
        g_pos[h_idx + 1] +=
          ((g_vel[h_idx + 1] - g_vel[h_idx + 2]) * dt +
           (g_acc[h_idx + 1] - 2 * g_acc[h_idx + 2] + g_acc[h_idx + 3]) * dt_2 +
           (1 * g_jerk[h_idx + 1] - 3 * g_jerk[h_idx + 2] +
            3 * g_jerk[h_idx + 3] - g_jerk[h_idx + 4]) *
           dt_3);
      }

      // write out:
      for (int h_idx = 0; h_idx < horizon - 1; h_idx++)
      {
        out_grad_position[b_addrs + d_idx * horizon + h_idx] = g_pos[h_idx + 1];
      }
      out_grad_position[b_addrs + d_idx * horizon + horizon - 1] = 0.0;
    }

    // Not used
    template<typename scalar_t>
    __global__ void backward_position_clique_kernel(
      scalar_t *out_grad_position, const scalar_t *grad_position,
      const scalar_t *grad_velocity, const scalar_t *grad_acceleration,
      const scalar_t *grad_jerk, const scalar_t *traj_dt, const int batch_size,
      const int horizon, const int dof)
    {
      // TODO: transpose h and dof to be able to directly read float2, float3, etc..
      const int tid   = blockDim.x * blockIdx.x + threadIdx.x;
      const int b_idx = tid / (horizon * dof);
      const int h_idx = (tid - b_idx * (horizon * dof)) / dof;
      const int d_idx = (tid - b_idx * horizon * dof - h_idx * dof);

      if ((b_idx >= batch_size) || (h_idx >= horizon) || (d_idx >= dof))
      {
        return;
      }
      const int b_addrs = b_idx * horizon * dof;

      if (h_idx == horizon - 1)
      {
        out_grad_position[b_addrs + (h_idx) * dof + d_idx] = 0.0;
        return;
      }

      // read gradients:
      const float dt = traj_dt[0];
      float g_u      = grad_position[b_addrs + (h_idx + 1) * dof + d_idx];

      float2 g_vel;
      float3 g_acc;
      float4 g_jerk;

      if (h_idx < horizon - 4) // && h_idx > 0)
      {
        g_vel = make_float2(grad_velocity[b_addrs + (h_idx + 1) * dof + d_idx],
                            grad_velocity[b_addrs + (h_idx + 2) * dof + d_idx]);
        g_acc = make_float3(grad_acceleration[b_addrs + (h_idx + 1) * dof + d_idx],
                            grad_acceleration[b_addrs + (h_idx + 2) * dof + d_idx],
                            grad_acceleration[b_addrs + (h_idx + 3) * dof + d_idx]);

        g_jerk = make_float4(grad_jerk[b_addrs + (h_idx + 1) * dof + d_idx],
                             grad_jerk[b_addrs + (h_idx + 2) * dof + d_idx],
                             grad_jerk[b_addrs + (h_idx + 3) * dof + d_idx],
                             grad_jerk[b_addrs + (h_idx + 4) * dof + d_idx]);
        g_u +=
          ((g_vel.x - g_vel.y) * dt +
           (g_acc.x - 2 * g_acc.y + g_acc.z) * dt * dt +
           (1 * g_jerk.x - 3 * g_jerk.y + 3 * g_jerk.z - g_jerk.w) * dt * dt * dt);
      }
      else if (h_idx == 0)
      {
        g_vel = make_float2(grad_velocity[b_addrs + (h_idx + 1) * dof + d_idx],
                            grad_velocity[b_addrs + (h_idx + 2) * dof + d_idx]);

        g_acc = make_float3(grad_acceleration[b_addrs + (h_idx + 1) * dof + d_idx],
                            grad_acceleration[b_addrs + (h_idx + 2) * dof + d_idx],
                            0.0);

        g_jerk = make_float4(grad_jerk[b_addrs + (h_idx + 1) * dof + d_idx],
                             grad_jerk[b_addrs + (h_idx + 2) * dof + d_idx],
                             grad_jerk[b_addrs + (h_idx + 3) * dof + d_idx], 0.0);
        g_u += ((g_vel.x - g_vel.y) * dt +
                (-1.0 * g_acc.x + 1 * g_acc.y) * dt * dt +
                (-1 * g_jerk.x + 2 * g_jerk.y - 1 * g_jerk.z) * dt * dt * dt);
      }
      else if (h_idx == horizon - 4)
      {
        g_vel = make_float2(grad_velocity[b_addrs + (h_idx + 1) * dof + d_idx],
                            grad_velocity[b_addrs + (h_idx + 2) * dof + d_idx]);
        g_acc = make_float3(grad_acceleration[b_addrs + (h_idx + 1) * dof + d_idx],
                            grad_acceleration[b_addrs + (h_idx + 2) * dof + d_idx],
                            grad_acceleration[b_addrs + (h_idx + 3) * dof + d_idx]);
        g_jerk = make_float4(grad_jerk[b_addrs + (h_idx + 1) * dof + d_idx],
                             grad_jerk[b_addrs + (h_idx + 2) * dof + d_idx],
                             grad_jerk[b_addrs + (h_idx + 3) * dof + d_idx], 0.0);
        g_u +=
          ((g_vel.x - g_vel.y) * dt +
           (g_acc.x - 2 * g_acc.y + g_acc.z) * dt * dt +
           (1 * g_jerk.x - 3 * g_jerk.y + 3 * g_jerk.z - g_jerk.w) * dt * dt * dt);
      }
      else if (h_idx == horizon - 3)
      {
        g_vel = make_float2(grad_velocity[b_addrs + (h_idx + 1) * dof + d_idx],
                            grad_velocity[b_addrs + (h_idx + 2) * dof + d_idx]);
        g_acc =
          make_float3(grad_acceleration[b_addrs + (h_idx + 1) * dof + d_idx],
                      grad_acceleration[b_addrs + (h_idx + 2) * dof + d_idx], 0);
        g_jerk =
          make_float4(grad_jerk[b_addrs + (h_idx + 1) * dof + d_idx],
                      grad_jerk[b_addrs + (h_idx + 2) * dof + d_idx], 0.0, 0.0);
        g_u +=
          ((g_vel.x - g_vel.y) * dt +
           (g_acc.x - 2 * g_acc.y + g_acc.z) * dt * dt +
           (1 * g_jerk.x - 3 * g_jerk.y + 3 * g_jerk.z - g_jerk.w) * dt * dt * dt);
      }
      else if (h_idx == horizon - 2)
      {
        g_vel =
          make_float2(grad_velocity[b_addrs + (h_idx + 1) * dof + d_idx], 0.0);
        g_acc = make_float3(grad_acceleration[b_addrs + (h_idx + 1) * dof + d_idx],
                            0, 0);
        g_jerk = make_float4(grad_jerk[b_addrs + (h_idx + 1) * dof + d_idx], 0.0,
                             0.0, 0.0);
        g_u +=
          ((g_vel.x - g_vel.y) * dt +
           (g_acc.x - 2 * g_acc.y + g_acc.z) * dt * dt +
           (1 * g_jerk.x - 3 * g_jerk.y + 3 * g_jerk.z - g_jerk.w) * dt * dt * dt);
      }

      out_grad_position[b_addrs + (h_idx) * dof + d_idx] = g_u;
    }
  } // namespace
}
std::vector<torch::Tensor>step_position_clique(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_position, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor traj_dt, const int batch_size, const int horizon,
  const int dof)
{
  using namespace Curobo::TensorStep;
  assert(horizon < MAX_H);

  const int k_size    = batch_size * dof;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 512)
  {
    threadsPerBlock = 512;
  }

  int blocksPerGrid   = (k_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    out_position.scalar_type(), "step_position_clique", ([&] {
    position_clique_loop_kernel<scalar_t>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
      out_position.data_ptr<scalar_t>(),
      out_velocity.data_ptr<scalar_t>(),
      out_acceleration.data_ptr<scalar_t>(),
      out_jerk.data_ptr<scalar_t>(), u_position.data_ptr<scalar_t>(),
      start_position.data_ptr<scalar_t>(),
      start_velocity.data_ptr<scalar_t>(),
      start_acceleration.data_ptr<scalar_t>(),
      traj_dt.data_ptr<scalar_t>(), batch_size, horizon, dof);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_position, out_velocity, out_acceleration, out_jerk };
}

std::vector<torch::Tensor>step_position_clique2(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_position, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor traj_dt, const int batch_size, const int horizon,
  const int dof,
  const int mode = -1)
{
  using namespace Curobo::TensorStep;

  assert(horizon > 5);

  // assert(horizon < MAX_H);
  const int k_size    = batch_size * dof * horizon;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }

  int blocksPerGrid   = (k_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (mode == BWD_DIFF)
  {
    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_position_clique", ([&] {
      position_clique_loop_kernel2<scalar_t, BWD_DIFF>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_position.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        traj_dt.data_ptr<scalar_t>(), batch_size, horizon, dof);
    }));
  }
  else if (mode == CENTRAL_DIFF)
  {
    assert(u_position.sizes()[1] == horizon - 4);

    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_position_clique", ([&] {
      position_clique_loop_kernel2<scalar_t, CENTRAL_DIFF>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_position.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        traj_dt.data_ptr<scalar_t>(), batch_size, horizon, dof);
    }));
  }
  else
  {
    assert(false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_position, out_velocity, out_acceleration, out_jerk };
}

std::vector<torch::Tensor>step_position_clique2_idx(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_position, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor start_idx, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof,
  const int mode = -1)
{
  using namespace Curobo::TensorStep;

  // assert(horizon < MAX_H);
  assert(horizon > 5);


  const int k_size    = batch_size * dof * horizon;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }

  int blocksPerGrid   = (k_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (mode == BWD_DIFF)
  {
    assert(false);
    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_position_clique", ([&] {
      position_clique_loop_idx_kernel2<scalar_t, BWD_DIFF>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_position.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        start_idx.data_ptr<int32_t>(), traj_dt.data_ptr<scalar_t>(),
        batch_size, horizon, dof);
    }));
  }

  else if (mode == CENTRAL_DIFF)
  {
    assert(u_position.sizes()[1] == horizon - 4);

    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_position_clique", ([&] {
      position_clique_loop_idx_kernel2<scalar_t, CENTRAL_DIFF>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_position.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        start_idx.data_ptr<int32_t>(), traj_dt.data_ptr<scalar_t>(),
        batch_size, horizon, dof);
    }));
  }
  else
  {
    assert(false);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_position, out_velocity, out_acceleration, out_jerk };
}

std::vector<torch::Tensor>backward_step_position_clique(
  torch::Tensor out_grad_position, const torch::Tensor grad_position,
  const torch::Tensor grad_velocity, const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof)
{
  using namespace Curobo::TensorStep;

  assert(horizon < MAX_H - 4);
  const int k_size    = batch_size * dof;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }

  int blocksPerGrid   = (k_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
    out_grad_position.scalar_type(), "backward_step_position_clique", ([&] {
    backward_position_clique_loop_kernel<scalar_t>
      << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
      out_grad_position.data_ptr<scalar_t>(),
      grad_position.data_ptr<scalar_t>(),
      grad_velocity.data_ptr<scalar_t>(),
      grad_acceleration.data_ptr<scalar_t>(),
      grad_jerk.data_ptr<scalar_t>(), traj_dt.data_ptr<scalar_t>(),
      batch_size, horizon, dof);
  }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_grad_position };
}

std::vector<torch::Tensor>backward_step_position_clique2(
  torch::Tensor out_grad_position, const torch::Tensor grad_position,
  const torch::Tensor grad_velocity, const torch::Tensor grad_acceleration,
  const torch::Tensor grad_jerk, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof,
  const int mode = -1)
{
  // assert(horizon < MAX_H - 4);
  using namespace Curobo::TensorStep;

  assert(horizon > 5);


  // const int k_size = batch_size * dof;
  const int k_size    = batch_size * dof * horizon;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 128)
  {
    threadsPerBlock = 128;
  }

  int blocksPerGrid   = (k_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (mode == BWD_DIFF)
  {
    assert(false); // not supported anymore
    AT_DISPATCH_FLOATING_TYPES(
      out_grad_position.scalar_type(), "backward_step_position_clique", ([&] {
      backward_position_clique_loop_backward_difference_kernel2<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_grad_position.data_ptr<scalar_t>(),
        grad_position.data_ptr<scalar_t>(),
        grad_velocity.data_ptr<scalar_t>(),
        grad_acceleration.data_ptr<scalar_t>(),
        grad_jerk.data_ptr<scalar_t>(), traj_dt.data_ptr<scalar_t>(),
        batch_size, horizon, dof);
    }));
  }
  else if (mode == CENTRAL_DIFF)
  {
    assert(out_grad_position.sizes()[1] == horizon - 4);
    AT_DISPATCH_FLOATING_TYPES(
      out_grad_position.scalar_type(), "backward_step_position_clique", ([&] {
      backward_position_clique_loop_central_difference_kernel2<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_grad_position.data_ptr<scalar_t>(),
        grad_position.data_ptr<scalar_t>(),
        grad_velocity.data_ptr<scalar_t>(),
        grad_acceleration.data_ptr<scalar_t>(),
        grad_jerk.data_ptr<scalar_t>(), traj_dt.data_ptr<scalar_t>(),
        batch_size, horizon, dof);
    }));
  }
  else
  {
    assert(false);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_grad_position };
}

std::vector<torch::Tensor>
step_acceleration(torch::Tensor out_position, torch::Tensor out_velocity,
                  torch::Tensor out_acceleration, torch::Tensor out_jerk,
                  const torch::Tensor u_acc, const torch::Tensor start_position,
                  const torch::Tensor start_velocity,
                  const torch::Tensor start_acceleration,
                  const torch::Tensor traj_dt, const int batch_size,
                  const int horizon, const int dof, const bool use_rk2 = true)
{
  assert(horizon < MAX_H);
  using namespace Curobo::TensorStep;

  const int k_size    = batch_size * dof;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 512)
  {
    threadsPerBlock = 512;
  }

  int blocksPerGrid   = (k_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_rk2)
  {
    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_acceleration", ([&] {
      acceleration_loop_rk2_kernel<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_acc.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        traj_dt.data_ptr<scalar_t>(), batch_size, horizon, dof);
    }));
  }

  else
  {
    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_acceleration", ([&] {
      acceleration_loop_kernel<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_acc.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        traj_dt.data_ptr<scalar_t>(), batch_size, horizon, dof);
    }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_position, out_velocity, out_acceleration, out_jerk };
}

std::vector<torch::Tensor>step_acceleration_idx(
  torch::Tensor out_position, torch::Tensor out_velocity,
  torch::Tensor out_acceleration, torch::Tensor out_jerk,
  const torch::Tensor u_acc, const torch::Tensor start_position,
  const torch::Tensor start_velocity, const torch::Tensor start_acceleration,
  const torch::Tensor start_idx, const torch::Tensor traj_dt,
  const int batch_size, const int horizon, const int dof,
  const bool use_rk2 = true)
{
  assert(horizon < MAX_H);
  using namespace Curobo::TensorStep;


  const int k_size    = batch_size * dof;
  int threadsPerBlock = k_size;

  if (threadsPerBlock > 512)
  {
    threadsPerBlock = 512;
  }

  int blocksPerGrid   = (k_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_rk2)
  {
    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_acceleration", ([&] {
      acceleration_loop_idx_rk2_kernel<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_acc.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        start_idx.data_ptr<int32_t>(), traj_dt.data_ptr<scalar_t>(),
        batch_size, horizon, dof);
    }));
  }
  else
  {
    AT_DISPATCH_FLOATING_TYPES(
      out_position.scalar_type(), "step_acceleration", ([&] {
      acceleration_loop_idx_kernel<scalar_t>
        << < blocksPerGrid, threadsPerBlock, 0, stream >> > (
        out_position.data_ptr<scalar_t>(),
        out_velocity.data_ptr<scalar_t>(),
        out_acceleration.data_ptr<scalar_t>(),
        out_jerk.data_ptr<scalar_t>(), u_acc.data_ptr<scalar_t>(),
        start_position.data_ptr<scalar_t>(),
        start_velocity.data_ptr<scalar_t>(),
        start_acceleration.data_ptr<scalar_t>(),
        start_idx.data_ptr<int32_t>(), traj_dt.data_ptr<scalar_t>(),
        batch_size, horizon, dof);
    }));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return { out_position, out_velocity, out_acceleration, out_jerk };
}
