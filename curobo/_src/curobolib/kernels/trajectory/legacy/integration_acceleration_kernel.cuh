/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace curobo{
    namespace trajectory{
    namespace legacy{

    template<typename ScalarType, int maxHorizon>
    __global__ void acceleration_loop_idx_kernel(
      ScalarType *out_position_mem, ScalarType *out_velocity_mem,
      ScalarType *out_acceleration_mem, ScalarType *out_jerk_mem,
      const ScalarType *u_acc, const ScalarType *start_position,
      const ScalarType *start_velocity, const ScalarType *start_acceleration,
      const int32_t *start_idx, const ScalarType *traj_dt, const int batch_size,
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
      float u_arr[maxHorizon], dt[maxHorizon];
      float out_pos[maxHorizon], out_vel[maxHorizon], out_acc[maxHorizon], out_jerk[maxHorizon];
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

    template<typename ScalarType, int maxHorizon>
    __global__ void acceleration_loop_idx_rk2_kernel(
      ScalarType *out_position_mem, ScalarType *out_velocity_mem,
      ScalarType *out_acceleration_mem, ScalarType *out_jerk_mem,
      const ScalarType *u_acc, const ScalarType *start_position,
      const ScalarType *start_velocity, const ScalarType *start_acceleration,
      const int32_t *start_idx, const ScalarType *traj_dt, const int batch_size,
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
      float u_arr[maxHorizon], dt[maxHorizon];
      float out_pos[maxHorizon], out_vel[maxHorizon], out_acc[maxHorizon], out_jerk[maxHorizon];
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

    }
}
}