/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once



namespace curobo
{
    namespace trajectory
    {
     namespace legacy{


        template<typename ScalarType, bool useStencil=true>
        __device__ __forceinline__ void compute_central_difference(ScalarType       *out_position_mem,
                                                                   ScalarType       *out_velocity_mem,
                                                                   ScalarType       *out_acceleration_mem,
                                                                   ScalarType       *out_jerk_mem,
                                                                   ScalarType       *out_dt_mem,
                                                                   const ScalarType *u_position,
                                                                   const ScalarType *start_position,
                                                                   const ScalarType *start_velocity,
                                                                   const ScalarType *start_acceleration,
                                                                   const ScalarType *goal_position,
                                                                   const ScalarType *goal_velocity,
                                                                   const ScalarType *goal_acceleration,
                                                                   const ScalarType *traj_dt,
                                                                   const uint8_t *use_implicit_goal_state,
                                                                   const int       batch_size,
                                                                   const int       horizon,
                                                                   const int       dof,
                                                                   const int       b_idx,
                                                                   const int       h_idx,
                                                                   const int       d_idx,
                                                                   const int       b_offset,
                                                                   const int       goal_offset)
        {
          const float dt = traj_dt[goal_offset]; // assume same dt across traj TODO: Implement variable dt
          const bool use_goal = use_implicit_goal_state[goal_offset];
          // dt here is actually 1/dt;
          const float dt_inv  = 1.0 / dt;
          const float fixed_jerk = 0.0;   // Note: start jerk can also be passed from global memory
          // read start state:
          float out_pos = 0.0, out_vel = 0.0, out_acc = 0.0, out_jerk = 0.0;
          float fixed_pos = 0.0, fixed_vel = 0.0, fixed_acc = 0.0;

          const int   b_addrs_action = b_idx * (horizon - 4) * dof;
          float       in_pos[5]; // create a 5 value scalar
          float goal_pos = 0.0;
          if (use_goal)
          {
            goal_pos = goal_position[goal_offset * dof + d_idx];
          }
      #pragma unroll 5

          for (int i = 0; i < 5; i++)
          {
            in_pos[i] = 0.0;
          }

          if (h_idx < 5)
          {
            fixed_pos = start_position[b_offset * dof + d_idx];
            fixed_vel = start_velocity[b_offset * dof + d_idx];
            fixed_acc = start_acceleration[b_offset * dof + d_idx];
          }

          if ((h_idx > 3) && (h_idx < horizon - 5))
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
                        (-1 * fixed_acc * (dt * dt) - (dt * dt * dt) * fixed_jerk) -
                        3.0f * dt *
                        fixed_vel + fixed_pos;
            in_pos[1] = -2.0f * fixed_acc * dt * dt - (4.0f / 3) * dt * dt * dt *
                        fixed_jerk - 2.0 * dt * fixed_vel + fixed_pos;
            in_pos[2] = -(3.0f / 2) * fixed_acc * dt * dt - (7.0f / 6) * dt * dt * dt *
                        fixed_jerk - dt * fixed_vel + fixed_pos;
            in_pos[3] = fixed_pos;
            in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
          }

          else if (h_idx == 1)
          {
            in_pos[0] = -2.0f * fixed_acc * dt * dt - (4.0f / 3) * dt * dt * dt *
                        fixed_jerk - 2.0 * dt * fixed_vel + fixed_pos;
            in_pos[1] = -(3.0f / 2) * fixed_acc * dt * dt - (7.0f / 6) * dt * dt * dt *
                        fixed_jerk - dt * fixed_vel + fixed_pos;


            in_pos[2] = fixed_pos;
            in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
            in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
          }

          else if (h_idx == 2)
          {
            in_pos[0] = -(3.0f / 2) * fixed_acc * dt * dt - (7.0f / 6) * dt * dt * dt *
                        fixed_jerk - dt * fixed_vel + fixed_pos;
            in_pos[1] = fixed_pos;
            in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
            in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
            in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
          }
          else if (h_idx == 3)
          {
            in_pos[0] = fixed_pos;
            in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
            in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
            in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
            in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];
          }
          else if (h_idx == horizon - 5)
          {
            in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
            in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
            in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
            in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
            in_pos[4] = u_position[b_addrs_action + (h_idx) * dof + d_idx];

                                   // d_idx];
            if(use_goal)
            {
              in_pos[4] = goal_pos;
            }

          }
          else if (h_idx == horizon - 4)
          {
            in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
            in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
            in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];
            in_pos[3] = u_position[b_addrs_action + (h_idx - 1) * dof + d_idx];
                                   // d_idx];
            if(use_goal)
            {
              in_pos[3] = goal_pos;
            }
            in_pos[4] = in_pos[3]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +

          }

          else if (h_idx == horizon - 3)
          {
            in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
            in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];

            in_pos[2] = u_position[b_addrs_action + (h_idx - 2) * dof + d_idx];

            if(use_goal)
            {
              in_pos[2] = goal_pos;
            }
            in_pos[3] = in_pos[2]; // u_position[b_addrs_action + (h_idx - 1 + 1) * dof + d_idx];
            in_pos[4] = in_pos[2]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +
                                   // d_idx];
          }
          else if (h_idx == horizon - 2)
          {
            in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
            in_pos[1] = u_position[b_addrs_action + (h_idx - 3) * dof + d_idx];
            if (use_goal)
            {
              in_pos[1] = goal_pos;
            }
            in_pos[2] = in_pos[1];
            in_pos[3] = in_pos[1]; // u_position[b_addrs_action + (h_idx - 1 + 1) * dof + d_idx];
            in_pos[4] = in_pos[1]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +
                                   // d_idx];
          }

          else if (h_idx == horizon - 1)
          {
            in_pos[0] = u_position[b_addrs_action + (h_idx - 4) * dof + d_idx];
            if(use_goal)
            {
              in_pos[0] = goal_pos;
            }
            in_pos[1] = in_pos[0];
            in_pos[2] = in_pos[0]; // u_position[b_addrs_action + (h_idx - 1 ) * dof + d_idx];
            in_pos[3] = in_pos[0]; // u_position[b_addrs_action + (h_idx - 1 + 1) * dof + d_idx];
            in_pos[4] = in_pos[0]; // in_pos[3]; //u_position[b_addrs_action + (h_idx - 1 + 2) * dof +
                                   // d_idx];
          }
          out_pos = in_pos[2];
          if (useStencil)
          {

          out_vel =
            ((0.083333333f) * in_pos[0] - (0.666666667f) * in_pos[1] + (0.666666667f) * in_pos[3] +
            (-0.083333333f) * in_pos[4]) * dt_inv;

          out_acc =
           ((-0.083333333f) * in_pos[0] + (1.333333333f) * in_pos[1] + (-2.5f) * in_pos[2] +
            (1.333333333f) * in_pos[3] + (-0.083333333f) * in_pos[4]) * dt_inv * dt_inv;

          }
          else
          {

          out_vel = (-0.5*in_pos[1] + 0.5*in_pos[3]) * dt_inv;



          out_acc = (1.0f * in_pos[1] - 2.0f * in_pos[2] + 1.0f * in_pos[3]) * dt_inv * dt_inv;
          }
          out_jerk = ((-(1.0f/2.0f)) * in_pos[0] + in_pos[1]  - in_pos[3] + ((1.0f/2.0f)) * in_pos[4]) *
                     (dt_inv * dt_inv * dt_inv);

          // write out:
          out_position_mem[b_idx * horizon * dof + h_idx * dof + d_idx]     = out_pos;
          out_velocity_mem[b_idx * horizon * dof + h_idx * dof + d_idx]     = out_vel;
          out_acceleration_mem[b_idx * horizon * dof + h_idx * dof + d_idx] = out_acc;
          out_jerk_mem[b_idx * horizon * dof + h_idx * dof + d_idx]         = out_jerk;

          if (h_idx == 0 && d_idx == 0)
          {
            out_dt_mem[b_idx] = dt;
          }
        }


    template<typename ScalarType, bool useStencil=true>
    __global__ void position_clique_loop_idx_bwd_kernel(
      ScalarType *out_grad_position, const ScalarType *grad_position,
      const ScalarType *grad_velocity, const ScalarType *grad_acceleration,
      const ScalarType *grad_jerk,
      const ScalarType *traj_dt,
      const int32_t *dt_idx,
      const uint8_t  *use_implicit_goal_state,
      const int batch_size,
      const int horizon,
      const int dof)
    {
      const int tid = blockDim.x * blockIdx.x + threadIdx.x;


      // number of threads = batch_size * dof * (horizon - 4)
      const int action_horizon = horizon - 4;

      const int b_idx = tid / (dof * action_horizon);
      //const int d_idx = tid % dof;
      //const int h_idx = (tid / dof) % action_horizon; // batch_size * dof * action_horizon threads are there.
      //const int ah_idx = (tid / dof) % action_horizon; // ah_idx

      const int h_idx = tid % action_horizon;
      const int ah_idx = tid % action_horizon;
      const int d_idx = (tid / action_horizon) % dof;

     if (tid >= batch_size * dof * action_horizon)
      {
        return;
      }
      const int b_addrs        = b_idx * horizon * dof;
      const int b_addrs_action = b_idx * (action_horizon) * dof;


      const int dt_offset = dt_idx[b_idx];
      const float dt = traj_dt[dt_offset]; // need to index this as well.
      const bool use_goal = use_implicit_goal_state[dt_offset];
      const float dt_inv =  1.0f / dt;
      // read gradients:
      float out_grad = 0.0;
      float g_pos[3];
      float g_vel[5];
      float g_acc[5];
      float g_jerk[5];
      float dt_inv_2 = dt_inv * dt_inv;
      float dt_inv_3 = dt_inv_2 * dt_inv;

      if (useStencil)
      {
      #pragma unroll
      for (int i = 0; i < 5; i++)
        { // five point stencil: hid - 2, hid - 1, hid, hid + 1, hid + 2
          // we write by shifting with hid-2
          // So we actually read hid-4
          // hid - 4, hid - 3, hid - 2, hid - 1, hid
          g_vel[i]  = grad_velocity[b_addrs + ((ah_idx) + i) * dof + d_idx];
          g_acc[i]  = grad_acceleration[b_addrs + ((ah_idx) + i) * dof + d_idx];
          g_jerk[i] = grad_jerk[b_addrs + ((ah_idx) + i) * dof + d_idx];
        }

      }
      else
      {
        #pragma unroll
        for (int i = 0; i < 5; i++)
        {
          g_jerk[i] = 0.0;
          g_vel[i] = 0.0;
          g_acc[i] = 0.0;
        }
        g_vel[1] = grad_velocity[b_addrs + (ah_idx + 1) * dof + d_idx];
        g_vel[3] = grad_velocity[b_addrs + (ah_idx + 3) * dof + d_idx];
        g_acc[1] = grad_acceleration[b_addrs + (ah_idx + 1) * dof + d_idx];
        g_acc[2] = grad_acceleration[b_addrs + (ah_idx + 2) * dof + d_idx];
        g_acc[3] = grad_acceleration[b_addrs + (ah_idx + 3) * dof + d_idx];
        g_jerk[0] = grad_jerk[b_addrs + (ah_idx + 0) * dof + d_idx];
        g_jerk[1] = grad_jerk[b_addrs + (ah_idx + 1) * dof + d_idx];
        g_jerk[3] = grad_jerk[b_addrs + (ah_idx + 3) * dof + d_idx];
        g_jerk[4] = grad_jerk[b_addrs + (ah_idx + 4) * dof + d_idx];


      }
      // h_idx ranges from 0 to horizon - 4


      g_pos[0] = grad_position[b_addrs + (ah_idx + 2) * dof + d_idx];
      g_pos[1] = 0.0;
      g_pos[2] = 0.0;



      if(ah_idx == action_horizon - 1)
      {
        if (use_goal)
        {
          g_pos[0] = 0.0;
        }
        else
        {
          g_pos[1] = grad_position[b_addrs + (h_idx + 3) * dof + d_idx];
          g_pos[2] = grad_position[b_addrs + (h_idx + 4) * dof + d_idx];
          g_pos[0] += g_pos[1] + g_pos[2];
        }
      }

      // compute gradients:

      out_grad = g_pos[0];

      if (ah_idx < action_horizon - 1)
      {
        if (useStencil)
        {
        out_grad += (-0.0833333330000000 * g_vel[0] + 0.666666667000000 * g_vel[1] + 0 * g_vel[2] - 0.666666667000000 * g_vel[3] + 0.0833333330000000 * g_vel[4]) * dt_inv;
        out_grad += (-0.0833333330000000 * g_acc[0] + 1.33333333300000 * g_acc[1] + (-2.50000000000000) * g_acc[2] + 1.33333333300000 * g_acc[3] + (-0.0833333330000000) * g_acc[4]) * dt_inv_2;
        }
        else
        {


        out_grad += (0.5 * g_vel[1] - 0.5 * g_vel[3]) * dt_inv;

        out_grad += (1.0f * g_acc[1] - 2.0f * g_acc[2] + 1.0f * g_acc[3]) * dt_inv_2;
        }
        out_grad += (0.5f * g_jerk[0] - 1.0f * g_jerk[1] + 1.0f * g_jerk[3] - 0.5f * g_jerk[4]) * dt_inv_3;
      }
      else
      {
        if (use_goal)
        {
          out_grad = 0.0;
          /*
          out_grad =
          -0.0833333330000000 * g_vel[0] * dt_inv +
          -0.0833333330000000 * g_acc[0] * dt_inv_2 +
          0.5 * g_jerk[0] * dt_inv_3;
          */
        }
        else
        {
          if (useStencil)
          {

          out_grad += (-0.0833333330000000 * g_vel[0] + 0.583333334000000 * g_vel[1] + 0.583333334000000 * g_vel[2] - 0.0833333330000000 * g_vel[3] + 0.0 * g_vel[4]) * dt_inv;
          out_grad += (-0.0833333330000000 * g_acc[0] + 1.25000000000000 * g_acc[1] + (-1.25000000000000) * g_acc[2] + 0.0833333330000000 * g_acc[3]) * dt_inv_2;
          }
          else
          {
          out_grad += (0.5 * g_vel[1] - 0.5 * g_vel[3]) * dt_inv;

          out_grad += (1.0f * g_acc[1] - 2.0f * g_acc[2] + 1.0f * g_acc[3]) * dt_inv_2;

          }
          out_grad += (0.5 * g_jerk[0] - 0.5 * g_jerk[1] - 0.5 * g_jerk[2] + 0.5 * g_jerk[3]) * dt_inv_3;

        }
      }


      // write out:
      // forward uses h_idx - 4, h_idx - 3, h_idx - 2, h_idx - 1, h_idx
      // h_idx : 0, horizon - 4. As we skip h_idx < 2 and h_idx >= horizon - 2

      // h_idx is only valid for h_idx < horizon - 4.
      out_grad_position[b_addrs_action + (ah_idx) * dof + d_idx] = out_grad;
    }



    template<typename ScalarType, bool useStencil=true>
    __global__ void position_clique_loop_idx_fwd_kernel(
      ScalarType *out_position_mem, ScalarType *out_velocity_mem,
      ScalarType *out_acceleration_mem, ScalarType *out_jerk_mem,
      ScalarType *out_dt_mem,
      const ScalarType *u_position,
      const ScalarType *start_position,
      const ScalarType *start_velocity, const ScalarType *start_acceleration,
      const ScalarType *goal_position,
      const ScalarType *goal_velocity, const ScalarType *goal_acceleration,
      const int32_t *start_idx,
      const int32_t *goal_idx,
      const ScalarType *traj_dt,
      const uint8_t *use_implicit_goal_state,
      const int batch_size,
      const int horizon, const int dof)
    {
      const int tid = blockDim.x * blockIdx.x + threadIdx.x;

      // number of threads = batch_size * dof * horizon;
      const int b_idx = tid / (dof * horizon);

      const int d_idx = tid % dof;
      const int h_idx = (tid / dof) % horizon;

      //const int h_idx = tid % horizon;
      //const int d_idx = (tid / horizon) % dof;

      if (tid >= batch_size * dof * horizon)
      {
        return;
      }


      const int b_offset = start_idx[b_idx];
      const int goal_offset = goal_idx[b_idx];


      compute_central_difference<ScalarType, useStencil>(out_position_mem,
                                out_velocity_mem,
                                out_acceleration_mem,
                                out_jerk_mem,
                                out_dt_mem,
                                u_position,
                                start_position,
                                start_velocity,
                                start_acceleration,
                                goal_position,
                                goal_velocity,
                                goal_acceleration,
                                traj_dt,
                                use_implicit_goal_state,
                                batch_size,
                                horizon,
                                dof,
                                b_idx,
                                h_idx,
                                d_idx,
                                b_offset,
                                goal_offset);

    }
    }
}
}