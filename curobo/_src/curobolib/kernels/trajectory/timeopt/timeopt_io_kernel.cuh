// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace curobo::trajectory::timeopt {

__global__ void fused_inputs_and_path(const float *controls, const float *start, const float *goal,
                                      const float *velocity, const float *acceleration,
                                      const float *jerk, const float *knot_dt,
                                      float2 *reference_speed, float *saved_controls,
                                      float *saved_start, float *saved_goal, bool *valid,
                                      bool *stationary, const int *segments, const float *local,
                                      float *path, float *q, float *qd, float *qdd, int knots,
                                      int dof, int points, int batch_size, int start_offset) {
  const int batch = blockIdx.y, tid = threadIdx.x;
  const int item = blockIdx.x * blockDim.x + tid, span = points * dof;
  const float dt = knot_dt[batch];
  float boundary[4];
  const int input = batch * dof + item % dof;
  bspline::compute_fixed_knots<3>(
      boundary,
      bspline::BoundaryConstraint(start[input], velocity[input], acceleration[input], jerk[input]),
      dt, dt * dt, dt * dt * dt);
  if (item < span) {
    const int point = item / dof, joint = item % dof, index = batch * span + item;
    float v[4];
    sample_joint(controls, boundary, goal, batch, joint, dof, knots, start_offset, segments[point],
                 local[point], v);
    for (int k = 0; k < 4; ++k) path[k * batch_size * span + index] = v[k];
    const int offset = batch * 3 * span + item;
    q[offset] = q[offset + span] = q[offset + 2 * span] = v[0];
    qd[offset] = qd[offset + span] = qdd[offset] = 0.0f;
    qd[offset + 2 * span] = qdd[offset + span] = v[1];
    qdd[offset + 2 * span] = v[2];
  }
  // Only block zero retains input and reduces metadata; no duplicate global writes.
  if (blockIdx.x != 0) return;
  const float duration = dt * (knots + 3 - start_offset);
  const float speed = start_offset ? 0.0f : 1.0f / (duration * duration);
  float finite = start_offset || (dt > 0.0f && isfinite(dt) && speed > 0.0f && isfinite(speed));
  float constant = 1.0f, static_start = 1.0f;
  for (int item = tid; item < knots * dof; item += blockDim.x) {
    const float value = controls[batch * knots * dof + item];
    saved_controls[batch * knots * dof + item] = value;
    finite *= isfinite(value);
    if (item < (knots - 1) * dof) constant *= value == start[batch * dof + item % dof];
  }
  if (tid < dof) {
    const int item = batch * dof + tid;
    static_start = velocity[item] == 0.0f && acceleration[item] == 0.0f && jerk[item] == 0.0f;
    for (int k = 0; k < 4; ++k) {
      saved_start[(batch * 4 + k) * dof + tid] = boundary[k];
      finite *= isfinite(boundary[k]);
      constant *= boundary[k] == start[item];
    }
    saved_goal[item] = goal[item];
    finite *= isfinite(start[item]) && isfinite(goal[item]);
    constant *= start[item] == goal[item];
  }
  __shared__ float scratch[3 * TIMEOPT_WARPS];
  finite = fused_block_min(finite, scratch);
  constant = fused_block_min(constant, scratch + TIMEOPT_WARPS);
  static_start = fused_block_min(static_start, scratch + 2 * TIMEOPT_WARPS);
  if (tid == 0) {
    valid[batch] = finite;
    stationary[batch] = constant;
    reference_speed[batch] = make_float2(speed, !start_offset && !static_start);
  }
}

// Uniform squared speed yields qdot=q', qddot=q'', jerk=q''', tau=g+alpha*B.
// Retain the three dynamics coefficients and reuse B/g directly for seed bounds.
__global__ void fused_coefficients_and_seed_bounds(const float *path, const float *torque,
                                                   float *coefficients, const float *vlim,
                                                   const float *alim, const float *jlim,
                                                   const float *lower, const float *upper,
                                                   float *out, int points, int dof, int batch_size,
                                                   int tiles) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x, batch = blockIdx.y;
  const int count = points * dof;
  float hi = fused_infinity(), lo = 0.0f, valid = 1.0f;
  if (item < count) {
    const int joint = item % dof, index = batch * count + item;
    const float v = fabsf(path[batch_size * count + index]);
    const float a = fabsf(path[2 * batch_size * count + index]);
    const float j = fabsf(path[3 * batch_size * count + index]);
    const int offset = batch * 3 * count + item;
    const float g = torque[offset];
    const float dynamic = torque[offset + 2 * count] - g;
    coefficients[offset] = torque[offset + count] - g;
    coefficients[offset + count] = dynamic;
    coefficients[offset + 2 * count] = g;
    valid = isfinite(v) && isfinite(a) && isfinite(j) && isfinite(dynamic) && isfinite(g);
    if (v > 0.0f) hi = fminf(hi, (vlim[joint] / v) * (vlim[joint] / v));
    if (a > 0.0f) hi = fminf(hi, alim[joint] / a);
    if (j > 0.0f) hi = fminf(hi, powf(jlim[joint] / j, 2.0f / 3.0f));
    if (dynamic != 0.0f) {
      const float left = (lower[joint] - g) / dynamic;
      const float right = (upper[joint] - g) / dynamic;
      lo = fmaxf(lo, fminf(left, right));
      hi = fminf(hi, fmaxf(left, right));
    } else
      valid *= g >= lower[joint] && g <= upper[joint];
  }
  __shared__ float scratch[3 * TIMEOPT_WARPS];
  hi = fused_block_min(hi, scratch);
  lo = -fused_block_min(-lo, scratch + TIMEOPT_WARPS);
  valid = fused_block_min(valid, scratch + 2 * TIMEOPT_WARPS);
  if (threadIdx.x == 0) {
    out[(batch * tiles + blockIdx.x) * 3] = hi;
    out[(batch * tiles + blockIdx.x) * 3 + 1] = lo;
    out[(batch * tiles + blockIdx.x) * 3 + 2] = valid;
  }
}

__device__ __forceinline__ float2 fused_seed(const float *bounds, const float2 *reference_speed,
                                             float *state, bool *feasible, bool *numerical,
                                             float *uniform, int tiles) {
  const int batch = blockIdx.y, tid = threadIdx.x;
  float hi = fused_infinity(), lo = 0.0f, valid = 1.0f;
  for (int tile = tid; tile < tiles; tile += blockDim.x) {
    hi = fminf(hi, bounds[(batch * tiles + tile) * 3]);
    lo = fmaxf(lo, bounds[(batch * tiles + tile) * 3 + 1]);
    valid = fminf(valid, bounds[(batch * tiles + tile) * 3 + 2]);
  }
  __shared__ float scratch[3 * TIMEOPT_WARPS];
  hi = fused_block_min(hi, scratch);
  lo = -fused_block_min(-lo, scratch + TIMEOPT_WARPS);
  valid = fused_block_min(valid, scratch + 2 * TIMEOPT_WARPS);
  valid *= hi > 0.0f && hi >= lo;
  if (!isfinite(hi)) hi = 1.0f;
  const float reference = reference_speed[batch].x;
  const bool fixed_start = reference_speed[batch].y != 0.0f;
  if (fixed_start)
    valid = reference > 0.0f && isfinite(reference);
  else if (reference != 0.0f)
    valid *= reference > 0.0f && isfinite(reference);
  float base =
      fixed_start ? (valid ? reference : 1.0f) : (valid ? fmaxf(0.5f * (hi + lo), 1e-12f) : 1.0f);
  float y = !fixed_start && valid ? (hi - 1e-4f * (hi - lo)) / base : 1.0f;
  if (!fixed_start && reference != 0.0f && valid) {
    // Static starts permit slowing a supplied time law to honor tighter timeopt limits.
    const float margin = 1e-4f * (hi - lo);
    base = fmaxf(fminf(reference, hi - margin), fmaxf(lo + margin, 1e-12f));
    y = 1.0f;
  }
  float *st = state + batch * F_SIZE;
  if (blockIdx.x == 0 && tid < TIMEOPT_CONTROLS)
    st[F_CONTROLS + tid] = st[F_BEST_CONTROLS + tid] = y;
  if (blockIdx.x == 0 && tid == 0) {
    st[F_SPEED_SCALE] = base;
    st[F_FIXED_CONTROLS] = fixed_start ? 2.0f : 0.0f;
    st[F_BEST_COST] = rsqrtf(y);
    st[F_ACCEPTED] = 0.0f;
    st[F_ITERATION_VALID] = st[F_NUMERICAL_OK] = 1.0f;
    feasible[batch] = valid;
    numerical[batch] = true;
    uniform[batch] = st[F_BEST_COST] * rsqrtf(base);
  }
  return make_float2(base, valid);
}

// Reconstruct analytic states and the exact sampled scale interval. Independent
// fine-grid benchmark validation remains outside the timed runtime.
__global__ void fused_output(const float *timing, const float4 *basis, const int *first,
                             const float *path, const float *torque, const float *vlim,
                             const float *alim, const float *jlim, const float *lower,
                             const float *upper, const float4 *time_basis, const int *time_first,
                             const float *time_weights, float *partial_bounds, float *dt,
                             float *output, float *output_torque, int points, int dof,
                             int batch_size, int intervals, int tiles) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x, batch = blockIdx.y;
  const int count = points * dof, plane = batch_size * count;
  const float *c = timing + batch * TIMEOPT_CONTROLS;
  float hi = fused_infinity(), lo = 0.0f, valid = 1.0f;
  if (item < count) {
    const int point = item / dof, joint = item % dof, index = batch * count + item;
    const float x = dot4(basis[point], c + first[point]);
    const float xp = dot4(basis[points + point], c + first[point]);
    const float xpp = dot4(basis[2 * points + point], c + first[point]);
    const float root = sqrtf(fmaxf(x, 0.0f));
    const float q1 = path[plane + index], q2 = path[2 * plane + index],
                q3 = path[3 * plane + index];
    const float velocity = q1 * root;
    const float acceleration = q2 * x + 0.5f * q1 * xp;
    const float jerk = root * (q3 * x + 1.5f * q2 * xp + 0.5f * q1 * xpp);
    const float dynamic =
        0.5f * torque[(batch * 3) * count + item] * xp + torque[(batch * 3 + 1) * count + item] * x;
    const float g = torque[(batch * 3 + 2) * count + item];
    const float v = fabsf(velocity), a = fabsf(acceleration), j = fabsf(jerk);
    valid = x > 0.0f && isfinite(x) && isfinite(v) && isfinite(a) && isfinite(j) &&
            isfinite(dynamic) && isfinite(g);
    if (v > 0.0f) hi = fminf(hi, (vlim[joint] / v) * (vlim[joint] / v));
    if (a > 0.0f) hi = fminf(hi, alim[joint] / a);
    if (j > 0.0f) hi = fminf(hi, powf(jlim[joint] / j, 2.0f / 3.0f));
    if (dynamic != 0.0f) {
      const float left = (lower[joint] - g) / dynamic, right = (upper[joint] - g) / dynamic;
      lo = fmaxf(lo, fminf(left, right));
      hi = fminf(hi, fmaxf(left, right));
    } else
      valid *= g >= lower[joint] && g <= upper[joint];
    if (point <= intervals) {
      const int out_count = (intervals + 1) * dof;
      const int out_plane = batch_size * out_count, out_index = batch * out_count + item;
      output[out_index] = path[index];
      output[out_plane + out_index] = velocity;
      output[2 * out_plane + out_index] = acceleration;
      output[3 * out_plane + out_index] = jerk;
      output_torque[out_index] = dynamic + g;
    }
  }
  if (item < intervals) {
    float value = 0.0f;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      const int q = item * 6 + k;
      value += time_weights[k] * rsqrtf(fmaxf(dot4(time_basis[q], c + time_first[q]), 1e-20f));
    }
    dt[batch * intervals + item] = value / intervals;
  }
  __shared__ float scratch[3 * TIMEOPT_WARPS];
  hi = fused_block_min(hi, scratch);
  lo = -fused_block_min(-lo, scratch + TIMEOPT_WARPS);
  valid = fused_block_min(valid, scratch + 2 * TIMEOPT_WARPS);
  if (threadIdx.x == 0) {
    partial_bounds[(batch * tiles + blockIdx.x) * 3] = hi;
    partial_bounds[(batch * tiles + blockIdx.x) * 3 + 1] = lo;
    partial_bounds[(batch * tiles + blockIdx.x) * 3 + 2] = valid;
  }
}

__device__ __forceinline__ void fused_output_status(const float *bounds, const bool *initialized,
                                                    const float *uniform, const bool *numerical,
                                                    const bool *input_valid, const bool *stationary,
                                                    float *output_uniform, bool *output_numerical,
                                                    bool *success, int *status, int tiles) {
  const int batch = blockIdx.x, tid = threadIdx.x;
  float hi = fused_infinity(), lo = 0.0f, valid = 1.0f;
  for (int tile = tid; tile < tiles; tile += blockDim.x) {
    hi = fminf(hi, bounds[(batch * tiles + tile) * 3]);
    lo = fmaxf(lo, bounds[(batch * tiles + tile) * 3 + 1]);
    valid = fminf(valid, bounds[(batch * tiles + tile) * 3 + 2]);
  }
  __shared__ float scratch[3 * TIMEOPT_WARPS];
  hi = fused_block_min(hi, scratch);
  lo = -fused_block_min(-lo, scratch + TIMEOPT_WARPS);
  valid = fused_block_min(valid, scratch + 2 * TIMEOPT_WARPS);
  valid *= hi > 0.0f && hi >= lo && hi >= 1.0f - 2e-5f && lo <= 1.0f + 2e-5f &&
           initialized[batch] && input_valid[batch];
  if (tid == 0) {
    success[batch] = valid;
    status[batch] = !input_valid[batch] ? 3 : (!initialized[batch] ? 1 : (valid ? 0 : 2));
    output_uniform[batch] =
        stationary[batch] ? 0.0f : (initialized[batch] ? uniform[batch] : fused_infinity());
    output_numerical[batch] = numerical[batch] && initialized[batch] && input_valid[batch];
  }
}

// One block per path scans time in fixed-size chunks and masks failed outputs.
__device__ __forceinline__ void fused_output_finalize(const float *timing, const float *dt,
                                                      const bool *stationary, const bool *success,
                                                      float *output_timing, float *output,
                                                      float *torque, float *time, int intervals,
                                                      int dof, int batch_size) {
  const int batch = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const float nan_value = __int_as_float(0x7fc00000);
  if (tid < TIMEOPT_CONTROLS)
    output_timing[batch * TIMEOPT_CONTROLS + tid] = timing[batch * TIMEOPT_CONTROLS + tid];
  if (!success[batch]) {
    const int count = (intervals + 1) * dof, plane = batch_size * count;
    for (int item = tid; item < count; item += blockDim.x) {
      const int idx = batch * count + item;
#pragma unroll
      for (int k = 0; k < 4; ++k) output[k * plane + idx] = nan_value;
      torque[idx] = nan_value;
    }
  }
  __shared__ float warp_sums[TIMEOPT_WARPS], carry;
  if (tid == 0) {
    carry = 0.0f;
    time[batch * (intervals + 1)] = success[batch] ? 0.0f : nan_value;
  }
  __syncthreads();
  for (int begin = 0; begin < intervals; begin += blockDim.x) {
    const int item = begin + tid;
    float value = item < intervals ? dt[batch * intervals + item] : 0.0f;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float previous = __shfl_up_sync(FULL_WARP, value, offset);
      if (lane >= offset) value += previous;
    }
    if (lane == 31) warp_sums[warp] = value;
    __syncthreads();
    float prefix = carry;
    for (int w = 0; w < warp; ++w) prefix += warp_sums[w];
    value += prefix;
    if (item < intervals)
      time[batch * (intervals + 1) + item + 1] =
          !success[batch] ? nan_value : (stationary[batch] ? 0.0f : value);
    __syncthreads();
    if (tid == blockDim.x - 1) carry = value;
    __syncthreads();
  }
}

// Publish status before prefix integration reads it within the same path block.
__global__ void fused_finish(const float *bounds, const bool *initialized, const float *uniform,
                             const bool *numerical, const bool *input_valid, const bool *stationary,
                             float *output_uniform, bool *output_numerical, bool *success,
                             int *status, const float *timing, const float *dt,
                             float *output_timing, float *output, float *torque, float *time,
                             int intervals, int dof, int batch_size, int tiles) {
  fused_output_status(bounds, initialized, uniform, numerical, input_valid, stationary,
                      output_uniform, output_numerical, success, status, tiles);
  __syncthreads();
  fused_output_finalize(timing, dt, stationary, success, output_timing, output, torque, time,
                        intervals, dof, batch_size);
}

}  // namespace curobo::trajectory::timeopt
