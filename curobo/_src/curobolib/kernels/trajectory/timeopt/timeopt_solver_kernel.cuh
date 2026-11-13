// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace curobo::trajectory::timeopt {

// One record per sample/joint. The four rows are squared velocity, acceleration,
// torque and the linear jerk factor; signed pairs share a derivative vector.
__global__ void fused_prepare(const int2 *meta, const int *offsets, const float4 *basis,
                              const float *path,
                              const float *torque, const float *vlim, const float *alim,
                              const float *jlim, const float *lower, const float *upper,
                              const float *seed_bounds, const float2 *reference_speed, float *state,
                              bool *feasible, bool *numerical, float *uniform, float4 *rows,
                              float *bounds, int points, int dof, int batch_size, int tiles,
                              int seed_tiles) {
  // Every tile reduces the small uniform bounds. Only tile zero publishes state;
  // all tiles use their local seed, so preparation needs no global barrier.
  const float2 seed =
      fused_seed(seed_bounds, reference_speed, state, feasible, numerical, uniform, seed_tiles);
  if (threadIdx.x >= TIMEOPT_RECORDS) return;
  const int local = blockIdx.x * TIMEOPT_RECORDS + threadIdx.x, batch = blockIdx.y;
  const int size = batch_size * tiles * TIMEOPT_RECORDS;
  const int row = batch * tiles * TIMEOPT_RECORDS + local;
  const int kind = meta[local].x, item = meta[local].y, first = offsets[blockIdx.x];
  float a[4][4] = {}, upper_bound = 1.0f, lower_bound = 1.0f;
  if (kind == JOINT_LIMITS && seed.y) {
    const int point = item / dof, joint = item % dof, count = points * dof;
    const float4 n0 = basis[point], n1 = basis[points + point], n2 = basis[2 * points + point];
    const float n[4] = {n0.x, n0.y, n0.z, n0.w};
    const float np[4] = {n1.x, n1.y, n1.z, n1.w};
    const float npp[4] = {n2.x, n2.y, n2.z, n2.w};
    const int index = batch * count + item;
    const float q1 = path[batch_size * count + index];
    const float q2 = path[2 * batch_size * count + index];
    const float q3 = path[3 * batch_size * count + index];
    const float dynamic_a = torque[batch * 3 * count + item];
    const float dynamic_b = torque[(batch * 3 + 1) * count + item];
    const float gravity = torque[(batch * 3 + 2) * count + item];
    const float tau_scale = fmaxf(fmaxf(fabsf(lower[joint]), fabsf(upper[joint])), 1e-12f);
    upper_bound = (upper[joint] - gravity) / tau_scale;
    lower_bound = (gravity - lower[joint]) / tau_scale;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      a[0][k] = seed.x * (q1 * q1 * n[k] / (vlim[joint] * vlim[joint]));
      a[1][k] = seed.x * ((q2 * n[k] + 0.5f * q1 * np[k]) / alim[joint]);
      a[2][k] = seed.x * ((0.5f * dynamic_a * np[k] + dynamic_b * n[k]) / tau_scale);
      a[3][k] = seed.x * ((q3 * n[k] + 1.5f * q2 * np[k] + 0.5f * q1 * npp[k]) / jlim[joint]);
    }
    // At s=0, fixed x and x' make velocity, acceleration and torque constant.
    // Exclude these rows from the barrier (they may sit exactly on a limit).
    // Final output validation still checks them; jerk remains a free constraint.
    if (point == 0 && reference_speed[batch].y != 0.0f) {
      for (int family = 0; family < 3; ++family)
        for (int k = 0; k < 4; ++k) a[family][k] = 0.0f;
      upper_bound = lower_bound = 1.0f;
    }
  } else if (kind == POSITIVE_CONTROL) {
    a[0][item - first] = -1.0f;
    upper_bound = 0.0f;
  } else if (kind == DURATION) {
    return;
  }
#pragma unroll
  for (int k = 0; k < 4; ++k)
    rows[k * size + row] = make_float4(a[k][0], a[k][1], a[k][2], a[k][3]);
  bounds[row] = upper_bound;
  bounds[size + row] = lower_bound;
}

__device__ __forceinline__ float fused_dual(float *row_state, const float *st, int row, int size,
                                            int constraint, bool initialize) {
  const int index = constraint * size + row;
  float dual = initialize ? 1e-4f : row_state[ROW_DUAL * size + index];
  if (!initialize && st[F_ACCEPTED])
    dual = fmaxf(dual + st[F_DUAL_STEP] * row_state[ROW_DUAL_DIRECTION * size + index], 1e-12f);
  row_state[ROW_DUAL * size + index] = dual;
  return dual;
}

__device__ __forceinline__ void fused_outer(float *values, float4 a4, float weight) {
  const float a[4] = {a4.x, a4.y, a4.z, a4.w};
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j <= i; ++j)
      values[ASSEMBLY_HESSIAN + i * (i + 1) / 2 + j] += weight * a[i] * a[j];
}

__global__ void fused_assemble(const int2 *meta, const int *offsets, const float4 *basis,
                               const float4 *rows, const float *bounds, const float *state,
                               float *row_state, float4 *jacobian, float *partials, int points,
                               int duration_points, int dof, int batch_size, int tiles,
                               int initialize) {
  const int tid = threadIdx.x / 4, family = threadIdx.x % 4;
  const int local = blockIdx.x * TIMEOPT_RECORDS + tid, batch = blockIdx.y;
  const int size = batch_size * tiles * TIMEOPT_RECORDS;
  const int row = batch * tiles * TIMEOPT_RECORDS + local;
  const int kind = meta[local].x, first = offsets[blockIdx.x];
  const float *st = state + batch * F_SIZE;
  const float *y = st + F_CONTROLS + first;
  float values[PARTIALS_PER_TILE] = {};
  // Sorted joint records make each point's first lane a shared timing producer.
  // A point cut by a tile boundary is produced by lane zero of the next tile.
  __shared__ float roots[TIMEOPT_RECORDS];
  const int leader = max(0, tid - meta[local].y % dof);
  if (kind == JOINT_LIMITS && family == 0 && tid == leader)
    roots[tid] = sqrtf(st[F_SPEED_SCALE] * dot4(basis[meta[local].y / dof], y));
  __syncthreads();
  if (kind == JOINT_LIMITS || kind == POSITIVE_CONTROL) {
    if (family == 0) {
      const float4 velocity = rows[row];
      float slack = (kind == POSITIVE_CONTROL ? 0.0f : 1.0f) - dot4(velocity, y);
      const float dual = fused_dual(row_state, st, row, size, 0, initialize);
      row_state[ROW_SLACK * size + row] = slack;
      values[ASSEMBLY_INVALID] += !(slack > 0.0f && isfinite(slack));
      fused_outer(values, velocity, dual / fmaxf(slack, 1e-12f));
    }
    if (kind == JOINT_LIMITS && family > 0) {
      const float4 a4 = rows[family * size + row];
      const float value = dot4(a4, y);
      const float root = roots[leader];
      float4 jac = a4;
      if (family == 3) {
        const float4 n = basis[meta[local].y / dof];
        const float cross = 0.5f * st[F_SPEED_SCALE] / root;
        jac = make_float4(root * a4.x + cross * value * n.x, root * a4.y + cross * value * n.y,
                          root * a4.z + cross * value * n.z, root * a4.w + cross * value * n.w);
        jacobian[row] = jac;
      }
      const float v = family == 3 ? root * value : value;
      const float sp = (family == 2 ? bounds[row] : 1.0f) - v;
      const float sn = (family == 2 ? bounds[size + row] : 1.0f) + v;
      const int positive = 2 * family - 1, negative = 2 * family;
      const float dp = fused_dual(row_state, st, row, size, positive, initialize);
      const float dn = fused_dual(row_state, st, row, size, negative, initialize);
      row_state[(ROW_SLACK + positive) * size + row] = sp;
      row_state[(ROW_SLACK + negative) * size + row] = sn;
      values[ASSEMBLY_INVALID] += !(sp > 0.0f && isfinite(sp) && sn > 0.0f && isfinite(sn));
      fused_outer(values, jac, dp / fmaxf(sp, 1e-12f) + dn / fmaxf(sn, 1e-12f));
      if (family == 3) {
        const float4 n4 = basis[meta[local].y / dof];
        const float n[4] = {n4.x, n4.y, n4.z, n4.w};
        const float a[4] = {a4.x, a4.y, a4.z, a4.w};
        const float cross = 0.5f * st[F_SPEED_SCALE] / root;
        const float x = st[F_SPEED_SCALE] * dot4(n4, y);
        const float second = -0.25f * st[F_SPEED_SCALE] * st[F_SPEED_SCALE] * value / (x * root);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j <= i; ++j)
            values[ASSEMBLY_HESSIAN + i * (i + 1) / 2 + j] +=
                (dp - dn) * (cross * (a[i] * n[j] + n[i] * a[j]) + second * n[i] * n[j]);
      }
    }
  } else if (kind == DURATION && family == 0) {
    const float4 a4 = basis[meta[local].y];
    const float a[4] = {a4.x, a4.y, a4.z, a4.w};
    const float z = fmaxf(dot4(a4, y), 1e-20f), root = rsqrtf(z);
    const float g = -0.5f * __frcp_rn(float(duration_points)) * root / z, h = -1.5f * g / z;
#pragma unroll
    for (int i = 0; i < 4; ++i) values[ASSEMBLY_GRADIENT + i] = g * a[i];
    fused_outer(values, a4, h);
  }
  __shared__ float scratch[PARTIALS_PER_TILE * TIMEOPT_WARPS];
  reduce_sums(values, partials + (batch * tiles + blockIdx.x) * PARTIALS_PER_TILE, scratch);
}

// Factor S*H*S + 1e-6*I = L*L^T, where S[i] = 1/sqrt(H[i,i]).
// Lane i owns band[k] = H[i,i-k], k=0..3, and overwrites it with L[i,i-k].
// One warp owns all 32 controls, with every lane participating in shuffles.
// A nonpositive/nonfinite pivot invalidates the iteration; clamping only keeps
// the remaining arithmetic finite while the caller suppresses the failed step.
__device__ __forceinline__ bool factor_banded_cholesky(float (&band)[4], float &diagonal_scale) {
  const int lane = threadIdx.x % 32;
  diagonal_scale = rsqrtf(fmaxf(band[0], 1e-20f));
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float other = __shfl_sync(FULL_WARP, diagonal_scale, max(0, lane - k));
    band[k] = lane >= k ? band[k] * diagonal_scale * other : 0.0f;
  }
  band[0] += 1e-6f;
  bool ok = true;
#pragma unroll
  for (int pivot = 0; pivot < TIMEOPT_CONTROLS; ++pivot) {
    const float diagonal = __shfl_sync(FULL_WARP, band[0], pivot);
    ok = ok && diagonal > 0.0f && isfinite(diagonal);
    const float inverse_root = rsqrtf(fmaxf(diagonal, 1e-20f));
    const float root = fmaxf(diagonal, 1e-20f) * inverse_root;
    float factor = 0.0f;
    if (lane == pivot) band[0] = root;
    if (lane > pivot && lane <= pivot + 3) {
      factor = band[lane - pivot] * inverse_root;
      band[lane - pivot] = factor;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float other = __shfl_sync(FULL_WARP, factor, max(0, lane - k));
      if (lane - k > pivot) band[k] -= factor * other;
    }
  }
  return ok;
}

// A bandwidth-three triangular substitution is a scan of affine maps on
// [x_i, x_{i-1}, x_{i-2}]. Composition doubles the solved prefix each round.
// The scalar recurrence is x_i = rhs_i + c0*x_(i-1) + c1*x_(i-2) + c2*x_(i-3).
// Five rounds cover all controls. Composition changes FP32 rounding relative
// to sequential substitution, so numerical checks must cover late iterations.
__device__ __forceinline__ float scan_banded_recurrence(float c0, float c1, float c2, float rhs) {
  const int lane = threadIdx.x % 32;
  float matrix[9] = {c0, c1, c2, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  float value[3] = {rhs, 0.0f, 0.0f};
#pragma unroll
  for (int offset = 1; offset < TIMEOPT_CONTROLS; offset *= 2) {
    float previous_matrix[9], previous_value[3], next_matrix[9], next_value[3];
#pragma unroll
    for (int k = 0; k < 9; ++k) previous_matrix[k] = __shfl_up_sync(FULL_WARP, matrix[k], offset);
#pragma unroll
    for (int k = 0; k < 3; ++k) previous_value[k] = __shfl_up_sync(FULL_WARP, value[k], offset);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      next_value[i] = value[i];
#pragma unroll
      for (int k = 0; k < 3; ++k) next_value[i] += matrix[i * 3 + k] * previous_value[k];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        next_matrix[i * 3 + j] = 0.0f;
#pragma unroll
        for (int k = 0; k < 3; ++k)
          next_matrix[i * 3 + j] += matrix[i * 3 + k] * previous_matrix[k * 3 + j];
      }
    }
    if (lane >= offset) {
#pragma unroll
      for (int k = 0; k < 9; ++k) matrix[k] = next_matrix[k];
#pragma unroll
      for (int k = 0; k < 3; ++k) value[k] = next_value[k];
    }
  }
  return value[0];
}

// Apply x = S * L^-T * L^-1 * S * rhs using the stored scaled factor.
// Reverse lane order for the upper solve so both passes use the same prefix scan.
__device__ __forceinline__ bool solve_banded_cholesky(const float (&band)[4], float diagonal_scale,
                                                      float rhs, float &solution) {
  const int lane = threadIdx.x % 32;
  const float inverse = 1.0f / fmaxf(band[0], 1e-20f);
  solution = rhs * diagonal_scale * inverse;
  solution =
      scan_banded_recurrence(-band[1] * inverse, -band[2] * inverse, -band[3] * inverse, solution);
  float upper[3];
#pragma unroll
  for (int k = 1; k <= 3; ++k) {
    const float next = __shfl_down_sync(FULL_WARP, band[k], k);
    upper[k - 1] = lane + k < TIMEOPT_CONTROLS ? -next * inverse : 0.0f;
  }
  const int reverse = TIMEOPT_CONTROLS - 1 - lane;
  solution = __shfl_sync(FULL_WARP, solution * inverse, reverse);
#pragma unroll
  for (int k = 0; k < 3; ++k) upper[k] = __shfl_sync(FULL_WARP, upper[k], reverse);
  solution = scan_banded_recurrence(upper[0], upper[1], upper[2], solution);
  solution = __shfl_sync(FULL_WARP, solution, reverse);
  solution *= diagonal_scale;
  return __all_sync(FULL_WARP, isfinite(solution));
}

// Independent warps reduce band entries and gradients across multiple blocks.
// A separate block reduces validity; the following kernel consumes all outputs.
__global__ void fused_factor_reduce(const float *partials, const int *offsets, const int *starts,
                                    float *state, int tiles) {
  const int batch = blockIdx.y, tid = threadIdx.x, lane = tid % 32;
  const int slot = blockIdx.x * TIMEOPT_WARPS + tid / 32;
  float *st = state + batch * F_SIZE;
  if (blockIdx.x == 5 * TIMEOPT_CONTROLS / TIMEOPT_WARPS) {
    __shared__ float scratch[TIMEOPT_WARPS];
    float invalid = 0.0f;
    for (int tile = tid; tile < tiles; tile += blockDim.x)
      invalid += partials[(batch * tiles + tile) * PARTIALS_PER_TILE + ASSEMBLY_INVALID];
    invalid = fused_block_sum(invalid, scratch);
    if (tid == 0) st[F_ITERATION_VALID] = invalid == 0.0f;
    return;
  }
  if (slot < TIMEOPT_CONTROLS * 4) {
    const int row = slot / 4, column = row - slot % 4;
    float value = 0.0f;
    if (column >= 0) {
      const int begin = starts[max(0, row - 3)],
                end = starts[min(column, TIMEOPT_CONTROLS - 4) + 1];
      for (int tile = begin + lane; tile < end; tile += 32) {
        const int r = row - offsets[tile], c = column - offsets[tile];
        value += partials[(batch * tiles + tile) * PARTIALS_PER_TILE + ASSEMBLY_HESSIAN +
                          r * (r + 1) / 2 + c];
      }
    }
    value = warp_sum(value);
    if (lane == 0) st[F_FACTOR + slot] = value;
  } else {
    const int row = slot - TIMEOPT_CONTROLS * 4;
    float value = 0.0f;
    const int begin = starts[max(0, row - 3)], end = starts[min(row, TIMEOPT_CONTROLS - 4) + 1];
    for (int tile = begin + lane; tile < end; tile += 32)
      value += partials[(batch * tiles + tile) * PARTIALS_PER_TILE + ASSEMBLY_GRADIENT + row -
                        offsets[tile]];
    value = warp_sum(value);
    if (lane == 0) st[F_GRADIENT + row] = value;
  }
}

// One warp factors the reduced matrix and solves the predictor RHS.
__global__ void fused_factor(float *state, bool *numerical) {
  const int batch = blockIdx.x, lane = threadIdx.x;
  float *st = state + batch * F_SIZE;
  float band[4] = {}, diagonal_scale = 1.0f, rhs = 0.0f, solution = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    // Eliminate the fixed variables from both rows and columns before factoring.
    band[k] = lane - k < st[F_FIXED_CONTROLS] ? float(k == 0) : st[F_FACTOR + lane * 4 + k];
  }
  rhs = lane < st[F_FIXED_CONTROLS] ? 0.0f : -st[F_GRADIENT + lane];
  bool ok = factor_banded_cholesky(band, diagonal_scale);
  ok = solve_banded_cholesky(band, diagonal_scale, rhs, solution) && ok;
  const float valid = __shfl_sync(FULL_WARP, lane == 0 ? st[F_ITERATION_VALID] * ok : 0.0f, 0);
#pragma unroll
  for (int k = 0; k < 4; ++k) st[F_FACTOR + lane * 4 + k] = band[k];
  st[F_DIAGONAL_SCALE + lane] = diagonal_scale;
  st[F_PREDICTOR + lane] = valid ? solution : 0.0f;
  if (lane == 0) {
    st[F_ITERATION_VALID] = valid;
    st[F_NUMERICAL_OK] *= valid;
    numerical[batch] = st[F_NUMERICAL_OK];
  }
}

// Shared signed-pair derivatives produce both corrector RHS terms in one pass.
__global__ void fused_affine(const int2 *meta, const int *offsets, const float4 *rows,
                             const float4 *jacobian, const float *state, float *row_state,
                             float *partials, int batch_size, int tiles) {
  const int local = blockIdx.x * TIMEOPT_RECORDS + threadIdx.x / 4, batch = blockIdx.y;
  const int size = batch_size * tiles * TIMEOPT_RECORDS;
  const int row = batch * tiles * TIMEOPT_RECORDS + local;
  const float *st = state + batch * F_SIZE;
  float values[PARTIALS_PER_TILE] = {}, ap = fused_infinity(), ad = fused_infinity();
  const int kind = meta[local].x;
  const int family = threadIdx.x % 4;
  if (kind == JOINT_LIMITS || (kind == POSITIVE_CONTROL && family == 0)) {
    const float4 a4 = family == 3 ? jacobian[row] : rows[family * size + row];
    const float a[4] = {a4.x, a4.y, a4.z, a4.w};
    const float ds_positive = -dot4(a4, st + F_PREDICTOR + offsets[blockIdx.x]);
    float left = 0.0f, right = 0.0f;
#pragma unroll
    for (int sign = 0; sign < 2; ++sign) {
      if (family == 0 && sign != 0) continue;
      const int constraint = family == 0 ? 0 : 2 * family - 1 + sign;
      const int index = constraint * size + row;
      const float s = row_state[ROW_SLACK * size + index],
                  dual = row_state[ROW_DUAL * size + index];
      const float safe = fmaxf(s, 1e-12f), ds = sign ? -ds_positive : ds_positive;
      const float dl = -dual - dual / safe * ds, product = ds * dl;
      row_state[ROW_AFFINE_PRODUCT * size + index] = product;
      ap = fminf(ap, fused_boundary(s, ds));
      ad = fminf(ad, fused_boundary(dual, dl));
      left += (sign ? -product : product) / safe;
      right += (sign ? -1.0f : 1.0f) / safe;
      values[AFFINE_MOMENTS] += s * dual;
      values[AFFINE_MOMENTS + 1] += ds * dual;
      values[AFFINE_MOMENTS + 2] += s * dl;
      values[AFFINE_MOMENTS + 3] += product;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      values[AFFINE_RHS + k] += a[k] * left;
      values[AFFINE_CENTERING_RHS + k] += a[k] * right;
    }
  }
  __shared__ float scratch[PARTIALS_PER_TILE * TIMEOPT_WARPS], min_scratch[2 * TIMEOPT_WARPS];
  ap = fused_block_min(ap, min_scratch);
  ad = fused_block_min(ad, min_scratch + TIMEOPT_WARPS);
  reduce_sums(values, partials + (batch * tiles + blockIdx.x) * PARTIALS_PER_TILE, scratch);
  // All output warps must finish before thread zero replaces the two minima.
  __syncthreads();
  if (threadIdx.x == 0) {
    partials[(batch * tiles + blockIdx.x) * PARTIALS_PER_TILE + AFFINE_PRIMAL_BOUND] = ap;
    partials[(batch * tiles + blockIdx.x) * PARTIALS_PER_TILE + AFFINE_DUAL_BOUND] = ad;
  }
}

// RHS components and centering statistics have independent reductions.
__global__ void fused_correct_reduce(const float *partials, const int *offsets, const int *starts,
                                     float *terms, float *state, int tiles, int constraints) {
  const int batch = blockIdx.y, tid = threadIdx.x, lane = tid % 32;
  float *st = state + batch * F_SIZE;
  if (blockIdx.x < TIMEOPT_CONTROLS / TIMEOPT_WARPS) {
    const int row = blockIdx.x * TIMEOPT_WARPS + tid / 32;
    float left = 0.0f, right = 0.0f;
    const int begin = starts[max(0, row - 3)], end = starts[min(row, TIMEOPT_CONTROLS - 4) + 1];
    for (int tile = begin + lane; tile < end; tile += 32) {
      const int k = row - offsets[tile];
      left += partials[(batch * tiles + tile) * PARTIALS_PER_TILE + AFFINE_RHS + k];
      right += partials[(batch * tiles + tile) * PARTIALS_PER_TILE + AFFINE_CENTERING_RHS + k];
    }
    left = warp_sum(left);
    right = warp_sum(right);
    if (lane == 0) {
      terms[batch * 2 * TIMEOPT_CONTROLS + row] = left;
      terms[(batch * 2 + 1) * TIMEOPT_CONTROLS + row] = right;
    }
    return;
  }
  __shared__ float scratch[6 * TIMEOPT_WARPS], moments[4];
  float sum[4] = {}, ap = fused_infinity(), ad = fused_infinity();
  for (int tile = tid; tile < tiles; tile += blockDim.x) {
    const float *p = partials + (batch * tiles + tile) * PARTIALS_PER_TILE;
    ap = fminf(ap, p[AFFINE_PRIMAL_BOUND]);
    ad = fminf(ad, p[AFFINE_DUAL_BOUND]);
#pragma unroll
    for (int k = 0; k < 4; ++k) sum[k] += p[AFFINE_MOMENTS + k];
  }
  ap = fused_block_min(ap, scratch);
  ad = fused_block_min(ad, scratch + TIMEOPT_WARPS);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float total = fused_block_sum(sum[k], scratch + (k + 2) * TIMEOPT_WARPS);
    if (tid == 0) moments[k] = total;
  }
  if (tid == 0) {
    ap = fminf(ap, 1.0f);
    ad = fminf(ad, 1.0f);
    const float gap = moments[0] / constraints;
    const float gap_aff =
        (moments[0] + ap * moments[1] + ad * moments[2] + ap * ad * moments[3]) / constraints;
    const float ratio = fminf(fmaxf(gap_aff / fmaxf(gap, 1e-30f), 0.0f), 1.0f);
    st[F_GAP] = gap;
    st[F_CENTERING] = ratio * ratio * ratio * gap;
  }
}

// One warp combines the reduced RHS and reuses the predictor's factor.
__global__ void fused_correct(const float *terms, float *state, bool *numerical) {
  const int batch = blockIdx.x, lane = threadIdx.x;
  float *st = state + batch * F_SIZE;
  float band[4] = {1.0f, 0.0f, 0.0f, 0.0f}, diagonal_scale = 1.0f, rhs = 0.0f, solution = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) band[k] = st[F_FACTOR + lane * 4 + k];
  const float left = terms[batch * 2 * TIMEOPT_CONTROLS + lane];
  const float right = terms[(batch * 2 + 1) * TIMEOPT_CONTROLS + lane];
  rhs = -st[F_GRADIENT + lane] + left - st[F_CENTERING] * right;
  if (lane < st[F_FIXED_CONTROLS]) rhs = 0.0f;
  diagonal_scale = st[F_DIAGONAL_SCALE + lane];
  const bool ok = solve_banded_cholesky(band, diagonal_scale, rhs, solution);
  const float valid = __shfl_sync(FULL_WARP, lane == 0 ? st[F_ITERATION_VALID] * ok : 0.0f, 0);
  st[F_CORRECTOR + lane] = valid ? solution : 0.0f;
  if (lane == 0) {
    st[F_ITERATION_VALID] = valid;
    st[F_NUMERICAL_OK] *= valid;
    numerical[batch] = st[F_NUMERICAL_OK];
  }
}

__global__ void fused_bounds(const int2 *meta, const int *offsets, const float4 *rows,
                             const float4 *jacobian, const float *state, float *row_state,
                             float *bounds, int batch_size, int tiles) {
  const int local = blockIdx.x * TIMEOPT_RECORDS + threadIdx.x / 4, batch = blockIdx.y;
  const int size = batch_size * tiles * TIMEOPT_RECORDS;
  const int row = batch * tiles * TIMEOPT_RECORDS + local;
  const float *st = state + batch * F_SIZE;
  float ap = fused_infinity(), ad = fused_infinity();
  const int kind = meta[local].x;
  const int family = threadIdx.x % 4;
  if (kind == JOINT_LIMITS || (kind == POSITIVE_CONTROL && family == 0)) {
    const float4 a = family == 3 ? jacobian[row] : rows[family * size + row];
    const float ds_positive = -dot4(a, st + F_CORRECTOR + offsets[blockIdx.x]);
#pragma unroll
    for (int sign = 0; sign < 2; ++sign) {
      if (family == 0 && sign != 0) continue;
      const int constraint = family == 0 ? 0 : 2 * family - 1 + sign;
      const int index = constraint * size + row;
      const float s = row_state[ROW_SLACK * size + index],
                  dual = row_state[ROW_DUAL * size + index];
      const float ds = sign ? -ds_positive : ds_positive;
      const float correction = row_state[ROW_AFFINE_PRODUCT * size + index] - st[F_CENTERING];
      const float dl = -dual - (correction + dual * ds) / fmaxf(s, 1e-12f);
      row_state[ROW_DUAL_DIRECTION * size + index] = dl;
      ap = fminf(ap, fused_boundary(s, ds));
      ad = fminf(ad, fused_boundary(dual, dl));
    }
  }
  __shared__ float scratch[2 * TIMEOPT_WARPS];
  ap = fused_block_min(ap, scratch);
  ad = fused_block_min(ad, scratch + TIMEOPT_WARPS);
  if (threadIdx.x == 0) {
    bounds[(batch * tiles + blockIdx.x) * 2] = ap;
    bounds[(batch * tiles + blockIdx.x) * 2 + 1] = ad;
  }
}

__global__ void fused_search_bounds(const float *step_bounds, float *state, int tiles) {
  const int batch = blockIdx.x, tid = threadIdx.x;
  float *st = state + batch * F_SIZE;
  float ap = fused_infinity(), ad = fused_infinity();
  for (int tile = tid; tile < tiles; tile += blockDim.x) {
    ap = fminf(ap, step_bounds[(batch * tiles + tile) * 2]);
    ad = fminf(ad, step_bounds[(batch * tiles + tile) * 2 + 1]);
  }
  __shared__ float scratch[2 * TIMEOPT_WARPS];
  ap = fminf(0.995f * fused_block_min(ap, scratch), 1.0f);
  ad = fminf(0.995f * fused_block_min(ad, scratch + TIMEOPT_WARPS), 1.0f);
  if (tid == 0) {
    st[F_PRIMAL_STEP] = ap;
    st[F_DUAL_STEP] = ad;
  }
}

// Candidate groups share the once-per-path step-bound reduction.
template <int Candidates>
__global__ void fused_search(const int2 *meta, const int *offsets, const float4 *basis,
                             const float4 *rows, const float *bounds, const float *state,
                             int *masks, int points, int dof, int batch_size, int tiles) {
  static_assert(Candidates >= 1 && Candidates <= LINE_SEARCH_STEPS &&
                LINE_SEARCH_STEPS % Candidates == 0);
  constexpr int GROUPS = LINE_SEARCH_STEPS / Candidates;
  const int tile = blockIdx.x / GROUPS, first_candidate = (blockIdx.x % GROUPS) * Candidates;
  const int tid = threadIdx.x, local = tile * TIMEOPT_RECORDS + tid / 4, batch = blockIdx.y;
  const int size = batch_size * tiles * TIMEOPT_RECORDS;
  const int row = batch * tiles * TIMEOPT_RECORDS + local, first = offsets[tile];
  const int kind = meta[local].x, family = tid % 4, record = tid / 4;
  const float *st = state + batch * F_SIZE;
  __shared__ int warp_masks[TIMEOPT_WARPS];
  __shared__ float roots[Candidates][TIMEOPT_RECORDS], candidates[Candidates][4];
  if (tid < Candidates * 4) {
    const int candidate = tid / 4, k = tid % 4;
    candidates[candidate][k] =
        __fadd_rn(st[F_CONTROLS + first + k],
                  __fmul_rn(st[F_PRIMAL_STEP] / float(1 << (first_candidate + candidate)),
                            st[F_CORRECTOR + first + k]));
  }
  __syncthreads();
  const int leader = max(0, record - meta[local].y % dof);
  if (kind == JOINT_LIMITS && family == 0 && record == leader) {
    const float4 n = basis[meta[local].y / dof];
#pragma unroll
    for (int candidate = 0; candidate < Candidates; ++candidate)
      roots[candidate][record] = sqrtf(st[F_SPEED_SCALE] * dot4(n, candidates[candidate]));
  }
  __syncthreads();
  int mask = st[F_ITERATION_VALID] ? 0 : ((1 << Candidates) - 1) << first_candidate;
  if ((kind == JOINT_LIMITS || (kind == POSITIVE_CONTROL && family == 0)) && !mask) {
    const float4 a = rows[family * size + row];
#pragma unroll
    for (int candidate = 0; candidate < Candidates; ++candidate) {
      float value = dot4(a, candidates[candidate]);
      if (family == 3) value *= roots[candidate][leader];
      bool valid = isfinite(value);
      if (family == 0)
        valid = valid && value < (kind == POSITIVE_CONTROL ? 0.0f : 1.0f);
      else if (family == 2)
        valid = valid && value < bounds[row] && -value < bounds[size + row];
      else
        valid = valid && fabsf(value) < 1.0f;
      if (!valid) mask |= 1 << (first_candidate + candidate);
    }
  }
  mask = fused_block_mask(mask, warp_masks);
  if (tid == 0) masks[batch * tiles * GROUPS + blockIdx.x] = mask;
}

__global__ void fused_accept(const int *masks, const float4 *basis, const int *point_first,
                             float *state, float *output, int mask_count, int duration_points) {
  const int batch = blockIdx.x, tid = threadIdx.x;
  float *st = state + batch * F_SIZE;
  __shared__ int warp_masks[TIMEOPT_WARPS], accepted;
  __shared__ float scratch[TIMEOPT_WARPS], cost;
  int mask = 0;
  for (int i = tid; i < mask_count; i += blockDim.x) mask |= masks[batch * mask_count + i];
  mask = fused_block_mask(mask, warp_masks);
  if (tid == 0) {
    accepted = __ffs((~mask) & ((1 << LINE_SEARCH_STEPS) - 1)) - 1;
    st[F_ACCEPTED] = accepted >= 0;
  }
  __syncthreads();
  if (tid < TIMEOPT_CONTROLS && accepted >= 0)
    st[F_CONTROLS + tid] =
        __fadd_rn(st[F_CONTROLS + tid],
                  __fmul_rn(st[F_PRIMAL_STEP] / float(1 << accepted), st[F_CORRECTOR + tid]));
  __syncthreads();
  float value = 0.0f;
  const float weight = __frcp_rn(float(duration_points));
  for (int q = tid; q < duration_points; q += blockDim.x)
    value += weight * rsqrtf(fmaxf(dot4(basis[q], st + F_CONTROLS + point_first[q]), 1e-20f));
  value = fused_block_sum(value, scratch);
  if (tid == 0) cost = value;
  __syncthreads();
  const bool improved = st[F_ITERATION_VALID] && cost < st[F_BEST_COST];
  if (tid < TIMEOPT_CONTROLS) {
    if (improved) st[F_BEST_CONTROLS + tid] = st[F_CONTROLS + tid];
    output[batch * TIMEOPT_CONTROLS + tid] = st[F_BEST_CONTROLS + tid] * st[F_SPEED_SCALE];
  }
  __syncthreads();
  if (tid == 0 && improved) st[F_BEST_COST] = cost;
}

}  // namespace curobo::trajectory::timeopt
