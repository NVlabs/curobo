// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace curobo::trajectory::seed_fitter {

constexpr int THREADS_PER_BLOCK = 256;

__global__ void resample_dense_kernel(
    const float* target_path, const float* arc_progress, float* arc_target,
    int seeds, int steps, int dof) {
  extern __shared__ float arc_workspace[];
  float* lengths = arc_workspace;
  float* cumulative = lengths + steps;
  const int seed = static_cast<int>(blockIdx.x);
  const int seed_offset = seed * steps * dof;
  for (int segment = static_cast<int>(threadIdx.x); segment < steps - 1;
       segment += static_cast<int>(blockDim.x)) {
    float length_squared = 0.0F;
    for (int joint = 0; joint < dof; ++joint) {
      const float delta =
          target_path[seed_offset + (segment + 1) * dof + joint] -
          target_path[seed_offset + segment * dof + joint];
      length_squared += delta * delta;
    }
    lengths[segment] = sqrtf(length_squared);
  }
  __syncthreads();
  // Preserve the original summation order; compute geometry once per path.
  if (threadIdx.x == 0) {
    cumulative[0] = 0.0F;
    for (int segment = 0; segment < steps - 1; ++segment) {
      cumulative[segment + 1] = cumulative[segment] + lengths[segment];
    }
  }
  __syncthreads();
  const float total_length = cumulative[steps - 1];
  for (int output_step = static_cast<int>(threadIdx.x); output_step < steps;
       output_step += static_cast<int>(blockDim.x)) {
    const float query_distance = arc_progress[output_step] * total_length;
    int low = 0;
    int high = steps - 2;
    while (low < high) {
      const int mid = (low + high) / 2;
      if (query_distance <= cumulative[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    const int selected_segment = low;
    const float selected_start = cumulative[low];
    const float selected_length = lengths[low];

    float alpha = 0.0F;
    if (total_length > 1.0E-8F && selected_length > 1.0E-8F) {
      alpha = fminf(
          fmaxf((query_distance - selected_start) / selected_length, 0.0F),
          1.0F);
    }
    for (int joint = 0; joint < dof; ++joint) {
      const float start =
          target_path[seed_offset + selected_segment * dof + joint];
      const float end =
          target_path[seed_offset + (selected_segment + 1) * dof + joint];
      arc_target[seed_offset + output_step * dof + joint] =
          start + alpha * (end - start);
    }
  }
}

__global__ void initialize_fit_kernel(
    const float* arc_target, const float* boundary_position,
    const float* start_position, const float* start_velocity,
    const float* start_acceleration, const float* start_jerk,
    const float* goal_position, const float* fit_matrix,
    const float* start_position_coefficient,
    const float* start_velocity_coefficient,
    const float* start_acceleration_coefficient,
    const float* start_jerk_coefficient,
    const float* goal_position_coefficient, float* position_residual,
    float* knots, int seeds, int steps, int dof, int knot_count) {
  const int free_knots = knot_count - 1;
  const int variable_count = seeds * knot_count * dof;
  for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       index < variable_count;
       index += static_cast<int>(blockDim.x * gridDim.x)) {
    const int seed = index / (knot_count * dof);
    const int variable = index - seed * knot_count * dof;
    const int knot = variable / dof;
    const int joint = variable - knot * dof;
    const int state_index = seed * dof + joint;
    if (knot < free_knots) {
      float value =
          start_position_coefficient[knot] * start_position[state_index] +
          start_velocity_coefficient[knot] * start_velocity[state_index] +
          start_acceleration_coefficient[knot] *
              start_acceleration[state_index] +
          start_jerk_coefficient[knot] * start_jerk[state_index] +
          goal_position_coefficient[knot] * goal_position[state_index];
      for (int step = 0; step < steps; ++step) {
        value += fit_matrix[knot * steps + step] *
                 arc_target[(seed * steps + step) * dof + joint];
      }
      knots[index] = value;
    } else {
      knots[index] = goal_position[state_index];
    }
    if (knot == 0) {
      for (int step = 0; step < steps; ++step) {
        const int output_index = (seed * steps + step) * dof + joint;
        position_residual[output_index] =
            arc_target[output_index] - boundary_position[output_index];
      }
    }
  }
}

__global__ void required_dt_kernel(
    const float* velocity, const float* acceleration, const float* jerk,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, int steps, int dof, float trajectory_dt,
    float dt_epsilon, float* required_dt) {
  __shared__ float velocity_scores[THREADS_PER_BLOCK];
  __shared__ float acceleration_scores[THREADS_PER_BLOCK];
  __shared__ float jerk_scores[THREADS_PER_BLOCK];
  const int seed = static_cast<int>(blockIdx.x);
  const int state_values = steps * dof;
  const int seed_offset = seed * state_values;
  float velocity_score = 0.0F;
  float acceleration_score = 0.0F;
  float jerk_score = 0.0F;
  for (int variable = static_cast<int>(threadIdx.x); variable < state_values;
       variable += static_cast<int>(blockDim.x)) {
    const int joint = variable % dof;
    const int index = seed_offset + variable;
    velocity_score =
        fmaxf(velocity_score, fabsf(velocity[index]) / max_velocity[joint]);
    acceleration_score = fmaxf(
        acceleration_score,
        fabsf(acceleration[index]) / max_acceleration[joint]);
    jerk_score = fmaxf(jerk_score, fabsf(jerk[index]) / max_jerk[joint]);
  }
  velocity_scores[threadIdx.x] = velocity_score;
  acceleration_scores[threadIdx.x] = acceleration_score;
  jerk_scores[threadIdx.x] = jerk_score;
  __syncthreads();

  for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
    if (static_cast<int>(threadIdx.x) < stride) {
      velocity_scores[threadIdx.x] = fmaxf(
          velocity_scores[threadIdx.x], velocity_scores[threadIdx.x + stride]);
      acceleration_scores[threadIdx.x] =
          fmaxf(acceleration_scores[threadIdx.x],
                acceleration_scores[threadIdx.x + stride]);
      jerk_scores[threadIdx.x] =
          fmaxf(jerk_scores[threadIdx.x], jerk_scores[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float scale = fmaxf(
        velocity_scores[0],
        fmaxf(sqrtf(acceleration_scores[0]),
              powf(jerk_scores[0], 1.0F / 3.0F)));
    required_dt[seed] =
        scale * trajectory_dt * (1.0F + dt_epsilon);
  }
}

__global__ void derivative_scale_kernel(
    const float* current_velocity, const float* current_acceleration,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, const float* basis_velocity,
    const float* basis_acceleration, const float* basis_jerk, int seeds,
    int steps, int dof, int free_knots, float path_normal_trace,
    float timing_regularization, float irls_epsilon,
    float* velocity_trace_workspace, float* acceleration_trace_workspace,
    float* jerk_trace_workspace, float* derivative_scale) {
  __shared__ float trace_sums[THREADS_PER_BLOCK];
  float local_trace = 0.0F;
  const int seed = static_cast<int>(blockIdx.x);
  const int state_values = steps * dof;
  for (int variable = static_cast<int>(threadIdx.x); variable < state_values;
       variable += static_cast<int>(blockDim.x)) {
    const int index = seed * state_values + variable;
    const int step = variable / dof;
    const int joint = variable - step * dof;
    float velocity_basis_squared = 0.0F;
    float acceleration_basis_squared = 0.0F;
    float jerk_basis_squared = 0.0F;
    for (int knot = 0; knot < free_knots; ++knot) {
      const int basis_index = step * free_knots + knot;
      const float velocity_basis = basis_velocity[basis_index];
      const float acceleration_basis = basis_acceleration[basis_index];
      const float jerk_basis = basis_jerk[basis_index];
      velocity_basis_squared += velocity_basis * velocity_basis;
      acceleration_basis_squared +=
          acceleration_basis * acceleration_basis;
      jerk_basis_squared += jerk_basis * jerk_basis;
    }
    const float inverse_velocity_squared =
        1.0F / (max_velocity[joint] * max_velocity[joint]);
    const float inverse_acceleration_squared =
        1.0F / (max_acceleration[joint] * max_acceleration[joint]);
    const float inverse_jerk_squared =
        1.0F / (max_jerk[joint] * max_jerk[joint]);
    const float velocity_ratio = fmaxf(
        fabsf(current_velocity[index]) / max_velocity[joint], irls_epsilon);
    const float acceleration_ratio =
        fmaxf(fabsf(current_acceleration[index]) / max_acceleration[joint],
              irls_epsilon);
    const float velocity_ratio_squared = velocity_ratio * velocity_ratio;
    const float velocity_weight =
        3.0F * velocity_ratio_squared * velocity_ratio_squared *
        inverse_velocity_squared;
    const float acceleration_weight =
        1.5F * acceleration_ratio * inverse_acceleration_squared;
    velocity_trace_workspace[index] =
        velocity_weight * velocity_basis_squared;
    acceleration_trace_workspace[index] =
        acceleration_weight * acceleration_basis_squared;
    jerk_trace_workspace[index] = inverse_jerk_squared * jerk_basis_squared;
    local_trace += velocity_weight * velocity_basis_squared;
    local_trace += acceleration_weight * acceleration_basis_squared;
    local_trace += inverse_jerk_squared * jerk_basis_squared;
  }
  trace_sums[threadIdx.x] = local_trace;
  __syncthreads();
  for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
    if (static_cast<int>(threadIdx.x) < stride) {
      trace_sums[threadIdx.x] += trace_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float trace = trace_sums[0] / static_cast<float>(dof);
    derivative_scale[seed] = timing_regularization * path_normal_trace / fmaxf(trace, 1.0E-12F);
  }
}

__global__ void solve_irls_kernel(
    const float* position_residual, const float* boundary_velocity,
    const float* boundary_acceleration, const float* boundary_jerk,
    const float* current_velocity, const float* current_acceleration,
    const float* current_knots, const float* goal_position,
    const float* max_velocity, const float* max_acceleration,
    const float* max_jerk, const float* basis_position,
    const float* basis_velocity, const float* basis_acceleration,
    const float* basis_jerk, const float* path_normal,
    const float* derivative_scale, int steps, int dof, int knots,
    float regularization, float trust_region_regularization,
    float path_diagonal_scale, float irls_epsilon, float* candidate_knots) {
  extern __shared__ float workspace[];
  const int free_knots = knots - 1;
  float* velocity_weight = workspace;
  float* acceleration_weight = velocity_weight + steps;
  float* normal = acceleration_weight + steps;
  float* rhs = normal + free_knots * free_knots;
  const int lane = static_cast<int>(blockIdx.x);
  const int seed = lane / dof;
  const int joint = lane - seed * dof;
  const int state_base = seed * steps * dof;
  const float inverse_velocity_squared =
      1.0F / (max_velocity[joint] * max_velocity[joint]);
  const float inverse_acceleration_squared =
      1.0F / (max_acceleration[joint] * max_acceleration[joint]);
  const float inverse_jerk_squared =
      1.0F / (max_jerk[joint] * max_jerk[joint]);

  for (int step = static_cast<int>(threadIdx.x); step < steps;
       step += static_cast<int>(blockDim.x)) {
    const int state_index = state_base + step * dof + joint;
    const float velocity_ratio = fmaxf(
        fabsf(current_velocity[state_index]) / max_velocity[joint],
        irls_epsilon);
    const float acceleration_ratio = fmaxf(
        fabsf(current_acceleration[state_index]) / max_acceleration[joint],
        irls_epsilon);
    const float velocity_ratio_squared = velocity_ratio * velocity_ratio;
    velocity_weight[step] =
        3.0F * velocity_ratio_squared * velocity_ratio_squared *
        inverse_velocity_squared;
    acceleration_weight[step] =
        1.5F * acceleration_ratio * inverse_acceleration_squared;
  }
  __syncthreads();

  const int matrix_values = free_knots * free_knots;
  for (int matrix_index = static_cast<int>(threadIdx.x);
       matrix_index < matrix_values;
       matrix_index += static_cast<int>(blockDim.x)) {
    const int row = matrix_index / free_knots;
    const int column = matrix_index - row * free_knots;
    if (column <= row) {
      float derivative_normal = 0.0F;
      for (int step = 0; step < steps; ++step) {
        const int row_index = step * free_knots + row;
        const int column_index = step * free_knots + column;
        derivative_normal +=
            basis_velocity[row_index] * velocity_weight[step] *
                basis_velocity[column_index] +
            basis_acceleration[row_index] * acceleration_weight[step] *
                basis_acceleration[column_index] +
            basis_jerk[row_index] * inverse_jerk_squared *
                basis_jerk[column_index];
      }
      float value = path_normal[matrix_index] + derivative_scale[seed] * derivative_normal;
      if (row == column) {
        value += (regularization + trust_region_regularization) *
                 path_diagonal_scale;
      }
      normal[matrix_index] = value;
    }
  }

  for (int row = static_cast<int>(threadIdx.x); row < free_knots;
       row += static_cast<int>(blockDim.x)) {
    float path_rhs = 0.0F;
    float derivative_rhs = 0.0F;
    for (int step = 0; step < steps; ++step) {
      const int state_index = state_base + step * dof + joint;
      const int basis_index = step * free_knots + row;
      path_rhs += position_residual[state_index] * basis_position[basis_index];
      derivative_rhs -=
          basis_velocity[basis_index] * velocity_weight[step] *
              boundary_velocity[state_index] +
          basis_acceleration[basis_index] * acceleration_weight[step] *
              boundary_acceleration[state_index] +
          basis_jerk[basis_index] * inverse_jerk_squared *
              boundary_jerk[state_index];
    }
    const int knot_index = (seed * knots + row) * dof + joint;
    rhs[row] = path_rhs + derivative_scale[seed] * derivative_rhs +
               trust_region_regularization * path_diagonal_scale * current_knots[knot_index];
  }
  __syncthreads();

  // Each column depends on previous columns; its off-diagonal rows are independent.
  // Keep the factorization within one warp so synchronization does not stall the block.
  if (threadIdx.x < 32) {
    for (int column = 0; column < free_knots; ++column) {
      if (threadIdx.x == 0) {
        float value = normal[column * free_knots + column];
        for (int inner = 0; inner < column; ++inner) {
          const float entry = normal[column * free_knots + inner];
          value -= entry * entry;
        }
        normal[column * free_knots + column] = sqrtf(fmaxf(value, 1.0E-20F));
      }
      __syncwarp();
      for (int row = column + 1 + static_cast<int>(threadIdx.x);
           row < free_knots; row += 32) {
        float value = normal[row * free_knots + column];
        for (int inner = 0; inner < column; ++inner) {
          value -= normal[row * free_knots + inner] *
                   normal[column * free_knots + inner];
        }
        normal[row * free_knots + column] =
            value / normal[column * free_knots + column];
      }
      __syncwarp();
    }
  }
  if (threadIdx.x == 0) {
    for (int row = 0; row < free_knots; ++row) {
      float value = rhs[row];
      for (int column = 0; column < row; ++column) {
        value -= normal[row * free_knots + column] * rhs[column];
      }
      rhs[row] = value / normal[row * free_knots + row];
    }
    for (int reverse_row = 0; reverse_row < free_knots; ++reverse_row) {
      const int row = free_knots - 1 - reverse_row;
      float value = rhs[row];
      for (int column = row + 1; column < free_knots; ++column) {
        const int candidate_index =
            (seed * knots + column) * dof + joint;
        value -= normal[column * free_knots + row] *
                 candidate_knots[candidate_index];
      }
      const int candidate_index = (seed * knots + row) * dof + joint;
      candidate_knots[candidate_index] =
          value / normal[row * free_knots + row];
    }
    candidate_knots[(seed * knots + free_knots) * dof + joint] =
        goal_position[seed * dof + joint];
  }
}

__global__ void accept_candidate_kernel(
    const float* candidate_knots, const float* candidate_velocity,
    const float* candidate_acceleration, const float* candidate_jerk,
    const float* candidate_dt, float* current_knots,
    float* current_velocity, float* current_acceleration,
    float* current_jerk, const float* current_dt, int seeds, int steps, int dof,
    int knot_count) {
  const int state_values = steps * dof;
  const int total_values = seeds * state_values;
  for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       index < total_values;
       index += static_cast<int>(blockDim.x * gridDim.x)) {
    const int seed = index / state_values;
    if (candidate_dt[seed] >= current_dt[seed]) {
      continue;
    }
    current_velocity[index] = candidate_velocity[index];
    current_acceleration[index] = candidate_acceleration[index];
    current_jerk[index] = candidate_jerk[index];
    const int variable = index - seed * state_values;
    if (variable < knot_count * dof) {
      current_knots[seed * knot_count * dof + variable] =
          candidate_knots[seed * knot_count * dof + variable];
    }
  }
}

__global__ void accept_candidate_dt_kernel(
    const float* candidate_dt, float* current_dt, int seeds) {
  // Launch after the state-copy kernel so every copy thread evaluates the
  // acceptance predicate against the same timestep.
  for (int seed = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
       seed < seeds;
       seed += static_cast<int>(blockDim.x * gridDim.x)) {
    if (candidate_dt[seed] < current_dt[seed]) {
      current_dt[seed] = candidate_dt[seed];
    }
  }
}

}  // namespace curobo::trajectory::seed_fitter
