/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

namespace curobo {
namespace optimization {
namespace pose_refiner {

constexpr int POSE_REFINER_SUCCESS = 0;
constexpr int POSE_REFINER_TOO_FEW_DEPTH_PIXELS = 1;
constexpr int POSE_REFINER_TOO_FEW_CORRESPONDENCES = 2;
constexpr int POSE_REFINER_LINEAR_SOLVE_FAILED = 3;

constexpr int RESULT_FINAL_ERROR = 0;
constexpr int RESULT_FINAL_LAMBDA = 1;
constexpr int RESULT_ITERATIONS = 0;
constexpr int RESULT_VALID_COUNT = 1;
constexpr int RESULT_ACCEPTED_STEPS = 2;
constexpr int RESULT_STATUS = 3;

__device__ __forceinline__ float clamp_float(float value, float minimum,
                                              float maximum) {
  return fminf(fmaxf(value, minimum), maximum);
}

__device__ __forceinline__ float positive_infinity() {
  return __int_as_float(0x7f800000);
}

__device__ __forceinline__ void copy_float_array(const float* source,
                                                  float* destination,
                                                  int count) {
  for (int index = 0; index < count; ++index) {
    destination[index] = source[index];
  }
}

__device__ __forceinline__ void quaternion_multiply_wxyz(const float* first,
                                                          const float* second,
                                                          float* output) {
  const float aw = first[0];
  const float ax = first[1];
  const float ay = first[2];
  const float az = first[3];
  const float bw = second[0];
  const float bx = second[1];
  const float by = second[2];
  const float bz = second[3];
  output[0] = aw * bw - ax * bx - ay * by - az * bz;
  output[1] = aw * bx + bw * ax + ay * bz - az * by;
  output[2] = aw * by + bw * ay + az * bx - ax * bz;
  output[3] = aw * bz + bw * az + ax * by - ay * bx;
}

__device__ __forceinline__ void normalize_quaternion_wxyz(float* quaternion) {
  const float norm = sqrtf(quaternion[0] * quaternion[0] +
                           quaternion[1] * quaternion[1] +
                           quaternion[2] * quaternion[2] +
                           quaternion[3] * quaternion[3]);
  if (isfinite(norm) && norm > 1.0e-7f) {
    const float inverse_norm = 1.0f / norm;
    quaternion[0] *= inverse_norm;
    quaternion[1] *= inverse_norm;
    quaternion[2] *= inverse_norm;
    quaternion[3] *= inverse_norm;
    return;
  }
  quaternion[0] = 1.0f;
  quaternion[1] = 0.0f;
  quaternion[2] = 0.0f;
  quaternion[3] = 0.0f;
}

__device__ __forceinline__ void rotate_by_quaternion_wxyz(
    const float* quaternion, const float* vector, float* output) {
  const float w = quaternion[0];
  const float qx = quaternion[1];
  const float qy = quaternion[2];
  const float qz = quaternion[3];
  const float uvx = qy * vector[2] - qz * vector[1];
  const float uvy = qz * vector[0] - qx * vector[2];
  const float uvz = qx * vector[1] - qy * vector[0];
  const float uuvx = qy * uvz - qz * uvy;
  const float uuvy = qz * uvx - qx * uvz;
  const float uuvz = qx * uvy - qy * uvx;
  output[0] = vector[0] + 2.0f * (w * uvx + uuvx);
  output[1] = vector[1] + 2.0f * (w * uvy + uuvy);
  output[2] = vector[2] + 2.0f * (w * uvz + uuvz);
}

__device__ __forceinline__ void euler_xyz_extrinsic_to_quaternion_wxyz(
    const float* euler, float* quaternion) {
  const float rx = 0.5f * euler[0];
  const float ry = 0.5f * euler[1];
  const float rz = 0.5f * euler[2];
  const float cx = cosf(rx);
  const float sx = sinf(rx);
  const float cy = cosf(ry);
  const float sy = sinf(ry);
  const float cz = cosf(rz);
  const float sz = sinf(rz);
  quaternion[0] = cx * cy * cz + sx * sy * sz;
  quaternion[1] = sx * cy * cz - cx * sy * sz;
  quaternion[2] = cx * sy * cz + sx * cy * sz;
  quaternion[3] = cx * cy * sz - sx * sy * cz;
}

__device__ __forceinline__ bool solve_lm_cholesky_6(
    const float* jtj, const float* jtr, float lambda_damping, float* delta) {
  float system[36];
  float cholesky[36];
  float intermediate[6];
  for (int index = 0; index < 36; ++index) {
    system[index] = jtj[index];
    cholesky[index] = 0.0f;
  }
  for (int index = 0; index < 6; ++index) {
    system[index * 6 + index] += lambda_damping;
  }

  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column <= row; ++column) {
      float sum = system[row * 6 + column];
      for (int index = 0; index < column; ++index) {
        sum -= cholesky[row * 6 + index] * cholesky[column * 6 + index];
      }
      if (row == column) {
        if (!isfinite(sum) || sum <= 1.0e-12f) {
          return false;
        }
        cholesky[row * 6 + column] = sqrtf(sum);
      } else {
        const float diagonal = cholesky[column * 6 + column];
        if (!isfinite(diagonal) || fabsf(diagonal) <= 1.0e-12f) {
          return false;
        }
        cholesky[row * 6 + column] = sum / diagonal;
      }
    }
  }

  for (int row = 0; row < 6; ++row) {
    float sum = -jtr[row];
    for (int index = 0; index < row; ++index) {
      sum -= cholesky[row * 6 + index] * intermediate[index];
    }
    intermediate[row] = sum / cholesky[row * 6 + row];
  }
  for (int row = 5; row >= 0; --row) {
    float sum = intermediate[row];
    for (int index = row + 1; index < 6; ++index) {
      sum -= cholesky[index * 6 + row] * delta[index];
    }
    delta[row] = sum / cholesky[row * 6 + row];
    if (!isfinite(delta[row])) {
      return false;
    }
  }
  return true;
}

__global__ void count_valid_depth_kernel(const float* depth, int pixel_count,
                                         float depth_minimum,
                                         float depth_maximum,
                                         int* valid_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) {
    return;
  }
  const float value = depth[index];
  if (isfinite(value) && value > depth_minimum && value < depth_maximum) {
    atomicAdd(valid_count, 1);
  }
}

__global__ void initialize_state_kernel(
    const float* initial_position, const float* initial_quaternion,
    const float* initial_jtj, const float* initial_jtr,
    const float* initial_sum_sq, const int* initial_valid_count,
    const int* valid_depth_count, int minimum_valid_depth_pixels,
    int minimum_valid_correspondences, float lambda_initial,
    float* best_position, float* best_quaternion, float* best_jtj,
    float* best_jtr, float* best_sum_sq, float* best_error,
    int* best_valid_count, float* lambda_damping, float* result_float,
    int* result_int) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  copy_float_array(initial_position, best_position, 3);
  copy_float_array(initial_quaternion, best_quaternion, 4);
  normalize_quaternion_wxyz(best_quaternion);
  copy_float_array(initial_jtj, best_jtj, 36);
  copy_float_array(initial_jtr, best_jtr, 6);
  best_sum_sq[0] = initial_sum_sq[0];
  best_valid_count[0] = initial_valid_count[0];
  best_error[0] = initial_valid_count[0] > 0
                      ? sqrtf(initial_sum_sq[0] /
                              (static_cast<float>(initial_valid_count[0]) + 1.0e-8f))
                      : positive_infinity();
  lambda_damping[0] = lambda_initial;

  result_float[RESULT_FINAL_ERROR] = best_error[0];
  result_float[RESULT_FINAL_LAMBDA] = lambda_initial;
  result_int[RESULT_ITERATIONS] = 0;
  result_int[RESULT_VALID_COUNT] = initial_valid_count[0];
  result_int[RESULT_ACCEPTED_STEPS] = 0;
  result_int[RESULT_STATUS] = POSE_REFINER_SUCCESS;
  if (valid_depth_count[0] < minimum_valid_depth_pixels) {
    result_int[RESULT_STATUS] = POSE_REFINER_TOO_FEW_DEPTH_PIXELS;
  } else if (initial_valid_count[0] <= minimum_valid_correspondences) {
    result_int[RESULT_STATUS] = POSE_REFINER_TOO_FEW_CORRESPONDENCES;
  }
}

__global__ void solve_candidate_kernel(
    const float* best_position, const float* best_quaternion,
    const float* best_jtj, const float* best_jtr,
    const float* lambda_damping, float* candidate_position,
    float* candidate_quaternion, float* predicted_reduction,
    int* result_int) {
  if (threadIdx.x != 0 || blockIdx.x != 0 ||
      result_int[RESULT_STATUS] != POSE_REFINER_SUCCESS) {
    return;
  }

  float delta[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (!solve_lm_cholesky_6(best_jtj, best_jtr, lambda_damping[0], delta)) {
    result_int[RESULT_STATUS] = POSE_REFINER_LINEAR_SOLVE_FAILED;
    return;
  }

  float jtj_delta[6];
  for (int row = 0; row < 6; ++row) {
    float sum = 0.0f;
    for (int column = 0; column < 6; ++column) {
      sum += best_jtj[row * 6 + column] * delta[column];
    }
    jtj_delta[row] = sum;
  }
  float linear_term = 0.0f;
  float quadratic_term = 0.0f;
  for (int index = 0; index < 6; ++index) {
    linear_term += delta[index] * best_jtr[index];
    quadratic_term += delta[index] * jtj_delta[index];
  }
  predicted_reduction[0] = -linear_term - 0.5f * quadratic_term;

  float delta_quaternion[4];
  euler_xyz_extrinsic_to_quaternion_wxyz(delta + 3, delta_quaternion);
  float rotated_position[3];
  rotate_by_quaternion_wxyz(delta_quaternion, best_position, rotated_position);
  candidate_position[0] = delta[0] + rotated_position[0];
  candidate_position[1] = delta[1] + rotated_position[1];
  candidate_position[2] = delta[2] + rotated_position[2];
  quaternion_multiply_wxyz(delta_quaternion, best_quaternion,
                           candidate_quaternion);
  normalize_quaternion_wxyz(candidate_quaternion);
}

__global__ void accept_candidate_kernel(
    const float* candidate_position, const float* candidate_quaternion,
    const float* candidate_jtj, const float* candidate_jtr,
    const float* candidate_sum_sq, const int* candidate_valid_count,
    float* best_position, float* best_quaternion, float* best_jtj,
    float* best_jtr, float* best_sum_sq, float* best_error,
    int* best_valid_count, float* lambda_damping,
    const float* predicted_reduction, int minimum_valid_correspondences,
    float lambda_factor, float lambda_minimum, float lambda_maximum,
    float* result_float, int* result_int) {
  if (threadIdx.x != 0 || blockIdx.x != 0 ||
      result_int[RESULT_STATUS] != POSE_REFINER_SUCCESS) {
    return;
  }

  const int candidate_count = candidate_valid_count[0];
  const bool has_enough_valid =
      candidate_count > minimum_valid_correspondences;
  const float candidate_error =
      has_enough_valid
          ? sqrtf(candidate_sum_sq[0] /
                  (static_cast<float>(candidate_count) + 1.0e-8f))
          : positive_infinity();
  const float actual_reduction = best_sum_sq[0] - candidate_sum_sq[0];
  const float trust_ratio =
      actual_reduction / (predicted_reduction[0] + 1.0e-8f);
  const bool step_accepted = has_enough_valid && trust_ratio >= 0.0f;

  if (step_accepted) {
    copy_float_array(candidate_position, best_position, 3);
    copy_float_array(candidate_quaternion, best_quaternion, 4);
    copy_float_array(candidate_jtj, best_jtj, 36);
    copy_float_array(candidate_jtr, best_jtr, 6);
    best_sum_sq[0] = candidate_sum_sq[0];
    best_error[0] = candidate_error;
    best_valid_count[0] = candidate_count;
    lambda_damping[0] =
        clamp_float(lambda_damping[0] / lambda_factor, lambda_minimum,
                    lambda_maximum);
    result_int[RESULT_ACCEPTED_STEPS] += 1;
  } else {
    lambda_damping[0] =
        clamp_float(lambda_damping[0] * lambda_factor, lambda_minimum,
                    lambda_maximum);
  }

  result_int[RESULT_ITERATIONS] += 1;
  result_float[RESULT_FINAL_ERROR] = best_error[0];
  result_float[RESULT_FINAL_LAMBDA] = lambda_damping[0];
  result_int[RESULT_VALID_COUNT] = best_valid_count[0];
}

__global__ void copy_best_pose_kernel(const float* best_position,
                                      const float* best_quaternion,
                                      float* output_position,
                                      float* output_quaternion) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 3) {
    output_position[index] = best_position[index];
  }
  if (index < 4) {
    output_quaternion[index] = best_quaternion[index];
  }
}

}  // namespace pose_refiner
}  // namespace optimization
}  // namespace curobo
