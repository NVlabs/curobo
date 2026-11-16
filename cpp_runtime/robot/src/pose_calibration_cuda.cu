// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>

#include "cuda_check.h"
#include "pose_calibration_cuda.h"

namespace curobo::robot::detail {
namespace {
constexpr int BLOCK = 256;
constexpr int TERMS = 44;

__device__ bool solve6(const float* h, const float* b, float damping, float* x) {
  float l[36] = {};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j <= i; ++j) {
      float value = h[i * 6 + j] + (i == j ? damping : 0.F);
      for (int k = 0; k < j; ++k) value -= l[i * 6 + k] * l[j * 6 + k];
      if (i == j) {
        if (!isfinite(value) || value <= 0.F) return false;
        l[i * 6 + j] = sqrtf(value);
      } else
        l[i * 6 + j] = value / l[j * 6 + j];
    }
  float y[6];
  for (int i = 0; i < 6; ++i) {
    float v = b[i];
    for (int j = 0; j < i; ++j) v -= l[i * 6 + j] * y[j];
    y[i] = v / l[i * 6 + i];
  }
  for (int i = 5; i >= 0; --i) {
    float v = y[i];
    for (int j = i + 1; j < 6; ++j) v -= l[j * 6 + i] * x[j];
    x[i] = v / l[i * 6 + i];
    if (!isfinite(x[i])) return false;
  }
  return true;
}

__device__ void quaternion_matrix(const float* q, float* r) {
  float w = q[0], x = q[1], y = q[2], z = q[3];
  r[0] = 1 - 2 * (y * y + z * z);
  r[1] = 2 * (x * y - w * z);
  r[2] = 2 * (x * z + w * y);
  r[3] = 2 * (x * y + w * z);
  r[4] = 1 - 2 * (x * x + z * z);
  r[5] = 2 * (y * z - w * x);
  r[6] = 2 * (x * z - w * y);
  r[7] = 2 * (y * z + w * x);
  r[8] = 1 - 2 * (x * x + y * y);
}

__device__ void left_update(float* t, const float* omega, const float* translation) {
  float theta = sqrtf(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]);
  float scale = theta < 1.e-10F ? .5F : sinf(theta * .5F) / theta;
  float q[4] = {cosf(theta * .5F), omega[0] * scale, omega[1] * scale, omega[2] * scale}, r[9],
        out[16];
  quaternion_matrix(q, r);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) {
      out[i * 4 + j] = j == 3 ? translation[i] : 0.F;
      for (int k = 0; k < 3; ++k) out[i * 4 + j] += r[i * 3 + k] * t[k * 4 + j];
    }
  for (int i = 0; i < 12; ++i) t[i] = out[i];
}

__device__ float nearest(const float* source, const float* observed, int count, int& index) {
  float best = INFINITY;
  index = -1;
  for (int i = 0; i < count; ++i) {
    float x = source[0] - observed[i * 3], y = source[1] - observed[i * 3 + 1],
          z = source[2] - observed[i * 3 + 2];
    float distance = x * x + y * y + z * z;
    if (distance < best) {
      best = distance;
      index = i;
    }
  }
  return sqrtf(best);
}

// A block owns one rotation hypothesis. Correspondences and Huber normal
// equations stay on the GPU for every iteration; no pairwise-distance matrix.
__global__ void icp_kernel(const float* points, const float* normals, int mesh_count,
                           const float* observed, int observed_count, CalibrationPose* poses,
                           int iterations, float threshold, float huber) {
  __shared__ float t[16], partial[8][TERMS];
  __shared__ int active, used, valid_count;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (threadIdx.x < 16) t[threadIdx.x] = poses[blockIdx.x].matrix[threadIdx.x];
  if (threadIdx.x == 0) {
    active = 1;
    used = 0;
    valid_count = 0;
  }
  __syncthreads();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    float terms[TERMS] = {};
    for (int p = threadIdx.x; p < mesh_count; p += BLOCK) {
      float s[3] = {}, n[3] = {};
      for (int i = 0; i < 3; ++i) {
        s[i] = t[i * 4 + 3];
        for (int k = 0; k < 3; ++k) {
          s[i] += t[i * 4 + k] * points[p * 3 + k];
          n[i] += t[i * 4 + k] * normals[p * 3 + k];
        }
      }
      int index;
      float distance = nearest(s, observed, observed_count, index);
      if (distance > threshold) continue;
      float residual = 0;
      for (int k = 0; k < 3; ++k) residual += (observed[index * 3 + k] - s[k]) * n[k];
      float weight = fabsf(residual) <= huber ? 1.F : huber / fabsf(residual);
      float j[6] = {s[1] * n[2] - s[2] * n[1],
                    s[2] * n[0] - s[0] * n[2],
                    s[0] * n[1] - s[1] * n[0],
                    n[0],
                    n[1],
                    n[2]};
      for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 6; ++k) terms[i * 6 + k] += weight * j[i] * j[k];
        terms[36 + i] += weight * j[i] * residual;
      }
      terms[43] += 1.F;
    }
    for (int k = 0; k < TERMS; ++k) {
      float value = terms[k];
      for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if (lane == 0) partial[warp][k] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total[TERMS] = {};
      for (int w = 0; w < 8; ++w)
        for (int k = 0; k < TERMS; ++k) total[k] += partial[w][k];
      valid_count = static_cast<int>(total[43]);
      used = iteration + 1;
      float delta[6] = {};
      active = valid_count >= 10 && solve6(total, total + 36, 1.e-6F, delta);
      if (active) {
        left_update(t, delta, delta + 3);
        float rotation = 0, translation = 0;
        for (int k = 0; k < 3; ++k) {
          rotation += delta[k] * delta[k];
          translation += delta[k + 3] * delta[k + 3];
        }
        // Check both components, including the final update. Pure rotations
        // must not exit early because their translation happens to be small.
        if (rotation < 1.e-10F && translation < 1.e-8F) active = 0;
      }
    }
    __syncthreads();
    if (!active) break;
  }
  float error = 0.F;
  for (int p = threadIdx.x; p < mesh_count; p += BLOCK) {
    float s[3] = {};
    for (int i = 0; i < 3; ++i) {
      s[i] = t[i * 4 + 3];
      for (int k = 0; k < 3; ++k) s[i] += t[i * 4 + k] * points[p * 3 + k];
    }
    int index;
    error += nearest(s, observed, observed_count, index);
  }
  for (int offset = 16; offset; offset /= 2) error += __shfl_down_sync(0xffffffff, error, offset);
  if (lane == 0) partial[warp][0] = error;
  __syncthreads();
  if (threadIdx.x < 16) poses[blockIdx.x].matrix[threadIdx.x] = t[threadIdx.x];
  if (threadIdx.x == 0) {
    error = 0;
    for (int w = 0; w < 8; ++w) error += partial[w][0];
    poses[blockIdx.x].error = error / mesh_count;
    poses[blockIdx.x].valid = valid_count;
    poses[blockIdx.x].iterations = used;
  }
}

__global__ void select_kernel(const CalibrationPose* poses, int count, CalibrationPose* best) {
  int index = 0;
  for (int i = 1; i < count; ++i)
    if (poses[i].error < poses[index].error) index = i;
  *best = poses[index];
  best->hypothesis = index;
}

// Same unsigned-distance LM equations as optim_pose_lm.py. A single thread
// solves the 6x6 system; BVH queries and reductions are separate AOT kernels.
__global__ void lm_kernel(const float* h, const float* g, const float* squared, const int* valid,
                          float* position, float* quaternion, CalibrationLmState* state) {
  bool accept = !state->initialized;
  if (state->initialized) {
    float rho = (state->sum_squared - *squared) / (state->predicted_reduction + 1.e-8F);
    accept = *valid > 10 && rho >= 0.F && isfinite(*squared);
    state->damping = fminf(1.e7F, fmaxf(1.e-7F, state->damping * (accept ? .1F : 10.F)));
    ++state->iterations;
  } else {
    state->damping = 1.e-3F;
    state->initialized = 1;
  }
  if (accept) {
    for (int i = 0; i < 3; ++i) state->position[i] = position[i];
    for (int i = 0; i < 4; ++i) state->quaternion[i] = quaternion[i];
    for (int i = 0; i < 36; ++i) state->hessian[i] = h[i];
    for (int i = 0; i < 6; ++i) state->gradient[i] = g[i];
    state->sum_squared = *squared;
    state->valid = *valid;
  }
  float delta[6] = {}, rhs[6];
  for (int i = 0; i < 6; ++i) rhs[i] = -state->gradient[i];
  if (!solve6(state->hessian, rhs, state->damping, delta))
    for (float& d : delta) d = 0;
  state->predicted_reduction = 0;
  for (int i = 0; i < 6; ++i) {
    state->predicted_reduction -= delta[i] * state->gradient[i];
    for (int k = 0; k < 6; ++k)
      state->predicted_reduction -= .5F * delta[i] * state->hessian[i * 6 + k] * delta[k];
  }
  // Pose.from_euler_xyz uses Rz * Ry * Rx, with left composition.
  float cx = cosf(delta[3] * .5F), sx = sinf(delta[3] * .5F), cy = cosf(delta[4] * .5F),
        sy = sinf(delta[4] * .5F), cz = cosf(delta[5] * .5F), sz = sinf(delta[5] * .5F);
  float q[4] = {cx * cy * cz + sx * sy * sz, sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz},
        r[9];
  quaternion_matrix(q, r);
  for (int i = 0; i < 3; ++i) {
    position[i] = delta[i];
    for (int k = 0; k < 3; ++k) position[i] += r[i * 3 + k] * state->position[k];
  }
  const float* p = state->quaternion;
  quaternion[0] = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3];
  quaternion[1] = q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2];
  quaternion[2] = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1];
  quaternion[3] = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0];
  float norm = 0;
  for (int i = 0; i < 4; ++i) norm += quaternion[i] * quaternion[i];
  for (int i = 0; i < 4; ++i) quaternion[i] *= rsqrtf(norm);
}
}  // namespace

void launch_calibration_icp(const float* p, const float* n, int m, const float* o, int count,
                            CalibrationPose* poses, int hypotheses, int iterations, float distance,
                            float huber, cudaStream_t stream) {
  icp_kernel<<<hypotheses, BLOCK, 0, stream>>>(p, n, m, o, count, poses, iterations, distance,
                                               huber);
  CRR_CUDA_CHECK(cudaGetLastError());
}
void launch_calibration_select(const CalibrationPose* poses, int count, CalibrationPose* best,
                               cudaStream_t stream) {
  select_kernel<<<1, 1, 0, stream>>>(poses, count, best);
  CRR_CUDA_CHECK(cudaGetLastError());
}
void launch_calibration_lm(const float* h, const float* g, const float* sq, const int* valid,
                           float* p, float* q, CalibrationLmState* state, cudaStream_t stream) {
  lm_kernel<<<1, 1, 0, stream>>>(h, g, sq, valid, p, q, state);
  CRR_CUDA_CHECK(cudaGetLastError());
}
}  // namespace curobo::robot::detail
