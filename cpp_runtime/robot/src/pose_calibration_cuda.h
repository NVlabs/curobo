// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include "curobo_robot_runtime/pose_calibration.h"

namespace curobo::robot::detail {
struct CalibrationPose {
  float matrix[16];
  float error;
  int valid;
  int iterations;
  int hypothesis;
};
struct CalibrationLmState {
  float position[3], quaternion[4];
  float hessian[36], gradient[6], sum_squared;
  int valid;
  float damping;
  float predicted_reduction;
  int iterations;
  int initialized;
};
void launch_calibration_icp(const float *points, const float *normals, int mesh_count,
                            const float *observed, int observed_count, CalibrationPose *poses,
                            int hypotheses, int iterations, float distance, float huber_delta,
                            cudaStream_t stream);
void launch_calibration_select(const CalibrationPose *poses, int hypotheses,
                               CalibrationPose *result, cudaStream_t stream);
void launch_calibration_lm(const float *hessian, const float *gradient, const float *sum_squared,
                           const int *valid, float *position, float *quaternion,
                           CalibrationLmState *state, cudaStream_t stream);
}  // namespace curobo::robot::detail
