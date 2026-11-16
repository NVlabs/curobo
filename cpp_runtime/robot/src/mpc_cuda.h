// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace curobo::robot::detail {
// Geometric path distance ignores progress along normalized direction [3].
// Tool positions/quaternions use [steps,tools,3] and [tools,4] wxyz; error_frame
// is [tools] (0=world, 1=goal). No cost weights enter this acceptance check.
cudaError_t launch_mpc_pose_acceptance(float *position, float *rotation,
                                       const float *tool_position, const float *goal_position,
                                       const float *goal_quaternion, const float *direction,
                                       const int32_t *error_frame, const uint8_t *constrain_path,
                                       int steps, int tools, cudaStream_t stream);
// A free-endpoint seed pads with its last knot, never a stale c-space goal.
cudaError_t launch_mpc_tail(const float *seed, float *goal, const uint8_t *implicit_goal,
                            int knots, int dof, cudaStream_t stream);
// Whole-knot advancement of the winning spatial seed.
cudaError_t launch_mpc_shift(const float *input, float *output, const float *goal,
                             const int *shift, int rows, int knots, int dof, cudaStream_t stream);
cudaError_t launch_mpc_broadcast(const float *input, float *output, int values, int rows,
                                 cudaStream_t stream);
// Rank valid timings without comparing objective costs across different dt.
cudaError_t launch_mpc_rank(uint8_t *success, const float *cost, const float *dt,
                            float *rank, int candidates, cudaStream_t stream);
cudaError_t launch_mpc_validate(const float *q, const float *v, const float *a, const float *j,
                                const float *q_bounds, const float *v_bounds,
                                const float *a_bounds, const float *j_bounds, const float *world,
                                const float *self, const float *spheres, int num_spheres,
                                float max_sphere_step, int *violation, int steps, int dof,
                                cudaStream_t stream, int batches = 1);
}  // namespace curobo::robot::detail
