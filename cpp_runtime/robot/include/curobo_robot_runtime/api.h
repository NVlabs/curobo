// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(CUROBO_ROBOT_RUNTIME_BUILD)
#define CUROBO_ROBOT_API __declspec(dllexport)
#else
#define CUROBO_ROBOT_API __declspec(dllimport)
#endif
#else
#define CUROBO_ROBOT_API __attribute__((visibility("default")))
#endif

typedef struct CuroboRobotRuntime CuroboRobotRuntime;

/** Robot dimensions fixed by the exported artifact. */
typedef struct CuroboRobotInfo {
  int num_links;
  int num_dof;
  int num_tool_frames;
} CuroboRobotInfo;

/**
 * Device tensors for fused forward kinematics and geometric Jacobians.
 *
 * All addresses refer to contiguous CUDA float32 storage. Input joint positions
 * have shape [batch_size, horizon, num_dof]. Outputs have shapes:
 *
 * - link_positions: [batch_size, horizon, num_tool_frames, 3]
 * - link_quaternions: [batch_size, horizon, num_tool_frames, 4], in wxyz order
 * - link_jacobians: [batch_size, horizon, num_tool_frames, 6, num_dof]
 *
 * Jacobian rows contain world-frame linear xyz followed by angular xyz. The
 * operation is enqueued on stream and does not synchronize it.
 */
typedef struct CuroboRobotKinematicsRequest {
  uint64_t joint_positions;
  uint64_t link_positions;
  uint64_t link_quaternions;
  uint64_t link_jacobians;
  uint64_t stream;
  int batch_size;
  int horizon;
} CuroboRobotKinematicsRequest;

/** Return the calling thread's most recent runtime error. */
CUROBO_ROBOT_API const char* curobo_robot_last_error(void);

/** Load a precompiled robot artifact on device_ordinal. */
CUROBO_ROBOT_API int curobo_robot_create(
    const char* artifact_dir,
    int device_ordinal,
    CuroboRobotRuntime** runtime);

/** Destroy a robot runtime. Passing null is allowed. */
CUROBO_ROBOT_API void curobo_robot_destroy(CuroboRobotRuntime* runtime);

/** Read dimensions fixed by the robot artifact. */
CUROBO_ROBOT_API int curobo_robot_get_info(
    CuroboRobotRuntime* runtime,
    CuroboRobotInfo* info);

/** Return an artifact-owned joint name, or null on error. */
CUROBO_ROBOT_API const char* curobo_robot_joint_name(
    CuroboRobotRuntime* runtime,
    int joint_index);

/** Return an artifact-owned tool-frame name, or null on error. */
CUROBO_ROBOT_API const char* curobo_robot_tool_frame_name(
    CuroboRobotRuntime* runtime,
    int tool_frame_index);

/** Enqueue fused forward kinematics and Jacobian computation. */
CUROBO_ROBOT_API int curobo_robot_compute_kinematics(
    CuroboRobotRuntime* runtime,
    const CuroboRobotKinematicsRequest* request);

#ifdef __cplusplus
}
#endif
