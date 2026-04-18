/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#define CUROBO_CLAMP_COLLISION_MAX_VALUE 1e5f
#define CUROBO_CLAMP_COLLISION_SQ_MAX_VALUE 100.0f
#define CUROBO_CLAMP_SELF_COLLISION_MAX_VALUE 1e5f
#define CUROBO_CLAMP_SELF_COLLISION_SQ_MAX_VALUE 100.0f
#define PI_F 3.141592654f
#define CUROBO_FLOAT32_PRECISION 1e-6f
#define VOLTA_PLUS true

#define CUDART_INF_F __int_as_float(0x7f800000)
#define CUDART_NINF_F __int_as_float(0xff800000)

// Geometry collision constants
#define CUROBO_VOXEL_UNOBSERVED_DISTANCE -1000.0f
#define CUROBO_DEFAULT_VOXEL_OFFSET 2



namespace curobo{
    namespace common{
        constexpr float  mathPI= 3.141592654f;
        constexpr float  fp32Precision= 1e-6f;
        constexpr float  clampCollisionMaxValue= 1e5f;
        constexpr float  clampCollisionSqMaxValue= 100.0f;
        constexpr float  clampSelfCollisionMaxValue= 1e5f;
        constexpr float  clampSelfCollisionSqMaxValue= 100.0f;
        constexpr int warpSize = 32;
        constexpr int blockSize = 1024;
        constexpr unsigned fullMask = 0xffffffffu;
        constexpr bool isVoltaPlus = true;

        // Geometry collision constants
        constexpr float voxelUnobservedDistance = -1000.0f;
        constexpr int defaultVoxelOffset = 2;
        constexpr float robustFloorThreshold = 1e-04f;
    }
}