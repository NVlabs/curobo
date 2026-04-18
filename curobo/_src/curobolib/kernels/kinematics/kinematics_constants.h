/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//#define M 4

#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5

#define MAX_FW_BATCH_PER_BLOCK 128    // tunable parameter for improving occupancy
#define MAX_BW_BATCH_PER_BLOCK 128 // tunable parameter for improving occupancy


namespace curobo{
    namespace kinematics{

        enum class JointType : int {
            Fixed = -1,
            XPrismatic = 0,
            YPrismatic = 1,
            ZPrismatic = 2,
            XRevolute = 3,
            YRevolute = 4,
            ZRevolute = 5
        };
        constexpr int  maxFwBatchPerBlock= 128;
        constexpr int  maxBwBatchPerBlock= 128;

    }
}