/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



namespace curobo {
namespace common {

    __host__ __forceinline__ float rnorm4df(float x, float y, float z, float w)
    {
        return 1.0f / sqrtf(x * x + y * y + z * z + w * w);
    }
} // namespace common
} // namespace curobo
