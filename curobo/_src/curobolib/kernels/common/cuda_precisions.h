/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>



#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BF16_TYPE_MACRO torch::kBFloat16