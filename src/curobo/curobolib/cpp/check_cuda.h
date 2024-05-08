/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), # x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), # x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x);  CHECK_CONTIGUOUS(x)

#define CHECK_FP8 defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 2
#define CHECK_INPUT_GUARD(x) CHECK_INPUT(x);   const at::cuda::OptionalCUDAGuard guard(x.device())

#if CHECK_FP8
    #define FP8_TYPE_MACRO torch::kFloat8_e4m3fn
    //constexpr const auto fp8_type = torch::kFloat8_e4m3fn;
#else
    #define FP8_TYPE_MACRO torch::kHalf
    //const constexpr auto fp8_type = torch::kHalf;
#endif 