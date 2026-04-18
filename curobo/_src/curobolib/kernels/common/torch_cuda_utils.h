/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <torch/extension.h>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>


namespace curobo{
    namespace common{
        inline void validate_cuda_input(const torch::Tensor& x, const std::string& name){
            AT_ASSERTM(x.is_cuda(),  name + " must be a CUDA tensor");
            AT_ASSERTM(x.is_contiguous(),  name + " must be contiguous");
        }
        inline cudaStream_t get_cuda_stream(){
            return at::cuda::getCurrentCUDAStream();
        }
        inline cudaStream_t get_current_cuda_stream(){
            return at::cuda::getCurrentCUDAStream();
        }
    }
}