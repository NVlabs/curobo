// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace curobo_robot_runtime {

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string("CUDA runtime error at ") + file + ":" +
                           std::to_string(line) + " for " + expression + ": " +
                           cudaGetErrorString(status));
}

inline void check_driver(CUresult status, const char* expression, const char* file, int line) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  throw std::runtime_error(std::string("CUDA driver error at ") + file + ":" +
                           std::to_string(line) + " for " + expression + ": " +
                           (name != nullptr ? name : "<unknown>") + " " +
                           (message != nullptr ? message : ""));
}

}  // namespace curobo_robot_runtime

#define CRR_CUDA_CHECK(expression) \
  ::curobo_robot_runtime::check_cuda((expression), #expression, __FILE__, __LINE__)

#define CRR_CU_CHECK(expression) \
  ::curobo_robot_runtime::check_driver((expression), #expression, __FILE__, __LINE__)
