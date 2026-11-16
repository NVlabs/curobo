// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace curobo_mapper_runtime {

inline void checkCuda(cudaError_t status, const char* expr, const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string("CUDA runtime error at ") + file + ":" +
                           std::to_string(line) + " for " + expr + ": " +
                           cudaGetErrorString(status));
}

inline void checkCu(CUresult status, const char* expr, const char* file, int line) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  throw std::runtime_error(std::string("CUDA driver error at ") + file + ":" +
                           std::to_string(line) + " for " + expr + ": " +
                           (name ? name : "<unknown>") + " " +
                           (message ? message : ""));
}

}  // namespace curobo_mapper_runtime

#define CMR_CUDA_CHECK(expr) \
  ::curobo_mapper_runtime::checkCuda((expr), #expr, __FILE__, __LINE__)

#define CMR_CU_CHECK(expr) \
  ::curobo_mapper_runtime::checkCu((expr), #expr, __FILE__, __LINE__)
