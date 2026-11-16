// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#include "cubin_kernel.h"

#include "cuda_check.h"

#include <utility>

namespace curobo_mapper_runtime {

CubinKernel::CubinKernel(std::string cubin_path, std::string symbol)
    : cubin_path_(std::move(cubin_path)), symbol_(std::move(symbol)) {}

CubinKernel::~CubinKernel() { unload(); }

CubinKernel::CubinKernel(CubinKernel&& other) noexcept
    : cubin_path_(std::move(other.cubin_path_)),
      symbol_(std::move(other.symbol_)),
      module_(other.module_),
      function_(other.function_) {
  other.module_ = nullptr;
  other.function_ = nullptr;
}

CubinKernel& CubinKernel::operator=(CubinKernel&& other) noexcept {
  if (this != &other) {
    unload();
    cubin_path_ = std::move(other.cubin_path_);
    symbol_ = std::move(other.symbol_);
    module_ = other.module_;
    function_ = other.function_;
    other.module_ = nullptr;
    other.function_ = nullptr;
  }
  return *this;
}

void CubinKernel::load() {
  if (function_ != nullptr) {
    return;
  }
  CMR_CU_CHECK(cuModuleLoad(&module_, cubin_path_.c_str()));
  CMR_CU_CHECK(cuModuleGetFunction(&function_, module_, symbol_.c_str()));
}

void CubinKernel::launch(Dim3 grid, Dim3 block, unsigned int shared_mem,
                         CUstream stream,
                         const std::vector<void*>& arg_ptrs) const {
  CMR_CU_CHECK(cuLaunchKernel(function_, grid.x, grid.y, grid.z, block.x, block.y,
                              block.z, shared_mem, stream,
                              const_cast<void**>(arg_ptrs.data()), nullptr));
}

void CubinKernel::unload() {
  function_ = nullptr;
  if (module_ != nullptr) {
    cuModuleUnload(module_);
    module_ = nullptr;
  }
}

}  // namespace curobo_mapper_runtime
