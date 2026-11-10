// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda.h>

#include <string>
#include <vector>

namespace curobo_mapper_runtime {

struct Dim3 {
  unsigned int x{1};
  unsigned int y{1};
  unsigned int z{1};
};

class CubinKernel {
 public:
  CubinKernel() = default;
  CubinKernel(std::string cubin_path, std::string symbol);
  ~CubinKernel();

  CubinKernel(const CubinKernel&) = delete;
  CubinKernel& operator=(const CubinKernel&) = delete;
  CubinKernel(CubinKernel&& other) noexcept;
  CubinKernel& operator=(CubinKernel&& other) noexcept;

  void load();
  void launch(Dim3 grid, Dim3 block, unsigned int shared_mem, CUstream stream,
              const std::vector<void*>& arg_ptrs) const;

  bool loaded() const { return function_ != nullptr; }
  const std::string& symbol() const { return symbol_; }

 private:
  void unload();

  std::string cubin_path_;
  std::string symbol_;
  CUmodule module_{nullptr};
  CUfunction function_{nullptr};
};

}  // namespace curobo_mapper_runtime
