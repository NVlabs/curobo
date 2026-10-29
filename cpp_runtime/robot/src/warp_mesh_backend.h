// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>

namespace curobo::robot::detail {

/** Private adapter around Warp's native mesh/BVH interface. */
class WarpMeshBackend final {
 public:
  WarpMeshBackend();
  ~WarpMeshBackend();

  WarpMeshBackend(const WarpMeshBackend&) = delete;
  WarpMeshBackend& operator=(const WarpMeshBackend&) = delete;

  uint64_t create_mesh(void* cuda_context, void* vertices, int num_vertices,
                       void* indices, int num_triangles) const;
  void destroy_mesh(uint64_t mesh_id) const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace curobo::robot::detail
