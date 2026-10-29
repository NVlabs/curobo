// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "warp_mesh_backend.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <version.h>
#include <warp.h>

namespace curobo::robot::detail {
namespace {

constexpr int WARP_BVH_CONSTRUCTOR_SAH = 0;
constexpr int WARP_BVH_LEAF_SIZE = 4;

}  // namespace

class WarpMeshBackend::Impl final {
 public:
  Impl() {
    const char* configured_path = std::getenv("CUROBO_WARP_LIBRARY");
    const char* library_path =
        configured_path != nullptr && configured_path[0] != '\0' ? configured_path : "warp.so";
    library_ = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      throw std::runtime_error(std::string("failed to load Warp native library '") +
                               library_path + "': " + dlerror());
    }
    try {
      init_ = load_symbol<decltype(init_)>("wp_init");
      get_error_ = load_symbol<decltype(get_error_)>("wp_get_error_string");
      is_cuda_enabled_ = load_symbol<decltype(is_cuda_enabled_)>("wp_is_cuda_enabled");
      create_ = load_symbol<decltype(create_)>("wp_mesh_create_device");
      destroy_ = load_symbol<decltype(destroy_)>("wp_mesh_destroy_device");
      if (init_(WP_VERSION_STRING) != 0) {
        throw std::runtime_error("failed to initialize Warp: " + error_string());
      }
      if (is_cuda_enabled_() == 0) {
        throw std::runtime_error("loaded Warp native library has no CUDA support");
      }
    } catch (...) {
      dlclose(library_);
      library_ = nullptr;
      throw;
    }
  }

  ~Impl() {
    if (library_ != nullptr) {
      dlclose(library_);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  uint64_t create_mesh(void* cuda_context, void* vertices, int num_vertices,
                       void* indices, int num_triangles) const {
    static_assert(sizeof(wp::vec3) == 3 * sizeof(float), "Warp vec3 layout changed");
    static_assert(sizeof(int) == sizeof(int32_t), "Warp triangle index layout changed");
    auto* point_pointer = static_cast<wp::vec3*>(vertices);
    auto* index_pointer = static_cast<int*>(indices);
    const wp::array_t<wp::vec3> points(point_pointer, num_vertices);
    const wp::array_t<wp::vec3> velocities;
    const wp::array_t<int> triangles(index_pointer, num_triangles * 3);
    const auto mesh_id = create_(cuda_context, points, velocities, triangles, num_vertices,
                                 num_triangles, 0, WARP_BVH_CONSTRUCTOR_SAH, nullptr,
                                 WARP_BVH_LEAF_SIZE);
    if (mesh_id == 0) {
      throw std::runtime_error("Warp failed to create mesh: " + error_string());
    }
    return mesh_id;
  }

  void destroy_mesh(uint64_t mesh_id) const noexcept { destroy_(mesh_id); }

 private:
  template <typename FunctionType>
  FunctionType load_symbol(const char* name) {
    dlerror();
    void* symbol = dlsym(library_, name);
    const char* error = dlerror();
    if (error != nullptr) {
      throw std::runtime_error(std::string("failed to resolve Warp symbol ") + name +
                               ": " + error);
    }
    static_assert(sizeof(FunctionType) == sizeof(symbol),
                  "function and object pointer sizes differ on this platform");
    FunctionType function = nullptr;
    std::memcpy(&function, &symbol, sizeof(function));
    return function;
  }

  std::string error_string() const {
    const char* error = get_error_();
    return error != nullptr ? error : "unknown Warp error";
  }

  void* library_{nullptr};
  decltype(&wp_init) init_{nullptr};
  decltype(&wp_get_error_string) get_error_{nullptr};
  decltype(&wp_is_cuda_enabled) is_cuda_enabled_{nullptr};
  decltype(&wp_mesh_create_device) create_{nullptr};
  decltype(&wp_mesh_destroy_device) destroy_{nullptr};
};

WarpMeshBackend::WarpMeshBackend() : impl_(std::make_unique<Impl>()) {}

WarpMeshBackend::~WarpMeshBackend() = default;

uint64_t WarpMeshBackend::create_mesh(void* cuda_context, void* vertices,
                                      int num_vertices, void* indices,
                                      int num_triangles) const {
  return impl_->create_mesh(cuda_context, vertices, num_vertices, indices, num_triangles);
}

void WarpMeshBackend::destroy_mesh(uint64_t mesh_id) const noexcept {
  impl_->destroy_mesh(mesh_id);
}

}  // namespace curobo::robot::detail
