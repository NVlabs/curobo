// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "scene_runtime.h"

#include "cuda_check.h"

#if defined(CUROBO_ROBOT_HAS_WARP_MESH)
#include "warp_mesh_backend.h"
#endif

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace curobo::robot::detail {
namespace {

constexpr int POSE_STRIDE = 8;
constexpr int BOUNDS_STRIDE = 8;
constexpr int SMALL_PARAM_STRIDE = 4;
constexpr int CAPSULE_PARAM_STRIDE = 8;
constexpr float MIN_QUATERNION_NORM = 1.0E-12F;
constexpr float CAPSULE_SEGMENT_EPSILON = 1.0E-6F;

static_assert(sizeof(std::array<float, 3>) == 3 * sizeof(float));
static_assert(sizeof(std::array<int32_t, 3>) == 3 * sizeof(int32_t));

size_t checked_bytes(size_t count, size_t element_size) {
  if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::invalid_argument("scene capacity is too large");
  }
  return count * element_size;
}

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    CRR_CUDA_CHECK(cudaMalloc(&pointer_, std::max<size_t>(bytes, 1)));
    if (bytes_ != 0) {
      const auto status = cudaMemset(pointer_, 0, bytes_);
      if (status != cudaSuccess) {
        cudaFree(pointer_);
        pointer_ = nullptr;
        CRR_CUDA_CHECK(status);
      }
    }
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void upload(const void* source, size_t bytes, size_t offset = 0) {
    if (offset > bytes_ || bytes > bytes_ - offset) {
      throw std::out_of_range("device scene buffer write exceeds capacity");
    }
    if (bytes == 0) {
      return;
    }
    auto* destination = static_cast<uint8_t*>(pointer_) + offset;
    CRR_CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
  }

  void copy_from_device_async(const void* source, size_t bytes, size_t offset,
                              cudaStream_t stream) {
    if (source == nullptr) {
      throw std::invalid_argument("source CUDA pointer is null");
    }
    if (offset > bytes_ || bytes > bytes_ - offset) {
      throw std::out_of_range("device scene buffer write exceeds capacity");
    }
    if (bytes != 0) {
      auto* destination = static_cast<uint8_t*>(pointer_) + offset;
      CRR_CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes,
                                     cudaMemcpyDeviceToDevice, stream));
    }
  }

  void clear() {
    if (bytes_ != 0) {
      CRR_CUDA_CHECK(cudaMemset(pointer_, 0, bytes_));
    }
  }

  void* pointer() const noexcept { return pointer_; }
  CUdeviceptr device_pointer() const noexcept {
    return reinterpret_cast<CUdeviceptr>(pointer_);
  }

 private:
  void* pointer_{nullptr};
  size_t bytes_{0};
};

struct Bounds {
  std::array<float, 3> lower{};
  std::array<float, 3> upper{};
};

struct ValidatedPose {
  std::array<float, 3> position{};
  std::array<float, 4> quaternion{};
};

bool finite(float value) { return std::isfinite(value); }

template <size_t Size>
void require_finite(const std::array<float, Size>& values, const char* description) {
  for (const auto value : values) {
    if (!finite(value)) {
      throw std::invalid_argument(std::string(description) + " must contain finite values");
    }
  }
}

void require_positive(const std::array<float, 3>& values, const char* description) {
  require_finite(values, description);
  for (const auto value : values) {
    if (value <= 0.0F) {
      throw std::invalid_argument(std::string(description) + " must be positive");
    }
  }
}

ValidatedPose validate_pose(const Pose& pose) {
  require_finite(pose.position, "pose position");
  require_finite(pose.quaternion, "pose quaternion");
  float norm_squared = 0.0F;
  for (const auto value : pose.quaternion) {
    norm_squared += value * value;
  }
  if (norm_squared <= MIN_QUATERNION_NORM) {
    throw std::invalid_argument("pose quaternion must have non-zero norm");
  }
  // Match Python Scene/Pose semantics: validate, but preserve the supplied
  // float quaternion. Renormalizing here perturbs inverse obstacle poses and
  // collision gradients enough to change optimizer convergence near contact.
  return {pose.position, pose.quaternion};
}

std::array<float, 3> rotate_vector(const std::array<float, 4>& quaternion,
                                   const std::array<float, 3>& vector) {
  const float w = quaternion.at(0);
  const float x = quaternion.at(1);
  const float y = quaternion.at(2);
  const float z = quaternion.at(3);
  const std::array<float, 3> cross{
      y * vector.at(2) - z * vector.at(1),
      z * vector.at(0) - x * vector.at(2),
      x * vector.at(1) - y * vector.at(0),
  };
  const std::array<float, 3> second_cross{
      y * cross.at(2) - z * cross.at(1),
      z * cross.at(0) - x * cross.at(2),
      x * cross.at(1) - y * cross.at(0),
  };
  return {
      vector.at(0) + 2.0F * (w * cross.at(0) + second_cross.at(0)),
      vector.at(1) + 2.0F * (w * cross.at(1) + second_cross.at(1)),
      vector.at(2) + 2.0F * (w * cross.at(2) + second_cross.at(2)),
  };
}

std::array<float, POSE_STRIDE> inverse_pose_row(const ValidatedPose& pose) {
  const std::array<float, 4> inverse_quaternion{
      pose.quaternion.at(0),
      -pose.quaternion.at(1),
      -pose.quaternion.at(2),
      -pose.quaternion.at(3),
  };
  const std::array<float, 3> negative_position{
      -pose.position.at(0), -pose.position.at(1), -pose.position.at(2)};
  const auto inverse_position = rotate_vector(inverse_quaternion, negative_position);
  return {
      inverse_position.at(0),
      inverse_position.at(1),
      inverse_position.at(2),
      inverse_quaternion.at(0),
      inverse_quaternion.at(1),
      inverse_quaternion.at(2),
      inverse_quaternion.at(3),
      0.0F,
  };
}

std::array<float, BOUNDS_STRIDE> world_bounds_row(const Bounds& bounds,
                                                   const ValidatedPose& pose) {
  std::array<float, 3> local_center{};
  std::array<float, 3> local_extent{};
  for (size_t axis = 0; axis < local_center.size(); ++axis) {
    local_center.at(axis) = 0.5F * (bounds.lower.at(axis) + bounds.upper.at(axis));
    local_extent.at(axis) = 0.5F * (bounds.upper.at(axis) - bounds.lower.at(axis));
  }
  const auto rotated_center = rotate_vector(pose.quaternion, local_center);

  const float w = pose.quaternion.at(0);
  const float x = pose.quaternion.at(1);
  const float y = pose.quaternion.at(2);
  const float z = pose.quaternion.at(3);
  const std::array<std::array<float, 3>, 3> rotation{{
      {1.0F - 2.0F * (y * y + z * z), 2.0F * (x * y - w * z),
       2.0F * (x * z + w * y)},
      {2.0F * (x * y + w * z), 1.0F - 2.0F * (x * x + z * z),
       2.0F * (y * z - w * x)},
      {2.0F * (x * z - w * y), 2.0F * (y * z + w * x),
       1.0F - 2.0F * (x * x + y * y)},
  }};
  std::array<float, 3> world_extent{};
  for (size_t row = 0; row < rotation.size(); ++row) {
    for (size_t column = 0; column < rotation.at(row).size(); ++column) {
      world_extent.at(row) +=
          std::abs(rotation.at(row).at(column)) * local_extent.at(column);
    }
  }
  return {
      pose.position.at(0) + rotated_center.at(0),
      pose.position.at(1) + rotated_center.at(1),
      pose.position.at(2) + rotated_center.at(2),
      0.0F,
      world_extent.at(0),
      world_extent.at(1),
      world_extent.at(2),
      0.0F,
  };
}

uint16_t float_to_half_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  uint32_t mantissa = bits & 0x007FFFFFU;
  int exponent = static_cast<int>((bits >> 23U) & 0xFFU) - 127 + 15;

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x00800000U;
    const int shift = 14 - exponent;
    uint32_t half_mantissa = mantissa >> shift;
    const uint32_t remainder_mask = (uint32_t{1} << shift) - 1U;
    const uint32_t remainder = mantissa & remainder_mask;
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
      ++half_mantissa;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }

  uint32_t half = static_cast<uint32_t>(exponent) << 10U;
  half |= mantissa >> 13U;
  const uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U) != 0U)) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

int type_index(ObstacleType type) {
  switch (type) {
    case ObstacleType::CUBOID:
      return 0;
    case ObstacleType::CAPSULE:
      return 1;
    case ObstacleType::CYLINDER:
      return 2;
    case ObstacleType::SPHERE:
      return 3;
    case ObstacleType::MESH:
      return 4;
    case ObstacleType::VOXEL_GRID:
      return 5;
  }
  throw std::invalid_argument("unknown obstacle type");
}

uint64_t next_obstacle_handle() {
  static std::atomic<uint64_t> next_handle{1};
  const auto handle = next_handle.fetch_add(1, std::memory_order_relaxed);
  if (handle == 0) {
    throw std::overflow_error("process exhausted obstacle handles");
  }
  return handle;
}

struct ObstacleStorage {
  ObstacleStorage(int requested_capacity, int param_stride)
      : capacity(requested_capacity),
        params(checked_bytes(static_cast<size_t>(capacity) * param_stride,
                             sizeof(float))),
        world_bounds(checked_bytes(
            static_cast<size_t>(capacity) * BOUNDS_STRIDE, sizeof(float))),
        inverse_poses(checked_bytes(static_cast<size_t>(capacity) * POSE_STRIDE,
                                    sizeof(float))),
        enabled(checked_bytes(static_cast<size_t>(capacity), sizeof(uint8_t))),
        obstacle_layers(
            checked_bytes(static_cast<size_t>(capacity), sizeof(uint32_t))),
        count(sizeof(int32_t)) {}

  void write_count() {
    const auto value = static_cast<int32_t>(loaded);
    count.upload(&value, sizeof(value));
  }

  int capacity{0};
  int loaded{0};
  DeviceBuffer params;
  DeviceBuffer world_bounds;
  DeviceBuffer inverse_poses;
  DeviceBuffer enabled;
  DeviceBuffer obstacle_layers;
  DeviceBuffer count;
};

class MeshResource final {
 public:
  MeshResource(size_t vertex_bytes, size_t index_bytes)
      : vertices(vertex_bytes), indices(index_bytes) {}

  ~MeshResource() {
#if defined(CUROBO_ROBOT_HAS_WARP_MESH)
    if (mesh_id != 0 && backend != nullptr) {
      backend->destroy_mesh(mesh_id);
    }
#endif
  }

  MeshResource(const MeshResource&) = delete;
  MeshResource& operator=(const MeshResource&) = delete;

  DeviceBuffer vertices;
  DeviceBuffer indices;
  uint64_t mesh_id{0};
#if defined(CUROBO_ROBOT_HAS_WARP_MESH)
  WarpMeshBackend* backend{nullptr};
#endif
};

struct MeshStorage {
  explicit MeshStorage(int requested_capacity)
      : capacity(requested_capacity),
        mesh_ids(
            checked_bytes(static_cast<size_t>(capacity), sizeof(uint64_t))),
        dimensions(checked_bytes(
            static_cast<size_t>(capacity) * SMALL_PARAM_STRIDE, sizeof(float))),
        world_bounds(checked_bytes(
            static_cast<size_t>(capacity) * BOUNDS_STRIDE, sizeof(float))),
        inverse_poses(checked_bytes(static_cast<size_t>(capacity) * POSE_STRIDE,
                                    sizeof(float))),
        enabled(checked_bytes(static_cast<size_t>(capacity), sizeof(uint8_t))),
        obstacle_layers(
            checked_bytes(static_cast<size_t>(capacity), sizeof(uint32_t))),
        count(sizeof(int32_t)),
        resources(static_cast<size_t>(capacity)) {}

  void write_count() {
    const auto value = static_cast<int32_t>(loaded);
    count.upload(&value, sizeof(value));
  }

  int capacity{0};
  int loaded{0};
  DeviceBuffer mesh_ids;
  DeviceBuffer dimensions;
  DeviceBuffer world_bounds;
  DeviceBuffer inverse_poses;
  DeviceBuffer enabled;
  DeviceBuffer obstacle_layers;
  DeviceBuffer count;
  std::vector<std::unique_ptr<MeshResource>> resources;
};

struct VoxelStorage {
  VoxelStorage(int requested_capacity, int requested_max_voxels)
      : capacity(requested_capacity),
        max_voxels_per_grid(requested_max_voxels),
        params(checked_bytes(static_cast<size_t>(capacity) * SMALL_PARAM_STRIDE,
                             sizeof(float))),
        dimensions(checked_bytes(
            static_cast<size_t>(capacity) * SMALL_PARAM_STRIDE, sizeof(float))),
        inverse_poses(checked_bytes(static_cast<size_t>(capacity) * POSE_STRIDE,
                                    sizeof(float))),
        enabled(checked_bytes(static_cast<size_t>(capacity), sizeof(uint8_t))),
        obstacle_layers(
            checked_bytes(static_cast<size_t>(capacity), sizeof(uint32_t))),
        features(
            checked_bytes(static_cast<size_t>(capacity) * max_voxels_per_grid,
                          sizeof(uint16_t))),
        count(sizeof(int32_t)) {}

  void write_count() {
    const auto value = static_cast<int32_t>(loaded);
    count.upload(&value, sizeof(value));
  }

  int capacity{0};
  int max_voxels_per_grid{0};
  int loaded{0};
  DeviceBuffer params;
  DeviceBuffer dimensions;
  DeviceBuffer inverse_poses;
  DeviceBuffer enabled;
  DeviceBuffer obstacle_layers;
  DeviceBuffer features;
  DeviceBuffer count;
};

enum class StorageKind {
  CUBOID,
  CAPSULE,
  CYLINDER,
  MESH,
  VOXEL_GRID,
};

struct Entry {
  ObstacleType type{ObstacleType::CUBOID};
  StorageKind storage{StorageKind::CUBOID};
  int slot{0};
  Bounds local_bounds{};
  size_t voxel_count{0};
};

void write_row(DeviceBuffer& buffer, int slot, int stride, const float* values) {
  const auto offset = checked_bytes(static_cast<size_t>(slot) * stride, sizeof(float));
  buffer.upload(values, checked_bytes(static_cast<size_t>(stride), sizeof(float)), offset);
}

void write_enabled(DeviceBuffer& buffer, int slot, bool enabled) {
  const uint8_t value = enabled ? 1U : 0U;
  buffer.upload(&value, sizeof(value), static_cast<size_t>(slot));
}

Bounds centered_bounds(const std::array<float, 3>& dimensions) {
  Bounds bounds;
  for (size_t axis = 0; axis < dimensions.size(); ++axis) {
    bounds.lower.at(axis) = -0.5F * dimensions.at(axis);
    bounds.upper.at(axis) = 0.5F * dimensions.at(axis);
  }
  return bounds;
}

}  // namespace

class SceneRuntime::Impl final {
 public:
  explicit Impl(const SceneConfig& requested_config)
      : config_(validate_config(requested_config)),
        cuboids_(config_.max_cuboids, SMALL_PARAM_STRIDE),
        capsules_(config_.max_capsules, CAPSULE_PARAM_STRIDE),
        cylinders_(config_.max_cylinders, SMALL_PARAM_STRIDE),
        meshes_(config_.max_meshes),
        voxels_(config_.max_voxel_grids, config_.max_voxels_per_grid) {}

  ~Impl() {
    const auto status = cudaSetDevice(config_.device_ordinal);
    if (status == cudaSuccess) {
      meshes_.resources.clear();
    }
  }

  const SceneConfig& config() const noexcept { return config_; }

  SceneDeviceView device_view() const noexcept {
    return {
        obstacle_view(cuboids_, SMALL_PARAM_STRIDE),
        obstacle_view(capsules_, CAPSULE_PARAM_STRIDE),
        obstacle_view(cylinders_, SMALL_PARAM_STRIDE),
        MeshDeviceView{
            meshes_.mesh_ids.device_pointer(),
            meshes_.dimensions.device_pointer(),
            meshes_.world_bounds.device_pointer(),
            meshes_.inverse_poses.device_pointer(),
            meshes_.enabled.device_pointer(),
            meshes_.obstacle_layers.device_pointer(),
            meshes_.count.device_pointer(),
            meshes_.capacity,
            0.1F,
        },
        VoxelDeviceView{
            voxels_.params.device_pointer(),
            voxels_.dimensions.device_pointer(),
            voxels_.inverse_poses.device_pointer(),
            voxels_.enabled.device_pointer(),
            voxels_.obstacle_layers.device_pointer(),
            voxels_.features.device_pointer(),
            voxels_.count.device_pointer(),
            voxels_.capacity,
            voxels_.max_voxels_per_grid,
            config_.max_esdf_distance,
        },
    };
  }

  int obstacle_count(ObstacleType type) const {
    return obstacle_counts_.at(static_cast<size_t>(type_index(type)));
  }

  bool mesh_support_available() const noexcept {
#if defined(CUROBO_ROBOT_HAS_WARP_MESH)
    try {
      activate_device();
      return load_mesh_backend();
    } catch (const std::exception& error) {
      mesh_backend_error_ = error.what();
      return false;
    }
#else
    return false;
#endif
  }

  ObstacleHandle add_cuboid(const Cuboid& cuboid) {
    require_positive(cuboid.dimensions, "cuboid dimensions");
    const auto pose = validate_pose(cuboid.pose);
    activate_device();
    ensure_add_allowed(cuboid.name, cuboids_.loaded, cuboids_.capacity, "cuboids");

    const int slot = cuboids_.loaded;
    const std::array<float, SMALL_PARAM_STRIDE> params{
        cuboid.dimensions.at(0), cuboid.dimensions.at(1), cuboid.dimensions.at(2), 0.0F};
    const auto bounds = centered_bounds(cuboid.dimensions);
    write_obstacle(cuboids_, slot, params.data(), SMALL_PARAM_STRIDE, bounds, pose);
    ++cuboids_.loaded;
    cuboids_.write_count();
    return register_obstacle(cuboid.name, ObstacleType::CUBOID, StorageKind::CUBOID,
                             slot, bounds);
  }

  ObstacleHandle add_capsule(const Capsule& capsule) {
    require_finite(capsule.base, "capsule base");
    require_finite(capsule.tip, "capsule tip");
    if (!finite(capsule.radius) || capsule.radius <= 0.0F) {
      throw std::invalid_argument("capsule radius must be finite and positive");
    }
    const auto pose = validate_pose(capsule.pose);
    return add_capsule_geometry(capsule.name, capsule.base, capsule.tip, capsule.radius,
                                ObstacleType::CAPSULE, pose);
  }

  ObstacleHandle add_sphere(const Sphere& sphere) {
    if (!finite(sphere.radius) || sphere.radius <= 0.0F) {
      throw std::invalid_argument("sphere radius must be finite and positive");
    }
    const auto pose = validate_pose(sphere.pose);
    const std::array<float, 3> center{0.0F, 0.0F, 0.0F};
    return add_capsule_geometry(sphere.name, center, center, sphere.radius,
                                ObstacleType::SPHERE, pose);
  }

  ObstacleHandle add_cylinder(const Cylinder& cylinder) {
    if (!finite(cylinder.radius) || cylinder.radius <= 0.0F ||
        !finite(cylinder.height) || cylinder.height <= 0.0F) {
      throw std::invalid_argument("cylinder radius and height must be finite and positive");
    }
    const auto pose = validate_pose(cylinder.pose);
    activate_device();
    ensure_add_allowed(cylinder.name, cylinders_.loaded, cylinders_.capacity, "cylinders");

    const int slot = cylinders_.loaded;
    const std::array<float, SMALL_PARAM_STRIDE> params{
        cylinder.radius, cylinder.height, 0.0F, 0.0F};
    const std::array<float, 3> dimensions{
        2.0F * cylinder.radius, 2.0F * cylinder.radius, cylinder.height};
    const auto bounds = centered_bounds(dimensions);
    write_obstacle(cylinders_, slot, params.data(), SMALL_PARAM_STRIDE, bounds, pose);
    ++cylinders_.loaded;
    cylinders_.write_count();
    return register_obstacle(cylinder.name, ObstacleType::CYLINDER,
                             StorageKind::CYLINDER, slot, bounds);
  }

  ObstacleHandle add_mesh(const Mesh& mesh) {
    const auto pose = validate_pose(mesh.pose);
    const auto bounds = validate_mesh(mesh);
    activate_device();
    ensure_add_allowed(mesh.name, meshes_.loaded, meshes_.capacity, "meshes");
#if defined(CUROBO_ROBOT_HAS_WARP_MESH)
    if (!load_mesh_backend()) {
      throw std::runtime_error(mesh_backend_error_);
    }

    const auto vertex_bytes = checked_bytes(mesh.vertices.size(), sizeof(mesh.vertices.front()));
    const auto index_bytes = checked_bytes(mesh.triangles.size(), sizeof(mesh.triangles.front()));
    auto resource = std::make_unique<MeshResource>(vertex_bytes, index_bytes);
    resource->vertices.upload(mesh.vertices.data(), vertex_bytes);
    resource->indices.upload(mesh.triangles.data(), index_bytes);
    CUcontext context = nullptr;
    CRR_CU_CHECK(cuCtxGetCurrent(&context));
    if (context == nullptr) {
      throw std::runtime_error("CUDA context is not active while creating a mesh");
    }
    resource->mesh_id = mesh_backend_->create_mesh(
        reinterpret_cast<void*>(context), resource->vertices.pointer(),
        static_cast<int>(mesh.vertices.size()), resource->indices.pointer(),
        static_cast<int>(mesh.triangles.size()));
    resource->backend = mesh_backend_.get();

    const int slot = meshes_.loaded;
    meshes_.mesh_ids.upload(&resource->mesh_id, sizeof(resource->mesh_id),
                            checked_bytes(static_cast<size_t>(slot), sizeof(uint64_t)));
    const std::array<float, SMALL_PARAM_STRIDE> dimensions{
        bounds.upper.at(0) - bounds.lower.at(0),
        bounds.upper.at(1) - bounds.lower.at(1),
        bounds.upper.at(2) - bounds.lower.at(2),
        0.0F,
    };
    write_row(meshes_.dimensions, slot, SMALL_PARAM_STRIDE, dimensions.data());
    write_pose_and_bounds(meshes_.inverse_poses, meshes_.world_bounds, slot, bounds, pose);
    write_enabled(meshes_.enabled, slot, true);
    const uint32_t common_layer = 1U;
    meshes_.obstacle_layers.upload(
        &common_layer, sizeof(common_layer),
        static_cast<size_t>(slot) * sizeof(common_layer));
    meshes_.resources.at(static_cast<size_t>(slot)) = std::move(resource);
    ++meshes_.loaded;
    meshes_.write_count();
    return register_obstacle(mesh.name, ObstacleType::MESH, StorageKind::MESH, slot,
                             bounds);
#else
    (void)pose;
    (void)bounds;
    throw std::runtime_error(
        "mesh support was not built; configure CMake with WARP_NATIVE_INCLUDE_DIR");
#endif
  }

  ObstacleHandle add_voxel_grid(const VoxelGrid& voxel_grid) {
    const auto grid_shape = validate_voxel_grid(voxel_grid);
    const auto pose = validate_pose(voxel_grid.pose);
    activate_device();
    ensure_add_allowed(voxel_grid.name, voxels_.loaded, voxels_.capacity, "voxel grids");

    const int slot = voxels_.loaded;
    const std::array<float, SMALL_PARAM_STRIDE> params{
        static_cast<float>(grid_shape.at(0)),
        static_cast<float>(grid_shape.at(1)),
        static_cast<float>(grid_shape.at(2)),
        voxel_grid.voxel_size,
    };
    const std::array<float, SMALL_PARAM_STRIDE> dimensions{
        voxel_grid.dimensions.at(0), voxel_grid.dimensions.at(1),
        voxel_grid.dimensions.at(2), 0.0F};
    write_row(voxels_.params, slot, SMALL_PARAM_STRIDE, params.data());
    write_row(voxels_.dimensions, slot, SMALL_PARAM_STRIDE, dimensions.data());
    const auto inverse_pose = inverse_pose_row(pose);
    write_row(voxels_.inverse_poses, slot, POSE_STRIDE, inverse_pose.data());
    write_enabled(voxels_.enabled, slot, true);
    const uint32_t common_layer = 1U;
    voxels_.obstacle_layers.upload(
        &common_layer, sizeof(common_layer),
        static_cast<size_t>(slot) * sizeof(common_layer));

    std::vector<uint16_t> half_esdf(static_cast<size_t>(voxels_.max_voxels_per_grid), 0);
    std::transform(voxel_grid.esdf.begin(), voxel_grid.esdf.end(), half_esdf.begin(),
                   float_to_half_bits);
    const auto feature_offset = checked_bytes(
        static_cast<size_t>(slot) * voxels_.max_voxels_per_grid, sizeof(uint16_t));
    voxels_.features.upload(half_esdf.data(),
                            checked_bytes(half_esdf.size(), sizeof(uint16_t)),
                            feature_offset);
    ++voxels_.loaded;
    voxels_.write_count();
    return register_obstacle(voxel_grid.name, ObstacleType::VOXEL_GRID,
                             StorageKind::VOXEL_GRID, slot, Bounds{},
                             voxel_grid.esdf.size());
  }

  uint64_t voxel_grid_cuda_address(ObstacleHandle handle) const {
    const auto& entry = find_entry(handle);
    if (entry.storage != StorageKind::VOXEL_GRID) {
      throw std::invalid_argument("obstacle is not a voxel grid");
    }
    const size_t element_offset =
        static_cast<size_t>(entry.slot) * voxels_.max_voxels_per_grid;
    return static_cast<uint64_t>(voxels_.features.device_pointer()) +
           checked_bytes(element_offset, sizeof(uint16_t));
  }

  void update_voxel_grid_cuda(ObstacleHandle handle, uint64_t source_esdf_fp16,
                              size_t voxel_count, uint64_t stream_handle) {
    activate_device();
    const auto& entry = find_entry(handle);
    if (entry.storage != StorageKind::VOXEL_GRID) {
      throw std::invalid_argument("obstacle is not a voxel grid");
    }
    if (source_esdf_fp16 == 0) {
      throw std::invalid_argument("source CUDA pointer is null");
    }
    if (voxel_count != entry.voxel_count) {
      throw std::invalid_argument("CUDA voxel ESDF size does not match grid shape");
    }
    const size_t offset = checked_bytes(
        static_cast<size_t>(entry.slot) * voxels_.max_voxels_per_grid,
        sizeof(uint16_t));
    voxels_.features.copy_from_device_async(
        reinterpret_cast<const void*>(source_esdf_fp16),
        checked_bytes(voxel_count, sizeof(uint16_t)), offset,
        reinterpret_cast<cudaStream_t>(stream_handle));
  }

  void update_obstacle_pose(ObstacleHandle handle, const Pose& requested_pose) {
    const auto pose = validate_pose(requested_pose);
    activate_device();
    const auto& entry = find_entry(handle);
    switch (entry.storage) {
      case StorageKind::CUBOID:
        write_pose_and_bounds(cuboids_.inverse_poses, cuboids_.world_bounds, entry.slot,
                              entry.local_bounds, pose);
        return;
      case StorageKind::CAPSULE:
        write_pose_and_bounds(capsules_.inverse_poses, capsules_.world_bounds, entry.slot,
                              entry.local_bounds, pose);
        return;
      case StorageKind::CYLINDER:
        write_pose_and_bounds(cylinders_.inverse_poses, cylinders_.world_bounds, entry.slot,
                              entry.local_bounds, pose);
        return;
      case StorageKind::MESH:
        write_pose_and_bounds(meshes_.inverse_poses, meshes_.world_bounds, entry.slot,
                              entry.local_bounds, pose);
        return;
      case StorageKind::VOXEL_GRID: {
        const auto inverse_pose = inverse_pose_row(pose);
        write_row(voxels_.inverse_poses, entry.slot, POSE_STRIDE, inverse_pose.data());
        return;
      }
    }
  }

  void set_obstacle_enabled(ObstacleHandle handle, bool enabled) {
    activate_device();
    const auto& entry = find_entry(handle);
    switch (entry.storage) {
      case StorageKind::CUBOID:
        write_enabled(cuboids_.enabled, entry.slot, enabled);
        return;
      case StorageKind::CAPSULE:
        write_enabled(capsules_.enabled, entry.slot, enabled);
        return;
      case StorageKind::CYLINDER:
        write_enabled(cylinders_.enabled, entry.slot, enabled);
        return;
      case StorageKind::MESH:
        write_enabled(meshes_.enabled, entry.slot, enabled);
        return;
      case StorageKind::VOXEL_GRID:
        write_enabled(voxels_.enabled, entry.slot, enabled);
        return;
    }
  }

  void set_obstacle_layers(ObstacleHandle handle, uint32_t layers) {
    activate_device();
    const auto& entry = find_entry(handle);
    DeviceBuffer* buffer = nullptr;
    switch (entry.storage) {
      case StorageKind::CUBOID:
        buffer = &cuboids_.obstacle_layers;
        break;
      case StorageKind::CAPSULE:
        buffer = &capsules_.obstacle_layers;
        break;
      case StorageKind::CYLINDER:
        buffer = &cylinders_.obstacle_layers;
        break;
      case StorageKind::MESH:
        buffer = &meshes_.obstacle_layers;
        break;
      case StorageKind::VOXEL_GRID:
        buffer = &voxels_.obstacle_layers;
        break;
    }
    buffer->upload(&layers, sizeof(layers),
                   static_cast<size_t>(entry.slot) * sizeof(layers));
  }

  void clear() {
    activate_device();
    cuboids_.loaded = 0;
    capsules_.loaded = 0;
    cylinders_.loaded = 0;
    meshes_.loaded = 0;
    voxels_.loaded = 0;
    cuboids_.enabled.clear();
    capsules_.enabled.clear();
    cylinders_.enabled.clear();
    meshes_.enabled.clear();
    meshes_.mesh_ids.clear();
    voxels_.enabled.clear();
    cuboids_.write_count();
    capsules_.write_count();
    cylinders_.write_count();
    meshes_.write_count();
    voxels_.write_count();
    meshes_.resources.clear();
    meshes_.resources.resize(static_cast<size_t>(meshes_.capacity));
    entries_.clear();
    names_.clear();
    obstacle_counts_.fill(0);
  }

 private:
  static SceneConfig validate_config(const SceneConfig& config) {
    if (config.device_ordinal < 0 || config.max_cuboids < 0 || config.max_capsules < 0 ||
        config.max_cylinders < 0 || config.max_meshes < 0 ||
        config.max_voxel_grids < 0 || config.max_voxels_per_grid < 0) {
      throw std::invalid_argument("scene device ordinal and capacities must be non-negative");
    }
    if (config.max_voxel_grids > 0 && config.max_voxels_per_grid == 0) {
      throw std::invalid_argument(
          "max_voxels_per_grid must be positive when voxel grids are enabled");
    }
    if (!finite(config.max_esdf_distance) || config.max_esdf_distance <= 0.0F) {
      throw std::invalid_argument("max_esdf_distance must be finite and positive");
    }
    CRR_CUDA_CHECK(cudaSetDevice(config.device_ordinal));
    return config;
  }

  void activate_device() const { CRR_CUDA_CHECK(cudaSetDevice(config_.device_ordinal)); }

  void ensure_add_allowed(const std::string& name, int loaded, int capacity,
                          const char* category) const {
    if (name.empty()) {
      throw std::invalid_argument("obstacle name must not be empty");
    }
    if (loaded >= capacity) {
      throw std::runtime_error(std::string("scene capacity is full for ") + category);
    }
    if (names_.find(name) != names_.end()) {
      throw std::runtime_error("scene obstacle name already exists: " + name);
    }
  }

  ObstacleHandle register_obstacle(const std::string& name, ObstacleType type,
                                   StorageKind storage, int slot, const Bounds& bounds,
                                   size_t voxel_count = 0) {
    const auto handle = next_obstacle_handle();
    entries_.emplace(handle, Entry{type, storage, slot, bounds, voxel_count});
    names_.insert(name);
    ++obstacle_counts_.at(static_cast<size_t>(type_index(type)));
    return ObstacleHandle(handle);
  }

  const Entry& find_entry(ObstacleHandle handle) const {
    if (!handle.valid()) {
      throw std::invalid_argument("obstacle handle is invalid");
    }
    const auto iterator = entries_.find(handle.value());
    if (iterator == entries_.end()) {
      throw std::invalid_argument("obstacle handle does not belong to this scene");
    }
    return iterator->second;
  }

  static void write_pose_and_bounds(DeviceBuffer& inverse_poses,
                                    DeviceBuffer& world_bounds, int slot,
                                    const Bounds& bounds, const ValidatedPose& pose) {
    const auto inverse_pose = inverse_pose_row(pose);
    const auto bounds_row = world_bounds_row(bounds, pose);
    write_row(inverse_poses, slot, POSE_STRIDE, inverse_pose.data());
    write_row(world_bounds, slot, BOUNDS_STRIDE, bounds_row.data());
  }

  static ObstacleDeviceView obstacle_view(const ObstacleStorage& storage,
                                          int param_stride) noexcept {
    return {
        storage.params.device_pointer(),
        storage.world_bounds.device_pointer(),
        storage.inverse_poses.device_pointer(),
        storage.enabled.device_pointer(),
        storage.obstacle_layers.device_pointer(),
        storage.count.device_pointer(),
        storage.capacity,
        param_stride,
    };
  }

  static void write_obstacle(ObstacleStorage& storage, int slot, const float* params,
                             int param_stride, const Bounds& bounds,
                             const ValidatedPose& pose) {
    write_row(storage.params, slot, param_stride, params);
    write_pose_and_bounds(storage.inverse_poses, storage.world_bounds, slot, bounds, pose);
    write_enabled(storage.enabled, slot, true);
    const uint32_t common_layer = 1U;
    storage.obstacle_layers.upload(
        &common_layer, sizeof(common_layer),
        static_cast<size_t>(slot) * sizeof(common_layer));
  }

  ObstacleHandle add_capsule_geometry(const std::string& name,
                                      const std::array<float, 3>& base,
                                      const std::array<float, 3>& tip, float radius,
                                      ObstacleType type, const ValidatedPose& pose) {
    activate_device();
    ensure_add_allowed(name, capsules_.loaded, capsules_.capacity, "capsules and spheres");
    std::array<float, 3> segment{};
    float length_squared = 0.0F;
    for (size_t axis = 0; axis < segment.size(); ++axis) {
      segment.at(axis) = tip.at(axis) - base.at(axis);
      length_squared += segment.at(axis) * segment.at(axis);
    }
    const float length = std::sqrt(length_squared);
    std::array<float, 3> axis{};
    if (length > CAPSULE_SEGMENT_EPSILON) {
      for (size_t index = 0; index < axis.size(); ++index) {
        axis.at(index) = segment.at(index) / length;
      }
    }
    const std::array<float, CAPSULE_PARAM_STRIDE> params{
        base.at(0), base.at(1), base.at(2), axis.at(0),
        axis.at(1), axis.at(2), length, radius};
    Bounds bounds;
    for (size_t index = 0; index < bounds.lower.size(); ++index) {
      bounds.lower.at(index) = std::min(base.at(index), tip.at(index)) - radius;
      bounds.upper.at(index) = std::max(base.at(index), tip.at(index)) + radius;
    }
    const int slot = capsules_.loaded;
    write_obstacle(capsules_, slot, params.data(), CAPSULE_PARAM_STRIDE, bounds, pose);
    ++capsules_.loaded;
    capsules_.write_count();
    return register_obstacle(name, type, StorageKind::CAPSULE, slot, bounds);
  }

  static Bounds validate_mesh(const Mesh& mesh) {
    if (mesh.vertices.empty()) {
      throw std::invalid_argument("mesh must contain at least one vertex");
    }
    if (mesh.triangles.empty()) {
      throw std::invalid_argument("mesh must contain at least one triangle");
    }
    if (mesh.vertices.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        mesh.triangles.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("mesh exceeds the supported index range");
    }
    Bounds bounds{mesh.vertices.front(), mesh.vertices.front()};
    for (const auto& vertex : mesh.vertices) {
      require_finite(vertex, "mesh vertices");
      for (size_t axis = 0; axis < vertex.size(); ++axis) {
        bounds.lower.at(axis) = std::min(bounds.lower.at(axis), vertex.at(axis));
        bounds.upper.at(axis) = std::max(bounds.upper.at(axis), vertex.at(axis));
      }
    }
    for (const auto& triangle : mesh.triangles) {
      for (const auto index : triangle) {
        if (index < 0 || static_cast<size_t>(index) >= mesh.vertices.size()) {
          throw std::invalid_argument("mesh triangle index is outside the vertex array");
        }
      }
    }
    return bounds;
  }

  std::array<int, 3> validate_voxel_grid(const VoxelGrid& voxel_grid) const {
    require_positive(voxel_grid.dimensions, "voxel grid dimensions");
    if (!finite(voxel_grid.voxel_size) || voxel_grid.voxel_size <= 0.0F) {
      throw std::invalid_argument("voxel size must be finite and positive");
    }
    std::array<int, 3> shape{};
    size_t voxel_count = 1;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      const auto rounded = std::nearbyint(voxel_grid.dimensions.at(axis) /
                                          voxel_grid.voxel_size);
      if (rounded <= 0.0F || rounded > static_cast<float>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("voxel grid shape is outside the supported range");
      }
      shape.at(axis) = static_cast<int>(rounded);
      if (voxel_count > std::numeric_limits<size_t>::max() /
                            static_cast<size_t>(shape.at(axis))) {
        throw std::invalid_argument("voxel grid shape is too large");
      }
      voxel_count *= static_cast<size_t>(shape.at(axis));
    }
    for (const auto value : voxel_grid.esdf) {
      if (!finite(value)) {
        throw std::invalid_argument("voxel ESDF must contain finite values");
      }
    }
    if (voxel_grid.esdf.size() != voxel_count) {
      throw std::invalid_argument("voxel ESDF size does not match the rounded grid shape");
    }
    if (voxel_count > static_cast<size_t>(config_.max_voxels_per_grid)) {
      throw std::invalid_argument("voxel ESDF exceeds max_voxels_per_grid");
    }
    return shape;
  }

#if defined(CUROBO_ROBOT_HAS_WARP_MESH)
  bool load_mesh_backend() const {
    if (mesh_backend_ != nullptr) {
      return true;
    }
    try {
      mesh_backend_ = std::make_unique<WarpMeshBackend>();
      mesh_backend_error_.clear();
      return true;
    } catch (const std::exception& error) {
      mesh_backend_error_ = error.what();
      return false;
    }
  }
#endif

  SceneConfig config_;
#if defined(CUROBO_ROBOT_HAS_WARP_MESH)
  mutable std::unique_ptr<WarpMeshBackend> mesh_backend_;
  mutable std::string mesh_backend_error_;
#endif
  ObstacleStorage cuboids_;
  ObstacleStorage capsules_;
  ObstacleStorage cylinders_;
  MeshStorage meshes_;
  VoxelStorage voxels_;
  std::unordered_map<uint64_t, Entry> entries_;
  std::unordered_set<std::string> names_;
  std::array<int, 6> obstacle_counts_{};
};

SceneRuntime::SceneRuntime(const SceneConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

SceneRuntime::~SceneRuntime() = default;

const SceneConfig& SceneRuntime::config() const { return impl_->config(); }

SceneDeviceView SceneRuntime::device_view() const { return impl_->device_view(); }

int SceneRuntime::obstacle_count(ObstacleType type) const {
  return impl_->obstacle_count(type);
}

bool SceneRuntime::mesh_support_available() const noexcept {
  return impl_->mesh_support_available();
}

ObstacleHandle SceneRuntime::add_cuboid(const Cuboid& cuboid) {
  return impl_->add_cuboid(cuboid);
}

ObstacleHandle SceneRuntime::add_capsule(const Capsule& capsule) {
  return impl_->add_capsule(capsule);
}

ObstacleHandle SceneRuntime::add_cylinder(const Cylinder& cylinder) {
  return impl_->add_cylinder(cylinder);
}

ObstacleHandle SceneRuntime::add_sphere(const Sphere& sphere) {
  return impl_->add_sphere(sphere);
}

ObstacleHandle SceneRuntime::add_mesh(const Mesh& mesh) {
  return impl_->add_mesh(mesh);
}

ObstacleHandle SceneRuntime::add_voxel_grid(const VoxelGrid& voxel_grid) {
  return impl_->add_voxel_grid(voxel_grid);
}

uint64_t SceneRuntime::voxel_grid_cuda_address(ObstacleHandle handle) const {
  return impl_->voxel_grid_cuda_address(handle);
}

void SceneRuntime::update_voxel_grid_cuda(ObstacleHandle handle,
                                          uint64_t source_esdf_fp16,
                                          std::size_t voxel_count,
                                          uint64_t stream) {
  impl_->update_voxel_grid_cuda(handle, source_esdf_fp16, voxel_count, stream);
}

void SceneRuntime::update_obstacle_pose(ObstacleHandle handle, const Pose& pose) {
  impl_->update_obstacle_pose(handle, pose);
}

void SceneRuntime::set_obstacle_enabled(ObstacleHandle handle, bool enabled) {
  impl_->set_obstacle_enabled(handle, enabled);
}

void SceneRuntime::set_obstacle_layers(ObstacleHandle handle, uint32_t layers) {
  impl_->set_obstacle_layers(handle, layers);
}

void SceneRuntime::clear() { impl_->clear(); }

}  // namespace curobo::robot::detail
