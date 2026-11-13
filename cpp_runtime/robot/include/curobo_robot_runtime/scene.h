// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace curobo::robot {

namespace detail {
class SceneRuntime;
class IkRuntime;
class TrajectoryOptimizerRuntime;
class RrtRuntime;
}

/**
 * A rigid pose in the world frame.
 *
 * Quaternions use wxyz order and are consumed as supplied to match the Python
 * runtime. Callers must provide a finite, non-zero, unit quaternion.
 */
struct Pose {
  std::array<float, 3> position{0.0F, 0.0F, 0.0F};
  std::array<float, 4> quaternion{1.0F, 0.0F, 0.0F, 0.0F};
};

/** Fixed GPU capacities for one deployment scene. */
struct SceneConfig {
  int device_ordinal{0};
  int max_cuboids{64};
  int max_capsules{64};
  int max_cylinders{32};
  int max_meshes{16};
  int max_voxel_grids{0};
  int max_voxels_per_grid{0};
  float max_esdf_distance{100.0F};
};

/** Public obstacle categories. Spheres use capsule storage internally. */
enum class ObstacleType {
  CUBOID,
  CAPSULE,
  CYLINDER,
  SPHERE,
  MESH,
  VOXEL_GRID,
};

/** Opaque identifier for an obstacle owned by a Scene. */
class ObstacleHandle final {
 public:
  ObstacleHandle() = default;

  bool valid() const noexcept { return value_ != 0; }
  uint64_t value() const noexcept { return value_; }

  friend bool operator==(ObstacleHandle left, ObstacleHandle right) noexcept {
    return left.value_ == right.value_;
  }

  friend bool operator!=(ObstacleHandle left, ObstacleHandle right) noexcept {
    return !(left == right);
  }

 private:
  explicit ObstacleHandle(uint64_t value) : value_(value) {}

  uint64_t value_{0};

  friend class Scene;
  friend class detail::SceneRuntime;
};

/** Axis-aligned box geometry expressed in its local frame. */
struct Cuboid {
  std::string name;
  Pose pose;
  std::array<float, 3> dimensions{1.0F, 1.0F, 1.0F};
};

/** Capsule geometry expressed in its local frame. */
struct Capsule {
  std::string name;
  Pose pose;
  std::array<float, 3> base{0.0F, 0.0F, 0.0F};
  std::array<float, 3> tip{0.0F, 0.0F, 0.0F};
  float radius{0.0F};
};

/** Z-axis cylinder geometry expressed in its local frame. */
struct Cylinder {
  std::string name;
  Pose pose;
  float radius{0.0F};
  float height{0.0F};
};

/** Sphere geometry centered at the obstacle's local origin. */
struct Sphere {
  std::string name;
  Pose pose;
  float radius{0.0F};
};

/** Indexed triangle mesh geometry expressed in its local frame. */
struct Mesh {
  std::string name;
  Pose pose;
  std::vector<std::array<float, 3>> vertices;
  std::vector<std::array<int32_t, 3>> triangles;
};

/** Dense ESDF grid centered at the obstacle's local origin. */
struct VoxelGrid {
  std::string name;
  Pose pose;
  std::array<float, 3> dimensions{0.0F, 0.0F, 0.0F};
  float voxel_size{0.0F};
  /** C-order values for shape dimensions / voxel_size, with z varying fastest. */
  std::vector<float> esdf;
};

/**
 * Fixed-capacity obstacle storage for one deployment scene.
 *
 * Scene mutations synchronously update stable CUDA buffers. Construct and
 * mutate the scene outside CUDA graph capture; solver calls may then reuse the
 * same buffers across graph replays. Query batches all collide with this one
 * scene, so the API intentionally has no environment index.
 */
class Scene final {
 public:
  explicit Scene(const SceneConfig& config = {});
  ~Scene();

  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;
  Scene(Scene&& other) noexcept;
  Scene& operator=(Scene&& other) noexcept;

  /** Return the immutable capacities and CUDA device selected at construction. */
  const SceneConfig& config() const;

  /** Return the number of loaded obstacles in a public category. */
  int obstacle_count(ObstacleType type) const;

  /** Return whether this build can create mesh acceleration structures. */
  bool mesh_support_available() const noexcept;

  /** Add a cuboid and return its scene-local handle. */
  ObstacleHandle add_cuboid(const Cuboid& cuboid);

  /** Add a capsule and return its scene-local handle. */
  ObstacleHandle add_capsule(const Capsule& capsule);

  /** Add a cylinder and return its scene-local handle. */
  ObstacleHandle add_cylinder(const Cylinder& cylinder);

  /** Add a sphere and return its scene-local handle. */
  ObstacleHandle add_sphere(const Sphere& sphere);

  /** Upload a triangle mesh, construct its BVH, and return its handle. */
  ObstacleHandle add_mesh(const Mesh& mesh);

  /** Add a dense ESDF voxel grid and return its scene-local handle. */
  ObstacleHandle add_voxel_grid(const VoxelGrid& voxel_grid);

  /**
   * Return the stable FP16 CUDA destination for one voxel grid.
   *
   * Fixed-shape GPU producers may write directly to this address, including
   * from a CUDA graph, instead of calling update_voxel_grid_cuda(). The Scene
   * retains ownership and the address is valid for the Scene lifetime.
   */
  uint64_t voxel_grid_cuda_address(ObstacleHandle handle) const;

  /**
   * Replace a voxel grid from contiguous FP16 CUDA storage.
   *
   * The copy is enqueued on `stream` and the destination address owned by
   * Scene remains stable, including across CUDA graph replays by consumers.
   */
  void update_voxel_grid_cuda(ObstacleHandle handle,
                              uint64_t source_esdf_fp16,
                              std::size_t voxel_count,
                              uint64_t stream = 0);

  /** Update an obstacle's world pose without reallocating scene metadata. */
  void update_obstacle_pose(ObstacleHandle handle, const Pose& pose);

  /** Enable or disable an obstacle without changing its loaded count. */
  void set_obstacle_enabled(ObstacleHandle handle, bool enabled);
  /** Set unsigned layer membership in stable scene-owned GPU storage. */
  void set_obstacle_layers(ObstacleHandle handle, uint32_t layers);

  /** Remove every obstacle while retaining fixed-capacity metadata buffers. */
  void clear();

 private:
  std::unique_ptr<detail::SceneRuntime> runtime_;

  friend class detail::IkRuntime;
  friend class detail::TrajectoryOptimizerRuntime;
  friend class detail::RrtRuntime;
};

}  // namespace curobo::robot
