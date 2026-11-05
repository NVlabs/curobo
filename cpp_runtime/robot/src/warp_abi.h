// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace curobo::robot::detail {

struct WarpLaunchBounds1 {
  int shape[1];
  size_t size;
  size_t coord_mult;
};

struct WarpLaunchBounds2 {
  int shape[2];
  size_t size;
  size_t coord_mult;
};

struct WarpShape {
  int dimensions[4];
};

struct WarpArray {
  void* data;
  void* gradient;
  WarpShape shape;
  int strides[4];
  uint16_t dimensions;
  uint16_t flags;
};

inline WarpArray warp_array_1d(uint64_t pointer, int count, int element_bytes) {
  return WarpArray{
      reinterpret_cast<void*>(pointer),
      nullptr,
      {{count, 0, 0, 0}},
      {element_bytes, 0, 0, 0},
      1,
      0,
  };
}

inline WarpArray warp_array_2d(uint64_t pointer, int rows, int columns,
                               int element_bytes) {
  return WarpArray{
      reinterpret_cast<void*>(pointer),
      nullptr,
      {{rows, columns, 0, 0}},
      {columns * element_bytes, element_bytes, 0, 0},
      2,
      0,
  };
}

inline WarpArray warp_array_3d(uint64_t pointer, int first, int second, int third,
                               int element_bytes) {
  return WarpArray{
      reinterpret_cast<void*>(pointer),
      nullptr,
      {{first, second, third, 0}},
      {second * third * element_bytes, third * element_bytes, element_bytes, 0},
      3,
      0,
  };
}

inline WarpArray warp_array_4d(uint64_t pointer, int first, int second,
                               int third, int fourth, int element_bytes) {
  return WarpArray{
      reinterpret_cast<void*>(pointer),
      nullptr,
      {{first, second, third, fourth}},
      {second * third * fourth * element_bytes,
       third * fourth * element_bytes, fourth * element_bytes, element_bytes},
      4,
      0,
  };
}

inline WarpLaunchBounds1 warp_bounds_1d(int count) {
  return WarpLaunchBounds1{{count}, static_cast<size_t>(count), 1};
}

inline WarpLaunchBounds1 warp_tiled_bounds_1d(int count, int block_size) {
  return WarpLaunchBounds1{
      {count},
      static_cast<size_t>(count) * static_cast<size_t>(block_size),
      static_cast<size_t>(block_size),
  };
}

inline WarpLaunchBounds2 warp_bounds_2d(int first, int second) {
  return WarpLaunchBounds2{
      {first, second},
      static_cast<size_t>(first) * static_cast<size_t>(second),
      1,
  };
}

inline WarpLaunchBounds2 warp_tiled_bounds_2d(int count, int block_size) {
  return WarpLaunchBounds2{
      {count, block_size},
      static_cast<size_t>(count) * static_cast<size_t>(block_size),
      1,
  };
}

struct CuboidDataWarp {
  WarpArray dimensions;
  WarpArray world_bounds;
  WarpArray inverse_poses;
  WarpArray enabled;
  WarpArray count;
  int32_t capacity;
  int32_t num_scenes;
};

struct CapsuleDataWarp {
  WarpArray params;
  WarpArray world_bounds;
  WarpArray inverse_poses;
  WarpArray enabled;
  WarpArray count;
  int32_t capacity;
  int32_t num_scenes;
};

using CylinderDataWarp = CapsuleDataWarp;

struct MeshDataWarp {
  WarpArray mesh_ids;
  WarpArray dimensions;
  WarpArray world_bounds;
  WarpArray inverse_poses;
  WarpArray enabled;
  WarpArray count;
  int32_t capacity;
  int32_t num_scenes;
  float max_distance;
};

struct VoxelDataWarp {
  WarpArray params;
  WarpArray dimensions;
  WarpArray inverse_poses;
  WarpArray enabled;
  WarpArray features;
  WarpArray count;
  int32_t voxels_per_grid;
  int32_t capacity;
  int32_t num_scenes;
  float max_distance;
};

}  // namespace curobo::robot::detail
