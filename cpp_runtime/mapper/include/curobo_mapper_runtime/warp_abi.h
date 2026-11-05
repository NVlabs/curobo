// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

// Host code includes CUDA driver/runtime headers in neighboring translation
// units. Defining this keeps Warp's tile support from declaring its own
// fallback float4 type, which conflicts with CUDA's vector_types.h.
#ifndef WP_ENABLE_CUDA
#define WP_ENABLE_CUDA 1
#endif

#include "builtin.h"

namespace curobo_mapper_runtime {

template <typename T>
inline wp::array_t<T> array1(uint64_t ptr, int dim0) {
  return wp::array_t<T>(reinterpret_cast<T*>(ptr), dim0);
}

template <typename T>
inline wp::array_t<T> array2(uint64_t ptr, int dim0, int dim1) {
  return wp::array_t<T>(reinterpret_cast<T*>(ptr), dim0, dim1);
}

template <typename T>
inline wp::array_t<T> array3(uint64_t ptr, int dim0, int dim1, int dim2) {
  return wp::array_t<T>(reinterpret_cast<T*>(ptr), dim0, dim1, dim2);
}

template <typename T>
inline wp::array_t<T> array4(uint64_t ptr, int dim0, int dim1, int dim2, int dim3) {
  return wp::array_t<T>(reinterpret_cast<T*>(ptr), dim0, dim1, dim2, dim3);
}

inline wp::launch_bounds_t<1> launchBounds1(int dim0) {
  wp::launch_bounds_t<1> bounds{};
  bounds.shape[0] = dim0;
  bounds.size = static_cast<size_t>(dim0);
  bounds.coord_mult = 1;
  return bounds;
}

inline wp::launch_bounds_t<2> launchBounds2(int dim0, int dim1) {
  wp::launch_bounds_t<2> bounds{};
  bounds.shape[0] = dim0;
  bounds.shape[1] = dim1;
  bounds.size = static_cast<size_t>(dim0) * static_cast<size_t>(dim1);
  bounds.coord_mult = 1;
  return bounds;
}

inline wp::launch_bounds_t<3> launchBounds3(int dim0, int dim1, int dim2) {
  wp::launch_bounds_t<3> bounds{};
  bounds.shape[0] = dim0;
  bounds.shape[1] = dim1;
  bounds.shape[2] = dim2;
  bounds.size = static_cast<size_t>(dim0) * static_cast<size_t>(dim1) *
                static_cast<size_t>(dim2);
  bounds.coord_mult = 1;
  return bounds;
}

inline wp::launch_bounds_t<4> launchBounds4(int dim0, int dim1, int dim2,
                                            int dim3) {
  wp::launch_bounds_t<4> bounds{};
  bounds.shape[0] = dim0;
  bounds.shape[1] = dim1;
  bounds.shape[2] = dim2;
  bounds.shape[3] = dim3;
  bounds.size = static_cast<size_t>(dim0) * static_cast<size_t>(dim1) *
                static_cast<size_t>(dim2) * static_cast<size_t>(dim3);
  bounds.coord_mult = 1;
  return bounds;
}

struct BlockSparseTSDFWarp {
  wp::array_t<wp::int64> hash_table;
  wp::array_t<wp::float16> block_data;
  wp::array_t<wp::float16> block_grid_rgb;
  wp::array_t<wp::float16> block_features;
  wp::array_t<wp::float16> block_feature_weight;
  wp::array_t<wp::int16> block_instance_state;
  wp::array_t<wp::float16> static_block_data;
  bool has_dynamic;
  bool has_static;
  bool has_features;
  bool has_instances;
  wp::int32 feature_dim;
  wp::int32 color_grid_size;
  wp::int32 feature_block_grid_size;
  wp::int32 instance_grid_size;
  wp::array_t<wp::int32> block_coords;
  wp::array_t<wp::int32> block_to_hash_slot;
  wp::array_t<wp::int32> free_list;
  wp::array_t<wp::int32> free_count;
  wp::array_t<wp::int32> num_allocated;
  wp::array_t<wp::int32> allocation_failures;
  wp::array_t<wp::float32> block_sums;
  wp::array_t<wp::int32> static_block_sums;
  wp::array_t<wp::int32> new_blocks;
  wp::array_t<wp::int32> new_block_count;
  wp::array_t<wp::int32> recycle_count;
  wp::vec_t<3, wp::float32> origin;
  wp::float32 voxel_size;
  wp::int32 hash_capacity;
  wp::int32 max_blocks;
  wp::float32 truncation_distance;
  wp::int32 block_size;
  wp::int32 grid_W;
  wp::int32 grid_H;
  wp::int32 grid_D;
};

struct PreparedSemanticFilterWarp {
  wp::array_t<wp::uint8> node_match;
  wp::array_t<wp::uint8> block_has_keep;
  bool exclude;
};

}  // namespace curobo_mapper_runtime
