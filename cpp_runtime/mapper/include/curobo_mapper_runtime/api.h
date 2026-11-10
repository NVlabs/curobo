// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(CUROBO_MAPPER_RUNTIME_BUILD)
#define CUROBO_MAPPER_API __declspec(dllexport)
#else
#define CUROBO_MAPPER_API __declspec(dllimport)
#endif
#else
#define CUROBO_MAPPER_API __attribute__((visibility("default")))
#endif

typedef struct CuroboMapperRuntime CuroboMapperRuntime;

typedef struct CuroboMapperFrame {
  uint64_t depth_images;
  uint64_t rgb_images;
  uint64_t intrinsics;
  uint64_t cam_positions;
  uint64_t cam_quaternions;
  /* Optional uint8 [camera, height, width] compact instance slots. */
  uint64_t instance_image;
  uint64_t stream;
} CuroboMapperFrame;

typedef struct CuroboMapperFeatureFrame {
  CuroboMapperFrame frame;
  uint64_t feature_grid;
} CuroboMapperFeatureFrame;

typedef struct CuroboMapperSemanticFilter {
  /* Optional float32 CUDA vector. Null when target_class is non-negative. */
  uint64_t target;
  /* Direct categorical target, or -1 to derive it from target. */
  int target_class;
  float min_similarity;
  int use_min_similarity;
  int require_argmax;
  int exclude;
  /* Additionally exclude this compact instance slot (1..255), or 0 for none.
   * Works with inclusion or exclusion filters and requires per-voxel features/instances. Only the filtered
   * TSDF view changes; stored geometry and instance evidence are preserved. */
  int exclude_instance_slot;
} CuroboMapperSemanticFilter;

/* Borrowed read-only CUDA storage. Pointers remain stable for runtime lifetime. */
typedef struct CuroboMapperDeviceView {
  uint64_t block_grid_rgb;
  uint64_t block_features;
  uint64_t block_feature_weight;
  uint64_t block_instance_state;
  uint64_t block_coords;
  uint64_t block_to_hash_slot;
  uint64_t num_allocated;
  uint64_t esdf_dist_field;
  int max_blocks;
  int block_size;
  int block_voxels;
  int color_grid_size;
  int color_grid_voxels;
  int feature_dim;
  int feature_block_grid_size;
  int feature_grid_voxels;
  int instance_grid_size;
  int instance_grid_voxels;
  int grid_shape[3];
  int esdf_shape[3];
  float origin[3];
  float voxel_size;
  float esdf_voxel_size;
} CuroboMapperDeviceView;

typedef struct CuroboMapperTextureFrame {
  uint64_t rgb_images;
  uint64_t visibility_depths;
  uint64_t intrinsics;
  uint64_t cam_positions;
  uint64_t cam_quaternions;
  uint64_t stream;
  int batch_count;
} CuroboMapperTextureFrame;

enum {
  CUROBO_MAPPER_POSE_REFINER_SUCCESS = 0,
  CUROBO_MAPPER_POSE_REFINER_TOO_FEW_DEPTH_PIXELS = 1,
  CUROBO_MAPPER_POSE_REFINER_TOO_FEW_CORRESPONDENCES = 2,
  CUROBO_MAPPER_POSE_REFINER_LINEAR_SOLVE_FAILED = 3
};

typedef struct CuroboMapperPoseRefinerConfig {
  int n_points;
  int minimum_valid_depth_pixels;
  int max_iterations;
  int inner_iterations;
  int n_samples_per_ray;
  int tile_block_dim;
  float distance_threshold;
  float min_valid_ratio;
  float depth_minimum_distance;
  float depth_maximum_distance;
  float minimum_tsdf_weight;
  float lambda_initial;
  float lambda_factor;
  float lambda_min;
  float lambda_max;
  float rho_min;
} CuroboMapperPoseRefinerConfig;

typedef struct CuroboMapperPoseRefinerResult {
  float final_error;
  float final_lambda;
  int iterations;
  int valid_count;
  int accepted_steps;
  int status;
} CuroboMapperPoseRefinerResult;

enum {
  CUROBO_MAPPER_MESH_REQUIRED_VERTICES = 0,
  CUROBO_MAPPER_MESH_REQUIRED_TRIANGLES = 1,
  CUROBO_MAPPER_MESH_WRITTEN_VERTICES = 2,
  CUROBO_MAPPER_MESH_WRITTEN_TRIANGLES = 3,
  CUROBO_MAPPER_MESH_SURFACE_CUBES = 4,
  CUROBO_MAPPER_MESH_OVERFLOW_TRIANGLES = 5,
  CUROBO_MAPPER_MESH_COUNTER_COUNT = 6
};

enum {
  CUROBO_MAPPER_MESH_OUTPUT_VERTICES = 1,
  CUROBO_MAPPER_MESH_OUTPUT_TRIANGLES = 2,
  CUROBO_MAPPER_MESH_OUTPUT_NORMALS = 4,
  CUROBO_MAPPER_MESH_OUTPUT_COLORS = 8,
  CUROBO_MAPPER_MESH_OUTPUT_FACE_NORMALS = 16,
  CUROBO_MAPPER_MESH_OUTPUT_ALL = 15
};

CUROBO_MAPPER_API const char* curobo_mapper_last_error(void);

CUROBO_MAPPER_API int curobo_mapper_default_pose_refiner_config(
    CuroboMapperPoseRefinerConfig* config);

CUROBO_MAPPER_API int curobo_mapper_create(
    const char* artifact_dir,
    int device_ordinal,
    CuroboMapperRuntime** runtime);

CUROBO_MAPPER_API void curobo_mapper_destroy(CuroboMapperRuntime* runtime);

CUROBO_MAPPER_API int curobo_mapper_reset(
    CuroboMapperRuntime* runtime,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_clear_instance_slot(
    CuroboMapperRuntime* runtime,
    int instance_slot,
    uint64_t stream);

/*
 * Remove dynamic TSDF/RGB/features owned by an instance, then clear its slot.
 * semantic_class_mask == 0 removes directly owned voxels. Otherwise matching
 * feature-argmax voxels are removed from blocks near the instance.
 */
CUROBO_MAPPER_API int curobo_mapper_remove_instance_geometry(
    CuroboMapperRuntime* runtime,
    int instance_slot,
    uint64_t semantic_class_mask,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_integrate(
    CuroboMapperRuntime* runtime,
    const CuroboMapperFrame* frame);

CUROBO_MAPPER_API int curobo_mapper_integrate_features(
    CuroboMapperRuntime* runtime,
    const CuroboMapperFeatureFrame* frame);

CUROBO_MAPPER_API int curobo_mapper_extract_occupied(
    CuroboMapperRuntime* runtime,
    uint64_t out_centers,
    uint64_t out_pool_indices,
    uint64_t out_count,
    int capacity,
    int surface_only,
    float sdf_threshold,
    uint64_t stream,
    int* required_count,
    int* written_count);

/*
 * Asynchronously sample observed TSDF zero-crossing squares, pruning |SDF|>=h/2.
 * CUDA outputs: positions float32[budget,3], source_voxels int32[budget,2]
 * (pool,local), count int32[1]. Only rows below count are valid; no refill.
 * instance_slot=-1 disables instance filtering. voxel_selection=0 disables the
 * optional CUDA uint8[max_blocks,block_voxels] selection. Filters intersect.
 * Geometry uses the full TSDF and renormalizes observed interpolation corners.
 * semantic_filter uses its existing target/target_class contract; instance
 * exclusion is not supported here. Warm up each budget before graph capture.
 * Calls/map writes must be serialized. No host count reads or TSDF mutations.
 */
CUROBO_MAPPER_API int curobo_mapper_extract_occupied_points(
    CuroboMapperRuntime* runtime, uint64_t out_positions,
    uint64_t out_source_voxels, uint64_t out_count, int sample_budget,
    int instance_slot, uint64_t voxel_selection,
    const CuroboMapperSemanticFilter* semantic_filter, int seed, uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_extract_occupied_semantic(
    CuroboMapperRuntime* runtime,
    uint64_t out_centers,
    uint64_t out_pool_indices,
    uint64_t out_count,
    int capacity,
    int surface_only,
    float sdf_threshold,
    const CuroboMapperSemanticFilter* semantic_filter,
    uint64_t stream,
    int* required_count,
    int* written_count);

/** Lifetime count of successfully instantiated ESDF graphs (host-only). */
CUROBO_MAPPER_API int curobo_mapper_get_esdf_graph_capture_count(CuroboMapperRuntime* runtime,
                                                                 uint64_t* count);

CUROBO_MAPPER_API int curobo_mapper_compute_esdf(
    CuroboMapperRuntime* runtime,
    uint64_t out_dist_field,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_compute_esdf_semantic(
    CuroboMapperRuntime* runtime,
    uint64_t out_dist_field,
    const CuroboMapperSemanticFilter* semantic_filter,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_render_depth(
    CuroboMapperRuntime* runtime,
    uint64_t intrinsics,
    uint64_t cam_positions,
    uint64_t cam_quaternions,
    uint64_t out_hit_points,
    uint64_t out_hit_normals,
    uint64_t out_hit_depths,
    uint64_t out_hit_mask,
    int camera_count,
    int image_height,
    int image_width,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_render_depth_semantic(
    CuroboMapperRuntime* runtime,
    uint64_t intrinsics,
    uint64_t cam_positions,
    uint64_t cam_quaternions,
    uint64_t out_hit_points,
    uint64_t out_hit_normals,
    uint64_t out_hit_depths,
    uint64_t out_hit_mask,
    int camera_count,
    int image_height,
    int image_width,
    const CuroboMapperSemanticFilter* semantic_filter,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_render_color(
    CuroboMapperRuntime* runtime,
    uint64_t intrinsics,
    uint64_t cam_positions,
    uint64_t cam_quaternions,
    uint64_t out_hit_points,
    uint64_t out_hit_normals,
    uint64_t out_hit_colors,
    uint64_t out_hit_depths,
    uint64_t out_hit_mask,
    int camera_count,
    int image_height,
    int image_width,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_render_color_semantic(
    CuroboMapperRuntime* runtime,
    uint64_t intrinsics,
    uint64_t cam_positions,
    uint64_t cam_quaternions,
    uint64_t out_hit_points,
    uint64_t out_hit_normals,
    uint64_t out_hit_colors,
    uint64_t out_hit_depths,
    uint64_t out_hit_mask,
    int camera_count,
    int image_height,
    int image_width,
    const CuroboMapperSemanticFilter* semantic_filter,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_render_features(
    CuroboMapperRuntime* runtime,
    uint64_t intrinsics,
    uint64_t cam_positions,
    uint64_t cam_quaternions,
    uint64_t out_hit_normals,
    uint64_t out_hit_depths,
    uint64_t out_hit_mask,
    uint64_t out_feature_valid,
    uint64_t out_feature_block_indices,
    uint64_t out_feature_node_indices,
    int camera_count,
    int image_height,
    int image_width,
    int patch_top,
    int patch_left,
    int patch_height,
    int patch_width,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_render_features_semantic(
    CuroboMapperRuntime* runtime,
    uint64_t intrinsics,
    uint64_t cam_positions,
    uint64_t cam_quaternions,
    uint64_t out_hit_normals,
    uint64_t out_hit_depths,
    uint64_t out_hit_mask,
    uint64_t out_feature_valid,
    uint64_t out_feature_block_indices,
    uint64_t out_feature_node_indices,
    int camera_count,
    int image_height,
    int image_width,
    int patch_top,
    int patch_left,
    int patch_height,
    int patch_width,
    const CuroboMapperSemanticFilter* semantic_filter,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_gather_features(
    CuroboMapperRuntime* runtime,
    uint64_t feature_valid,
    uint64_t feature_block_indices,
    uint64_t feature_node_indices,
    uint64_t out_features,
    int camera_count,
    int image_height,
    int image_width,
    int channel_start,
    int channel_count,
    int output_float16,
    uint64_t stream);

CUROBO_MAPPER_API int curobo_mapper_extract_mesh(
    CuroboMapperRuntime* runtime,
    uint64_t out_vertices,
    uint64_t out_normals,
    uint64_t out_colors,
    int vertex_capacity,
    float level,
    int surface_only,
    int refine_iterations,
    uint64_t stream,
    int* required_vertices,
    int* required_triangles,
    int* written_vertices,
    int* written_triangles);

CUROBO_MAPPER_API int curobo_mapper_extract_mesh_semantic(
    CuroboMapperRuntime* runtime,
    uint64_t out_vertices,
    uint64_t out_normals,
    uint64_t out_colors,
    int vertex_capacity,
    float level,
    int surface_only,
    int refine_iterations,
    const CuroboMapperSemanticFilter* semantic_filter,
    uint64_t stream,
    int* required_vertices,
    int* required_triangles,
    int* written_vertices,
    int* written_triangles);

CUROBO_MAPPER_API int curobo_mapper_project_mesh_texture(
    CuroboMapperRuntime* runtime,
    const CuroboMapperTextureFrame* frame,
    uint64_t vertices,
    int vertex_count,
    uint64_t out_uvs,
    uint64_t out_colors,
    uint64_t out_texture_atlas,
    uint64_t texture_atlas_capacity_bytes,
    float texture_depth_tolerance_m,
    int* texture_atlas_height,
    int* texture_atlas_width);

CUROBO_MAPPER_API int curobo_mapper_refine_pose(
    CuroboMapperRuntime* runtime,
    uint64_t depth_image,
    uint64_t intrinsics,
    uint64_t initial_position,
    uint64_t initial_quaternion,
    uint64_t out_position,
    uint64_t out_quaternion,
    const CuroboMapperPoseRefinerConfig* config,
    uint64_t stream,
    CuroboMapperPoseRefinerResult* result);

CUROBO_MAPPER_API int curobo_mapper_num_allocated(
    CuroboMapperRuntime* runtime,
    uint64_t stream,
    int* num_allocated);

CUROBO_MAPPER_API int curobo_mapper_get_device_view(
    CuroboMapperRuntime* runtime,
    CuroboMapperDeviceView* view);

CUROBO_MAPPER_API int curobo_mapper_config_int(
    CuroboMapperRuntime* runtime,
    const char* name,
    int* value);

CUROBO_MAPPER_API int curobo_mapper_config_float(
    CuroboMapperRuntime* runtime,
    const char* name,
    float* value);

// Load a static TSDF seed into an empty mapper and restore it on reset.
CUROBO_MAPPER_API int curobo_mapper_load_static_seed(CuroboMapperRuntime *runtime, const char *path,
                                                     uint64_t stream);

// Update the two fixed-shape TSDF parameters that are safe to tune while the
// runtime is live. The decay applies on the next integration; the minimum
// weight applies to extraction, rendering, and ESDF site generation.
CUROBO_MAPPER_API int curobo_mapper_set_tsdf_runtime_parameters(
    CuroboMapperRuntime* runtime,
    float decay_factor,
    float minimum_tsdf_weight);

#ifdef __cplusplus
}
#endif
