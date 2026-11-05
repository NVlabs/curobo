// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#include "mapper_runtime.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

using curobo_mapper_runtime::MapperRuntimeImpl;

struct CuroboMapperRuntime {
  std::unique_ptr<MapperRuntimeImpl> impl;
};

namespace {

thread_local std::string g_last_error;

int fail(const std::exception& exc) {
  g_last_error = exc.what();
  return -1;
}

int failMessage(const char* message) {
  g_last_error = message;
  return -1;
}

template <typename Fn>
int wrap(Fn&& fn) {
  try {
    fn();
    g_last_error.clear();
    return 0;
  } catch (const std::exception& exc) {
    return fail(exc);
  } catch (...) {
    return failMessage("unknown C++ exception");
  }
}

MapperRuntimeImpl& checked(CuroboMapperRuntime* runtime) {
  if (runtime == nullptr || runtime->impl == nullptr) {
    throw std::runtime_error("CuroboMapperRuntime is null");
  }
  return *runtime->impl;
}

}  // namespace

extern "C" {
int curobo_mapper_load_static_seed(CuroboMapperRuntime *runtime, const char *path,
                                   uint64_t stream) {
  return wrap([&] {
    if (!path) throw std::invalid_argument("static seed path is null");
    checked(runtime).loadStaticSeed(path, stream);
  });
}

const char* curobo_mapper_last_error(void) {
  return g_last_error.c_str();
}

int curobo_mapper_default_pose_refiner_config(
    CuroboMapperPoseRefinerConfig* config) {
  return wrap([&] {
    if (config == nullptr) {
      throw std::runtime_error("pose refiner config output pointer is null");
    }
    *config = curobo_mapper_runtime::defaultPoseRefinerConfig();
  });
}

int curobo_mapper_create(const char* artifact_dir, int device_ordinal,
                         CuroboMapperRuntime** runtime) {
  if (runtime == nullptr) {
    return failMessage("runtime output pointer is null");
  }
  *runtime = nullptr;
  return wrap([&] {
    if (artifact_dir == nullptr) {
      throw std::runtime_error("artifact_dir is null");
    }
    auto holder = std::make_unique<CuroboMapperRuntime>();
    holder->impl = std::make_unique<MapperRuntimeImpl>(artifact_dir, device_ordinal);
    *runtime = holder.release();
  });
}

void curobo_mapper_destroy(CuroboMapperRuntime* runtime) {
  delete runtime;
}

int curobo_mapper_reset(CuroboMapperRuntime* runtime, uint64_t stream) {
  return wrap([&] { checked(runtime).reset(stream); });
}

int curobo_mapper_clear_instance_slot(CuroboMapperRuntime* runtime,
                                      int instance_slot,
                                      uint64_t stream) {
  return wrap(
      [&] { checked(runtime).clearInstanceSlot(instance_slot, stream); });
}

int curobo_mapper_remove_instance_geometry(CuroboMapperRuntime* runtime,
                                           int instance_slot,
                                           uint64_t semantic_class_mask,
                                           uint64_t stream) {
  return wrap([&] {
    checked(runtime).removeInstanceGeometry(instance_slot,
                                            semantic_class_mask, stream);
  });
}

int curobo_mapper_integrate(CuroboMapperRuntime* runtime,
                            const CuroboMapperFrame* frame) {
  return wrap([&] {
    if (frame == nullptr) {
      throw std::runtime_error("frame is null");
    }
    checked(runtime).integrate(*frame);
  });
}

int curobo_mapper_integrate_features(CuroboMapperRuntime* runtime,
                                     const CuroboMapperFeatureFrame* frame) {
  return wrap([&] {
    if (frame == nullptr) {
      throw std::runtime_error("feature frame is null");
    }
    checked(runtime).integrate(frame->frame, frame->feature_grid);
  });
}

int curobo_mapper_extract_occupied_points(
    CuroboMapperRuntime* runtime, uint64_t out_positions, uint64_t out_source_voxels,
    uint64_t out_count, int sample_budget, int instance_slot, uint64_t voxel_selection,
    const CuroboMapperSemanticFilter* semantic_filter, int seed, uint64_t stream) {
  return wrap([&] {
    checked(runtime).extract_occupied_points(out_positions, out_source_voxels, out_count,
        sample_budget, instance_slot, voxel_selection, semantic_filter, seed, stream);
  });
}

int curobo_mapper_extract_occupied(CuroboMapperRuntime* runtime,
                                   uint64_t out_centers,
                                   uint64_t out_pool_indices,
                                   uint64_t out_count,
                                   int capacity,
                                   int surface_only,
                                   float sdf_threshold,
                                   uint64_t stream,
                                   int* required_count,
                                   int* written_count) {
  return wrap([&] {
    checked(runtime).extractOccupied(out_centers, out_pool_indices, out_count,
                                     capacity, surface_only != 0, sdf_threshold,
                                     stream, required_count, written_count);
  });
}

int curobo_mapper_extract_occupied_semantic(
    CuroboMapperRuntime* runtime, uint64_t out_centers,
    uint64_t out_pool_indices, uint64_t out_count, int capacity,
    int surface_only, float sdf_threshold,
    const CuroboMapperSemanticFilter* semantic_filter, uint64_t stream,
    int* required_count, int* written_count) {
  return wrap([&] {
    if (semantic_filter == nullptr) {
      throw std::runtime_error("semantic_filter is null");
    }
    checked(runtime).extractOccupied(
        out_centers, out_pool_indices, out_count, capacity, surface_only != 0,
        sdf_threshold, stream, required_count, written_count, semantic_filter);
  });
}

int curobo_mapper_get_esdf_graph_capture_count(CuroboMapperRuntime* runtime, uint64_t* count) {
  return wrap([&] {
    if (count == nullptr) throw std::invalid_argument("graph count pointer is null");
    *count = checked(runtime).esdfGraphCaptureCount();
  });
}

int curobo_mapper_compute_esdf(CuroboMapperRuntime* runtime,
                               uint64_t out_dist_field,
                               uint64_t stream) {
  return wrap([&] { checked(runtime).computeEsdf(out_dist_field, stream); });
}

int curobo_mapper_compute_esdf_semantic(
    CuroboMapperRuntime* runtime, uint64_t out_dist_field,
    const CuroboMapperSemanticFilter* semantic_filter, uint64_t stream) {
  return wrap([&] {
    if (semantic_filter == nullptr) {
      throw std::runtime_error("semantic_filter is null");
    }
    checked(runtime).computeEsdf(out_dist_field, stream, semantic_filter);
  });
}

int curobo_mapper_render_depth(CuroboMapperRuntime* runtime,
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
                               uint64_t stream) {
  return wrap([&] {
    checked(runtime).renderDepth(intrinsics, cam_positions, cam_quaternions,
                                 out_hit_points, out_hit_normals, out_hit_depths,
                                 out_hit_mask, camera_count, image_height,
                                 image_width, stream);
  });
}

int curobo_mapper_render_depth_semantic(
    CuroboMapperRuntime* runtime, uint64_t intrinsics,
    uint64_t cam_positions, uint64_t cam_quaternions, uint64_t out_hit_points,
    uint64_t out_hit_normals, uint64_t out_hit_depths, uint64_t out_hit_mask,
    int camera_count, int image_height, int image_width,
    const CuroboMapperSemanticFilter* semantic_filter, uint64_t stream) {
  return wrap([&] {
    if (semantic_filter == nullptr) {
      throw std::runtime_error("semantic_filter is null");
    }
    checked(runtime).renderDepth(
        intrinsics, cam_positions, cam_quaternions, out_hit_points,
        out_hit_normals, out_hit_depths, out_hit_mask, camera_count,
        image_height, image_width, stream, semantic_filter);
  });
}

int curobo_mapper_render_color(CuroboMapperRuntime* runtime,
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
                               uint64_t stream) {
  return wrap([&] {
    checked(runtime).renderColor(intrinsics, cam_positions, cam_quaternions,
                                 out_hit_points, out_hit_normals, out_hit_colors,
                                 out_hit_depths, out_hit_mask, camera_count,
                                 image_height, image_width, stream);
  });
}

int curobo_mapper_render_color_semantic(
    CuroboMapperRuntime* runtime, uint64_t intrinsics,
    uint64_t cam_positions, uint64_t cam_quaternions, uint64_t out_hit_points,
    uint64_t out_hit_normals, uint64_t out_hit_colors,
    uint64_t out_hit_depths, uint64_t out_hit_mask, int camera_count,
    int image_height, int image_width,
    const CuroboMapperSemanticFilter* semantic_filter, uint64_t stream) {
  return wrap([&] {
    if (semantic_filter == nullptr) {
      throw std::runtime_error("semantic_filter is null");
    }
    checked(runtime).renderColor(
        intrinsics, cam_positions, cam_quaternions, out_hit_points,
        out_hit_normals, out_hit_colors, out_hit_depths, out_hit_mask,
        camera_count, image_height, image_width, stream, semantic_filter);
  });
}

int curobo_mapper_render_features(
    CuroboMapperRuntime* runtime, uint64_t intrinsics, uint64_t cam_positions,
    uint64_t cam_quaternions, uint64_t out_hit_normals,
    uint64_t out_hit_depths, uint64_t out_hit_mask, uint64_t out_feature_valid,
    uint64_t out_feature_block_indices, uint64_t out_feature_node_indices,
    int camera_count, int image_height, int image_width, int patch_top,
    int patch_left, int patch_height, int patch_width, uint64_t stream) {
  return wrap([&] {
    checked(runtime).renderFeatures(
        intrinsics, cam_positions, cam_quaternions, out_hit_normals,
        out_hit_depths, out_hit_mask, out_feature_valid,
        out_feature_block_indices, out_feature_node_indices, camera_count,
        image_height, image_width, patch_top, patch_left, patch_height,
        patch_width, stream);
  });
}

int curobo_mapper_render_features_semantic(
    CuroboMapperRuntime* runtime, uint64_t intrinsics, uint64_t cam_positions,
    uint64_t cam_quaternions, uint64_t out_hit_normals,
    uint64_t out_hit_depths, uint64_t out_hit_mask, uint64_t out_feature_valid,
    uint64_t out_feature_block_indices, uint64_t out_feature_node_indices,
    int camera_count, int image_height, int image_width, int patch_top,
    int patch_left, int patch_height, int patch_width,
    const CuroboMapperSemanticFilter* semantic_filter, uint64_t stream) {
  return wrap([&] {
    if (semantic_filter == nullptr) {
      throw std::runtime_error("semantic_filter is null");
    }
    checked(runtime).renderFeatures(
        intrinsics, cam_positions, cam_quaternions, out_hit_normals,
        out_hit_depths, out_hit_mask, out_feature_valid,
        out_feature_block_indices, out_feature_node_indices, camera_count,
        image_height, image_width, patch_top, patch_left, patch_height,
        patch_width, stream, semantic_filter);
  });
}

int curobo_mapper_gather_features(
    CuroboMapperRuntime* runtime, uint64_t feature_valid,
    uint64_t feature_block_indices, uint64_t feature_node_indices,
    uint64_t out_features, int camera_count, int image_height, int image_width,
    int channel_start, int channel_count, int output_float16, uint64_t stream) {
  return wrap([&] {
    checked(runtime).gatherFeatures(
        feature_valid, feature_block_indices, feature_node_indices, out_features,
        camera_count, image_height, image_width, channel_start, channel_count,
        output_float16 != 0, stream);
  });
}

int curobo_mapper_extract_mesh(
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
    int* written_triangles) {
  return wrap([&] {
    checked(runtime).extractMesh(out_vertices, out_normals, out_colors,
                                 vertex_capacity, level, surface_only != 0,
                                 refine_iterations, stream, required_vertices,
                                 required_triangles, written_vertices,
                                 written_triangles);
  });
}

int curobo_mapper_extract_mesh_semantic(
    CuroboMapperRuntime* runtime, uint64_t out_vertices,
    uint64_t out_normals, uint64_t out_colors, int vertex_capacity,
    float level, int surface_only, int refine_iterations,
    const CuroboMapperSemanticFilter* semantic_filter, uint64_t stream,
    int* required_vertices, int* required_triangles, int* written_vertices,
    int* written_triangles) {
  return wrap([&] {
    if (semantic_filter == nullptr) {
      throw std::runtime_error("semantic_filter is null");
    }
    checked(runtime).extractMesh(
        out_vertices, out_normals, out_colors, vertex_capacity, level,
        surface_only != 0, refine_iterations, stream, required_vertices,
        required_triangles, written_vertices, written_triangles,
        semantic_filter);
  });
}

int curobo_mapper_project_mesh_texture(
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
    int* texture_atlas_width) {
  return wrap([&] {
    if (frame == nullptr) {
      throw std::runtime_error("frame is null");
    }
    checked(runtime).projectMeshTexture(*frame, vertices, vertex_count, out_uvs,
                                        out_colors, out_texture_atlas,
                                        texture_atlas_capacity_bytes,
                                        texture_depth_tolerance_m,
                                        texture_atlas_height,
                                        texture_atlas_width);
  });
}

int curobo_mapper_refine_pose(
    CuroboMapperRuntime* runtime,
    uint64_t depth_image,
    uint64_t intrinsics,
    uint64_t initial_position,
    uint64_t initial_quaternion,
    uint64_t out_position,
    uint64_t out_quaternion,
    const CuroboMapperPoseRefinerConfig* config,
    uint64_t stream,
    CuroboMapperPoseRefinerResult* result) {
  return wrap([&] {
    checked(runtime).refinePose(depth_image, intrinsics, initial_position,
                                initial_quaternion, out_position, out_quaternion,
                                config, stream, result);
  });
}

int curobo_mapper_num_allocated(CuroboMapperRuntime* runtime,
                                uint64_t stream,
                                int* num_allocated) {
  return wrap([&] {
    if (num_allocated == nullptr) {
      throw std::runtime_error("num_allocated output pointer is null");
    }
    *num_allocated = checked(runtime).numAllocated(stream);
  });
}

int curobo_mapper_get_device_view(CuroboMapperRuntime* runtime,
                                  CuroboMapperDeviceView* view) {
  return wrap([&] {
    if (view == nullptr) {
      throw std::runtime_error("device view output pointer is null");
    }
    *view = checked(runtime).deviceView();
  });
}

int curobo_mapper_config_int(CuroboMapperRuntime* runtime,
                             const char* name,
                             int* value) {
  return wrap([&] {
    if (name == nullptr || value == nullptr) {
      throw std::runtime_error("config_int received null pointer");
    }
    *value = checked(runtime).configInt(name);
  });
}

int curobo_mapper_config_float(CuroboMapperRuntime* runtime,
                               const char* name,
                               float* value) {
  return wrap([&] {
    if (name == nullptr || value == nullptr) {
      throw std::runtime_error("config_float received null pointer");
    }
    *value = checked(runtime).configFloat(name);
  });
}

int curobo_mapper_set_tsdf_runtime_parameters(
    CuroboMapperRuntime* runtime,
    float decay_factor,
    float minimum_tsdf_weight) {
  return wrap([&] {
    checked(runtime).setTsdfRuntimeParameters(decay_factor,
                                              minimum_tsdf_weight);
  });
}

}  // extern "C"
