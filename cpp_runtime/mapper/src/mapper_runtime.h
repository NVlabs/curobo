// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cubin_kernel.h"
#include "curobo_mapper_runtime/api.h"
#include "curobo_mapper_runtime/warp_abi.h"

namespace curobo_mapper_runtime {

struct KernelManifest {
  std::string cubin;
  std::string symbol;
  unsigned int shared_mem{0};
};

struct MapperConfig {
  int num_cameras{1};
  int image_height{0};
  int image_width{0};
  int grid_D{0};
  int grid_H{0};
  int grid_W{0};
  int esdf_nx{0};
  int esdf_ny{0};
  int esdf_nz{0};
  int block_size{4};
  int block_voxels{64};
  int num_samples{1};
  int num_block_key_candidates{0};
  int max_blocks{0};
  int hash_capacity{0};
  int max_visible_blocks{0};
  int max_support_pixels_per_block_camera{0};
  int pose_refiner_n_samples_per_ray{1};
  int color_grid_size{1};
  int color_grid_voxels{1};
  int feature_dim{0};
  int feature_block_grid_size{1};
  int feature_grid_voxels{1};
  int instance_grid_size{0};
  int instance_grid_voxels{1};
  int feature_grid_height{0};
  int feature_grid_width{0};
  int feature_channels_per_thread{4};
  int max_feature_tile_channels{4096};
  bool use_tiled_feature_kernel{false};
  int texture_num_cameras{1};
  int texture_camera_image_height{0};
  int texture_camera_image_width{0};
  float origin[3]{0.0f, 0.0f, 0.0f};
  float voxel_size{0.0f};
  float esdf_voxel_size{0.0f};
  float truncation_distance{0.0f};
  float depth_minimum_distance{0.1f};
  float depth_maximum_distance{5.0f};
  float minimum_tsdf_weight{0.1f};
  float adjacent_skip_steps{1.0f};
  float decay_factor{1.0f};
  float frustum_decay_factor{1.0f};
  float accumulator_w_max{1000.0f};
};

struct MapperManifest {
  MapperConfig config;
  std::map<std::string, KernelManifest> kernels;
  KernelManifest pba_flood_z;
  KernelManifest pba_maurer_axis;
  KernelManifest pba_color_axis;
  std::vector<int> mc_num_triangles;
  std::vector<int> mc_triangle_table;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void allocate(size_t bytes);
  void memsetAsync(int value, cudaStream_t stream);

  uint64_t ptr() const { return reinterpret_cast<uint64_t>(ptr_); }
  void* raw() const { return ptr_; }
  size_t bytes() const { return bytes_; }

 private:
  void release();

  void* ptr_{nullptr};
  size_t bytes_{0};
};

struct SurfacePointScratch {
  DeviceBuffer proposals;
  DeviceBuffer sources;
  DeviceBuffer valid;
  DeviceBuffer offsets;
  DeviceBuffer scan_temp;
};

struct SurfacePointWorkspace {
  DeviceBuffer flags;
  DeviceBuffer counts;
  DeviceBuffer offsets;
  DeviceBuffer scan_temp;
  DeviceBuffer query;
  DeviceBuffer target;
  DeviceBuffer selection_dummy;
  std::map<int, SurfacePointScratch> outputs;
  int groups{0};
};

struct MapperBuffers {
  DeviceBuffer hash_table;
  DeviceBuffer block_data;
  DeviceBuffer block_grid_rgb;
  DeviceBuffer block_features;
  DeviceBuffer block_feature_weight;
  DeviceBuffer block_instance_state;
  DeviceBuffer static_block_data;
  DeviceBuffer block_coords;
  DeviceBuffer block_to_hash_slot;
  DeviceBuffer free_list;
  DeviceBuffer free_count;
  DeviceBuffer num_allocated;
  DeviceBuffer allocation_failures;
  DeviceBuffer block_sums;
  DeviceBuffer static_block_sums;
  DeviceBuffer frustum_flags;
  DeviceBuffer new_blocks;
  DeviceBuffer new_block_count;
  DeviceBuffer recycle_count;

  DeviceBuffer block_keys;
  DeviceBuffer pool_indices;
  DeviceBuffer visible_epoch;
  DeviceBuffer visible_count;
  DeviceBuffer pool_to_visible_slot;
  DeviceBuffer support_counts;
  DeviceBuffer support_pixels;
  DeviceBuffer support_overflow_count;
  DeviceBuffer occupied_count;
  DeviceBuffer semantic_target;
  DeviceBuffer semantic_node_match;
  DeviceBuffer semantic_block_has_keep;
  DeviceBuffer instance_removal_block_mask;

  DeviceBuffer esdf_site_index;
  DeviceBuffer esdf_pba_buffer;
  DeviceBuffer esdf_dist_field;
  DeviceBuffer esdf_origin;
  DeviceBuffer esdf_voxel_size;
  DeviceBuffer esdf_query;

  DeviceBuffer active_block_count;
  DeviceBuffer active_block_idx;
  DeviceBuffer surface_count;
  DeviceBuffer surface_block_idx;
  DeviceBuffer surface_cube_idx;
  DeviceBuffer triangle_count;
  DeviceBuffer mc_num_triangles;
  DeviceBuffer mc_triangle_table;
  DeviceBuffer render_hit_points;
  DeviceBuffer render_hit_normals;
  DeviceBuffer render_hit_depths;
  DeviceBuffer render_hit_mask;
  DeviceBuffer texture_projection_scores;
  DeviceBuffer texture_fallback_flags;
  DeviceBuffer texture_fallback_offsets;
  DeviceBuffer texture_fallback_count;
  DeviceBuffer texture_scan_temp;

  DeviceBuffer pose_valid_depth_count;
  DeviceBuffer pose_best_position;
  DeviceBuffer pose_best_quaternion;
  DeviceBuffer pose_candidate_position;
  DeviceBuffer pose_candidate_quaternion;
  DeviceBuffer pose_best_jtj;
  DeviceBuffer pose_best_jtr;
  DeviceBuffer pose_best_sum_sq;
  DeviceBuffer pose_best_error;
  DeviceBuffer pose_best_valid_count;
  DeviceBuffer pose_candidate_jtj;
  DeviceBuffer pose_candidate_jtr;
  DeviceBuffer pose_candidate_sum_sq;
  DeviceBuffer pose_candidate_valid_count;
  DeviceBuffer pose_lambda;
  DeviceBuffer pose_pred_reduction;
  DeviceBuffer pose_result;
};

struct MapperKernels {
  CubinKernel compute_block_keys;
  CubinKernel mark_blocks_in_frustum;
  CubinKernel allocate_visible_blocks;
  CubinKernel build_support_pixels;
  CubinKernel clear_new_blocks;
  CubinKernel clear_new_block_grid_rgb;
  CubinKernel clear_new_block_features;
  CubinKernel clear_new_block_instances;
  CubinKernel clear_instance_slot;
  CubinKernel mark_instance_removal_blocks;
  CubinKernel remove_instance_geometry;
  CubinKernel integrate_voxels;
  CubinKernel integrate_block_grid_rgb;
  CubinKernel integrate_features;
  CubinKernel integrate_instances;
  CubinKernel rescale_accumulators;
  CubinKernel rescale_block_grid_rgb;
  CubinKernel apply_decay_and_sum;
  CubinKernel recycle_empty_blocks;
  CubinKernel count_occupied;
  CubinKernel set_point_query;
  CubinKernel set_point_target_class;
  CubinKernel count_point_edges;
  CubinKernel sample_point_edges;
  CubinKernel compact_surface_points;
  CubinKernel extract_occupied;
  CubinKernel seed_esdf_gather;
  CubinKernel compute_esdf_distance;
  CubinKernel raycast_depth;
  CubinKernel raycast_color;
  CubinKernel raycast_features;
  CubinKernel gather_features_fp32;
  CubinKernel gather_features_fp16;
  CubinKernel prepare_semantic_node_matches;
  CubinKernel prepare_semantic_argmax_node_matches;
  CubinKernel prepare_semantic_block_summary;
  CubinKernel count_occupied_semantic;
  CubinKernel extract_occupied_semantic;
  CubinKernel seed_esdf_gather_semantic;
  CubinKernel compute_esdf_distance_semantic;
  CubinKernel raycast_depth_semantic;
  CubinKernel raycast_color_semantic;
  CubinKernel raycast_features_semantic;
  CubinKernel append_active_blocks_semantic;
  CubinKernel append_surface_cubes_from_blocks_semantic;
  CubinKernel append_active_blocks;
  CubinKernel count_surface_cubes_from_blocks;
  CubinKernel append_surface_cubes_from_blocks;
  CubinKernel count_total_triangles;
  CubinKernel generate_mesh;
  CubinKernel project_mesh_uvs;
  CubinKernel pose_refine_alignment;
  CubinKernel pose_refine_count_valid_depth;
  CubinKernel pose_refine_initialize_state;
  CubinKernel pose_refine_solve_candidate;
  CubinKernel pose_refine_accept_candidate;
  CubinKernel pose_refine_copy_best_pose;
  CubinKernel pba_flood_z;
  CubinKernel pba_maurer_axis;
  CubinKernel pba_color_axis;
};

class MapperRuntimeImpl {
 public:
  MapperRuntimeImpl(std::string artifact_dir, int device_ordinal);
  ~MapperRuntimeImpl();

  void reset(uint64_t stream_handle);
  void loadStaticSeed(const std::string &path, uint64_t stream_handle);
  void clearInstanceSlot(int instance_slot, uint64_t stream_handle);
  void removeInstanceGeometry(int instance_slot,
                              uint64_t semantic_class_mask,
                              uint64_t stream_handle);
  void integrate(const CuroboMapperFrame& frame, uint64_t feature_grid = 0);
  void extract_occupied_points(uint64_t out_positions, uint64_t out_source_voxels,
      uint64_t out_count, int sample_budget, int instance_slot, uint64_t voxel_selection,
      const CuroboMapperSemanticFilter* semantic_filter, int seed, uint64_t stream_handle);
  void extractOccupied(uint64_t out_centers, uint64_t out_pool_indices,
                       uint64_t out_count, int capacity, bool surface_only,
                       float sdf_threshold, uint64_t stream_handle,
                       int* required_count, int* written_count,
                       const CuroboMapperSemanticFilter* semantic_filter = nullptr);
  void computeEsdf(uint64_t out_dist_field, uint64_t stream_handle,
                   const CuroboMapperSemanticFilter* semantic_filter = nullptr);
  uint64_t esdfGraphCaptureCount() const { return esdf_graph_capture_count_; }
  void renderDepth(uint64_t intrinsics, uint64_t cam_positions,
                   uint64_t cam_quaternions, uint64_t out_hit_points,
                   uint64_t out_hit_normals, uint64_t out_hit_depths,
                   uint64_t out_hit_mask, int camera_count, int image_height,
                   int image_width, uint64_t stream_handle,
                   const CuroboMapperSemanticFilter* semantic_filter = nullptr);
  void renderColor(uint64_t intrinsics, uint64_t cam_positions,
                   uint64_t cam_quaternions, uint64_t out_hit_points,
                   uint64_t out_hit_normals, uint64_t out_hit_colors,
                   uint64_t out_hit_depths, uint64_t out_hit_mask,
                   int camera_count, int image_height, int image_width,
                   uint64_t stream_handle,
                   const CuroboMapperSemanticFilter* semantic_filter = nullptr);
  void renderFeatures(uint64_t intrinsics, uint64_t cam_positions,
                      uint64_t cam_quaternions, uint64_t out_hit_normals,
                      uint64_t out_hit_depths, uint64_t out_hit_mask,
                      uint64_t out_feature_valid,
                      uint64_t out_feature_block_indices,
                      uint64_t out_feature_node_indices, int camera_count,
                      int image_height, int image_width, int patch_top,
                      int patch_left, int patch_height, int patch_width,
                      uint64_t stream_handle,
                      const CuroboMapperSemanticFilter* semantic_filter = nullptr);
  void gatherFeatures(uint64_t feature_valid,
                      uint64_t feature_block_indices,
                      uint64_t feature_node_indices, uint64_t out_features,
                      int camera_count, int image_height, int image_width,
                      int channel_start, int channel_count, bool output_float16,
                      uint64_t stream_handle);
  void extractMesh(uint64_t out_vertices, uint64_t out_normals,
                   uint64_t out_colors, int vertex_capacity, float level,
                   bool surface_only, int refine_iterations,
                   uint64_t stream_handle, int* required_vertices,
                   int* required_triangles, int* written_vertices,
                   int* written_triangles,
                   const CuroboMapperSemanticFilter* semantic_filter = nullptr);
  void projectMeshTexture(const CuroboMapperTextureFrame& frame,
                          uint64_t vertices, int vertex_count,
                          uint64_t out_uvs, uint64_t out_colors,
                          uint64_t out_texture_atlas,
                          uint64_t texture_atlas_capacity_bytes,
                          float texture_depth_tolerance_m,
                          int* texture_atlas_height,
                          int* texture_atlas_width);
  void refinePose(uint64_t depth_image, uint64_t intrinsics,
                  uint64_t initial_position, uint64_t initial_quaternion,
                  uint64_t out_position, uint64_t out_quaternion,
                  const CuroboMapperPoseRefinerConfig* config,
                  uint64_t stream_handle, CuroboMapperPoseRefinerResult* result);
  int numAllocated(uint64_t stream_handle);
  CuroboMapperDeviceView deviceView() const;

  int configInt(const std::string& name) const;
  float configFloat(const std::string& name) const;
  void setTsdfRuntimeParameters(float decay_factor,
                                float minimum_tsdf_weight);

 private:
  void allocateBuffers();
  void loadKernels();
  BlockSparseTSDFWarp tsdfWarp() const;
  void ensureMeshScratch();
  void ensure_point_scratch(int sample_budget, cudaStream_t stream);
  void ensureRenderScratch(int n_pixels);
  void ensureTextureScoreScratch(int triangle_count);
  void ensureTexturePostprocessScratch(int triangle_count);
  void launchRender(bool with_color, uint64_t intrinsics, uint64_t cam_positions,
                    uint64_t cam_quaternions, uint64_t out_hit_points,
                    uint64_t out_hit_normals, uint64_t out_hit_colors,
                    uint64_t out_hit_depths, uint64_t out_hit_mask,
                    int camera_count, int image_height, int image_width,
                    uint64_t stream_handle,
                    const CuroboMapperSemanticFilter* semantic_filter);
  PreparedSemanticFilterWarp prepareSemanticFilter(
      const CuroboMapperSemanticFilter& semantic_filter, cudaStream_t cuda_stream,
      CUstream cu_stream, const uint64_t* esdf_query = nullptr);
  uint64_t visibilityDepthForTextureFrame(const CuroboMapperTextureFrame& frame);
  void maybeDecay(const CuroboMapperFrame& frame, cudaStream_t cuda_stream,
                  CUstream cu_stream);
  void launchPba(cudaStream_t cuda_stream, CUstream cu_stream);
  void enqueueEsdf(uint64_t out_dist_field, cudaStream_t cuda_stream, CUstream cu_stream,
                   const CuroboMapperSemanticFilter* semantic_filter, bool dynamic_query = false);
  int semanticPreparationBlockCapacity() const;
  void destroyEsdfGraph() noexcept;
  void launchPoseAlignment(uint64_t depth_image, uint64_t intrinsics,
                           uint64_t position, uint64_t quaternion,
                           const CuroboMapperPoseRefinerConfig& config,
                           int stride, int out_w, int n_pixels,
                           uint64_t out_jtj, uint64_t out_jtr,
                           uint64_t out_sum_sq, uint64_t out_valid_count,
                           CUstream cu_stream);

  CubinKernel makeKernel(const KernelManifest& manifest) const;
  const KernelManifest& kernelManifest(const std::string& name) const;

  Dim3 gridFor(size_t linear_size, unsigned int block_dim = 256) const;

  std::string artifact_dir_;
  int device_ordinal_{0};
  MapperManifest manifest_;
  MapperConfig config_;
  MapperBuffers buffers_;
  std::vector<std::vector<char>> static_seed_;
  int static_seed_blocks_{0};
  void restoreStaticSeed(cudaStream_t stream);
  SurfacePointWorkspace points_;
  bool mesh_scratch_initialized_{false};
  MapperKernels kernels_;
  int frame_epoch_{0};
  int frame_count_{0};
  int cached_num_allocated_{0};
  cudaGraph_t esdf_graph_{nullptr};
  uint64_t esdf_graph_capture_count_{0};
  cudaGraphExec_t esdf_graph_executable_{nullptr};
  uint64_t esdf_graph_stream_{0};
  int esdf_graph_semantic_block_capacity_{0};
  bool esdf_graph_has_filter_{false};
  CuroboMapperSemanticFilter esdf_graph_filter_{};
};

MapperManifest loadManifest(const std::string& artifact_dir);
CuroboMapperPoseRefinerConfig defaultPoseRefinerConfig();

}  // namespace curobo_mapper_runtime
