// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#include "device_scan.h"

#include "cuda_check.h"
#include "curobo_mapper_runtime/api.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>

namespace curobo_mapper_runtime {
__global__ void fillHalfInfinityKernel(uint16_t *output, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = 0x7c00;
}
void fillHalfInfinity(uint16_t *output, int count, cudaStream_t stream) {
  fillHalfInfinityKernel<<<(count + 255) / 256, 256, 0, stream>>>(output, count);
  CMR_CUDA_CHECK(cudaGetLastError());
}

size_t exclusiveSumTempStorageBytes(int num_items) {
  size_t temp_storage_bytes = 0;
  CMR_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr, temp_storage_bytes, static_cast<const int*>(nullptr),
      static_cast<int*>(nullptr), num_items));
  return temp_storage_bytes;
}

size_t reduceSumTempStorageBytes(int num_items) {
  size_t temp_storage_bytes = 0;
  CMR_CUDA_CHECK(cub::DeviceReduce::Sum(
      nullptr, temp_storage_bytes, static_cast<const int*>(nullptr),
      static_cast<int*>(nullptr), num_items));
  return temp_storage_bytes;
}

void exclusiveSumInt32(void* temp_storage, size_t temp_storage_bytes,
                       const int* input, int* output, int num_items,
                       cudaStream_t stream) {
  CMR_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      temp_storage, temp_storage_bytes, input, output, num_items, stream));
}

void reduceSumInt32(void* temp_storage, size_t temp_storage_bytes,
                    const int* input, int* output, int num_items,
                    cudaStream_t stream) {
  CMR_CUDA_CHECK(cub::DeviceReduce::Sum(
      temp_storage, temp_storage_bytes, input, output, num_items, stream));
}

__global__ void setFastMeshCountsFromBlockCountsKernel(
    const int* block_triangle_counts, const int* block_triangle_offsets,
    int num_blocks, int triangle_capacity, int* mesh_counts) {
  int total_triangles = 0;
  if (num_blocks > 0) {
    const int last = num_blocks - 1;
    total_triangles =
        block_triangle_offsets[last] + block_triangle_counts[last];
  }

  const int written_triangles = max(0, min(total_triangles, triangle_capacity));
  mesh_counts[CUROBO_MAPPER_MESH_REQUIRED_TRIANGLES] = total_triangles;
  mesh_counts[CUROBO_MAPPER_MESH_REQUIRED_VERTICES] = total_triangles * 3;
  mesh_counts[CUROBO_MAPPER_MESH_WRITTEN_TRIANGLES] = written_triangles;
  mesh_counts[CUROBO_MAPPER_MESH_WRITTEN_VERTICES] = written_triangles * 3;
  mesh_counts[CUROBO_MAPPER_MESH_OVERFLOW_TRIANGLES] =
      max(0, total_triangles - triangle_capacity);
}

void setFastMeshCountsFromBlockCounts(const int* block_triangle_counts,
                                      const int* block_triangle_offsets,
                                      int num_blocks, int triangle_capacity,
                                      int* mesh_counts, cudaStream_t stream) {
  setFastMeshCountsFromBlockCountsKernel<<<1, 1, 0, stream>>>(
      block_triangle_counts, block_triangle_offsets, num_blocks,
      triangle_capacity, mesh_counts);
  CMR_CUDA_CHECK(cudaGetLastError());
}

__global__ void fillFloatKernel(float* output, float value, int num_items) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    output[idx] = value;
  }
}

void fillFloat(float* output, float value, int num_items, cudaStream_t stream) {
  if (num_items <= 0) {
    return;
  }
  constexpr int block_dim = 256;
  const int grid_dim = (num_items + block_dim - 1) / block_dim;
  fillFloatKernel<<<grid_dim, block_dim, 0, stream>>>(output, value, num_items);
  CMR_CUDA_CHECK(cudaGetLastError());
}

__global__ void packTextureAtlasKernel(const uint8_t* rgb_images,
                                       int total_cameras, int image_height,
                                       int image_width, int atlas_columns,
                                       uint8_t* texture_atlas) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixels_per_camera = image_height * image_width;
  const int total_pixels = total_cameras * pixels_per_camera;
  if (idx >= total_pixels) {
    return;
  }

  const int camera_idx = idx / pixels_per_camera;
  const int pixel_idx = idx - camera_idx * pixels_per_camera;
  const int y = pixel_idx / image_width;
  const int x = pixel_idx - y * image_width;
  const int atlas_row = camera_idx / atlas_columns;
  const int atlas_col = camera_idx - atlas_row * atlas_columns;
  const int atlas_width = atlas_columns * image_width;
  const int atlas_x = atlas_col * image_width + x;
  const int atlas_y = atlas_row * image_height + y;
  const int src = idx * 3;
  const int dst = (atlas_y * atlas_width + atlas_x) * 3;
  texture_atlas[dst + 0] = rgb_images[src + 0];
  texture_atlas[dst + 1] = rgb_images[src + 1];
  texture_atlas[dst + 2] = rgb_images[src + 2];
}

void packTextureAtlas(const uint8_t* rgb_images, int batch_count,
                      int cameras_per_batch, int image_height, int image_width,
                      int atlas_columns, uint8_t* texture_atlas,
                      cudaStream_t stream) {
  const int total_cameras = batch_count * cameras_per_batch;
  const int total_pixels = total_cameras * image_height * image_width;
  if (total_pixels <= 0) {
    return;
  }
  constexpr int block_dim = 256;
  const int grid_dim = (total_pixels + block_dim - 1) / block_dim;
  packTextureAtlasKernel<<<grid_dim, block_dim, 0, stream>>>(
      rgb_images, total_cameras, image_height, image_width, atlas_columns,
      texture_atlas);
  CMR_CUDA_CHECK(cudaGetLastError());
}

__device__ bool validUv(float u, float v) {
  return isfinite(u) && isfinite(v) && u >= 0.0f && u <= 1.0f && v >= 0.0f &&
         v <= 1.0f;
}

__global__ void markTextureFallbackTrianglesKernel(const float* texture_uvs,
                                                   int triangle_count,
                                                   int* fallback_flags) {
  const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tri_idx >= triangle_count) {
    return;
  }
  const int base = tri_idx * 6;
  const bool valid =
      validUv(texture_uvs[base + 0], texture_uvs[base + 1]) &&
      validUv(texture_uvs[base + 2], texture_uvs[base + 3]) &&
      validUv(texture_uvs[base + 4], texture_uvs[base + 5]);
  fallback_flags[tri_idx] = valid ? 0 : 1;
}

void markTextureFallbackTriangles(const float* texture_uvs, int triangle_count,
                                  int* fallback_flags, cudaStream_t stream) {
  if (triangle_count <= 0) {
    return;
  }
  constexpr int block_dim = 256;
  const int grid_dim = (triangle_count + block_dim - 1) / block_dim;
  markTextureFallbackTrianglesKernel<<<grid_dim, block_dim, 0, stream>>>(
      texture_uvs, triangle_count, fallback_flags);
  CMR_CUDA_CHECK(cudaGetLastError());
}

__device__ int clampInt(int value, int lo, int hi) {
  return max(lo, min(value, hi));
}

__device__ uint8_t averageColor(uint8_t a, uint8_t b, uint8_t c) {
  return static_cast<uint8_t>(floorf((static_cast<float>(a) +
                                      static_cast<float>(b) +
                                      static_cast<float>(c)) /
                                         3.0f +
                                     0.5f));
}

__global__ void finalizeTextureAtlasKernel(
    float* texture_uvs, int triangle_count, const int* fallback_flags,
    const int* fallback_offsets, uint8_t* colors, uint8_t* texture_atlas,
    int base_atlas_height, int atlas_width, int final_atlas_height) {
  const int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tri_idx >= triangle_count) {
    return;
  }

  const int vertex_base = tri_idx * 3;
  const int uv_base = vertex_base * 2;
  if (fallback_flags[tri_idx] == 0) {
    const float v_scale =
        static_cast<float>(base_atlas_height) / static_cast<float>(final_atlas_height);
    for (int local = 0; local < 3; ++local) {
      const int vertex_idx = vertex_base + local;
      const int uv_idx = uv_base + local * 2;
      const float u = min(1.0f, max(0.0f, texture_uvs[uv_idx + 0]));
      const float v = min(1.0f, max(0.0f, texture_uvs[uv_idx + 1]));
      const int px =
          clampInt(static_cast<int>(floorf(u * static_cast<float>(atlas_width - 1) +
                                           0.5f)),
                   0, atlas_width - 1);
      const int py =
          clampInt(static_cast<int>(floorf(v * static_cast<float>(base_atlas_height - 1) +
                                           0.5f)),
                   0, base_atlas_height - 1);
      const int atlas_idx = (py * atlas_width + px) * 3;
      const int color_idx = vertex_idx * 3;
      colors[color_idx + 0] = texture_atlas[atlas_idx + 0];
      colors[color_idx + 1] = texture_atlas[atlas_idx + 1];
      colors[color_idx + 2] = texture_atlas[atlas_idx + 2];
      texture_uvs[uv_idx + 0] = u;
      texture_uvs[uv_idx + 1] = v * v_scale;
    }
    return;
  }

  const int slot = fallback_offsets[tri_idx];
  const int patch_size = 2;
  const int patches_per_row = max(1, atlas_width / patch_size);
  const int patch_x = (slot % patches_per_row) * patch_size;
  const int patch_y = base_atlas_height + (slot / patches_per_row) * patch_size;

  const int c0 = (vertex_base + 0) * 3;
  const int c1 = (vertex_base + 1) * 3;
  const int c2 = (vertex_base + 2) * 3;
  const int p00 = (patch_y * atlas_width + patch_x) * 3;
  const int p10 = (patch_y * atlas_width + patch_x + 1) * 3;
  const int p01 = ((patch_y + 1) * atlas_width + patch_x) * 3;
  const int p11 = ((patch_y + 1) * atlas_width + patch_x + 1) * 3;
  for (int ch = 0; ch < 3; ++ch) {
    texture_atlas[p00 + ch] = colors[c0 + ch];
    texture_atlas[p10 + ch] = colors[c1 + ch];
    texture_atlas[p01 + ch] = colors[c2 + ch];
    texture_atlas[p11 + ch] =
        averageColor(colors[c0 + ch], colors[c1 + ch], colors[c2 + ch]);
  }

  texture_uvs[uv_base + 0] =
      (static_cast<float>(patch_x) + 0.5f) / static_cast<float>(atlas_width);
  texture_uvs[uv_base + 1] =
      (static_cast<float>(patch_y) + 0.5f) / static_cast<float>(final_atlas_height);
  texture_uvs[uv_base + 2] =
      (static_cast<float>(patch_x) + 1.5f) / static_cast<float>(atlas_width);
  texture_uvs[uv_base + 3] =
      (static_cast<float>(patch_y) + 0.5f) / static_cast<float>(final_atlas_height);
  texture_uvs[uv_base + 4] =
      (static_cast<float>(patch_x) + 0.5f) / static_cast<float>(atlas_width);
  texture_uvs[uv_base + 5] =
      (static_cast<float>(patch_y) + 1.5f) / static_cast<float>(final_atlas_height);
}

void finalizeTextureAtlas(float* texture_uvs, int triangle_count,
                          const int* fallback_flags,
                          const int* fallback_offsets, uint8_t* colors,
                          uint8_t* texture_atlas, int base_atlas_height,
                          int atlas_width, int final_atlas_height,
                          cudaStream_t stream) {
  if (triangle_count <= 0) {
    return;
  }
  constexpr int block_dim = 256;
  const int grid_dim = (triangle_count + block_dim - 1) / block_dim;
  finalizeTextureAtlasKernel<<<grid_dim, block_dim, 0, stream>>>(
      texture_uvs, triangle_count, fallback_flags, fallback_offsets, colors,
      texture_atlas, base_atlas_height, atlas_width, final_atlas_height);
  CMR_CUDA_CHECK(cudaGetLastError());
}

__global__ void exclude_instance_nodes_kernel(const uint16_t* instance_state, uint8_t* node_match,
                                              const int* num_allocated, int block_capacity,
                                              int voxels_per_block, int slot, bool exclude,
                                              const uint64_t* esdf_query) {
  if (esdf_query != nullptr) slot = static_cast<int>(esdf_query[1]);
  if (slot == 0) return;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = min(*num_allocated, block_capacity) * voxels_per_block;
  if (index < count && (instance_state[index] & 0xffU) == slot) {
    node_match[index] = exclude ? 1 : 0;
  }
}

void exclude_instance_nodes(const uint16_t* instance_state, uint8_t* node_match,
                            const int* num_allocated, int block_capacity, int voxels_per_block,
                            int slot, bool exclude, cudaStream_t stream, const uint64_t* esdf_query) {
  constexpr int BLOCK_DIM = 256;
  const int count = block_capacity * voxels_per_block;
  exclude_instance_nodes_kernel<<<(count + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM, 0, stream>>>(
      instance_state, node_match, num_allocated, block_capacity, voxels_per_block, slot, exclude,
      esdf_query);
  CMR_CUDA_CHECK(cudaGetLastError());
}

__global__ void setEsdfQueryKernel(uint64_t* query, uint64_t output, int slot) {
  query[0] = output;
  query[1] = static_cast<uint64_t>(slot);
}

__global__ void copyEsdfOutputKernel(const uint16_t* distances, int count, const uint64_t* query) {
  auto* output = reinterpret_cast<uint16_t*>(query[0]);
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output != nullptr && index < count) output[index] = distances[index];
}

void setEsdfQuery(uint64_t* query, uint64_t output, int instance_slot, cudaStream_t stream) {
  setEsdfQueryKernel<<<1, 1, 0, stream>>>(query, output, instance_slot);
  CMR_CUDA_CHECK(cudaGetLastError());
}

void copyEsdfOutput(const uint16_t* distances, int count, const uint64_t* query,
                    cudaStream_t stream) {
  copyEsdfOutputKernel<<<(count + 255) / 256, 256, 0, stream>>>(distances, count, query);
  CMR_CUDA_CHECK(cudaGetLastError());
}

}  // namespace curobo_mapper_runtime
