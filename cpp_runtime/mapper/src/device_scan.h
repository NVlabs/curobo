// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace curobo_mapper_runtime {

size_t exclusiveSumTempStorageBytes(int num_items);
size_t reduceSumTempStorageBytes(int num_items);

void exclusiveSumInt32(void* temp_storage, size_t temp_storage_bytes,
                       const int* input, int* output, int num_items,
                       cudaStream_t stream);
void reduceSumInt32(void* temp_storage, size_t temp_storage_bytes,
                    const int* input, int* output, int num_items,
                    cudaStream_t stream);

void setFastMeshCountsFromBlockCounts(const int* block_triangle_counts,
                                      const int* block_triangle_offsets,
                                      int num_blocks, int triangle_capacity,
                                      int* mesh_counts, cudaStream_t stream);

void fillFloat(float* output, float value, int num_items, cudaStream_t stream);
void fillHalfInfinity(uint16_t *output, int num_items, cudaStream_t stream);

// Remove an instance from either an inclusion or exclusion semantic mask.
void exclude_instance_nodes(const uint16_t* instance_state, uint8_t* node_match,
                            const int* num_allocated, int block_capacity, int voxels_per_block,
                            int slot, bool exclude, cudaStream_t stream, const uint64_t* esdf_query = nullptr);

// Fixed-address parameters keep selected instance and output changes outside
// graph capture. The captured copy remains part of the ESDF graph.
void setEsdfQuery(uint64_t* query, uint64_t output, int instance_slot, cudaStream_t stream);
void copyEsdfOutput(const uint16_t* distances, int count, const uint64_t* query,
                    cudaStream_t stream);

void packTextureAtlas(const uint8_t* rgb_images, int batch_count,
                      int cameras_per_batch, int image_height, int image_width,
                      int atlas_columns, uint8_t* texture_atlas,
                      cudaStream_t stream);

void markTextureFallbackTriangles(const float* texture_uvs, int triangle_count,
                                  int* fallback_flags, cudaStream_t stream);

void finalizeTextureAtlas(float* texture_uvs, int triangle_count,
                          const int* fallback_flags,
                          const int* fallback_offsets, uint8_t* colors,
                          uint8_t* texture_atlas, int base_atlas_height,
                          int atlas_width, int final_atlas_height,
                          cudaStream_t stream);

}  // namespace curobo_mapper_runtime
