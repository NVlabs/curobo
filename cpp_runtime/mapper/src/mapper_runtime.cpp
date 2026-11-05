// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#include "mapper_runtime.h"

#include "cuda_check.h"
#include "device_scan.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace curobo_mapper_runtime {
namespace {

constexpr int kWarpBlockDim = 256;
constexpr int kProjectiveTextureAtlasMaxWidthPx = 16384;
constexpr const char* kPoseAlignmentKernelName =
    "ray_sdf_alignment_block_sparse_tiled_kernel";
constexpr const char* kPoseCountValidDepthKernelName =
    "pose_refiner_count_valid_depth_kernel";
constexpr const char* kPoseInitializeStateKernelName =
    "pose_refiner_initialize_state_kernel";
constexpr const char* kPoseSolveCandidateKernelName =
    "pose_refiner_solve_candidate_kernel";
constexpr const char* kPoseAcceptCandidateKernelName =
    "pose_refiner_accept_candidate_kernel";
constexpr const char* kPoseCopyBestPoseKernelName =
    "pose_refiner_copy_best_pose_kernel";
constexpr std::array<const char*, 5> kPoseLmKernelNames = {
    kPoseCountValidDepthKernelName, kPoseInitializeStateKernelName,
    kPoseSolveCandidateKernelName, kPoseAcceptCandidateKernelName,
    kPoseCopyBestPoseKernelName};

static_assert(CUROBO_MAPPER_POSE_REFINER_SUCCESS == 0);
static_assert(CUROBO_MAPPER_POSE_REFINER_TOO_FEW_DEPTH_PIXELS == 1);
static_assert(CUROBO_MAPPER_POSE_REFINER_TOO_FEW_CORRESPONDENCES == 2);
static_assert(CUROBO_MAPPER_POSE_REFINER_LINEAR_SOLVE_FAILED == 3);
static_assert(offsetof(CuroboMapperPoseRefinerResult, final_error) == 0);
static_assert(offsetof(CuroboMapperPoseRefinerResult, final_lambda) ==
              sizeof(float));
static_assert(offsetof(CuroboMapperPoseRefinerResult, iterations) ==
              2 * sizeof(float));
static_assert(offsetof(CuroboMapperPoseRefinerResult, valid_count) ==
              2 * sizeof(float) + sizeof(int));
static_assert(offsetof(CuroboMapperPoseRefinerResult, accepted_steps) ==
              2 * sizeof(float) + 2 * sizeof(int));
static_assert(offsetof(CuroboMapperPoseRefinerResult, status) ==
              2 * sizeof(float) + 3 * sizeof(int));
static_assert(sizeof(CuroboMapperPoseRefinerResult) ==
              2 * sizeof(float) + 4 * sizeof(int));

bool has_pose_lm_kernels(const std::map<std::string, KernelManifest>& kernels) {
  return std::all_of(kPoseLmKernelNames.begin(), kPoseLmKernelNames.end(),
                     [&](const char* name) {
                       return kernels.find(name) != kernels.end();
                     });
}

std::string readText(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

size_t findKey(const std::string& text, const std::string& key, size_t pos = 0) {
  const std::string needle = "\"" + key + "\"";
  size_t found = text.find(needle, pos);
  while (found != std::string::npos) {
    size_t cursor = found + needle.size();
    while (cursor < text.size() &&
           (text[cursor] == ' ' || text[cursor] == '\n' || text[cursor] == '\r' ||
            text[cursor] == '\t')) {
      ++cursor;
    }
    if (cursor < text.size() && text[cursor] == ':') {
      return found;
    }
    found = text.find(needle, found + needle.size());
  }
  throw std::runtime_error("manifest key not found: " + key);
}

std::string extractObject(const std::string& text, const std::string& key) {
  const size_t key_pos = findKey(text, key);
  const size_t begin = text.find('{', key_pos);
  if (begin == std::string::npos) {
    throw std::runtime_error("manifest object not found for key: " + key);
  }
  int depth = 0;
  for (size_t i = begin; i < text.size(); ++i) {
    if (text[i] == '{') {
      ++depth;
    } else if (text[i] == '}') {
      --depth;
      if (depth == 0) {
        return text.substr(begin, i - begin + 1);
      }
    }
  }
  throw std::runtime_error("unterminated manifest object for key: " + key);
}

std::string extractArrayText(const std::string& text, const std::string& key) {
  const size_t key_pos = findKey(text, key);
  const size_t begin = text.find('[', key_pos);
  if (begin == std::string::npos) {
    throw std::runtime_error("manifest array not found for key: " + key);
  }
  const size_t end = text.find(']', begin);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated manifest array for key: " + key);
  }
  return text.substr(begin + 1, end - begin - 1);
}

std::string extractString(const std::string& text, const std::string& key) {
  const size_t key_pos = findKey(text, key);
  const size_t colon = text.find(':', key_pos);
  const size_t begin = text.find('"', colon + 1);
  if (colon == std::string::npos || begin == std::string::npos) {
    throw std::runtime_error("manifest string not found for key: " + key);
  }
  const size_t end = text.find('"', begin + 1);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated manifest string for key: " + key);
  }
  return text.substr(begin + 1, end - begin - 1);
}

std::string extractStringOr(const std::string& text, const std::string& key,
                            const std::string& default_value) {
  try {
    return extractString(text, key);
  } catch (const std::runtime_error&) {
    return default_value;
  }
}

double extractDouble(const std::string& text, const std::string& key) {
  const size_t key_pos = findKey(text, key);
  const size_t colon = text.find(':', key_pos);
  if (colon == std::string::npos) {
    throw std::runtime_error("manifest numeric value not found for key: " + key);
  }
  char* end = nullptr;
  const double value = std::strtod(text.c_str() + colon + 1, &end);
  if (end == text.c_str() + colon + 1) {
    throw std::runtime_error("failed to parse manifest number for key: " + key);
  }
  return value;
}

int extractInt(const std::string& text, const std::string& key) {
  return static_cast<int>(std::llround(extractDouble(text, key)));
}

int extractIntOr(const std::string& text, const std::string& key,
                 int default_value) {
  try {
    return extractInt(text, key);
  } catch (const std::runtime_error&) {
    return default_value;
  }
}

bool extractBoolOr(const std::string& text, const std::string& key,
                   bool default_value) {
  try {
    const size_t key_pos = findKey(text, key);
    const size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos) {
      throw std::runtime_error("manifest bool value not found for key: " + key);
    }
    size_t cursor = colon + 1;
    while (cursor < text.size() &&
           (text[cursor] == ' ' || text[cursor] == '\n' || text[cursor] == '\r' ||
            text[cursor] == '\t')) {
      ++cursor;
    }
    if (text.compare(cursor, 4, "true") == 0 || text.compare(cursor, 1, "1") == 0) {
      return true;
    }
    if (text.compare(cursor, 5, "false") == 0 || text.compare(cursor, 1, "0") == 0) {
      return false;
    }
    throw std::runtime_error("failed to parse manifest bool for key: " + key);
  } catch (const std::runtime_error&) {
    return default_value;
  }
}

std::vector<int> extractIntArray(const std::string& text, const std::string& key) {
  const std::string body = extractArrayText(text, key);
  std::vector<int> out;
  const char* cur = body.c_str();
  char* end = nullptr;
  while (*cur != '\0') {
    const long value = std::strtol(cur, &end, 10);
    if (end != cur) {
      out.push_back(static_cast<int>(value));
      cur = end;
    } else {
      ++cur;
    }
  }
  return out;
}

std::vector<float> extractFloatArray(const std::string& text, const std::string& key) {
  const std::string body = extractArrayText(text, key);
  std::vector<float> out;
  const char* cur = body.c_str();
  char* end = nullptr;
  while (*cur != '\0') {
    const double value = std::strtod(cur, &end);
    if (end != cur) {
      out.push_back(static_cast<float>(value));
      cur = end;
    } else {
      ++cur;
    }
  }
  return out;
}

KernelManifest parseKernelObject(const std::string& object) {
  KernelManifest manifest;
  manifest.cubin = extractString(object, "cubin");
  manifest.symbol = extractString(object, "symbol");
  manifest.shared_mem = static_cast<unsigned int>(extractInt(object, "shared_mem"));
  return manifest;
}

void requirePositive(const char* name, int value) {
  if (value <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
}

void requireNonNegative(const char* name, int value) {
  if (value < 0) {
    throw std::runtime_error(std::string(name) + " must be non-negative");
  }
}

cudaStream_t toCudaStream(uint64_t stream_handle) {
  return reinterpret_cast<cudaStream_t>(stream_handle);
}

CUstream toCuStream(uint64_t stream_handle) {
  return reinterpret_cast<CUstream>(stream_handle);
}

size_t bytesFor(size_t count, size_t element_size) {
  return count * element_size;
}

int featureStorageBlocks(const MapperConfig& config) {
  return config.feature_dim > 0 ? config.max_blocks : 1;
}

int featureStorageNodes(const MapperConfig& config) {
  return config.feature_dim > 0 ? config.feature_grid_voxels : 1;
}

int featureStorageChannels(const MapperConfig& config) {
  return config.feature_dim > 0 ? config.feature_dim : 1;
}

int instanceStorageBlocks(const MapperConfig& config) {
  return config.instance_grid_size > 0 ? config.max_blocks : 1;
}

int instanceStorageNodes(const MapperConfig& config) {
  return config.instance_grid_size > 0 ? config.instance_grid_voxels : 1;
}

int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

int projectiveTextureAtlasColumns(int total_cameras, int image_width) {
  if (total_cameras <= 0) {
    throw std::runtime_error("texture total camera count must be positive");
  }
  if (image_width <= 0) {
    throw std::runtime_error("texture image width must be positive");
  }
  const int max_columns =
      std::max(1, kProjectiveTextureAtlasMaxWidthPx / image_width);
  return std::min(total_cameras, max_columns);
}

uint64_t offsetPtr(uint64_t ptr, size_t byte_offset) {
  return ptr + static_cast<uint64_t>(byte_offset);
}

void launch_pose_refiner_count_valid_depth(
    const CubinKernel& kernel, const float* depth, int pixel_count,
    float depth_minimum, float depth_maximum, int* valid_count,
    CUstream stream) {
  std::vector<void*> args{&depth, &pixel_count, &depth_minimum, &depth_maximum,
                          &valid_count};
  kernel.launch(
      Dim3{static_cast<unsigned int>(cdiv(pixel_count, kWarpBlockDim)), 1, 1},
      Dim3{kWarpBlockDim, 1, 1}, 0, stream, args);
}

void launch_pose_refiner_initialize_state(
    const CubinKernel& kernel, const float* initial_position,
    const float* initial_quaternion, const float* initial_jtj,
    const float* initial_jtr, const float* initial_sum_sq,
    const int* initial_valid_count, const int* valid_depth_count,
    int minimum_valid_depth_pixels, int minimum_valid_correspondences,
    float lambda_initial, float* best_position, float* best_quaternion,
    float* best_jtj, float* best_jtr, float* best_sum_sq, float* best_error,
    int* best_valid_count, float* lambda_damping, float* result_float,
    int* result_int, CUstream stream) {
  std::vector<void*> args{
      &initial_position,      &initial_quaternion,
      &initial_jtj,           &initial_jtr,
      &initial_sum_sq,        &initial_valid_count,
      &valid_depth_count,     &minimum_valid_depth_pixels,
      &minimum_valid_correspondences,
      &lambda_initial,        &best_position,
      &best_quaternion,       &best_jtj,
      &best_jtr,              &best_sum_sq,
      &best_error,            &best_valid_count,
      &lambda_damping,        &result_float,
      &result_int};
  kernel.launch(Dim3{1, 1, 1}, Dim3{1, 1, 1}, 0, stream, args);
}

void launch_pose_refiner_solve_candidate(
    const CubinKernel& kernel, const float* best_position,
    const float* best_quaternion, const float* best_jtj,
    const float* best_jtr, const float* lambda_damping,
    float* candidate_position, float* candidate_quaternion,
    float* predicted_reduction, int* result_int, CUstream stream) {
  std::vector<void*> args{&best_position,       &best_quaternion,
                          &best_jtj,            &best_jtr,
                          &lambda_damping,      &candidate_position,
                          &candidate_quaternion, &predicted_reduction,
                          &result_int};
  kernel.launch(Dim3{1, 1, 1}, Dim3{1, 1, 1}, 0, stream, args);
}

void launch_pose_refiner_accept_candidate(
    const CubinKernel& kernel, const float* candidate_position,
    const float* candidate_quaternion, const float* candidate_jtj,
    const float* candidate_jtr, const float* candidate_sum_sq,
    const int* candidate_valid_count, float* best_position,
    float* best_quaternion, float* best_jtj, float* best_jtr,
    float* best_sum_sq, float* best_error, int* best_valid_count,
    float* lambda_damping, const float* predicted_reduction,
    int minimum_valid_correspondences, float lambda_factor,
    float lambda_minimum, float lambda_maximum, float* result_float,
    int* result_int, CUstream stream) {
  std::vector<void*> args{
      &candidate_position,          &candidate_quaternion,
      &candidate_jtj,               &candidate_jtr,
      &candidate_sum_sq,            &candidate_valid_count,
      &best_position,               &best_quaternion,
      &best_jtj,                    &best_jtr,
      &best_sum_sq,                 &best_error,
      &best_valid_count,            &lambda_damping,
      &predicted_reduction,         &minimum_valid_correspondences,
      &lambda_factor,               &lambda_minimum,
      &lambda_maximum,              &result_float,
      &result_int};
  kernel.launch(Dim3{1, 1, 1}, Dim3{1, 1, 1}, 0, stream, args);
}

void launch_pose_refiner_copy_best_pose(
    const CubinKernel& kernel, const float* best_position,
    const float* best_quaternion, float* output_position,
    float* output_quaternion, CUstream stream) {
  std::vector<void*> args{&best_position, &best_quaternion, &output_position,
                          &output_quaternion};
  kernel.launch(Dim3{1, 1, 1}, Dim3{4, 1, 1}, 0, stream, args);
}

CuroboMapperPoseRefinerConfig normalizedPoseRefinerConfig(
    const CuroboMapperPoseRefinerConfig* requested,
    const MapperConfig& mapper_config) {
  CuroboMapperPoseRefinerConfig cfg =
      requested != nullptr ? *requested : defaultPoseRefinerConfig();
  if (cfg.n_points <= 0) {
    cfg.n_points = defaultPoseRefinerConfig().n_points;
  }
  if (cfg.minimum_valid_depth_pixels <= 0) {
    cfg.minimum_valid_depth_pixels =
        defaultPoseRefinerConfig().minimum_valid_depth_pixels;
  }
  if (cfg.max_iterations <= 0) {
    cfg.max_iterations = defaultPoseRefinerConfig().max_iterations;
  }
  if (cfg.inner_iterations <= 0) {
    cfg.inner_iterations = defaultPoseRefinerConfig().inner_iterations;
  }
  if (cfg.n_samples_per_ray <= 0) {
    cfg.n_samples_per_ray = mapper_config.pose_refiner_n_samples_per_ray;
  }
  if (cfg.tile_block_dim <= 0) {
    cfg.tile_block_dim = defaultPoseRefinerConfig().tile_block_dim;
  }
  if (cfg.depth_minimum_distance <= 0.0f) {
    cfg.depth_minimum_distance =
        defaultPoseRefinerConfig().depth_minimum_distance;
  }
  if (cfg.depth_maximum_distance <= cfg.depth_minimum_distance) {
    cfg.depth_maximum_distance =
        defaultPoseRefinerConfig().depth_maximum_distance;
  }
  if (cfg.minimum_tsdf_weight <= 0.0f) {
    cfg.minimum_tsdf_weight = defaultPoseRefinerConfig().minimum_tsdf_weight;
  }
  if (cfg.distance_threshold <= 0.0f) {
    cfg.distance_threshold = defaultPoseRefinerConfig().distance_threshold;
  }
  if (cfg.lambda_initial <= 0.0f) {
    cfg.lambda_initial = defaultPoseRefinerConfig().lambda_initial;
  }
  if (cfg.lambda_factor <= 1.0f) {
    cfg.lambda_factor = defaultPoseRefinerConfig().lambda_factor;
  }
  if (cfg.lambda_min <= 0.0f) {
    cfg.lambda_min = defaultPoseRefinerConfig().lambda_min;
  }
  if (cfg.lambda_max <= cfg.lambda_min) {
    cfg.lambda_max = defaultPoseRefinerConfig().lambda_max;
  }
  return cfg;
}

}  // namespace

CuroboMapperPoseRefinerConfig defaultPoseRefinerConfig() {
  CuroboMapperPoseRefinerConfig cfg{};
  cfg.n_points = (1280 * 720) / 10;
  cfg.minimum_valid_depth_pixels = 100;
  cfg.max_iterations = 100;
  cfg.inner_iterations = 4;
  cfg.n_samples_per_ray = 1;
  cfg.tile_block_dim = 256;
  cfg.distance_threshold = 0.1f;
  cfg.min_valid_ratio = 0.1f;
  cfg.depth_minimum_distance = 0.1f;
  cfg.depth_maximum_distance = 10.0f;
  cfg.minimum_tsdf_weight = 2.0f;
  cfg.lambda_initial = 1.0e-3f;
  cfg.lambda_factor = 10.0f;
  cfg.lambda_min = 1.0e-7f;
  cfg.lambda_max = 1.0e7f;
  cfg.rho_min = 0.25f;
  return cfg;
}

DeviceBuffer::~DeviceBuffer() { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

void DeviceBuffer::allocate(size_t bytes) {
  release();
  bytes_ = bytes;
  if (bytes_ > 0) {
    CMR_CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
  }
}

void DeviceBuffer::memsetAsync(int value, cudaStream_t stream) {
  if (ptr_ != nullptr && bytes_ > 0) {
    CMR_CUDA_CHECK(cudaMemsetAsync(ptr_, value, bytes_, stream));
  }
}

void DeviceBuffer::release() {
  if (ptr_ != nullptr) {
    cudaFree(ptr_);
    ptr_ = nullptr;
  }
  bytes_ = 0;
}

MapperManifest loadManifest(const std::string& artifact_dir) {
  const auto path = std::filesystem::path(artifact_dir) / "mapper_aot_manifest.json";
  const std::string json = readText(path);
  const std::string config_obj = extractObject(json, "config");
  const std::string kernels_obj = extractObject(json, "kernels");
  const std::string pba_obj = extractObject(json, "pba");

  MapperManifest manifest;
  MapperConfig& cfg = manifest.config;
  cfg.num_cameras = extractInt(config_obj, "num_cameras");
  cfg.image_height = extractInt(config_obj, "image_height");
  cfg.image_width = extractInt(config_obj, "image_width");
  cfg.block_size = extractInt(config_obj, "block_size");
  cfg.block_voxels = cfg.block_size * cfg.block_size * cfg.block_size;
  cfg.num_samples = extractInt(config_obj, "num_samples");
  cfg.pose_refiner_n_samples_per_ray =
      extractIntOr(config_obj, "pose_refiner_n_samples_per_ray", 1);
  cfg.num_block_key_candidates = extractInt(config_obj, "num_block_key_candidates");
  cfg.max_blocks = extractInt(config_obj, "max_blocks");
  cfg.hash_capacity = extractInt(config_obj, "hash_capacity");
  cfg.max_visible_blocks = extractInt(config_obj, "max_visible_blocks_per_integration");
  cfg.max_support_pixels_per_block_camera =
      extractInt(config_obj, "max_support_pixels_per_block_camera");
  cfg.color_grid_size = extractIntOr(config_obj, "color_grid_size", 1);
  cfg.color_grid_voxels = cfg.color_grid_size * cfg.color_grid_size * cfg.color_grid_size;
  cfg.feature_dim = extractIntOr(config_obj, "feature_dim", 0);
  cfg.feature_block_grid_size = extractInt(config_obj, "feature_block_grid_size");
  cfg.feature_grid_voxels =
      cfg.feature_block_grid_size * cfg.feature_block_grid_size *
      cfg.feature_block_grid_size;
  cfg.instance_grid_size = extractIntOr(config_obj, "instance_grid_size", 0);
  cfg.instance_grid_voxels =
      std::max(1, cfg.instance_grid_size * cfg.instance_grid_size *
                      cfg.instance_grid_size);
  cfg.feature_grid_height = extractIntOr(config_obj, "feature_grid_height", 0);
  cfg.feature_grid_width = extractIntOr(config_obj, "feature_grid_width", 0);
  cfg.feature_channels_per_thread =
      extractIntOr(config_obj, "feature_channels_per_thread", 4);
  cfg.max_feature_tile_channels =
      extractIntOr(config_obj, "max_feature_tile_channels", 4096);
  const std::string feature_kernel = extractStringOr(
      config_obj, "feature_integration_kernel",
      "integrate_features_from_support_grouped_kernel");
  cfg.use_tiled_feature_kernel =
      feature_kernel == "integrate_features_from_support_tiled_kernel";
  if (!cfg.use_tiled_feature_kernel &&
      feature_kernel != "integrate_features_from_support_grouped_kernel") {
    throw std::runtime_error("unsupported feature_integration_kernel: " +
                             feature_kernel);
  }
  cfg.texture_num_cameras = extractIntOr(config_obj, "texture_num_cameras",
                                         cfg.num_cameras);
  cfg.texture_camera_image_height =
      extractIntOr(config_obj, "texture_camera_image_height", cfg.image_height);
  cfg.texture_camera_image_width =
      extractIntOr(config_obj, "texture_camera_image_width", cfg.image_width);
  cfg.voxel_size = static_cast<float>(extractDouble(config_obj, "voxel_size"));
  cfg.esdf_voxel_size = static_cast<float>(extractDouble(config_obj, "esdf_voxel_size"));
  cfg.truncation_distance =
      static_cast<float>(extractDouble(config_obj, "truncation_distance"));
  cfg.depth_minimum_distance =
      static_cast<float>(extractDouble(config_obj, "depth_minimum_distance"));
  cfg.depth_maximum_distance =
      static_cast<float>(extractDouble(config_obj, "depth_maximum_distance"));
  cfg.minimum_tsdf_weight =
      static_cast<float>(extractDouble(config_obj, "minimum_tsdf_weight"));
  cfg.adjacent_skip_steps =
      static_cast<float>(extractDouble(config_obj, "adjacent_skip_steps"));
  cfg.decay_factor = static_cast<float>(extractDouble(config_obj, "decay_factor"));
  cfg.frustum_decay_factor =
      static_cast<float>(extractDouble(config_obj, "frustum_decay_factor"));
  cfg.accumulator_w_max =
      static_cast<float>(extractDouble(config_obj, "accumulator_w_max"));

  const auto grid_shape = extractIntArray(config_obj, "grid_shape");
  if (grid_shape.size() != 3) {
    throw std::runtime_error("grid_shape must have 3 values");
  }
  cfg.grid_D = grid_shape[0];
  cfg.grid_H = grid_shape[1];
  cfg.grid_W = grid_shape[2];

  const auto esdf_grid_shape = extractIntArray(config_obj, "esdf_grid_shape");
  if (esdf_grid_shape.size() != 3) {
    throw std::runtime_error("esdf_grid_shape must have 3 values");
  }
  cfg.esdf_nx = esdf_grid_shape[0];
  cfg.esdf_ny = esdf_grid_shape[1];
  cfg.esdf_nz = esdf_grid_shape[2];

  const auto origin = extractFloatArray(config_obj, "origin");
  if (origin.size() != 3) {
    throw std::runtime_error("origin must have 3 values");
  }
  cfg.origin[0] = origin[0];
  cfg.origin[1] = origin[1];
  cfg.origin[2] = origin[2];

  requirePositive("num_cameras", cfg.num_cameras);
  requirePositive("image_height", cfg.image_height);
  requirePositive("image_width", cfg.image_width);
  requirePositive("block_size", cfg.block_size);
  requirePositive("max_blocks", cfg.max_blocks);
  requirePositive("hash_capacity", cfg.hash_capacity);
  requirePositive("max_visible_blocks_per_integration", cfg.max_visible_blocks);
  requirePositive("max_support_pixels_per_block_camera",
                  cfg.max_support_pixels_per_block_camera);
  requirePositive("color_grid_size", cfg.color_grid_size);
  requirePositive("color_grid_voxels", cfg.color_grid_voxels);
  requireNonNegative("feature_dim", cfg.feature_dim);
  requirePositive("feature_block_grid_size", cfg.feature_block_grid_size);
  requirePositive("feature_grid_voxels", cfg.feature_grid_voxels);
  requireNonNegative("instance_grid_size", cfg.instance_grid_size);
  if (cfg.instance_grid_size > cfg.block_size) {
    throw std::runtime_error("instance_grid_size must not exceed block_size");
  }
  requirePositive("feature_channels_per_thread", cfg.feature_channels_per_thread);
  requirePositive("max_feature_tile_channels", cfg.max_feature_tile_channels);
  if (cfg.feature_dim > 0) {
    requirePositive("feature_grid_height", cfg.feature_grid_height);
    requirePositive("feature_grid_width", cfg.feature_grid_width);
  }
  requirePositive("texture_num_cameras", cfg.texture_num_cameras);
  requirePositive("texture_camera_image_height", cfg.texture_camera_image_height);
  requirePositive("texture_camera_image_width", cfg.texture_camera_image_width);

  const std::vector<std::string> required_kernel_names = {
      "mark_blocks_in_frustum_kernel",
      "recycle_empty_blocks_kernel",
      "apply_decay_and_sum_kernel",
      "compute_block_keys_only_kernel",
      "allocate_visible_blocks_from_keys_kernel",
      "build_support_pixels_from_keys_kernel",
      "clear_new_blocks_kernel",
      "clear_new_block_grid_rgb_kernel",
      "clear_new_block_features_kernel",
      "clear_new_block_instances_kernel",
      "clear_instance_slot_kernel",
      "mark_instance_removal_blocks_kernel",
      "remove_instance_geometry_kernel",
      "integrate_voxels_kernel",
      "integrate_block_grid_rgb_kernel",
      "integrate_instance_grid_kernel",
      "rescale_block_accumulators_kernel",
      "rescale_block_grid_rgb_kernel",
      "count_occupied_voxels_kernel",
      "extract_occupied_voxels_kernel",
      "set_point_query_kernel",
      "set_point_target_class_kernel",
      "count_point_edges_kernel",
      "sample_point_edges_kernel",
      "compact_surface_points_kernel",
      "seed_esdf_sites_gather_kernel",
      "compute_esdf_from_min_tsdf_kernel",
      "raycast_block_sparse_kernel",
      "raycast_block_sparse_color_kernel",
      "append_active_blocks_kernel",
      "count_surface_cubes_from_blocks_kernel",
      "append_surface_cubes_from_blocks_kernel",
      "count_total_triangles_kernel",
      "generate_mesh_kernel",
      "project_mesh_uvs_kernel",
  };
  for (const std::string& name : required_kernel_names) {
    manifest.kernels.emplace(name, parseKernelObject(extractObject(kernels_obj, name)));
  }
  if (cfg.feature_dim > 0) {
    const std::vector<std::string> feature_kernel_names = {
        cfg.use_tiled_feature_kernel
            ? "integrate_features_from_support_tiled_kernel"
            : "integrate_features_from_support_grouped_kernel",
        "raycast_block_sparse_feature_kernel",
        "gather_rendered_features_fp32_kernel",
        "gather_rendered_features_fp16_kernel",
        "prepare_semantic_node_matches_kernel",
        "prepare_semantic_argmax_node_matches_kernel",
        "prepare_semantic_block_summary_kernel",
        "count_occupied_voxels_semantic_kernel",
        "extract_occupied_voxels_semantic_kernel",
        "seed_esdf_sites_gather_semantic_kernel",
        "compute_esdf_from_min_tsdf_semantic_kernel",
        "raycast_block_sparse_semantic_kernel",
        "raycast_block_sparse_color_semantic_kernel",
        "raycast_block_sparse_feature_semantic_kernel",
        "append_active_blocks_semantic_kernel",
        "append_surface_cubes_from_blocks_semantic_kernel",
    };
    for (const std::string& name : feature_kernel_names) {
      manifest.kernels.emplace(name,
                               parseKernelObject(extractObject(kernels_obj, name)));
    }
  }
  try {
    manifest.kernels.emplace(kPoseAlignmentKernelName,
                             parseKernelObject(extractObject(kernels_obj,
                                                             kPoseAlignmentKernelName)));
  } catch (const std::runtime_error&) {
    // Artifacts generated before pose refinement support do not contain this
    // kernel. The refine-pose API checks and reports this explicitly.
  }
  try {
    for (const char* name : kPoseLmKernelNames) {
      manifest.kernels.emplace(name, parseKernelObject(extractObject(kernels_obj, name)));
    }
  } catch (const std::runtime_error&) {
    for (const char* name : kPoseLmKernelNames) {
      manifest.kernels.erase(name);
    }
  }

  manifest.pba_flood_z = parseKernelObject(extractObject(pba_obj, "flood_z"));
  manifest.pba_maurer_axis = parseKernelObject(extractObject(pba_obj, "maurer_axis"));
  manifest.pba_color_axis = parseKernelObject(extractObject(pba_obj, "color_axis"));
  const std::string mc_tables_obj = extractObject(json, "mc_tables");
  manifest.mc_num_triangles = extractIntArray(mc_tables_obj, "num_triangles");
  manifest.mc_triangle_table = extractIntArray(mc_tables_obj, "triangle_table");
  if (manifest.mc_num_triangles.size() != 256) {
    throw std::runtime_error("mc_tables.num_triangles must contain 256 values");
  }
  if (manifest.mc_triangle_table.size() != 256 * 16) {
    throw std::runtime_error("mc_tables.triangle_table must contain 4096 values");
  }
  return manifest;
}

MapperRuntimeImpl::MapperRuntimeImpl(std::string artifact_dir, int device_ordinal)
    : artifact_dir_(std::move(artifact_dir)), device_ordinal_(device_ordinal) {
  CMR_CU_CHECK(cuInit(0));
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  manifest_ = loadManifest(artifact_dir_);
  config_ = manifest_.config;
  allocateBuffers();
  loadKernels();
  reset(0);
}

MapperRuntimeImpl::~MapperRuntimeImpl() { destroyEsdfGraph(); }

void MapperRuntimeImpl::allocateBuffers() {
  const size_t max_blocks = static_cast<size_t>(config_.max_blocks);
  const size_t block_voxels = static_cast<size_t>(config_.block_voxels);
  const size_t max_visible = static_cast<size_t>(config_.max_visible_blocks);
  const size_t num_cameras = static_cast<size_t>(config_.num_cameras);
  const size_t support_capacity =
      static_cast<size_t>(config_.max_support_pixels_per_block_camera);
  const size_t color_grid_voxels = static_cast<size_t>(config_.color_grid_voxels);
  const size_t esdf_voxels =
      static_cast<size_t>(config_.esdf_nx) * static_cast<size_t>(config_.esdf_ny) *
      static_cast<size_t>(config_.esdf_nz);

  buffers_.hash_table.allocate(bytesFor(config_.hash_capacity, sizeof(int64_t)));
  buffers_.block_data.allocate(bytesFor(max_blocks * block_voxels * 2, sizeof(uint16_t)));
  buffers_.block_grid_rgb.allocate(
      bytesFor(max_blocks * color_grid_voxels * 4, sizeof(uint16_t)));
  buffers_.block_features.allocate(
      bytesFor(static_cast<size_t>(featureStorageBlocks(config_)) *
                   static_cast<size_t>(featureStorageNodes(config_)) *
                   static_cast<size_t>(featureStorageChannels(config_)),
               sizeof(uint16_t)));
  buffers_.block_feature_weight.allocate(
      bytesFor(static_cast<size_t>(featureStorageBlocks(config_)) *
                   static_cast<size_t>(featureStorageNodes(config_)),
               sizeof(uint16_t)));
  buffers_.block_instance_state.allocate(
      bytesFor(static_cast<size_t>(instanceStorageBlocks(config_)) *
                   static_cast<size_t>(instanceStorageNodes(config_)),
               sizeof(int16_t)));
  buffers_.static_block_data.allocate(bytesFor(1, sizeof(uint16_t)));
  buffers_.block_coords.allocate(bytesFor(max_blocks * 3, sizeof(int32_t)));
  buffers_.block_to_hash_slot.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.free_list.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.free_count.allocate(sizeof(int32_t));
  buffers_.num_allocated.allocate(sizeof(int32_t));
  buffers_.allocation_failures.allocate(sizeof(int32_t));
  buffers_.block_sums.allocate(bytesFor(max_blocks, sizeof(float)));
  buffers_.static_block_sums.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.frustum_flags.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.new_blocks.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.new_block_count.allocate(sizeof(int32_t));
  buffers_.recycle_count.allocate(sizeof(int32_t));

  buffers_.block_keys.allocate(
      bytesFor(config_.num_block_key_candidates, sizeof(int64_t)));
  buffers_.pool_indices.allocate(bytesFor(max_visible, sizeof(int32_t)));
  buffers_.visible_epoch.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.visible_count.allocate(sizeof(int32_t));
  buffers_.pool_to_visible_slot.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.support_counts.allocate(bytesFor(max_visible * num_cameras, sizeof(int32_t)));
  buffers_.support_pixels.allocate(
      bytesFor(max_visible * num_cameras * support_capacity, sizeof(int32_t)));
  buffers_.support_overflow_count.allocate(sizeof(int32_t));
  buffers_.occupied_count.allocate(sizeof(int32_t));
  buffers_.semantic_target.allocate(
      bytesFor(static_cast<size_t>(std::max(1, config_.feature_dim)), sizeof(float)));
  buffers_.semantic_node_match.allocate(
      bytesFor(static_cast<size_t>(featureStorageBlocks(config_)) *
                   static_cast<size_t>(featureStorageNodes(config_)),
               sizeof(uint8_t)));
  buffers_.semantic_block_has_keep.allocate(
      bytesFor(static_cast<size_t>(featureStorageBlocks(config_)), sizeof(uint8_t)));
  buffers_.instance_removal_block_mask.allocate(
      bytesFor(max_blocks, sizeof(uint8_t)));

  buffers_.esdf_site_index.allocate(bytesFor(esdf_voxels, sizeof(int32_t)));
  buffers_.esdf_pba_buffer.allocate(bytesFor(esdf_voxels, sizeof(int32_t)));
  buffers_.esdf_dist_field.allocate(bytesFor(esdf_voxels, sizeof(uint16_t)));
  buffers_.esdf_origin.allocate(bytesFor(3, sizeof(float)));
  buffers_.esdf_voxel_size.allocate(sizeof(float));
  buffers_.esdf_query.allocate(2 * sizeof(uint64_t));

  buffers_.pose_valid_depth_count.allocate(sizeof(int32_t));
  buffers_.pose_best_position.allocate(bytesFor(3, sizeof(float)));
  buffers_.pose_best_quaternion.allocate(bytesFor(4, sizeof(float)));
  buffers_.pose_candidate_position.allocate(bytesFor(3, sizeof(float)));
  buffers_.pose_candidate_quaternion.allocate(bytesFor(4, sizeof(float)));
  buffers_.pose_best_jtj.allocate(bytesFor(36, sizeof(float)));
  buffers_.pose_best_jtr.allocate(bytesFor(6, sizeof(float)));
  buffers_.pose_best_sum_sq.allocate(sizeof(float));
  buffers_.pose_best_error.allocate(sizeof(float));
  buffers_.pose_best_valid_count.allocate(sizeof(int32_t));
  buffers_.pose_candidate_jtj.allocate(bytesFor(36, sizeof(float)));
  buffers_.pose_candidate_jtr.allocate(bytesFor(6, sizeof(float)));
  buffers_.pose_candidate_sum_sq.allocate(sizeof(float));
  buffers_.pose_candidate_valid_count.allocate(sizeof(int32_t));
  buffers_.pose_lambda.allocate(sizeof(float));
  buffers_.pose_pred_reduction.allocate(sizeof(float));
  buffers_.pose_result.allocate(sizeof(CuroboMapperPoseRefinerResult));
}

CubinKernel MapperRuntimeImpl::makeKernel(const KernelManifest& manifest) const {
  const auto cubin_path = (std::filesystem::path(artifact_dir_) / manifest.cubin).string();
  return CubinKernel(cubin_path, manifest.symbol);
}

const KernelManifest& MapperRuntimeImpl::kernelManifest(const std::string& name) const {
  const auto it = manifest_.kernels.find(name);
  if (it == manifest_.kernels.end()) {
    throw std::runtime_error("kernel missing from manifest: " + name);
  }
  return it->second;
}

void MapperRuntimeImpl::loadKernels() {
  kernels_.set_point_query = makeKernel(kernelManifest("set_point_query_kernel"));
  kernels_.set_point_target_class = makeKernel(kernelManifest("set_point_target_class_kernel"));
  kernels_.count_point_edges = makeKernel(kernelManifest("count_point_edges_kernel"));
  kernels_.sample_point_edges = makeKernel(kernelManifest("sample_point_edges_kernel"));
  kernels_.compact_surface_points = makeKernel(kernelManifest("compact_surface_points_kernel"));
  kernels_.set_point_query.load();
  kernels_.set_point_target_class.load();
  kernels_.count_point_edges.load();
  kernels_.sample_point_edges.load();
  kernels_.compact_surface_points.load();
  kernels_.compute_block_keys =
      makeKernel(kernelManifest("compute_block_keys_only_kernel"));
  kernels_.mark_blocks_in_frustum =
      makeKernel(kernelManifest("mark_blocks_in_frustum_kernel"));
  kernels_.allocate_visible_blocks =
      makeKernel(kernelManifest("allocate_visible_blocks_from_keys_kernel"));
  kernels_.build_support_pixels =
      makeKernel(kernelManifest("build_support_pixels_from_keys_kernel"));
  kernels_.clear_new_blocks = makeKernel(kernelManifest("clear_new_blocks_kernel"));
  kernels_.clear_new_block_grid_rgb =
      makeKernel(kernelManifest("clear_new_block_grid_rgb_kernel"));
  kernels_.clear_new_block_features =
      makeKernel(kernelManifest("clear_new_block_features_kernel"));
  kernels_.clear_new_block_instances =
      makeKernel(kernelManifest("clear_new_block_instances_kernel"));
  kernels_.clear_instance_slot =
      makeKernel(kernelManifest("clear_instance_slot_kernel"));
  kernels_.mark_instance_removal_blocks =
      makeKernel(kernelManifest("mark_instance_removal_blocks_kernel"));
  kernels_.remove_instance_geometry =
      makeKernel(kernelManifest("remove_instance_geometry_kernel"));
  kernels_.integrate_voxels = makeKernel(kernelManifest("integrate_voxels_kernel"));
  kernels_.integrate_block_grid_rgb =
      makeKernel(kernelManifest("integrate_block_grid_rgb_kernel"));
  kernels_.integrate_instances =
      makeKernel(kernelManifest("integrate_instance_grid_kernel"));
  if (config_.feature_dim > 0) {
    kernels_.integrate_features = makeKernel(kernelManifest(
        config_.use_tiled_feature_kernel
            ? "integrate_features_from_support_tiled_kernel"
            : "integrate_features_from_support_grouped_kernel"));
  }
  kernels_.rescale_accumulators =
      makeKernel(kernelManifest("rescale_block_accumulators_kernel"));
  kernels_.rescale_block_grid_rgb =
      makeKernel(kernelManifest("rescale_block_grid_rgb_kernel"));
  kernels_.apply_decay_and_sum =
      makeKernel(kernelManifest("apply_decay_and_sum_kernel"));
  kernels_.recycle_empty_blocks =
      makeKernel(kernelManifest("recycle_empty_blocks_kernel"));
  kernels_.count_occupied = makeKernel(kernelManifest("count_occupied_voxels_kernel"));
  kernels_.extract_occupied =
      makeKernel(kernelManifest("extract_occupied_voxels_kernel"));
  kernels_.seed_esdf_gather =
      makeKernel(kernelManifest("seed_esdf_sites_gather_kernel"));
  kernels_.compute_esdf_distance =
      makeKernel(kernelManifest("compute_esdf_from_min_tsdf_kernel"));
  kernels_.raycast_depth =
      makeKernel(kernelManifest("raycast_block_sparse_kernel"));
  kernels_.raycast_color =
      makeKernel(kernelManifest("raycast_block_sparse_color_kernel"));
  if (config_.feature_dim > 0) {
    kernels_.raycast_features =
        makeKernel(kernelManifest("raycast_block_sparse_feature_kernel"));
    kernels_.gather_features_fp32 =
        makeKernel(kernelManifest("gather_rendered_features_fp32_kernel"));
    kernels_.gather_features_fp16 =
        makeKernel(kernelManifest("gather_rendered_features_fp16_kernel"));
    kernels_.prepare_semantic_node_matches =
        makeKernel(kernelManifest("prepare_semantic_node_matches_kernel"));
    kernels_.prepare_semantic_argmax_node_matches = makeKernel(
        kernelManifest("prepare_semantic_argmax_node_matches_kernel"));
    kernels_.prepare_semantic_block_summary =
        makeKernel(kernelManifest("prepare_semantic_block_summary_kernel"));
    kernels_.count_occupied_semantic =
        makeKernel(kernelManifest("count_occupied_voxels_semantic_kernel"));
    kernels_.extract_occupied_semantic =
        makeKernel(kernelManifest("extract_occupied_voxels_semantic_kernel"));
    kernels_.seed_esdf_gather_semantic =
        makeKernel(kernelManifest("seed_esdf_sites_gather_semantic_kernel"));
    kernels_.compute_esdf_distance_semantic =
        makeKernel(kernelManifest("compute_esdf_from_min_tsdf_semantic_kernel"));
    kernels_.raycast_depth_semantic =
        makeKernel(kernelManifest("raycast_block_sparse_semantic_kernel"));
    kernels_.raycast_color_semantic =
        makeKernel(kernelManifest("raycast_block_sparse_color_semantic_kernel"));
    kernels_.raycast_features_semantic = makeKernel(
        kernelManifest("raycast_block_sparse_feature_semantic_kernel"));
    kernels_.append_active_blocks_semantic =
        makeKernel(kernelManifest("append_active_blocks_semantic_kernel"));
    kernels_.append_surface_cubes_from_blocks_semantic = makeKernel(
        kernelManifest("append_surface_cubes_from_blocks_semantic_kernel"));
  }
  kernels_.append_active_blocks =
      makeKernel(kernelManifest("append_active_blocks_kernel"));
  kernels_.count_surface_cubes_from_blocks =
      makeKernel(kernelManifest("count_surface_cubes_from_blocks_kernel"));
  kernels_.append_surface_cubes_from_blocks =
      makeKernel(kernelManifest("append_surface_cubes_from_blocks_kernel"));
  kernels_.count_total_triangles =
      makeKernel(kernelManifest("count_total_triangles_kernel"));
  kernels_.generate_mesh =
      makeKernel(kernelManifest("generate_mesh_kernel"));
  kernels_.project_mesh_uvs =
      makeKernel(kernelManifest("project_mesh_uvs_kernel"));
  if (manifest_.kernels.find(kPoseAlignmentKernelName) != manifest_.kernels.end()) {
    kernels_.pose_refine_alignment = makeKernel(kernelManifest(kPoseAlignmentKernelName));
  }
  if (has_pose_lm_kernels(manifest_.kernels)) {
    kernels_.pose_refine_count_valid_depth =
        makeKernel(kernelManifest(kPoseCountValidDepthKernelName));
    kernels_.pose_refine_initialize_state =
        makeKernel(kernelManifest(kPoseInitializeStateKernelName));
    kernels_.pose_refine_solve_candidate =
        makeKernel(kernelManifest(kPoseSolveCandidateKernelName));
    kernels_.pose_refine_accept_candidate =
        makeKernel(kernelManifest(kPoseAcceptCandidateKernelName));
    kernels_.pose_refine_copy_best_pose =
        makeKernel(kernelManifest(kPoseCopyBestPoseKernelName));
  }
  kernels_.pba_flood_z = makeKernel(manifest_.pba_flood_z);
  kernels_.pba_maurer_axis = makeKernel(manifest_.pba_maurer_axis);
  kernels_.pba_color_axis = makeKernel(manifest_.pba_color_axis);

  kernels_.compute_block_keys.load();
  kernels_.mark_blocks_in_frustum.load();
  kernels_.allocate_visible_blocks.load();
  kernels_.build_support_pixels.load();
  kernels_.clear_new_blocks.load();
  kernels_.clear_new_block_grid_rgb.load();
  kernels_.clear_new_block_features.load();
  kernels_.clear_new_block_instances.load();
  kernels_.clear_instance_slot.load();
  kernels_.mark_instance_removal_blocks.load();
  kernels_.remove_instance_geometry.load();
  kernels_.integrate_voxels.load();
  kernels_.integrate_block_grid_rgb.load();
  kernels_.integrate_instances.load();
  if (config_.feature_dim > 0) {
    kernels_.integrate_features.load();
  }
  kernels_.rescale_accumulators.load();
  kernels_.rescale_block_grid_rgb.load();
  kernels_.apply_decay_and_sum.load();
  kernels_.recycle_empty_blocks.load();
  kernels_.count_occupied.load();
  kernels_.extract_occupied.load();
  kernels_.seed_esdf_gather.load();
  kernels_.compute_esdf_distance.load();
  kernels_.raycast_depth.load();
  kernels_.raycast_color.load();
  if (config_.feature_dim > 0) {
    kernels_.raycast_features.load();
    kernels_.gather_features_fp32.load();
    kernels_.gather_features_fp16.load();
    kernels_.prepare_semantic_node_matches.load();
    kernels_.prepare_semantic_argmax_node_matches.load();
    kernels_.prepare_semantic_block_summary.load();
    kernels_.count_occupied_semantic.load();
    kernels_.extract_occupied_semantic.load();
    kernels_.seed_esdf_gather_semantic.load();
    kernels_.compute_esdf_distance_semantic.load();
    kernels_.raycast_depth_semantic.load();
    kernels_.raycast_color_semantic.load();
    kernels_.raycast_features_semantic.load();
    kernels_.append_active_blocks_semantic.load();
    kernels_.append_surface_cubes_from_blocks_semantic.load();
  }
  kernels_.append_active_blocks.load();
  kernels_.count_surface_cubes_from_blocks.load();
  kernels_.append_surface_cubes_from_blocks.load();
  kernels_.count_total_triangles.load();
  kernels_.generate_mesh.load();
  kernels_.project_mesh_uvs.load();
  if (manifest_.kernels.find(kPoseAlignmentKernelName) != manifest_.kernels.end()) {
    kernels_.pose_refine_alignment.load();
  }
  if (has_pose_lm_kernels(manifest_.kernels)) {
    kernels_.pose_refine_count_valid_depth.load();
    kernels_.pose_refine_initialize_state.load();
    kernels_.pose_refine_solve_candidate.load();
    kernels_.pose_refine_accept_candidate.load();
    kernels_.pose_refine_copy_best_pose.load();
  }
  kernels_.pba_flood_z.load();
  kernels_.pba_maurer_axis.load();
  kernels_.pba_color_axis.load();
}

Dim3 MapperRuntimeImpl::gridFor(size_t linear_size, unsigned int block_dim) const {
  const size_t blocks = std::max<size_t>(1, (linear_size + block_dim - 1) / block_dim);
  return Dim3{static_cast<unsigned int>(blocks), 1, 1};
}

BlockSparseTSDFWarp MapperRuntimeImpl::tsdfWarp() const {
  BlockSparseTSDFWarp tsdf{};
  tsdf.hash_table = array1<wp::int64>(buffers_.hash_table.ptr(), config_.hash_capacity);
  tsdf.block_data = array3<wp::float16>(buffers_.block_data.ptr(), config_.max_blocks,
                                        config_.block_voxels, 2);
  tsdf.block_grid_rgb = array3<wp::float16>(buffers_.block_grid_rgb.ptr(),
                                            config_.max_blocks,
                                            config_.color_grid_voxels, 4);
  tsdf.block_features = array3<wp::float16>(buffers_.block_features.ptr(),
                                            featureStorageBlocks(config_),
                                            featureStorageNodes(config_),
                                            featureStorageChannels(config_));
  tsdf.block_feature_weight = array2<wp::float16>(buffers_.block_feature_weight.ptr(),
                                                  featureStorageBlocks(config_),
                                                  featureStorageNodes(config_));
  tsdf.block_instance_state = array2<wp::int16>(
      buffers_.block_instance_state.ptr(), instanceStorageBlocks(config_),
      instanceStorageNodes(config_));
  tsdf.static_block_data = array2<wp::float16>(buffers_.static_block_data.ptr(),
                                               static_seed_.empty() ? 1 : config_.max_blocks,
                                               static_seed_.empty() ? 1 : config_.block_voxels);
  tsdf.has_dynamic = true;
  tsdf.has_static = !static_seed_.empty();
  tsdf.has_features = config_.feature_dim > 0;
  tsdf.has_instances = config_.instance_grid_size > 0;
  tsdf.feature_dim = config_.feature_dim;
  tsdf.color_grid_size = config_.color_grid_size;
  tsdf.feature_block_grid_size = config_.feature_block_grid_size;
  tsdf.instance_grid_size = config_.instance_grid_size;
  tsdf.block_coords = array1<wp::int32>(buffers_.block_coords.ptr(), config_.max_blocks * 3);
  tsdf.block_to_hash_slot =
      array1<wp::int32>(buffers_.block_to_hash_slot.ptr(), config_.max_blocks);
  tsdf.free_list = array1<wp::int32>(buffers_.free_list.ptr(), config_.max_blocks);
  tsdf.free_count = array1<wp::int32>(buffers_.free_count.ptr(), 1);
  tsdf.num_allocated = array1<wp::int32>(buffers_.num_allocated.ptr(), 1);
  tsdf.allocation_failures = array1<wp::int32>(buffers_.allocation_failures.ptr(), 1);
  tsdf.block_sums = array1<wp::float32>(buffers_.block_sums.ptr(), config_.max_blocks);
  tsdf.static_block_sums =
      array1<wp::int32>(buffers_.static_block_sums.ptr(), config_.max_blocks);
  tsdf.new_blocks = array1<wp::int32>(buffers_.new_blocks.ptr(), config_.max_blocks);
  tsdf.new_block_count = array1<wp::int32>(buffers_.new_block_count.ptr(), 1);
  tsdf.recycle_count = array1<wp::int32>(buffers_.recycle_count.ptr(), 1);
  tsdf.origin =
      wp::vec_t<3, wp::float32>(config_.origin[0], config_.origin[1], config_.origin[2]);
  tsdf.voxel_size = config_.voxel_size;
  tsdf.hash_capacity = config_.hash_capacity;
  tsdf.max_blocks = config_.max_blocks;
  tsdf.truncation_distance = config_.truncation_distance;
  tsdf.block_size = config_.block_size;
  tsdf.grid_W = config_.grid_W;
  tsdf.grid_H = config_.grid_H;
  tsdf.grid_D = config_.grid_D;
  return tsdf;
}

PreparedSemanticFilterWarp MapperRuntimeImpl::prepareSemanticFilter(
    const CuroboMapperSemanticFilter& semantic_filter, cudaStream_t cuda_stream, CUstream cu_stream,
    const uint64_t* esdf_query) {
  if (config_.feature_dim <= 0) {
    throw std::runtime_error(
        "semantic filtering requires an artifact exported with feature_dim > 0");
  }
  const bool categorical = semantic_filter.target_class >= 0;
  if (semantic_filter.exclude_instance_slot < 0 || semantic_filter.exclude_instance_slot > 255 ||
      (semantic_filter.exclude_instance_slot != 0 &&
       (config_.instance_grid_size != config_.block_size ||
        config_.feature_block_grid_size != config_.block_size))) {
    throw std::invalid_argument("instance exclusion requires a slot in 1..255 and per-voxel semantic filtering");
  }
  if (!categorical && semantic_filter.target == 0) {
    throw std::runtime_error(
        "semantic filter requires target or non-negative target_class");
  }
  if (categorical && semantic_filter.target_class >= config_.feature_dim) {
    throw std::runtime_error("semantic filter target_class is out of range");
  }
  if (categorical && semantic_filter.use_min_similarity != 0) {
    throw std::runtime_error(
        "categorical semantic filter does not support cosine similarity");
  }
  if (semantic_filter.use_min_similarity == 0 &&
      semantic_filter.require_argmax == 0) {
    throw std::runtime_error(
        "semantic filter requires cosine similarity or argmax matching");
  }
  if (semantic_filter.use_min_similarity != 0 &&
      (!std::isfinite(semantic_filter.min_similarity) ||
       semantic_filter.min_similarity < -1.0f ||
       semantic_filter.min_similarity > 1.0f)) {
    throw std::runtime_error(
        "semantic filter min_similarity must be finite and in [-1, 1]");
  }

  std::vector<float> target(static_cast<size_t>(config_.feature_dim), 0.0F);
  if (!categorical) {
    CMR_CUDA_CHECK(cudaMemcpyAsync(
        target.data(), reinterpret_cast<const void*>(semantic_filter.target),
        bytesFor(target.size(), sizeof(float)), cudaMemcpyDeviceToHost,
        cuda_stream));
    CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  }
  float norm_squared = 0.0f;
  int target_argmax = categorical ? semantic_filter.target_class : 0;
  for (int channel = 0; channel < config_.feature_dim; ++channel) {
    if (!categorical && !std::isfinite(target[static_cast<size_t>(channel)])) {
      throw std::runtime_error("semantic filter target contains a non-finite value");
    }
    const float value = target[static_cast<size_t>(channel)];
    norm_squared += value * value;
    if (!categorical && value > target[static_cast<size_t>(target_argmax)]) {
      target_argmax = channel;
    }
  }
  if (semantic_filter.use_min_similarity != 0) {
    if (!(norm_squared > 0.0f) || !std::isfinite(norm_squared)) {
      throw std::runtime_error(
          "cosine semantic filtering requires a finite, nonzero target norm");
    }
    const float inverse_norm = 1.0f / std::sqrt(norm_squared);
    for (float& value : target) {
      value *= inverse_norm;
    }
  }
  if (!categorical) {
    CMR_CUDA_CHECK(cudaMemcpyAsync(
        buffers_.semantic_target.raw(), target.data(),
        bytesFor(target.size(), sizeof(float)), cudaMemcpyHostToDevice,
        cuda_stream));
  }

  auto tsdf = tsdfWarp();
  auto target_array =
      array1<wp::float32>(buffers_.semantic_target.ptr(), config_.feature_dim);
  auto node_match = array2<wp::uint8>(buffers_.semantic_node_match.ptr(),
                                      config_.max_blocks,
                                      config_.feature_grid_voxels);
  auto block_has_keep = array1<wp::uint8>(
      buffers_.semantic_block_has_keep.ptr(), config_.max_blocks);
  wp::float32 min_similarity = semantic_filter.use_min_similarity != 0
                                   ? semantic_filter.min_similarity
                                   : -1.0f;
  wp::int32 target_argmax_w = target_argmax;
  wp::int32 use_cosine = semantic_filter.use_min_similarity != 0 ? 1 : 0;
  wp::int32 use_argmax = semantic_filter.require_argmax != 0 ? 1 : 0;
  constexpr unsigned int kSemanticBlockDim = 64;
  if (categorical) {
    auto dim = launchBounds2(semanticPreparationBlockCapacity(),
                             config_.feature_grid_voxels);
    std::vector<void*> args{&dim, &tsdf, &target_argmax_w, &node_match};
    kernels_.prepare_semantic_argmax_node_matches.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("prepare_semantic_argmax_node_matches_kernel").shared_mem,
        cu_stream, args);
  } else {
    auto dim = launchBounds3(semanticPreparationBlockCapacity(),
                             config_.feature_grid_voxels, kSemanticBlockDim);
    std::vector<void*> args{&dim,          &tsdf,        &target_array,
                            &min_similarity, &target_argmax_w, &use_cosine,
                            &use_argmax,   &node_match};
    kernels_.prepare_semantic_node_matches.launch(
        gridFor(dim.size, kSemanticBlockDim),
        Dim3{kSemanticBlockDim, 1, 1},
        kernelManifest("prepare_semantic_node_matches_kernel").shared_mem,
        cu_stream, args);
  }
  {
    if (semantic_filter.exclude_instance_slot != 0 ||
        (esdf_query != nullptr &&
         config_.instance_grid_size == config_.block_size &&
         config_.feature_block_grid_size == config_.block_size)) {
      exclude_instance_nodes(reinterpret_cast<const uint16_t*>(buffers_.block_instance_state.ptr()),
                             reinterpret_cast<uint8_t*>(buffers_.semantic_node_match.ptr()),
                             reinterpret_cast<const int*>(buffers_.num_allocated.ptr()),
                             semanticPreparationBlockCapacity(), config_.block_voxels,
                             semantic_filter.exclude_instance_slot, semantic_filter.exclude != 0,
                             cuda_stream, esdf_query);
    }
    auto dim = launchBounds2(semanticPreparationBlockCapacity(),
                             kSemanticBlockDim);
    wp::int32 exclude = semantic_filter.exclude != 0 ? 1 : 0;
    std::vector<void*> args{&dim, &tsdf, &node_match, &exclude,
                            &block_has_keep};
    kernels_.prepare_semantic_block_summary.launch(
        gridFor(dim.size, kSemanticBlockDim),
        Dim3{kSemanticBlockDim, 1, 1},
        kernelManifest("prepare_semantic_block_summary_kernel").shared_mem,
        cu_stream, args);
  }

  PreparedSemanticFilterWarp prepared{};
  prepared.node_match = node_match;
  prepared.block_has_keep = block_has_keep;
  prepared.exclude = semantic_filter.exclude != 0;
  return prepared;
}

void MapperRuntimeImpl::loadStaticSeed(const std::string &path, uint64_t stream_handle) {
  if (frame_count_ != 0 || cached_num_allocated_ != 0 || !static_seed_.empty())
    throw std::runtime_error("Static seed must be loaded into an empty new mapper");
  std::ifstream file(path, std::ios::binary);
  char magic[8];
  int32_t header[8];
  float geometry[5];
  file.read(magic, 8);
  file.read(reinterpret_cast<char *>(header), sizeof(header));
  file.read(reinterpret_cast<char *>(geometry), sizeof(geometry));
  if (!file || std::string(magic, 8) != "STTSDF01" || header[0] != config_.hash_capacity ||
      header[1] != config_.max_blocks || header[2] != config_.block_size || header[3] <= 0 ||
      header[3] > config_.max_blocks || header[4] != config_.grid_D ||
      header[5] != config_.grid_H || header[6] != config_.grid_W ||
      header[7] != config_.color_grid_voxels ||
      std::abs(geometry[0] - config_.voxel_size) > 1.e-7F ||
      std::abs(geometry[1] - config_.truncation_distance) > 1.e-7F)
    throw std::runtime_error("Static seed geometry does not match mapper");
  for (int axis = 0; axis < 3; ++axis)
    if (std::abs(geometry[2 + axis] - config_.origin[axis]) > 1.e-6F)
      throw std::runtime_error("Static seed origin does not match mapper");
  const size_t n = header[3];
  const size_t sizes[]{
      size_t(config_.hash_capacity) * 8,    n * 3 * 4, n * 4, n * 4, n * config_.block_voxels * 2,
      n * config_.color_grid_voxels * 4 * 2};
  std::vector<std::vector<char>> seed;
  for (auto size : sizes) {
    seed.emplace_back(size);
    file.read(seed.back().data(), size);
    if (!file) throw std::runtime_error("Truncated static seed");
  }
  if (file.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Unexpected static seed payload");
  destroyEsdfGraph();
  buffers_.static_block_data.allocate(size_t(config_.max_blocks) * config_.block_voxels * 2);
  static_seed_ = std::move(seed);
  static_seed_blocks_ = header[3];
  restoreStaticSeed(toCudaStream(stream_handle));
}
void MapperRuntimeImpl::restoreStaticSeed(cudaStream_t stream) {
  if (static_seed_.empty()) return;
  fillHalfInfinity(reinterpret_cast<uint16_t *>(buffers_.static_block_data.raw()),
                   config_.max_blocks * config_.block_voxels, stream);
  DeviceBuffer *targets[]{&buffers_.hash_table,         &buffers_.block_coords,
                          &buffers_.block_to_hash_slot, &buffers_.static_block_sums,
                          &buffers_.static_block_data,  &buffers_.block_grid_rgb};
  for (size_t i = 0; i < static_seed_.size(); ++i)
    CMR_CUDA_CHECK(cudaMemcpyAsync(targets[i]->raw(), static_seed_[i].data(),
                                   static_seed_[i].size(), cudaMemcpyHostToDevice, stream));
  CMR_CUDA_CHECK(cudaMemcpyAsync(buffers_.num_allocated.raw(), &static_seed_blocks_, sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(stream));
  cached_num_allocated_ = static_seed_blocks_;
}

void MapperRuntimeImpl::reset(uint64_t stream_handle) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  cudaStream_t stream = toCudaStream(stream_handle);
  buffers_.hash_table.memsetAsync(0xff, stream);
  buffers_.block_data.memsetAsync(0, stream);
  buffers_.block_grid_rgb.memsetAsync(0, stream);
  buffers_.block_features.memsetAsync(0, stream);
  buffers_.block_feature_weight.memsetAsync(0, stream);
  buffers_.block_instance_state.memsetAsync(0, stream);
  buffers_.static_block_data.memsetAsync(0, stream);
  buffers_.block_coords.memsetAsync(0, stream);
  buffers_.block_to_hash_slot.memsetAsync(0xff, stream);
  buffers_.free_list.memsetAsync(0, stream);
  buffers_.free_count.memsetAsync(0, stream);
  buffers_.num_allocated.memsetAsync(0, stream);
  buffers_.allocation_failures.memsetAsync(0, stream);
  buffers_.block_sums.memsetAsync(0, stream);
  buffers_.static_block_sums.memsetAsync(0, stream);
  buffers_.frustum_flags.memsetAsync(0, stream);
  buffers_.new_blocks.memsetAsync(0, stream);
  buffers_.new_block_count.memsetAsync(0, stream);
  buffers_.recycle_count.memsetAsync(0, stream);
  buffers_.visible_epoch.memsetAsync(0, stream);
  buffers_.visible_count.memsetAsync(0, stream);
  buffers_.support_counts.memsetAsync(0, stream);
  buffers_.support_overflow_count.memsetAsync(0, stream);
  buffers_.occupied_count.memsetAsync(0, stream);
  buffers_.semantic_target.memsetAsync(0, stream);
  buffers_.semantic_node_match.memsetAsync(0, stream);
  buffers_.semantic_block_has_keep.memsetAsync(0, stream);
  buffers_.instance_removal_block_mask.memsetAsync(0, stream);
  buffers_.esdf_site_index.memsetAsync(0xff, stream);
  buffers_.esdf_pba_buffer.memsetAsync(0xff, stream);
  buffers_.esdf_dist_field.memsetAsync(0, stream);
  buffers_.active_block_count.memsetAsync(0, stream);
  buffers_.surface_count.memsetAsync(0, stream);
  buffers_.triangle_count.memsetAsync(0, stream);
  buffers_.pose_valid_depth_count.memsetAsync(0, stream);
  buffers_.pose_best_position.memsetAsync(0, stream);
  buffers_.pose_best_quaternion.memsetAsync(0, stream);
  buffers_.pose_candidate_position.memsetAsync(0, stream);
  buffers_.pose_candidate_quaternion.memsetAsync(0, stream);
  buffers_.pose_best_jtj.memsetAsync(0, stream);
  buffers_.pose_best_jtr.memsetAsync(0, stream);
  buffers_.pose_best_sum_sq.memsetAsync(0, stream);
  buffers_.pose_best_error.memsetAsync(0, stream);
  buffers_.pose_best_valid_count.memsetAsync(0, stream);
  buffers_.pose_candidate_jtj.memsetAsync(0, stream);
  buffers_.pose_candidate_jtr.memsetAsync(0, stream);
  buffers_.pose_candidate_sum_sq.memsetAsync(0, stream);
  buffers_.pose_candidate_valid_count.memsetAsync(0, stream);
  buffers_.pose_lambda.memsetAsync(0, stream);
  buffers_.pose_pred_reduction.memsetAsync(0, stream);
  buffers_.pose_result.memsetAsync(0, stream);
  CMR_CUDA_CHECK(cudaMemcpyAsync(buffers_.esdf_origin.raw(), config_.origin,
                                 sizeof(float) * 3, cudaMemcpyHostToDevice, stream));
  CMR_CUDA_CHECK(cudaMemcpyAsync(buffers_.esdf_voxel_size.raw(), &config_.esdf_voxel_size,
                                 sizeof(float), cudaMemcpyHostToDevice, stream));
  frame_epoch_ = 0;
  frame_count_ = 0;
  cached_num_allocated_ = 0;
  restoreStaticSeed(stream);
}

void MapperRuntimeImpl::clearInstanceSlot(int instance_slot,
                                          uint64_t stream_handle) {
  if (config_.instance_grid_size <= 0) {
    throw std::runtime_error(
        "clearInstanceSlot requires an artifact with instance_grid_size > 0");
  }
  if (instance_slot <= 0 || instance_slot > 255) {
    throw std::runtime_error("instance_slot must be in [1, 255]");
  }
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  const int num_blocks = numAllocated(stream_handle);
  if (num_blocks <= 0) return;
  auto cu_stream = toCuStream(stream_handle);
  auto state = array2<wp::int16>(buffers_.block_instance_state.ptr(),
                                 config_.max_blocks,
                                 config_.instance_grid_voxels);
  auto num_allocated = array1<wp::int32>(buffers_.num_allocated.ptr(), 1);
  wp::int32 slot = instance_slot;
  wp::int32 max_blocks = config_.max_blocks;
  auto dim = launchBounds2(num_blocks, config_.instance_grid_voxels);
  std::vector<void*> args{&dim, &state, &num_allocated, &slot, &max_blocks};
  kernels_.clear_instance_slot.launch(
      gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("clear_instance_slot_kernel").shared_mem, cu_stream, args);
}

void MapperRuntimeImpl::removeInstanceGeometry(
    int instance_slot, uint64_t semantic_class_mask,
    uint64_t stream_handle) {
  if (config_.instance_grid_size <= 0) {
    throw std::runtime_error(
        "removeInstanceGeometry requires an artifact with instance storage");
  }
  if (instance_slot <= 0 || instance_slot > 255) {
    throw std::runtime_error("instance_slot must be in [1, 255]");
  }
  if (config_.color_grid_size != config_.block_size) {
    throw std::runtime_error(
        "instance geometry removal requires dense block RGB");
  }
  if (semantic_class_mask != 0 &&
      (config_.feature_dim <= 0 || config_.feature_dim > 64 ||
       config_.feature_block_grid_size != config_.block_size)) {
    throw std::runtime_error(
        "semantic instance removal requires dense features with at most 64 classes");
  }
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  const int num_blocks = numAllocated(stream_handle);
  if (num_blocks <= 0) return;
  auto cu_stream = toCuStream(stream_handle);
  auto block_data = array3<wp::float16>(buffers_.block_data.ptr(),
                                        config_.max_blocks,
                                        config_.block_voxels, 2);
  auto block_sums =
      array1<wp::float32>(buffers_.block_sums.ptr(), config_.max_blocks);
  auto block_grid_rgb = array3<wp::float16>(
      buffers_.block_grid_rgb.ptr(), config_.max_blocks,
      config_.color_grid_voxels, 4);
  auto block_features = array3<wp::float16>(
      buffers_.block_features.ptr(), featureStorageBlocks(config_),
      featureStorageNodes(config_), featureStorageChannels(config_));
  auto block_feature_weight = array2<wp::float16>(
      buffers_.block_feature_weight.ptr(), featureStorageBlocks(config_),
      featureStorageNodes(config_));
  auto state = array2<wp::int16>(buffers_.block_instance_state.ptr(),
                                 config_.max_blocks,
                                 config_.instance_grid_voxels);
  auto block_remove_mask = array1<wp::uint8>(
      buffers_.instance_removal_block_mask.ptr(), config_.max_blocks);
  auto num_allocated = array1<wp::int32>(buffers_.num_allocated.ptr(), 1);
  wp::int32 slot = instance_slot;
  std::uint32_t semantic_mask_low =
      static_cast<std::uint32_t>(semantic_class_mask);
  std::uint32_t semantic_mask_high =
      static_cast<std::uint32_t>(semantic_class_mask >> 32U);
  wp::int32 max_blocks = config_.max_blocks;
  if (semantic_class_mask != 0) {
    auto hash_table =
        array1<wp::int64>(buffers_.hash_table.ptr(), config_.hash_capacity);
    auto block_coords =
        array1<wp::int32>(buffers_.block_coords.ptr(), config_.max_blocks * 3);
    auto block_to_hash_slot = array1<wp::int32>(
        buffers_.block_to_hash_slot.ptr(), config_.max_blocks);
    wp::int32 hash_capacity = config_.hash_capacity;
    auto mark_dim = launchBounds1(num_blocks);
    std::vector<void*> mark_args{&mark_dim,
                                 &hash_table,
                                 &block_coords,
                                 &block_to_hash_slot,
                                 &state,
                                 &num_allocated,
                                 &slot,
                                 &hash_capacity,
                                 &max_blocks,
                                 &block_remove_mask};
    kernels_.mark_instance_removal_blocks.launch(
        gridFor(mark_dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("mark_instance_removal_blocks_kernel").shared_mem,
        cu_stream, mark_args);
  }
  auto remove_dim = launchBounds2(num_blocks, config_.block_voxels);
  std::vector<void*> remove_args{
      &remove_dim,           &block_data,         &block_sums,
      &block_grid_rgb,       &block_features,     &block_feature_weight,
      &state,                &block_remove_mask,  &num_allocated,
      &slot,                 &semantic_mask_low,  &semantic_mask_high,
      &max_blocks};
  kernels_.remove_instance_geometry.launch(
      gridFor(remove_dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("remove_instance_geometry_kernel").shared_mem,
      cu_stream, remove_args);

  auto clear_dim = launchBounds2(num_blocks, config_.instance_grid_voxels);
  std::vector<void*> clear_args{
      &clear_dim, &state, &num_allocated, &slot, &max_blocks};
  kernels_.clear_instance_slot.launch(
      gridFor(clear_dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("clear_instance_slot_kernel").shared_mem, cu_stream,
      clear_args);
}

void MapperRuntimeImpl::maybeDecay(const CuroboMapperFrame& frame,
                                   cudaStream_t cuda_stream, CUstream cu_stream) {
  if (frame_count_ <= 0) {
    return;
  }
  if (config_.decay_factor >= 1.0f && config_.frustum_decay_factor >= 1.0f) {
    return;
  }

  const int num_blocks = numAllocated(frame.stream);
  if (num_blocks <= 0) {
    buffers_.recycle_count.memsetAsync(0, cuda_stream);
    return;
  }
  if (num_blocks > config_.max_blocks) {
    throw std::runtime_error("allocated block count exceeds exported capacity");
  }

  auto block_coords = array1<wp::int32>(buffers_.block_coords.ptr(),
                                        config_.max_blocks * 3);
  auto block_to_hash_slot =
      array1<wp::int32>(buffers_.block_to_hash_slot.ptr(), config_.max_blocks);
  auto num_allocated = array1<wp::int32>(buffers_.num_allocated.ptr(), 1);
  auto intrinsics = array3<wp::float32>(frame.intrinsics, config_.num_cameras, 3, 3);
  auto cam_positions = array2<wp::float32>(frame.cam_positions, config_.num_cameras, 3);
  auto cam_quaternions =
      array2<wp::float32>(frame.cam_quaternions, config_.num_cameras, 4);
  auto frustum_flags =
      array1<wp::int32>(buffers_.frustum_flags.ptr(), config_.max_blocks);
  wp::float32 depth_min = config_.depth_minimum_distance;
  wp::float32 depth_max = config_.depth_maximum_distance;
  wp::int32 max_blocks = config_.max_blocks;
  wp::int32 use_frustum = config_.frustum_decay_factor < 1.0f ? 1 : 0;

  if (use_frustum != 0) {
    buffers_.frustum_flags.memsetAsync(0, cuda_stream);
    auto dim_mark = launchBounds2(num_blocks, config_.num_cameras);
    std::vector<void*> args{&dim_mark,
                            &block_coords,
                            &block_to_hash_slot,
                            &num_allocated,
                            &intrinsics,
                            &cam_positions,
                            &cam_quaternions,
                            &depth_min,
                            &depth_max,
                            &frustum_flags,
                            &max_blocks};
    kernels_.mark_blocks_in_frustum.launch(
        gridFor(dim_mark.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("mark_blocks_in_frustum_kernel").shared_mem, cu_stream, args);
  }

  buffers_.block_sums.memsetAsync(0, cuda_stream);
  auto block_data =
      array3<wp::float16>(buffers_.block_data.ptr(), config_.max_blocks,
                          config_.block_voxels, 2);
  auto block_grid_rgb = array3<wp::float16>(buffers_.block_grid_rgb.ptr(),
                                            config_.max_blocks,
                                            config_.color_grid_voxels, 4);
  auto block_features = array3<wp::float16>(buffers_.block_features.ptr(),
                                            featureStorageBlocks(config_),
                                            featureStorageNodes(config_),
                                            featureStorageChannels(config_));
  auto block_feature_weight = array2<wp::float16>(
      buffers_.block_feature_weight.ptr(), featureStorageBlocks(config_),
      featureStorageNodes(config_));
  auto block_sums = array1<wp::float32>(buffers_.block_sums.ptr(), config_.max_blocks);
  wp::float32 time_decay = config_.decay_factor;
  wp::float32 frustum_decay = config_.frustum_decay_factor;
  wp::int32 block_voxels_i = config_.block_voxels;
  wp::int32 color_grid_cells = config_.color_grid_voxels * 4;
  wp::int32 feature_dim = config_.feature_dim;
  wp::int32 feature_cells = config_.feature_dim > 0
                                ? config_.feature_grid_voxels * config_.feature_dim
                                : 0;
  const int decay_cells = std::max(config_.block_voxels,
                                   std::max(config_.color_grid_voxels * 4,
                                            static_cast<int>(feature_cells)));
  auto dim_apply = launchBounds2(num_blocks, decay_cells);
  {
    std::vector<void*> args{&dim_apply,
                            &block_data,
                            &block_grid_rgb,
                            &block_features,
                            &block_feature_weight,
                            &frustum_flags,
                            &block_sums,
                            &num_allocated,
                            &time_decay,
                            &frustum_decay,
                            &use_frustum,
                            &block_voxels_i,
                            &color_grid_cells,
                            &feature_dim,
                            &feature_cells,
                            &max_blocks};
    kernels_.apply_decay_and_sum.launch(
        gridFor(dim_apply.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("apply_decay_and_sum_kernel").shared_mem, cu_stream, args);
  }

  buffers_.recycle_count.memsetAsync(0, cuda_stream);
  auto static_block_sums =
      array1<wp::int32>(buffers_.static_block_sums.ptr(), config_.max_blocks);
  auto hash_table = array1<wp::int64>(buffers_.hash_table.ptr(), config_.hash_capacity);
  auto free_list = array1<wp::int32>(buffers_.free_list.ptr(), config_.max_blocks);
  auto free_count = array1<wp::int32>(buffers_.free_count.ptr(), 1);
  auto recycle_count = array1<wp::int32>(buffers_.recycle_count.ptr(), 1);
  auto dim_recycle = launchBounds1(num_blocks);
  {
    std::vector<void*> args{&dim_recycle,
                            &block_sums,
                            &static_block_sums,
                            &block_to_hash_slot,
                            &hash_table,
                            &free_list,
                            &free_count,
                            &num_allocated,
                            &max_blocks,
                            &recycle_count};
    kernels_.recycle_empty_blocks.launch(
        gridFor(dim_recycle.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("recycle_empty_blocks_kernel").shared_mem, cu_stream, args);
  }
}

void MapperRuntimeImpl::integrate(const CuroboMapperFrame& frame,
                                  uint64_t feature_grid) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  cudaStream_t cuda_stream = toCudaStream(frame.stream);
  CUstream cu_stream = toCuStream(frame.stream);

  maybeDecay(frame, cuda_stream, cu_stream);

  buffers_.new_block_count.memsetAsync(0, cuda_stream);
  buffers_.visible_count.memsetAsync(0, cuda_stream);
  buffers_.support_counts.memsetAsync(0, cuda_stream);
  buffers_.support_overflow_count.memsetAsync(0, cuda_stream);

  const int n_candidates = config_.num_block_key_candidates;
  auto dim_keys = launchBounds1(n_candidates);
  auto intrinsics = array3<wp::float32>(frame.intrinsics, config_.num_cameras, 3, 3);
  auto cam_positions = array2<wp::float32>(frame.cam_positions, config_.num_cameras, 3);
  auto cam_quaternions =
      array2<wp::float32>(frame.cam_quaternions, config_.num_cameras, 4);
  auto depth_images =
      array3<wp::float32>(frame.depth_images, config_.num_cameras, config_.image_height,
                          config_.image_width);
  wp::float32 depth_min = config_.depth_minimum_distance;
  wp::float32 depth_max = config_.depth_maximum_distance;
  auto block_keys = array1<wp::int64>(buffers_.block_keys.ptr(), n_candidates);
  {
    std::vector<void*> args{&dim_keys, &intrinsics, &cam_positions, &cam_quaternions,
                            &depth_images, &depth_min, &depth_max, &block_keys};
    kernels_.compute_block_keys.launch(
        gridFor(dim_keys.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("compute_block_keys_only_kernel").shared_mem, cu_stream, args);
  }

  frame_epoch_ += 1;
  if (frame_epoch_ <= 0) {
    frame_epoch_ = 1;
    buffers_.visible_epoch.memsetAsync(0, cuda_stream);
  }

  auto hash_table = array1<wp::int64>(buffers_.hash_table.ptr(), config_.hash_capacity);
  auto block_coords = array1<wp::int32>(buffers_.block_coords.ptr(), config_.max_blocks * 3);
  auto block_to_hash_slot =
      array1<wp::int32>(buffers_.block_to_hash_slot.ptr(), config_.max_blocks);
  auto num_allocated = array1<wp::int32>(buffers_.num_allocated.ptr(), 1);
  auto allocation_failures = array1<wp::int32>(buffers_.allocation_failures.ptr(), 1);
  auto free_list = array1<wp::int32>(buffers_.free_list.ptr(), config_.max_blocks);
  auto free_count = array1<wp::int32>(buffers_.free_count.ptr(), 1);
  auto new_blocks = array1<wp::int32>(buffers_.new_blocks.ptr(), config_.max_blocks);
  auto new_block_count = array1<wp::int32>(buffers_.new_block_count.ptr(), 1);
  auto visible_epoch = array1<wp::int32>(buffers_.visible_epoch.ptr(), config_.max_blocks);
  auto visible_count = array1<wp::int32>(buffers_.visible_count.ptr(), 1);
  auto pool_indices = array1<wp::int32>(buffers_.pool_indices.ptr(),
                                        config_.max_visible_blocks);
  auto pool_to_visible_slot =
      array1<wp::int32>(buffers_.pool_to_visible_slot.ptr(), config_.max_blocks);
  wp::int32 n_keys = n_candidates;
  wp::int32 hash_capacity = config_.hash_capacity;
  wp::int32 max_blocks = config_.max_blocks;
  wp::int32 visible_capacity = config_.max_visible_blocks;
  wp::int32 frame_epoch = frame_epoch_;
  {
    std::vector<void*> args{&dim_keys,
                            &block_keys,
                            &n_keys,
                            &hash_table,
                            &hash_capacity,
                            &block_coords,
                            &block_to_hash_slot,
                            &num_allocated,
                            &allocation_failures,
                            &max_blocks,
                            &free_list,
                            &free_count,
                            &new_blocks,
                            &new_block_count,
                            &visible_epoch,
                            &visible_count,
                            &frame_epoch,
                            &pool_indices,
                            &pool_to_visible_slot,
                            &visible_capacity};
    kernels_.allocate_visible_blocks.launch(
        gridFor(dim_keys.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("allocate_visible_blocks_from_keys_kernel").shared_mem, cu_stream,
        args);
  }

  int num_visible = 0;
  int num_allocated_host = 0;
  CMR_CUDA_CHECK(cudaMemcpyAsync(&num_visible, buffers_.visible_count.raw(),
                                 sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  CMR_CUDA_CHECK(cudaMemcpyAsync(&num_allocated_host, buffers_.num_allocated.raw(),
                                 sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  cached_num_allocated_ = num_allocated_host;
  if (num_visible <= 0) {
    frame_count_ += 1;
    return;
  }
  if (num_visible > config_.max_visible_blocks) {
    throw std::runtime_error("visible block count exceeds exported capacity");
  }

  auto support_counts = array2<wp::int32>(buffers_.support_counts.ptr(),
                                          config_.max_visible_blocks,
                                          config_.num_cameras);
  auto support_pixels =
      array3<wp::int32>(buffers_.support_pixels.ptr(), config_.max_visible_blocks,
                        config_.num_cameras,
                        config_.max_support_pixels_per_block_camera);
  auto support_overflow_count =
      array1<wp::int32>(buffers_.support_overflow_count.ptr(), 1);
  {
    std::vector<void*> args{&dim_keys,
                            &block_keys,
                            &n_keys,
                            &hash_table,
                            &hash_capacity,
                            &max_blocks,
                            &visible_epoch,
                            &frame_epoch,
                            &pool_to_visible_slot,
                            &visible_capacity,
                            &support_counts,
                            &support_pixels,
                            &support_overflow_count};
    kernels_.build_support_pixels.launch(
        gridFor(dim_keys.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("build_support_pixels_from_keys_kernel").shared_mem, cu_stream,
        args);
  }

  auto block_data =
      array3<wp::float16>(buffers_.block_data.ptr(), config_.max_blocks,
                          config_.block_voxels, 2);
  auto block_grid_rgb = array3<wp::float16>(buffers_.block_grid_rgb.ptr(),
                                            config_.max_blocks,
                                            config_.color_grid_voxels, 4);
  auto block_features = array3<wp::float16>(buffers_.block_features.ptr(),
                                            featureStorageBlocks(config_),
                                            featureStorageNodes(config_),
                                            featureStorageChannels(config_));
  auto block_feature_weight = array2<wp::float16>(
      buffers_.block_feature_weight.ptr(), featureStorageBlocks(config_),
      featureStorageNodes(config_));
  auto block_instance_state = array2<wp::int16>(
      buffers_.block_instance_state.ptr(), instanceStorageBlocks(config_),
      instanceStorageNodes(config_));
  wp::int32 max_blocks_i = config_.max_blocks;
  auto dim_clear =
      launchBounds2(std::min(num_visible, config_.max_blocks), config_.block_voxels);
  {
    std::vector<void*> args{&dim_clear, &block_data, &new_blocks,
                            &new_block_count, &max_blocks_i};
    kernels_.clear_new_blocks.launch(
        gridFor(dim_clear.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("clear_new_blocks_kernel").shared_mem, cu_stream, args);
  }
  {
    auto dim_clear_grid =
        launchBounds2(std::min(num_visible, config_.max_blocks),
                      config_.color_grid_voxels * 4);
    std::vector<void*> args{&dim_clear_grid, &block_grid_rgb, &new_blocks,
                            &new_block_count, &max_blocks_i};
    kernels_.clear_new_block_grid_rgb.launch(
        gridFor(dim_clear_grid.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("clear_new_block_grid_rgb_kernel").shared_mem, cu_stream, args);
  }
  if (config_.feature_dim > 0) {
    auto dim_clear_features = launchBounds3(std::min(num_visible, config_.max_blocks),
                                            config_.feature_grid_voxels,
                                            config_.feature_dim);
    std::vector<void*> args{&dim_clear_features,
                            &block_features,
                            &block_feature_weight,
                            &new_blocks,
                            &new_block_count,
                            &max_blocks_i};
    kernels_.clear_new_block_features.launch(
        gridFor(dim_clear_features.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("clear_new_block_features_kernel").shared_mem, cu_stream,
        args);
  }
  if (config_.instance_grid_size > 0) {
    auto dim_clear_instances = launchBounds2(
        std::min(num_visible, config_.max_blocks),
        config_.instance_grid_voxels);
    std::vector<void*> args{&dim_clear_instances, &block_instance_state,
                            &new_blocks, &new_block_count, &max_blocks_i};
    kernels_.clear_new_block_instances.launch(
        gridFor(dim_clear_instances.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("clear_new_block_instances_kernel").shared_mem,
        cu_stream, args);
  }

  wp::int32 n_visible = num_visible;
  auto dim_integrate = launchBounds2(num_visible, config_.block_voxels);
  {
    std::vector<void*> args{&dim_integrate,
                            &pool_indices,
                            &n_visible,
                            &intrinsics,
                            &cam_positions,
                            &cam_quaternions,
                            &depth_images,
                            &depth_min,
                            &depth_max,
                            &block_coords,
                            &block_data};
    kernels_.integrate_voxels.launch(
        gridFor(dim_integrate.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("integrate_voxels_kernel").shared_mem, cu_stream, args);
  }

  auto rgb_images =
      array2<wp::vec_t<3, wp::uint8>>(frame.rgb_images,
                                      config_.num_cameras * config_.image_height,
                                      config_.image_width);
  {
    auto dim_grid_rgb = launchBounds2(num_visible, config_.color_grid_voxels);
    std::vector<void*> args{&dim_grid_rgb,
                            &pool_indices,
                            &n_visible,
                            &intrinsics,
                            &cam_positions,
                            &cam_quaternions,
                            &depth_images,
                            &rgb_images,
                            &support_counts,
                            &support_pixels,
                            &depth_min,
                            &depth_max,
                            &block_coords,
                            &block_grid_rgb};
    kernels_.integrate_block_grid_rgb.launch(
        gridFor(dim_grid_rgb.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("integrate_block_grid_rgb_kernel").shared_mem, cu_stream, args);
  }

  if (frame.instance_image != 0) {
    if (config_.instance_grid_size <= 0) {
      throw std::runtime_error(
          "instance_image was provided but instance_grid_size == 0");
    }
    auto instance_image = array3<wp::uint8>(
        frame.instance_image, config_.num_cameras, config_.image_height,
        config_.image_width);
    auto dim_instances =
        launchBounds2(num_visible, config_.instance_grid_voxels);
    std::vector<void*> args{&dim_instances,
                            &pool_indices,
                            &n_visible,
                            &intrinsics,
                            &cam_positions,
                            &cam_quaternions,
                            &depth_images,
                            &instance_image,
                            &depth_min,
                            &depth_max,
                            &block_coords,
                            &block_instance_state};
    kernels_.integrate_instances.launch(
        gridFor(dim_instances.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("integrate_instance_grid_kernel").shared_mem, cu_stream,
        args);
  }

  if (feature_grid != 0) {
    if (config_.feature_dim <= 0) {
      throw std::runtime_error(
          "feature_grid was provided but the exported mapper has feature_dim == 0");
    }
    auto feature_grid_array = array4<wp::float16>(
        feature_grid, config_.num_cameras, config_.feature_grid_height,
        config_.feature_grid_width, config_.feature_dim);
    std::vector<void*> args;
    const std::string kernel_name =
        config_.use_tiled_feature_kernel
            ? "integrate_features_from_support_tiled_kernel"
            : "integrate_features_from_support_grouped_kernel";
    if (config_.use_tiled_feature_kernel) {
      const int tile_channels =
          std::max(1, std::min(config_.feature_dim,
                               config_.max_feature_tile_channels));
      const int channel_tiles = cdiv(config_.feature_dim, tile_channels);
      auto dim_features = launchBounds4(num_visible, config_.feature_grid_voxels,
                                        channel_tiles, 64);
      args = {&dim_features,
              &pool_indices,
              &n_visible,
              &intrinsics,
              &cam_positions,
              &cam_quaternions,
              &depth_images,
              &support_counts,
              &support_pixels,
              &feature_grid_array,
              &depth_min,
              &depth_max,
              &block_coords,
              &block_features,
              &block_feature_weight};
      kernels_.integrate_features.launch(
          gridFor(dim_features.size, 64), Dim3{64, 1, 1},
          kernelManifest(kernel_name).shared_mem, cu_stream, args);
    } else {
      const int channel_groups =
          cdiv(config_.feature_dim, config_.feature_channels_per_thread);
      auto dim_features = launchBounds3(num_visible, config_.feature_grid_voxels,
                                        channel_groups);
      args = {&dim_features,
              &pool_indices,
              &n_visible,
              &intrinsics,
              &cam_positions,
              &cam_quaternions,
              &depth_images,
              &support_counts,
              &support_pixels,
              &feature_grid_array,
              &depth_min,
              &depth_max,
              &block_coords,
              &block_features,
              &block_feature_weight};
      kernels_.integrate_features.launch(
          gridFor(dim_features.size), Dim3{kWarpBlockDim, 1, 1},
          kernelManifest(kernel_name).shared_mem, cu_stream, args);
    }
  }

  wp::float32 w_max = config_.accumulator_w_max;
  if (config_.feature_dim > 0) {
    auto dim_feature_rescale = launchBounds3(num_visible, config_.feature_grid_voxels,
                                             config_.feature_dim);
    std::vector<void*> args{&dim_feature_rescale,
                            &pool_indices,
                            &n_visible,
                            &w_max,
                            &block_features,
                            &block_feature_weight};
    kernels_.rescale_accumulators.launch(
        gridFor(dim_feature_rescale.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("rescale_block_accumulators_kernel").shared_mem, cu_stream,
        args);
  }
  {
    auto dim_grid_rescale = launchBounds2(num_visible, config_.color_grid_voxels * 3);
    std::vector<void*> args{&dim_grid_rescale, &pool_indices, &n_visible, &w_max,
                            &block_grid_rgb};
    kernels_.rescale_block_grid_rgb.launch(
        gridFor(dim_grid_rescale.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("rescale_block_grid_rgb_kernel").shared_mem, cu_stream, args);
  }
  frame_count_ += 1;
}

void MapperRuntimeImpl::extractOccupied(uint64_t out_centers, uint64_t out_pool_indices,
                                        uint64_t out_count, int capacity,
                                        bool surface_only, float sdf_threshold,
                                        uint64_t stream_handle, int* required_count,
                                        int* written_count,
                                        const CuroboMapperSemanticFilter* semantic_filter) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  cudaStream_t cuda_stream = toCudaStream(stream_handle);
  CUstream cu_stream = toCuStream(stream_handle);
  if (capacity < 0) {
    throw std::runtime_error("occupied output capacity must be non-negative");
  }

  int num_alloc = numAllocated(stream_handle);
  if (required_count != nullptr) {
    *required_count = 0;
  }
  if (written_count != nullptr) {
    *written_count = 0;
  }
  buffers_.occupied_count.memsetAsync(0, cuda_stream);
  if (out_count != 0) {
    CMR_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(out_count), 0, sizeof(int),
                                   cuda_stream));
  }
  if (num_alloc <= 0) {
    return;
  }

  auto tsdf = tsdfWarp();
  wp::float32 min_weight = config_.minimum_tsdf_weight;
  wp::int32 surface_only_i = surface_only ? 1 : 0;
  wp::float32 threshold = sdf_threshold > 0.0f ? sdf_threshold : config_.voxel_size;
  auto count_array = array1<wp::int32>(buffers_.occupied_count.ptr(), 1);
  auto dim = launchBounds2(num_alloc, config_.block_voxels);
  PreparedSemanticFilterWarp prepared{};
  if (semantic_filter != nullptr) {
    prepared = prepareSemanticFilter(*semantic_filter, cuda_stream, cu_stream);
  }
  const bool query_only =
      capacity == 0 || out_centers == 0 || out_pool_indices == 0;
  if (query_only) {
    if (semantic_filter != nullptr) {
      std::vector<void*> args{&dim, &tsdf, &prepared, &min_weight,
                              &surface_only_i, &threshold, &count_array};
      kernels_.count_occupied_semantic.launch(
          gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
          kernelManifest("count_occupied_voxels_semantic_kernel").shared_mem,
          cu_stream, args);
    } else {
      std::vector<void*> args{&dim, &tsdf, &min_weight, &surface_only_i,
                              &threshold, &count_array};
      kernels_.count_occupied.launch(
          gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
          kernelManifest("count_occupied_voxels_kernel").shared_mem, cu_stream,
          args);
    }
    int count = 0;
    CMR_CUDA_CHECK(cudaMemcpyAsync(&count, buffers_.occupied_count.raw(), sizeof(int),
                                   cudaMemcpyDeviceToHost, cuda_stream));
    CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    if (required_count != nullptr) *required_count = count;
    return;
  }

  void* out_count_ptr =
      out_count != 0 ? reinterpret_cast<void*>(out_count) : buffers_.occupied_count.raw();
  CMR_CUDA_CHECK(cudaMemsetAsync(out_count_ptr, 0, sizeof(int), cuda_stream));
  auto out_centers_array = array2<wp::float32>(out_centers, capacity, 3);
  auto out_pool_idx_array = array1<wp::int32>(out_pool_indices, capacity);
  auto out_count_array =
      array1<wp::int32>(reinterpret_cast<uint64_t>(out_count_ptr), 1);
  wp::int32 max_voxels = capacity;
  if (semantic_filter != nullptr) {
    std::vector<void*> args{&dim,
                            &tsdf,
                            &prepared,
                            &min_weight,
                            &surface_only_i,
                            &threshold,
                            &out_centers_array,
                            &out_pool_idx_array,
                            &out_count_array,
                            &max_voxels};
    kernels_.extract_occupied_semantic.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("extract_occupied_voxels_semantic_kernel").shared_mem,
        cu_stream, args);
  } else {
    std::vector<void*> args{&dim,
                            &tsdf,
                            &min_weight,
                            &surface_only_i,
                            &threshold,
                            &out_centers_array,
                            &out_pool_idx_array,
                            &out_count_array,
                            &max_voxels};
    kernels_.extract_occupied.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("extract_occupied_voxels_kernel").shared_mem, cu_stream, args);
  }
  int written = 0;
  CMR_CUDA_CHECK(cudaMemcpyAsync(&written, out_count_ptr, sizeof(int),
                                 cudaMemcpyDeviceToHost, cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  if (required_count != nullptr) {
    *required_count = written;
  }
  if (written_count != nullptr) {
    *written_count = std::min(written, capacity);
  }
}

void MapperRuntimeImpl::launchPba(cudaStream_t cuda_stream, CUstream cu_stream) {
  const int nx = config_.esdf_nx;
  const int ny = config_.esdf_ny;
  const int nz = config_.esdf_nz;
  const int sx0 = nz;
  const int sy0 = ny;
  const int sz0 = nx;
  int* buf0 = reinterpret_cast<int*>(buffers_.esdf_site_index.raw());
  int* buf1 = reinterpret_cast<int*>(buffers_.esdf_pba_buffer.raw());

  auto launch_pba = [&](CubinKernel& kernel, Dim3 grid, Dim3 block, int* input,
                        int* output, int sx, int sy, int sz) {
    std::vector<void*> args{&input, &output, &sx, &sy, &sz};
    kernel.launch(grid, block, 0, cu_stream, args);
  };

  launch_pba(kernels_.pba_flood_z, Dim3{static_cast<unsigned int>(cdiv(sx0, 32)),
                                        static_cast<unsigned int>(cdiv(sy0, 4)), 1},
             Dim3{32, 4, 1}, buf0, buf1, sx0, sy0, sz0);
  launch_pba(kernels_.pba_maurer_axis, Dim3{static_cast<unsigned int>(cdiv(sx0, 32)),
                                            static_cast<unsigned int>(cdiv(sz0, 4)), 1},
             Dim3{32, 4, 1}, buf1, buf0, sx0, sy0, sz0);
  launch_pba(kernels_.pba_color_axis, Dim3{static_cast<unsigned int>(cdiv(sx0, 32)),
                                           static_cast<unsigned int>(sz0), 1},
             Dim3{32, 2, 1}, buf0, buf1, sx0, sy0, sz0);
  launch_pba(kernels_.pba_maurer_axis, Dim3{static_cast<unsigned int>(cdiv(sy0, 32)),
                                            static_cast<unsigned int>(cdiv(sz0, 4)), 1},
             Dim3{32, 4, 1}, buf1, buf0, sy0, sx0, sz0);
  launch_pba(kernels_.pba_color_axis, Dim3{static_cast<unsigned int>(cdiv(sy0, 32)),
                                           static_cast<unsigned int>(sz0), 1},
             Dim3{32, 2, 1}, buf0, buf1, sy0, sx0, sz0);

  const size_t esdf_bytes =
      bytesFor(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz),
               sizeof(int32_t));
  CMR_CUDA_CHECK(cudaMemcpyAsync(buffers_.esdf_site_index.raw(),
                                 buffers_.esdf_pba_buffer.raw(), esdf_bytes,
                                 cudaMemcpyDeviceToDevice, cuda_stream));
}

int MapperRuntimeImpl::semanticPreparationBlockCapacity() const {
  const int required = std::max(1, cached_num_allocated_);
  int capacity = 1;
  while (capacity < required && capacity <= config_.max_blocks / 2) {
    capacity *= 2;
  }
  return capacity < required ? config_.max_blocks
                             : std::min(capacity, config_.max_blocks);
}

void MapperRuntimeImpl::destroyEsdfGraph() noexcept {
  if (esdf_graph_executable_ != nullptr) {
    cudaGraphExecDestroy(esdf_graph_executable_);
  }
  if (esdf_graph_ != nullptr) {
    cudaGraphDestroy(esdf_graph_);
  }
  esdf_graph_ = nullptr;
  esdf_graph_executable_ = nullptr;
  esdf_graph_stream_ = 0;
  esdf_graph_semantic_block_capacity_ = 0;
  esdf_graph_has_filter_ = false;
  esdf_graph_filter_ = {};
}

void MapperRuntimeImpl::enqueueEsdf(uint64_t out_dist_field, cudaStream_t cuda_stream,
                                    CUstream cu_stream,
                                    const CuroboMapperSemanticFilter* semantic_filter,
                                    bool dynamic_query) {
  const int n_esdf = config_.esdf_nx * config_.esdf_ny * config_.esdf_nz;

  buffers_.esdf_site_index.memsetAsync(0xff, cuda_stream);
  buffers_.esdf_dist_field.memsetAsync(0, cuda_stream);

  auto tsdf = tsdfWarp();
  auto site_index = array1<wp::int32>(buffers_.esdf_site_index.ptr(), n_esdf);
  auto esdf_origin = array1<wp::float32>(buffers_.esdf_origin.ptr(), 3);
  auto esdf_voxel_size = array1<wp::float32>(buffers_.esdf_voxel_size.ptr(), 1);
  wp::float32 min_weight = config_.minimum_tsdf_weight;
  auto dim_seed = launchBounds1(n_esdf);
  PreparedSemanticFilterWarp prepared{};
  if (semantic_filter != nullptr) {
    prepared = prepareSemanticFilter(
        *semantic_filter, cuda_stream, cu_stream,
        dynamic_query ? reinterpret_cast<const uint64_t*>(buffers_.esdf_query.ptr()) : nullptr);
    std::vector<void*> args{&dim_seed, &tsdf, &prepared, &site_index,
                            &esdf_origin, &esdf_voxel_size, &min_weight};
    kernels_.seed_esdf_gather_semantic.launch(
        gridFor(dim_seed.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("seed_esdf_sites_gather_semantic_kernel").shared_mem,
        cu_stream, args);
  } else {
    std::vector<void*> args{&dim_seed, &tsdf, &site_index, &esdf_origin,
                            &esdf_voxel_size, &min_weight};
    kernels_.seed_esdf_gather.launch(
        gridFor(dim_seed.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("seed_esdf_sites_gather_kernel").shared_mem, cu_stream, args);
  }

  launchPba(cuda_stream, cu_stream);

  auto dist_field = array1<wp::float16>(buffers_.esdf_dist_field.ptr(), n_esdf);
  wp::float32 skip_steps = config_.adjacent_skip_steps;
  if (semantic_filter != nullptr) {
    std::vector<void*> args{&dim_seed,
                            &site_index,
                            &esdf_voxel_size,
                            &dist_field,
                            &tsdf,
                            &prepared,
                            &min_weight,
                            &esdf_origin,
                            &skip_steps};
    kernels_.compute_esdf_distance_semantic.launch(
        gridFor(dim_seed.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("compute_esdf_from_min_tsdf_semantic_kernel").shared_mem,
        cu_stream, args);
  } else {
    std::vector<void*> args{&dim_seed,
                            &site_index,
                            &esdf_voxel_size,
                            &dist_field,
                            &tsdf,
                            &min_weight,
                            &esdf_origin,
                            &skip_steps};
    kernels_.compute_esdf_distance.launch(
        gridFor(dim_seed.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("compute_esdf_from_min_tsdf_kernel").shared_mem, cu_stream,
        args);
  }

  if (dynamic_query) {
    copyEsdfOutput(reinterpret_cast<const uint16_t*>(buffers_.esdf_dist_field.ptr()), n_esdf,
                   reinterpret_cast<const uint64_t*>(buffers_.esdf_query.ptr()), cuda_stream);
  } else if (out_dist_field != 0) {
    const size_t bytes =
        bytesFor(static_cast<size_t>(n_esdf), sizeof(uint16_t));
    CMR_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(out_dist_field),
                                   buffers_.esdf_dist_field.raw(), bytes,
                                   cudaMemcpyDeviceToDevice, cuda_stream));
  }
}

void MapperRuntimeImpl::computeEsdf(
    uint64_t out_dist_field, uint64_t stream_handle,
    const CuroboMapperSemanticFilter* semantic_filter) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  cudaStream_t cuda_stream = toCudaStream(stream_handle);
  CUstream cu_stream = toCuStream(stream_handle);

  // Vector targets require host validation and normalization. The fixed
  // categorical filter and unfiltered paths are entirely device-side and can
  // be captured because the exported ESDF shape and all buffer addresses are
  // immutable for the runtime lifetime.
  const bool graph_eligible =
      stream_handle != 0 &&
      (semantic_filter == nullptr || semantic_filter->target_class >= 0);
  if (!graph_eligible) {
    enqueueEsdf(out_dist_field, cuda_stream, cu_stream, semantic_filter);
    return;
  }

  const bool has_filter = semantic_filter != nullptr;
  // Validate changing values even on replay; layout-dependent validation also
  // runs during capture in prepareSemanticFilter.
  if (has_filter &&
      (semantic_filter->exclude_instance_slot < 0 || semantic_filter->exclude_instance_slot > 255 ||
       (semantic_filter->exclude_instance_slot != 0 &&
        (config_.instance_grid_size != config_.block_size ||
         config_.feature_block_grid_size != config_.block_size)))) {
    throw std::invalid_argument("invalid ESDF instance exclusion");
  }
  setEsdfQuery(reinterpret_cast<uint64_t*>(buffers_.esdf_query.ptr()), out_dist_field,
               has_filter ? semantic_filter->exclude_instance_slot : 0, cuda_stream);
  const int semantic_capacity =
      has_filter ? semanticPreparationBlockCapacity() : 0;
  const auto same_filter = [&]() {
    if (!has_filter) return true;
    return esdf_graph_filter_.target == semantic_filter->target &&
           esdf_graph_filter_.target_class == semantic_filter->target_class &&
           esdf_graph_filter_.min_similarity == semantic_filter->min_similarity &&
           esdf_graph_filter_.use_min_similarity == semantic_filter->use_min_similarity &&
           esdf_graph_filter_.require_argmax == semantic_filter->require_argmax &&
           esdf_graph_filter_.exclude == semantic_filter->exclude;
  };
  if (esdf_graph_executable_ != nullptr &&
      esdf_graph_stream_ == stream_handle &&
      esdf_graph_semantic_block_capacity_ == semantic_capacity &&
      esdf_graph_has_filter_ == has_filter && same_filter()) {
    CMR_CUDA_CHECK(cudaGraphLaunch(esdf_graph_executable_, cuda_stream));
    return;
  }

  destroyEsdfGraph();
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  CMR_CUDA_CHECK(cudaStreamBeginCapture(cuda_stream,
                                        cudaStreamCaptureModeThreadLocal));
  try {
    enqueueEsdf(out_dist_field, cuda_stream, cu_stream, semantic_filter, true);
    CMR_CUDA_CHECK(cudaStreamEndCapture(cuda_stream, &esdf_graph_));
    CMR_CUDA_CHECK(
        cudaGraphInstantiate(&esdf_graph_executable_, esdf_graph_, 0));
    ++esdf_graph_capture_count_;
  } catch (...) {
    cudaGraph_t abandoned_graph = nullptr;
    cudaStreamEndCapture(cuda_stream, &abandoned_graph);
    if (abandoned_graph != nullptr) cudaGraphDestroy(abandoned_graph);
    destroyEsdfGraph();
    throw;
  }
  esdf_graph_stream_ = stream_handle;
  esdf_graph_semantic_block_capacity_ = semantic_capacity;
  esdf_graph_has_filter_ = has_filter;
  if (has_filter) esdf_graph_filter_ = *semantic_filter;
  CMR_CUDA_CHECK(cudaGraphLaunch(esdf_graph_executable_, cuda_stream));
}

void MapperRuntimeImpl::ensureRenderScratch(int n_pixels) {
  const size_t needed_points = bytesFor(static_cast<size_t>(n_pixels) * 3, sizeof(float));
  const size_t needed_depths = bytesFor(n_pixels, sizeof(float));
  const size_t needed_mask = bytesFor(n_pixels, sizeof(uint8_t));
  if (buffers_.render_hit_points.bytes() < needed_points) {
    buffers_.render_hit_points.allocate(needed_points);
    buffers_.render_hit_normals.allocate(needed_points);
  }
  if (buffers_.render_hit_depths.bytes() < needed_depths) {
    buffers_.render_hit_depths.allocate(needed_depths);
  }
  if (buffers_.render_hit_mask.bytes() < needed_mask) {
    buffers_.render_hit_mask.allocate(needed_mask);
  }
}

void MapperRuntimeImpl::launchRender(bool with_color, uint64_t intrinsics,
                                     uint64_t cam_positions,
                                     uint64_t cam_quaternions,
                                     uint64_t out_hit_points,
                                     uint64_t out_hit_normals,
                                     uint64_t out_hit_colors,
                                     uint64_t out_hit_depths,
                                     uint64_t out_hit_mask, int camera_count,
                                     int image_height, int image_width,
                                     uint64_t stream_handle,
                                     const CuroboMapperSemanticFilter* semantic_filter) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  cudaStream_t cuda_stream = toCudaStream(stream_handle);
  CUstream cu_stream = toCuStream(stream_handle);
  if (camera_count <= 0) {
    throw std::runtime_error("render camera count must be positive");
  }
  if (image_height <= 0 || image_width <= 0) {
    throw std::runtime_error("render image shape must be positive");
  }
  if (intrinsics == 0 || cam_positions == 0 || cam_quaternions == 0 ||
      out_hit_points == 0 || out_hit_normals == 0 || out_hit_depths == 0 ||
      out_hit_mask == 0 || (with_color && out_hit_colors == 0)) {
    throw std::runtime_error("render received a null device pointer");
  }
  const int pixels_per_camera = image_height * image_width;
  const int n_pixels = camera_count * pixels_per_camera;
  auto dim = launchBounds1(n_pixels);
  auto intrinsics_array = array3<wp::float32>(intrinsics, camera_count, 3, 3);
  auto position_array = array2<wp::float32>(cam_positions, camera_count, 3);
  auto quaternion_array = array2<wp::float32>(cam_quaternions, camera_count, 4);
  auto tsdf = tsdfWarp();
  wp::float32 depth_min = config_.depth_minimum_distance;
  wp::float32 depth_max = config_.depth_maximum_distance;
  wp::float32 min_weight = config_.minimum_tsdf_weight;
  auto hit_points = array2<wp::float32>(out_hit_points, n_pixels, 3);
  auto hit_normals = array2<wp::float32>(out_hit_normals, n_pixels, 3);
  auto hit_depths = array1<wp::float32>(out_hit_depths, n_pixels);
  auto hit_mask = array1<wp::uint8>(out_hit_mask, n_pixels);
  wp::int32 H = image_height;
  wp::int32 W = image_width;
  wp::int32 pixels_per_camera_w = pixels_per_camera;
  PreparedSemanticFilterWarp prepared{};
  if (semantic_filter != nullptr) {
    prepared = prepareSemanticFilter(*semantic_filter, cuda_stream, cu_stream);
  }
  if (with_color) {
    auto hit_colors = array2<wp::uint8>(out_hit_colors, n_pixels, 3);
    if (semantic_filter != nullptr) {
      std::vector<void*> args{&dim,
                              &intrinsics_array,
                              &position_array,
                              &quaternion_array,
                              &tsdf,
                              &prepared,
                              &depth_min,
                              &depth_max,
                              &min_weight,
                              &hit_points,
                              &hit_normals,
                              &hit_colors,
                              &hit_depths,
                              &hit_mask,
                              &H,
                              &W,
                              &pixels_per_camera_w};
      kernels_.raycast_color_semantic.launch(
          gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
          kernelManifest("raycast_block_sparse_color_semantic_kernel").shared_mem,
          cu_stream, args);
      return;
    }
    std::vector<void*> args{&dim,
                            &intrinsics_array,
                            &position_array,
                            &quaternion_array,
                            &tsdf,
                            &depth_min,
                            &depth_max,
                            &min_weight,
                            &hit_points,
                            &hit_normals,
                            &hit_colors,
                            &hit_depths,
                            &hit_mask,
                            &H,
                            &W,
                            &pixels_per_camera_w};
    kernels_.raycast_color.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("raycast_block_sparse_color_kernel").shared_mem, cu_stream,
        args);
  } else {
    if (semantic_filter != nullptr) {
      std::vector<void*> args{&dim,
                              &intrinsics_array,
                              &position_array,
                              &quaternion_array,
                              &tsdf,
                              &prepared,
                              &depth_min,
                              &depth_max,
                              &min_weight,
                              &hit_points,
                              &hit_normals,
                              &hit_depths,
                              &hit_mask,
                              &H,
                              &W,
                              &pixels_per_camera_w};
      kernels_.raycast_depth_semantic.launch(
          gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
          kernelManifest("raycast_block_sparse_semantic_kernel").shared_mem,
          cu_stream, args);
      return;
    }
    std::vector<void*> args{&dim,
                            &intrinsics_array,
                            &position_array,
                            &quaternion_array,
                            &tsdf,
                            &depth_min,
                            &depth_max,
                            &min_weight,
                            &hit_points,
                            &hit_normals,
                            &hit_depths,
                            &hit_mask,
                            &H,
                            &W,
                            &pixels_per_camera_w};
    kernels_.raycast_depth.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("raycast_block_sparse_kernel").shared_mem, cu_stream, args);
  }
}

void MapperRuntimeImpl::renderDepth(uint64_t intrinsics, uint64_t cam_positions,
                                    uint64_t cam_quaternions,
                                    uint64_t out_hit_points,
                                    uint64_t out_hit_normals,
                                    uint64_t out_hit_depths,
                                    uint64_t out_hit_mask, int camera_count,
                                    int image_height, int image_width,
                                    uint64_t stream_handle,
                                    const CuroboMapperSemanticFilter* semantic_filter) {
  launchRender(false, intrinsics, cam_positions, cam_quaternions, out_hit_points,
               out_hit_normals, 0, out_hit_depths, out_hit_mask, camera_count,
               image_height, image_width, stream_handle, semantic_filter);
}

void MapperRuntimeImpl::renderColor(uint64_t intrinsics, uint64_t cam_positions,
                                    uint64_t cam_quaternions,
                                    uint64_t out_hit_points,
                                    uint64_t out_hit_normals,
                                    uint64_t out_hit_colors,
                                    uint64_t out_hit_depths,
                                    uint64_t out_hit_mask, int camera_count,
                                    int image_height, int image_width,
                                    uint64_t stream_handle,
                                    const CuroboMapperSemanticFilter* semantic_filter) {
  launchRender(true, intrinsics, cam_positions, cam_quaternions, out_hit_points,
               out_hit_normals, out_hit_colors, out_hit_depths, out_hit_mask,
               camera_count, image_height, image_width, stream_handle,
               semantic_filter);
}

void MapperRuntimeImpl::renderFeatures(
    uint64_t intrinsics, uint64_t cam_positions, uint64_t cam_quaternions,
    uint64_t out_hit_normals, uint64_t out_hit_depths, uint64_t out_hit_mask,
    uint64_t out_feature_valid, uint64_t out_feature_block_indices,
    uint64_t out_feature_node_indices, int camera_count, int image_height,
    int image_width, int patch_top, int patch_left, int patch_height,
    int patch_width, uint64_t stream_handle,
    const CuroboMapperSemanticFilter* semantic_filter) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  if (config_.feature_dim <= 0) {
    throw std::runtime_error(
        "render_features requires an artifact exported with feature_dim > 0");
  }
  if (camera_count <= 0 || image_height <= 0 || image_width <= 0 ||
      patch_height <= 0 || patch_width <= 0) {
    throw std::runtime_error("feature render shapes must be positive");
  }
  if (patch_top < 0 || patch_left < 0 || patch_top + patch_height > image_height ||
      patch_left + patch_width > image_width) {
    throw std::runtime_error("feature render patch is outside the image bounds");
  }
  if (intrinsics == 0 || cam_positions == 0 || cam_quaternions == 0 ||
      out_hit_normals == 0 || out_hit_depths == 0 || out_hit_mask == 0 ||
      out_feature_valid == 0 || out_feature_block_indices == 0 ||
      out_feature_node_indices == 0) {
    throw std::runtime_error("render_features received a null device pointer");
  }

  cudaStream_t cuda_stream = toCudaStream(stream_handle);
  CUstream cu_stream = toCuStream(stream_handle);
  const int pixels_per_camera = patch_height * patch_width;
  const int n_pixels = camera_count * pixels_per_camera;
  auto dim = launchBounds1(n_pixels);
  auto intrinsics_array = array3<wp::float32>(intrinsics, camera_count, 3, 3);
  auto position_array = array2<wp::float32>(cam_positions, camera_count, 3);
  auto quaternion_array = array2<wp::float32>(cam_quaternions, camera_count, 4);
  auto tsdf = tsdfWarp();
  wp::float32 depth_min = config_.depth_minimum_distance;
  wp::float32 depth_max = config_.depth_maximum_distance;
  wp::float32 min_weight = config_.minimum_tsdf_weight;
  auto hit_normals = array2<wp::float32>(out_hit_normals, n_pixels, 3);
  auto hit_depths = array1<wp::float32>(out_hit_depths, n_pixels);
  auto hit_mask = array1<wp::uint8>(out_hit_mask, n_pixels);
  auto feature_valid = array1<wp::uint8>(out_feature_valid, n_pixels);
  auto feature_block_idx =
      array1<wp::int32>(out_feature_block_indices, n_pixels);
  auto feature_node_idx = array1<wp::int16>(out_feature_node_indices, n_pixels);
  wp::int32 patch_top_w = patch_top;
  wp::int32 patch_left_w = patch_left;
  wp::int32 patch_height_w = patch_height;
  wp::int32 patch_width_w = patch_width;
  wp::int32 pixels_per_camera_w = pixels_per_camera;
  if (semantic_filter != nullptr) {
    PreparedSemanticFilterWarp prepared =
        prepareSemanticFilter(*semantic_filter, cuda_stream, cu_stream);
    std::vector<void*> args{
        &dim,               &intrinsics_array,  &position_array,
        &quaternion_array,  &tsdf,              &prepared,
        &depth_min,         &depth_max,          &min_weight,
        &hit_normals,       &hit_depths,         &hit_mask,
        &feature_valid,     &feature_block_idx,  &feature_node_idx,
        &patch_top_w,       &patch_left_w,       &patch_height_w,
        &patch_width_w,     &pixels_per_camera_w};
    kernels_.raycast_features_semantic.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("raycast_block_sparse_feature_semantic_kernel").shared_mem,
        cu_stream, args);
    return;
  }
  std::vector<void*> args{
      &dim,               &intrinsics_array,  &position_array,
      &quaternion_array,  &tsdf,              &depth_min,
      &depth_max,         &min_weight,         &hit_normals,
      &hit_depths,        &hit_mask,           &feature_valid,
      &feature_block_idx, &feature_node_idx,   &patch_top_w,
      &patch_left_w,      &patch_height_w,     &patch_width_w,
      &pixels_per_camera_w};
  kernels_.raycast_features.launch(
      gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("raycast_block_sparse_feature_kernel").shared_mem,
      cu_stream, args);
}

void MapperRuntimeImpl::gatherFeatures(
    uint64_t feature_valid, uint64_t feature_block_indices,
    uint64_t feature_node_indices, uint64_t out_features, int camera_count,
    int image_height, int image_width, int channel_start, int channel_count,
    bool output_float16, uint64_t stream_handle) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  if (config_.feature_dim <= 0) {
    throw std::runtime_error(
        "gather_features requires an artifact exported with feature_dim > 0");
  }
  if (camera_count <= 0 || image_height <= 0 || image_width <= 0 ||
      channel_count <= 0) {
    throw std::runtime_error("feature gather shapes must be positive");
  }
  if (channel_start < 0 || channel_start + channel_count > config_.feature_dim) {
    throw std::runtime_error("feature gather channel range is outside feature_dim");
  }
  if (feature_valid == 0 || feature_block_indices == 0 ||
      feature_node_indices == 0 || out_features == 0) {
    throw std::runtime_error("gather_features received a null device pointer");
  }

  CUstream cu_stream = toCuStream(stream_handle);
  const int pixels_per_camera = image_height * image_width;
  const int n_pixels = camera_count * pixels_per_camera;
  auto dim = launchBounds2(n_pixels, channel_count);
  auto tsdf = tsdfWarp();
  auto block_indices = array1<wp::int32>(feature_block_indices, n_pixels);
  auto node_indices = array1<wp::int16>(feature_node_indices, n_pixels);
  auto valid = array1<wp::uint8>(feature_valid, n_pixels);
  wp::int32 source_width = image_width;
  wp::int32 source_pixels_per_camera = pixels_per_camera;
  wp::int32 patch_top = 0;
  wp::int32 patch_left = 0;
  wp::int32 patch_height = image_height;
  wp::int32 patch_width = image_width;
  wp::int32 channel_start_w = channel_start;
  if (output_float16) {
    auto output = array2<wp::float16>(out_features, n_pixels, channel_count);
    std::vector<void*> args{
        &dim,        &tsdf,       &block_indices, &node_indices,
        &valid,      &source_width, &source_pixels_per_camera,
        &patch_top,  &patch_left, &patch_height,  &patch_width,
        &channel_start_w, &output};
    kernels_.gather_features_fp16.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("gather_rendered_features_fp16_kernel").shared_mem,
        cu_stream, args);
  } else {
    auto output = array2<wp::float32>(out_features, n_pixels, channel_count);
    std::vector<void*> args{
        &dim,        &tsdf,       &block_indices, &node_indices,
        &valid,      &source_width, &source_pixels_per_camera,
        &patch_top,  &patch_left, &patch_height,  &patch_width,
        &channel_start_w, &output};
    kernels_.gather_features_fp32.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("gather_rendered_features_fp32_kernel").shared_mem,
        cu_stream, args);
  }
}

void MapperRuntimeImpl::ensureMeshScratch() {
  if (mesh_scratch_initialized_) {
    return;
  }
  const size_t max_blocks = static_cast<size_t>(config_.max_blocks);
  const size_t max_surface_cubes =
      max_blocks * static_cast<size_t>(config_.block_voxels);
  buffers_.active_block_count.allocate(sizeof(int32_t));
  buffers_.active_block_idx.allocate(bytesFor(max_blocks, sizeof(int32_t)));
  buffers_.surface_count.allocate(sizeof(int32_t));
  buffers_.surface_block_idx.allocate(bytesFor(max_surface_cubes, sizeof(int32_t)));
  buffers_.surface_cube_idx.allocate(bytesFor(max_surface_cubes, sizeof(int32_t)));
  buffers_.triangle_count.allocate(sizeof(int32_t));
  buffers_.mc_num_triangles.allocate(bytesFor(256, sizeof(int32_t)));
  buffers_.mc_triangle_table.allocate(bytesFor(256 * 16, sizeof(int32_t)));
  CMR_CUDA_CHECK(cudaMemcpy(buffers_.mc_num_triangles.raw(),
                            manifest_.mc_num_triangles.data(),
                            bytesFor(256, sizeof(int32_t)),
                            cudaMemcpyHostToDevice));
  CMR_CUDA_CHECK(cudaMemcpy(buffers_.mc_triangle_table.raw(),
                            manifest_.mc_triangle_table.data(),
                            bytesFor(256 * 16, sizeof(int32_t)),
                            cudaMemcpyHostToDevice));

  mesh_scratch_initialized_ = true;
}

void MapperRuntimeImpl::extractMesh(
    uint64_t out_vertices, uint64_t out_normals, uint64_t out_colors,
    int vertex_capacity, float level, bool surface_only, int refine_iterations,
    uint64_t stream_handle, int* required_vertices, int* required_triangles,
    int* written_vertices, int* written_triangles,
    const CuroboMapperSemanticFilter* semantic_filter) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  cudaStream_t cuda_stream = toCudaStream(stream_handle);
  CUstream cu_stream = toCuStream(stream_handle);
  if (vertex_capacity < 0) {
    throw std::runtime_error("mesh output capacity must be non-negative");
  }
  if (refine_iterations < 0) {
    throw std::runtime_error("mesh refine_iterations must be non-negative");
  }
  if (required_vertices != nullptr) *required_vertices = 0;
  if (required_triangles != nullptr) *required_triangles = 0;
  if (written_vertices != nullptr) *written_vertices = 0;
  if (written_triangles != nullptr) *written_triangles = 0;

  const int num_alloc = numAllocated(stream_handle);
  if (num_alloc <= 0) {
    return;
  }
  if (num_alloc > config_.max_blocks) {
    throw std::runtime_error("allocated block count exceeds exported capacity");
  }

  const size_t max_surface_cubes =
      static_cast<size_t>(config_.max_blocks) * static_cast<size_t>(config_.block_voxels);
  if (max_surface_cubes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("mesh surface scratch exceeds int32 launch limits");
  }
  ensureMeshScratch();

  auto tsdf = tsdfWarp();
  auto active_count = array1<wp::int32>(buffers_.active_block_count.ptr(), 1);
  auto active_block_idx =
      array1<wp::int32>(buffers_.active_block_idx.ptr(), config_.max_blocks);
  auto surface_count = array1<wp::int32>(buffers_.surface_count.ptr(), 1);
  wp::float32 level_w = level;
  wp::float32 surface_band =
      surface_only ? config_.truncation_distance : 0.0f;
  wp::float32 min_weight = config_.minimum_tsdf_weight;
  PreparedSemanticFilterWarp prepared{};
  if (semantic_filter != nullptr) {
    prepared = prepareSemanticFilter(*semantic_filter, cuda_stream, cu_stream);
  }

  buffers_.active_block_count.memsetAsync(0, cuda_stream);
  {
    auto dim_active = launchBounds1(num_alloc);
    if (semantic_filter != nullptr) {
      std::vector<void*> args{&dim_active, &tsdf, &prepared, &active_count,
                              &active_block_idx};
      kernels_.append_active_blocks_semantic.launch(
          gridFor(dim_active.size), Dim3{kWarpBlockDim, 1, 1},
          kernelManifest("append_active_blocks_semantic_kernel").shared_mem,
          cu_stream, args);
    } else {
      std::vector<void*> args{&dim_active, &tsdf, &active_count,
                              &active_block_idx};
      kernels_.append_active_blocks.launch(
          gridFor(dim_active.size), Dim3{kWarpBlockDim, 1, 1},
          kernelManifest("append_active_blocks_kernel").shared_mem, cu_stream,
          args);
    }
  }

  int n_active = 0;
  CMR_CUDA_CHECK(cudaMemcpyAsync(&n_active, buffers_.active_block_count.raw(),
                                 sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  if (n_active <= 0) {
    return;
  }
  if (n_active > config_.max_blocks) {
    throw std::runtime_error("active block count exceeds exported capacity");
  }

  wp::int32 n_active_w = n_active;
  auto dim_surface = launchBounds2(n_active, config_.block_voxels);
  auto surface_block_idx =
      array1<wp::int32>(buffers_.surface_block_idx.ptr(),
                        static_cast<int>(max_surface_cubes));
  auto surface_cube_idx =
      array1<wp::int32>(buffers_.surface_cube_idx.ptr(),
                        static_cast<int>(max_surface_cubes));
  buffers_.surface_count.memsetAsync(0, cuda_stream);
  if (semantic_filter != nullptr) {
    std::vector<void*> args{&dim_surface,
                            &tsdf,
                            &prepared,
                            &active_block_idx,
                            &n_active_w,
                            &level_w,
                            &surface_band,
                            &min_weight,
                            &surface_count,
                            &surface_block_idx,
                            &surface_cube_idx};
    kernels_.append_surface_cubes_from_blocks_semantic.launch(
        gridFor(dim_surface.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("append_surface_cubes_from_blocks_semantic_kernel")
            .shared_mem,
        cu_stream, args);
  } else {
    std::vector<void*> args{&dim_surface,
                            &tsdf,
                            &active_block_idx,
                            &n_active_w,
                            &level_w,
                            &surface_band,
                            &min_weight,
                            &surface_count,
                            &surface_block_idx,
                            &surface_cube_idx};
    kernels_.append_surface_cubes_from_blocks.launch(
        gridFor(dim_surface.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("append_surface_cubes_from_blocks_kernel").shared_mem,
        cu_stream, args);
  }

  int n_surfaces = 0;
  CMR_CUDA_CHECK(cudaMemcpyAsync(&n_surfaces, buffers_.surface_count.raw(),
                                 sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  if (n_surfaces <= 0) {
    return;
  }
  if (static_cast<size_t>(n_surfaces) > max_surface_cubes) {
    throw std::runtime_error("surface cube count exceeds runtime scratch capacity");
  }

  auto num_tris_table = array1<wp::int32>(buffers_.mc_num_triangles.ptr(), 256);
  auto triangle_count = array1<wp::int32>(buffers_.triangle_count.ptr(), 1);
  wp::int32 n_surfaces_w = n_surfaces;
  auto dim_count_tris = launchBounds1(n_surfaces);
  buffers_.triangle_count.memsetAsync(0, cuda_stream);
  {
    std::vector<void*> args{&dim_count_tris,
                            &tsdf,
                            &level_w,
                            &min_weight,
                            &surface_block_idx,
                            &surface_cube_idx,
                            &n_surfaces_w,
                            &num_tris_table,
                            &triangle_count};
    kernels_.count_total_triangles.launch(
        gridFor(dim_count_tris.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("count_total_triangles_kernel").shared_mem, cu_stream,
        args);
  }

  int total_tris = 0;
  CMR_CUDA_CHECK(cudaMemcpyAsync(&total_tris, buffers_.triangle_count.raw(),
                                 sizeof(int), cudaMemcpyDeviceToHost, cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  if (total_tris < 0 ||
      total_tris > static_cast<int>(std::numeric_limits<int>::max() / 3)) {
    throw std::runtime_error("approximate mesh exceeds int32 output limits");
  }
  const int total_vertices = total_tris * 3;
  if (required_vertices != nullptr) *required_vertices = total_vertices;
  if (required_triangles != nullptr) *required_triangles = total_tris;
  if (total_tris <= 0) {
    return;
  }

  if (vertex_capacity == 0 || out_vertices == 0 || out_normals == 0 ||
      out_colors == 0) {
    return;
  }
  if (vertex_capacity < total_vertices) {
    throw std::runtime_error("mesh output buffers are smaller than required count");
  }

  auto tri_table = array1<wp::int32>(buffers_.mc_triangle_table.ptr(), 256 * 16);
  auto vertices = array1<wp::vec_t<3, wp::float32>>(out_vertices, total_vertices);
  auto normals = array1<wp::vec_t<3, wp::float32>>(out_normals, total_vertices);
  auto colors = array1<wp::vec_t<3, wp::uint8>>(out_colors, total_vertices);
  wp::int32 triangle_capacity_w = total_tris;
  wp::int32 refine_iterations_w = refine_iterations;
  buffers_.triangle_count.memsetAsync(0, cuda_stream);
  {
    std::vector<void*> args{&dim_count_tris,
                            &tsdf,
                            &level_w,
                            &min_weight,
                            &refine_iterations_w,
                            &surface_block_idx,
                            &surface_cube_idx,
                            &n_surfaces_w,
                            &tri_table,
                            &vertices,
                            &normals,
                            &colors,
                            &triangle_count,
                            &triangle_capacity_w};
    kernels_.generate_mesh.launch(
        gridFor(dim_count_tris.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("generate_mesh_kernel").shared_mem, cu_stream, args);
  }
  if (written_vertices != nullptr) *written_vertices = total_vertices;
  if (written_triangles != nullptr) *written_triangles = total_tris;
}

void MapperRuntimeImpl::ensureTextureScoreScratch(int triangle_count) {
  const size_t needed = bytesFor(static_cast<size_t>(triangle_count), sizeof(float));
  if (buffers_.texture_projection_scores.bytes() < needed) {
    buffers_.texture_projection_scores.allocate(needed);
  }
}

void MapperRuntimeImpl::ensureTexturePostprocessScratch(int triangle_count) {
  const size_t count_bytes =
      bytesFor(static_cast<size_t>(std::max(1, triangle_count)), sizeof(int32_t));
  if (buffers_.texture_fallback_flags.bytes() < count_bytes) {
    buffers_.texture_fallback_flags.allocate(count_bytes);
  }
  if (buffers_.texture_fallback_offsets.bytes() < count_bytes) {
    buffers_.texture_fallback_offsets.allocate(count_bytes);
  }
  if (buffers_.texture_fallback_count.bytes() < sizeof(int32_t)) {
    buffers_.texture_fallback_count.allocate(sizeof(int32_t));
  }
  const size_t scan_bytes = exclusiveSumTempStorageBytes(triangle_count);
  const size_t reduce_bytes = reduceSumTempStorageBytes(triangle_count);
  const size_t temp_bytes = std::max(scan_bytes, reduce_bytes);
  if (buffers_.texture_scan_temp.bytes() < temp_bytes) {
    buffers_.texture_scan_temp.allocate(temp_bytes);
  }
}

uint64_t MapperRuntimeImpl::visibilityDepthForTextureFrame(
    const CuroboMapperTextureFrame& frame) {
  if (frame.visibility_depths != 0) {
    return frame.visibility_depths;
  }
  if (frame.batch_count <= 0) {
    throw std::runtime_error("texture batch_count must be positive");
  }
  const int total_cameras = frame.batch_count * config_.texture_num_cameras;
  const int n_pixels = total_cameras *
                       config_.texture_camera_image_height *
                       config_.texture_camera_image_width;
  ensureRenderScratch(n_pixels);
  launchRender(false, frame.intrinsics, frame.cam_positions, frame.cam_quaternions,
               buffers_.render_hit_points.ptr(), buffers_.render_hit_normals.ptr(),
               0, buffers_.render_hit_depths.ptr(), buffers_.render_hit_mask.ptr(),
               total_cameras, config_.texture_camera_image_height,
               config_.texture_camera_image_width, frame.stream, nullptr);
  return buffers_.render_hit_depths.ptr();
}

void MapperRuntimeImpl::projectMeshTexture(
    const CuroboMapperTextureFrame& frame, uint64_t vertices_ptr, int vertex_count,
    uint64_t out_uvs, uint64_t out_colors, uint64_t out_texture_atlas,
    uint64_t texture_atlas_capacity_bytes, float texture_depth_tolerance_m,
    int* texture_atlas_height, int* texture_atlas_width) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  cudaStream_t cuda_stream = toCudaStream(frame.stream);
  CUstream cu_stream = toCuStream(frame.stream);

  if (texture_atlas_height == nullptr || texture_atlas_width == nullptr) {
    throw std::runtime_error("texture atlas dimension output pointers are null");
  }
  *texture_atlas_height = 0;
  *texture_atlas_width = 0;
  if (frame.batch_count <= 0) {
    throw std::runtime_error("texture batch_count must be positive");
  }
  if (vertex_count < 0) {
    throw std::runtime_error("texture projection vertex count must be non-negative");
  }
  if ((vertex_count % 3) != 0) {
    throw std::runtime_error("texture projection requires triangle-soup vertices");
  }
  if (frame.rgb_images == 0 || frame.intrinsics == 0 ||
      frame.cam_positions == 0 || frame.cam_quaternions == 0 ||
      out_texture_atlas == 0) {
    throw std::runtime_error("texture projection received a null device pointer");
  }

  const int cameras_per_batch = config_.texture_num_cameras;
  const int total_cameras = frame.batch_count * cameras_per_batch;
  const int image_height = config_.texture_camera_image_height;
  const int image_width = config_.texture_camera_image_width;
  const int atlas_columns =
      projectiveTextureAtlasColumns(total_cameras, image_width);
  const int base_atlas_rows = cdiv(total_cameras, atlas_columns);
  const int base_atlas_height = base_atlas_rows * image_height;
  const int atlas_width = atlas_columns * image_width;
  const size_t base_atlas_bytes =
      bytesFor(static_cast<size_t>(base_atlas_height) *
                   static_cast<size_t>(atlas_width) * 3,
               sizeof(uint8_t));
  if (texture_atlas_capacity_bytes < base_atlas_bytes) {
    *texture_atlas_height = base_atlas_height;
    *texture_atlas_width = atlas_width;
    throw std::runtime_error("texture atlas output buffer is smaller than base atlas");
  }

  CMR_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(out_texture_atlas), 0,
                                 base_atlas_bytes, cuda_stream));
  packTextureAtlas(reinterpret_cast<const uint8_t*>(frame.rgb_images),
                   frame.batch_count, cameras_per_batch, image_height,
                   image_width, atlas_columns,
                   reinterpret_cast<uint8_t*>(out_texture_atlas), cuda_stream);

  if (vertex_count == 0) {
    *texture_atlas_height = base_atlas_height;
    *texture_atlas_width = atlas_width;
    return;
  }
  if (vertices_ptr == 0 || out_uvs == 0 || out_colors == 0) {
    throw std::runtime_error("texture projection received a null mesh output pointer");
  }

  const int triangle_count = vertex_count / 3;
  ensureTextureScoreScratch(triangle_count);
  ensureTexturePostprocessScratch(triangle_count);
  fillFloat(reinterpret_cast<float*>(buffers_.texture_projection_scores.raw()),
            -1.0e20f, triangle_count, cuda_stream);

  const uint64_t visibility_depths_ptr = visibilityDepthForTextureFrame(frame);
  auto vertices = array1<wp::vec_t<3, wp::float32>>(vertices_ptr, vertex_count);
  auto vertex_uvs = array1<wp::vec_t<2, wp::float32>>(out_uvs, vertex_count);
  auto projection_scores =
      array1<wp::float32>(buffers_.texture_projection_scores.ptr(), triangle_count);
  auto colors = array1<wp::vec_t<3, wp::uint8>>(out_colors, vertex_count);
  auto tsdf = tsdfWarp();
  auto dim = launchBounds1(triangle_count);
  wp::int32 vertex_count_w = vertex_count;
  wp::float32 depth_min = config_.depth_minimum_distance;
  wp::float32 depth_max = config_.depth_maximum_distance;
  wp::float32 tolerance = texture_depth_tolerance_m;
  wp::int32 atlas_camera_count = total_cameras;
  wp::int32 atlas_columns_w = atlas_columns;
  wp::int32 fill_missing_colors_w = 1;
  const size_t depth_batch_bytes =
      bytesFor(static_cast<size_t>(cameras_per_batch) *
                   static_cast<size_t>(image_height) *
                   static_cast<size_t>(image_width),
               sizeof(float));
  const size_t intrinsics_batch_bytes =
      bytesFor(static_cast<size_t>(cameras_per_batch) * 9, sizeof(float));
  const size_t positions_batch_bytes =
      bytesFor(static_cast<size_t>(cameras_per_batch) * 3, sizeof(float));
  const size_t quaternions_batch_bytes =
      bytesFor(static_cast<size_t>(cameras_per_batch) * 4, sizeof(float));

  for (int batch_idx = 0; batch_idx < frame.batch_count; ++batch_idx) {
    auto visibility_depths = array3<wp::float32>(
        offsetPtr(visibility_depths_ptr, static_cast<size_t>(batch_idx) *
                                             depth_batch_bytes),
        cameras_per_batch, image_height, image_width);
    auto intrinsics = array3<wp::float32>(
        offsetPtr(frame.intrinsics, static_cast<size_t>(batch_idx) *
                                      intrinsics_batch_bytes),
        cameras_per_batch, 3, 3);
    auto cam_positions = array2<wp::float32>(
        offsetPtr(frame.cam_positions, static_cast<size_t>(batch_idx) *
                                           positions_batch_bytes),
        cameras_per_batch, 3);
    auto cam_quaternions = array2<wp::float32>(
        offsetPtr(frame.cam_quaternions, static_cast<size_t>(batch_idx) *
                                             quaternions_batch_bytes),
        cameras_per_batch, 4);
    wp::int32 camera_offset = batch_idx * cameras_per_batch;

    std::vector<void*> args{&dim,
                            &vertices,
                            &vertex_count_w,
                            &visibility_depths,
                            &intrinsics,
                            &cam_positions,
                            &cam_quaternions,
                            &depth_min,
                            &depth_max,
                            &tolerance,
                            &camera_offset,
                            &atlas_camera_count,
                            &atlas_columns_w,
                            &vertex_uvs,
                            &projection_scores,
                            &colors,
                            &fill_missing_colors_w,
                            &tsdf};
    kernels_.project_mesh_uvs.launch(
        gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("project_mesh_uvs_kernel").shared_mem, cu_stream, args);
  }

  auto* fallback_flags =
      reinterpret_cast<int32_t*>(buffers_.texture_fallback_flags.raw());
  auto* fallback_offsets =
      reinterpret_cast<int32_t*>(buffers_.texture_fallback_offsets.raw());
  auto* fallback_count =
      reinterpret_cast<int32_t*>(buffers_.texture_fallback_count.raw());
  markTextureFallbackTriangles(reinterpret_cast<const float*>(out_uvs),
                               triangle_count, fallback_flags, cuda_stream);
  reduceSumInt32(buffers_.texture_scan_temp.raw(),
                 buffers_.texture_scan_temp.bytes(), fallback_flags,
                 fallback_count, triangle_count, cuda_stream);
  int fallback_count_host = 0;
  CMR_CUDA_CHECK(cudaMemcpyAsync(&fallback_count_host, fallback_count,
                                 sizeof(int32_t), cudaMemcpyDeviceToHost,
                                 cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

  const int patch_size = 2;
  const int patches_per_row = std::max(1, atlas_width / patch_size);
  const int fallback_rows = cdiv(fallback_count_host, patches_per_row) * patch_size;
  const int final_atlas_height = base_atlas_height + fallback_rows;
  const size_t final_atlas_bytes =
      bytesFor(static_cast<size_t>(final_atlas_height) *
                   static_cast<size_t>(atlas_width) * 3,
               sizeof(uint8_t));
  *texture_atlas_height = final_atlas_height;
  *texture_atlas_width = atlas_width;
  if (texture_atlas_capacity_bytes < final_atlas_bytes) {
    throw std::runtime_error("texture atlas output buffer is smaller than final atlas");
  }
  if (final_atlas_bytes > base_atlas_bytes) {
    CMR_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(offsetPtr(out_texture_atlas, base_atlas_bytes)),
        0, final_atlas_bytes - base_atlas_bytes, cuda_stream));
  }
  exclusiveSumInt32(buffers_.texture_scan_temp.raw(),
                    buffers_.texture_scan_temp.bytes(), fallback_flags,
                    fallback_offsets, triangle_count, cuda_stream);
  finalizeTextureAtlas(reinterpret_cast<float*>(out_uvs), triangle_count,
                       fallback_flags, fallback_offsets,
                       reinterpret_cast<uint8_t*>(out_colors),
                       reinterpret_cast<uint8_t*>(out_texture_atlas),
                       base_atlas_height, atlas_width, final_atlas_height,
                       cuda_stream);
}

void MapperRuntimeImpl::launchPoseAlignment(
    uint64_t depth_image, uint64_t intrinsics, uint64_t position,
    uint64_t quaternion, const CuroboMapperPoseRefinerConfig& config,
    int stride, int out_w, int n_pixels, uint64_t out_jtj, uint64_t out_jtr,
    uint64_t out_sum_sq, uint64_t out_valid_count, CUstream cu_stream) {
  if (manifest_.kernels.find(kPoseAlignmentKernelName) == manifest_.kernels.end()) {
    throw std::runtime_error(
        "artifact does not contain the pose alignment kernel; regenerate it with the "
        "updated exporter");
  }
  const int n_total_samples = n_pixels * config.n_samples_per_ray;
  auto dim = launchBounds1(n_total_samples);
  auto depth = array2<wp::float32>(depth_image, config_.image_height,
                                   config_.image_width);
  auto intrinsics_array = array2<wp::float32>(intrinsics, 3, 3);
  auto position_array = array1<wp::float32>(position, 3);
  auto quaternion_array = array1<wp::float32>(quaternion, 4);
  auto tsdf = tsdfWarp();
  wp::float32 min_weight = config.minimum_tsdf_weight;
  wp::float32 depth_min = config.depth_minimum_distance;
  wp::float32 depth_max = config.depth_maximum_distance;
  wp::float32 distance_threshold = config.distance_threshold;
  wp::int32 stride_i = stride;
  wp::int32 out_w_i = out_w;
  wp::int32 n_pixels_i = n_pixels;
  auto jtj = array1<wp::float32>(out_jtj, 36);
  auto jtr = array1<wp::float32>(out_jtr, 6);
  auto sum_sq = array1<wp::float32>(out_sum_sq, 1);
  auto valid_count = array1<wp::int32>(out_valid_count, 1);

  std::vector<void*> args{&dim,
                          &depth,
                          &intrinsics_array,
                          &position_array,
                          &quaternion_array,
                          &tsdf,
                          &min_weight,
                          &depth_min,
                          &depth_max,
                          &distance_threshold,
                          &stride_i,
                          &out_w_i,
                          &n_pixels_i,
                          &jtj,
                          &jtr,
                          &sum_sq,
                          &valid_count};
  kernels_.pose_refine_alignment.launch(
      gridFor(dim.size, static_cast<unsigned int>(config.tile_block_dim)),
      Dim3{static_cast<unsigned int>(config.tile_block_dim), 1, 1},
      kernelManifest(kPoseAlignmentKernelName).shared_mem, cu_stream, args);
}

void MapperRuntimeImpl::refinePose(
    uint64_t depth_image, uint64_t intrinsics, uint64_t initial_position,
    uint64_t initial_quaternion, uint64_t out_position, uint64_t out_quaternion,
    const CuroboMapperPoseRefinerConfig* requested_config, uint64_t stream_handle,
    CuroboMapperPoseRefinerResult* result) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  if (depth_image == 0 || intrinsics == 0 || initial_position == 0 ||
      initial_quaternion == 0 || out_position == 0 || out_quaternion == 0) {
    throw std::runtime_error("refine_pose received a null device pointer");
  }
  if (manifest_.kernels.find(kPoseAlignmentKernelName) == manifest_.kernels.end() ||
      !has_pose_lm_kernels(manifest_.kernels)) {
    throw std::runtime_error(
        "artifact does not contain all pose refinement kernels; regenerate it with the "
        "updated exporter");
  }

  const CuroboMapperPoseRefinerConfig config =
      normalizedPoseRefinerConfig(requested_config, config_);
  if (config.n_samples_per_ray != config_.pose_refiner_n_samples_per_ray) {
    throw std::runtime_error(
        "pose refiner n_samples_per_ray does not match the generated artifact");
  }
  if (config.tile_block_dim <= 0) {
    throw std::runtime_error("pose refiner tile_block_dim must be positive");
  }

  cudaStream_t cuda_stream = toCudaStream(stream_handle);
  CUstream cu_stream = toCuStream(stream_handle);
  auto* result_float = reinterpret_cast<float*>(buffers_.pose_result.raw());
  auto* result_int = reinterpret_cast<int*>(offsetPtr(
      buffers_.pose_result.ptr(), offsetof(CuroboMapperPoseRefinerResult, iterations)));

  const int image_height = config_.image_height;
  const int image_width = config_.image_width;
  const int image_pixels = image_height * image_width;
  const int stride =
      std::max(1, static_cast<int>(std::sqrt(static_cast<double>(image_pixels) /
                                             static_cast<double>(config.n_points))));
  const int out_h = cdiv(image_height, stride);
  const int out_w = cdiv(image_width, stride);
  const int n_pixels = out_h * out_w;

  buffers_.pose_valid_depth_count.memsetAsync(0, cuda_stream);
  buffers_.pose_candidate_jtj.memsetAsync(0, cuda_stream);
  buffers_.pose_candidate_jtr.memsetAsync(0, cuda_stream);
  buffers_.pose_candidate_sum_sq.memsetAsync(0, cuda_stream);
  buffers_.pose_candidate_valid_count.memsetAsync(0, cuda_stream);

  launch_pose_refiner_count_valid_depth(
      kernels_.pose_refine_count_valid_depth,
      reinterpret_cast<const float*>(depth_image), image_pixels,
      config.depth_minimum_distance, config.depth_maximum_distance,
      reinterpret_cast<int*>(buffers_.pose_valid_depth_count.raw()), cu_stream);

  launchPoseAlignment(depth_image, intrinsics, initial_position, initial_quaternion,
                      config, stride, out_w, n_pixels,
                      buffers_.pose_candidate_jtj.ptr(),
                      buffers_.pose_candidate_jtr.ptr(),
                      buffers_.pose_candidate_sum_sq.ptr(),
                      buffers_.pose_candidate_valid_count.ptr(), cu_stream);

  launch_pose_refiner_initialize_state(
      kernels_.pose_refine_initialize_state,
      reinterpret_cast<const float*>(initial_position),
      reinterpret_cast<const float*>(initial_quaternion),
      reinterpret_cast<const float*>(buffers_.pose_candidate_jtj.raw()),
      reinterpret_cast<const float*>(buffers_.pose_candidate_jtr.raw()),
      reinterpret_cast<const float*>(buffers_.pose_candidate_sum_sq.raw()),
      reinterpret_cast<const int*>(buffers_.pose_candidate_valid_count.raw()),
      reinterpret_cast<const int*>(buffers_.pose_valid_depth_count.raw()),
      config.minimum_valid_depth_pixels, config.minimum_valid_depth_pixels,
      config.lambda_initial,
      reinterpret_cast<float*>(buffers_.pose_best_position.raw()),
      reinterpret_cast<float*>(buffers_.pose_best_quaternion.raw()),
      reinterpret_cast<float*>(buffers_.pose_best_jtj.raw()),
      reinterpret_cast<float*>(buffers_.pose_best_jtr.raw()),
      reinterpret_cast<float*>(buffers_.pose_best_sum_sq.raw()),
      reinterpret_cast<float*>(buffers_.pose_best_error.raw()),
      reinterpret_cast<int*>(buffers_.pose_best_valid_count.raw()),
      reinterpret_cast<float*>(buffers_.pose_lambda.raw()),
      result_float, result_int, cu_stream);

  CuroboMapperPoseRefinerResult local_result{};
  CMR_CUDA_CHECK(cudaMemcpyAsync(&local_result, buffers_.pose_result.raw(),
                                 sizeof(local_result), cudaMemcpyDeviceToHost,
                                 cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

  if (local_result.status == CUROBO_MAPPER_POSE_REFINER_SUCCESS) {
    for (int iter = 0; iter < config.max_iterations; ++iter) {
      launch_pose_refiner_solve_candidate(
          kernels_.pose_refine_solve_candidate,
          reinterpret_cast<const float*>(buffers_.pose_best_position.raw()),
          reinterpret_cast<const float*>(buffers_.pose_best_quaternion.raw()),
          reinterpret_cast<const float*>(buffers_.pose_best_jtj.raw()),
          reinterpret_cast<const float*>(buffers_.pose_best_jtr.raw()),
          reinterpret_cast<const float*>(buffers_.pose_lambda.raw()),
          reinterpret_cast<float*>(buffers_.pose_candidate_position.raw()),
          reinterpret_cast<float*>(buffers_.pose_candidate_quaternion.raw()),
          reinterpret_cast<float*>(buffers_.pose_pred_reduction.raw()),
          result_int, cu_stream);

      buffers_.pose_candidate_jtj.memsetAsync(0, cuda_stream);
      buffers_.pose_candidate_jtr.memsetAsync(0, cuda_stream);
      buffers_.pose_candidate_sum_sq.memsetAsync(0, cuda_stream);
      buffers_.pose_candidate_valid_count.memsetAsync(0, cuda_stream);
      launchPoseAlignment(depth_image, intrinsics,
                          buffers_.pose_candidate_position.ptr(),
                          buffers_.pose_candidate_quaternion.ptr(), config, stride,
                          out_w, n_pixels, buffers_.pose_candidate_jtj.ptr(),
                          buffers_.pose_candidate_jtr.ptr(),
                          buffers_.pose_candidate_sum_sq.ptr(),
                          buffers_.pose_candidate_valid_count.ptr(), cu_stream);

      launch_pose_refiner_accept_candidate(
          kernels_.pose_refine_accept_candidate,
          reinterpret_cast<const float*>(buffers_.pose_candidate_position.raw()),
          reinterpret_cast<const float*>(buffers_.pose_candidate_quaternion.raw()),
          reinterpret_cast<const float*>(buffers_.pose_candidate_jtj.raw()),
          reinterpret_cast<const float*>(buffers_.pose_candidate_jtr.raw()),
          reinterpret_cast<const float*>(buffers_.pose_candidate_sum_sq.raw()),
          reinterpret_cast<const int*>(buffers_.pose_candidate_valid_count.raw()),
          reinterpret_cast<float*>(buffers_.pose_best_position.raw()),
          reinterpret_cast<float*>(buffers_.pose_best_quaternion.raw()),
          reinterpret_cast<float*>(buffers_.pose_best_jtj.raw()),
          reinterpret_cast<float*>(buffers_.pose_best_jtr.raw()),
          reinterpret_cast<float*>(buffers_.pose_best_sum_sq.raw()),
          reinterpret_cast<float*>(buffers_.pose_best_error.raw()),
          reinterpret_cast<int*>(buffers_.pose_best_valid_count.raw()),
          reinterpret_cast<float*>(buffers_.pose_lambda.raw()),
          reinterpret_cast<const float*>(buffers_.pose_pred_reduction.raw()),
          config.minimum_valid_depth_pixels, config.lambda_factor,
          config.lambda_min, config.lambda_max, result_float, result_int,
          cu_stream);
    }
  }

  launch_pose_refiner_copy_best_pose(
      kernels_.pose_refine_copy_best_pose,
      reinterpret_cast<const float*>(buffers_.pose_best_position.raw()),
      reinterpret_cast<const float*>(buffers_.pose_best_quaternion.raw()),
      reinterpret_cast<float*>(out_position), reinterpret_cast<float*>(out_quaternion),
      cu_stream);

  if (result != nullptr) {
    CMR_CUDA_CHECK(cudaMemcpyAsync(result, buffers_.pose_result.raw(),
                                   sizeof(*result), cudaMemcpyDeviceToHost,
                                   cuda_stream));
    CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  }
}

void MapperRuntimeImpl::ensure_point_scratch(int sample_budget, cudaStream_t stream) {
  if (points_.outputs.count(sample_budget) != 0) return;
  cudaStreamCaptureStatus capture;
  CMR_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error("Warm up the surface-point budget before graph capture");
  constexpr size_t GROUP = 64;
  const size_t voxels = static_cast<size_t>(config_.max_blocks) * config_.block_voxels;
  if (voxels > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / 3)
    throw std::runtime_error("Surface sampling exceeds int32 crossing-index limits");
  if (points_.flags.bytes() == 0) {
    points_.groups = static_cast<int>((voxels + GROUP - 1) / GROUP);
    const int entries = points_.groups + 1;
    points_.flags.allocate(voxels);
    points_.counts.allocate(bytesFor(entries, sizeof(int32_t)));
    points_.offsets.allocate(bytesFor(entries, sizeof(int32_t)));
    points_.scan_temp.allocate(exclusiveSumTempStorageBytes(entries));
    points_.query.allocate(7 * sizeof(int32_t));
    points_.target.allocate(bytesFor(std::max(1, config_.feature_dim), sizeof(float)));
    points_.selection_dummy.allocate(1);
  }
  SurfacePointScratch scratch;
  scratch.proposals.allocate(bytesFor(sample_budget, 3 * sizeof(float)));
  scratch.sources.allocate(bytesFor(sample_budget, 2 * sizeof(int32_t)));
  scratch.valid.allocate(bytesFor(sample_budget + 1, sizeof(int32_t)));
  scratch.offsets.allocate(bytesFor(sample_budget + 1, sizeof(int32_t)));
  scratch.scan_temp.allocate(exclusiveSumTempStorageBytes(sample_budget + 1));
  scratch.valid.memsetAsync(0, stream);  // Trailing scan sentinel is always zero.
  points_.outputs.emplace(sample_budget, std::move(scratch));
}

void MapperRuntimeImpl::extract_occupied_points(
    uint64_t out_positions, uint64_t out_source_voxels, uint64_t out_count,
    int sample_budget, int instance_slot, uint64_t voxel_selection,
    const CuroboMapperSemanticFilter* filter, int seed, uint64_t stream_handle) {
  CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  if (!out_positions || !out_source_voxels || !out_count || sample_budget <= 0 ||
      sample_budget == std::numeric_limits<int32_t>::max() || seed < 0)
    throw std::invalid_argument("Surface sampling requires outputs, positive budget and nonnegative seed");
  if (instance_slot < -1 || instance_slot > 255 ||
      (instance_slot >= 0 && config_.instance_grid_size == 0))
    throw std::invalid_argument("Invalid surface-point instance slot or disabled instance storage");
  if (filter != nullptr) {
    if (config_.feature_dim == 0 || (!filter->require_argmax && !filter->use_min_similarity) ||
        filter->exclude_instance_slot > 0 ||
        (!filter->target && (filter->target_class < 0 || filter->target_class >= config_.feature_dim)) ||
        (filter->use_min_similarity && (!std::isfinite(filter->min_similarity) ||
          filter->min_similarity < -1.0f || filter->min_similarity > 1.0f)))
      throw std::invalid_argument("Invalid surface-point semantic filter");
  }
  auto stream = toCudaStream(stream_handle);
  auto cu_stream = toCuStream(stream_handle);
  ensure_point_scratch(sample_budget, stream);
  auto& scratch = points_.outputs.at(sample_budget);
  auto tsdf = tsdfWarp();
  auto query = array1<wp::int32>(points_.query.ptr(), 7);
  wp::int32 slot = instance_slot, use_semantic = filter != nullptr;
  wp::int32 use_selection = voxel_selection != 0, sample_seed = seed;
  wp::int32 use_argmax = filter != nullptr && filter->require_argmax;
  wp::int32 exclude = filter != nullptr && filter->exclude;
  wp::int32 use_cosine = filter != nullptr && filter->use_min_similarity;
  wp::float32 min_weight = config_.minimum_tsdf_weight;
  wp::float32 min_similarity = use_cosine ? filter->min_similarity : -1.0f;
  auto dim = launchBounds1(1);
  std::vector<void*> query_args{&dim, &query, &slot, &use_semantic, &use_selection,
                                &sample_seed, &use_argmax, &exclude, &use_cosine};
  kernels_.set_point_query.launch(gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("set_point_query_kernel").shared_mem, cu_stream, query_args);
  uint64_t target_pointer = points_.target.ptr();
  if (filter != nullptr && filter->target != 0) target_pointer = filter->target;
  auto target = array1<wp::float32>(target_pointer, std::max(1, config_.feature_dim));
  if (filter != nullptr && filter->target == 0) {
    dim = launchBounds1(config_.feature_dim);
    wp::int32 label = filter->target_class;
    std::vector<void*> args{&dim, &target, &label};
    kernels_.set_point_target_class.launch(gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
        kernelManifest("set_point_target_class_kernel").shared_mem, cu_stream, args);
  }
  auto selection = array2<wp::uint8>(voxel_selection ? voxel_selection : points_.selection_dummy.ptr(),
      voxel_selection ? config_.max_blocks : 1, voxel_selection ? config_.block_voxels : 1);
  auto flags = array2<wp::uint8>(points_.flags.ptr(), config_.max_blocks, config_.block_voxels);
  auto counts = array1<wp::int32>(points_.counts.ptr(), points_.groups + 1);
  auto offsets = array1<wp::int32>(points_.offsets.ptr(), points_.groups + 1);
  points_.counts.memsetAsync(0, stream);
  dim = launchBounds1(65536);  // Matches POINT_SCAN_THREADS in the shared Warp kernel.
  std::vector<void*> count_args{&dim, &tsdf, &target, &selection, &query,
                                &min_weight, &min_similarity, &flags, &counts};
  kernels_.count_point_edges.launch(gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("count_point_edges_kernel").shared_mem, cu_stream, count_args);
  exclusiveSumInt32(points_.scan_temp.raw(), points_.scan_temp.bytes(),
      static_cast<const int*>(points_.counts.raw()), static_cast<int*>(points_.offsets.raw()),
      points_.groups + 1, stream);
  auto proposals = array2<wp::float32>(scratch.proposals.ptr(), sample_budget, 3);
  auto sources = array2<wp::int32>(scratch.sources.ptr(), sample_budget, 2);
  auto valid = array1<wp::int32>(scratch.valid.ptr(), sample_budget + 1);
  auto sample_offsets = array1<wp::int32>(scratch.offsets.ptr(), sample_budget + 1);
  dim = launchBounds1(sample_budget);
  std::vector<void*> sample_args{&dim, &tsdf, &flags, &offsets, &query, &min_weight,
                                 &proposals, &sources, &valid};
  kernels_.sample_point_edges.launch(gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("sample_point_edges_kernel").shared_mem, cu_stream, sample_args);
  exclusiveSumInt32(scratch.scan_temp.raw(), scratch.scan_temp.bytes(),
      static_cast<const int*>(scratch.valid.raw()), static_cast<int*>(scratch.offsets.raw()),
      sample_budget + 1, stream);
  auto positions = array2<wp::float32>(out_positions, sample_budget, 3);
  auto source_voxels = array2<wp::int32>(out_source_voxels, sample_budget, 2);
  auto count = array1<wp::int32>(out_count, 1);
  std::vector<void*> compact_args{&dim, &proposals, &sources, &valid, &sample_offsets,
                                  &positions, &source_voxels, &count};
  kernels_.compact_surface_points.launch(gridFor(dim.size), Dim3{kWarpBlockDim, 1, 1},
      kernelManifest("compact_surface_points_kernel").shared_mem, cu_stream, compact_args);
}

int MapperRuntimeImpl::numAllocated(uint64_t stream_handle) {
  cudaStream_t cuda_stream = toCudaStream(stream_handle);
  int value = 0;
  CMR_CUDA_CHECK(cudaMemcpyAsync(&value, buffers_.num_allocated.raw(), sizeof(int),
                                 cudaMemcpyDeviceToHost, cuda_stream));
  CMR_CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  return value;
}

CuroboMapperDeviceView MapperRuntimeImpl::deviceView() const {
  CuroboMapperDeviceView view{};
  view.block_grid_rgb = buffers_.block_grid_rgb.ptr();
  view.block_features = buffers_.block_features.ptr();
  view.block_feature_weight = buffers_.block_feature_weight.ptr();
  view.block_instance_state = buffers_.block_instance_state.ptr();
  view.block_coords = buffers_.block_coords.ptr();
  view.block_to_hash_slot = buffers_.block_to_hash_slot.ptr();
  view.num_allocated = buffers_.num_allocated.ptr();
  view.esdf_dist_field = buffers_.esdf_dist_field.ptr();
  view.max_blocks = config_.max_blocks;
  view.block_size = config_.block_size;
  view.block_voxels = config_.block_voxels;
  view.color_grid_size = config_.color_grid_size;
  view.color_grid_voxels = config_.color_grid_voxels;
  view.feature_dim = config_.feature_dim;
  view.feature_block_grid_size = config_.feature_block_grid_size;
  view.feature_grid_voxels = config_.feature_grid_voxels;
  view.instance_grid_size = config_.instance_grid_size;
  view.instance_grid_voxels = config_.instance_grid_voxels;
  view.grid_shape[0] = config_.grid_D;
  view.grid_shape[1] = config_.grid_H;
  view.grid_shape[2] = config_.grid_W;
  view.esdf_shape[0] = config_.esdf_nx;
  view.esdf_shape[1] = config_.esdf_ny;
  view.esdf_shape[2] = config_.esdf_nz;
  std::copy(std::begin(config_.origin), std::end(config_.origin), view.origin);
  view.voxel_size = config_.voxel_size;
  view.esdf_voxel_size = config_.esdf_voxel_size;
  return view;
}

int MapperRuntimeImpl::configInt(const std::string& name) const {
  if (name == "num_cameras") return config_.num_cameras;
  if (name == "image_height") return config_.image_height;
  if (name == "image_width") return config_.image_width;
  if (name == "grid_D") return config_.grid_D;
  if (name == "grid_H") return config_.grid_H;
  if (name == "grid_W") return config_.grid_W;
  if (name == "esdf_nx") return config_.esdf_nx;
  if (name == "esdf_ny") return config_.esdf_ny;
  if (name == "esdf_nz") return config_.esdf_nz;
  if (name == "block_size") return config_.block_size;
  if (name == "num_samples") return config_.num_samples;
  if (name == "max_blocks") return config_.max_blocks;
  if (name == "hash_capacity") return config_.hash_capacity;
  if (name == "pose_refiner_n_samples_per_ray")
    return config_.pose_refiner_n_samples_per_ray;
  if (name == "feature_dim") return config_.feature_dim;
  if (name == "feature_block_grid_size") return config_.feature_block_grid_size;
  if (name == "feature_grid_voxels") return config_.feature_grid_voxels;
  if (name == "instance_grid_size") return config_.instance_grid_size;
  if (name == "instance_grid_voxels") return config_.instance_grid_voxels;
  if (name == "feature_grid_height") return config_.feature_grid_height;
  if (name == "feature_grid_width") return config_.feature_grid_width;
  if (name == "feature_channels_per_thread")
    return config_.feature_channels_per_thread;
  if (name == "max_feature_tile_channels")
    return config_.max_feature_tile_channels;
  if (name == "use_tiled_feature_kernel")
    return config_.use_tiled_feature_kernel ? 1 : 0;
  if (name == "color_grid_size") return config_.color_grid_size;
  if (name == "color_grid_voxels") return config_.color_grid_voxels;
  if (name == "texture_num_cameras") return config_.texture_num_cameras;
  if (name == "texture_camera_image_height")
    return config_.texture_camera_image_height;
  if (name == "texture_camera_image_width")
    return config_.texture_camera_image_width;
  if (name == "max_visible_blocks_per_integration") return config_.max_visible_blocks;
  if (name == "max_support_pixels_per_block_camera")
    return config_.max_support_pixels_per_block_camera;
  throw std::runtime_error("unknown int config key: " + name);
}

float MapperRuntimeImpl::configFloat(const std::string& name) const {
  if (name == "voxel_size") return config_.voxel_size;
  if (name == "esdf_voxel_size") return config_.esdf_voxel_size;
  if (name == "truncation_distance") return config_.truncation_distance;
  if (name == "depth_minimum_distance") return config_.depth_minimum_distance;
  if (name == "depth_maximum_distance") return config_.depth_maximum_distance;
  if (name == "minimum_tsdf_weight") return config_.minimum_tsdf_weight;
  if (name == "adjacent_skip_steps") return config_.adjacent_skip_steps;
  if (name == "decay_factor") return config_.decay_factor;
  if (name == "frustum_decay_factor") return config_.frustum_decay_factor;
  if (name == "accumulator_w_max") return config_.accumulator_w_max;
  if (name == "origin_x") return config_.origin[0];
  if (name == "origin_y") return config_.origin[1];
  if (name == "origin_z") return config_.origin[2];
  throw std::runtime_error("unknown float config key: " + name);
}

void MapperRuntimeImpl::setTsdfRuntimeParameters(
    float decay_factor, float minimum_tsdf_weight) {
  if (!std::isfinite(decay_factor) || decay_factor <= 0.0f ||
      decay_factor > 1.0f) {
    throw std::invalid_argument("decay_factor must be finite and in (0, 1]");
  }
  if (!std::isfinite(minimum_tsdf_weight) || minimum_tsdf_weight <= 0.0f ||
      minimum_tsdf_weight > config_.accumulator_w_max) {
    throw std::invalid_argument(
        "minimum_tsdf_weight must be finite, positive, and no larger than "
        "accumulator_w_max");
  }
  if (minimum_tsdf_weight != config_.minimum_tsdf_weight &&
      esdf_graph_executable_ != nullptr) {
    CMR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
    CMR_CUDA_CHECK(cudaDeviceSynchronize());
    destroyEsdfGraph();
  }
  config_.decay_factor = decay_factor;
  config_.minimum_tsdf_weight = minimum_tsdf_weight;
}

}  // namespace curobo_mapper_runtime
