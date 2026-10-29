// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "robot_runtime.h"

#include "cuda_check.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace curobo_robot_runtime {
namespace {

std::string read_text(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::ostringstream output;
  output << stream.rdbuf();
  return output.str();
}

std::vector<uint8_t> read_binary(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("failed to open " + path.string());
  }
  const auto end = stream.tellg();
  if (end < 0) {
    throw std::runtime_error("failed to determine size of " + path.string());
  }
  std::vector<uint8_t> data(static_cast<size_t>(end));
  stream.seekg(0, std::ios::beg);
  if (!data.empty() && !stream.read(reinterpret_cast<char*>(data.data()), end)) {
    throw std::runtime_error("failed to read " + path.string());
  }
  return data;
}

size_t value_position(const std::string& json, const std::string& key) {
  const auto key_position = json.find('"' + key + '"');
  if (key_position == std::string::npos) {
    throw std::runtime_error("manifest is missing key: " + key);
  }
  const auto colon = json.find(':', key_position + key.size() + 2);
  if (colon == std::string::npos) {
    throw std::runtime_error("manifest key has no value: " + key);
  }
  auto position = colon + 1;
  while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position]))) {
    ++position;
  }
  return position;
}

std::string parse_string(const std::string& json, size_t* position) {
  if (*position >= json.size() || json[*position] != '"') {
    throw std::runtime_error("expected a JSON string");
  }
  ++*position;
  std::string result;
  while (*position < json.size()) {
    const char character = json[(*position)++];
    if (character == '"') {
      return result;
    }
    if (character != '\\') {
      result.push_back(character);
      continue;
    }
    if (*position >= json.size()) {
      break;
    }
    const char escaped = json[(*position)++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        result.push_back(escaped);
        break;
      case 'b':
        result.push_back('\b');
        break;
      case 'f':
        result.push_back('\f');
        break;
      case 'n':
        result.push_back('\n');
        break;
      case 'r':
        result.push_back('\r');
        break;
      case 't':
        result.push_back('\t');
        break;
      default:
        throw std::runtime_error("unsupported JSON escape in manifest");
    }
  }
  throw std::runtime_error("unterminated JSON string in manifest");
}

std::string extract_string(const std::string& json, const std::string& key) {
  auto position = value_position(json, key);
  return parse_string(json, &position);
}

int extract_int(const std::string& json, const std::string& key) {
  const auto position = value_position(json, key);
  size_t consumed = 0;
  const long value = std::stol(json.substr(position), &consumed);
  if (consumed == 0 || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("manifest integer is out of range: " + key);
  }
  return static_cast<int>(value);
}

float extract_float(const std::string& json, const std::string& key) {
  const auto position = value_position(json, key);
  size_t consumed = 0;
  const float value = std::stof(json.substr(position), &consumed);
  if (consumed == 0 || !std::isfinite(value)) {
    throw std::runtime_error("manifest float is invalid: " + key);
  }
  return value;
}

bool has_key(const std::string& json, const std::string& key) {
  return json.find('"' + key + '"') != std::string::npos;
}

std::string extract_object(const std::string& json, const std::string& key) {
  const auto start = value_position(json, key);
  if (start >= json.size() || json[start] != '{') {
    throw std::runtime_error("manifest key is not an object: " + key);
  }
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (auto position = start; position < json.size(); ++position) {
    const char character = json[position];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == '{') {
      ++depth;
    } else if (character == '}' && --depth == 0) {
      return json.substr(start, position - start + 1);
    }
  }
  throw std::runtime_error("unterminated manifest object: " + key);
}

std::vector<std::string> extract_string_array(const std::string& json,
                                              const std::string& key) {
  auto position = value_position(json, key);
  if (position >= json.size() || json[position] != '[') {
    throw std::runtime_error("manifest key is not an array: " + key);
  }
  ++position;
  std::vector<std::string> result;
  while (position < json.size()) {
    while (position < json.size() &&
           (std::isspace(static_cast<unsigned char>(json[position])) ||
            json[position] == ',')) {
      ++position;
    }
    if (position < json.size() && json[position] == ']') {
      return result;
    }
    result.push_back(parse_string(json, &position));
  }
  throw std::runtime_error("unterminated manifest array: " + key);
}

TensorManifest extract_tensor(const std::string& tensors, const std::string& name) {
  const auto object = extract_object(tensors, name);
  const int bytes = extract_int(object, "bytes");
  if (bytes < 0) {
    throw std::runtime_error("tensor byte count must be non-negative: " + name);
  }
  return TensorManifest{
      extract_string(object, "file"),
      extract_string(object, "dtype"),
      static_cast<size_t>(bytes),
  };
}

IkKernelManifest extract_ik_kernel(const std::string& kernels,
                                   const std::string& name) {
  if (!has_key(kernels, name)) {
    return {};
  }
  const auto object = extract_object(kernels, name);
  IkKernelManifest kernel;
  kernel.cubin = extract_string(object, "cubin");
  kernel.symbol = extract_string(object, "symbol");
  kernel.source = extract_string(object, "source");
  kernel.shared_memory = extract_int(object, "shared_memory");
  if (has_key(object, "block_dim")) {
    kernel.block_dim = extract_int(object, "block_dim");
  }
  if (has_key(object, "max_obstacles")) {
    kernel.max_obstacles = extract_int(object, "max_obstacles");
  }
  return kernel;
}

KernelManifest extract_kinematics_kernel(const std::string& kernels,
                                         const std::string& name) {
  const auto object = extract_object(kernels, name);
  return KernelManifest{
      extract_string(object, "cubin"),
      extract_string(object, "symbol"),
      extract_string(object, "source"),
      extract_int(object, "threads_per_batch"),
      extract_int(object, "max_batches_per_block"),
      extract_int(object, "max_threads_per_block"),
      extract_int(object, "max_dynamic_shared_memory"),
  };
}

size_t checked_product(size_t left, size_t right, const char* description) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    throw std::runtime_error(std::string(description) + " size overflow");
  }
  return left * right;
}

void validate_tensor(const TensorManifest& tensor, const std::string& expected_dtype,
                     size_t expected_bytes, const std::string& name) {
  if (tensor.dtype != expected_dtype) {
    throw std::runtime_error(name + " has dtype " + tensor.dtype + ", expected " +
                             expected_dtype);
  }
  if (tensor.bytes != expected_bytes) {
    throw std::runtime_error(name + " has " + std::to_string(tensor.bytes) +
                             " bytes, expected " + std::to_string(expected_bytes));
  }
}

std::vector<uint8_t> read_tensor(const std::filesystem::path& artifact_dir,
                                 const TensorManifest& tensor,
                                 const std::string& name) {
  auto data = read_binary(artifact_dir / tensor.file);
  if (data.size() != tensor.bytes) {
    throw std::runtime_error(name + " file has " + std::to_string(data.size()) +
                             " bytes, expected " + std::to_string(tensor.bytes));
  }
  return data;
}

void validate_tool_offsets(const std::vector<uint8_t>& offsets_data,
                           size_t contributor_count) {
  if (offsets_data.size() % sizeof(int16_t) != 0) {
    throw std::runtime_error("tool_joint_links_offsets byte count is invalid");
  }
  const auto count = offsets_data.size() / sizeof(int16_t);
  std::vector<int16_t> offsets(count);
  std::memcpy(offsets.data(), offsets_data.data(), offsets_data.size());
  int previous = 0;
  for (const int offset : offsets) {
    if (offset < previous || static_cast<size_t>(offset) > contributor_count) {
      throw std::runtime_error("tool_joint_links_offsets is not a valid CSR offset array");
    }
    previous = offset;
  }
  if (offsets.empty() || static_cast<size_t>(offsets.back()) != contributor_count) {
    throw std::runtime_error(
        "tool_joint_links_offsets does not terminate at tool_joint_links_data size");
  }
}

}  // namespace

RobotManifest load_manifest(const std::string& artifact_dir) {
  const auto path = std::filesystem::path(artifact_dir) / "robot_kinematics_manifest.json";
  const auto json = read_text(path);
  const auto tensors = extract_object(json, "tensors");
  const auto kernel = extract_object(json, "kernel");

  RobotManifest manifest;
  manifest.schema_version = extract_int(json, "schema_version");
  manifest.arch = extract_string(json, "arch");
  manifest.num_links = extract_int(json, "num_links");
  manifest.num_dof = extract_int(json, "num_dof");
  manifest.num_tool_frames = extract_int(json, "num_tool_frames");
  if (has_key(json, "link_names")) {
    manifest.link_names = extract_string_array(json, "link_names");
  }
  manifest.joint_names = extract_string_array(json, "joint_names");
  manifest.tool_frames = extract_string_array(json, "tool_frames");
  manifest.fixed_transforms = extract_tensor(tensors, "fixed_transforms");
  manifest.joint_map_type = extract_tensor(tensors, "joint_map_type");
  manifest.joint_map = extract_tensor(tensors, "joint_map");
  manifest.link_map = extract_tensor(tensors, "link_map");
  manifest.tool_frame_map = extract_tensor(tensors, "tool_frame_map");
  manifest.joint_offset_map = extract_tensor(tensors, "joint_offset_map");
  manifest.tool_joint_links_data = extract_tensor(tensors, "tool_joint_links_data");
  manifest.tool_joint_links_offsets =
      extract_tensor(tensors, "tool_joint_links_offsets");
  if (has_key(tensors, "joint_link_influence_words")) {
    manifest.joint_link_influence_words =
        extract_tensor(tensors, "joint_link_influence_words");
    manifest.joint_links_data = extract_tensor(tensors, "joint_links_data");
    manifest.joint_links_offsets = extract_tensor(tensors, "joint_links_offsets");
    manifest.link_masses_com = extract_tensor(tensors, "link_masses_com");
  }
  manifest.kernel = KernelManifest{
      extract_string(kernel, "cubin"),
      extract_string(kernel, "symbol"),
      extract_string(kernel, "source"),
      extract_int(kernel, "threads_per_batch"),
      extract_int(kernel, "max_batches_per_block"),
      extract_int(kernel, "max_threads_per_block"),
      extract_int(kernel, "max_dynamic_shared_memory"),
  };

  if (manifest.schema_version != 1) {
    throw std::runtime_error("unsupported robot artifact schema version: " +
                             std::to_string(manifest.schema_version));
  }
  if (manifest.num_links <= 0 || manifest.num_dof <= 0 ||
      manifest.num_tool_frames <= 0) {
    throw std::runtime_error("robot artifact dimensions must be positive");
  }
  if (manifest.joint_names.size() != static_cast<size_t>(manifest.num_dof)) {
    throw std::runtime_error("joint_names count does not match num_dof");
  }
  if (manifest.tool_frames.size() != static_cast<size_t>(manifest.num_tool_frames)) {
    throw std::runtime_error("tool_frames count does not match num_tool_frames");
  }
  if (!manifest.link_names.empty() &&
      manifest.link_names.size() != static_cast<size_t>(manifest.num_links)) {
    throw std::runtime_error("link_names count does not match num_links");
  }
  if (manifest.kernel.source != "curobo_kinematics_kernel") {
    throw std::runtime_error("robot artifact does not use the in-tree cuRobo kernel");
  }
  if (manifest.kernel.threads_per_batch != 32 ||
      manifest.kernel.max_batches_per_block <= 0 ||
      manifest.kernel.max_threads_per_block < manifest.kernel.threads_per_batch ||
      manifest.kernel.max_dynamic_shared_memory <= 0) {
    throw std::runtime_error("robot artifact has an invalid kinematics launch policy");
  }

  const auto num_links = static_cast<size_t>(manifest.num_links);
  const auto num_dof = static_cast<size_t>(manifest.num_dof);
  const auto num_tool_frames = static_cast<size_t>(manifest.num_tool_frames);
  validate_tensor(manifest.fixed_transforms, "float32",
                  checked_product(num_links, 12 * sizeof(float), "fixed transforms"),
                  "fixed_transforms");
  validate_tensor(manifest.joint_map_type, "int8", num_links, "joint_map_type");
  validate_tensor(manifest.joint_map, "int16", num_links * sizeof(int16_t),
                  "joint_map");
  validate_tensor(manifest.link_map, "int16", num_links * sizeof(int16_t),
                  "link_map");
  validate_tensor(manifest.tool_frame_map, "int16",
                  num_tool_frames * sizeof(int16_t), "tool_frame_map");
  validate_tensor(manifest.joint_offset_map, "float32",
                  checked_product(num_links, 2 * sizeof(float), "joint offsets"),
                  "joint_offset_map");
  if (manifest.tool_joint_links_data.dtype != "int16" ||
      manifest.tool_joint_links_data.bytes % sizeof(int16_t) != 0) {
    throw std::runtime_error("tool_joint_links_data must contain int16 values");
  }
  const auto offset_count = checked_product(num_tool_frames, num_dof, "tool offsets") + 1;
  validate_tensor(manifest.tool_joint_links_offsets, "int16",
                  offset_count * sizeof(int16_t), "tool_joint_links_offsets");
  if (!manifest.joint_link_influence_words.file.empty()) {
    const auto influence_words = (num_links + 31) / 32;
    validate_tensor(manifest.joint_link_influence_words, "int32",
                    num_links * influence_words * sizeof(int32_t),
                    "joint_link_influence_words");
    if (manifest.joint_links_data.dtype != "int16" ||
        manifest.joint_links_data.bytes == 0 ||
        manifest.joint_links_data.bytes % sizeof(int16_t) != 0) {
      throw std::runtime_error("joint_links_data must contain int16 values");
    }
    validate_tensor(manifest.joint_links_offsets, "int16",
                    (num_dof + 1) * sizeof(int16_t), "joint_links_offsets");
    validate_tensor(manifest.link_masses_com, "float32",
                    num_links * 4 * sizeof(float), "link_masses_com");
  }
  return manifest;
}

IkManifest load_ik_manifest(const std::string& artifact_dir) {
  const auto path = std::filesystem::path(artifact_dir) / "robot_ik_manifest.json";
  const auto json = read_text(path);
  const auto tensors = extract_object(json, "tensors");
  const auto kernels = extract_object(json, "kernels");
  const auto capacities = extract_object(json, "scene_capacities");
  const auto optimizer = extract_object(json, "optimizer");
  const auto kinematics_object = extract_object(kernels, "kinematics");

  IkManifest manifest;
  manifest.schema_version = extract_int(json, "schema_version");
  manifest.arch = extract_string(json, "arch");
  manifest.num_links = extract_int(json, "num_links");
  manifest.num_dof = extract_int(json, "num_dof");
  manifest.num_tool_frames = extract_int(json, "num_tool_frames");
  manifest.num_spheres = extract_int(json, "num_spheres");
  manifest.num_residuals = extract_int(json, "num_residuals");
  manifest.max_seeds = extract_int(json, "max_seeds");
  manifest.num_seed_samples = extract_int(json, "num_seed_samples");
  manifest.max_goalset = extract_int(json, "max_goalset");
  manifest.num_self_collision_pairs = extract_int(json, "num_self_collision_pairs");
  manifest.max_cuboids = extract_int(capacities, "cuboid");
  manifest.max_capsules = extract_int(capacities, "capsule");
  manifest.max_cylinders = extract_int(capacities, "cylinder");
  manifest.max_meshes = extract_int(capacities, "mesh");
  manifest.max_voxel_grids = extract_int(capacities, "voxel");
  manifest.history = extract_int(json, "history");
  manifest.num_line_search = extract_int(json, "num_line_search");
  manifest.optimizer_iterations = extract_int(optimizer, "num_iterations");
  manifest.convergence_iteration =
      extract_int(optimizer, "convergence_iteration");
  manifest.epsilon = extract_float(optimizer, "epsilon");
  manifest.initial_step_scale = extract_float(optimizer, "initial_step_scale");
  manifest.armijo_threshold = extract_float(optimizer, "armijo_threshold");
  manifest.curvature_threshold = extract_float(optimizer, "curvature_threshold");
  manifest.cost_delta_threshold =
      extract_float(optimizer, "cost_delta_threshold");
  manifest.cost_relative_threshold =
      extract_float(optimizer, "cost_relative_threshold");
  manifest.position_weight = extract_float(optimizer, "position_weight");
  manifest.orientation_weight = extract_float(optimizer, "orientation_weight");
  manifest.joint_position_lower = extract_tensor(tensors, "joint_position_lower");
  manifest.joint_position_upper = extract_tensor(tensors, "joint_position_upper");
  manifest.default_joint_position = extract_tensor(tensors, "default_joint_position");
  manifest.seed_positions = extract_tensor(tensors, "seed_positions");
  manifest.link_spheres = extract_tensor(tensors, "link_spheres");
  manifest.sphere_collision_mask =
      extract_tensor(tensors, "sphere_collision_mask");
  manifest.link_sphere_map = extract_tensor(tensors, "link_sphere_map");
  if (manifest.num_self_collision_pairs > 0) {
    manifest.self_collision_padding =
        extract_tensor(tensors, "self_collision_padding");
    manifest.self_collision_pairs =
        extract_tensor(tensors, "self_collision_pairs");
  }
  manifest.position_bounds = extract_tensor(tensors, "ik_position_bounds");
  manifest.velocity_bounds = extract_tensor(tensors, "ik_velocity_bounds");
  manifest.cspace_weight = extract_tensor(tensors, "ik_cspace_weight");
  manifest.cspace_activation_distance =
      extract_tensor(tensors, "ik_cspace_activation_distance");
  manifest.cspace_target_weight =
      extract_tensor(tensors, "ik_cspace_target_weight");
  manifest.cspace_target_dof_weight =
      extract_tensor(tensors, "ik_cspace_target_dof_weight");
  manifest.cspace_regularization_weight =
      extract_tensor(tensors, "ik_cspace_regularization_weight");
  manifest.start_cspace_weight =
      extract_tensor(tensors, "ik_start_cspace_weight");
  manifest.start_cspace_dof_weight =
      extract_tensor(tensors, "ik_start_cspace_dof_weight");
  manifest.world_collision_weight =
      extract_tensor(tensors, "ik_world_collision_weight");
  manifest.self_collision_weight =
      extract_tensor(tensors, "ik_self_collision_weight");
  manifest.optimizer_collision_activation_distance =
      extract_tensor(tensors, "ik_optimizer_collision_activation_distance");
  manifest.line_search_scale = extract_tensor(tensors, "ik_line_search_scale");
  manifest.optimization_step_max =
      extract_tensor(tensors, "ik_optimization_step_max");
  manifest.kinematics = KernelManifest{
      extract_string(kinematics_object, "cubin"),
      extract_string(kinematics_object, "symbol"),
      extract_string(kinematics_object, "source"),
      extract_int(kinematics_object, "threads_per_batch"),
      extract_int(kinematics_object, "max_batches_per_block"),
      extract_int(kinematics_object, "max_threads_per_block"),
      extract_int(kinematics_object, "max_dynamic_shared_memory"),
  };
  manifest.kinematics_backward =
      extract_ik_kernel(kernels, "kinematics_backward");
  manifest.initialize = extract_ik_kernel(kernels, "initialize");
  manifest.error = extract_ik_kernel(kernels, "error");
  manifest.update = extract_ik_kernel(kernels, "update");
  manifest.rank = extract_ik_kernel(kernels, "rank");
  manifest.finalize = extract_ik_kernel(kernels, "finalize");
  manifest.levenberg_marquardt =
      extract_ik_kernel(kernels, "levenberg_marquardt");
  manifest.cspace = extract_ik_kernel(kernels, "cspace");
  manifest.prepare_initial_state =
      extract_ik_kernel(kernels, "prepare_initial_state");
  manifest.prepare_search_actions =
      extract_ik_kernel(kernels, "prepare_search_actions");
  manifest.prepare_search_actions_backtracking =
      extract_ik_kernel(kernels, "prepare_search_actions_backtracking");
  manifest.combine_lbfgs_rollout =
      extract_ik_kernel(kernels, "combine_lbfgs_rollout");
  manifest.line_search = extract_ik_kernel(kernels, "line_search");
  manifest.line_search_backtracking = extract_ik_kernel(kernels, "line_search_backtracking");
  manifest.lbfgs = extract_ik_kernel(kernels, "lbfgs");
  manifest.collision_cuboid = extract_ik_kernel(kernels, "collision_cuboid");
  manifest.collision_capsule = extract_ik_kernel(kernels, "collision_capsule");
  manifest.collision_cylinder = extract_ik_kernel(kernels, "collision_cylinder");
  manifest.collision_mesh = extract_ik_kernel(kernels, "collision_mesh");
  manifest.collision_voxel = extract_ik_kernel(kernels, "collision_voxel");
  manifest.self_collision = extract_ik_kernel(kernels, "self_collision");

  if (manifest.schema_version != 13) {
    throw std::runtime_error("unsupported robot IK artifact schema version: " +
                             std::to_string(manifest.schema_version));
  }
  if (manifest.num_links <= 0 || manifest.num_dof <= 0 ||
      manifest.num_tool_frames <= 0 ||
      manifest.num_spheres < 0 || manifest.num_residuals <= 0 ||
      manifest.max_seeds <= 0 ||
      manifest.num_seed_samples < manifest.max_seeds ||
      manifest.max_goalset <= 0 ||
      manifest.num_self_collision_pairs < 0 || manifest.history <= 0 ||
      manifest.num_line_search <= 0 || manifest.num_line_search > 32 ||
      manifest.optimizer_iterations <= 0 || manifest.convergence_iteration <= 0 ||
      manifest.epsilon <= 0.0F || manifest.initial_step_scale <= 0.0F ||
      manifest.position_weight <= 0.0F || manifest.orientation_weight <= 0.0F) {
    throw std::runtime_error("robot IK artifact dimensions are invalid");
  }
  const auto dof = static_cast<size_t>(manifest.num_dof);
  const auto sphere_count = static_cast<size_t>(manifest.num_spheres);
  validate_tensor(manifest.joint_position_lower, "float32", dof * sizeof(float),
                  "joint_position_lower");
  validate_tensor(manifest.joint_position_upper, "float32", dof * sizeof(float),
                  "joint_position_upper");
  validate_tensor(manifest.default_joint_position, "float32", dof * sizeof(float),
                  "default_joint_position");
  validate_tensor(manifest.seed_positions, "float32",
                  static_cast<size_t>(manifest.num_seed_samples) * dof *
                      sizeof(float),
                  "seed_positions");
  validate_tensor(manifest.sphere_collision_mask, "uint32",
                  static_cast<size_t>(manifest.num_spheres) * sizeof(uint32_t),
                  "sphere_collision_mask");
  validate_tensor(manifest.link_spheres, "float32", sphere_count * 4 * sizeof(float),
                  "link_spheres");
  validate_tensor(manifest.link_sphere_map, "int16", sphere_count * sizeof(int16_t),
                  "link_sphere_map");
  validate_tensor(manifest.position_bounds, "float32", 2 * dof * sizeof(float),
                  "ik_position_bounds");
  validate_tensor(manifest.velocity_bounds, "float32", 2 * dof * sizeof(float),
                  "ik_velocity_bounds");
  validate_tensor(manifest.cspace_weight, "float32", 2 * sizeof(float),
                  "ik_cspace_weight");
  validate_tensor(manifest.cspace_activation_distance, "float32",
                  2 * sizeof(float), "ik_cspace_activation_distance");
  validate_tensor(manifest.cspace_target_weight, "float32", sizeof(float),
                  "ik_cspace_target_weight");
  validate_tensor(manifest.cspace_target_dof_weight, "float32",
                  dof * sizeof(float), "ik_cspace_target_dof_weight");
  validate_tensor(manifest.cspace_regularization_weight, "float32",
                  2 * sizeof(float), "ik_cspace_regularization_weight");
  validate_tensor(manifest.start_cspace_weight, "float32", sizeof(float),
                  "ik_start_cspace_weight");
  validate_tensor(manifest.start_cspace_dof_weight, "float32",
                  dof * sizeof(float), "ik_start_cspace_dof_weight");
  validate_tensor(manifest.world_collision_weight, "float32", sizeof(float),
                  "ik_world_collision_weight");
  validate_tensor(manifest.self_collision_weight, "float32", sizeof(float),
                  "ik_self_collision_weight");
  validate_tensor(manifest.optimizer_collision_activation_distance, "float32",
                  sizeof(float), "ik_optimizer_collision_activation_distance");
  validate_tensor(manifest.line_search_scale, "float32",
                  static_cast<size_t>(manifest.num_line_search) * sizeof(float),
                  "ik_line_search_scale");
  validate_tensor(manifest.optimization_step_max, "float32", dof * sizeof(float),
                  "ik_optimization_step_max");
  if (manifest.num_self_collision_pairs > 0) {
    validate_tensor(manifest.self_collision_padding, "float32",
                    sphere_count * sizeof(float), "self_collision_padding");
    validate_tensor(
        manifest.self_collision_pairs, "int16",
        static_cast<size_t>(manifest.num_self_collision_pairs) * 2 * sizeof(int16_t),
        "self_collision_pairs");
    if (manifest.self_collision.cubin.empty() ||
        manifest.self_collision.symbol.empty() ||
        manifest.self_collision.source != "curobo_self_collision_kernel" ||
        manifest.self_collision.shared_memory < 0 ||
        manifest.self_collision.block_dim <= 0) {
      throw std::runtime_error("robot IK artifact has invalid self-collision metadata");
    }
  }
       for (const auto* kernel : {
           &manifest.initialize,
           &manifest.error,
           &manifest.update,
           &manifest.finalize,
           &manifest.levenberg_marquardt,
           &manifest.cspace,
           &manifest.prepare_initial_state,
           &manifest.prepare_search_actions,
           &manifest.prepare_search_actions_backtracking,
           &manifest.combine_lbfgs_rollout,
       }) {
    if (kernel->cubin.empty() || kernel->symbol.empty() ||
        kernel->source != "curobo_python_warp_kernel" ||
        kernel->shared_memory < 0 || kernel->block_dim <= 0) {
      throw std::runtime_error("robot IK artifact has invalid Warp kernel metadata");
    }
  }
  if (manifest.kinematics_backward.cubin.empty() ||
      manifest.kinematics_backward.symbol.empty() ||
      manifest.kinematics_backward.source != "curobo_kinematics_kernel" ||
      manifest.kinematics_backward.shared_memory < 0 ||
      manifest.kinematics_backward.block_dim <= 0) {
    throw std::runtime_error("robot IK artifact has invalid backward kinematics metadata");
  }
  if (manifest.lbfgs.cubin.empty() || manifest.lbfgs.symbol.empty() ||
      manifest.lbfgs.source != "curobo_lbfgs_kernel" ||
      manifest.lbfgs.shared_memory < 0 || manifest.lbfgs.block_dim <= 0 ||
      manifest.line_search.cubin.empty() || manifest.line_search.symbol.empty() ||
      manifest.line_search.source != "curobo_line_search_kernel" ||
      manifest.line_search.shared_memory < 0 ||
      manifest.line_search.block_dim <= 0 ||
      manifest.line_search_backtracking.cubin.empty() ||
      manifest.line_search_backtracking.symbol.empty() ||
      manifest.line_search_backtracking.source != "curobo_line_search_kernel" ||
      manifest.line_search_backtracking.shared_memory < 0 ||
      manifest.line_search_backtracking.block_dim <= 0) {
    throw std::runtime_error("robot IK artifact has invalid L-BFGS metadata");
  }
  if (manifest.kinematics.source != "curobo_kinematics_kernel") {
    throw std::runtime_error("robot IK artifact does not use the in-tree kinematics kernel");
  }
  return manifest;
}

TrajectoryOptimizationManifest load_trajectory_optimization_manifest(
    const std::string& artifact_dir) {
  const auto path =
      std::filesystem::path(artifact_dir) / "robot_trajopt_manifest.json";
  const auto json = read_text(path);
  const auto tensors = extract_object(json, "tensors");
  const auto kernels = extract_object(json, "kernels");
  const auto optimizer = extract_object(json, "optimizer");
  const auto capacities = extract_object(json, "scene_capacities");

  TrajectoryOptimizationManifest manifest;
  manifest.schema_version = extract_int(json, "schema_version");
  manifest.arch = extract_string(json, "arch");
  manifest.num_dof = extract_int(json, "num_dof");
  manifest.num_links = extract_int(json, "num_links");
  manifest.num_tool_frames = extract_int(json, "num_tool_frames");
  manifest.num_spheres = extract_int(json, "num_spheres");
  manifest.num_self_collision_pairs =
      extract_int(json, "num_self_collision_pairs");
  manifest.num_knots = extract_int(json, "num_knots");
  manifest.trajectory_steps = extract_int(json, "trajectory_steps");
  manifest.interpolation_buffer_size =
      extract_int(json, "interpolation_buffer_size");
  manifest.spline_degree = extract_int(json, "spline_degree");
  manifest.max_seeds = extract_int(json, "max_seeds");
  manifest.history = extract_int(json, "history");
  manifest.num_line_search = extract_int(json, "num_line_search");
  manifest.base_dt = extract_float(json, "base_dt");
  manifest.minimum_dt = extract_float(json, "minimum_dt");
  manifest.maximum_dt = extract_float(json, "maximum_dt");
  manifest.interpolation_dt = extract_float(json, "interpolation_dt");
  manifest.epsilon = extract_float(optimizer, "epsilon");
  manifest.initial_step_scale = extract_float(optimizer, "initial_step_scale");
  manifest.armijo_threshold = extract_float(optimizer, "armijo_threshold");
  manifest.curvature_threshold = extract_float(optimizer, "curvature_threshold");
  manifest.convergence_iteration =
      extract_int(optimizer, "convergence_iteration");
  manifest.cost_delta_threshold =
      extract_float(optimizer, "cost_delta_threshold");
  manifest.cost_relative_threshold =
      extract_float(optimizer, "cost_relative_threshold");
  manifest.max_cuboids = extract_int(capacities, "cuboid");
  manifest.max_capsules = extract_int(capacities, "capsule");
  manifest.max_cylinders = extract_int(capacities, "cylinder");
  manifest.max_meshes = extract_int(capacities, "mesh");
  manifest.max_voxel_grids = extract_int(capacities, "voxel");

  manifest.position_bounds = extract_tensor(tensors, "position_bounds");
  manifest.velocity_bounds = extract_tensor(tensors, "velocity_bounds");
  manifest.acceleration_bounds = extract_tensor(tensors, "acceleration_bounds");
  manifest.jerk_bounds = extract_tensor(tensors, "jerk_bounds");
  manifest.state_bound_weight = extract_tensor(tensors, "state_bound_weight");
  manifest.state_activation_distance =
      extract_tensor(tensors, "state_activation_distance");
  manifest.state_regularization_weight =
      extract_tensor(tensors, "state_regularization_weight");
  manifest.cspace_target_weight =
      extract_tensor(tensors, "cspace_target_weight");
  manifest.cspace_non_terminal_weight =
      extract_tensor(tensors, "cspace_non_terminal_weight");
  manifest.cspace_target_dof_weight =
      extract_tensor(tensors, "cspace_target_dof_weight");
  manifest.tool_pose_weight = extract_tensor(tensors, "tool_pose_weight");
  manifest.world_collision_weight =
      extract_tensor(tensors, "world_collision_weight");
  manifest.self_collision_weight =
      extract_tensor(tensors, "self_collision_weight");
  manifest.metrics_world_collision_weight =
      extract_tensor(tensors, "metrics_world_collision_weight");
  manifest.metrics_self_collision_weight =
      extract_tensor(tensors, "metrics_self_collision_weight");
  manifest.validation_self_collision_weight =
      extract_tensor(tensors, "validation_self_collision_weight");
  manifest.line_search_scale = extract_tensor(tensors, "line_search_scale");
  manifest.optimization_step_max =
      extract_tensor(tensors, "optimization_step_max");
  manifest.link_spheres = extract_tensor(tensors, "link_spheres");
  manifest.sphere_collision_mask =
      extract_tensor(tensors, "sphere_collision_mask");
  manifest.link_sphere_map = extract_tensor(tensors, "link_sphere_map");
  if (manifest.num_self_collision_pairs > 0) {
    manifest.self_collision_padding =
        extract_tensor(tensors, "self_collision_padding");
    manifest.self_collision_pairs =
        extract_tensor(tensors, "self_collision_pairs");
  }

  manifest.kinematics = extract_kinematics_kernel(kernels, "kinematics");
  manifest.kinematics_backward =
      extract_ik_kernel(kernels, "kinematics_backward");
  manifest.bspline_forward = extract_ik_kernel(kernels, "bspline_forward");
  manifest.bspline_single_dt =
      extract_ik_kernel(kernels, "bspline_single_dt");
  manifest.bspline_backward = extract_ik_kernel(kernels, "bspline_backward");
  manifest.time_scaled_dt = extract_ik_kernel(kernels, "time_scaled_dt");
  manifest.cspace = extract_ik_kernel(kernels, "cspace");
  manifest.pose = extract_ik_kernel(kernels, "pose");
  manifest.collision_cuboid = extract_ik_kernel(kernels, "collision_cuboid");
  manifest.collision_capsule = extract_ik_kernel(kernels, "collision_capsule");
  manifest.collision_cylinder = extract_ik_kernel(kernels, "collision_cylinder");
  manifest.collision_mesh = extract_ik_kernel(kernels, "collision_mesh");
  manifest.collision_voxel = extract_ik_kernel(kernels, "collision_voxel");
  manifest.metrics_collision_cuboid =
      extract_ik_kernel(kernels, "metrics_collision_cuboid");
  manifest.metrics_collision_capsule =
      extract_ik_kernel(kernels, "metrics_collision_capsule");
  manifest.metrics_collision_cylinder =
      extract_ik_kernel(kernels, "metrics_collision_cylinder");
  manifest.metrics_collision_mesh =
      extract_ik_kernel(kernels, "metrics_collision_mesh");
  manifest.metrics_collision_voxel =
      extract_ik_kernel(kernels, "metrics_collision_voxel");
  manifest.validation_collision_cuboid =
      extract_ik_kernel(kernels, "validation_collision_cuboid");
  manifest.validation_collision_capsule =
      extract_ik_kernel(kernels, "validation_collision_capsule");
  manifest.validation_collision_cylinder =
      extract_ik_kernel(kernels, "validation_collision_cylinder");
  manifest.validation_collision_mesh =
      extract_ik_kernel(kernels, "validation_collision_mesh");
  manifest.validation_collision_voxel =
      extract_ik_kernel(kernels, "validation_collision_voxel");
  manifest.self_collision = extract_ik_kernel(kernels, "self_collision");
  manifest.initialize_actions = extract_ik_kernel(kernels, "initialize_actions");
  manifest.prepare_pose_criteria =
      extract_ik_kernel(kernels, "prepare_pose_criteria");
  manifest.expand_trajectory_dt =
      extract_ik_kernel(kernels, "expand_trajectory_dt");
  manifest.reduce_required_dt =
      extract_ik_kernel(kernels, "reduce_required_dt");
  manifest.calculate_interpolation_horizon =
      extract_ik_kernel(kernels, "calculate_interpolation_horizon");
  manifest.validate_interpolated =
      extract_ik_kernel(kernels, "validate_interpolated");
  manifest.prepare_initial_state =
      extract_ik_kernel(kernels, "prepare_initial_state");
  manifest.prepare_search_actions =
      extract_ik_kernel(kernels, "prepare_search_actions");
  manifest.prepare_search_actions_backtracking =
      extract_ik_kernel(kernels, "prepare_search_actions_backtracking");
  manifest.combine_rollout = extract_ik_kernel(kernels, "combine_rollout");
  manifest.accumulate = extract_ik_kernel(kernels, "accumulate");
  manifest.finalize = extract_ik_kernel(kernels, "finalize");
  manifest.line_search = extract_ik_kernel(kernels, "line_search");
  manifest.line_search_backtracking = extract_ik_kernel(kernels, "line_search_backtracking");
  manifest.lbfgs = extract_ik_kernel(kernels, "lbfgs");

  if (manifest.schema_version != 18) {
    throw std::runtime_error("unsupported TrajOpt artifact schema; re-export the robot artifact");
  }
  if (manifest.num_dof <= 0 ||
      manifest.num_links <= 0 || manifest.num_tool_frames <= 0 ||
      manifest.num_spheres <= 0 || manifest.num_self_collision_pairs < 0 ||
      manifest.num_knots <= 5 || manifest.trajectory_steps <= 16 ||
      manifest.interpolation_buffer_size < manifest.trajectory_steps ||
      manifest.spline_degree != 3 || manifest.max_seeds <= 0 ||
      manifest.history <= 0 || manifest.num_line_search <= 0 ||
      manifest.num_line_search > 32 || manifest.base_dt <= 0.0F ||
      manifest.minimum_dt <= 0.0F ||
      manifest.maximum_dt < manifest.minimum_dt ||
      manifest.interpolation_dt <= 0.0F || manifest.epsilon <= 0.0F ||
      manifest.initial_step_scale <= 0.0F ||
      manifest.convergence_iteration <= 0) {
    throw std::runtime_error("robot TrajOpt artifact metadata is invalid");
  }
  const auto dof = static_cast<size_t>(manifest.num_dof);
  const auto sphere_count = static_cast<size_t>(manifest.num_spheres);
  const auto opt_dim = checked_product(
      static_cast<size_t>(manifest.num_knots), dof, "TrajOpt action");
  for (const auto* tensor : {
           &manifest.position_bounds,
           &manifest.velocity_bounds,
           &manifest.acceleration_bounds,
           &manifest.jerk_bounds,
       }) {
    validate_tensor(*tensor, "float32", 2 * dof * sizeof(float),
                    "TrajOpt joint bounds");
  }
  for (const auto* tensor : {
           &manifest.state_bound_weight,
           &manifest.state_activation_distance,
           &manifest.state_regularization_weight,
       }) {
    validate_tensor(*tensor, "float32", 5 * sizeof(float),
                    "TrajOpt state weight");
  }
  validate_tensor(manifest.cspace_target_weight, "float32", sizeof(float),
                  "cspace_target_weight");
  validate_tensor(manifest.cspace_non_terminal_weight, "float32", sizeof(float),
                  "cspace_non_terminal_weight");
  validate_tensor(manifest.cspace_target_dof_weight, "float32",
                  dof * sizeof(float), "cspace_target_dof_weight");
  validate_tensor(manifest.tool_pose_weight, "float32", 2 * sizeof(float),
                  "tool_pose_weight");
  validate_tensor(manifest.world_collision_weight, "float32", sizeof(float),
                  "world_collision_weight");
  validate_tensor(manifest.self_collision_weight, "float32", sizeof(float),
                  "self_collision_weight");
  validate_tensor(manifest.metrics_world_collision_weight, "float32", sizeof(float),
                  "metrics_world_collision_weight");
  validate_tensor(manifest.metrics_self_collision_weight, "float32", sizeof(float),
                  "metrics_self_collision_weight");
  validate_tensor(manifest.validation_self_collision_weight, "float32", sizeof(float),
                  "validation_self_collision_weight");
  validate_tensor(manifest.line_search_scale, "float32",
                  static_cast<size_t>(manifest.num_line_search) * sizeof(float),
                  "line_search_scale");
  validate_tensor(manifest.optimization_step_max, "float32",
                  opt_dim * sizeof(float), "optimization_step_max");
  validate_tensor(manifest.sphere_collision_mask, "uint32",
                  static_cast<size_t>(manifest.num_spheres) * sizeof(uint32_t),
                  "sphere_collision_mask");
  validate_tensor(manifest.link_spheres, "float32",
                  sphere_count * 4 * sizeof(float), "link_spheres");
  validate_tensor(manifest.link_sphere_map, "int16",
                  sphere_count * sizeof(int16_t), "link_sphere_map");
  if (manifest.num_self_collision_pairs > 0) {
    validate_tensor(manifest.self_collision_padding, "float32",
                    sphere_count * sizeof(float), "self_collision_padding");
    validate_tensor(
        manifest.self_collision_pairs, "int16",
        static_cast<size_t>(manifest.num_self_collision_pairs) * 2 * sizeof(int16_t),
        "self_collision_pairs");
  }
  if (manifest.kinematics.source != "curobo_kinematics_kernel") {
    throw std::runtime_error("TrajOpt artifact has an invalid kinematics kernel");
  }
  for (const auto* kernel : {
           &manifest.kinematics_backward,
           &manifest.bspline_forward,
           &manifest.bspline_single_dt,
           &manifest.bspline_backward,
           &manifest.time_scaled_dt,
           &manifest.cspace,
           &manifest.pose,
           &manifest.initialize_actions,
           &manifest.prepare_pose_criteria,
           &manifest.expand_trajectory_dt,
           &manifest.reduce_required_dt,
           &manifest.calculate_interpolation_horizon,
           &manifest.validate_interpolated,
           &manifest.prepare_initial_state,
           &manifest.prepare_search_actions,
           &manifest.prepare_search_actions_backtracking,
           &manifest.combine_rollout,
           &manifest.accumulate,
           &manifest.finalize,
           &manifest.line_search,
           &manifest.line_search_backtracking,
           &manifest.lbfgs,
       }) {
    if (kernel->cubin.empty() || kernel->symbol.empty() ||
        kernel->shared_memory < 0 || kernel->block_dim <= 0) {
      throw std::runtime_error("TrajOpt artifact has invalid kernel metadata");
    }
  }
  return manifest;
}

RolloutManifest load_rollout_manifest(const std::string& artifact_dir) {
  const auto path =
      std::filesystem::path(artifact_dir) / "robot_rollout_manifest.json";
  const auto json = read_text(path);
  const auto tensors = extract_object(json, "tensors");
  const auto profiles = extract_object(json, "profiles");
  const auto load_profile = [&](const char* name) {
    const auto profile_json = extract_object(profiles, name);
    const auto kernels = extract_object(profile_json, "kernels");
    RolloutKernelProfile profile;
    profile.threads_per_batch =
        extract_int(profile_json, "threads_per_batch");
    profile.cache_floats_per_link =
        extract_int(profile_json, "cache_floats_per_link");
    profile.forward = extract_ik_kernel(kernels, "forward");
    profile.backward = extract_ik_kernel(kernels, "backward");
    return profile;
  };

  RolloutManifest manifest;
  manifest.schema_version = extract_int(json, "schema_version");
  manifest.arch = extract_string(json, "arch");
  manifest.num_dof = extract_int(json, "num_dof");
  manifest.num_links = extract_int(json, "num_links");
  manifest.num_original_links = extract_int(json, "num_original_links");
  manifest.num_levels = extract_int(json, "num_levels");
  manifest.level_link_indices = extract_int(json, "level_link_indices");
  manifest.fixed_transforms = extract_tensor(tensors, "fixed_transforms");
  manifest.link_masses_com = extract_tensor(tensors, "link_masses_com");
  manifest.link_inertias = extract_tensor(tensors, "link_inertias");
  manifest.joint_map_type = extract_tensor(tensors, "joint_map_type");
  manifest.joint_map = extract_tensor(tensors, "joint_map");
  manifest.link_map = extract_tensor(tensors, "link_map");
  manifest.chain_map = extract_tensor(tensors, "chain_map");
  manifest.joint_offset_map = extract_tensor(tensors, "joint_offset_map");
  manifest.gravity = extract_tensor(tensors, "gravity");
  manifest.level_starts = extract_tensor(tensors, "level_starts");
  manifest.level_links = extract_tensor(tensors, "level_links");
  manifest.original_link_masses_com =
      extract_tensor(tensors, "original_link_masses_com");
  manifest.original_link_inertias =
      extract_tensor(tensors, "original_link_inertias");
  manifest.external_force_owners =
      extract_tensor(tensors, "external_force_owners");
  manifest.external_force_transforms =
      extract_tensor(tensors, "external_force_transforms");
  manifest.effort_bounds = extract_tensor(tensors, "effort_bounds");
  manifest.ik = load_profile("ik");
  manifest.rrt = load_profile("rrt");
  manifest.trajopt = load_profile("trajopt");

  if (manifest.schema_version != 1 || manifest.arch.empty() ||
      manifest.num_dof <= 0 || manifest.num_links <= 0 ||
      manifest.num_original_links < manifest.num_links ||
      manifest.num_levels <= 0 || manifest.level_link_indices <= 0 ||
      manifest.level_link_indices % 2 != 0) {
    throw std::runtime_error("robot rollout artifact metadata is invalid");
  }
  const auto links = static_cast<size_t>(manifest.num_links);
  const auto original_links =
      static_cast<size_t>(manifest.num_original_links);
  const auto dof = static_cast<size_t>(manifest.num_dof);
  validate_tensor(manifest.fixed_transforms, "float32",
                  links * 12 * sizeof(float), "rollout fixed_transforms");
  validate_tensor(manifest.link_masses_com, "float32",
                  links * 4 * sizeof(float), "rollout link_masses_com");
  validate_tensor(manifest.link_inertias, "float32",
                  links * 8 * sizeof(float), "rollout link_inertias");
  validate_tensor(manifest.joint_map_type, "int8", links,
                  "rollout joint_map_type");
  for (const auto* tensor : {&manifest.joint_map, &manifest.link_map,
                             &manifest.chain_map}) {
    validate_tensor(*tensor, "int16", links * sizeof(int16_t),
                    "rollout link topology");
  }
  validate_tensor(manifest.joint_offset_map, "float32",
                  links * 2 * sizeof(float), "rollout joint_offset_map");
  validate_tensor(manifest.gravity, "float32", 6 * sizeof(float),
                  "rollout gravity");
  validate_tensor(manifest.level_starts, "int16",
                  static_cast<size_t>(manifest.num_levels + 1) *
                      sizeof(int16_t),
                  "rollout level_starts");
  validate_tensor(manifest.level_links, "int16",
                  static_cast<size_t>(manifest.level_link_indices) *
                      sizeof(int16_t),
                  "rollout level_links");
  validate_tensor(manifest.original_link_masses_com, "float32",
                  original_links * 4 * sizeof(float),
                  "rollout original_link_masses_com");
  validate_tensor(manifest.original_link_inertias, "float32",
                  original_links * 8 * sizeof(float),
                  "rollout original_link_inertias");
  validate_tensor(manifest.external_force_owners, "int16",
                  original_links * sizeof(int16_t),
                  "rollout external_force_owners");
  validate_tensor(manifest.external_force_transforms, "float32",
                  original_links * 12 * sizeof(float),
                  "rollout external_force_transforms");
  validate_tensor(manifest.effort_bounds, "float32",
                  2 * dof * sizeof(float), "rollout effort_bounds");

  for (const auto* profile : {&manifest.ik, &manifest.rrt,
                              &manifest.trajopt}) {
    if ((profile->threads_per_batch != 1 &&
         profile->threads_per_batch != 2 &&
         profile->threads_per_batch != 4 &&
         profile->threads_per_batch != 16 &&
         profile->threads_per_batch != 32) ||
        profile->cache_floats_per_link !=
            (profile->threads_per_batch > 1 ? 32 : 20) ||
        profile->forward.cubin.empty() || profile->forward.symbol.empty() ||
        profile->forward.source != "curobo_rnea_kernel" ||
        profile->forward.block_dim <= 0 ||
        profile->forward.block_dim % profile->threads_per_batch != 0 ||
        profile->backward.cubin.empty() || profile->backward.symbol.empty() ||
        profile->backward.source != "curobo_rnea_kernel" ||
        profile->backward.block_dim <= 0 ||
        profile->backward.block_dim % profile->threads_per_batch != 0) {
      throw std::runtime_error("robot rollout profile metadata is invalid");
    }
  }
  return manifest;
}

RrtManifest load_rrt_manifest(const std::string& artifact_dir) {
  const auto path = std::filesystem::path(artifact_dir) / "robot_rrt_manifest.json";
  const auto json = read_text(path);
  const auto tensors = extract_object(json, "tensors");
  const auto kernels = extract_object(json, "kernels");
  const auto capacities = extract_object(json, "scene_capacities");

  RrtManifest manifest;
  manifest.schema_version = extract_int(json, "schema_version");
  manifest.arch = extract_string(json, "arch");
  manifest.num_dof = extract_int(json, "num_dof");
  manifest.num_links = extract_int(json, "num_links");
  manifest.num_tool_frames = extract_int(json, "num_tool_frames");
  manifest.num_spheres = extract_int(json, "num_spheres");
  manifest.num_self_collision_pairs =
      extract_int(json, "num_self_collision_pairs");
  manifest.samples_per_iteration =
      extract_int(json, "samples_per_iteration");
  manifest.max_iterations = extract_int(json, "max_iterations");
  manifest.max_nodes_per_tree = extract_int(json, "max_nodes_per_tree");
  manifest.fixed_steer_steps = extract_int(json, "fixed_steer_steps");
  manifest.direct_steer_steps = extract_int(json, "direct_steer_steps");
  manifest.max_path_nodes = extract_int(json, "max_path_nodes");
  manifest.sample_buffer_size = extract_int(json, "sample_buffer_size");
  manifest.max_batched_lanes = extract_int(json, "max_batched_lanes");
  manifest.shortcut_path_capacity =
      extract_int(json, "shortcut_path_capacity");
  manifest.shortcut_state_capacity =
      extract_int(json, "shortcut_state_capacity");
  manifest.shortcut_edge_capacity =
      extract_int(json, "shortcut_edge_capacity");
  manifest.cspace_resolution = extract_float(json, "cspace_resolution");
  manifest.max_extension_distance =
      extract_float(json, "max_extension_distance");
  manifest.similarity_threshold =
      extract_float(json, "similarity_threshold");
  manifest.max_cuboids = extract_int(capacities, "cuboid");
  manifest.max_capsules = extract_int(capacities, "capsule");
  manifest.max_cylinders = extract_int(capacities, "cylinder");
  manifest.max_meshes = extract_int(capacities, "mesh");
  manifest.max_voxel_grids = extract_int(capacities, "voxel");

  manifest.position_lower = extract_tensor(tensors, "position_lower");
  manifest.position_upper = extract_tensor(tensors, "position_upper");
  manifest.distance_weight = extract_tensor(tensors, "distance_weight");
  manifest.samples = extract_tensor(tensors, "samples");
  manifest.shortcut_pair_sources =
      extract_tensor(tensors, "shortcut_pair_sources");
  manifest.shortcut_pair_destinations =
      extract_tensor(tensors, "shortcut_pair_destinations");
  manifest.shortcut_edge_lookup =
      extract_tensor(tensors, "shortcut_edge_lookup");
  manifest.link_spheres = extract_tensor(tensors, "link_spheres");
  manifest.sphere_collision_mask =
      extract_tensor(tensors, "sphere_collision_mask");
  manifest.link_sphere_map = extract_tensor(tensors, "link_sphere_map");
  if (manifest.num_self_collision_pairs > 0) {
    manifest.self_collision_padding =
        extract_tensor(tensors, "self_collision_padding");
    manifest.self_collision_pairs =
        extract_tensor(tensors, "self_collision_pairs");
  }

  manifest.kinematics = extract_kinematics_kernel(kernels, "kinematics");
  manifest.collision_cuboid = extract_ik_kernel(kernels, "collision_cuboid");
  manifest.collision_capsule = extract_ik_kernel(kernels, "collision_capsule");
  manifest.collision_cylinder =
      extract_ik_kernel(kernels, "collision_cylinder");
  manifest.collision_mesh = extract_ik_kernel(kernels, "collision_mesh");
  manifest.collision_voxel = extract_ik_kernel(kernels, "collision_voxel");
  manifest.self_collision = extract_ik_kernel(kernels, "self_collision");
  manifest.nearest_target = extract_ik_kernel(kernels, "nearest_target");
  manifest.packed_append = extract_ik_kernel(kernels, "packed_append");
  manifest.reconstruct_path = extract_ik_kernel(kernels, "reconstruct_path");
  manifest.prepare_samples = extract_ik_kernel(kernels, "prepare_samples");
  manifest.steer_edges = extract_ik_kernel(kernels, "steer_edges");
  manifest.evaluate_edges = extract_ik_kernel(kernels, "evaluate_edges");
  manifest.prepare_direct_edges =
      extract_ik_kernel(kernels, "prepare_direct_edges");
  manifest.interpolate_direct_edges =
      extract_ik_kernel(kernels, "interpolate_direct_edges");
  manifest.reduce_direct_edges =
      extract_ik_kernel(kernels, "reduce_direct_edges");
  manifest.classify_states = extract_ik_kernel(kernels, "classify_states");
  manifest.select_connection =
      extract_ik_kernel(kernels, "select_connection");
  manifest.shortcut_setup = extract_ik_kernel(kernels, "shortcut_setup");
  manifest.shortcut_fill_edges =
      extract_ik_kernel(kernels, "shortcut_fill_edges");
  manifest.shortcut_reduce_edges =
      extract_ik_kernel(kernels, "shortcut_reduce_edges");
  manifest.shortcut_select_path =
      extract_ik_kernel(kernels, "shortcut_select_path");

  if (manifest.schema_version != 8 || manifest.num_dof <= 0 ||
      manifest.num_links <= 0 || manifest.num_tool_frames <= 0 ||
      manifest.num_spheres <= 0 || manifest.num_self_collision_pairs < 0 ||
      manifest.samples_per_iteration <= 0 ||
      manifest.samples_per_iteration > 256 || manifest.max_iterations <= 0 ||
      manifest.max_nodes_per_tree <= 0 || manifest.fixed_steer_steps < 3 ||
      manifest.direct_steer_steps < 2 || manifest.max_path_nodes < 2 ||
      manifest.sample_buffer_size <= 0 || manifest.max_batched_lanes <= 0 ||
      manifest.shortcut_path_capacity < 3 ||
      manifest.shortcut_state_capacity < 2 ||
      manifest.shortcut_edge_capacity <= 0) {
    throw std::runtime_error("robot RRT artifact metadata is invalid");
  }
  const auto required_nodes =
      static_cast<size_t>(2) +
      checked_product(static_cast<size_t>(manifest.samples_per_iteration),
                      static_cast<size_t>(manifest.max_iterations),
                      "RRT required nodes");
  const auto expected_shortcut_edges =
      static_cast<size_t>(manifest.shortcut_path_capacity - 1) *
      static_cast<size_t>(manifest.shortcut_path_capacity - 2) / 2;
  const auto sample_storage_size = std::max(
      static_cast<size_t>(manifest.sample_buffer_size),
      checked_product(
          checked_product(static_cast<size_t>(2),
                          static_cast<size_t>(manifest.max_batched_lanes),
                          "RRT sample storage lanes"),
          static_cast<size_t>(manifest.samples_per_iteration),
          "RRT sample storage"));
  if (required_nodes > static_cast<size_t>(manifest.max_nodes_per_tree) ||
      expected_shortcut_edges !=
          static_cast<size_t>(manifest.shortcut_edge_capacity) ||
      manifest.max_path_nodes < 2 * (manifest.max_iterations + 2) ||
      manifest.cspace_resolution <= 0.0F ||
      manifest.max_extension_distance <= 0.0F ||
      manifest.max_extension_distance >
          (manifest.fixed_steer_steps - 2) * manifest.cspace_resolution ||
      manifest.similarity_threshold <= 0.0F || manifest.max_cuboids < 0 ||
      manifest.max_capsules < 0 || manifest.max_cylinders < 0 ||
      manifest.max_meshes < 0 || manifest.max_voxel_grids < 0 ||
      manifest.prepare_samples.block_dim <= 0 ||
      manifest.steer_edges.block_dim < manifest.fixed_steer_steps ||
      manifest.steer_edges.block_dim < manifest.num_dof ||
      manifest.evaluate_edges.block_dim < manifest.fixed_steer_steps ||
      manifest.evaluate_edges.block_dim < manifest.num_dof ||
      manifest.prepare_direct_edges.block_dim <= 0 ||
      manifest.interpolate_direct_edges.block_dim <= 0 ||
      manifest.reduce_direct_edges.block_dim <= 0 ||
      manifest.nearest_target.block_dim <= 0 ||
      manifest.max_nodes_per_tree % manifest.nearest_target.block_dim != 0) {
    throw std::runtime_error("robot RRT artifact metadata is invalid");
  }

  const auto dof = static_cast<size_t>(manifest.num_dof);
  const auto spheres = static_cast<size_t>(manifest.num_spheres);
  validate_tensor(manifest.position_lower, "float32", dof * sizeof(float),
                  "RRT position_lower");
  validate_tensor(manifest.position_upper, "float32", dof * sizeof(float),
                  "RRT position_upper");
  validate_tensor(manifest.distance_weight, "float32", dof * sizeof(float),
                  "RRT distance_weight");
  validate_tensor(
      manifest.samples, "float32",
      checked_product(
          sample_storage_size,
          dof * sizeof(float), "RRT samples"),
      "RRT samples");
  validate_tensor(manifest.shortcut_pair_sources, "int32",
                  expected_shortcut_edges * sizeof(int32_t),
                  "RRT shortcut_pair_sources");
  validate_tensor(manifest.shortcut_pair_destinations, "int32",
                  expected_shortcut_edges * sizeof(int32_t),
                  "RRT shortcut_pair_destinations");
  validate_tensor(
      manifest.shortcut_edge_lookup, "int32",
      static_cast<size_t>(manifest.shortcut_path_capacity) *
          static_cast<size_t>(manifest.shortcut_path_capacity) *
          sizeof(int32_t),
      "RRT shortcut_edge_lookup");
  validate_tensor(manifest.sphere_collision_mask, "uint32",
                  static_cast<size_t>(manifest.num_spheres) * sizeof(uint32_t),
                  "sphere_collision_mask");
  validate_tensor(manifest.link_spheres, "float32",
                  spheres * 4 * sizeof(float), "RRT link_spheres");
  validate_tensor(manifest.link_sphere_map, "int16",
                  spheres * sizeof(int16_t), "RRT link_sphere_map");
  if (manifest.num_self_collision_pairs > 0) {
    validate_tensor(manifest.self_collision_padding, "float32",
                    spheres * sizeof(float), "RRT self_collision_padding");
    validate_tensor(
        manifest.self_collision_pairs, "int16",
        static_cast<size_t>(manifest.num_self_collision_pairs) * 2 *
            sizeof(int16_t),
        "RRT self_collision_pairs");
  }
  if (manifest.kinematics.source != "curobo_kinematics_kernel") {
    throw std::runtime_error("RRT artifact has an invalid kinematics kernel");
  }
  for (const auto* kernel : {
           &manifest.nearest_target,
           &manifest.packed_append,
           &manifest.reconstruct_path,
           &manifest.prepare_samples,
           &manifest.steer_edges,
           &manifest.evaluate_edges,
           &manifest.prepare_direct_edges,
           &manifest.interpolate_direct_edges,
           &manifest.reduce_direct_edges,
           &manifest.classify_states,
           &manifest.select_connection,
           &manifest.shortcut_setup,
           &manifest.shortcut_fill_edges,
           &manifest.shortcut_reduce_edges,
           &manifest.shortcut_select_path,
       }) {
    if (kernel->cubin.empty() || kernel->symbol.empty() ||
        kernel->source != "curobo_python_warp_kernel" ||
        kernel->shared_memory < 0 || kernel->block_dim <= 0) {
      throw std::runtime_error("RRT artifact has invalid Warp kernel metadata");
    }
  }
  if (manifest.num_self_collision_pairs > 0 &&
      (manifest.self_collision.cubin.empty() ||
       manifest.self_collision.symbol.empty() ||
       manifest.self_collision.source != "curobo_self_collision_kernel" ||
       manifest.self_collision.block_dim <= 0)) {
    throw std::runtime_error("RRT artifact has invalid self-collision metadata");
  }
  return manifest;
}

TimeOptManifest load_timeopt_manifest(const std::string &artifact_dir, int knots,
                                        bool reference) {
  const auto filename = "timeopt_" + std::to_string(knots) + "_" +
                        std::to_string(static_cast<int>(reference)) + ".json";
  const auto json = read_text(std::filesystem::path(artifact_dir) / filename);
  if (extract_int(json, "schema_version") != 3) {
    throw std::runtime_error("unsupported timeopt table schema; re-export timeopt tables");
  }
  TimeOptManifest manifest;
  manifest.num_dof = extract_int(json, "num_dof");
  manifest.num_knots = extract_int(json, "num_knots");
  manifest.reference = extract_int(json, "reference");
  manifest.intervals = extract_int(json, "intervals");
  manifest.points = extract_int(json, "points");
  manifest.tiles = extract_int(json, "tiles");
  const auto tensors = extract_object(json, "tensors");
  for (const auto *name :
       {"basis", "point_first", "meta", "offsets", "starts",
        "time_basis", "time_first", "time_weights", "segments", "local", "polynomial", "nodes"}) {
    manifest.tensors.emplace(name, extract_tensor(tensors, name));
  }
  return manifest;
}

MotionPlanningManifest load_motion_planning_manifest(
    const std::string& artifact_dir) {
  const auto path =
      std::filesystem::path(artifact_dir) / "robot_motion_manifest.json";
  const auto json = read_text(path);
  const auto tensors = extract_object(json, "tensors");
  const auto kernels = extract_object(json, "kernels");

  MotionPlanningManifest manifest;
  manifest.schema_version = extract_int(json, "schema_version");
  manifest.arch = extract_string(json, "arch");
  manifest.fit = extract_string(json, "fit");
  manifest.num_dof = extract_int(json, "num_dof");
  manifest.num_knots = extract_int(json, "num_knots");
  manifest.trajectory_steps = extract_int(json, "trajectory_steps");
  manifest.interpolation_steps = extract_int(json, "interpolation_steps");
  manifest.max_rrt_path_nodes = extract_int(json, "max_rrt_path_nodes");
  manifest.max_trajopt_seeds = extract_int(json, "max_trajopt_seeds");
  manifest.grasp_contact_link_names =
      extract_string_array(json, "grasp_contact_link_names");
  manifest.irls_iterations = extract_int(json, "irls_iterations");
  manifest.regularization = extract_float(json, "regularization");
  manifest.timing_regularization =
      extract_float(json, "timing_regularization");
  manifest.trust_region_regularization =
      extract_float(json, "trust_region_regularization");
  manifest.irls_epsilon = extract_float(json, "irls_epsilon");
  manifest.dt_epsilon = extract_float(json, "dt_epsilon");
  manifest.trajectory_dt = extract_float(json, "trajectory_dt");
  manifest.path_normal_trace = extract_float(json, "path_normal_trace");
  manifest.path_diagonal_scale =
      extract_float(json, "path_diagonal_scale");
  manifest.arc_progress = extract_tensor(tensors, "arc_progress");
  manifest.fit_matrix = extract_tensor(tensors, "fit_matrix");
  manifest.start_coefficient = extract_tensor(tensors, "start_coefficient");
  manifest.start_velocity_coefficient =
      extract_tensor(tensors, "start_velocity_coefficient");
  manifest.start_acceleration_coefficient =
      extract_tensor(tensors, "start_acceleration_coefficient");
  manifest.start_jerk_coefficient =
      extract_tensor(tensors, "start_jerk_coefficient");
  manifest.goal_coefficient = extract_tensor(tensors, "goal_coefficient");
  manifest.basis_position = extract_tensor(tensors, "basis_position");
  manifest.basis_velocity = extract_tensor(tensors, "basis_velocity");
  manifest.basis_acceleration =
      extract_tensor(tensors, "basis_acceleration");
  manifest.basis_jerk = extract_tensor(tensors, "basis_jerk");
  manifest.path_normal = extract_tensor(tensors, "path_normal");
  manifest.max_velocity = extract_tensor(tensors, "max_velocity");
  manifest.max_acceleration = extract_tensor(tensors, "max_acceleration");
  manifest.max_jerk = extract_tensor(tensors, "max_jerk");
  manifest.resample_path = extract_ik_kernel(kernels, "resample_path");
  manifest.prepare_seed_paths =
      extract_ik_kernel(kernels, "prepare_seed_paths");
  if (manifest.schema_version <= 8) {
    manifest.resample_dense = extract_ik_kernel(kernels, "resample_dense");
    manifest.initialize_fit = extract_ik_kernel(kernels, "initialize_fit");
    manifest.accept_candidate =
        extract_ik_kernel(kernels, "accept_candidate");
  }

  if (manifest.schema_version != 10 ||
      manifest.arch.empty() ||
      manifest.fit != "bspline_time_optimal_irls" ||
      manifest.num_dof <= 0 || manifest.num_knots <= 5 ||
      manifest.trajectory_steps <= 5 || manifest.interpolation_steps <= 0 ||
      manifest.max_rrt_path_nodes < 2 || manifest.max_trajopt_seeds <= 0 ||
      manifest.irls_iterations <= 0 || manifest.regularization < 0.0F ||
      manifest.timing_regularization < 0.0F ||
      manifest.trust_region_regularization < 0.0F ||
      manifest.irls_epsilon <= 0.0F || manifest.dt_epsilon < 0.0F ||
      manifest.trajectory_dt <= 0.0F || manifest.path_normal_trace <= 0.0F ||
      manifest.path_diagonal_scale <= 0.0F) {
    throw std::runtime_error("robot motion-planning artifact metadata is invalid");
  }
  const auto free_knots = static_cast<size_t>(manifest.num_knots - 1);
  const auto trajectory_steps =
      static_cast<size_t>(manifest.trajectory_steps);
  validate_tensor(manifest.arc_progress, "float32",
                  trajectory_steps * sizeof(float), "motion arc_progress");
  validate_tensor(
      manifest.fit_matrix, "float32",
      checked_product(free_knots, trajectory_steps, "motion fit matrix") *
          sizeof(float),
      "motion fit_matrix");
  validate_tensor(manifest.start_coefficient, "float32",
                  free_knots * sizeof(float),
                  "motion start_coefficient");
  validate_tensor(manifest.start_velocity_coefficient, "float32",
                  free_knots * sizeof(float),
                  "motion start_velocity_coefficient");
  validate_tensor(manifest.start_acceleration_coefficient, "float32",
                  free_knots * sizeof(float),
                  "motion start_acceleration_coefficient");
  validate_tensor(manifest.start_jerk_coefficient, "float32",
                  free_knots * sizeof(float),
                  "motion start_jerk_coefficient");
  validate_tensor(manifest.goal_coefficient, "float32",
                  free_knots * sizeof(float), "motion goal_coefficient");
  for (const auto* tensor : {&manifest.basis_position,
                             &manifest.basis_velocity,
                             &manifest.basis_acceleration,
                             &manifest.basis_jerk}) {
    validate_tensor(*tensor, "float32",
                    checked_product(trajectory_steps, free_knots,
                                    "motion B-spline basis") *
                        sizeof(float),
                    "motion B-spline basis");
  }
  validate_tensor(manifest.path_normal, "float32",
                  checked_product(free_knots, free_knots,
                                  "motion path normal") *
                      sizeof(float),
                  "motion path_normal");
  for (const auto* tensor : {&manifest.max_velocity,
                             &manifest.max_acceleration,
                             &manifest.max_jerk}) {
    validate_tensor(*tensor, "float32",
                    static_cast<size_t>(manifest.num_dof) * sizeof(float),
                    "motion derivative limit");
  }
  std::vector<const IkKernelManifest*> required_kernels{
      &manifest.resample_path,
      &manifest.prepare_seed_paths,
  };
  if (manifest.schema_version <= 8) {
    required_kernels.insert(required_kernels.end(),
                            {&manifest.resample_dense,
                             &manifest.initialize_fit,
                             &manifest.accept_candidate});
  }
  for (const auto* kernel : required_kernels) {
    if (kernel->cubin.empty() || kernel->symbol.empty() ||
        kernel->source != "curobo_python_warp_kernel" ||
        kernel->shared_memory < 0 || kernel->block_dim <= 0) {
      throw std::runtime_error(
          "motion-planning artifact has invalid Warp kernel metadata");
    }
  }
  return manifest;
}

std::vector<uint8_t> load_tensor_data(const std::string& artifact_dir,
                                      const TensorManifest& tensor,
                                      const std::string& name) {
  return read_tensor(std::filesystem::path(artifact_dir), tensor, name);
}

DeviceBuffer::~DeviceBuffer() { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : pointer_(other.pointer_), bytes_(other.bytes_) {
  other.pointer_ = nullptr;
  other.bytes_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    pointer_ = other.pointer_;
    bytes_ = other.bytes_;
    other.pointer_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

void DeviceBuffer::allocate(size_t bytes) {
  release();
  CRR_CUDA_CHECK(cudaMalloc(&pointer_, std::max<size_t>(bytes, 1)));
  bytes_ = bytes;
}

void DeviceBuffer::upload(const std::vector<uint8_t>& data) {
  allocate(data.size());
  if (!data.empty()) {
    CRR_CUDA_CHECK(
        cudaMemcpy(pointer_, data.data(), data.size(), cudaMemcpyHostToDevice));
  }
}

void DeviceBuffer::clear(cudaStream_t stream) const {
  if (bytes_ != 0) {
    CRR_CUDA_CHECK(cudaMemsetAsync(pointer_, 0, bytes_, stream));
  }
}

void DeviceBuffer::release() {
  if (pointer_ != nullptr) {
    cudaFree(pointer_);
    pointer_ = nullptr;
  }
  bytes_ = 0;
}

KernelModule::~KernelModule() {
  function_ = nullptr;
  if (module_ != nullptr) {
    cuModuleUnload(module_);
    module_ = nullptr;
  }
}

void KernelModule::load(const std::string& cubin_path, const std::string& symbol) {
  if (module_ != nullptr) {
    throw std::runtime_error("kernel module is already loaded");
  }
  CRR_CU_CHECK(cuModuleLoad(&module_, cubin_path.c_str()));
  CRR_CU_CHECK(cuModuleGetFunction(&function_, module_, symbol.c_str()));
}

void KernelModule::set_max_dynamic_shared_memory(int bytes) const {
  if (function_ == nullptr) {
    throw std::runtime_error("kernel is not loaded");
  }
  if (bytes < 0) {
    throw std::invalid_argument("dynamic shared memory must be non-negative");
  }
  CRR_CU_CHECK(cuFuncSetAttribute(
      function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes));
}

void KernelModule::launch(unsigned int grid_x, unsigned int block_x,
                          unsigned int shared_memory, CUstream stream,
                          const std::vector<void*>& arguments) const {
  if (function_ == nullptr) {
    throw std::runtime_error("kernel is not loaded");
  }
  CRR_CU_CHECK(cuLaunchKernel(function_, grid_x, 1, 1, block_x, 1, 1, shared_memory,
                              stream, const_cast<void**>(arguments.data()), nullptr));
}

RobotRuntimeImpl::RobotRuntimeImpl(std::string artifact_dir, int device_ordinal)
    : artifact_dir_(std::move(artifact_dir)), device_ordinal_(device_ordinal) {
  CRR_CU_CHECK(cuInit(0));
  CRR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  manifest_ = load_manifest(artifact_dir_);
  validate_device_architecture();
  load_model();
  const auto cubin_path =
      (std::filesystem::path(artifact_dir_) / manifest_.kernel.cubin).string();
  kernel_.load(cubin_path, manifest_.kernel.symbol);
}

RobotRuntimeImpl::~RobotRuntimeImpl() {
  // Members release CUDA resources after this body executes.
  cudaSetDevice(device_ordinal_);
}

void RobotRuntimeImpl::validate_device_architecture() const {
  cudaDeviceProp properties{};
  CRR_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_ordinal_));
  const auto device_arch =
      "sm_" + std::to_string(properties.major) + std::to_string(properties.minor);
  if (manifest_.arch != device_arch) {
    throw std::runtime_error("robot artifact targets " + manifest_.arch +
                             ", but selected device is " + device_arch);
  }
}

void RobotRuntimeImpl::load_model() {
  const auto directory = std::filesystem::path(artifact_dir_);
  auto fixed_transforms = read_tensor(directory, manifest_.fixed_transforms,
                                      "fixed_transforms");
  auto joint_map_type =
      read_tensor(directory, manifest_.joint_map_type, "joint_map_type");
  auto joint_map = read_tensor(directory, manifest_.joint_map, "joint_map");
  auto link_map = read_tensor(directory, manifest_.link_map, "link_map");
  auto tool_frame_map =
      read_tensor(directory, manifest_.tool_frame_map, "tool_frame_map");
  auto joint_offset_map =
      read_tensor(directory, manifest_.joint_offset_map, "joint_offset_map");
  auto tool_joint_links_data = read_tensor(
      directory, manifest_.tool_joint_links_data, "tool_joint_links_data");
  auto tool_joint_links_offsets = read_tensor(
      directory, manifest_.tool_joint_links_offsets, "tool_joint_links_offsets");
  validate_tool_offsets(tool_joint_links_offsets,
                        tool_joint_links_data.size() / sizeof(int16_t));

  fixed_transforms_.upload(fixed_transforms);
  joint_map_type_.upload(joint_map_type);
  joint_map_.upload(joint_map);
  link_map_.upload(link_map);
  tool_frame_map_.upload(tool_frame_map);
  joint_offset_map_.upload(joint_offset_map);
  tool_joint_links_data_.upload(tool_joint_links_data);
  tool_joint_links_offsets_.upload(tool_joint_links_offsets);
  if (!manifest_.joint_link_influence_words.file.empty()) {
    joint_link_influence_words_.upload(read_tensor(
        directory, manifest_.joint_link_influence_words,
        "joint_link_influence_words"));
    joint_links_data_.upload(
        read_tensor(directory, manifest_.joint_links_data, "joint_links_data"));
    joint_links_offsets_.upload(read_tensor(
        directory, manifest_.joint_links_offsets, "joint_links_offsets"));
    link_masses_com_.upload(
        read_tensor(directory, manifest_.link_masses_com, "link_masses_com"));
  }
}

CuroboRobotInfo RobotRuntimeImpl::info() const {
  return CuroboRobotInfo{
      manifest_.num_links,
      manifest_.num_dof,
      manifest_.num_tool_frames,
  };
}

const std::string& RobotRuntimeImpl::joint_name(int index) const {
  if (index < 0 || index >= manifest_.num_dof) {
    throw std::out_of_range("joint index is out of range");
  }
  return manifest_.joint_names.at(static_cast<size_t>(index));
}

const std::string& RobotRuntimeImpl::tool_frame_name(int index) const {
  if (index < 0 || index >= manifest_.num_tool_frames) {
    throw std::out_of_range("tool-frame index is out of range");
  }
  return manifest_.tool_frames.at(static_cast<size_t>(index));
}

RobotRuntimeImpl::ModelDeviceView RobotRuntimeImpl::model_device_view() const {
  return ModelDeviceView{
      fixed_transforms_.ptr(),
      joint_map_type_.ptr(),
      joint_map_.ptr(),
      link_map_.ptr(),
      tool_frame_map_.ptr(),
      joint_offset_map_.ptr(),
      tool_joint_links_data_.ptr(),
      tool_joint_links_offsets_.ptr(),
      joint_link_influence_words_.ptr(),
      joint_links_data_.ptr(),
      joint_links_offsets_.ptr(),
      link_masses_com_.ptr(),
  };
}

void RobotRuntimeImpl::compute_kinematics(
    const CuroboRobotKinematicsRequest& request) const {
  CRR_CUDA_CHECK(cudaSetDevice(device_ordinal_));
  if (request.joint_positions == 0 || request.link_positions == 0 ||
      request.link_quaternions == 0 || request.link_jacobians == 0) {
    throw std::invalid_argument("kinematics CUDA tensor addresses must be nonzero");
  }
  if (request.batch_size <= 0 || request.horizon <= 0) {
    throw std::invalid_argument("batch_size and horizon must be positive");
  }
  const int64_t flattened_batch =
      static_cast<int64_t>(request.batch_size) * static_cast<int64_t>(request.horizon);
  if (flattened_batch > INT_MAX) {
    throw std::invalid_argument("batch_size * horizon exceeds the kernel integer range");
  }

  const auto& policy = manifest_.kernel;
  const int64_t shared_memory_per_batch_value =
      static_cast<int64_t>(manifest_.num_links) * 12 * sizeof(float) * 2;
  if (shared_memory_per_batch_value > policy.max_dynamic_shared_memory) {
    throw std::runtime_error("robot requires more dynamic shared memory than the artifact allows");
  }
  const int shared_memory_per_batch = static_cast<int>(shared_memory_per_batch_value);
  const int thread_limited_batches =
      policy.max_threads_per_block / policy.threads_per_batch;
  const int shared_memory_limited_batches =
      policy.max_dynamic_shared_memory / shared_memory_per_batch;
  const int batches_per_block = std::min(
      {static_cast<int>(flattened_batch), policy.max_batches_per_block,
       thread_limited_batches, shared_memory_limited_batches});
  if (batches_per_block <= 0) {
    throw std::runtime_error("kinematics launch policy produced an empty block");
  }

  const auto block_x =
      static_cast<unsigned int>(batches_per_block * policy.threads_per_batch);
  const auto grid_x = static_cast<unsigned int>(
      (flattened_batch + batches_per_block - 1) / batches_per_block);
  const auto shared_memory =
      static_cast<unsigned int>(batches_per_block * shared_memory_per_batch);

  CUdeviceptr link_positions = request.link_positions;
  CUdeviceptr link_quaternions = request.link_quaternions;
  CUdeviceptr null_pointer = 0;
  CUdeviceptr link_jacobians = request.link_jacobians;
  CUdeviceptr joint_positions = request.joint_positions;
  CUdeviceptr fixed_transforms = fixed_transforms_.ptr();
  CUdeviceptr joint_map_type = joint_map_type_.ptr();
  CUdeviceptr joint_map = joint_map_.ptr();
  CUdeviceptr link_map = link_map_.ptr();
  CUdeviceptr tool_frame_map = tool_frame_map_.ptr();
  CUdeviceptr tool_joint_links_data = tool_joint_links_data_.ptr();
  CUdeviceptr tool_joint_links_offsets = tool_joint_links_offsets_.ptr();
  CUdeviceptr joint_offset_map = joint_offset_map_.ptr();
  int kernel_batch_size = static_cast<int>(flattened_batch);
  int horizon = request.horizon;
  int num_spheres = 0;
  int num_environments = 1;
  int num_links = manifest_.num_links;
  int num_dof = manifest_.num_dof;
  int num_tool_frames = manifest_.num_tool_frames;

  std::vector<void*> arguments = {
      &link_positions,
      &link_quaternions,
      &null_pointer,
      &null_pointer,
      &link_jacobians,
      &null_pointer,
      &joint_positions,
      &fixed_transforms,
      &null_pointer,
      &null_pointer,
      &joint_map_type,
      &joint_map,
      &link_map,
      &tool_frame_map,
      &null_pointer,
      &tool_joint_links_data,
      &tool_joint_links_offsets,
      &joint_offset_map,
      &null_pointer,
      &kernel_batch_size,
      &horizon,
      &num_spheres,
      &num_environments,
      &num_links,
      &num_dof,
      &num_tool_frames,
  };
  const auto stream = reinterpret_cast<CUstream>(static_cast<uintptr_t>(request.stream));
  kernel_.launch(grid_x, block_x, shared_memory, stream, arguments);
}

}  // namespace curobo_robot_runtime
