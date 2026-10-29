// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "rollout_runtime.h"

#include "cuda_check.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace curobo::robot::detail {
namespace {

unsigned int grid_for(int count, int block_size) {
  return static_cast<unsigned int>((count + block_size - 1) / block_size);
}

template <typename ValueType>
std::vector<ValueType> decode(const std::vector<uint8_t>& bytes,
                              const char* name) {
  if (bytes.size() % sizeof(ValueType) != 0) {
    throw std::runtime_error(std::string(name) + " byte size is invalid");
  }
  std::vector<ValueType> result(bytes.size() / sizeof(ValueType));
  if (!bytes.empty()) {
    std::memcpy(result.data(), bytes.data(), bytes.size());
  }
  return result;
}

void load_kernel(const std::string& artifact_directory,
                 const curobo_robot_runtime::IkKernelManifest& manifest,
                 curobo_robot_runtime::KernelModule* kernel) {
  kernel->load(
      (std::filesystem::path(artifact_directory) / manifest.cubin).string(),
      manifest.symbol);
  if (manifest.shared_memory > 48000) {
    kernel->set_max_dynamic_shared_memory(manifest.shared_memory);
  }
}

using Matrix6 = std::array<double, 36>;

double& at(Matrix6* matrix, int row, int column) {
  return matrix->at(static_cast<size_t>(row * 6 + column));
}

double at(const Matrix6& matrix, int row, int column) {
  return matrix.at(static_cast<size_t>(row * 6 + column));
}

Matrix6 multiply(const Matrix6& left, const Matrix6& right) {
  Matrix6 result{};
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      for (int inner = 0; inner < 6; ++inner) {
        at(&result, row, column) +=
            at(left, row, inner) * at(right, inner, column);
      }
    }
  }
  return result;
}

Matrix6 transpose(const Matrix6& matrix) {
  Matrix6 result{};
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      at(&result, row, column) = at(matrix, column, row);
    }
  }
  return result;
}

Matrix6 spatial_inertia(const float* mass_com, const float* inertia) {
  const double x = mass_com[0];
  const double y = mass_com[1];
  const double z = mass_com[2];
  const double mass = mass_com[3];
  const std::array<double, 9> skew{0.0, -z, y, z, 0.0, -x, -y, x, 0.0};
  Matrix6 result{};
  const std::array<double, 9> center_inertia{
      inertia[0], inertia[3], inertia[4], inertia[3], inertia[1],
      inertia[5], inertia[4], inertia[5], inertia[2]};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      double shifted = 0.0;
      for (int inner = 0; inner < 3; ++inner) {
        shifted += skew[static_cast<size_t>(inner * 3 + row)] *
                   skew[static_cast<size_t>(inner * 3 + column)];
      }
      at(&result, row, column) =
          center_inertia[static_cast<size_t>(row * 3 + column)] +
          mass * shifted;
      at(&result, row, column + 3) =
          mass * skew[static_cast<size_t>(row * 3 + column)];
      at(&result, row + 3, column) =
          mass * skew[static_cast<size_t>(column * 3 + row)];
      at(&result, row + 3, column + 3) = row == column ? mass : 0.0;
    }
  }
  return result;
}

Matrix6 motion_transform(const float* transform) {
  Matrix6 result{};
  std::array<double, 9> rotation_transpose{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      rotation_transpose[static_cast<size_t>(row * 3 + column)] =
          transform[column * 3 + row];
    }
  }
  const double x = transform[9];
  const double y = transform[10];
  const double z = transform[11];
  const std::array<double, 9> skew{0.0, -z, y, z, 0.0, -x, -y, x, 0.0};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const double rotation =
          rotation_transpose[static_cast<size_t>(row * 3 + column)];
      at(&result, row, column) = rotation;
      at(&result, row + 3, column + 3) = rotation;
      for (int inner = 0; inner < 3; ++inner) {
        at(&result, row + 3, column) -=
            rotation_transpose[static_cast<size_t>(row * 3 + inner)] *
            skew[static_cast<size_t>(inner * 3 + column)];
      }
    }
  }
  return result;
}

void pack_spatial_inertia(const Matrix6& spatial, float* mass_com,
                          float* inertia) {
  const double mass =
      (at(spatial, 3, 3) + at(spatial, 4, 4) + at(spatial, 5, 5)) / 3.0;
  std::array<double, 9> center_cross{};
  if (std::abs(mass) > 1.0E-12) {
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        center_cross[static_cast<size_t>(row * 3 + column)] =
            0.5 * (at(spatial, row, column + 3) -
                   at(spatial, column, row + 3)) /
            mass;
      }
    }
  }
  mass_com[0] = static_cast<float>(center_cross[7]);
  mass_com[1] = static_cast<float>(center_cross[2]);
  mass_com[2] = static_cast<float>(center_cross[3]);
  mass_com[3] = static_cast<float>(mass);
  std::array<double, 9> center_inertia{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      double shifted = 0.0;
      for (int inner = 0; inner < 3; ++inner) {
        shifted += center_cross[static_cast<size_t>(inner * 3 + row)] *
                   center_cross[static_cast<size_t>(inner * 3 + column)];
      }
      center_inertia[static_cast<size_t>(row * 3 + column)] =
          0.5 * (at(spatial, row, column) + at(spatial, column, row)) -
          mass * shifted;
    }
  }
  inertia[0] = static_cast<float>(center_inertia[0]);
  inertia[1] = static_cast<float>(center_inertia[4]);
  inertia[2] = static_cast<float>(center_inertia[8]);
  inertia[3] = static_cast<float>(center_inertia[1]);
  inertia[4] = static_cast<float>(center_inertia[2]);
  inertia[5] = static_cast<float>(center_inertia[5]);
  inertia[6] = 0.0F;
  inertia[7] = 0.0F;
}

bool finite_values(const float* values, size_t count) {
  return std::all_of(values, values + count,
                     [](float value) { return std::isfinite(value); });
}

}  // namespace

RolloutRuntime::RolloutRuntime(std::string artifact_directory,
                               int device_ordinal,
                               const curobo_robot_runtime::RobotManifest&
                                   robot_manifest,
                               RolloutProfile profile)
    : artifact_directory_(std::move(artifact_directory)),
      link_names_(robot_manifest.link_names),
      manifest_(
          curobo_robot_runtime::load_rollout_manifest(artifact_directory_)) {
  CRR_CUDA_CHECK(cudaSetDevice(device_ordinal));
  if (manifest_.arch != robot_manifest.arch ||
      manifest_.num_dof != robot_manifest.num_dof ||
      link_names_.size() != static_cast<size_t>(manifest_.num_original_links)) {
    throw std::runtime_error("rollout and robot link dimensions do not match");
  }
  switch (profile) {
    case RolloutProfile::IK:
      profile_ = manifest_.ik;
      break;
    case RolloutProfile::RRT:
      profile_ = manifest_.rrt;
      break;
    case RolloutProfile::TRAJOPT:
      profile_ = manifest_.trajopt;
      break;
  }
  const auto load = [&](const curobo_robot_runtime::TensorManifest& tensor,
                        const char* name,
                        curobo_robot_runtime::DeviceBuffer* destination) {
    destination->upload(curobo_robot_runtime::load_tensor_data(
        artifact_directory_, tensor, name));
  };
  load(manifest_.fixed_transforms, "rollout fixed_transforms",
       &fixed_transforms_);
  load(manifest_.link_masses_com, "rollout link_masses_com",
       &link_masses_com_);
  load(manifest_.link_inertias, "rollout link_inertias", &link_inertias_);
  load(manifest_.joint_map_type, "rollout joint_map_type", &joint_map_type_);
  load(manifest_.joint_map, "rollout joint_map", &joint_map_);
  load(manifest_.link_map, "rollout link_map", &link_map_);
  load(manifest_.chain_map, "rollout chain_map", &chain_map_);
  load(manifest_.joint_offset_map, "rollout joint_offset_map",
       &joint_offset_map_);
  load(manifest_.gravity, "rollout gravity", &gravity_);
  load(manifest_.level_starts, "rollout level_starts", &level_starts_);
  load(manifest_.level_links, "rollout level_links", &level_links_);
  load(manifest_.effort_bounds, "rollout effort_bounds", &effort_bounds_);
  original_masses_com_ = decode<float>(
      curobo_robot_runtime::load_tensor_data(
          artifact_directory_, manifest_.original_link_masses_com,
          "rollout original_link_masses_com"),
      "original_link_masses_com");
  original_inertias_ = decode<float>(
      curobo_robot_runtime::load_tensor_data(
          artifact_directory_, manifest_.original_link_inertias,
          "rollout original_link_inertias"),
      "original_link_inertias");
  external_force_owners_ = decode<int16_t>(
      curobo_robot_runtime::load_tensor_data(
          artifact_directory_, manifest_.external_force_owners,
          "rollout external_force_owners"),
      "external_force_owners");
  external_force_transforms_ = decode<float>(
      curobo_robot_runtime::load_tensor_data(
          artifact_directory_, manifest_.external_force_transforms,
          "rollout external_force_transforms"),
      "external_force_transforms");
  load_kernel(artifact_directory_, profile_.forward, &forward_kernel_);
  load_kernel(artifact_directory_, profile_.backward, &backward_kernel_);
}

void RolloutRuntime::launch_inverse_dynamics(
    CUdeviceptr position, CUdeviceptr velocity, CUdeviceptr acceleration,
    CUdeviceptr effort, CUdeviceptr forward_cache, int states,
    CUstream stream) const {
  if (states <= 0) {
    throw std::invalid_argument("inverse-dynamics state count must be positive");
  }
  const int batches_per_block =
      profile_.forward.block_dim / profile_.threads_per_batch;
  CUdeviceptr fixed_transforms = fixed_transforms_.ptr();
  CUdeviceptr link_masses_com = link_masses_com_.ptr();
  CUdeviceptr link_inertias = link_inertias_.ptr();
  CUdeviceptr joint_map_type = joint_map_type_.ptr();
  CUdeviceptr joint_map = joint_map_.ptr();
  CUdeviceptr link_map = link_map_.ptr();
  CUdeviceptr chain_map = chain_map_.ptr();
  CUdeviceptr joint_offset_map = joint_offset_map_.ptr();
  CUdeviceptr gravity = gravity_.ptr();
  CUdeviceptr level_starts = level_starts_.ptr();
  CUdeviceptr level_links = level_links_.ptr();
  CUdeviceptr null_pointer = 0;
  int num_levels = manifest_.num_levels;
  int num_links = manifest_.num_links;
  std::vector<void*> arguments{
      &effort,          &position,          &velocity,
      &acceleration,    &fixed_transforms,  &link_masses_com,
      &link_inertias,   &joint_map_type,    &joint_map,
      &link_map,        &chain_map,         &joint_offset_map,
      &gravity,         &level_starts,      &level_links,
      &forward_cache,   &null_pointer,      &null_pointer,
      &null_pointer,    &null_pointer,      &null_pointer,
      &states,          &num_levels,        &num_links,
  };
  forward_kernel_.launch(
      grid_for(states, batches_per_block),
      static_cast<unsigned int>(profile_.forward.block_dim),
      static_cast<unsigned int>(profile_.forward.shared_memory), stream,
      arguments);
}

void RolloutRuntime::launch_inverse_dynamics_backward(
    CUdeviceptr effort_gradient, CUdeviceptr position, CUdeviceptr velocity,
    CUdeviceptr forward_cache, CUdeviceptr gradient_position,
    CUdeviceptr gradient_velocity, CUdeviceptr gradient_acceleration,
    int states, CUstream stream) const {
  if (states <= 0) {
    throw std::invalid_argument("inverse-dynamics state count must be positive");
  }
  const int batches_per_block =
      profile_.backward.block_dim / profile_.threads_per_batch;
  CUdeviceptr fixed_transforms = fixed_transforms_.ptr();
  CUdeviceptr link_masses_com = link_masses_com_.ptr();
  CUdeviceptr link_inertias = link_inertias_.ptr();
  CUdeviceptr joint_map_type = joint_map_type_.ptr();
  CUdeviceptr joint_map = joint_map_.ptr();
  CUdeviceptr link_map = link_map_.ptr();
  CUdeviceptr chain_map = chain_map_.ptr();
  CUdeviceptr joint_offset_map = joint_offset_map_.ptr();
  CUdeviceptr gravity = gravity_.ptr();
  CUdeviceptr level_starts = level_starts_.ptr();
  CUdeviceptr level_links = level_links_.ptr();
  CUdeviceptr null_pointer = 0;
  int num_levels = manifest_.num_levels;
  int num_links = manifest_.num_links;
  std::vector<void*> arguments{
      &gradient_position, &gradient_velocity, &gradient_acceleration,
      &null_pointer,      &null_pointer,      &null_pointer,
      &null_pointer,      &effort_gradient,  &position,
      &velocity,          &fixed_transforms,  &link_masses_com,
      &link_inertias,     &joint_map_type,    &joint_map,
      &link_map,          &chain_map,         &joint_offset_map,
      &gravity,           &level_starts,      &level_links,
      &forward_cache,     &states,            &num_levels,
      &num_links,
  };
  backward_kernel_.launch(
      grid_for(states, batches_per_block),
      static_cast<unsigned int>(profile_.backward.block_dim),
      static_cast<unsigned int>(profile_.backward.shared_memory), stream,
      arguments);
}

void RolloutRuntime::update_link_inertial(
    const LinkInertialUpdate& update) {
  update_links_inertial({update});
}

void RolloutRuntime::update_links_inertial(
    const LinkInertialUpdates& updates) {
  if (updates.empty()) {
    throw std::invalid_argument("inertial update list cannot be empty");
  }
  auto new_masses_com = original_masses_com_;
  auto new_inertias = original_inertias_;
  for (const auto& update : updates) {
    if (!update.mass && !update.center_of_mass && !update.inertia) {
      throw std::invalid_argument(
          "an inertial update must provide mass, center_of_mass, or inertia");
    }
    const auto found =
        std::find(link_names_.begin(), link_names_.end(), update.link_name);
    if (found == link_names_.end()) {
      throw std::invalid_argument("unknown inertial link: " + update.link_name);
    }
    const auto index = static_cast<size_t>(found - link_names_.begin());
    if (update.mass) {
      if (!std::isfinite(*update.mass) || *update.mass < 0.0F) {
        throw std::invalid_argument("link mass must be finite and non-negative");
      }
      new_masses_com.at(index * 4 + 3) = *update.mass;
    }
    if (update.center_of_mass) {
      if (!finite_values(update.center_of_mass->data(), 3)) {
        throw std::invalid_argument("link center_of_mass must be finite");
      }
      std::copy(update.center_of_mass->begin(), update.center_of_mass->end(),
                new_masses_com.begin() + static_cast<ptrdiff_t>(index * 4));
    }
    if (update.inertia) {
      if (!finite_values(update.inertia->data(), 6)) {
        throw std::invalid_argument("link inertia must be finite");
      }
      std::copy(update.inertia->begin(), update.inertia->end(),
                new_inertias.begin() + static_cast<ptrdiff_t>(index * 8));
    }
  }

  std::vector<float> reduced_masses_com;
  std::vector<float> reduced_inertias;
  recompute_reduced_model(new_masses_com, new_inertias,
                          &reduced_masses_com, &reduced_inertias);
  CRR_CUDA_CHECK(cudaDeviceSynchronize());
  CRR_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(link_masses_com_.ptr()),
                            reduced_masses_com.data(),
                            reduced_masses_com.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  CRR_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(link_inertias_.ptr()),
                            reduced_inertias.data(),
                            reduced_inertias.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
  original_masses_com_ = std::move(new_masses_com);
  original_inertias_ = std::move(new_inertias);
}

void RolloutRuntime::recompute_reduced_model(
    const std::vector<float>& original_masses_com,
    const std::vector<float>& original_inertias,
    std::vector<float>* reduced_masses_com,
    std::vector<float>* reduced_inertias) const {
  std::vector<Matrix6> aggregate(static_cast<size_t>(manifest_.num_links));
  for (int link = 0; link < manifest_.num_original_links; ++link) {
    const int owner = external_force_owners_.at(static_cast<size_t>(link));
    if (owner < 0) {
      continue;
    }
    const auto inertia = spatial_inertia(
        original_masses_com.data() + static_cast<size_t>(link * 4),
        original_inertias.data() + static_cast<size_t>(link * 8));
    const auto transform = motion_transform(
        external_force_transforms_.data() + static_cast<size_t>(link * 12));
    const auto transformed =
        multiply(transpose(transform), multiply(inertia, transform));
    auto& destination = aggregate.at(static_cast<size_t>(owner));
    for (size_t element = 0; element < destination.size(); ++element) {
      destination[element] += transformed[element];
    }
  }
  reduced_masses_com->resize(static_cast<size_t>(manifest_.num_links * 4));
  reduced_inertias->resize(static_cast<size_t>(manifest_.num_links * 8));
  for (int link = 0; link < manifest_.num_links; ++link) {
    pack_spatial_inertia(
        aggregate.at(static_cast<size_t>(link)),
        reduced_masses_com->data() + static_cast<size_t>(link * 4),
        reduced_inertias->data() + static_cast<size_t>(link * 8));
  }
}

}  // namespace curobo::robot::detail
