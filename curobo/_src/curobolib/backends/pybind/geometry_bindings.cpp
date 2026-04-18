/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <map>
#include <torch/extension.h>

#include <vector>

namespace curobo{
  namespace geometry{
  namespace self_collision {

// CUDA forward declarations
void self_collision_distance(
  torch::Tensor out_distance,
  torch::Tensor out_vec,
  torch::Tensor pair_distance, // batch_size x num_collision_pairs
  torch::Tensor sparse_index,
  const torch::Tensor robot_spheres,    // batch_size x n_spheres x 3
  const torch::Tensor sphere_padding, // n_spheres
  const torch::Tensor weight,
  const torch::Tensor pair_locations,
  torch::Tensor block_batch_max_value,
  torch::Tensor block_batch_max_index,
  const int num_blocks_per_batch,
  const int max_threads_per_block,
  const int batch_size,
  const int horizon,
  const int nspheres,
  const int num_collision_pairs,
  const bool store_pair_distance,
  const bool compute_grad);

}
}
}
// C++ interface



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

  m.def("self_collision_distance", &curobo::geometry::self_collision::self_collision_distance);
}
