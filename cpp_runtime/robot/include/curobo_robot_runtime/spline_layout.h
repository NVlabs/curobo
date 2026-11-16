// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace curobo::robot {

/** Fixed spline geometry exported by the Python configuration.
 * Query native_horizon when allocating native state or Cartesian reference
 * buffers. Padding and reference_intervals describe the stationary endpoints.
 */
struct SplineLayout {
  int num_knots{0};
  int degree{0};
  int interpolation_steps{0};
  int native_horizon{0};
  int padding{0};
  int reference_intervals{0};
};
}  // namespace curobo::robot
