/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


namespace curobo{
  namespace common{
    template<typename T>
    __forceinline__ __device__ void compute_augmented_langrange_multiplier(
      float distance,
      T gradient,
      float constraint_weight, float constraint_sq_weight,
      float max_value,
      float &scaled_distance,
      T &scaled_gradient,
      float &scaled_constraint_weight)
    {

      float multiplier = fmaxf(0.0f, constraint_weight + constraint_sq_weight * distance);
      if (distance <= 0.0f)
      {
        distance = 0.0f;
        gradient *= 0.0f;
        multiplier = 0.0f;
      }
      scaled_constraint_weight = fminf(max_value, multiplier);

      // Cost: λv + 0.5μv²
      scaled_distance = (constraint_weight + 0.5f * constraint_sq_weight * distance) * distance;

      // Gradient: (λ + μv) * gradient
      scaled_gradient = gradient * fmaxf(0.0f, constraint_weight + constraint_sq_weight * distance);

    }
}
}