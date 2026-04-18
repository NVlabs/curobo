/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


namespace curobo{
    namespace geometry{


    struct CollisionPair { // 64 bits. This could be optimized to store with 32 bits?
      float   d; // 32 bits
      int16_t i; // 16 bits
      int16_t j; // 16 bits

      // Default constructor
      __host__ __device__ CollisionPair() : d(0.0f), i(0), j(0) {}

      // Constructor with parameters
      __host__ __device__ CollisionPair(float distance, int16_t idx_i, int16_t idx_j) : d(distance), i(idx_i), j(idx_j) {}

      explicit __host__ __device__ CollisionPair(float value): d(value), i(0), j(0) {}



      // Comparison operator - compares based on distance value

      __host__ __device__ bool operator>(const CollisionPair& other) const {
        return d > other.d;
      }

      __host__ __device__ bool operator<(const CollisionPair& other) const {
        return d < other.d;
      }

      __host__ __device__ bool operator>=(const CollisionPair& other) const {
        return d >= other.d;
      }

      __host__ __device__ bool operator<=(const CollisionPair& other) const {
        return d <= other.d;
      }

      __host__ __device__ bool operator==(const CollisionPair& other) const {
        return d == other.d;
      }

      __host__ __device__ bool operator!=(const CollisionPair& other) const {
        return d != other.d;
      }

      //  max function for CollisionPair
     friend __host__ __device__ __forceinline__ CollisionPair max(const CollisionPair& a, const CollisionPair& b) {
      return a.d > b.d ? a : b;
    }

    //  min function for CollisionPair
    friend __host__ __device__ __forceinline__ CollisionPair min(const CollisionPair& a, const CollisionPair& b) {
      return a.d < b.d ? a : b;
    }
    };

    // Helper function to create CollisionPair instances
    __host__ __device__ __forceinline__ CollisionPair make_collision_pair(float distance = 0.0f, int16_t idx_i = 0, int16_t idx_j = 0) {
      CollisionPair result;
      result.d = distance;
      result.i = idx_i;
      result.j = idx_j;
      return result;
    }



    // Convert CollisionPair to uint64_t (nvcc-safe version using union)
    __host__ __device__ __forceinline__ uint64_t reinterpret_as_uint64(const CollisionPair& pair) {
      union {
        CollisionPair as_pair;
        uint64_t as_uint64;
      } converter = {pair};
      return converter.as_uint64;
    }

    // Convert uint64_t to CollisionPair (nvcc-safe version using union)
    __host__ __device__ __forceinline__ CollisionPair reinterpret_as_collision_pair(uint64_t value) {
      union {
        uint64_t as_uint64;
        CollisionPair as_pair;
      } converter = {value};
      return converter.as_pair;
    }


    __device__ CollisionPair __shfl_down_sync(const unsigned mask, const CollisionPair &a, const int &offset)
    {
        // convert collision pair to uint64_t

        uint64_t a_uint64 = reinterpret_as_uint64(a);
        uint64_t other_value =  ::__shfl_down_sync(mask,  a_uint64, offset);

        return reinterpret_as_collision_pair(other_value);
    }



  }
}