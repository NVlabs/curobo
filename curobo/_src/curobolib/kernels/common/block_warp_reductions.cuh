/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once
#include "curobo_constants.h"


namespace curobo{
  namespace common{

// Enum for reduction operation types
enum class ReductionOp {
    SUM,
    MAX
};

// Operation functors
template<ReductionOp Op>
struct ReductionOperator;

template<>
struct ReductionOperator<ReductionOp::SUM> {
    template<typename T>
    __device__ __forceinline__ static T apply(const T& a, const T& b) {
        return a + b;
    }

    template<typename T>
    __device__ __forceinline__ static T identity() {
        return T(0);
    }
};

template<>
struct ReductionOperator<ReductionOp::MAX> {
    template<typename T>
    __device__ __forceinline__ static T apply(const T& a, const T& b) {
        return (a > b) ? a : b;
    }

    template<typename T>
    __device__ __forceinline__ static T identity() {
        return T(CUDART_NINF_F);
    }
};

// Templated warp reduction with operation support
template<typename ValueType, ReductionOp Op = ReductionOp::SUM>
__forceinline__ __device__ ValueType warp_reduce(ValueType v, const int num_elements)
{
    const unsigned active_mask = __activemask();
    ValueType val = v;
    const uint8_t lane_idx = threadIdx.x % curobo::common::warpSize;
    const uint8_t valid_lanes = static_cast<uint8_t>(
      num_elements <= 0 ? 0 :
      (num_elements < curobo::common::warpSize ? num_elements : curobo::common::warpSize));
    const unsigned mask = __ballot_sync(active_mask, lane_idx < valid_lanes);
    if (lane_idx >= valid_lanes)
    {
        return val;
    }

    #pragma unroll
    for (uint8_t offset = curobo::common::warpSize / 2; offset > 0; offset /= 2) {
        const ValueType other_value = __shfl_down_sync(mask, val, offset);
        const uint8_t source_lane = lane_idx + offset;
        if (source_lane < valid_lanes && ((1u << source_lane) & mask)) {
            val = ReductionOperator<Op>::apply(val, other_value);
        }
    }

    return val;
}

// Templated block reduction with operation support
template<typename ValueType, ReductionOp Op = ReductionOp::SUM>
__forceinline__ __device__ void block_reduce(ValueType v,
                                          int num_elements,
                                          ValueType *data,
                                          ValueType *result)
{
    const int warp_start = (threadIdx.x / curobo::common::warpSize) * curobo::common::warpSize;
    const int remaining_elements = num_elements - warp_start;
    const int warp_elements = remaining_elements > curobo::common::warpSize ?
      curobo::common::warpSize : (remaining_elements > 0 ? remaining_elements : 0);
    ValueType val = warp_reduce<ValueType, Op>(v, warp_elements);
    const uint8_t leader = 0;

    if (num_elements <= curobo::common::warpSize) // only one warp is active
    {
        if (threadIdx.x == leader)
        {
            result[0] = val;
        }
        __syncthreads(); // syncwarp might be sufficient here.
        return;
    }

    if (threadIdx.x % 32 == leader)
    {
        data[(threadIdx.x) / 32] = val; // store reduced value per warp
    }

    __syncthreads();

    const uint32_t elems = (num_elements + 31) / 32; // number of warps for the given dimension that have a value

    ValueType val2 = (threadIdx.x < elems) ? data[threadIdx.x] : ReductionOperator<Op>::template identity<ValueType>();
    val2 = warp_reduce<ValueType, Op>(val2, elems);

    if (threadIdx.x == leader)
    {
        result[0] = val2;
    }

    __syncthreads();
}

// Templated segmented block reduction with operation support
template<typename ValueType, ReductionOp Op = ReductionOp::SUM>
__forceinline__ __device__ void segmented_block_reduce(
                                          ValueType register_value,
                                          int num_elements, // total number of elements per segment
                                          ValueType *data, // at least n_segments * warps_per_segment
                                          ValueType *result,// at least n_segments
                                          const int local_thread_idx,
                                          const int segment_idx)
{
    ValueType val = warp_reduce<ValueType, Op>(
      register_value,
      num_elements < curobo::common::warpSize ? num_elements : curobo::common::warpSize);
    const uint8_t leader = 0;
    const int warps_per_segment = (num_elements + 31) / 32; // Need to compute this correctly.

    if (num_elements < 32) // only one warp is active
    {
        if (local_thread_idx == leader)
        {
            result[segment_idx] = val;
        }
        __syncthreads(); // syncwarp might be sufficient here.
        return;
    }

    if (local_thread_idx % 32 == leader)
    {
        data[segment_idx * warps_per_segment + ((local_thread_idx) / 32)] = val; // store reduced value per warp
    }

    __syncthreads();

    ValueType val2 = (local_thread_idx < warps_per_segment) ?
                  data[segment_idx * warps_per_segment + (local_thread_idx)] :
                  ReductionOperator<Op>::template identity<ValueType>();

    val2 = warp_reduce<ValueType, Op>(val2, warps_per_segment);

    if (local_thread_idx == leader)
    {
        result[segment_idx] = val2;
    }

    __syncthreads();
}

// create templated functions:

template<typename ValueType>
__forceinline__ __device__ void block_reduce_sum(ValueType v, int num_elements, ValueType *data, ValueType *result)
{
    block_reduce<ValueType, ReductionOp::SUM>(v, num_elements, data, result);
}

template<typename ValueType>
__forceinline__ __device__ void block_reduce_max(ValueType v, int num_elements, ValueType *data, ValueType *result)
{
    block_reduce<ValueType, ReductionOp::MAX>(v, num_elements, data, result);
}

// warp reductions:

template<typename ValueType>
__forceinline__ __device__ ValueType warp_reduce_sum(ValueType v, int num_elements)
{
    return warp_reduce<ValueType, ReductionOp::SUM>(v, num_elements);
}

template<typename ValueType>
__forceinline__ __device__ ValueType warp_reduce_max(ValueType v, int num_elements)
{
    return warp_reduce<ValueType, ReductionOp::MAX>(v, num_elements);
}

// segmented reductions:

template<typename ValueType>
__forceinline__ __device__  void segmented_block_reduce_sum(ValueType v, int num_elements, ValueType *data, ValueType *result, int local_thread_idx, int segment_idx)
{
    segmented_block_reduce<ValueType, ReductionOp::SUM>(v, num_elements, data, result, local_thread_idx, segment_idx);
}


template<typename ValueType>
__forceinline__ __device__ void segmented_block_reduce_max(ValueType v, int num_elements, ValueType *data, ValueType *result, int local_thread_idx, int segment_idx)
{
    segmented_block_reduce<ValueType, ReductionOp::MAX>(v, num_elements, data, result, local_thread_idx, segment_idx);
}

}
}
