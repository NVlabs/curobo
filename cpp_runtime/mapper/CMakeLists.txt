# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

project(curobo_mapper_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
include(CTest)

set(WARP_NATIVE_INCLUDE_DIR "" CACHE PATH "Path to warp/native headers")
if(NOT WARP_NATIVE_INCLUDE_DIR)
  message(FATAL_ERROR "Set -DWARP_NATIVE_INCLUDE_DIR=/path/to/site-packages/warp/native")
endif()

add_library(curobo_mapper_runtime SHARED
  src/c_api.cpp
  src/cubin_kernel.cpp
  src/device_scan.cu
  src/mapper_runtime.cpp
)

target_include_directories(curobo_mapper_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  SYSTEM PRIVATE
    ${WARP_NATIVE_INCLUDE_DIR}
)

target_link_libraries(curobo_mapper_runtime
  PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)
