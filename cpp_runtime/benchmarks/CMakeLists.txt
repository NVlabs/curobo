# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(curobo_benchmarks LANGUAGES CXX)

set(BUILD_TESTING OFF CACHE BOOL "Build runtime tests")
add_subdirectory(../robot robot)
add_executable(curobo_benchmark benchmark.cpp)
target_compile_features(curobo_benchmark PRIVATE cxx_std_17)
target_link_libraries(curobo_benchmark PRIVATE curobo_robot_runtime)
