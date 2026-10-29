# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH
      "CUDA compiler")
endif()

project(curobo_robot_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

set(WARP_NATIVE_INCLUDE_DIR "" CACHE PATH
    "Path to Warp's native headers; enables mesh acceleration support")

add_library(curobo_robot_runtime SHARED
  src/attachment_runtime.cpp
  src/c_api.cpp
  src/ik_runtime.cpp
  src/inverse_kinematics.cpp
  src/kinematics.cpp
  src/motion_planner.cpp
  src/motion_planner_cuda.cu
  src/motion_planner_runtime.cpp
  src/mpc.cpp
  src/mpc_cuda.cu
  src/robot_runtime.cpp
  src/rollout_runtime.cpp
  src/rollout_streams.cpp
  src/rrt.cpp
  src/rrt_runtime.cpp
  src/scene.cpp
  src/grasp_collision_runtime.cpp
  src/scene_runtime.cpp
  src/sphere_fit.cpp
  src/trajectory_optimization.cpp
  src/trajectory_seed_fitter_cuda.cu
  src/trajectory_optimizer_runtime.cpp
  src/trajectory_seed_fitter.cpp
  src/trajectory_seed_fitter_runtime.cpp
  src/trajectory_timeopt.cpp
  src/trajectory_timeopt_cuda.cu
)

set_source_files_properties(src/trajectory_seed_fitter_cuda.cu src/trajectory_timeopt_cuda.cu PROPERTIES
  COMPILE_OPTIONS "--ftz=true;--fmad=true;--prec-div=false;--prec-sqrt=false"
)

target_compile_definitions(curobo_robot_runtime
  PRIVATE CUROBO_ROBOT_RUNTIME_BUILD
)

target_include_directories(curobo_robot_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../curobo/_src/curobolib/kernels
          ${CMAKE_CURRENT_SOURCE_DIR}/../../curobo/_src/curobolib/kernels/trajectory
          ${CMAKE_CURRENT_SOURCE_DIR}/../../curobo/_src/curobolib/kernels/common
)

target_link_libraries(curobo_robot_runtime
  PUBLIC
    CUDA::cudart
  PRIVATE
    CUDA::cuda_driver
)

if(WARP_NATIVE_INCLUDE_DIR)
  if(NOT EXISTS "${WARP_NATIVE_INCLUDE_DIR}/warp.h" OR
     NOT EXISTS "${WARP_NATIVE_INCLUDE_DIR}/version.h")
    message(FATAL_ERROR
      "WARP_NATIVE_INCLUDE_DIR must contain warp.h and version.h")
  endif()
  target_include_directories(curobo_robot_runtime SYSTEM PRIVATE ${WARP_NATIVE_INCLUDE_DIR})
  target_sources(curobo_robot_runtime PRIVATE src/warp_mesh_backend.cpp)
  target_compile_definitions(curobo_robot_runtime PRIVATE CUROBO_ROBOT_HAS_WARP_MESH=1)
  target_link_libraries(curobo_robot_runtime PRIVATE ${CMAKE_DL_LIBS})
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(curobo_robot_timeopt_output_test tests/timeopt_output_test.cpp)
  target_include_directories(curobo_robot_timeopt_output_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_timeopt_output_test PRIVATE curobo_robot_runtime)
  add_test(NAME curobo_robot_timeopt_output_test COMMAND curobo_robot_timeopt_output_test)
  add_executable(curobo_robot_timeopt_test tests/trajectory_timeopt_test.cpp)
  target_include_directories(curobo_robot_timeopt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_timeopt_test PRIVATE curobo_robot_runtime CUDA::cuda_driver)
  add_executable(curobo_robot_mpc_test tests/mpc_test.cpp)
  target_include_directories(curobo_robot_mpc_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_mpc_test PRIVATE curobo_robot_runtime)
  add_executable(curobo_robot_mpc_shift_test tests/mpc_shift_test.cpp)
  target_include_directories(curobo_robot_mpc_shift_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_mpc_shift_test PRIVATE curobo_robot_runtime)
  add_test(NAME curobo_robot_mpc_shift_test COMMAND curobo_robot_mpc_shift_test)
  add_executable(curobo_robot_mpc_continuation_test tests/mpc_continuation_test.cu)
  target_include_directories(curobo_robot_mpc_continuation_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../curobo/_src/curobolib/kernels)
  target_link_libraries(curobo_robot_mpc_continuation_test PRIVATE curobo_robot_runtime)
  add_test(NAME curobo_robot_mpc_continuation_test COMMAND curobo_robot_mpc_continuation_test)
  add_executable(curobo_robot_grasp_collision_policy_test tests/grasp_collision_policy_test.cpp)
  target_include_directories(curobo_robot_grasp_collision_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_grasp_collision_policy_test PRIVATE curobo_robot_runtime)
  add_test(NAME curobo_robot_grasp_collision_policy_test COMMAND curobo_robot_grasp_collision_policy_test)
  add_executable(curobo_robot_collision_layers_test tests/collision_layers_test.cpp)
  target_include_directories(curobo_robot_collision_layers_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_collision_layers_test PRIVATE curobo_robot_runtime CUDA::cuda_driver)

  add_executable(curobo_robot_grasp_pose_preparation_test tests/grasp_pose_preparation_test.cpp)
  target_include_directories(curobo_robot_grasp_pose_preparation_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_grasp_pose_preparation_test PRIVATE curobo_robot_runtime)
  add_test(NAME curobo_robot_grasp_pose_preparation_test COMMAND curobo_robot_grasp_pose_preparation_test)

  add_executable(curobo_robot_ik_shared_model_test tests/ik_shared_model_test.cpp)
  target_include_directories(curobo_robot_ik_shared_model_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_ik_shared_model_test PRIVATE curobo_robot_runtime CUDA::cuda_driver)

  add_executable(curobo_robot_ik_parallelism_test tests/ik_parallelism_test.cpp)
  target_include_directories(curobo_robot_ik_parallelism_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_ik_parallelism_test PRIVATE curobo_robot_runtime)

  add_executable(curobo_robot_trajopt_parallelism_test tests/trajopt_parallelism_test.cpp)
  target_link_libraries(curobo_robot_trajopt_parallelism_test PRIVATE curobo_robot_runtime)

  add_executable(curobo_robot_ik_test tests/ik_test.cpp)
  target_include_directories(curobo_robot_ik_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_ik_test PRIVATE curobo_robot_runtime)

  add_executable(curobo_robot_scene_test tests/scene_test.cpp)
  target_include_directories(curobo_robot_scene_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_scene_test PRIVATE curobo_robot_runtime)
  add_test(NAME curobo_robot_scene_test COMMAND curobo_robot_scene_test)

  add_executable(curobo_robot_sphere_fit_test tests/sphere_fit_test.cpp)
  target_include_directories(curobo_robot_sphere_fit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(curobo_robot_sphere_fit_test PRIVATE curobo_robot_runtime)
  add_test(NAME curobo_robot_sphere_fit_test COMMAND curobo_robot_sphere_fit_test)

  add_executable(curobo_robot_trajopt_test tests/trajopt_test.cpp)
  target_link_libraries(curobo_robot_trajopt_test PRIVATE curobo_robot_runtime)

  add_executable(curobo_robot_rrt_test tests/rrt_test.cpp)
  target_link_libraries(curobo_robot_rrt_test PRIVATE curobo_robot_runtime)

  add_executable(curobo_robot_motion_test tests/motion_planner_test.cpp)
  target_link_libraries(curobo_robot_motion_test PRIVATE curobo_robot_runtime)

  add_executable(curobo_robot_planner_ik_model_test tests/planner_ik_model_test.cpp)
  target_include_directories(curobo_robot_planner_ik_model_test PRIVATE src)
  target_link_libraries(curobo_robot_planner_ik_model_test PRIVATE curobo_robot_runtime)

  add_executable(curobo_robot_seed_fitter_test tests/trajectory_seed_fitter_test.cpp)
  target_link_libraries(curobo_robot_seed_fitter_test PRIVATE curobo_robot_runtime)
endif()
