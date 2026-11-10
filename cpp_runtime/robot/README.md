<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Robot C++ runtime

Load an exported robot and run cuRobo from C++17 with CUDA. Python is used to prepare
artifacts; the deployed application loads precompiled kernels and robot data.

Follow the [runtime setup guide](../README.md) to export a robot and build the libraries.
The commands below run from the repository root.

## Available APIs

Public headers are in [`include/curobo_robot_runtime/`](include/curobo_robot_runtime/).
The namespace is `curobo::robot`.

| API | Purpose | Contract |
| --- | --- | --- |
| `Kinematics` | Fused FK and geometric Jacobians | [FK](REFERENCE.md#api-contract) |
| `InverseKinematics` | Multi-seed, goalset and batched IK | [IK](REFERENCE.md#inverse-kinematics-api) |
| `TrajectoryOptimizer` | Joint-space and Cartesian trajectory optimization | [TrajOpt](REFERENCE.md#trajectory-optimization-api) |
| `RrtPlanner` | Bounded bidirectional RRT | [RRT](REFERENCE.md#rrt-api) |
| `TrajectorySeedFitter`, `TrajectoryTimeOpt` | B-spline fitting and trajectory retiming | [Fitting](REFERENCE.md#trajectory-seed-fitter-api), [retiming](REFERENCE.md#timeopt-executable-trajectories) |
| `MotionPlanner` | Joint, pose, waypoint and grasp planning | [Planning](REFERENCE.md#motion-planning-api) |
| `MpcController` | Joint-space and Cartesian MPC | [MPC](REFERENCE.md#joint-space-and-cartesian-mpc) |
| `Scene`, sphere fitting and attachments | Fixed-capacity obstacles and grasped objects | [Scene](REFERENCE.md#scene-api), [attachments](REFERENCE.md#sphere-fitting-and-attached-objects) |
| `PoseCalibration` | Global/seeded ICP and mesh-distance refinement | [Calibration](REFERENCE.md#pose-calibration); link `curobo_calibration_runtime` |

The [C ABI](include/curobo_robot_runtime/api.h) exposes FK and Jacobians. Planning APIs use
the C++ interfaces. Calibration is a separate library and takes posed triangles and observed
points from the application.

## Run a minimal C++ application

After exporting `cpp_runtime/robot/artifacts/franka`, save this as `app/main.cpp`:

```cpp
#include <curobo_robot_runtime/kinematics.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

using DeviceBuffer = std::unique_ptr<float, decltype(&cudaFree)>;

DeviceBuffer allocate_floats(std::size_t count) {
  float* pointer = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)));
  return DeviceBuffer(pointer, &cudaFree);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: curobo_fk ARTIFACT_DIRECTORY\n";
    return 1;
  }
  try {
    curobo::robot::Kinematics kinematics(argv[1]);
    const auto info = kinematics.info();
    auto joints = allocate_floats(info.num_dof);
    auto positions = allocate_floats(info.num_tool_frames * 3);
    auto quaternions = allocate_floats(info.num_tool_frames * 4);
    auto jacobians = allocate_floats(info.num_tool_frames * 6 * info.num_dof);
    check_cuda(cudaMemset(joints.get(), 0, info.num_dof * sizeof(float)));

    // Reuse this instance and these buffers for subsequent FK calls.
    kinematics.forward(joints.get(), positions.get(), quaternions.get(),
                       jacobians.get(), 1, 1);
    check_cuda(cudaStreamSynchronize(nullptr));
    std::vector<float> xyz(info.num_tool_frames * 3);
    check_cuda(cudaMemcpy(xyz.data(), positions.get(), xyz.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::cout << "First tool xyz: " << xyz[0] << " " << xyz[1] << " " << xyz[2] << "\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
```

This computes FK at zero joint positions; it does not plan or execute a motion.
The runtime allocates model data once. The application owns its CUDA input/output buffers.

Save this as `app/CMakeLists.txt`. `CUROBO_SOURCE_DIR` is the absolute checkout path:

```cmake
cmake_minimum_required(VERSION 3.22)
project(curobo_fk_example LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 12.3 REQUIRED)
set(BUILD_TESTING OFF CACHE BOOL "Build runtime tests")
add_subdirectory("${CUROBO_SOURCE_DIR}/cpp_runtime/robot" curobo_robot)
add_executable(curobo_fk main.cpp)
target_link_libraries(curobo_fk PRIVATE curobo_robot_runtime)
```

Build and run with the same `CUDA_ARCH` selected in the setup guide:

```bash
cmake -S app -B cpp_runtime/build/app \
  -DCUROBO_SOURCE_DIR="$PWD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build cpp_runtime/build/app --target curobo_fk -j8
cpp_runtime/build/app/curobo_fk cpp_runtime/robot/artifacts/franka
```

`add_subdirectory` propagates the public include path and CUDA runtime dependency.
Applications using mesh obstacles or calibration must also configure the Warp native backend
as described in the [setup guide](../README.md#build-the-runtime).

## Add planning to an application

1. Allocate a `Scene` with capacities covering the largest expected obstacle set. Populate it
   before constructing the planner and keep it alive while solvers refer to it.
2. Construct one `MotionPlanner` with the validated Python-equivalent settings and matching
   artifact. Its default configuration enables internal CUDA graphs.
3. Allocate contiguous CUDA buffers from `robot_info()` and `info()`. Supply targets through
   `PoseMotionPlanningRequest`, `WaypointMotionPlanningRequest`, or `GraspPlanningRequest`.
4. Warm representative requests, including direct and RRT fallback paths, before measurement.
   There is no separate native `MotionPlanner::warmup()` method; call the normal planning APIs.
5. Reuse the scene, planner, requests, and buffers. Check success and use only the valid returned
   horizon. Borrowed candidate and time-optimal trajectory views belong to the planner and may
   be overwritten by the next solve; copy them if they must outlive that call.

The [planning reference](REFERENCE.md#motion-planning-api) gives request fields and examples.
`plan()`, `plan_pose()`, and grasp orchestration observe GPU status on the host and are
synchronous. They internally replay captured solver stages; they are not a single externally
capturable CUDA graph. FK is asynchronous on the supplied stream.

## Verify the deployment

Build tests by configuring the robot runtime with `-DBUILD_TESTING=ON`, then run:

```bash
cmake -S cpp_runtime/robot -B cpp_runtime/build/robot \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DBUILD_TESTING=ON
cmake --build cpp_runtime/build/robot -j8
ctest --test-dir cpp_runtime/build/robot --output-on-failure
cpp_runtime/build/robot/curobo_robot_ik_test cpp_runtime/robot/artifacts/franka
cpp_runtime/build/robot/curobo_robot_motion_test cpp_runtime/robot/artifacts/franka
```

CTest covers registered native checks. Artifact-backed solver executables are additional GPU
checks; the [test reference](REFERENCE.md#test) lists parity-test environment variables and
individual executables. Calibration's GPU test needs separately exported calibration kernels.
Use the [benchmark runner](../benchmarks/README.md) after quality checks pass.
