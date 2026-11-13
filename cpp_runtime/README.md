<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# cuRobo C++ runtime

Deploy robot computation and volumetric mapping from C++ with precompiled CUDA kernels.
Prototype and validate the task in Python first, then export the qualified configuration.
See the [agent workflow](../AGENTS.md) for instance reuse, collision caches, CUDA graphs,
robot import, and locked-joint updates.

## What is available

| Component | Functionality | Entry point |
| --- | --- | --- |
| Robot computation | FK, Jacobians, multi-seed/goalset/batched IK | [Robot guide](robot/README.md) |
| Motion generation | TrajOpt, bidirectional RRT, spline fitting, time-optimal retiming, joint/pose/waypoint/grasp planning, joint/Cartesian MPC | [Robot APIs](robot/README.md#available-apis) |
| Collision geometry | Single-scene obstacles, primitives/meshes/ESDFs, sphere fitting, attachments, payload inertials | [Scene and attachments](robot/REFERENCE.md#scene-api) |
| Pose calibration | Global or seeded ICP and mesh-distance refinement | [Separate calibration API](robot/include/curobo_robot_runtime/pose_calibration.h) |
| Mapping | RGB-D TSDF, ESDF, occupied voxels/points, features/instances, semantic queries, raycasts, meshes/textures, camera pose refinement | [Mapper guide](mapper/README.md) |
| Benchmarks | Reproducible native IK and motion-planning workloads | [Benchmark guide](benchmarks/README.md) |

Robot APIs use `curobo::robot`; FK also has a C ABI. The mapper exposes a C ABI.
Python's feature set is broader: native mapper LiDAR integration and static obstacle stamping
are not currently exposed. A basic URDF export supplies FK/Jacobians and IK; full collision-aware
motion generation requires a cuRobo robot configuration with spheres and limits.
Calibration is built as `curobo_calibration_runtime`, separately from the robot library.

## Prepare the environment

Run commands from the repository root in an installed cuRobo environment with a working CUDA
GPU and export dependencies. Follow the [installation guide](../docs/getting-started/installation.rst).
Building needs CMake 3.22+, a C++17 compiler, and a CUDA toolkit containing `nvcc`.
The robot runtime requires CUDA Toolkit 12.3 or newer. The toolkit must support the target GPU.

```bash
source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
CUDA_ARCH="$(python -c 'import torch; print("%d%d" % torch.cuda.get_device_capability())')"
WARP_ROOT="$(python -c 'from pathlib import Path; import warp; print(Path(warp.__file__).parent)')"
```

These variables select the local GPU and Warp headers. For another GPU or ARM64 Jetson,
follow [cross-device export and deployment](../docs/guides/export_cpp_runtime.rst).

## Export artifacts

Export a full Franka artifact for the active GPU:

```bash
python cpp_runtime/robot/tools/export_robot.py \
  --robot-config curobo/content/configs/robot/franka.yml \
  --tool-frame panda_hand \
  --out cpp_runtime/robot/artifacts/franka
```

Use the robot YAML and task settings that passed Python quality checks. Configure seed, goalset,
obstacle, attachment, and interpolation capacities for the application before export. For
waypoint workloads, set `--num-knots N` and use the same count in the native planner configuration.
See the [export reference](robot/REFERENCE.md#export-a-robot) for options.

Export a small mapper configuration:

```bash
python cpp_runtime/mapper/tools/export_mapper_aot.py \
  --config cpp_runtime/mapper/configs/small_mapper.json \
  --out cpp_runtime/mapper/artifacts/small_mapper
```

First-time mapper export compiles geometry, feature, semantic, and pose-refinement kernels
and can take many minutes. The exporter reports its kernel count and compilation phases.

Keep manifests, model data, and GPU kernels together. Re-export after incompatible source/schema,
model, layout, or GPU changes. The C++ process loads these files; it does not run Python or
compile robot kernels at startup. Artifacts and builds are ignored by Git.
Set `CUROBO_SOURCE_DIR` only when the exporter should intentionally use another checkout.

## Build the runtime

Build Release libraries for the same GPU architecture as the artifacts:

```bash
cmake -S cpp_runtime/robot -B cpp_runtime/build/robot \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DBUILD_TESTING=OFF -DWARP_NATIVE_INCLUDE_DIR="${WARP_ROOT}/native"
cmake --build cpp_runtime/build/robot -j8

cmake -S cpp_runtime/mapper -B cpp_runtime/build/mapper \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DBUILD_TESTING=OFF -DWARP_NATIVE_INCLUDE_DIR="${WARP_ROOT}/native"
cmake --build cpp_runtime/build/mapper -j8
```

The outputs are `libcurobo_robot_runtime.so`, `libcurobo_calibration_runtime.so`, and
`libcurobo_mapper_runtime.so` in their build directories. The robot build can omit
`WARP_NATIVE_INCLUDE_DIR` for primitive/voxel scenes; triangle-mesh collision and calibration
need the native mesh backend. The mapper build always needs the Warp headers.

For robot mesh queries and calibration, provide the matching Warp native library:

```bash
export CUROBO_WARP_LIBRARY="${WARP_ROOT}/bin/warp.so"
```

Deploy the native libraries, exported artifacts, and compatible CUDA libraries. For mesh-enabled
applications, also deploy matching `warp.so` and set `CUROBO_WARP_LIBRARY` to its deployed path.
The target application does not need Python, PyTorch, or the Python Warp package.
Build host libraries for the deployment CPU architecture. Preserve the executable's RPATH
layout or add the deployed library directory to `LD_LIBRARY_PATH`.

## Use the libraries

Start with the [complete FK application and CMake example](robot/README.md#run-a-minimal-c-application).
Then use the request-based [planning APIs](robot/REFERENCE.md#motion-planning-api) or the
[mapper lifecycle](mapper/README.md#use-from-c) for the task validated in Python.

1. Load artifacts and allocate workspaces and collision capacities once.
2. Warm representative calls before timing, including fallback branches used by the application.
3. Reuse instances and stable input/output addresses. Update scene slots, attachments, and payload
   inertials through the public APIs instead of reconstructing solvers for each request.
4. Check success, trajectory length, collision validity, joint/dynamics limits, and pose tolerances
   before consuming results. Copy borrowed outputs if needed beyond their documented lifetime.

Use contiguous CUDA buffers with the shapes in each header. Robot numeric arrays are float32,
quaternions use `wxyz`, and joint order comes from the exported model. Mapper images and features
have their own documented dtypes. Keep caller-owned buffers alive until their CUDA stream completes.
Native planning uses internal CUDA graphs by default; host-side stage orchestration is synchronous
and is not one externally capturable graph.

Locked/mimic joint values are resolved during robot export. The Python in-place kinematics
update recipe does not imply a native `update_kinematics_config` API; re-export changed locked
states unless the native public API explicitly supports that model change.

## Validate and measure

See [native checks and Python parity](robot/README.md#verify-the-deployment) and
[mapper smoke tests](mapper/README.md#notes). Artifact-dependent tests require an exported artifact;
a skipped test is not deployment validation.

Retain the same robot, obstacles, collision margins, seed/iteration budgets, tolerances,
dynamics settings, and graph policy for Python/native comparisons. Compare warmed instances on
the same hardware and report success/quality alongside compute time. The
[benchmark runner](benchmarks/README.md) retains inputs, exclusions, and provenance.
