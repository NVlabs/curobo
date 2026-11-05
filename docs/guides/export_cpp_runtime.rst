.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Export the C++ runtime for another GPU
======================================

Prepare robot artifacts on a development GPU, then deploy them to a different CUDA GPU,
including an ARM64 Jetson Thor. The deployment machine needs the C++ runtime, CUDA driver
and runtime libraries, and exported artifacts. PyTorch, Warp, and cuRobo Python packages
are needed only on the export machine. Mesh scenes additionally need Warp's native mesh
backend; the cuboid-only benchmark does not.

Describe the deployment GPU
---------------------------

Copy the standalone device probe to the deployment machine and run it with system Python.
It uses only Python's standard library and ``libcuda.so.1``:

.. code-block:: bash

   scp cpp_runtime/robot/tools/export_device.py user@target-host:/tmp/export_device.py
   ssh user@target-host \
     'python3 /tmp/export_device.py --out /tmp/thor.json'
   scp user@target-host:/tmp/thor.json ./thor.json

Replace ``user@target-host`` with your deployment machine's SSH destination.
``--device N`` selects a visible CUDA device
ordinal; the default is zero. The JSON records the GPU name, cubin architecture, SM count,
and opt-in shared-memory limit. Probe the intended GPU partition when using MIG.

Export robot artifacts
----------------------

Run the exporter in the development environment:

.. code-block:: bash

   source .venv/bin/activate
   python cpp_runtime/robot/tools/export_robot.py \
     --robot-config curobo/content/configs/robot/franka.yml \
     --target-device thor.json \
     --out cpp_runtime/robot/artifacts/franka_thor

Omit ``--target-device`` to keep exporting for the active local GPU. Use a separate output
directory for each target. The robot kinematics manifest retains the device description;
all exported CUDA and Warp cubins use its architecture. L-BFGS uses its shared-memory limit,
and RNEA launch profiles use its SM count. Foreign cubins are compiled without loading them
on the export GPU; validate them on the deployment GPU before relying on performance results.

The development machine still needs a supported CUDA GPU for robot preparation and IK input
sampling. Its NVRTC and Warp compilers must both support the target architecture. Warp bundles
its own CUDA compiler: installing CUDA 13 on the host does not upgrade a CUDA 12 Warp wheel.
For Thor (``sm_110``), use a CUDA 13 Warp build on the export host. For example, the Linux x86-64
CUDA 13 wheel for Warp 1.17 is available from
`NVIDIA's Warp releases <https://github.com/NVIDIA/warp/releases/tag/v1.17.0>`_:

.. code-block:: bash

   uv pip install \
     'https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0+cu13-py3-none-manylinux_2_28_x86_64.whl'

Choose the release and CPU architecture matching your development environment. The exporter
checks compiler architecture support before writing robot artifacts. See
`Warp installation instructions <https://github.com/NVIDIA/warp/blob/main/docs/user_guide/installation.rst>`_
for other builds.

Prepare benchmark inputs locally
--------------------------------

The benchmark runner can prepare target artifacts and frozen inputs without building or
executing a host benchmark:

.. code-block:: bash

   python cpp_runtime/benchmarks/run_benchmarks.py \
     --suite motion --max-problems 2 --prepare-only \
     --target-device thor.json \
     --output cpp_runtime/benchmarks/outputs/thor_smoke

The example prepares a partial smoke workload. Remove ``--max-problems 2`` for the full motion
corpus; use ``--suite all`` to include IK. Preparation needs the repository's ``benchmark`` extra
on the development machine. The device description participates in the export cache key,
and ``--target-device`` requires ``--prepare-only`` to prevent execution of target kernels locally.

Copy the prepared output directory to the deployment machine. Keep ``inputs.json``, source
provenance, exclusions, and artifact hashes with the results. Paths in ``inputs.json`` refer to
the export machine: replace their output-directory prefix when supplying native arguments.

Build and run on the deployment machine
---------------------------------------

GPU cubins are independent of the host CPU architecture. The runtime library and benchmark
executable must be built for the deployment CPU. An x86-64 executable cannot run on ARM64 Thor.
Building on Thor avoids needing an ARM64 cross-compilation toolchain.

Transfer these source paths, preserving their repository-relative layout:

- ``cpp_runtime/robot/`` (exclude generated artifacts and caches);
- ``cpp_runtime/benchmarks/CMakeLists.txt`` and ``cpp_runtime/benchmarks/benchmark.cpp``;
- ``curobo/_src/curobolib/kernels/``.

On Thor, build with the existing CUDA toolkit and C++17 compiler:

.. code-block:: bash

   export PATH=/usr/local/cuda/bin:$PATH
   cmake -S cpp_runtime/benchmarks -B build/thor \
     -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110 -DBUILD_TESTING=OFF
   cmake --build build/thor --target curobo_benchmark -j2

For another GPU, replace ``110`` with the device description's architecture number. Keep the
matching ``libcurobo_robot_runtime.so`` with the executable, preserving the build layout or
setting ``LD_LIBRARY_PATH`` to the library directory. A CUDA toolkit supporting the target
architecture is required for this build. Export with a toolkit compatible with the target
driver; a newer driver capability reported by ``nvidia-smi`` does not identify the installed
toolkit version.

The native motion invocation is:

.. code-block:: text

   curobo_benchmark motion ARTIFACT INPUT RESULTS PAYLOAD_3KG UNUSED MAX_CUBOIDS NUM_KNOTS

Use the dataset artifact/input and capacities from ``inputs.json``. ``PAYLOAD_3KG=0`` selects the
unloaded case and ``1`` selects the 3 kg case. Both enable inverse dynamics and planning effort
limits by default. Append ``--disable-inverse-dynamics`` to disable effort checks in IK, RRT,
and TrajOpt while retaining the payload mass. Time optimization and its final dynamics
validation remain enabled. Pass ``0`` for the unused argument; the executable runs every
problem in the prepared input. The native IK invocation is:

.. code-block:: text

   curobo_benchmark ik ARTIFACT INPUT RESULTS COLLISION_FREE CONFIG

Run each dataset/payload or robot/mode separately. Copy native ``.jsonl`` and ``.states.f32``
results back to the development environment for full trajectory scoring. The executable
retains its normal CUDA graphs and sustained warmup.

Capture kernel timings
----------------------

Measure the unprofiled workload first, then run the same native command in a separate output
directory with Nsight Systems:

.. code-block:: bash

   nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
     --cuda-graph-trace=node --capture-range=cudaProfilerApi \
     --capture-range-end=stop --export=sqlite --output=thor_profile \
     build/thor/curobo_benchmark motion \
     ARTIFACT INPUT PROFILE_RESULTS DYNAMICS 0 MAX_CUBOIDS NUM_KNOTS --profile
   nsys stats --report cuda_gpu_kern_sum thor_profile.nsys-rep

The benchmark captures up to 50 warmed motion calls or five IK batches. Node tracing exposes
individual graph kernels. Summed kernel durations can exceed wall time because streams
overlap; profile timings are diagnostic. Monitor actual clocks, temperature, and concurrent
GPU activity during measurement. Use Nsight Compute on the dominant kernels when hardware
counter access is available.
