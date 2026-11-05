Benchmarks & Profiling
======================

Recorded C++ Results
--------------------

Measured October 30–31, 2026 on an NVIDIA GeForce RTX 5090 with CUDA 13.0,
driver 595.99.02, and a Release build targeting SM 120. These are historical
measurements, rather than results from the current checkout.

CUDA graphs were enabled. Timings use synchronized host wall time after graph
warmup and at least five seconds of sustained warmup.

Motion planning
~~~~~~~~~~~~~~~

The Franka workload contains 2,590 scored MotionBenchMaker and MPINets problems
per payload, after 10 canonical exclusions. Planning time includes retries and
time optimization; scene updates, input/output copies, and scoring are excluded.

.. list-table:: Motion planning results
   :header-rows: 1

   * - Payload
     - Successful problems
     - Mean planning time (ms)
     - Mean motion duration (s)
   * - 0 kg
     - 2,582 / 2,590 (99.69%)
     - 4.464
     - 1.177
   * - 3 kg
     - 2,582 / 2,590 (99.69%)
     - 4.868
     - 1.266

Planning means include failed calls; motion durations include successful plans
only. Both cases use 32 IK seeds, four trajectory seeds, 16 spline controls,
one 50-step initial optimization pass, a separate 25-step refinement pass,
and up to 20 planning attempts.

The recorded unloaded case disables planning effort checks; the 3 kg case enables
them. Both retain dynamics in time optimization. The current runner enables
planning effort checks for both payloads, so its default unloaded timings use a
different policy.

Inverse kinematics
~~~~~~~~~~~~~~~~~~

Each case uses five batches of 100 reachable targets. Times are mean times for
an entire 100-query batch, including unsuccessful queries. Pose tolerances are
5 mm and 0.05 rad. Collision-free mode checks self collision and the
``collision_table.yml`` scene. G1 uses four wrist/ankle targets and 35 optimization
variables; dual UR10e uses two tool targets and Franka uses one.

.. list-table:: Inverse kinematics results
   :header-rows: 1

   * - Robot
     - Pose IK (ms/batch)
     - Success (%)
     - Collision-free IK (ms/batch)
     - Success (%)
   * - G1
     - 19.201
     - 100
     - 162.564
     - 90.2
   * - Dual UR10e
     - 3.574
     - 100
     - 8.434
     - 99.4
   * - Franka
     - 0.551
     - 100
     - 0.559
     - 100

Run Benchmarks
--------------

The native runner requires CMake, a C++17 compiler, a CUDA toolkit, and a supported
NVIDIA GPU. From the repository root:

.. code-block:: bash

   source .venv/bin/activate
   uv pip install -e ".[benchmark]"
   python cpp_runtime/benchmarks/run_benchmarks.py \
     --output cpp_runtime/benchmarks/outputs/full_run

Use ``--suite motion`` or ``--suite ik`` to select a workload. ``--max-problems 2``
produces a partial motion smoke run. ``--disable-inverse-dynamics`` disables
planning effort checks for both payloads while retaining time-optimization
dynamics checks.

Outputs include ``summary.json``, ``tables.rst``, source and artifact hashes,
frozen inputs, exclusions, failures, and trajectory data. Native export and
cross-device deployment are covered in :doc:`../guides/export_cpp_runtime`.

Profiling
---------

The native runner accepts ``--profile node`` or ``--profile graph`` for Nsight
Systems traces. Profiled timings are diagnostic and should be kept separate
from unprofiled results.

For inverse-dynamics CUDA kernel timings:

.. code-block:: bash

   python benchmark/inverse_dynamics_kernel_benchmark.py
