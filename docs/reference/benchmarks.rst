Benchmarks & Profiling
======================

Run this script to reproduce the timings:

.. code-block:: bash

   source .venv/bin/activate
   uv pip install -e ".[benchmark]"
   python cpp_runtime/benchmarks/run_benchmarks.py \
     --output cpp_runtime/benchmarks/outputs/full_run

Metrics use warmed C++ Release builds with CUDA graphs on an RTX 5090
(CUDA 13.0, SM 120) and Jetson AGX Orin in MAXN mode (CUDA 13.2, SM 87).
Timings are synchronized host wall time.

Motion planning
---------------

Franka without payload: 2,590 MotionBenchMaker and MPINets problems, after 10 exclusions.
Planning time includes retries and TimeOpt; scene updates, I/O, and scoring are
excluded. Planning means include failures; motion durations use successful plans.

.. list-table:: Motion planning with inverse dynamics
   :header-rows: 1

   * - GPU
     - Successful problems
     - Mean planning time (ms)
     - Mean motion duration (s)
   * - RTX 5090
     - 2,583 / 2,590 (99.73%)
     - 3.797
     - 1.189
   * - Jetson AGX Orin
     - 2,583 / 2,590 (99.73%)
     - 25.662
     - 1.188

Without inverse dynamics
~~~~~~~~~~~~~~~~~~~~~~~~

With ``--disable-inverse-dynamics``, success checks the goal, collisions, and joint
kinematic limits. Torque feasibility is unchecked, including during TimeOpt and
final validation.

.. list-table:: Motion planning without inverse dynamics
   :header-rows: 1

   * - GPU
     - Kinematic successes
     - Mean planning time (ms)
     - Mean motion duration (s)
   * - RTX 5090
     - 2,583 / 2,590 (99.73%)
     - 3.348
     - 1.168
   * - Jetson AGX Orin
     - 2,583 / 2,590 (99.73%)
     - 20.800
     - 1.170

Inverse kinematics
------------------

Mean time per 100-query batch, five batches per case; tolerances are 5 mm and 0.05 rad.

.. list-table:: Inverse kinematics on RTX 5090
   :header-rows: 1

   * - Robot
     - Pose IK (ms/batch)
     - Success (%)
     - Collision-free IK (ms/batch)
     - Success (%)
   * - G1
     - 19.026
     - 100
     - 138.448
     - 89.0
   * - Dual UR10e
     - 3.574
     - 100
     - 8.460
     - 99.4
   * - Franka
     - 0.556
     - 100
     - 0.563
     - 100

Profiling
---------

Use ``--profile node`` or ``--profile graph`` for Nsight Systems traces.
Profiled timings are diagnostic.
