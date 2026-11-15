Benchmarks & Profiling
======================

Run this script to reproduce the timings:

.. code-block:: bash

   source .venv/bin/activate
   uv pip install -e ".[benchmark]"
   python cpp_runtime/benchmarks/run_benchmarks.py \
     --output cpp_runtime/benchmarks/outputs/full_run

Metrics measured on an NVIDIA GeForce RTX 5090 are shown below. The C++ runtime
uses CUDA 13.0, a Release build targeting SM 120, and CUDA graphs. Timings are
warmed, synchronized host wall time.

Motion planning
---------------

Franka: 2,590 MotionBenchMaker and MPINets problems per payload, after 10 exclusions.
Planning time includes retries and TimeOpt; scene updates, I/O, and scoring are
excluded. Planning means include failures; motion durations use successful plans.

.. list-table:: Motion planning with inverse dynamics
   :header-rows: 1

   * - Payload
     - Successful problems
     - Mean planning time (ms)
     - Mean motion duration (s)
   * - 0 kg
     - 2,583 / 2,590 (99.73%)
     - 3.797
     - 1.189
   * - 3 kg
     - 2,583 / 2,590 (99.73%)
     - 3.974
     - 1.253

Without inverse dynamics
~~~~~~~~~~~~~~~~~~~~~~~~

With ``--disable-inverse-dynamics``, success checks the goal, collisions, and joint
kinematic limits. Torque feasibility is unchecked, including during TimeOpt and
final validation.

.. list-table:: Motion planning without inverse dynamics
   :header-rows: 1

   * - Payload
     - Kinematic successes
     - Mean planning time (ms)
     - Mean motion duration (s)
   * - 0 kg
     - 2,583 / 2,590 (99.73%)
     - 3.348
     - 1.168
   * - 3 kg
     - 2,583 / 2,590 (99.73%)
     - 3.355
     - 1.170

Inverse kinematics
------------------

Mean time per 100-query batch, five batches per case; tolerances are 5 mm and 0.05 rad.

.. list-table:: Inverse kinematics results
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
