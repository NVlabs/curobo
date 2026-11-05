Benchmarks & Profiling
======================

Latest C++ Runtime Results
**************************

Measured October 30--31, 2026 (America/Los_Angeles), using an NVIDIA GeForce RTX 5090, CUDA 13.0,
NVIDIA driver 595.99.02, and a Release build targeting SM 120. All measured
planning and IK calls use the standalone C++ runtime with CUDA graphs enabled.
The measured working tree is based on revision ``842eed7``, including the
local planner and kernel improvements. Exact source, executable, raw-result, input,
and artifact hashes are retained with the results; the revision alone does not
identify this build.

The GPU used Prefer Maximum Performance (PowerMizer mode 1) at a 575 W power limit.
Clocks were not locked. The CPU governor and energy preference were ``performance``.
No foreign GPU compute process was observed during the native benchmark runs.
Both motion planning and IK use at least five seconds of sustained public-API warmup
on the measured instance, after their initial graph warmup calls and outside timing.
The motion run contains 5,180 planning calls. The updated IK run contains 3,000 queries.

These measurements use synchronized host wall time. Hardware, seed counts,
timing boundaries, and time optimization differ from the earlier Python/RTX 6000
Ada measurements, so these values are not a matched comparison with those results.

Motion Generation
-----------------

The full MotionBenchMaker and Motion Policy Networks corpus contains 2,600
problems. The canonical negative-``collision_buffer_ik`` filter excludes 10,
leaving 799 MotionBenchMaker and 1,791 MPINets problems per payload setting.
Success includes all 2,590 scored problems; the remaining table statistics use
successful plans. Both settings use 32 IK seeds, four TrajOpt seeds,
``outer_iterations=1`` (one 50-step initial solve/check pass per optimization),
one separate 25-step refinement pass, at most 20 planning attempts, and time optimization.
Waypoint planning defaults to ``outer_iterations=4``; these tables measure single-pose
planning. See :doc:`api_overview` for the pass-count API.

The recorded 3 kg setting enforces effort limits in IK, RRT, TrajOpt, and time optimization.
The recorded unloaded setting disables planning effort checks and retains dynamics in
time optimization. The current benchmark runner enables planning inverse dynamics for
both payloads by default; ``--disable-inverse-dynamics`` restores the earlier unloaded
planning policy for either payload. The existing unloaded timings below use the earlier
policy and have not been remeasured with the new default.
The world reserves 41 cuboids, the maximum across the full eligible corpus;
capsule, cylinder, mesh, and voxel capacities are zero. The motion interpolation
buffer contains 1,000 samples, with 16 spatial spline controls.

Plan time includes the complete synchronous ``plan_pose()`` call, retries, and
time optimization. Scene updates, input/output copies, and metric computation are
outside timing. Across **all** calls, including failures, the mean is
**4.464 ms at 0 kg** and
**4.868 ms at 3 kg**.
The successful-only timing distributions appear below.

Path length, duration, and maximum absolute jerk describe the returned executable
time-optimized trajectory. Energy is the integral approximation
``sum(abs(torque * velocity)) * dt``; torque is the maximum absolute joint torque.
Pinocchio RNEA computes both from the executable samples with matching robot and
payload inertia. The October 31 cleanup removes unused backward calculations from final
validation rollouts. The 3 kg mean changed from 4.883 to 4.868 ms in one comparison;
this small difference does not establish a robust speedup. Success and configured iteration
budgets are unchanged. Native before/after validation passed 38,176 values covering
feasibility, refinement, effort limits, and CUDA graph replay.

The independent Pinocchio audit of the preceding October 30 run verified start configurations
and endpoint pose tolerances for all 5,164 successful plans and found no effort-limit
violations in its 3 kg trajectories (maximum torque/limit ratio: 0.9502).

Unloaded (0 kg):

+---------------------+----------------------------------------------------------------------+
| Metric              | Value                                                                |
+=====================+======================================================================+
| Successful problems | 2582 / 2590                                                          |
+---------------------+----------------------------------------------------------------------+
| Success %           | 99.69                                                                |
+---------------------+----------------------------------------------------------------------+
| Plan Time (s)       | mean: 0.00441 ± 0.00127  median: 0.00414  75%: 0.00445  98%: 0.00790 |
+---------------------+----------------------------------------------------------------------+
| Position Error (mm) | mean: 0.011 ± 0.174  median: 0.000  75%: 0.000  98%: 0.038           |
+---------------------+----------------------------------------------------------------------+
| Path Length (rad.)  | mean: 3.191 ± 1.108  median: 3.297  75%: 3.866  98%: 5.462           |
+---------------------+----------------------------------------------------------------------+
| Motion Time(s)      | mean: 1.177 ± 0.356  median: 1.175  75%: 1.405  98%: 2.040           |
+---------------------+----------------------------------------------------------------------+
| Jerk (rad/s³)       | mean: 448.483 ± 25.876  median: 453.104  75%: 467.386  98%: 474.955  |
+---------------------+----------------------------------------------------------------------+
| Energy (J)          | mean: 56.030 ± 33.057  median: 48.129  75%: 72.630  98%: 147.517     |
+---------------------+----------------------------------------------------------------------+
| Torque (N·m)        | mean: 48.029 ± 17.197  median: 46.037  75%: 60.689  98%: 82.208      |
+---------------------+----------------------------------------------------------------------+

With torque limits at full payload of 3 kg:

+---------------------+----------------------------------------------------------------------+
| Metric              | Value                                                                |
+=====================+======================================================================+
| Successful problems | 2582 / 2590                                                          |
+---------------------+----------------------------------------------------------------------+
| Success %           | 99.69                                                                |
+---------------------+----------------------------------------------------------------------+
| Plan Time (s)       | mean: 0.00481 ± 0.00222  median: 0.00449  75%: 0.00468  98%: 0.00987 |
+---------------------+----------------------------------------------------------------------+
| Position Error (mm) | mean: 0.013 ± 0.172  median: 0.000  75%: 0.000  98%: 0.035           |
+---------------------+----------------------------------------------------------------------+
| Path Length (rad.)  | mean: 3.347 ± 1.281  median: 3.404  75%: 4.073  98%: 6.441           |
+---------------------+----------------------------------------------------------------------+
| Motion Time(s)      | mean: 1.266 ± 0.454  median: 1.248  75%: 1.471  98%: 2.563           |
+---------------------+----------------------------------------------------------------------+
| Jerk (rad/s³)       | mean: 442.662 ± 43.960  median: 453.064  75%: 466.823  98%: 474.915  |
+---------------------+----------------------------------------------------------------------+
| Energy (J)          | mean: 77.423 ± 40.698  median: 69.589  75%: 98.609  98%: 184.741     |
+---------------------+----------------------------------------------------------------------+
| Torque (N·m)        | mean: 66.019 ± 15.766  median: 69.272  75%: 81.836  98%: 82.631      |
+---------------------+----------------------------------------------------------------------+

Latest Inverse Kinematics Results
*********************************

Each robot and mode uses five batches of 100 reachable targets, sampled with
seed 2. Valid configurations accumulate across rejection-sampling draws until
all 100 targets are available; the joint bounds and collision checks are unchanged.
The October 31 rerun reuses the frozen joint configurations and takes its settings directly from
``benchmark/ik_benchmark.py``. LM first refines a larger pool, then selects the smaller
L-BFGS seed set. Both modes retain early exit and CUDA graphs.

.. list-table:: Shared Python and C++ IK settings
   :header-rows: 1

   * - Robot
     - LM seeds
     - LM iterations
     - L-BFGS seeds: pose / collision-free
     - L-BFGS iterations
   * - G1
     - 64
     - 30
     - 2 / 16
     - 160
   * - Dual UR10e
     - 128
     - 20
     - 2 / 16
     - 80
   * - Franka
     - 32
     - 16
     - 2 / 8
     - 80

Pose tolerances are 5 mm and 0.05 rad. Collision optimization activates at 2.5 mm;
final validation requires zero collision cost. The collision-free mode checks self collision
and the ``collision_table.yml`` scene. All six cases use the five-second sustained warmup.

Times are the mean synchronized wall time for an entire **100-query batch**.
Errors are the mean of the successful-query 90th percentiles from each batch.
Failed queries remain in the success denominator, and zero-success batches remain
in timing. Native FK target construction is outside timing. First-use graph capture
for a branch not reached during warmup remains included.

The G1 benchmark uses ``unitree_g1_29dof.yml``: 29 actuated joints plus six floating-base
coordinates (35 optimization variables), with no articulated finger joints. Its four targets
are ``left_wrist_yaw_link``, ``right_wrist_yaw_link``, ``left_ankle_roll_link``, and
``right_ankle_roll_link``. Each target tracks position and orientation. Dual UR10e has
12 variables and two tool frames; Franka has seven variables and one tool frame.

The separate ``unitree_g1_29dof_retarget.yml`` retains its 14 tool frames. On the same
500 joint configurations per mode, the Python benchmark achieves 100% pose-only and
96.0% collision-free success with four targets, compared with 60.6% and 60.2% with
the retargeting profile. Seed counts and iteration budgets are unchanged; this comparison
changes the target constraints. Native G1 success is 100% pose-only and 90.2% collision-free;
the collision-free gap relative to Python remains. Python achieves 100% in both modes for
dual UR10e and Franka; native dual-UR10e collision-free success is 99.4%.

+----+----------------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+
|    | robot                | IK-time(ms)   | Batch-Size   | Success-IK   | Position-Error(mm)   | Orientation-Error(deg)   | C-Free-IK-time(ms)   | Success-C-Free-IK   | Position-Error-C-Free-IK(mm)   | Orientation-Error-C-Free-IK(deg)   |
+====+======================+===============+==============+==============+======================+==========================+======================+=====================+================================+====================================+
| 0  | unitree_g1_29dof.yml | 19.2012       | 100          | 100          | 0.000969868          | 2.74406e-05              | 162.564              | 90.2                | 0.0917246                      | 0.0178088                          |
+----+----------------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+
| 1  | dual_ur10e.yml       | 3.57369       | 100          | 100          | 0.000133293          | 1.19755e-05              | 8.43368              | 99.4                | 0.000103888                    | 1.57285e-05                        |
+----+----------------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+
| 2  | franka.yml           | 0.550719      | 100          | 100          | 0.000143497          | 1.08788e-05              | 0.558616             | 100                 | 0.000169512                    | 1.25649e-05                        |
+----+----------------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+

Running Benchmarks
******************

Install the ``benchmark`` extra for artifact preparation, datasets, and scoring:

.. code-block:: bash

    source .venv/bin/activate
    pip install -e ".[benchmark]"
    # or: uv sync --extra benchmark

The C++ benchmarks require CMake, a C++17 compiler, the CUDA toolkit, and a supported
NVIDIA GPU. The runner builds the native executable for the active GPU, exports
current robot artifacts, freezes inputs, and saves every attempt and trajectory:

.. code-block:: bash

    python cpp_runtime/benchmarks/run_benchmarks.py \
      --output cpp_runtime/benchmarks/outputs/full_run

``summary.json`` contains aggregates and ``tables.rst`` contains rendered tables.
The output also retains explicit exclusions, failures, source provenance, hashes,
and float32 trajectory buffers. Motion measurements are retained in
``cpp_runtime/benchmarks/outputs/three_ms_review_20261031/cost_only/``. Its parent directory
contains the before/after comparison, kernel and graph profiles, native reference checks,
and GPU telemetry. The preceding run and its independent ``trajectory_audit.json`` remain in
``cpp_runtime/benchmarks/outputs/final_motion_ik_20261030/``. The runner's
``README.md`` documents the input and output contract. Updated IK measurements are in
``cpp_runtime/benchmarks/outputs/g1_four_targets_20261031/native/``. The parent directory
retains the Python G1 comparison and the provenance of the reused joint configurations.

Inverse Kinematics
------------------

.. code-block:: bash

    python cpp_runtime/benchmarks/run_benchmarks.py --suite ik \
      --output cpp_runtime/benchmarks/outputs/ik_run

The Python benchmark samples configurations directly using the shared rejection sampler:

.. code-block:: bash

    python benchmark/ik_benchmark.py --save_path benchmark/results

The output directory is created automatically. Valid partial draws accumulate until each
100-query batch is full; no saved JSON inputs are needed.

Motion Planning
---------------

This runs both the unloaded and 3 kg settings with planning inverse dynamics enabled
across the full corpus:

.. code-block:: bash

    python cpp_runtime/benchmarks/run_benchmarks.py --suite motion \
      --output cpp_runtime/benchmarks/outputs/motion_run

``--max-problems 2`` limits motion planning to two problems per corpus for a smoke
run. Limited summaries are marked partial and must not replace the full tables.
``--prepare-only`` builds and freezes inputs without measured benchmark calls.
``--disable-inverse-dynamics`` disables effort checks in motion-planning IK, RRT, and
TrajOpt for both payloads. Time optimization retains its dynamics checks. The effective
setting is recorded in the inputs, provenance, summary, and native motion results.

Inverse Dynamics Kernel Profiling
---------------------------------

The separate Python kernel profiler measures inverse-dynamics forward and backward
CUDA kernel execution across robots and batch sizes. It is not used for the C++
runtime tables above:

.. code-block:: bash

    python benchmark/inverse_dynamics_kernel_benchmark.py

Use ``--robots`` and ``--batch-sizes`` to select these kernel workloads.
