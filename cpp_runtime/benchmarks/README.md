# C++ runtime benchmarks

Run the motion-planning and inverse-kinematics workloads reported in
[`docs/reference/benchmarks.rst`](../../docs/reference/benchmarks.rst) from the repository root:

```bash
source .venv/bin/activate
python cpp_runtime/benchmarks/run_benchmarks.py \
  --output cpp_runtime/benchmarks/outputs/full_run
```

Install the repository's `benchmark` extra first. CMake, a C++17 compiler, the CUDA
toolkit, and a supported NVIDIA GPU are required. Python prepares robot artifacts,
samples reachable IK targets, converts dataset obstacles to OBBs, and computes
trajectory metrics. All measured planning and IK calls execute in the standalone
`curobo_benchmark` C++ executable, using the public runtime APIs with CUDA graphs.

Use `--suite motion` or `--suite ik` to select a workload. Use `--max-problems 2`
for a motion smoke run; its summary is marked partial. `--prepare-only` builds and
exports inputs without running measured benchmarks. Run without limits for the
documentation tables. Keep each independent measurement in its own output directory.

The runner builds the runtime in Release mode for the active GPU architecture and
exports kernels from the current checkout. Matching source/configuration hashes
permit artifact reuse within the same output directory. Export and build diagnostics
are retained in `.log` files; native failures stop the run.

The output directory contains `summary.json`, rendered tables in `tables.rst`, source provenance and
SHA-256 hashes, frozen input files, explicit corpus exclusions, every native result
(including failures), and binary float32 executable trajectories. Motion metric
files retain dataset/scene/problem identities. A `.states.f32` file stores each
successful result as `[4, steps, dof]` in position/velocity/acceleration/jerk order;
`state_offset` in its matching JSONL record counts float32 elements.

Motion planning runs the unloaded and torque-limited 3 kg Franka cases, with the
canonical per-dataset finger locks, 0.2 rad joint-bound expansion, OBB conversion,
and negative-`collision_buffer_ik` exclusions. Native defaults use 32 IK seeds,
four TrajOpt seeds, 50 + 25 iterations, 20 planning attempts, and time optimization. Both direct and
RRT planning are warmed on the first problem before measurement. Planning time is
synchronized host wall time for `plan_pose()`, including retries and time
optimization, and excludes scene updates, input copies, output copies, and scoring.

Motion artifacts and the native Scene reserve only cuboids: the capacity is the
maximum OBB count across all selected eligible problems in both corpora (41 for
the full benchmark). Capsule, cylinder, mesh, and voxel capacities are zero.
The motion interpolation buffer contains 1,000 samples. These export options
participate in the artifact cache fingerprint.

Motion statistics use successful plans; success includes every scored problem.
The summary also retains timing across all attempts. Path length is the sum of
joint-space Euclidean segment lengths, motion duration comes from the native
time-optimized result, and jerk is the maximum absolute joint jerk. Energy is
`sum(abs(torque * velocity)) * dt` and torque is the maximum absolute joint torque,
computed with Pinocchio RNEA on the returned executable states and matching payload
inertia. These metrics describe the model's trajectory, not measured hardware motion.

IK uses five batches of 100 reachable targets per robot and mode, generated with
seed 2 and the existing collision rejection sampler. Native FK constructs all
tool-frame targets outside timing. Both modes use 32 native IK seeds, 16 LM
iterations and the exported L-BFGS profile (160 iterations for G1), with early exit.
Samples are mapped by joint name to the native model; joints pruned by the pose-only
sampler retain the exported defaults. G1 uses `unitree_g1_29dof_retarget.yml` in both modes.
The collision-free mode checks self collision and `collision_table.yml`.
Three solves warm each measured instance. Timings are mean synchronized wall time
per complete batch. Errors are the mean of each batch's successful-query 90th
percentile, in mm and degrees. Zero-success batches still count in success and timing.
First-use graph capture for a branch not exercised during warmup remains included.

Run the result-contract tests with:

```bash
pytest -q -n 0 cpp_runtime/benchmarks/tests
```
