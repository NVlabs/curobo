# C++ runtime benchmarks

For preparing inputs and kernels on a development GPU and running only C++ on another
device, see [Export the C++ runtime for another GPU](../../docs/guides/export_cpp_runtime.rst).
`--prepare-only --target-device DEVICE.json` skips the host build and prepares target kernels.

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
`--num-knots N` selects the motion planner's initialization-time control count (12..1024,
default 16); the runner exports matching artifacts and records the count in its results.

The runner builds the runtime in Release mode for the active GPU architecture and
exports kernels from the current checkout. Matching source/configuration hashes
permit artifact reuse within the same output directory. Export and build diagnostics
are retained in `.log` files; native failures stop the run.

`--profile node` records kernels inside CUDA graphs with Nsight Systems;
`--profile graph` records whole graph spans. Each case captures its first 50 measured
calls (five batches for IK), after the normal five-second warmup, through the same public
API. Remaining calls still run. Traces and SQLite exports use the `_profile` suffix.
Profiled timing summaries are diagnostic and must not be used as unprofiled performance results.

The output directory contains `summary.json`, rendered tables in `tables.rst`, source provenance and
SHA-256 hashes, frozen input files, explicit corpus exclusions, every native result
(including failures), and binary float32 executable trajectories. Motion metric
files retain dataset/scene/problem identities. A `.states.f32` file stores each
successful result as `[4, steps, dof]` in position/velocity/acceleration/jerk order;
`state_offset` in its matching JSONL record counts float32 elements.

Motion planning runs the unloaded and 3 kg Franka cases, with the
canonical per-dataset finger locks, 0.2 rad joint-bound expansion, OBB conversion,
and negative-`collision_buffer_ik` exclusions. Native defaults use 32 IK seeds,
four TrajOpt seeds, `outer_iterations=1` (one 50-step initial solve/check pass),
25 refinement iterations, 20 planning attempts, and time optimization. Both direct and
RRT planning are warmed on the first problem before measurement. The benchmark then repeats
public `plan_pose()` calls for at least five seconds to warm sustained GPU performance; the
extra call count and elapsed warmup time are recorded in the native log. Planning time is
synchronized host wall time for `plan_pose()`, including retries and time
optimization, and excludes scene updates, input copies, output copies, and scoring.

Both payload cases enable inverse dynamics and effort limits in IK, RRT, and TrajOpt
by default. `--disable-inverse-dynamics` disables those planning effort checks for both
payloads; the time optimizer and its final dynamics validation remain enabled.
The setting is recorded as `motion_inverse_dynamics` in the prepared `inputs.json`,
provenance, and summary. Native motion records retain `payload_mass_kg` and
`inverse_dynamics` independently. Historical unloaded results with planning effort checks
disabled describe the earlier protocol and must be distinguished from the new default.

Motion artifacts and the native Scene reserve only cuboids: the capacity is the
maximum OBB count across all selected eligible problems in both corpora (41 for
the full benchmark). Capsule, cylinder, mesh, and voxel capacities are zero.
The motion interpolation buffer contains at least 1,000 samples, growing with the configured
spline's worst-case dense horizon. Counts 12 and 16 retain the 1,000-sample buffer.
These export options
participate in the artifact cache fingerprint.

Motion statistics use successful plans; success includes every scored problem.
The summary also retains timing across all attempts. Path length is the sum of
joint-space Euclidean segment lengths, motion duration comes from the native
time-optimized result, and jerk is the maximum absolute joint jerk. Energy is
`sum(abs(torque * velocity)) * dt` and torque is the maximum absolute joint torque,
computed with Pinocchio RNEA on the returned executable states and matching payload
inertia. These metrics describe the model's trajectory, not measured hardware motion.

IK uses five batches of 100 reachable targets per robot and mode, generated with
seed 2 and the existing collision rejection sampler. Valid configurations accumulate across
draws until the batch is full; rejected samples never enter the inputs. Native FK constructs all
tool-frame targets outside timing. Both frontends use `create_ik_config` from
`benchmark/ik_benchmark.py`; the runner records the effective Python solver settings
and passes them to the public native IK API, with early exit enabled.

| Robot | LM seeds | LM iterations | L-BFGS seeds: pose / collision-free | L-BFGS iterations |
|---|---:|---:|---:|---:|
| G1 | 64 | 30 | 2 / 16 | 160 |
| Dual UR10e | 128 | 20 | 2 / 16 | 80 |
| Franka | 32 | 16 | 2 / 8 | 80 |

LM candidates are ranked by pose error before selecting the smaller L-BFGS pool.
Position/orientation tolerances are 5 mm / 0.05 rad, and collision optimization
activates at 2.5 mm. `--ik-input-dir PREVIOUS_OUTPUT` reuses saved native IK inputs.
The Python benchmark samples configurations directly, with optional `--robot` filtering.
Both frontends share the rejection-sampling helper, which retains partial valid draws.
Native timings retain the five-second warmup and synchronized wall-time protocol.

Samples are mapped by joint name to the native model; joints pruned by the pose-only
sampler retain the exported defaults. G1 uses `unitree_g1_29dof.yml` in both modes,
targeting `left_wrist_yaw_link`, `right_wrist_yaw_link`, `left_ankle_roll_link`, and
`right_ankle_roll_link`. The separate retargeting profile retains its 14 targets.
The collision-free mode checks self collision and `collision_table.yml`.
Three initial solves and at least five seconds of sustained public-API solves warm each
measured instance, using the same warmup helper as motion planning. Timings are mean synchronized wall time
per complete batch. Errors are the mean of each batch's successful-query 90th
percentile, in mm and degrees. Zero-success batches still count in success and timing.
First-use graph capture for a branch not exercised during warmup remains included.

Run the result-contract tests with:

```bash
pytest -q -n 0 cpp_runtime/benchmarks/tests
```
