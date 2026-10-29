# cuRobo Robot C++ Runtime

This directory owns backend-neutral C++ APIs that operate on a robot model. It
currently provides fused forward kinematics, geometric Jacobians, multi-seed
goalset inverse kinematics, time-optimal B-spline seed fitting, multi-seed
joint-space trajectory optimization with uniform timestep refinement, bounded
bidirectional RRT, composed joint- and pose-space motion planning, fixed-target
MPC, voxel-volume
sphere fitting, attached-object collision geometry, and single-scene obstacle
storage.
Its wider scope includes:

- forward kinematics and Jacobians;
- inverse kinematics;
- collision queries; and
- motion generation, including planning and trajectory optimization.

Public headers must describe robot inputs, outputs, memory ownership, and CUDA
stream behavior without exposing backend-specific types. The implementation
uses cuRobo CUDA kernels and exported Warp kernels.

## Export a Robot

Generate a robot-specific artifact from a cuRobo YAML configuration. Repeat
`--tool-frame` to request more than one output link:

```bash
python cpp_runtime/robot/tools/export_robot.py \
  --robot-config curobo/content/configs/robot/franka.yml \
  --tool-frame panda_hand \
  --out cpp_runtime/robot/artifacts/franka
```

Reserve fixed collision storage for a runtime-held payload without editing the
authoring YAML:

```bash
python cpp_runtime/robot/tools/export_robot.py \
  --robot-config /path/to/robot.yml \
  --out /path/to/artifact \
  --attachment-parent-link tool_link \
  --attachment-spheres 32
```

This adds an identity-fixed `attached_object` link and emits the resolved YAML
inside the artifact. IK, RRT, and TrajOpt then expose the same 32-sphere
attachment capacity.

A basic URDF can be exported without a cuRobo robot configuration:

```bash
python cpp_runtime/robot/tools/export_robot.py \
  --urdf /path/to/robot.urdf \
  --base-link base_link \
  --tool-frame tool_link \
  --out cpp_runtime/robot/artifacts/robot
```

The exporter resolves locked and mimic joints, generates the packed topology and
deterministic IK seeds, and compiles the existing in-tree cuRobo kinematics,
pose-cost, Levenberg-Marquardt, B-spline, L-BFGS, RNEA, world-collision, and
self-collision kernels, plus the existing Warp RRT tree kernels, for the active
GPU architecture. It derives the seed fitter's exact B-spline bases, geometric
operator, derivative limits, and IRLS policy from the current Python fitter.
The deployed runtime does not parse URDF files or compile kernels at startup.
TimeOpt table schema 3 shares the duration and constraint basis and averages inverse
speed on the uniform grid without a weight table. Regenerate older timeopt tables with
`python cpp_runtime/robot/tools/export_timeopt.py /path/to/artifact --dof NUM_DOF`
(pass `--knots` for nondefault control counts).
The library build compiles fitter-specific CUDA reductions and small IRLS
solves; those kernels reuse the exported B-spline forward kernel and Python-
derived fitter tensors. Pass `--kinematics-only` when the artifact only needs
forward kinematics and Jacobians.

The IK artifact keeps the same 2,000-entry Halton cache as Python SeedIK by
default, so repeated `plan_pose()` attempts consume fresh seed windows. Set
`--ik-seed-samples` only when intentionally changing that retry schedule.

The motion-planning benchmark expands each active joint position limit by
0.2 radians. Export its parity artifact with `--joint-limit-expansion 0.2` so
IK, TrajOpt, and RRT derive their bounds and Halton samples from the same joint
space. The option defaults to zero for deployment artifacts.

C++-specific Warp glue factories live in `kernels/`. They are export-time
implementation details and are not part of `curobo/_src`; the exporter still
imports the production cuRobo kernels directly wherever an existing kernel
already implements the required operation.

A full cuRobo robot configuration also exports TrajOpt, RRT, and their composed
motion planner because it supplies the collision spheres and joint limits needed
to validate states and trajectories. Basic URDF export currently creates only
the kinematics and IK artifacts.

## Build

A CUDA toolkit containing `nvcc` is required to build the robot runtime.

```bash
cmake -S cpp_runtime/robot -B cpp_runtime/build/robot
cmake --build cpp_runtime/build/robot -j
```

Cuboids, capsules, cylinders, spheres, and voxel grids need only CUDA. To enable
triangle meshes, point CMake at the native headers from the same Warp installation
that will provide the runtime library:

```bash
WARP_ROOT="$(python -c \
  'from pathlib import Path; import warp; print(Path(warp.__file__).parent)')"
cmake -S cpp_runtime/robot -B cpp_runtime/build/robot \
  -DWARP_NATIVE_INCLUDE_DIR="${WARP_ROOT}/native"
cmake --build cpp_runtime/build/robot -j
export CUROBO_WARP_LIBRARY="${WARP_ROOT}/bin/warp.so"
```

The native library version is checked when mesh support is first used. Mesh vertices
and triangle indices are uploaded once, and the runtime delegates BVH construction to
Warp's existing native implementation.

Only MPC enables `line_search_backtracking` by default; IK and TrajOpt disable it.
Each MPC problem owns one GPU beta value: failed probe fans multiply it by `0.1`, accepted steps
reset it to `1`. Retries leave the search direction and L-BFGS history unchanged
and consume the fixed CUDA-graph iteration budget. Set the solver configuration
flag to `false` to disable this policy. Re-export existing artifacts after this
kernel ABI update (IK schema 13, TrajOpt schema 18). The artifacts contain separate
fixed-fan and backtracking kernels, selected before graph capture. The fixed-fan
candidate and line-search kernels do not read or update the backtracking scale.

## API Contract

Construct `curobo::robot::Kinematics` from an artifact directory and call
`forward()` with caller-owned CUDA buffers. A stable C ABI is also available in
`include/curobo_robot_runtime/api.h`.

```cpp
#include <curobo_robot_runtime/kinematics.h>

curobo::robot::Kinematics kinematics("/path/to/robot/artifact");
const auto info = kinematics.info();
kinematics.forward(joint_positions, link_positions, link_quaternions,
                   link_jacobians, batch_size, horizon, stream);
```

For input shape `[batch, horizon, num_dof]`, one fused launch writes:

- tool positions `[batch, horizon, num_tool_frames, 3]`;
- tool quaternions `[batch, horizon, num_tool_frames, 4]` in `wxyz` order; and
- world-frame Jacobians `[batch, horizon, num_tool_frames, 6, num_dof]`, with
  linear `xyz` rows followed by angular `xyz` rows.

All tensors are contiguous CUDA float32 memory. The call is asynchronous and
ordered on the supplied CUDA stream. Create the runtime before CUDA graph
capture; its kinematics calls can then be captured without allocation or
synchronization.

## Inverse Kinematics API

Construct one fixed-workspace solver for an exported robot. An optional `Scene`
is shared by every goal in a solve request.

```cpp
#include <curobo_robot_runtime/inverse_kinematics.h>

curobo::robot::InverseKinematicsConfig config;
config.max_batch_size = 1;
config.max_goalset = 64;
config.num_seeds = 32;
config.check_effort_limits = true;
curobo::robot::InverseKinematics ik("/path/to/robot/artifact", config, &scene);

curobo::robot::IkSolveRequest request;
request.goal_positions = goal_positions;
request.goal_quaternions = goal_quaternions;
request.current_positions = current_positions;  // Optional ranking reference.
request.solution = solution;
request.success = success;
request.position_error = position_error;
request.orientation_error = orientation_error;
request.collision_cost = collision_cost;
request.goalset_index = goalset_index;
request.num_goalset = num_goalset;
request.num_solutions = num_solutions;
request.stream = stream;
ik.solve(request);
```

Inputs and outputs are caller-owned contiguous CUDA buffers. Goals have shapes
`[batch, num_tool_frames, num_goalset, 3]` and
`[batch, num_tool_frames, num_goalset, 4]`; quaternions use `wxyz` order. Set
`num_goalset = 1` for a single pose. Optional seeds have shape
`[batch, num_seeds, num_dof]`. Optional current positions have shape
`[batch, num_dof]` and bias ranking toward nearby configurations. Set
`num_solutions` to return multiple distinct ranked candidates. Solutions have
shape `[batch, num_solutions, num_dof]`, metrics have shape
`[batch, num_solutions]`, and selected goal indices have shape
`[batch, num_solutions, num_tool_frames]`; the index output is required for
goalsets. By default, the solver evaluates the LM result and synchronizes once
to return immediately when each batch item has at least one valid solution. LM
and L-BFGS have separate internal CUDA Graphs, retain stable buffers, and recapture
only when request addresses or dimensions change. Set `exit_early=false` to
always run L-BFGS and preserve a fully asynchronous call, including when
wrapping `solve()` in a caller-owned graph. Set `use_cuda_graph=false` only for
eager debugging or caller-owned capture. A solver owns one workspace and must
not execute concurrent calls.

Goalset scoring uses one CUDA block per seed, with goal poses distributed across
threads and a stable reduction to the lowest-cost goal. Final validation computes
each seed's feasibility and ranking score once, then ranks seeds in parallel.
Both stages remain inside the solver's CUDA Graphs; equal scores retain the
lower goal/seed index. These kernels use IK artifact schema 13.

IK first runs pose-only Levenberg-Marquardt seed refinement, with no world- or
self-collision cost in the refinement loop. It then evaluates pose convergence,
joint bounds, world collision, and self collision and, when `exit_early=true`,
skips L-BFGS if every batch item has at least one valid ranked result. The
fallback L-BFGS reuses the current cuRobo pose metric, c-space cost, fused
forward and reverse kinematics, line search, world collision, and exact
self-collision kernels.
World- and self-collision gradients are added inside backward kinematics, without
an intermediate sum buffer or separate merge launch, in IK and TrajOpt. A zero
`lbfgs_iterations` uses the iteration count stored in the artifact's Python IK
profile; `lm_iterations` configures the preceding LM stage.
When `check_effort_limits=true`, L-BFGS includes the RNEA effort cost and its
configuration gradient, and final ranking rejects solutions outside the
exported effort limits. The pose-only LM iterations remain unchanged.
L-BFGS and final validation run dynamics and kinematics concurrently, then
evaluate pose, C-space, world-collision, and self-collision terms on separate
persistent streams before joining their results.

The referenced `Scene` must outlive the solver.

World collision layers use unsigned 32-bit masks. `Scene::set_obstacle_layers()` sets an
obstacle's membership; `set_link_collision_mask()` selects the layers checked by a robot
link's spheres. Both default to bit zero (value `1`). A zero mask excludes all world layers;
it does not disable self-collision. Matching several bits checks an obstacle only once.
Updates retain GPU addresses and are visible to subsequent CUDA Graph replays. Complete
previous solver calls before changing policy; these synchronous setters run outside capture.

`MotionPlanner` shares link-local spheres, masks, and attachment state between ordinary IK,
grasp IK, RRT, and TrajOpt, while keeping solver workspaces separate. Native scenes remain
single-environment. Re-export artifacts for IK schema 13, TrajOpt schema 18, and RRT schema 8
before loading them into the updated runtime.

`GraspPlanningRequest::collision_policy` enables the two-field grasp path. The caller owns
coherent full/target-excluded scene grids and finite robot-frame GPU target points until
`plan_grasp` returns. Contact links use the target-excluded field for initial IK/insertion;
transit uses the full field. Lift fits a conservative 4x4x2 sphere cover in the selected
tool frame and uses the target-excluded field for robot and attachment. The attachment link
must coincide with the single grasp tool frame and reserve 32 spheres. This temporary
override never changes live attachment ownership; all exits restore prior geometry/masks.
Normal queries can use `MotionPlanner::set_collision_mask(1U | 2U)` for COMMON plus FULL.
The mapper keeps ESDF output and instance selection GPU-resident across graph replay.
Recorded-scene and Ada qualification are still required before physical execution.

## Trajectory Optimization API

`curobo::robot::TrajectoryOptimizer` optimizes one joint-space request with
multiple parallel seeds. It accepts common fallback endpoints and optional
paired start/goal configurations per seed. It returns 16 optimized cubic
B-spline knots plus the current default 81-state trajectory. Query `info()`
instead of hard-coding those artifact dimensions.

```cpp
#include <curobo_robot_runtime/trajectory_optimization.h>

curobo::robot::TrajectoryOptimizationConfig config;
config.num_seeds = 4;
config.max_iterations = 25;
config.refinement_iterations = 25;
config.trajectory_dt = 0.15F;
config.finetune_attempts = 2;
config.finetune_dt_scale = 0.6F;
config.check_effort_limits = true;  // Optional RNEA cost and feasibility check.
curobo::robot::TrajectoryOptimizer trajopt(
    "/path/to/robot/artifact", config, &scene);

const auto info = trajopt.info();
curobo::robot::TrajectoryOptimizationRequest request;
request.start_position = start_position;
request.start_velocity = start_velocity;          // Optional [num_dof].
request.start_acceleration = start_acceleration;  // Optional [num_dof].
request.goal_position = goal_position;
request.seed_knots = seed_knots;  // Optional [num_seeds, num_knots, num_dof].
request.seed_start_positions = seed_starts;  // Optional [num_seeds, num_dof].
request.seed_goal_positions = seed_goals;  // Optional [num_seeds, num_dof].
request.seed_valid = ik_success;  // Optional uint8 [num_seeds].
request.optimized_knots = optimized_knots;
request.trajectory_position = trajectory_position;
request.trajectory_velocity = trajectory_velocity;
request.trajectory_acceleration = trajectory_acceleration;
request.trajectory_jerk = trajectory_jerk;
request.success = success;
request.cost = cost;
request.collision_cost = collision_cost;
request.trajectory_dt = trajectory_dt;
request.selected_seed = selected_seed;
request.stream = stream;
trajopt.optimize(request);
```

All pointers are caller-owned contiguous CUDA storage. Trajectory outputs have
shape `[trajectory_steps, num_dof]`; scalar outputs have one element. A null
start-derivative pointer requests a zero boundary value, and a null seed pointer
creates straight joint-space knots. Caller-provided seed paths and paired
endpoints are optimized in parallel. Masked seeds cannot be selected, and
`selected_seed` identifies the returned trajectory and endpoint pair.
TrajOpt keeps its deployment shape fixed at `config.num_seeds`; pad shorter
candidate lists and clear the corresponding `seed_valid` entries.

When effort checking is enabled, RNEA and kinematics run concurrently on
persistent non-blocking CUDA streams. Independent pose, C-space,
world-collision, and self-collision costs then fan out onto their own streams
and rejoin before the rollout result is assembled. This stream topology is
always enabled and is retained by CUDA Graph capture.
When `check_effort_limits=true`, TrajOpt evaluates the dense trajectory with
cuRobo's RNEA forward kernel, applies the exported effort-limit C-space cost,
propagates its gradient through the RNEA VJP, and rejects trajectories outside
the exported effort bounds. Dense interpolated validation also evaluates RNEA
from its interpolated position, velocity, and acceleration, catching torque
peaks between optimization samples. The option defaults to false.

For a constrained Cartesian phase, also provide one goal pose per tool frame,
set `constrain_tool_path`, and select the free translation direction and frame.
TrajOpt then uses cuRobo's production pose-distance kernel: terminal states must
reach the full pose, while every intermediate state must preserve orientation
and remain on the requested line within the configured pose tolerances.

When `finetune_attempts` is positive, the runtime uses cuRobo's B-spline
time-scaling kernel to compute the uniform timestep required by velocity,
acceleration, and jerk limits. Each refinement starts from the best retained
knots at `required_dt * finetune_dt_scale`; a faster result replaces the prior
one only when it remains feasible. `trajectory_dt` reports the selected result.
The fixed number of passes remains allocation-free, asynchronous, and CUDA
Graph capturable.

This component is joint-space TrajOpt, not the complete motion planner: pose
goals must first be converted with IK. It reuses cuRobo's current cubic B-spline,
time-scaling, state cost, kinematics VJP, swept obstacle collision, exact
self-collision, approximate-Wolfe, and L-BFGS kernels. The referenced `Scene`
must outlive the optimizer.

### TimeOpt executable trajectories

`TrajectoryOptimizationConfig::enable_timeopt=true` optimizes timing for every retained seed
before final validation and selection by shortest valid duration. It defaults to
false. TimeOpt validation checks collision, joint bounds, velocity, acceleration,
jerk, and torque even when `check_effort_limits` is false. The timeopt shares the
optimizer's dynamics model; payload updates are visible to captured CUDA graphs.

After a successful solve, `TrajectoryOptimizer::timeopt_trajectory()` returns
borrowed `TimeOptTrajectory` buffers. Position, velocity, acceleration, and jerk
have shape `[capacity, num_dof]`; `sample_count`, `dt`, and `duration` are CUDA
scalars. Only the first `sample_count` states are executable, with uniform
`dt <= interpolation_dt` and the exact endpoint included. Candidates that exceed
the exported interpolation buffer are rejected. `timing_controls` has shape
`[num_timing_controls]` (currently 32) and belongs to the same selected seed as
the sampled states and request's optimized knots. These are the physical cubic
squared-speed controls `x(s) = (ds/dt)^2` on normalized path coordinate `s` in
`[0, 1]`, with a clamped cubic basis; they are not timestamps. The buffers remain
valid until the next solve. `spatial_controls` has shape
`[num_knots + 6, num_dof]` and includes the reconstructed start boundary and
repeated goal boundary. It defines a uniform cardinal cubic with
`num_spatial_controls - 3` spans over the same normalized coordinate. These
complete controls, together with the timing controls, reconstruct the selected
path without inferring boundaries from samples or the request's action knots.
All pointers are null and both control counts are zero when timeopt is
disabled. Failed timeopt seed selection clears both control buffers and sets
`sample_count` to zero. Access is asynchronous on the request's CUDA stream;
consumers must order their copies on that stream or synchronize before reading.
Native consumers must rebuild against this header when upgrading the runtime.

The request's knot and state outputs retain their original uniform timing for
geometry inspection. Execution with timeopt enabled uses `timeopt_trajectory()`.
Validation is sampled and does not certify continuous-time feasibility. The
exporter's `--interpolation-buffer-size` option sets the sample capacity.

Set `execution_speed_scale` in the optimizer config, or call
`set_execution_speed_scale(scale)` on the optimizer/planner before solving, to
validate the actual execution derivatives and torque at a fixed scale in
`(0,1]`. The nominal timeopt constraints must also pass. Captured graphs observe
the updated scale through a fixed device scalar. Returned controls, samples,
and durations remain nominal; execution applies velocity/acceleration/jerk
scaling by `scale`, `scale^2`, and `scale^3`. A later change to speed or payload
requires another solve before execution.

## RRT API

`curobo::robot::RrtPlanner` plans paired joint-space candidates with bounded
bidirectional RRT lanes and returns a collision-checked, visibility-shortened
waypoint chain for every successful lane in the winning stage. Query `info()`
to allocate the artifact-defined output capacity.

```cpp
#include <curobo_robot_runtime/rrt.h>

curobo::robot::RrtConfig config;
config.max_iterations = 32;
config.max_plans = 32;
config.check_effort_limits = true;
curobo::robot::RrtPlanner planner("/path/to/robot/artifact", config, &scene);

const auto info = planner.info();
curobo::robot::RrtRequest request;
request.start_position = start_position;
request.goal_position = goal_position;
request.path = path;
request.path_count = path_count;
request.success = success;
request.iterations = iterations;
request.stream = stream;
request.num_plans = num_ik_solutions;
request.plan_valid = ik_success;  // Optional uint8 [num_plans].
planner.plan(request);
planner.reset_seed();  // Optional: restart the exported Halton schedule.
```

Start and goal have matching shape `[num_plans, num_dof]`; lane `i` always
plans `start[i]` to `goal[i]`. This is the contract used by Python `plan_pose`,
where the current state is repeated across the valid IK solutions. The
caller-owned path buffer has shape
`[num_plans, max_path_nodes, num_dof]`. Success, path count, and iteration
count have shape `[num_plans]`. The search stops after a complete direct or RRT
stage produces at least one path, preserving every lane that succeeded in that
stage.

The planner reuses cuRobo's current packed-tree nearest-neighbor, append, path
reconstruction, shortcut, fused kinematics, obstacle collision, and exact
self-collision kernels. Every lane consumes distinct rows from the exported
scrambled-Halton schedule, which advances across calls until `reset_seed()` is
called. Complete direct edges, parallel growth edges, and shortcut feasibility
all use fixed workspace capacities and CUDA Graph replay, with no allocation
during `plan()`. The bounded outer search observes two connection indices on
the host each iteration, so `plan()` is synchronous on the request stream and
is not end-to-end CUDA Graph capturable. One planner must not execute concurrent
calls, and its `Scene` must outlive it.

When `check_effort_limits=true`, terminal states and every parallel steering
sample must satisfy the exported static gravity and payload torque limits.
RRT has no trajectory timing, so velocity and acceleration are zero for this
check.
Each feasibility rollout runs dynamics concurrently with kinematics and fans
world- and self-collision checks onto independent persistent streams. The
internal RRT CUDA Graphs retain that stream topology.

This component produces a collision-valid geometric path. It remains useful
independently; `MotionPlanner` below fits it and runs TrajOpt. Pose-goal IK and
projection constraints remain separate.

## Trajectory Seed Fitter API

`curobo::robot::TrajectorySeedFitter` exposes the fixed-shape behavior of
Python's `BSplineTimeOptimalSeedFitter` without requiring PyTorch in the
deployed process:

```cpp
#include <curobo_robot_runtime/trajectory_seed_fitter.h>

curobo::robot::TrajectorySeedFitterConfig config;
config.num_seeds = 4;
curobo::robot::TrajectorySeedFitter fitter(
    "/path/to/robot/artifact", config);

curobo::robot::TrajectorySeedFitterRequest request;
request.target_position = target_position;
request.start_position = start_position;
request.goal_position = goal_position;
request.fitted_knots = fitted_knots;
request.required_dt = required_dt;
request.stream = stream;
fitter.fit(request);
```

Targets have shape `[num_seeds, trajectory_steps, num_dof]`. Starts and goals
have shape `[num_seeds, num_dof]`; optional start velocity, acceleration, and
jerk use the same shape. Output knots have shape
`[num_seeds, num_knots, num_dof]`, and required timestep has shape
`[num_seeds]`. The fitter uses the production arc-length reparameterization,
acceleration-regularized geometric fit, two derivative-limit IRLS updates, and
improvement-only acceptance. Calls are asynchronous and CUDA Graph
capturable; a non-default stable stream enables the fitter's internal graph
cache.

## Motion Planning API

`curobo::robot::MotionPlanner` composes the joint-space stages. By default it
first runs straight-seeded TrajOpt. If that fails, it runs RRT, reparameterizes
the raw path by joint-space arc length, fits it to the exported cubic B-spline
representation, and runs TrajOpt again.

```cpp
#include <curobo_robot_runtime/motion_planner.h>

curobo::robot::MotionPlanningConfig config;
config.trajectory_optimization.num_seeds = 4;
config.trajectory_optimization.finetune_attempts = 2;
config.inverse_kinematics.check_effort_limits = true;
config.rrt.check_effort_limits = true;
config.trajectory_optimization.check_effort_limits = true;
config.rrt.max_iterations = 32;
config.rrt.max_plans = config.trajectory_optimization.num_seeds;
config.inverse_kinematics.cspace_rank_weight = 5.0F;
curobo::robot::MotionPlanner planner(
    "/path/to/robot/artifact", config, &scene);

const auto info = planner.info();
curobo::robot::MotionPlanningRequest request;
request.start_position = start_position;
request.start_velocity = start_velocity;          // Optional [num_dof].
request.start_acceleration = start_acceleration;  // Optional [num_dof].
request.goal_position = goal_position;
request.seed_start_positions = seed_starts;  // Optional [num_seeds, num_dof].
request.seed_goal_positions = seed_goals;  // Optional [num_seeds, num_dof].
request.seed_valid = ik_success;  // Optional uint8 [num_seeds].
request.optimized_knots = optimized_knots;
request.trajectory_position = trajectory_position;
request.trajectory_velocity = trajectory_velocity;
request.trajectory_acceleration = trajectory_acceleration;
request.trajectory_jerk = trajectory_jerk;
request.success = success;
request.cost = cost;
request.collision_cost = collision_cost;
request.used_rrt = used_rrt;
request.rrt_iterations = rrt_iterations;
request.trajectory_dt = trajectory_dt;
request.selected_seed = selected_seed;
request.stream = stream;
planner.plan(request);
```

`config.trajectory_optimization.enable_timeopt=true` enables timeopt inside
the planner's retry loop for both direct and RRT-seeded TrajOpt. After a successful
`plan()` or `plan_pose()`, `planner.timeopt_trajectory()` returns the selected
executable states and timing controls under the same lifetime and sampling
contract described above.

For `plan_grasp()`, optional caller-owned CUDA outputs
`GraspPhaseOutput::spatial_controls`, `timing_controls`, and `timeopt_duration`
are copied before the next phase overwrites the solver workspace. Only outputs
of successful phases are executable. Optional
`GraspPlanningRequest::phase_inertial_updates[0..2]` apply the appropriate
payload before approach, insertion, and lift; empty entries preserve the
current model. The last applied inertials remain active after the call.

Update payload dynamics with the same API on `InverseKinematics`,
`RrtPlanner`, `TrajectoryOptimizer`, or the composed `MotionPlanner`:

```cpp
curobo::robot::LinkInertialUpdate payload;
payload.link_name = "attached_object";
payload.mass = 1.2F;
payload.center_of_mass = std::array<float, 3>{0.0F, 0.0F, 0.08F};
payload.inertia =
    std::array<float, 6>{0.004F, 0.004F, 0.002F, 0.0F, 0.0F, 0.0F};
planner.update_link_inertial(payload);
```

The update replaces only provided fields, re-collapses fixed links into their
movable parent exactly as the Python dynamics model does, and copies the result
into stable CUDA buffers. Existing solver graphs remain valid. The call waits
for prior device work and completes synchronously. Use
`update_links_inertial()` to recompute once for multiple links.

The output shapes match `TrajectoryOptimizer`; `trajectory_dt`, `used_rrt`,
`rrt_iterations`, and `selected_seed` are one-element CUDA outputs. The planner
runs RRT on every valid paired endpoint candidate, fits each successful graph
path into its corresponding TrajOpt seed, and retains linear fallback seeds for
lanes without a graph path. Set
`attempt_direct_trajectory_optimization = false` to select the graph pipeline
immediately. Stage selection observes device status on the host, so `plan()` is
synchronous and not end-to-end CUDA Graph capturable. By default, the planner
internally captures and replays the direct and RRT-seeded TrajOpt stages when
their request addresses remain stable; set `use_cuda_graph = false` for eager
debugging.

Direct planning uses boundary-fitted B-spline seeds for every configured lane.
RRT paths use the same `TrajectorySeedFitter`, including its nonlinear IRLS
stage, before TrajOpt. TrajOpt then applies cuRobo's uniform timestep finetuning
passes. IK candidates are ranked against the current joint position using
`inverse_kinematics.cspace_rank_weight`; velocity and acceleration are then
enforced as trajectory start-boundary conditions.

For the single-pose equivalent of Python `MotionPlanner.plan_pose`, pass tool
goals directly to the composed planner:

```cpp
curobo::robot::PoseMotionPlanningRequest pose_request;
pose_request.goal_positions = goal_positions;        // [num_tool_frames, 3]
pose_request.goal_quaternions = goal_quaternions;    // wxyz
pose_request.current_position = current_position;    // [num_dof]
pose_request.current_velocity = current_velocity;    // Optional [num_dof]
pose_request.current_acceleration = current_acceleration;  // Optional [num_dof]
pose_request.selected_goal_position = selected_goal; // Optional [num_dof]
pose_request.optimized_knots = optimized_knots;
pose_request.trajectory_position = trajectory_position;
pose_request.trajectory_velocity = trajectory_velocity;
pose_request.trajectory_acceleration = trajectory_acceleration;
pose_request.trajectory_jerk = trajectory_jerk;
pose_request.success = success;
pose_request.cost = cost;
pose_request.collision_cost = collision_cost;
pose_request.trajectory_dt = trajectory_dt;
pose_request.position_error = position_error;        // Optional scalar
pose_request.orientation_error = orientation_error;  // Optional scalar
pose_request.used_rrt = used_rrt;
pose_request.rrt_iterations = rrt_iterations;
pose_request.selected_seed = selected_seed;           // Optional scalar
pose_request.planning_attempts = planning_attempts;   // Optional scalar
planner.plan_pose(pose_request);
```

`plan_pose()` requests ranked IK solutions for all TrajOpt lanes, fits the
direct seeds, and retries up to `max_attempts`. Starting at
`enable_graph_attempt`, it runs paired RRT over the valid IK solutions, fits
the successful paths, and sends them through TrajOpt. Defaults match the
single-goal Python policy: five attempts, graph planning from attempt one, one
time-optimal refinement, and `finetune_dt_scale = 0.65`. It supports one pose
per tool frame; goalsets are intentionally not part of this deployment API.

For the Python-compatible grasp pipeline, pass a goalset directly to
`plan_grasp()`:

```cpp
curobo::robot::GraspPlanningRequest grasp_request;
grasp_request.grasp_positions = grasp_positions;       // [tools, grasps, 3]
grasp_request.grasp_quaternions = grasp_quaternions;   // [tools, grasps, 4]
grasp_request.current_position = current_position;
grasp_request.current_velocity = current_velocity;      // Optional [num_dof]
grasp_request.current_acceleration = current_acceleration;  // Optional [num_dof]
grasp_request.num_grasps = num_grasps;
grasp_request.approach = approach_outputs;
grasp_request.grasp = insertion_outputs;
grasp_request.lift = lift_outputs;
grasp_request.success = success;
grasp_request.goalset_success = goalset_success;
grasp_request.goalset_index = goalset_index;
const auto status = planner.plan_grasp(grasp_request);
```

The planner disables the artifact's `grasp_contact_link_names` while selecting
the grasp, restores them for the approach, and disables them again for the
linear insertion and lift. Every exit path restores collision geometry. The
default goalset stage uses the IK solver's current-state ranking; set
`use_trajopt_for_goalset_selection` to select through the full motion pipeline.
Insertion reuses the selected grasp joint configuration without solving IK again.
Trajectory optimization, Cartesian-path validation, and collision checks still run.
Lift solves IK for its new endpoint. Both use single-pass constrained TrajOpt by default; set
`use_double_pass_trajopt_for_constrained_phases` for one timestep-refinement
pass. The C++ runtime does not expose Python's projected-RRT option; ordinary
pose planning still uses RRT as its unconstrained fallback.

Grasp planning stages dynamic inputs and outputs through fixed internal CUDA
addresses. Goal selection, approach, insertion, and lift therefore keep their
own reusable graph executables without constructing additional TrajOpt
instances or recapturing when caller buffer addresses change. Results are
copied to each phase's distinct caller-owned CUDA buffers before the shared
workspace is reused. The returned `GraspPlanningStatus` distinguishes goalset,
approach, insertion, and lift failures; the one-element device success outputs
mirror Python's phase result flags.

Selected, approach, and lift poses are prepared on the GPU. Contact-link toggles
update device sphere radii without uploading the full sphere array. Host success
reads remain at phase boundaries. TrajOpt validates and retains state/joint values
in parallel, then reduces the per-seed ranks and copies the winner in parallel
(TrajOpt artifact schema 18).

## Joint-space and Cartesian MPC

`MpcController` refines a motion plan toward a fixed joint-space goal or a Cartesian target.
It uses the existing B-spline implicit joint goal boundary, Warp cspace state/smoothness
cost, swept world collision, and self collision. Cartesian phases use a free joint
endpoint and pose/path costs in the same graph. There is no online IK, RRT,
torque cost, or continuous timestep optimization. It requires a cubic B-spline
artifact with capacity for at least four seeds.

At phase entry, `update_dt(initial_plan_sample_dt)` latches the speed-scaled
initial plan timing. Four geometrically spaced candidates start at its knot
spacing (with a 10 ms floor) and end at the larger of 100 ms or twice that spacing.
They remain fixed throughout the phase: neither a failure nor the selected winner
changes this set. The 100 ms default is not a hard input cap, and initial plans
are never clamped to a faster trajectory.
All four candidates solve in one CUDA graph. The shortest feasible candidate
wins; the result includes `selected_candidate` and its **sample** `trajectory_dt`.

Cartesian running pose cost defaults to full strength along the motion direction
(`along_direction_weight=1.0`); terminal pose cost is always full strength.
Path acceptance uses geometric perpendicular position error and orientation error
at every sample, independently of that weight. Remaining distance along the axis
is not a violation. The direction may be specified in world or goal frame, and
direction/target updates reuse the captured graph. Measured goal completion stays
with the executor; swept collision and joint-limit validation remain enabled.

`reset()` selects joint-space tracking with a pinned endpoint. `reset_cartesian()`
selects pose/path tracking with a free joint endpoint. Both replace the spatial
seed but retain the phase timing candidates. `update_cartesian_target()` changes
only the target. L-BFGS history is reset inside the graph on every solve.
Both modes share one solve graph; inactive objectives use zero GPU weights.

Full exports include weights from `curobo/content/configs/task/mpc/lbfgs_mpc.yml`.
To add those weights to an existing artifact without recompiling kernels:

```bash
python cpp_runtime/robot/tools/export_mpc_weights.py --out /path/to/artifact
```

`robot_mpc_manifest.json` schema 2 records the exact source YAML and the exported
float32 weight values instead of checksums. The binary weight layout is unchanged.

Create MPC from the planner that produced the initial trajectory. They share the
pre-FK collision spheres and layer masks, including later attachment updates.
The planner and scene must outlive MPC; serialize planning, MPC, and scene updates.

```cpp
#include <curobo_robot_runtime/mpc.h>

curobo::robot::MpcConfig config;
config.trajectory_dt = optimized_plan_sample_dt;
curobo::robot::MpcController mpc(planner, config);
mpc.reset(selected_joint_goal_cuda, initial_free_knots_cuda, stream);

// Commanded q/v/a and right-sided jerk at the scheduled integer-knot handoff.
auto candidate = mpc.step(q_handoff_cuda, v_handoff_cuda, a_handoff_cuda,
                          jerk_handoff_cuda, executed_whole_knots, stream);
// Reconstruct/validate a quintic using candidate.trajectory_dt, then submit it.
// Only after execution admission/ACK:
if (candidate.feasible && controller_applied_the_validated_replacement) {
  mpc.accept(stream);
}
```

The goal, accepted seed, timings and optimizer buffers remain GPU-resident.
Each step shifts the accepted spatial knots once by the executed whole-knot
count, then copies that seed to all four candidates. They share the commanded
q/v/a/right-sided-jerk boundary and have independent timing and optimizer state.
The winner becomes the next warm start only through `accept()`; an unaccepted
or infeasible proposal never replaces it. Pass zero additional advancement when
retrying at the same handoff. L-BFGS starts fresh for all candidates every solve.
A fixed-iteration CUDA graph (50 iterations by default) runs the batch and
validation. Goal/state/ESDF values and shared sphere updates do not recapture it.
Fractional shifting is unsupported. Four-knot replacements occur every
`4 * samples_per_knot * candidate.trajectory_dt` seconds, where
`samples_per_knot = (trajectory_steps - 1) / (num_knots + 4)` for cubic artifacts.
The new interval applies only after the replacement is acknowledged. Never
stretch an executing command's timestamps. `update_dt()` is a phase-entry call:
it changes the GPU timing grid without recapture and requires a new phase reset.
The cubic boundary consumes jerk; artifacts exported before that coefficient fix
must have their B-spline kernels rebuilt. Nonzero starting jerk must come from the
accepted command's right-sided continuation, never from a rejected solve.
`step()` synchronizes to return feasibility; its output CUDA arrays remain owned
by MPC until the next step. Warm up before starting execution.

The execution layer must validate its actual quintic replacement, match q/v/a
at handoff, and discard late or invalid results. `validate_trajectory()` checks
fixed-size CUDA q/v/a/jerk samples with a separate captured graph. It uses swept
world collision, self collision, joint limits, and a sphere-displacement sampling
guard. Native candidate validation additionally samples at at most 4 ms; the
fixed artifact validation buffer must cover the entire horizon. New native
exports reserve 2,000 samples per lane. Regenerate older artifacts to use this
capacity: it is compiled into the validation kernels, not only the manifest.
Timing grids beyond artifact dt limits or buffer capacity are rejected, never
truncated or checked at a coarser resolution to make them fit.

On rejection, validate a deceleration spline, stop/hold, and keep solving. Do not
restart braking on every rejected update. A stop is not inherently collision-free:
if an obstacle appears inside the stopping distance, neither continuing the old
plan nor blindly braking is a certified recovery. Hardware transport and task
sequencing belong to the caller; this class only solves and validates trajectories.

### Offline quintic-controller test

Use a full Franka artifact with `panda_hand`, 32 attachment spheres, at least one
voxel grid, 16 free knots, and 81 trajectory samples:

```bash
cpp_runtime/build/robot/curobo_robot_mpc_test /path/to/franka/artifact
```

The test starts with `plan_pose()`, then uses its optimized sample period with
four-knot replanning intervals, a one-knot scheduled handoff, a 200 ms quintic stop,
and execution checks at at most 4 ms spacing. Cases cover clear motion,
blocked-goal stop/recovery with cuboids and GPU-updated ESDF, permanent obstruction,
late updates, shared attachments,
goal changes, stream changes, invalid states, and reset. Cartesian cases check
small-motion recovery, lift completion, path rejection, switching back to joint-space
tracking, and updated-timestep parity with a fresh solver without recapture. The test deliberately
rejects a colliding 400 ms brake and sparsely sampled collision queries.
Printed solve times exclude first capture but include synchronous candidate
validation; they exclude execution-spline validation and transport.

`curobo_robot_mpc_shift_test` checks integer shifts, seed padding, four-lane
broadcast/validation/ranking, and geometric path acceptance under graph replay.
`curobo_robot_mpc_continuation_test` checks
q/v/a/jerk continuity through the production cubic kernel at two sample periods,
with pinned and free endpoints. Its zero-jerk negative control reproduces the
handover seam. Both tests are registered with CTest.

This is a reference-trajectory simulation, not robot dynamics or a real-time
controller test. It does not establish hardware safety. In particular, finite
swept sampling is not a continuous-time proof, some clear-path updates still
require braking at the default iteration budget, and deployment needs checks
with the actual robot geometry, controller limits, tracking error, scene age,
transport latency, and stopping distance.

## Scene API

`curobo::robot::Scene` owns fixed-capacity GPU obstacle data shared by IK,
motion-planning, and MPC APIs:

```cpp
#include <curobo_robot_runtime/scene.h>

curobo::robot::SceneConfig config;
config.max_cuboids = 32;
config.max_capsules = 32;  // Shared by capsules and spheres.
curobo::robot::Scene scene(config);

curobo::robot::Cuboid table;
table.name = "table";
table.dimensions = {1.0F, 0.8F, 0.1F};
table.pose.position = {0.0F, 0.0F, -0.05F};
const auto table_handle = scene.add_cuboid(table);

scene.set_obstacle_enabled(table_handle, false);
```

A `Scene` always represents one deployment scene. Batched IK or trajectory queries
reuse that scene; there is no `num_envs` setting or environment index. Capacities are
fixed at construction so device addresses remain stable. Adding obstacles or changing
their poses and enable flags updates those buffers synchronously and must happen outside
CUDA graph capture and while no solver is using the scene.

The scene capacities used with IK must not exceed the capacities compiled into
the robot artifact. The export defaults match `SceneConfig`; increase the
exporter's `--max-*` arguments when a deployment requires larger capacities.

Supported geometry is:

- cuboids, capsules, and local-Z cylinders;
- spheres, stored exactly as zero-length capsules;
- indexed triangle meshes with an optional Warp BVH backend; and
- dense float16 ESDF voxel grids in C order, with z varying fastest.

After adding a voxel grid, `update_voxel_grid_cuda` asynchronously copies a
contiguous FP16 CUDA ESDF into the Scene-owned stable allocation. This is the
normal live-mapping path: the source may change, while collision-query and CUDA
graph consumers keep the same Scene pointer.

The runtime accepts already decoded mesh vertices and triangle indices. File parsing and
scene-format conversion stay outside the deployment API.

## Sphere Fitting and Attached Objects

The runtime includes a deterministic voxel-volume fitter for approximating a
watertight triangle mesh with inscribed spheres. It deliberately omits MorphIt:
fitting runs infrequently when the grasp state changes, while planning continues
to use the normal fixed collision buffers.

```cpp
#include <curobo_robot_runtime/motion_planner.h>
#include <curobo_robot_runtime/sphere_fit.h>

curobo::robot::SphereFitConfig fit_config;
fit_config.num_spheres =
    planner.attachment_sphere_capacity("attached_object");
const auto fit = curobo::robot::fit_spheres_to_mesh(object_mesh, fit_config);

curobo::robot::SphereAttachment attachment;
attachment.link_name = "attached_object";
attachment.object_in_link = object_pose_in_gripper;
attachment.spheres = fit.spheres;
planner.attach_object(attachment);
ik.attach_object(attachment);

// Remove the corresponding world obstacle while it is grasped.
scene.set_obstacle_enabled(object_handle, false);

// Restore the artifact geometry when releasing the object.
planner.detach_object();
ik.detach_object();
scene.set_obstacle_enabled(object_handle, true);
```

Mesh vertices and fitted spheres use the mesh-local frame. `object_in_link`
maps those centers into the target robot-link frame and accepts `wxyz`
quaternions. The artifact must reserve sphere slots on that link through
`extra_collision_spheres`; unused slots are disabled with negative radii.
Attachment mutation is synchronous, preserves device addresses used by CUDA
Graphs, and must not overlap a solve or plan call. `MotionPlanner` updates its
IK, RRT, and TrajOpt sphere buffers. Standalone IK, RRT, and TrajOpt instances
must each receive the same attachment explicitly.

The simple voxel method requires a closed, watertight volume. Open scans and
thin shells should be repaired or converted to a closed proxy before fitting.

## Test

Static contract tests run without generated artifacts:

```bash
pytest -q cpp_runtime/robot/tests
```

The focused native scene and sphere-fitting tests are available through CTest:

```bash
ctest --test-dir cpp_runtime/build/robot --output-on-failure
```

To compare the C++ runtime with Python cuRobo on CUDA:

```bash
CUROBO_ROBOT_TEST_ARTIFACT_DIR=cpp_runtime/robot/artifacts/franka \
  CUROBO_ROBOT_TEST_LIBRARY=cpp_runtime/build/robot/libcurobo_robot_runtime.so \
  CUROBO_ROBOT_IK_TEST_EXECUTABLE=cpp_runtime/build/robot/curobo_robot_ik_test \
  CUROBO_ROBOT_TRAJOPT_TEST_EXECUTABLE=cpp_runtime/build/robot/curobo_robot_trajopt_test \
  CUROBO_ROBOT_RRT_TEST_EXECUTABLE=cpp_runtime/build/robot/curobo_robot_rrt_test \
  CUROBO_ROBOT_SEED_FITTER_TEST_EXECUTABLE=cpp_runtime/build/robot/curobo_robot_seed_fitter_test \
  CUROBO_ROBOT_MOTION_TEST_EXECUTABLE=cpp_runtime/build/robot/curobo_robot_motion_test \
  pytest -q cpp_runtime/robot/tests/test_robot_runtime.py
```

The native IK executable covers L-BFGS and LM, single-pose and goalset
selection, batched solves, caller-provided seeds, CUDA Graph replay, and Scene
rejection:

```bash
cpp_runtime/build/robot/curobo_robot_ik_test \
  cpp_runtime/robot/artifacts/franka
```

The native TrajOpt executable covers multi-seed goal selection, optional RNEA
effort limits, timestep refinement, multi-stream CUDA Graph replay, and a cuboid
that intersects the straight seed and requires optimization:

```bash
cpp_runtime/build/robot/curobo_robot_trajopt_test \
  cpp_runtime/robot/artifacts/franka
```

The native RRT executable covers collision-free direct planning, an
obstacle-avoiding search, exact endpoints, and collision-validity of every
returned edge:

```bash
cpp_runtime/build/robot/curobo_robot_rrt_test \
  cpp_runtime/robot/artifacts/franka
```

The native motion-planner executable covers direct TrajOpt, the complete RRT
fallback, and grasp goalset selection through constrained approach, insertion,
and lift. It FK-validates every sample of the linear phases:

```bash
cpp_runtime/build/robot/curobo_robot_motion_test \
  cpp_runtime/robot/artifacts/franka
```

The native seed-fitter executable covers the production geometric fit, IRLS
refinement, exact goal knot, required timestep, and CUDA Graph replay:

```bash
cpp_runtime/build/robot/curobo_robot_seed_fitter_test \
  cpp_runtime/robot/artifacts/franka
```
