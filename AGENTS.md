<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agent Instructions

This file is the entry point for coding agents working in this repository. Keep
changes small, inspect the existing code before editing, and follow the
repository-specific rules in `.agent/`.

## Use cuRobo: prototype, validate, then deploy

Start in Python. Establish motion quality with the intended robot, scene, payload,
targets, collision margins, and joint/dynamics limits before optimizing deployment.
Use public imports from `curobo.motion_planner`, `curobo.inverse_kinematics`,
`curobo.kinematics`, `curobo.scene`, and `curobo.types`.

Use float32 CUDA tensors, meters, radians, seconds, and `wxyz` quaternions. Read joint
and tool ordering from the model, and associate joint names with `JointState` inputs.

**Construct once, warm up, and reuse.** Keep one motion planner per compatible robot/task
configuration and one standalone IK instance only when the application needs it. A
`MotionPlanner` already owns an IK solver. Reuse `planner.ik_solver` for that model instead
of constructing a second IK solver. Never build planners inside a control/request loop,
and do not maintain a cache of planners indexed by locked-joint positions.

### Set up and choose the task

Follow the [installation guide](docs/getting-started/installation.rst) for the CUDA-specific
dependencies. From the repository root, activate the environment and run a headless GPU check:

```bash
source .venv/bin/activate
python -m curobo.examples.getting_started.motion_planning --mode pose
```

The example reports planning success and writes a trajectory plot. First-use compilation
and graph capture can take longer than warmed requests; exclude them from latency measurements.
Use the [examples index](curobo/examples/README.rst) to find runnable scripts.

Choose the public API for the task:

| Task | API and starting point |
| --- | --- |
| FK and Jacobians | `Kinematics`; [tutorial][fk] |
| IK for tool poses | `InverseKinematics.solve_pose()`; [tutorial][ik] |
| Joint/tool-pose paths | `MotionPlanner.plan_cspace()` / `plan_pose()`; [tutorial][planning] |
| Cartesian waypoints | `MotionPlanner.plan_pose_waypoints()`; [tutorial][waypoints] |
| Parallel problems | [BatchMotionPlanner][batch]; allocate `max_batch_size` in `MotionPlannerCfg` |
| Reactive control | [ModelPredictiveControl][mpc-api]; [tutorial][mpc] |
| Motion retargeting | [MotionRetargeter][retarget-api]; [tutorial][retarget] |
| RGB-D mapping/features | [perception APIs][perception]; [mapping][mapping] / [features][features] |

[fk]: docs/getting-started/forward_kinematics.rst
[ik]: docs/getting-started/inverse_kinematics.rst
[planning]: docs/getting-started/motion_planning.rst
[waypoints]: docs/getting-started/cartesian_waypoints.rst
[batch]: curobo/batch_motion_planner.py
[mpc-api]: curobo/model_predictive_control.py
[mpc]: docs/getting-started/reactive_control.rst
[retarget-api]: curobo/motion_retargeter.py
[retarget]: docs/getting-started/humanoid_retargeting.rst
[perception]: curobo/perception.py
[mapping]: docs/getting-started/volumetric_mapping.rst
[features]: docs/getting-started/feature_mapping.rst

IK finds configurations; use a motion planner when the path to a configuration must also
avoid collisions. Batch planning returns per-problem success; inspect each result independently.

### Import and qualify the robot

Use an existing robot YAML when available. For a new robot, build a cuRobo configuration
from its URDF and meshes, fit collision spheres, and inspect the result:

```bash
source .venv/bin/activate
python -m curobo.examples.getting_started.build_robot_model \
  --urdf /absolute/path/robot.urdf --asset-path /absolute/path/meshes \
  --tool-frames tool_link --output /absolute/path/robot.yml \
  --compute-metrics --visualize
```

Check sphere coverage, protrusion, thin links, fingers, tool geometry, and the base's
clearance from its mounting surface. Review self-collision ignore pairs, base/tool
frames, joint ordering, locked/mimic joints, joint limits, and default joint positions.
Inspect several configurations, not only the home pose. A robot loading successfully
does not establish that its collision model is suitable for planning.
See [sphere fitting](docs/reference/sphere_fitting.rst) and the
[robot-builder example](curobo/examples/getting_started/build_robot_model.py).

### Allocate collision caches and warm the Python instance

Size `collision_cache` for the largest expected scene before construction. Keep
`use_cuda_graph=True` for normal operation. Keep debug tracing off during performance
measurement; `store_debug=True` can disable graphs. Warm the instance before the first
timed request, including task-specific branches, goalset sizes, and graph fallback.

```python
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

planner = MotionPlanner(MotionPlannerCfg.create(
    robot="franka.yml",
    scene_model="collision_table.yml",
    collision_cache={"cuboid": 8, "mesh": 2},
    graph_planner_algorithm="rrt",
    use_cuda_graph=True,
))
planner.warmup(enable_graph=True, num_warmup_iterations=5)

start = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
)
target = Pose.from_list([0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0])  # xyz, wxyz
goal = GoalToolPose.from_poses({"panda_hand": target})
result = planner.plan_pose(goal, start, max_attempts=2)
assert result is not None and bool(result.success.any())

trajectory = result.get_interpolated_plan()  # Trim padding for this single result.
assert trajectory.dt is not None
dt = float(trajectory.dt.reshape(-1)[0].item())
assert dt > 0.0
joint_names = trajectory.joint_names
positions = trajectory.position  # [..., valid_steps, num_joints], in joint_names order.
assert joint_names is not None and len(joint_names) == positions.shape[-1]
duration = (positions.shape[-2] - 1) * dt
# Keep planner alive and reuse it for subsequent requests.
```

The default start above is for the standalone example. In an application, build `JointState`
from measured positions and the associated joint names, then assign
`start = start.reorder(planner.joint_names)`. Refresh the measured state for every request.
`JointState.from_position()` assumes zero velocity and acceleration; supply measured derivatives
through `JointState` when planning from a moving state.

Express goals and obstacle poses in the same planning frame as FK: the configured robot base
frame for this fixed-base model. Transform camera/world targets into that frame first.
The goal's key must name a configured tool frame; a flange and a tool tip can have different
poses. `GoalToolPose.from_poses()` constructs the required goal tensor layout from named poses.
Normalize external quaternions and convert `xyzw` sources to `wxyz` before passing them in.

The planning call uses the default `enable_graph_attempt=1`: try direct planning first,
then enable RRT seeding on the second attempt. Set it to `0` to enable RRT from the first
attempt. This option controls graph-based path search, independently of CUDA graph replay.
Keep the policy fixed when comparing timings. See
[iteration budgets](docs/reference/api_overview.rst)
for `outer_iterations`, `max_attempts`, and refinement settings.

After success, use the trimmed trajectory's positions and available derivatives with its
stored `dt`. Read ordering from `trajectory.joint_names`, which can include locked joints;
select/reorder the controller's commanded joints explicitly. Raw `result.solution`
contains optimizer actions, and `interpolated_trajectory` can contain padding. For batches,
use each successful problem's `interpolated_last_tstep`; `get_interpolated_plan()` supports
only a single result. Plan and validate offline before submitting a trajectory to hardware.

For standalone `InverseKinematics`, warm representative `solve_pose()` calls; do not
assume it has the motion planner's `warmup()` method. Keep GPU tensors and buffers
resident where possible, and synchronize CUDA when measuring elapsed GPU work.

### Update the scene and grasped objects

Replace the world within the allocated collision capacities using the same planner:

```python
from curobo.scene import Cuboid, Scene

scene = Scene(cuboid=[
    Cuboid(name="table", dims=[4.0, 4.0, 0.2], pose=[0.0, 0.0, -0.2, 1, 0, 0, 0]),
    Cuboid(name="box", dims=[0.1, 0.1, 0.1], pose=[0.7, 0.3, 0.1, 1, 0, 0, 0]),
])
planner.update_world(scene)
planner.scene_collision_checker.update_obstacle_pose(
    "box", Pose.from_list([0.7, 0.4, 0.1, 1, 0, 0, 0])
)
planner.reset_seed()  # Clear graph-planner nodes after the obstacle pose update.
```

`update_world()` replaces the scene, so include all obstacles that should remain active.
Serialize updates against active solves. Retain collision storage and solver instances across
calls. Exceeding capacities or changing fixed tensor layouts requires explicit
resizing/reinitialization and another warmup.

For grasped objects, reserve an attachment link and enough collision sphere slots in the robot
configuration before construction. Use `planner.attachment_manager.attach_from_scene()` with
the grasp joint state, object names, attachment link, and object pose offset to fit/update
attached geometry and disable its world copy. Use `detach()` to restore the link spheres and
re-enable the world object; update the world pose to the release pose before replanning.
Inspect attachment coverage just as for robot links. See the
[attachment API](curobo/_src/collision/attachment_manager.py) for argument/frame contracts.

Collision attachment does not by itself set payload dynamics. Use `planner.update_link_inertial()`
or `update_links_inertial()` with the appropriate link-local mass, center of mass, and inertia
when evaluating payload/effort limits. These calls replace link properties; combine the link
and payload inertials before updating. Restore the unloaded inertials on release. See the
[planner API](curobo/_src/motion/motion_planner.py) for the update signatures.

### Update locked-joint values without rebuilding the planner

For a gripper with a very small motion range, keep the locked finger values fixed if
the collision spheres conservatively cover that range and preserve the required clearance.
Do not update them for every opening/closing command. Use the update below only when
collision-model inspection or collision checks show that the fixed state is inadequate,
such as false collisions near the fingers or missing coverage at another opening.

When only the values of already-locked joints change, prepare compatible kinematics
parameters and call `Kinematics.update_kinematics_config()`. It copies into existing
tensors so captured graphs retain their buffer addresses. For the planner above:

```python
import torch
from curobo.kinematics import KinematicsCfg

updated = KinematicsCfg.from_robot_yaml_file(
    "franka.yml",
    lock_joints={"panda_finger_joint1": 0.02, "panda_finger_joint2": 0.02},
)
torch.cuda.synchronize()  # Complete outstanding work before changing the shared model.
planner.kinematics.update_kinematics_config(updated.kinematics_config)
planner.reset_seed()  # Discard graph-planner nodes built for the previous locked state.
```

Replacement parameters must include the current attachments, collision masks, and payload
inertials: the update copies those tensors too. The example starts from the unmodified
Franka model; carry live model changes forward when preparing an application's replacement.

The standard planner factory shares model tensors across its IK, trajectory, and
graph-planning stages. If an application intentionally owns a separate IK instance,
update `ik.kinematics` with the same parameters too. Serialize model updates against
active solves. Prepare parameter changes outside the hot solve loop where possible.

Keep the robot topology, active/locked joint sets and ordering, tool frames, tensor
shapes, sphere capacity, device, and dtype unchanged. Changing which joints are active
is a different model/layout and can require rebuilding and warming the solver.
Changing a locked joint's numeric position is not a reason to build or cache another
motion planner. Call `planner.destroy()` when the application is finished with it.

### Diagnose a failed request

Check these in order before increasing solver budgets:

1. Check CUDA device/dtype, finite values, tensor shapes, joint/tool names, quaternion order,
   and coordinate transforms. Compare FK of the measured state with the expected tool pose.
2. Inspect the start state for self/world collisions and joint-limit violations, including
   the mounting surface, gripper, attached objects, and collision sphere fit.
3. Check that the target is reachable with the configured active/locked joints and tool frame.
   A standalone IK solve can help distinguish endpoint feasibility from path feasibility.
4. Review pose/path tolerances, collision margins, joint/dynamics limits, and payload settings.
   Keep collision checking enabled while diagnosing the request.
5. Inspect returned success masks and error/metric fields when a result is available; handle
   `None` and unsuccessful results without consuming a trajectory. Then consider seed counts,
   RRT fallback, `max_attempts`, and `outer_iterations` for difficult cases.

Use [runtime debugging options](docs/reference/runtime_configuration.rst) for targeted diagnosis.
Restore CUDA graphs and disable debug tracing before measuring steady-state performance.

### Check quality before moving to C++

Validate representative reachable and difficult targets, self/world collisions,
trajectory clearance, endpoint/path tolerances, joint velocity/acceleration/jerk,
payload/effort limits, and failure handling. Check success and the valid trajectory
horizon; do not consume unsuccessful or padded results. Measure warmed throughput
and latency only after those checks pass.

Then export the qualified configuration and build the [C++ runtime](cpp_runtime/README.md).
Persist the final locked-joint values, collision geometry, and payload settings used in the
quality checks so that native export uses the same model.
Use **2–4× faster compute** as a native deployment target, not a guarantee for every
robot/workload. Confirm quality parity and speed on matching hardware, inputs, seed and
iteration budgets, graph policy, and timing boundaries. Keep the same construct/warm/reuse
policy in C++; avoid Python conversions and host/device copies in the timed compute path.
Native exported locked-joint values and model layouts are fixed unless a public native
API explicitly supports updating them; do not assume the Python update API exists in C++.

## Rule Files

- Read `.agent/design_principles.mdc` before non-trivial code changes.
- Use `.agent/python_rules.mdc` and `.agent/package_format_rules.mdc` for Python
  code.
- Use `.agent/cpp_rules.mdc` for C++, CUDA, and Warp-adjacent native code.
- Use `.agent/tests.mdc` before adding or changing tests.
- Use `.agent/sphinx_md.mdc`, `.agent/documenter.mdc`, and
  `.agent/diataxis.mdc` for documentation changes.
- Use `.agent/file_naming.mdc` when creating or moving implementation files.

## Development Workflow

- Prefer `rg`/`rg --files` for repository search.
- Activate `.venv` before running Python commands when it exists.
- Preserve user changes in the working tree. Do not revert unrelated files.
- Keep public APIs minimal and user-focused. Do not expose implementation details
  unless they are required by the user-facing contract.
- Prefer existing local patterns and helpers over new abstractions.
- For new NVIDIA-authored files, use
  `SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
  followed by `SPDX-License-Identifier: Apache-2.0`. Preserve existing copyright
  years and upstream notices when modifying older files.
- Add focused tests for behavioral changes. Do not use `float64` in tests.
- Keep CUDA graph enabled for normal runtime, examples, and performance-sensitive
  paths. Disable CUDA graph only in tests or debugging paths where eager
  execution helps catch correctness issues.

## Python Conventions

- Add type hints to all function signatures.
- Use concise docstrings that describe behavior, inputs, outputs, and tensor
  shapes when relevant.
- For dataclasses, document fields where they are declared.
- Use logging/error helpers from `curobo/_src/util/logging.py` instead of raw
  prints or ad hoc exceptions in library code.
- Keep `_src` package initializers marker-only unless they intentionally build a
  public API aggregation layer.
- Public modules under `curobo/` should curate user-facing imports. Internal
  code should import concrete implementation modules directly.

## Naming

- Internal implementation files under `curobo/_src/` should use the
  category-first pattern from `.agent/file_naming.mdc`.
- Public API modules under `curobo/` should use descriptive, user-friendly names.

## Documentation

- Prefer Markdown with MyST syntax for new Sphinx documentation.
- Keep docs in the correct Diataxis category: tutorial, guide, concept, or
  reference.
- Write user-facing docs around the outcome and contract, not internal
  implementation details.
