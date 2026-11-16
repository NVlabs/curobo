.. _api:

Python API
============

.. automodule:: curobo
   :no-index:

.. toctree::
   :hidden:

   ../api/modules

Motion planning iteration budgets
---------------------------------

``MotionPlanner`` and ``BatchMotionPlanner`` accept a positive integer
``outer_iterations`` on each pose-planning call:

.. list-table:: Initial solve/check passes
   :header-rows: 1

   * - Method
     - Default ``outer_iterations``
     - Maximum optimizer steps with default configuration
   * - ``plan_pose``
     - 2
     - 100 per trajectory-optimization solve
   * - ``plan_pose_waypoints``
     - 4
     - 200 per trajectory-optimization solve

Each pass runs ``initial_check_interval`` optimizer steps (50 by default), checks
feasibility, and stops continuation on success. Further passes retain the iterate and
optimizer history. The budget applies independently to direct and RRT-seeded solves;
it does not limit the total work across planning retries.

``max_attempts`` limits planning retries. ``finetune_attempts`` controls additional
refinement passes, each using a separate 25-step budget by default. Changing
``outer_iterations`` on one call does not change later calls' defaults. Waypoint planning
uses four passes in both position-only and position-and-orientation tracking modes.

Python ``plan_cspace`` uses ``TrajOptSolverCfg.initial_max_iters`` (default 100) rather
than a per-call ``outer_iterations`` argument. Standalone TrajOpt uses the same cap,
with an optional per-call ``initial_iters`` override.

The C++ equivalents are ``PoseMotionPlanningRequest::outer_iterations`` (default 2)
and ``WaypointMotionPlanningRequest::outer_iterations`` (default 4). The waypoint
field supplies the budget independently of the embedded ``motion`` request. C++
recaptures its optimization graph when the pass count changes; Python reuses its
25-step optimizer graphs within each solve/check pass.

C++ joint-space planning uses ``MotionPlanningRequest::outer_iterations`` (default 2).
Standalone TrajOpt uses ``TrajectoryOptimizationConfig::max_iterations`` (default 100);
a positive ``TrajectoryOptimizationRequest::outer_iterations`` overrides that cap.

Ordinary pose and joint-space planning default to a 100-step initial budget, checking
success after 50 steps. Success proceeds to refinement; failure continues the same
optimizer state for another 50 steps. The default optimizer collision activation
distance is 0.0025 m for IK and TrajOpt in Python and for TrajOpt in C++. Native IK
preserves the exported activation distance by default (``-1``); newly generated
exports use the Python default of 0.0025 m.

Complete spline trajectories
----------------------------

``curobo.types.SplineTrajectory`` contains a complete cubic path, joint names in
control-point order, and either ``UniformTiming`` or ``TimeOptTiming``. Its
``duration`` is a device tensor in seconds. ``control_points`` has shape
``[B, C, D]`` and includes boundary support; the active path has ``C - 3`` spans.
The timing variants contain:

* ``UniformTiming.knot_dt``: seconds per spatial span, shape ``[B]``.
* ``TimeOptTiming.controls``: squared-progress-speed coefficients, shape ``[B, 32]``.
* ``TimeOptTiming.time``: integrated seconds at uniformly spaced progress nodes,
  shape ``[B, G]``. Its final entry is the duration.

A successful motion-planning or trajectory-optimization result exposes
``get_trajectory(batch_index=None, seed_index=None)``. Specify an index whenever
that result contains multiple problems or retained seeds. Failed selections raise
an error. The selected value contains one trajectory, with ``B = 1``.

``trajectory.interpolate(dt)`` evaluates positions, velocities, accelerations, and
jerks on CUDA and returns a ``JointState`` of shape ``[1, N, D]``. ``dt`` is a
positive maximum interval in seconds. The returned ``JointState.dt`` stores the
actual interval, adjusted to include both endpoints exactly. A stationary
zero-duration TimeOpt path returns one state, zero derivatives, and ``dt=None``.
Interpolation allocates output and reads duration; call it outside CUDA capture,
once when preparing a controller buffer. The spline does not cache sampled states.

``trajectory.clone()`` copies all controls, timing, and names.
``trajectory.to(device)`` transfers the complete value to CPU or CUDA; a
same-device call may return the original value. CPU storage supports controller
handoff; interpolation requires CUDA. Python copies follow PyTorch's current-stream
ordering. Coordinate separate streams explicitly before consuming or reusing buffers.
Spline joint names describe active coordinates; existing interpolated-result helpers
can also include locked joints. Existing sampled-result APIs remain available.

Grasp results expose ``get_approach_trajectory()``, ``get_grasp_trajectory()``, and
``get_lift_trajectory()``. Each successful phase retains its complete spline before
another phase reuses the solver. An omitted phase returns ``None``; an attempted
phase that failed raises an error. Overall grasp success must be checked separately.

C++ provides the same value and timing variants through
``curobo_robot_runtime/spline_trajectory.h``. ``MotionPlanner::trajectory()`` and
``TrajectoryOptimizer::trajectory()`` return the selected successful spline;
``TrajectoryTimeOpt::trajectory(seed)`` selects a TimeOpt seed. Native
``interpolate(dt, stream)`` returns owning CUDA buffers with joint names,
``sample_count``, and actual ``dt``. Planner and optimizer accessors require the
latest request's output buffers to remain alive until selection. CPU device
``-1`` supports storage and transfer;
nonnegative device values are CUDA ordinals. Native checked copies synchronize the
supplied stream. One-sample stationary output has ``dt=0``. Native grasp phase
getters return ``std::optional<SplineTrajectory>`` with the same omission and failure
rules as Python.
