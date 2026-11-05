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
