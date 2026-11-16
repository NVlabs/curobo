.. _runtime_configuration:

Runtime Configuration
=====================

.. automodule:: curobo._src.runtime
   :members:
   :undoc-members:

Inverse dynamics in motion planning
----------------------------------

``MotionPlannerCfg.create(enable_inverse_dynamics=False)`` disables inverse-dynamics
computation and effort constraints throughout IK, path search, trajectory optimization,
dense validation, and time optimization. Collision checking and joint position,
velocity, acceleration, and jerk limits remain active. ``enable_timeopt`` remains
independent and defaults to true. The inverse-dynamics option defaults to true,
preserving existing per-stage settings.

Standalone ``TrajOptSolverCfg.create`` and ``TrajectoryTimeOptCfg`` expose the same
option. Timeopt results return ``torque=None`` when disabled. Such plans do not
provide torque feasibility guarantees and do not require effort limits or inertials
for time optimization.

The native equivalents are ``MotionPlanningConfig::enable_inverse_dynamics``,
``TrajectoryOptimizationConfig::enable_inverse_dynamics``, and
``TrajectoryTimeOptConfig::enable_inverse_dynamics``. Setting the planner option
false overrides every stage's effort-check flag. Standalone native timeopt returns
a null torque pointer and does not load the rollout/dynamics manifest when disabled.
Both Python and native execution retain CUDA graph support.
