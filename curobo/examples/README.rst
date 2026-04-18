cuRobo Examples
===============

Examples are split into three categories:

* **Getting Started** (``getting_started/``): Step-by-step tutorials for beginners.
  Each script is paired with a Sphinx page under ``docs/getting-started/``.
* **Guides** (``guides/``): Scripts backing the problem-oriented how-tos under
  ``docs/guides/``.
* **Reference** (``reference/``): Interactive comparison / debugging demos for specific
  features (e.g., sphere fitting, pose estimation).

Getting Started
---------------

- ``build_robot_model.py``: Build a cuRobo robot model from a URDF.
- ``forward_kinematics.py``: Compute forward kinematics from joint configurations.
- ``inverse_kinematics.py``: Solve inverse kinematics for end-effector poses.
- ``motion_planning.py``: Plan collision-free trajectories.
- ``reactive_control.py``: Run model-predictive control.
- ``humanoid_retargeting.py``: Retarget humanoid motions with self-collision.
- ``volumetric_mapping.py``: Build TSDF / ESDF maps for collision checking.

Guides
------

- ``guides/custom_optimization.py``: Minimise the Rosenbrock function with cuRobo's
  optimizers (drives ``docs/guides/custom_optimization.rst`` and
  ``docs/guides/optimization_problem.rst``).

Reference
---------

- ``reference/sphere_fit_comparison.py``: Interactive Viser viewer comparing
  sphere-fitting methods on robot link meshes.
- ``reference/robot_pose_calibration.py``: Interactive ICP / SDF pose-estimation
  demo in Viser.
