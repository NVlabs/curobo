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
- ``cartesian_waypoints.py``: Follow Cartesian waypoints and preview a planned trajectory.
- ``reactive_control.py``: Run model-predictive control.
- ``humanoid_retargeting.py``: Retarget humanoid motions with self-collision.
- ``volumetric_mapping.py``: Build TSDF / ESDF maps for collision checking.
- ``feature_mapping.py``: Fuse neural image features into a TSDF map and query semantic regions.

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
- ``reference/lidar_volumetric_mapping.py``: Fuse LiDAR clouds into a TSDF with optional
  camera features, text queries, and textured mesh export.
- ``reference/live_volumetric_mapping_mpc.py``: Fuse live RealSense RGB-D frames into a
  volumetric map and run a collision-aware Franka MPC demo.
