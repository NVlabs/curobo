<!--
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# Changelog

## Version 0.7.3

### New Features
- Add start state checks for world collision, self-collision, and joint limits.
- Add finetune with dt scaling for `motion_gen.plan_single_js` to get more time optimal
trajectories in joint space planning.
- Improve joint space planning convergence, now succeeds in more planning problems with higher
accuracy.

### Changes in default behavior
- Some warp kernels are now compiled based on runtime parameters (dof), causing a slowdown in load
time for motion_gen. To avoid this slowdown, add an environment variable `CUROBO_USE_LRU_CACHE=1`
which will cache the runtime generated kernels.

### BugFixes & Misc.
- Fix bug in evaluator to account for dof maximum acceleration and jerk.
- Add unit test for different acceleration and jerk limits.
- Add a check in self-collision kernels to avoid computing over inactive threads.
- Add `link_poses` as an additional property to kinematics to be more descriptive.
- Add `g_dim` check for `int` in batched planning.
- Add `link_poses` for motion_gen.warmup() in batch planning mode.
- Add `link_poses` as input to `batch_goalset`.
- Add finetune js trajopt solver.
- Pass raw velocity, acceleration, and jerk values to dt computation function to prevent
interpolation errors from causing out of joint limit failures
- Add `finetune_js_dt_scale` with a default value > 1.0 as joint space trajectories are
time optimal in sparse obstacle environments.
- Add note on deterministic behavior to website. Use lbfgs history < 12 for deterministic
optimization results.
- Add warning when adding a mesh with the same name as in existing cache.
- Remove warmup for batch motion gen reacher isaac sim example.
- Fix python examples in getting started webpage.
- Refactor warp mesh query kernels to use a `wp.func` for signed distance queries.

## Version 0.7.2

### New Features
- Significant improvements for generating slow trajectories. Added re-timing post processing to
slow down optimized trajectories. Use `MotionGenPlanConfig.time_dilation_factor<1.0` to slow down a
planned trajectory. This is more robust than setting `velocity_scale<1.0` and also allows for
changing the speed of trajectories between planning calls
- `curobo.util.logger` adds `logger_name` as an input, enabling use of logging api with other
packages.

### Changes in default behavior
- Move `CudaRobotModelState` from `curobo.cuda_robot_model.types` to
`curobo.cuda_robot_model.cuda_robot_model`
- Activation distance for bound cost in now a ratio instead of absolute value to account for very
small range of joint limits when `velocity_scale<0.1`.
- `TrajResult` is renamed to `TrajOptResult` to be consistent with other solvers.
- Order of inputs to `get_batch_interpolated_trajectory` has changed.
- `MpcSolverConfig.load_from_robot_config` uses `world_model` instead of `world_cfg` to be
consistent with other wrappers.

### BugFixes & Misc.
- Fix bug in `MotionGen.plan_batch_env` where graph planner was being set to True. This also fixes
isaac sim example `batch_motion_gen_reacher.py`.
- Add `min_dt` as a parameter to `MotionGenConfig` and `TrajOptSolverConfig` to improve readability
and allow for having smaller `interpolation_dt`.
- Add `epsilon` to `min_dt` to make sure after time scaling, joint temporal values are not exactly
at their limits.
- Remove 0.02 offset for `max_joint_vel` and `max_joint_acc` in `TrajOptSolver`
- Bound cost now scales the cost by `1/limit_range**2` when `limit_range<1.0` to be robust to small
joint limits.
- Added documentation for `curobo.util.logger`, `curobo.wrap.reacher.motion_gen`,
`curobo.wrap.reacher.mpc`, and `curobo.wrap.reacher.trajopt`.
- When interpolation buffer is smaller than required, a new buffer is created with a warning
instead of raising an exception.
- `torch.cuda.synchronize()` now only synchronizes specified cuda device with
`torch.cuda.synchronize(device=self.tensor_args.device)`
- Added python example for MPC.

## Version 0.7.1

### New Features
- Add mimic joint parsing and optimization support. Check `ur5e_robotiq_2f_140.yml`.
- Add `finetune_dt_scale` as a parameter to `MotionGenPlanConfig` to dynamically change the
time-optimal scaling on a per problem instance.
- `MotionGen.plan_single()` will now try finetuning in a for-loop, with larger and larger dt
until convergence. This also warm starts from previous failure.
- Add `high_precision` mode to `MotionGenConfig` to support `<1mm` convergence.

### Changes in default behavior
- collision_sphere_buffer now supports having offset per link. Also, collision_sphere_buffer only
applies to world collision while self_collision_buffer applies for self collision. Previously,
self_collision_buffer was added on top of collision_sphere_buffer.
- `TrajEvaluatorConfig` cannot be initialized without dof as now per-joint jerk and acceleration
limits are used. Use `TrajEvaluatorConfig.from_basic()` to initialize similar to previous behavior.
- `finetune_dt_scale` default value is 0.9 from 0.95.

### BugFixes & Misc.
- Fix bug in `WorldVoxelCollision` where `env_query_idx` was being overwritten.
- Fix bug in `WorldVoxelCollision` where parent collision types were not getting called in some
cases.
- Change voxelization dimensions to include 1 extra voxel per dim.
- Added `seed` parameter to `IKSolverConfig`.
- Added `sampler_seed` parameter `RolloutConfig`.
- Fixed bug in `links_goal_pose` where tensor could be non contiguous.
- Improved `ik_solver` success by removing gaussian projection of seed samples.
- Added flag to sample from ik seeder instead of `rollout_fn` sampler.
- Added ik startup profiler to `benchmark/curobo_python_profile.py`.
- Reduced branching in Kinematics kernels and added mimic joint computations.
- Add init_cache to WorldVoxelCollision to create cache for Mesh and Cuboid obstacles.
- `TrajEvaluator` now uses per-joint acceleration and jerk limits.
- Fixed regression in `batch_motion_gen_reacher.py` example where robot's position was not being
set correctly.
- Switched from smooth l2 to l2 for BoundCost as that gives better convergence.
- `requires_grad` is explicitly stored in a varaible before `tensor.detach()` in warp kernel calls
as this can get set to False in some instances.
- Fix dt update in `MotionGen.plan_single_js()` where dt was not reset after finetunestep, causing
joint space planner to fail often.
- Improve joint space planner success by changing smooth l2 distance cost to l2 distance. Also,
added fallback to graph planner when linear path is not possible.
- Retuned weigths for IKSolver, now 98th percentile accuracy is 10 micrometers wtih 16 seeds
(vs 24 seeds previously).
- Switch float8 precision check from `const` to macro to avoid compile errors in older nvcc, this
fixes docker build issues for isaac sim 2023.1.0.

## Version 0.7.0
### Changes in default behavior
- Increased default collision cache to 50 in RobotWorld.
- Changed `CSpaceConfig.position_limit_clip` default to 0 as previous default of 0.01 can make
default start state in examples be out of bounds.
- MotionGen uses parallel_finetune by default. To get previous motion gen behavior, pass
`warmup(parallel_finetune=False)` and `MotionGenPlanConfig(parallel_finetune=False)`.
- MotionGen loads Mesh Collision checker instead of Primitive by default.
- UR10e and UR5e now don't have a collision sphere at tool frame for world collision checking. This
sphere is only active for self collision avoidance.
- With torch>=2.0, cuRobo will use `torch.compile` instead of `torch.jit.script` to generate fused
kernels. This can take several seconds during the first run. To enable this feature, set
environment variable `export CUROBO_TORCH_COMPILE_DISABLE=0`.

### Breaking Changes
- Renamed `copy_if_not_none` to `clone_if_not_none` to be more descriptive. Now `copy_if_not_none`
will try to copy data into reference.
- Renamed `n_envs` in curobo.opt module to avoid confusion between parallel environments and
parallel problems in optimization.
- Added more inputs to pose distance kernels. Check `curobolib/geom.py`.
- Pose cost `run_vec_weight` should now be `[0,0,0,0,0,0]` instead of `[1,1,1,1,1,1]`
- ``max_distance`` is now tensor from ``float`` and is an input to collision kernels.
- Order of inputs to ``SweptSdfMeshWarpPy`` has changed.


### New Features
- Add function to disable and enable collision for specific links in KinematicsTensorConfig.
- Add goal index to reacher results to return index of goal reached when goalset planning.
- Add locked joint state update api in MotionGen class.
- Add goalset warmup padding to handle varied number of goals during goalset planning and also when
calling plan_single after warmup of goalset.
- Add new trajopt config to allow for smooth solutions at slow speeds (`velocity_scale<=0.25`).
Also add error when `velocity_scale<0.1`.
- Add experimental robot image segmentation module to enable robot removal in depth images.
- Add constrained planning mode to motion_gen.
- Use `torch.compile` to leverage better kernel fusion in place of `torch.jit.script`.
- Significantly improved collision computation for cuboids and meshes. Mesh collision checker is
now only 2x slower than cuboid (from 5x slower). Optimization convergence is also improved.
- LBFGS kernels now support ``history <= 31`` from ``history <= 15``.
- 2x faster LBFGS kernel that allocates upto 68kb of shared memory, preventing use in CUDA devices
with compute capability ``<7.0``.
- On benchmarking Dataset, Planning time is now 42ms on average from 50ms. Higher quality solutions
are also obtained. See [benchmarks](https://curobo.org/source/getting_started/4_benchmarks.html)
for more details.
- Add ``WorldCollisionVoxel``, a new collision checking implementation that uses a voxel grid
of signed distances (SDF) to compute collision avoidance metrics. Documentation coming soon, see
``benchmark/curobo_voxel_benchmark.py`` for an example.
- Add API for ESDF computation from world representations, see
``WorldCollision.get_esdf_in_bounding_box()``.
- Add partial support for isaac sim 2023.1.1. Most examples run for UR robots. `Franka Panda` is
unstable.

### BugFixes & Misc.
- refactored wp.index() instances to `[]` to avoid errors in future releases of warp.
- Fix bug in gaussian transformation to ensure values are not -1 or +1.
- Fix bug in ik_solver loading ee_link_name from argument.
- Fix bug in batch_goalset planning, where pose cost was selected as GOALSET instead of
BATCH_GOALSET.
- Added package data to also export `.so` files.
- Fixed bug in transforming link visual mesh offset when reading from urdf.
- Fixed bug in MotionGenPlanConfig.clone() that didn't clone the state of parallel_finetune.
- Increased weighting from 1.0 to 10.0 for optimized_dt in TrajEvaluator to select shorter
trajectories.
- Improved determinism by setting global seed for random in `graph_nx.py`.
- Added option to clear obstacles in WorldPrimitiveCollision.
- Raise error when reference of tensors change in MotionGen, IKSolver, and TrajOpt when cuda graph
is enabled.
- plan_single will get converted to plan_goalset when a plan_goalset was used to initialize cuda
graph.
- plan_goalset will pad for extra goals when called with less number of goal than initial creation.
- Improved API documentation for Optimizer class.
- Set `use_cuda_graph` to `True` as default from `None` in `MotionGenConfig.load_from_robot_config`
- Add batched mode to robot image segmentation, supports single robot multiple camera and batch
robot batch camera.
- Add `log_warn` import to `arm_reacher.py`
- Remove negative radius check in self collision kernel to allow for self collision checking with
spheres of negative radius.
- Added `conftest.py` to disable `torch.compile` for tests.
- Added UR5e robot with robotiq gripper (2f-140) with improved sphere model.
- Fix bug in aarch64.dockerfile where curobo was cloned to wrong path.
- Fix bug in aarch64.dockerfile where python was used instead of python3.
- Remove unused variables in kernels.
- Added ``pybind11`` as a dependency as some pytorch dockers for Jetson do not have this installed.
- Fix incorrect dimensions in ``MotionGenResult.success`` in ``MotionGen.plan_batch()`` when
trajectory optimization fails.
- Added unit tests for collision checking functions.
- Fix bug in linear interpolation which was not reading the new ``optimized_dt`` to interpolate
velocity, acceleration, and jerk.
- Remove torch.jit.script wrapper for lbfgs as it causes TorchScript error if history is different
between trajopt and finetune_trajopt.


### Known Bugs (WIP)
- `Franka Panda` robot loading from urdf in isaac sim 2023.1.1 is unstable.

## Version 0.6.2
### New Features
- Added support for actuated axis to be negative (i.e., urdf joints with `<axis xyz="0 -1 0"/>` are
now natively supported).
- Improved gradient calculation to account for terminal state. Trajectory optimization can reach
within 1mm of accuracy (median across 2600 problems at 0.017mm).
- Improved estimation of previous positions based on start velocity and acceleration. This enables
Trajectory optimization to optimize from non-zero start velocity and accelerations.
- Added graph planner and finetuning step to joint space planning (motion_gen.plan_single_js). This
improves success and motion quality when planning to reach joint space targets.
- Added finetuning across many seeds in motion_gen, improving success rate and motion quality.
- Add urdf support to usd helper to export optimization steps as animated usd files for debugging
motion generation. Check `examples/usd_examples.py` for an example.
- Retuned weights for IK and Trajectory optimization. This (+ other fixes) significantly improves
pose reaching accuracy, IK accuracy improves by 100x (98th percentile < 10 micrometers) and motion
generation median at 0.017mm (with). IK now solves most problems with 24 seeds (vs 30 seeds prev.).
Run `benchmark/ik_benchmark.py` to get the latest results.
- Added `external_asset_path` to robot configuration to help in loading urdf and meshes from an
external directory.


### BugFixes & Misc.
- Update nvblox wrappers to work with v0.0.5 without segfaults. Significantly improves stability.
- Remove mimic joints in franka panda to maintain compatibility with Isaac Sim 2023.1.0 and
2022.2.1
- Cleanup docker scripts. Use `build_docker.sh` instead of `build_dev_docker.sh`. Added isaac sim
development docker.
- Fixed bug in backward kinematics kernel, helped improve IK and TO pose reaching accuracy..
- Changed `panda_finger_joint2` from `<axis xyz="0 1 0"/>`
 to `<axis xyz="0 -1 0"/>` in `franka_panda.urdf` to match real robot urdf as cuRobo now supports
 negative axis.
- Changed benchmarking scripts to use lock joint state of [0.025,0.025] for mpinets dataset.
- Added scaling of mesh to Mesh.get_trimesh_mesh() to help in debugging mesh world.
- Improved stability and accuracy of MPPI for MPC.
- Added NaN checking in STOMP covariance computation to account for cases when cholesky decomp
fails.
- Added ground truth collision check validation in `benchmarks/curobo_nvblox_benchmark.py`.

### Performance Regressions
- cuRobo now generates significantly shorter paths then previous version. E.g., cuRobo obtains
2.2 seconds 98th percentile motion time on the 2600 problems (`benchmark/curobo_benchmark.py`),
where previously it was at 3 seconds (1.36x quicker motions). This was obtained by retuning the
weights and slight reformulations of trajectory optimization. These changes have led to a slight
degrade in planning time, 20ms slower on 4090 and 40ms on ORIN MAXN. We will address this slow down
in a later release. One way to avoid this regression is to set `finetune_dt_scale=1.05` in
`MotionGenConfig.load_from_robot_config()`.


## Version 0.6.1

- Added changes to `examples/isaac_sim` to support Isaac Sim 2023.1.0
- Added dockerfiles and notes to run cuRobo from a docker
- Minor cleanup of examples
- Added option to generate log with UsdHelper from URDF file (check `examples/usd_example.py`)
- Fix typos in robot sphere generation tutorial (thanks @cedricgoubard)

## Version 0.6.0

- First version of CuRobo.
