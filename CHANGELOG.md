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

## Version 0.6.3
### Changes in default behavior
- Increased default collision cache to 50 in RobotWorld.
- Changed `CSpaceConfig.position_limit_clip` default to 0 as previous default of 0.01 can make 
default start state in examples be out of bounds. 
- MotionGen uses parallel_finetune by default. To get previous motion gen behavior, pass 
`warmup(parallel_finetune=False)` and `MotionGenPlanConfig(parallel_finetune=False)`.
- MotionGen loads Mesh Collision checker instead of Primitive by default.

### Breaking Changes
- Renamed `copy_if_not_none` to `clone_if_not_none` to be more descriptive. Now `copy_if_not_none`
will try to copy data into reference.
- Renamed `n_envs` in curobo.opt module to avoid confusion between parallel environments and 
parallel problems in optimization.
- Added more inputs to pose distance kernels. Check `curobolib/geom.py`.
- Pose cost `run_vec_weight` should now be `[0,0,0,0,0,0]` instead of `[1,1,1,1,1,1]`

### New Features
- Add function to disable and enable collision for specific links in KinematicsTensorConfig. 
- Add goal index to reacher results to return index of goal reached when goalset planning.
- Add locked joint state update api in MotionGen class.
- Add goalset warmup padding to handle varied number of goals during goalset planning and also when
calling plan_single after warmup of goalset. 
- Add new trajopt config to allow for smooth solutions at slow speeds (`velocity_scale<=0.25`). Also
add error when `velocity_scale<0.1`.
- Add experimental robot image segmentation module to enable robot removal in depth images.
- Add constrained planning mode to motion_gen.

### BugFixes & Misc.
- refactored wp.index() instances to `[]` to avoid errors in future releases of warp.
- Fix bug in gaussian transformation to ensure values are not -1 or +1.
- Fix bug in ik_solver loading ee_link_name from argument.
- Fix bug in batch_goalset planning, where pose cost was selected as GOALSET instead of 
BATCH_GOALSET.
- Added package data to also export `.so` files.
- Fixed bug in transforming link visual mesh offset when reading from urdf. 
- Fixed bug in MotionGenPlanConfig.clone() that didn't clone the state of parallel_finetune.
- Increased weighting from 1.0 to 5.0 for optimized_dt in TrajEvaluator to select shorter 
trajectories.
- Improved determinism by setting global seed for random in `graph_nx.py`.
- Added option to clear obstacles in WorldPrimitiveCollision.
- Raise error when reference of tensors change in MotionGen, IKSolver, and TrajOpt when cuda graph
is enabled.
- plan_single will get converted to plan_goalset when a plan_goalset was used to initialize cuda 
graph.
- plan_goalset will pad for extra goals when called with less number of goal than initial creation.
- Improved API documentation for Optimizer class.
- Improved benchmark timings, now within 15ms of results reported in technical report. Added
numbers to benchmark [webpage](https://curobo.org/source/getting_started/4_benchmarks.html) for 
easy reference.
- Set `use_cuda_graph` to `True` as default from `None` in `MotionGenConfig.load_from_robot_config`

### Known Bugs (WIP)
- Examples don't run in Isaac Sim 2023.1.1 due to behavior change in urdf importer.

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
- Remove mimic joints in franka panda to maintain compatibility with Isaac Sim 2023.1.0 and 2022.2.1
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
2.2 seconds 98th percentile motion time on the 2600 problems (`benchmark/curobo_benchmark.py`), where
previously it was at 3 seconds (1.36x quicker motions). This was obtained by retuning the weights and
slight reformulations of trajectory optimization. These changes have led to a slight degrade in 
planning time, 20ms slower on 4090 and 40ms on ORIN MAXN. We will address this slow down in a later
release. One way to avoid this regression is to set `finetune_dt_scale=1.05` in 
`MotionGenConfig.load_from_robot_config()`.


## Version 0.6.1

- Added changes to `examples/isaac_sim` to support Isaac Sim 2023.1.0
- Added dockerfiles and notes to run cuRobo from a docker
- Minor cleanup of examples
- Added option to generate log with UsdHelper from URDF file (check `examples/usd_example.py`)
- Fix typos in robot sphere generation tutorial (thanks @cedricgoubard)

## Version 0.6.0

- First version of CuRobo.
