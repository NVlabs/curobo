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
