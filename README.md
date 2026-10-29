<!-- SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# cuRobo

*CUDA Accelerated Robot Library*

**[Documentation](https://nvlabs.github.io/curobo) | [Paper](https://arxiv.org/abs/2603.05493)**

> [!NOTE]
> cuRoboV2 is a significant rewrite and the public API has changed from cuRobo v1.
> If you depend on the v1 API, pin to the [`v0.7.8`](https://github.com/NVlabs/curobo/tree/v0.7.8) tag.

cuRobo is a CUDA-accelerated library for robot motion generation, built on
PyTorch, CUDA, and Warp. It provides GPU-parallel algorithms for
forward/inverse kinematics, collision checking, trajectory optimization,
geometric planning, GPU-native perception, and whole-body motion generation,
scaling from single-arm manipulators to high-DoF humanoids.

Key capabilities:
- **Dynamics-aware trajectory optimization** with B-spline representation enforcing smoothness and torque limits
- **GPU-native ESDF perception** that generates dense signed distance fields from depth images, up to 10x faster than state-of-the-art
- **Scalable whole-body computation** including topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision for high-DoF robots
- **Collision-free motion generation** combining IK, geometric planning, and trajectory optimization.

## C++ Runtime

The [`cpp_runtime/`](cpp_runtime/README.md) directory contains backend-neutral
C++ APIs for robot kinematics, goalset inverse kinematics, paired-candidate
time-optimal B-spline seed fitting, joint-space trajectory optimization with
timestep refinement, paired-lane bidirectional RRT over IK solutions, composed
joint-space motion planning, single-pose IK-to-RRT-to-TrajOpt planning, goalset
grasp approach/insertion/lift, single-scene collision data, voxel sphere fitting,
grasped-object attachment, and the block-sparse mapper. Python remains the source
of truth for production cuRobo and Warp kernels, while C++-specific export glue
lives under `cpp_runtime/robot/kernels/`. Generated architecture-specific artifacts
are not committed.

## Citation

If you found this work useful, please cite cuRoboV2,

```
@misc{curobo_v2,
      title={cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots},
      author={Balakumar Sundaralingam and Adithyavairavan Murali and Stan Birchfield},
      year={2026},
      eprint={2603.05493},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Contributing

Contributions are welcome. Bugs: [open an issue](https://github.com/NVlabs/curobo/issues). General usage questions: [GitHub Discussions](https://github.com/NVlabs/curobo/discussions). For pull requests, please read [`CONTRIBUTING.md`](CONTRIBUTING.md). All commits must include a DCO sign-off (`git commit -s`).

## License

cuRobo is released under the [Apache 2.0 license](LICENSE).

The example robot assets bundled in this repository are provided under their respective licenses. See [LICENSE_ASSETS](LICENSE_ASSETS) for details.

| Asset directory | License | Local license file |
| --- | --- | --- |
| [franka_description](curobo/content/assets/robot/franka_description/) | Apache-2.0 | [LICENSE](curobo/content/assets/robot/franka_description/LICENSE) |
| [g1](curobo/content/assets/robot/g1/) | BSD-3-Clause | [LICENSE](curobo/content/assets/robot/g1/LICENSE) |
| [ur_description](curobo/content/assets/robot/ur_description/) | BSD-3-Clause | [LICENSE](curobo/content/assets/robot/ur_description/LICENSE) |

## Third-Party Software

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

Dependency licenses and attributions are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.  Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses.  You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.**
