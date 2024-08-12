<!--
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# cuRobo (For SPARROWS comparison)

*CUDA Accelerated Robot Library*

Use [Discussions](https://github.com/NVlabs/curobo/discussions) for questions on using this package.

Use [Issues](https://github.com/NVlabs/curobo/issues) if you find a bug.


cuRobo's collision-free motion planner is available for commercial applications as a
MoveIt plugin: [Isaac ROS cuMotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion)

For business inquiries of this python library, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)


**Check [curobo.org](https://curobo.org) for installing and getting started with examples!**

## Installation
Run the following command to create a conda environment before running any examples:
```
conda env create -f environment.yml
```

Then run the following command to activate this environment:
```
conda activate curobo-comparison-env
```

Check your python version by running the following command. curobo only works on python<=3.10
```
python3 --version
```

Install curobo by running the following command:
```
pip install -e . --no-build-isolation
```

Test if the installation is successful by running the following command for testing:
```
python3 -m pytest .
```

## Run comparisons
You need to copy `no_filter_planning_results/` folder under `curobo/` first.
This is the result of SPARROWS, which stores the obstacles information and the trajectory information as well. 

Go to `examples/` folder and create a folder to store the comparison results:
```
mkdir comparison-results/
```

Run the following python script to test curobo on SPARROWS scenarios.
Change the value of variable `obs_num` at the front of the script to 10, 20, 40 to test scenarios with different number of obstacles.
```
python3 comparison_for_sparrows.py 
```

## Setup a simple example
You can edit the `world_file` as a yaml file.
Refer to `../src/curobo/content/configs/world/simple_scenario.yml` as an example, where you only need to define a couple of boxes (cuboids).

Edit variables `start_state_tensor` and `goal_state_tensor` for start and goal.

The results are automatically stored in a mat file called `curobo_trajectory.mat`.
You can use a simple matlab script `visualize_trajectory.m` to plot the robot motion.

## Overview

cuRobo is a CUDA accelerated library containing a suite of robotics algorithms that run significantly faster than existing implementations leveraging parallel compute. cuRobo currently provides the following algorithms: (1) forward and inverse kinematics,
(2) collision checking between robot and world, with the world represented as Cuboids, Meshes, and Depth images, (3) numerical optimization with gradient descent, L-BFGS, and MPPI, (4) geometric planning, (5) trajectory optimization, (6) motion generation that combines inverse kinematics, geometric planning, and trajectory optimization to generate global motions within 30ms.

<p align="center">
<img width="500" src="images/robot_demo.gif">
</p>


cuRobo performs trajectory optimization across many seeds in parallel to find a solution. cuRobo's trajectory optimization penalizes jerk and accelerations, encouraging smoother and shorter trajectories. Below we compare cuRobo's motion generation on the left to a BiRRT planner for the motion planning phases in a pick and place task.

<p align="center">
<img width="500" src="images/rrt_compare.gif">
</p>


## Citation

If you found this work useful, please cite the below report,

```
@misc{curobo_report23,
      title={cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation},
      author={Balakumar Sundaralingam and Siva Kumar Sastry Hari and Adam Fishman and Caelan Garrett
              and Karl Van Wyk and Valts Blukis and Alexander Millane and Helen Oleynikova and Ankur Handa
              and Fabio Ramos and Nathan Ratliff and Dieter Fox},
      year={2023},
      eprint={2310.17274},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```