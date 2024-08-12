#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
This module contains GPU accelerated kinematics leveraging CUDA. Kinematics computations enable
mapping from a robot's joint configuration to the pose of the robot's links in Cartesian space
(with reference to the robot's base link). In cuRobo, robot's geometry is approximated with spheres
and their positions are also computed as part of kinematics. This mapping is differentiable,
enabling their use in optimization problems and as part of neural networks.


..  figure:: ../images/robot_representation.png
    :width: 400px
    :align: center

    Robot representation in cuRobo is shown for the Franka Panda robot.



Kinematics in CuRobo currently supports single axis actuated joints, where the joint can be actuated
as prismatic or revolute joints. Continuous joints are approximated to revolute joints with limits
at [-6, +6] radians. Mimic joints are not supported, so convert mimic joints to independent joints.

CuRobo loads a robot's kinematic tree from :class:`~types.KinematicsTensorConfig`. This config is
generated using :class:`~cuda_robot_generator.CudaRobotGenerator`. A parser base class
:class:`~kinematics_parser.KinematicsParser` is provided to help with parsing kinematics from
standard formats. Kinematics parsing from URDF is implemented in
:class:`~urdf_kinematics_parser.UrdfKinematicsParser`. An experimental USD kinematics parser is
provided in :class:`~usd_kinematics_parser.UsdKinematicsParser`, which is missing an additional
transform between the joint origin and link origin, so this might not work for all robots. An
example workflow for setting up a robot from URDF is shown below:

.. graphviz::

   digraph {
    rankdir=LR;
   bgcolor="#808080";
   edge [color = "#FFFFFF"; fontsize=10];
   node [shape="box", style="rounded, filled", fontsize=12, color="#76b900", fontcolor="#FFFFFF"];
   "CudaRobotGenerator" [color="#FFFFFF", fontcolor="#000000"]
   "UrdfKinematicsParser" [fillcolor="#FFFFFF", fontcolor="#000000", style="box, filled", color="#000000"]
   "CudaRobotGenerator" [color="#FFFFFF", fontcolor="#000000"]
    "URDF" [fillcolor="#FFFFFF", fontcolor="#000000", style="box, filled, dashed", color="#000000"]
    "XRDF" [fillcolor="#FFFFFF", fontcolor="#000000", style="box, filled, dashed", color="#000000"]
    "cuRobo YML" [fillcolor="#FFFFFF", fontcolor="#000000", style="box, filled", color="#000000"]

   "CudaRobotGeneratorConfig" -> "CudaRobotGenerator";
   "CudaRobotGenerator" -> "UrdfKinematicsParser" [dir="both"];
   "CudaRobotGenerator" -> "CudaRobotModelConfig";
   "URDF" -> "cuRobo YML";
   "XRDF" -> "cuRobo YML" [style="dashed",label="Optional", fontcolor="#FFFFFF"];
   "cuRobo YML" -> "CudaRobotGeneratorConfig";

   }


In addition to parsing data from a kinematics file (urdf, usd), CuRobo also needs a sphere
representation of the robot that approximates the volume of the robot's links with spheres.
Several other parameters are also needed to represent kinematics in CuRobo. A tutorial on setting up a
robot is provided in :ref:`tut_robot_configuration`. cuRobo also supports using
`XRDF <https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html>`_ for representing
the additional parameters of the robot that are not available in URDF.

Once a robot configuration file is setup, you can pass this to
:class:`~cuda_robot_model.CudaRobotModelConfig` to generate an instance of kinematics configuraiton.
:class:`~cuda_robot_model.CudaRobotModel` takes this configuration and provides access to kinematics
computations.

.. note::
    :class:`~cuda_robot_model.CudaRobotModel` creates memory tensors that are used by CUDA kernels
    while :class:`~cuda_robot_model.CudaRobotModelConfig` contains only the robot kinematics
    configuration. To reduce memory overhead, you can pass one instance of
    :class:`~cuda_robot_model.CudaRobotModelConfig` to many instances of
    :class:`~cuda_robot_model.CudaRobotModel`.

"""
