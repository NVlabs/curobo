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
This module contains :meth:`MotionGen` class that provides a high-level interface for motion
generation. Motion Generation can take goals either as joint configurations
:meth:`MotionGen.plan_single_js` or as Cartesian poses :meth:`MotionGen.plan_single`. When Cartesian
pose is given as target, inverse kinematics is first done to generate seeds for trajectory
optimization. Motion generation fallback to using a graph planner when linear interpolated
trajectory optimization seeds are not successful. Reaching one Cartesian pose in a goalset is also
supported using :meth:`MotionGen.plan_goalset`. Batched planning in the same world environment (
:meth:`MotionGen.plan_batch`) and different world environments (:meth:`MotionGen.plan_batch_env`)
is also provided.


.. raw:: html

    <p>
    <video autoplay="True" loop="True" muted="True" preload="auto" width="100%"><source src="../videos/ur10_real_timer.mp4" type="video/mp4"></video>
    </p>


A motion generation request can be configured using :meth:`MotionGenPlanConfig`. The result
of motion generation is returned as a :meth:`MotionGenResult`. A minimal example is availble at
:ref:`python_motion_gen_example`.

"""

from __future__ import annotations

# Standard Library
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler
import warp as wp

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Cuboid, Obstacle, WorldConfig
from curobo.graph.graph_base import GraphConfig, GraphPlanBase, GraphResult
from curobo.graph.prm import PRMStar
from curobo.rollout.arm_reacher import ArmReacher
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.types.tensor import T_BDOF, T_DOF, T_BValue_float
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import tensor_repeat_seeds
from curobo.util.trajectory import InterpolateType, get_batch_interpolated_trajectory
from curobo.util.warp import init_warp
from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.evaluator import TrajEvaluator, TrajEvaluatorConfig
from curobo.wrap.reacher.ik_solver import IKResult, IKSolver, IKSolverConfig
from curobo.wrap.reacher.trajopt import TrajOptResult, TrajOptSolver, TrajOptSolverConfig
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType


@dataclass
class MotionGenConfig:
    """Configuration dataclass for creating a motion generation instance."""

    #: number of IK seeds to run per query problem.
    ik_seeds: int

    #: number of graph planning seeds to use per query problem.
    graph_seeds: int

    #: number of trajectory optimization seeds to use per query problem.
    trajopt_seeds: int

    #: number of seeds to run trajectory optimization per trajopt_seed.
    noisy_trajopt_seeds: int

    #: number of IK seeds to use for batched queries.
    batch_ik_seeds: int

    #: number of trajectory optimization seeds to use for batched queries.
    batch_trajopt_seeds: int

    #: instance of robot configuration shared across all solvers.
    robot_cfg: RobotConfig

    #: instance of IK solver to use for motion generation.
    ik_solver: IKSolver

    #: instance of graph planner to use.
    graph_planner: GraphPlanBase

    #: instance of trajectory optimization solver to use for reaching Cartesian poses.
    trajopt_solver: TrajOptSolver

    #: instance of trajectory optimization solver to use for reaching joint space targets.
    js_trajopt_solver: TrajOptSolver

    #: instance of trajectory optimization solver for final fine tuning for joint space targets.
    finetune_js_trajopt_solver: TrajOptSolver
    #: instance of trajectory optimization solver for final fine tuning.
    finetune_trajopt_solver: TrajOptSolver

    #: interpolation to use for getting dense waypoints from optimized solution.
    interpolation_type: InterpolateType

    #: maximum number of steps to interpolate
    interpolation_steps: int

    #: instance of world collision checker.
    world_coll_checker: WorldCollision

    #: device to load motion generation.
    tensor_args: TensorDeviceType

    #: number of IK iterations to run for initializing trajectory optimization
    partial_ik_iters: int

    #: number of iterations to run trajectory optimization when seeded from a graph plan.
    graph_trajopt_iters: Optional[int] = None

    #: store debugging information in MotionGenResult
    store_debug_in_result: bool = False

    #: interpolation dt to use for output trajectory.
    interpolation_dt: float = 0.01

    #: scale initial dt by this value to finetune trajectory optimization.
    finetune_dt_scale: float = 0.9

    #: record compute ops as cuda graphs and replay recorded graphs for upto 10x faster execution.
    use_cuda_graph: bool = True

    #: After 100 iterations of trajectory optimization, a new dt is computed that pushes the
    #: velocity, acceleration, or jerk limits to the maximum. This new dt is then used with a
    #: reduction :attr:`MotionGenPlanConfig.finetune_dt_scale` to run 200+ iterations of trajectory
    #: optimization. If trajectory optimization fails with the new dt, the new dt is increased and
    #: tried again until :attr:`MotionGenPlanConfig.finetune_attempts`.
    optimize_dt: bool = True

    @staticmethod
    def load_from_robot_config(
        robot_cfg: Union[Union[str, Dict], RobotConfig],
        world_model: Optional[Union[Union[str, Dict], WorldConfig]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        num_ik_seeds: int = 32,
        num_graph_seeds: int = 4,
        num_trajopt_seeds: int = 4,
        num_batch_ik_seeds: int = 32,
        num_batch_trajopt_seeds: int = 1,
        num_trajopt_noisy_seeds: int = 1,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        cspace_threshold: float = 0.05,
        world_coll_checker=None,
        base_cfg_file: str = "base_cfg.yml",
        particle_ik_file: str = "particle_ik.yml",
        gradient_ik_file: str = "gradient_ik_autotune.yml",
        graph_file: str = "graph.yml",
        particle_trajopt_file: str = "particle_trajopt.yml",
        gradient_trajopt_file: str = "gradient_trajopt.yml",
        finetune_trajopt_file: Optional[str] = None,
        trajopt_tsteps: int = 32,
        interpolation_steps: int = 5000,
        interpolation_dt: float = 0.02,
        interpolation_type: InterpolateType = InterpolateType.LINEAR_CUDA,
        use_cuda_graph: bool = True,
        self_collision_check: bool = True,
        self_collision_opt: bool = True,
        grad_trajopt_iters: Optional[int] = None,
        trajopt_seed_ratio: Dict[str, int] = {"linear": 1.0, "bias": 0.0},
        ik_opt_iters: Optional[int] = None,
        ik_particle_opt: bool = True,
        collision_checker_type: Optional[CollisionCheckerType] = CollisionCheckerType.MESH,
        sync_cuda_time: Optional[bool] = None,
        trajopt_particle_opt: bool = True,
        traj_evaluator_config: Optional[TrajEvaluatorConfig] = None,
        traj_evaluator: Optional[TrajEvaluator] = None,
        minimize_jerk: bool = True,
        filter_robot_command: bool = False,
        use_gradient_descent: bool = False,
        collision_cache: Optional[Dict[str, int]] = None,
        n_collision_envs: Optional[int] = None,
        ee_link_name: Optional[str] = None,
        use_es_ik: Optional[bool] = None,
        use_es_trajopt: Optional[bool] = None,
        es_ik_learning_rate: float = 1.0,
        es_trajopt_learning_rate: float = 1.0,
        use_ik_fixed_samples: Optional[bool] = None,
        use_trajopt_fixed_samples: Optional[bool] = None,
        evaluate_interpolated_trajectory: bool = True,
        partial_ik_iters: int = 2,
        fixed_iters_trajopt: Optional[bool] = None,
        store_ik_debug: bool = False,
        store_trajopt_debug: bool = False,
        graph_trajopt_iters: Optional[int] = None,
        collision_max_outside_distance: Optional[float] = None,
        collision_activation_distance: Optional[float] = None,
        trajopt_dt: Optional[float] = None,
        js_trajopt_dt: Optional[float] = None,
        js_trajopt_tsteps: Optional[int] = None,
        trim_steps: Optional[List[int]] = None,
        store_debug_in_result: bool = False,
        finetune_trajopt_iters: Optional[int] = None,
        smooth_weight: List[float] = None,
        finetune_smooth_weight: Optional[List[float]] = None,
        state_finite_difference_mode: Optional[str] = None,
        finetune_dt_scale: float = 0.9,
        minimum_trajectory_dt: Optional[float] = None,
        maximum_trajectory_time: Optional[float] = None,
        maximum_trajectory_dt: Optional[float] = None,
        velocity_scale: Optional[Union[List[float], float]] = None,
        acceleration_scale: Optional[Union[List[float], float]] = None,
        jerk_scale: Optional[Union[List[float], float]] = None,
        optimize_dt: bool = True,
        project_pose_to_goal_frame: bool = True,
        ik_seed: int = 1531,
        graph_seed: int = 1531,
        high_precision: bool = False,
        use_cuda_graph_trajopt_metrics: bool = False,
        trajopt_fix_terminal_action: bool = True,
        trajopt_js_fix_terminal_action: bool = True,
    ):
        """Create a motion generation configuration from robot and world configuration.

        Args:
            robot_cfg: Robot configuration to use for motion generation. This can be a path to a
                yaml file, a dictionary, or an instance of :class:`RobotConfig`. See
                :ref:`available_robot_list` for a list of available robots. You can also create a
                a configuration file for your robot using :ref:`tut_robot_configuration`.
            world_model: World configuration to use for motion generation. This can be a path to a
                yaml file, a dictionary, or an instance of :class:`WorldConfig`. See
                :ref:`world_collision` for more details.
            tensor_args: Numerical precision and compute device to use for motion generation.
            num_ik_seeds: Number of seeds to use for solving inverse kinematics. Default of 32 is
                found to be a good number for most cases. In sparse environments, a lower number of
                16 can also be used.
            num_graph_seeds: Number of seeds to use for graph planner per problem query. When graph
                planning is used to generate seeds for trajectory optimization, graph planner will
                attempt to find collision-free paths from the start state to the many inverse
                kinematics solutions.
            num_trajopt_seeds: Number of seeds to use for trajectory optimization per problem
                query. Default of 4 is found to be a good number for most cases. Increasing this
                will increase memory usage.
            num_batch_ik_seeds: Number of seeds to use for inverse kinematics during batched
                planning. Default of 32 is found to be a good number for most cases.
            num_batch_trajopt_seeds: Number of seeds to use for trajectory optimization during
                batched planning. Using more than 1 will disable graph planning for batched
                planning.
            num_trajopt_noisy_seeds: Number of augmented trajectories to use per trajectory seed.
                The augmentation is done by adding random noise to the trajectory. This
                augmentation has not been found to be useful and it's recommended to keep this to
                1. The noisy seeds can also be used in conjunction with the trajopt_seed_ratio to
                generate seeds that go through a bias point.
            position_threshold: Position threshold in meters between reached position and target
                position used to measure success.
            rotation_threshold: Rotation threshold between reached orientation and target
                orientation used to measure success. The metric is q^T * q, where q is the
                quaternion difference between target and reached orientation. The metric is not
                easy to interpret and a future release will provide a more intuitive metric. For
                now, use 0.05 as a good value.
            cspace_threshold: Joint space threshold in radians for revolute joints and meters for
                linear joints between reached joint configuration and target joint configuration
                used to measure success. Default of 0.05 has been found to be a good value for most
                cases.
            world_coll_checker: Instance of world collision checker to use for motion generation.
                Leaving this to None will create a new instance of world collision checker using
                the provided attr:`world_model`.
            base_cfg_file: Base configuration file containing convergence and constraint criteria
                to measure success.
            particle_ik_file: Optimizer configuration file to use for particle-based optimization
                during inverse kinematics.
            gradient_ik_file: Optimizer configuration file to use for gradient-based optimization
                during inverse kinematics.
            graph_file: Configuration file to use for graph planner.
            particle_trajopt_file: Optimizer configuration file to use for particle-based
                optimization during trajectory optimization.
            gradient_trajopt_file: Optimizer configuration file to use for gradient-based
                optimization during trajectory optimization.
            finetune_trajopt_file: Optimizer configuration file to use for finetuning trajectory
                optimization.
            trajopt_tsteps: Number of waypoints to use for trajectory optimization. Default of 32
                is found to be a good number for most cases.
            interpolation_steps: Buffer size to use for storing interpolated trajectory. Default of
                5000 is found to be a good number for most cases.
            interpolation_dt: Time step in seconds to use for generating interpolated trajectory
                from optimized trajectory. Change this if you want to generate a trajectory with
                a fixed timestep between waypoints.
            interpolation_type: Interpolation type to use for generating dense waypoints from
                optimized trajectory. Default of
                :py:attr:`curobo.util.trajectory.InterpolateType.LINEAR_CUDA` is found to be a
                good choice for most cases. Other suitable options for real robot execution are
                :py:attr:`curobo.util.trajectory.InterpolateType.QUINTIC` and
                :py:attr:`curobo.util.trajectory.InterpolateType.CUBIC`.
            use_cuda_graph: Record compute ops as cuda graphs and replay recorded graphs where
                implemented. This can speed up execution by upto 10x. Default of True is
                recommended. Enabling this will prevent changing problem type or batch size
                after the first call to the solver.
            self_collision_check: Enable self collision checks for generated motions. Default of
                True is recommended. Set this to False to debug planning failures. Setting this to
                False will also set self_collision_opt to False.
            self_collision_opt: Enable self collision cost during optimization (IK, TrajOpt).
                Default of True is recommended.
            grad_trajopt_iters: Number of iterations to run trajectory optimization.
            trajopt_seed_ratio: Ratio of linear and bias seeds to use for trajectory optimization.
                Linear seed will generate a linear interpolated trajectory from start state
                to IK solutions. Bias seed will add a mid-waypoint through the retract
                configuration. Default of 1.0 linear and 0.0 bias is recommended. This can be
                changed to 0.5 linear and 0.5 bias, along with changing trajopt_noisy_seeds to 2.
            ik_opt_iters: Number of iterations to run inverse kinematics.
            ik_particle_opt: Enable particle-based optimization during inverse kinematics. Default
                of True is recommended as particle-based optimization moves the random seeds to
                a regions of local minima.
            collision_checker_type: Type of collision checker to use for motion generation. Default
                of CollisionCheckerType.MESH supports world represented by Cuboids and Meshes. See
                :ref:`world_collision` for more details.
            sync_cuda_time: Synchronize with host using :py:func:`torch.cuda.synchronize` before
                measuring compute time.
            trajopt_particle_opt: Enable particle-based optimization during trajectory
                optimization. Default of True is recommended as particle-based optimization moves
                the interpolated seeds away from bad local minima.
            traj_evaluator_config: Configuration for trajectory evaluator. Default of None will
                create a new instance of TrajEvaluatorConfig. After trajectory optimization across
                many seeds, the best trajectory is selected based on this configuration. This
                evaluator also checks if the optimized dt is within
                :py:attr:`curobo.wrap.reacher.evaluator.TrajEvaluatorConfig.max_dt`. This check is
                needed to measure smoothness of the optimized trajectory as bad trajectories can
                have very high dt to fit within velocity, acceleration, and jerk limits.
            traj_evaluator: Instance of trajectory evaluator to use for trajectory optimization.
                Default of None will create a new instance of TrajEvaluator. In case you want to
                use a custom evaluator, you can create a child instance of TrajEvaluator and
                pass it.
            minimize_jerk: Minimize jerk as regularization during trajectory optimizaiton.
            filter_robot_command: Filter generated trajectory to remove finite difference
                artifacts. Default of True is recommended.
            use_gradient_descent: Use gradient descent instead of L-BFGS for trajectory
                optimization.
            collision_cache: Cache of obstacles to create to load obstacles between planning calls.
                An example: ``{"obb": 10, "mesh": 10}``, to create a cache of 10 cuboids and 10
                meshes.
            n_collision_envs: Number of collision environments to create for batched planning
                across different environments. Only used for :py:meth:`MotionGen.plan_batch_env`
                and :py:meth:`MotionGen.plan_batch_env_goalset`.
            ee_link_name: End effector link/frame to use for reaching Cartesian poses. Default of
                None will use the end effector link from the robot configuration. This cannot
                be changed after creating the robot configuration.
            use_es_ik: Use evolutionary strategy for as the particle-based optimizer for inverse
                kinematics. Default of None will use MPPI as the optimization algorithm. ES is not
                recommended as it's unstable and provided optimization parameters were not tuned.
            use_es_trajopt: Use evolutionary strategy as the particle-based optimizer for
                trajectory optimization. Default of None will use MPPI as the optimization
                algorithm. ES is not recommended as it's unstable and provided optimization
                parameters were not tuned.
            es_ik_learning_rate: Learning rate to use for evolutionary strategy in IK.
            es_trajopt_learning_rate: Learning rate to use for evolutionary strategy in TrajOpt.
            use_ik_fixed_samples: Use fixed samples of noise during particle-based optimization
                of IK. Default of None will use the setting from the optimizer configuration file
                (``particle_ik.yml``).
            use_trajopt_fixed_samples: Use fixed samples of noise during particle-based
                optimization of trajectory optimization. Default of None will use the setting from
                the optimizer configuration file (``particle_trajopt.yml``).
            evaluate_interpolated_trajectory: Evaluate interpolated trajectory after optimization.
                Default of True is recommended to ensure the optimized trajectory is not passing
                through very thin obstacles.
            partial_ik_iters: Number of iterations of L-BFGS to run inverse kinematics when
                only partial IK is needed.
            fixed_iters_trajopt: Use fixed number of iterations of L-BFGS for trajectory
                optimization. Default of None will use the setting from the optimizer
                configuration. In most cases, fixed iterations of solvers are run as current
                solvers treat constraints as costs and there is no guarantee that the constraints
                will be satisfied. Instead of checking constraints between iterations of a solver
                and exiting, it's computationally cheaper to run a fixed number of iterations. In
                addition, running fixed iterations of solvers is more robust to outlier problems.
            store_ik_debug: Store debugging information such as values of optimization variables
                at every iteration in IK result. Setting this to True will set
                :attr:`use_cuda_graph` to False.
            store_trajopt_debug: Store debugging information such as values of optimization
                variables in TrajOpt result. Setting this to True will set :attr:`use_cuda_graph`
                to False.
            graph_trajopt_iters: Number of iterations to run trajectory optimization when seeded
                from a graph plan. Default of None will use the same number of iterations as
                linear seeded trajectory optimization. This can be set to a higher value of 200
                in case where trajectories obtained are not of requird quality.
            collision_max_outside_distance: Maximum distance to check for collision outside a
                obstacle. Increasing this value will slow down collision checks with Meshes as
                closest point queries will be run up to this distance outside an obstacle.
            collision_activation_distance: Distance in meters to activate collision cost. A good
                value to start with is 0.01 meters. Increase the distance if the robot needs to
                stay further away from obstacles.
            trajopt_dt: Time step in seconds to use for trajectory optimization. A good value to
                start with is 0.15 seconds. This value is used to compute velocity, acceleration,
                and jerk values for waypoints through finite difference.
            js_trajopt_dt: Time step in seconds to use for trajectory optimization when reaching
                joint space targets. A good value to start with is 0.15 seconds. This value is used
                to compute velocity, acceleration, and jerk values for waypoints through finite
                difference.
            js_trajopt_tsteps: Number of waypoints to use for trajectory optimization when reaching
                joint space targets. Default of None will use the same number of waypoints as
                Cartesian trajectory optimization.
            trim_steps: Trim waypoints from optimized trajectory. The optimized trajectory will
                contain the start state at index 0 and have the last two waypoints be the same
                as T-2 as trajectory optimization implicitly optimizes for zero acceleration and
                velocity at the last waypoint. An example: ``[1,-2]`` will trim the first waypoint
                and last 3 waypoints from the optimized trajectory.
            store_debug_in_result: Store debugging information in MotionGenResult. This value is
                set to True if either store_ik_debug or store_trajopt_debug is set to True.
            finetune_trajopt_iters: Number of iterations to run trajectory optimization for
                finetuning after an initial collision-free trajectory is obtained.
            smooth_weight: Override smooth weight for trajectory optimization. It's not recommended
                to set this value for most cases.
            finetune_smooth_weight: Override smooth weight for finetuning trajectory optimization.
                It's not recommended to set this value for most cases.
            state_finite_difference_mode: Finite difference mode to use for computing velocity,
                acceleration, and jerk values. Default of None will use the setting from the
                optimizer configuration file. The default finite difference method is a five
                point stencil to compute the derivatives as this is accurate and provides
                faster convergence compared to backward or central difference methods.
            finetune_dt_scale: Scale initial estimated dt by this value to finetune trajectory
                optimization. This is deprecated and will be removed in future releases. Use
                :py:attr:`MotionGenPlanConfig.finetune_dt_scale` instead.
            minimum_trajectory_dt: Minimum time step in seconds allowed for trajectory
                optimization.
            maximum_trajectory_time: Maximum time in seconds allowed for trajectory optimization.
            maximum_trajectory_dt: Maximum time step in seconds allowed for trajectory
                optimization.
            velocity_scale: Scale velocity limits by this value. Default of None will not scale
                the velocity limits. To generate slower trajectories, use
                :py:attr:`MotionGenPlanConfig.time_dilation_factor` < 1.0 instead. Changing this
                value is not recommended as it changes the scale of cost terms and they would
                require retuning.
            acceleration_scale: Scale acceleration limits by this value. Default of None will not
                scale the acceleration limits. To generate slower trajectories, use
                :py:attr:`MotionGenPlanConfig.time_dilation_factor` < 1.0 instead. Changing this
                value is not recommended as it changes the scale of cost terms and they would
                require retuning.
            jerk_scale: Scale jerk limits by this value. Default of None will not scale the jerk
                limits. To generate slower trajectories, use
                :py:attr:`MotionGenPlanConfig.time_dilation_factor` < 1.0 instead. Changing this
                value is not recommended as it changes the scale of cost terms and they would
                require retuning.
            optimize_dt: Optimize dt during trajectory optimization. Default of True is
                recommended to find time-optimal trajectories. Setting this to False will use the
                provided :attr:`trajopt_dt` for trajectory optimization. Setting to False is
                required when optimizing from a non-static start state.
            project_pose_to_goal_frame: Project pose to goal frame when calculating distance
                between reached and goal pose. Use this to constrain motion to specific axes
                either in the global frame or the goal frame.
            ik_seed: Random seed to use for inverse kinematics.
            graph_seed: Random seed to use for graph planner.
            high_precision: Use high precision settings for motion generation. This will increase
                the number of iterations for optimization solvers and reduce the thresholds for
                position to 1mm and rotation to 0.025. Default of False is recommended for most
                cases as standard motion generation settings reach within 0.5mm on most problems.
            use_cuda_graph_trajopt_metrics: Flag to enable cuda_graph when evaluating interpolated
                trajectories after trajectory optimization. If interpolation_buffer is smaller
                than interpolated trajectory, then the buffers will be re-created. This can cause
                existing cuda graph to be invalid.
            trajopt_fix_terminal_action: Flag to disable optimizing for final state. When true,
                the final state is unchanged from initial seed. When false, terminal state can
                change based on cost. Setting to False will lead to worse accuracy at target
                pose (>0.1mm). Setting to True can achieve < 0.01mm accuracy.
            trajopt_js_fix_terminal_action: Flag to disable optimizing for final state for joint
                space target planning. When true, the final state is unchanged from initial seed.
                When false, terminal state can change based on cost. Setting to False will lead to
                worse accuracy at target joint configuration.

        Returns:
            MotionGenConfig: Instance of motion generation configuration.
        """
        if position_threshold <= 0.001:
            high_precision = True
        if high_precision:
            finetune_trajopt_iters = (
                300 if finetune_trajopt_iters is None else max(300, finetune_trajopt_iters)
            )
            if grad_trajopt_iters is None:
                grad_trajopt_iters = 200
            grad_trajopt_iters = max(200, grad_trajopt_iters)
            position_threshold = min(position_threshold, 0.001)
            rotation_threshold = min(rotation_threshold, 0.025)
            cspace_threshold = min(cspace_threshold, 0.01)
        init_warp(tensor_args=tensor_args)
        if js_trajopt_tsteps is not None:
            log_warn("js_trajopt_tsteps is deprecated, use trajopt_tsteps instead.")
            trajopt_tsteps = js_trajopt_tsteps
        if trajopt_tsteps is not None:
            js_trajopt_tsteps = trajopt_tsteps
        if velocity_scale is not None and isinstance(velocity_scale, float):
            log_warn(
                "To slow down trajectories, use MotionGenPlanConfig.time_dilation_factor"
                + " instead of velocity_scale"
            )
            velocity_scale = [velocity_scale]

        if acceleration_scale is not None and isinstance(acceleration_scale, float):
            log_warn(
                "To slow down trajectories, use MotionGenPlanConfig.time_dilation_factor"
                + " instead of acceleration_scale"
            )
            acceleration_scale = [acceleration_scale]
        if jerk_scale is not None and isinstance(jerk_scale, float):
            jerk_scale = [jerk_scale]

        if store_ik_debug or store_trajopt_debug:
            store_debug_in_result = True

        if (
            velocity_scale is not None
            and min(velocity_scale) < 0.1
            and finetune_trajopt_file is None
            and maximum_trajectory_dt is None
        ):
            log_error(
                "velocity scale<0.1 requires a user determined maximum_trajectory_dt as"
                + " default scaling will likely fail. A good value to start with would be 30"
                + " seconds"
            )

        if maximum_trajectory_dt is None:
            maximum_trajectory_dt = 0.15
        maximum_trajectory_dt_acc = maximum_trajectory_dt
        maximum_trajectory_dt_vel = maximum_trajectory_dt
        if (
            acceleration_scale is not None
            and min(acceleration_scale) < 1.0
            and maximum_trajectory_dt <= 0.2
        ):
            maximum_trajectory_dt_acc = (
                np.sqrt(1.0 / min(acceleration_scale)) * maximum_trajectory_dt * 3
            )
        if (
            velocity_scale is not None
            and min(velocity_scale) < 1.0
            and maximum_trajectory_dt <= 0.2
        ):
            maximum_trajectory_dt_vel = (1.0 / min(velocity_scale)) * maximum_trajectory_dt * 3
        maximum_trajectory_dt = max(maximum_trajectory_dt_acc, maximum_trajectory_dt_vel)
        if maximum_trajectory_dt is not None:
            if trajopt_dt is None:
                trajopt_dt = maximum_trajectory_dt
            if js_trajopt_dt is None:
                js_trajopt_dt = maximum_trajectory_dt
        if acceleration_scale is not None and min(acceleration_scale) < 0.5:
            fixed_iters_trajopt = True
        if velocity_scale is not None and min(velocity_scale) < 0.5:
            fixed_iters_trajopt = True

        if (
            velocity_scale is not None
            and min(velocity_scale) <= 0.25
            and finetune_trajopt_file is None
        ):
            finetune_trajopt_file = "finetune_trajopt_slow.yml"

        if isinstance(robot_cfg, str):
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))["robot_cfg"]
        elif isinstance(robot_cfg, Dict) and "robot_cfg" in robot_cfg.keys():
            robot_cfg = robot_cfg["robot_cfg"]
        if isinstance(robot_cfg, RobotConfig):
            if (
                ee_link_name is not None
                and robot_cfg.kinematics.kinematics_config.ee_link != ee_link_name
            ):
                log_error("ee link cannot be changed after creating RobotConfig")
            if (
                acceleration_scale is not None
                and torch.max(robot_cfg.kinematics.kinematics_config.cspace.acceleration_scale)
                != acceleration_scale
            ):
                log_error("acceleration_scale cannot be changed after creating RobotConfig")
            if (
                velocity_scale is not None
                and torch.max(robot_cfg.kinematics.kinematics_config.cspace.velocity_scale)
                != velocity_scale
            ):
                log_error("velocity cannot be changed after creating RobotConfig")
        else:
            if ee_link_name is not None:
                robot_cfg["kinematics"]["ee_link"] = ee_link_name
            if jerk_scale is not None:
                robot_cfg["kinematics"]["cspace"]["jerk_scale"] = jerk_scale
            if acceleration_scale is not None:
                robot_cfg["kinematics"]["cspace"]["acceleration_scale"] = acceleration_scale
            if velocity_scale is not None:
                robot_cfg["kinematics"]["cspace"]["velocity_scale"] = velocity_scale

            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        if minimum_trajectory_dt is None:
            minimum_trajectory_dt = interpolation_dt
        elif minimum_trajectory_dt < interpolation_dt:
            log_error("minimum_trajectory_dt cannot be lower than interpolation_dt")
        if traj_evaluator_config is None:
            if maximum_trajectory_dt is not None:
                max_dt = maximum_trajectory_dt
            if maximum_trajectory_time is not None:
                max_dt = maximum_trajectory_time / trajopt_tsteps
            if acceleration_scale is not None:
                max_dt = max_dt * (1.0 / np.sqrt(min(acceleration_scale)))
            traj_evaluator_config = TrajEvaluatorConfig.from_basic(
                min_dt=minimum_trajectory_dt, max_dt=max_dt, dof=robot_cfg.kinematics.dof
            )
        traj_evaluator_config.max_acc = robot_cfg.kinematics.get_joint_limits().acceleration[1]

        traj_evaluator_config.max_jerk = robot_cfg.kinematics.get_joint_limits().jerk[1]

        if isinstance(world_model, str):
            world_model = load_yaml(join_path(get_world_configs_path(), world_model))

        base_config_data = load_yaml(join_path(get_task_configs_path(), base_cfg_file))
        if collision_cache is not None:
            base_config_data["world_collision_checker_cfg"]["cache"] = collision_cache
        if n_collision_envs is not None:
            base_config_data["world_collision_checker_cfg"]["n_envs"] = n_collision_envs
        if collision_max_outside_distance is not None:
            if collision_max_outside_distance < 0.0:
                log_error("collision_max_outside_distance cannot be negative")
            base_config_data["world_collision_checker_cfg"][
                "max_distance"
            ] = collision_max_outside_distance
        if collision_checker_type is not None:
            # log_info("updating collision checker type to ",collision_checker_type)
            base_config_data["world_collision_checker_cfg"]["checker_type"] = collision_checker_type
        if not self_collision_check:
            base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0

        if world_coll_checker is None and world_model is not None:
            world_cfg = WorldCollisionConfig.load_from_dict(
                base_config_data["world_collision_checker_cfg"], world_model, tensor_args
            )
            world_coll_checker = create_collision_checker(world_cfg)
        ik_solver_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_model,
            tensor_args,
            num_ik_seeds,
            position_threshold,
            rotation_threshold,
            world_coll_checker,
            base_config_data,
            particle_ik_file,
            gradient_ik_file,
            use_cuda_graph=use_cuda_graph,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_opt,
            grad_iters=ik_opt_iters,
            use_particle_opt=ik_particle_opt,
            sync_cuda_time=sync_cuda_time,
            use_es=use_es_ik,
            es_learning_rate=es_ik_learning_rate,
            use_fixed_samples=use_ik_fixed_samples,
            store_debug=store_ik_debug,
            collision_activation_distance=collision_activation_distance,
            project_pose_to_goal_frame=project_pose_to_goal_frame,
            seed=ik_seed,
            high_precision=high_precision,
        )

        ik_solver = IKSolver(ik_solver_cfg)

        graph_cfg = GraphConfig.load_from_robot_config(
            robot_cfg,
            world_model,
            tensor_args,
            world_coll_checker,
            base_config_data,
            graph_file,
            use_cuda_graph=use_cuda_graph,
            seed=graph_seed,
        )
        graph_cfg.interpolation_dt = interpolation_dt
        graph_cfg.interpolation_steps = interpolation_steps

        graph_planner = PRMStar(graph_cfg)

        trajopt_cfg = TrajOptSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_model,
            tensor_args=tensor_args,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            world_coll_checker=world_coll_checker,
            base_cfg_file=base_config_data,
            particle_file=particle_trajopt_file,
            gradient_file=gradient_trajopt_file,
            traj_tsteps=trajopt_tsteps,
            interpolation_type=interpolation_type,
            interpolation_steps=interpolation_steps,
            use_cuda_graph=use_cuda_graph,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_opt,
            grad_trajopt_iters=grad_trajopt_iters,
            num_seeds=num_trajopt_noisy_seeds,
            seed_ratio=trajopt_seed_ratio,
            interpolation_dt=interpolation_dt,
            use_particle_opt=trajopt_particle_opt,
            traj_evaluator_config=traj_evaluator_config,
            traj_evaluator=traj_evaluator,
            use_gradient_descent=use_gradient_descent,
            use_es=use_es_trajopt,
            es_learning_rate=es_trajopt_learning_rate,
            use_fixed_samples=use_trajopt_fixed_samples,
            evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
            fixed_iters=fixed_iters_trajopt,
            store_debug=store_trajopt_debug,
            collision_activation_distance=collision_activation_distance,
            trajopt_dt=trajopt_dt,
            store_debug_in_result=store_debug_in_result,
            smooth_weight=smooth_weight,
            cspace_threshold=cspace_threshold,
            state_finite_difference_mode=state_finite_difference_mode,
            filter_robot_command=filter_robot_command,
            minimize_jerk=minimize_jerk,
            optimize_dt=optimize_dt,
            project_pose_to_goal_frame=project_pose_to_goal_frame,
            use_cuda_graph_metrics=use_cuda_graph_trajopt_metrics,
            fix_terminal_action=trajopt_fix_terminal_action,
        )
        trajopt_solver = TrajOptSolver(trajopt_cfg)

        js_trajopt_cfg = TrajOptSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_model,
            tensor_args=tensor_args,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            world_coll_checker=world_coll_checker,
            base_cfg_file=base_config_data,
            particle_file=particle_trajopt_file,
            gradient_file=gradient_trajopt_file,
            traj_tsteps=js_trajopt_tsteps,
            interpolation_type=interpolation_type,
            interpolation_steps=interpolation_steps,
            use_cuda_graph=use_cuda_graph,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_opt,
            grad_trajopt_iters=grad_trajopt_iters,
            interpolation_dt=interpolation_dt,
            use_particle_opt=trajopt_particle_opt,
            traj_evaluator_config=traj_evaluator_config,
            traj_evaluator=traj_evaluator,
            use_gradient_descent=use_gradient_descent,
            use_es=use_es_trajopt,
            es_learning_rate=es_trajopt_learning_rate,
            use_fixed_samples=use_trajopt_fixed_samples,
            evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
            fixed_iters=fixed_iters_trajopt,
            store_debug=store_trajopt_debug,
            collision_activation_distance=collision_activation_distance,
            trajopt_dt=js_trajopt_dt,
            store_debug_in_result=store_debug_in_result,
            smooth_weight=smooth_weight,
            cspace_threshold=cspace_threshold,
            state_finite_difference_mode=state_finite_difference_mode,
            minimize_jerk=minimize_jerk,
            filter_robot_command=filter_robot_command,
            optimize_dt=optimize_dt,
            num_seeds=num_trajopt_noisy_seeds,
            use_cuda_graph_metrics=use_cuda_graph_trajopt_metrics,
            fix_terminal_action=trajopt_js_fix_terminal_action,
        )
        js_trajopt_solver = TrajOptSolver(js_trajopt_cfg)

        if finetune_trajopt_file is None:
            finetune_trajopt_file = "finetune_trajopt.yml"

        finetune_trajopt_cfg = TrajOptSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_model,
            tensor_args=tensor_args,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            world_coll_checker=world_coll_checker,
            base_cfg_file=base_config_data,
            particle_file=particle_trajopt_file,
            gradient_file=finetune_trajopt_file,
            traj_tsteps=trajopt_tsteps,
            interpolation_type=interpolation_type,
            interpolation_steps=interpolation_steps,
            use_cuda_graph=use_cuda_graph,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_opt,
            grad_trajopt_iters=finetune_trajopt_iters,
            interpolation_dt=interpolation_dt,
            use_particle_opt=False,
            traj_evaluator_config=traj_evaluator_config,
            traj_evaluator=traj_evaluator,
            evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
            fixed_iters=fixed_iters_trajopt,
            store_debug=store_trajopt_debug,
            collision_activation_distance=collision_activation_distance,
            trajopt_dt=trajopt_dt,
            store_debug_in_result=store_debug_in_result,
            smooth_weight=finetune_smooth_weight,
            cspace_threshold=cspace_threshold,
            state_finite_difference_mode=state_finite_difference_mode,
            minimize_jerk=minimize_jerk,
            trim_steps=trim_steps,
            use_gradient_descent=use_gradient_descent,
            filter_robot_command=filter_robot_command,
            optimize_dt=optimize_dt,
            project_pose_to_goal_frame=project_pose_to_goal_frame,
            use_cuda_graph_metrics=use_cuda_graph_trajopt_metrics,
            fix_terminal_action=trajopt_fix_terminal_action,
        )

        finetune_trajopt_solver = TrajOptSolver(finetune_trajopt_cfg)

        finetune_js_trajopt_cfg = TrajOptSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_model,
            tensor_args=tensor_args,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            world_coll_checker=world_coll_checker,
            base_cfg_file=base_config_data,
            particle_file=particle_trajopt_file,
            gradient_file=finetune_trajopt_file,
            traj_tsteps=js_trajopt_tsteps,
            interpolation_type=interpolation_type,
            interpolation_steps=interpolation_steps,
            use_cuda_graph=use_cuda_graph,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_opt,
            grad_trajopt_iters=grad_trajopt_iters,
            interpolation_dt=interpolation_dt,
            use_particle_opt=False,
            traj_evaluator_config=traj_evaluator_config,
            traj_evaluator=traj_evaluator,
            use_gradient_descent=use_gradient_descent,
            use_es=use_es_trajopt,
            es_learning_rate=es_trajopt_learning_rate,
            use_fixed_samples=use_trajopt_fixed_samples,
            evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
            fixed_iters=fixed_iters_trajopt,
            store_debug=store_trajopt_debug,
            collision_activation_distance=collision_activation_distance,
            trajopt_dt=js_trajopt_dt,
            store_debug_in_result=store_debug_in_result,
            smooth_weight=smooth_weight,
            cspace_threshold=cspace_threshold,
            state_finite_difference_mode=state_finite_difference_mode,
            minimize_jerk=minimize_jerk,
            filter_robot_command=filter_robot_command,
            optimize_dt=optimize_dt,
            num_seeds=num_trajopt_noisy_seeds,
            use_cuda_graph_metrics=use_cuda_graph_trajopt_metrics,
            fix_terminal_action=trajopt_js_fix_terminal_action,
        )
        finetune_js_trajopt_solver = TrajOptSolver(finetune_js_trajopt_cfg)

        if graph_trajopt_iters is not None:
            graph_trajopt_iters = math.ceil(
                graph_trajopt_iters / finetune_trajopt_solver.solver.newton_optimizer.inner_iters
            )
        else:
            graph_trajopt_iters = finetune_trajopt_solver.solver.newton_optimizer.outer_iters + 2
        return MotionGenConfig(
            num_ik_seeds,
            num_graph_seeds,
            num_trajopt_seeds,
            num_trajopt_noisy_seeds,
            num_batch_ik_seeds,
            num_batch_trajopt_seeds,
            robot_cfg,
            ik_solver,
            graph_planner,
            trajopt_solver=trajopt_solver,
            js_trajopt_solver=js_trajopt_solver,
            finetune_js_trajopt_solver=finetune_js_trajopt_solver,
            finetune_trajopt_solver=finetune_trajopt_solver,
            interpolation_type=interpolation_type,
            interpolation_steps=interpolation_steps,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            partial_ik_iters=partial_ik_iters,
            graph_trajopt_iters=graph_trajopt_iters,
            store_debug_in_result=store_debug_in_result,
            interpolation_dt=interpolation_dt,
            finetune_dt_scale=finetune_dt_scale,
            use_cuda_graph=use_cuda_graph,
            optimize_dt=optimize_dt,
        )


@dataclass
class MotionGenPlanConfig:
    """Configuration for querying motion generation."""

    #: Use graph planner to generate collision-free seed for trajectory optimization.
    enable_graph: bool = False

    #: Enable trajectory optimization.
    enable_opt: bool = True

    #: Use neural network IK seed for solving inverse kinematics. Not implemented.
    use_nn_ik_seed: bool = False

    #: Run trajectory optimization only if graph planner is successful.
    need_graph_success: bool = False

    #: Maximum number of attempts allowed to solve the motion generation problem.
    max_attempts: int = 60

    #: Maximum time in seconds allowed to solve the motion generation problem.
    timeout: float = 10.0

    #: Number of failed attempts at which to fallback to a graph planner for obtaining trajectory
    #: seeds.
    enable_graph_attempt: Optional[int] = 3

    #: Number of failed attempts at which to disable graph planner. This has not been found to be
    #: useful.
    disable_graph_attempt: Optional[int] = None

    #: Number of IK attempts allowed before returning a failure. Set this to a low value (5) to
    #: save compute time when an unreachable goal is given.
    ik_fail_return: Optional[int] = None

    #: Full IK solving takes 10% of the planning time. Setting this to True will run only 50
    #: iterations of IK instead of 100 and then run trajecrtory optimization without checking if
    #: IK is successful. This can reduce the planning time by 5% but generated solutions can
    #: have larger motion time and path length. Leave this to False for most cases.
    partial_ik_opt: bool = False

    #: Number of seeds to use for solving inverse kinematics. Chanigng this will cause the existing
    #: CUDAGraphs to be invalidated. Instead, set num_ik_seeds when creating
    #: :meth:`MotionGenConfig`.
    num_ik_seeds: Optional[int] = None

    #: Number of seeds to use for graph planner. We found 4 to be a good number for most cases. The
    #: default value is set when creating :meth:`MotionGenConfig` so leave this to None.
    num_graph_seeds: Optional[int] = None

    #: Number of seeds to use for trajectory optimization. We found 12 to be a good number for most
    #: cases. The default value is set when creating :meth:`MotionGenConfig` so leave this to None.
    num_trajopt_seeds: Optional[int] = None

    #: Ratio of successful motion generation queries to total queries in batch planning mode. Motion
    #: generation is queries upto :attr:`MotionGenPlanConfig.max_attempts` until this ratio is met.
    success_ratio: float = 1

    #: Return a failure if the query is invalid. Set this to False to debug a query that is invalid.
    fail_on_invalid_query: bool = True

    #: use start config as regularization for IK instead of
    #: :meth:`curobo.types.robot.RobotConfig.kinematics.kinematics_config.retract_config`
    use_start_state_as_retract: bool = True

    #: Use a custom pose cost metric for trajectory optimization. This is useful for adding
    #: additional constraints to motion generation, such as constraining the end-effector's motion
    #: to a plane or a line or hold orientation while moving. This is also useful for only reaching
    #: partial pose (e.g., only position). See :meth:`curobo.rollout.cost.pose_cost.PoseCostMetric`
    #: for more details.
    pose_cost_metric: Optional[PoseCostMetric] = None

    #: Run finetuning trajectory optimization after running 100 iterations of
    #: trajectory optimization. This will provide shorter and smoother trajectories. When
    #: :attr:`MotionGenConfig.optimize_dt` is True, this flag will also scale the trajectory
    #: optimization by a new dt. Leave this to True for most cases. If you are not interested in
    #: finding time-optimal solutions and only want to use motion generation as a feasibility check,
    #: set this to False. Note that when set to False, the resulting trajectory is only guaranteed
    #: to be collision-free and within joint limits. When False, it's not guaranteed to be smooth
    #: and might not execute on a real robot.
    enable_finetune_trajopt: bool = True

    #: Run finetuning trajectory optimization across all trajectory optimization seeds. If this is
    #: set to False, then only 1 successful seed per query is selected and finetuned. When False,
    #: we have observed that the resulting trajectory is not as optimal as when set to True.
    parallel_finetune: bool = True

    #: Scale dt by this value before running finetuning trajectory optimization. This enables
    #: trajectory optimization to find shorter paths and also account for smoothness over variable
    #: length trajectories. This is only used when :attr:`MotionGenConfig.optimize_dt` is True.
    finetune_dt_scale: Optional[float] = 0.85

    #: Number of attempts to run finetuning trajectory optimization. Every attempt will increase the
    #: :attr:`MotionGenPlanConfig.finetune_dt_scale` by
    #: :attr:`MotionGenPlanConfig.finetune_dt_decay` as a path couldn't be found with the previous
    #: smaller dt.
    finetune_attempts: int = 5

    #: Decay factor used to increase :attr:`MotionGenPlanConfig.finetune_dt_scale` when optimization
    #: fails to find a solution. This is only used when :attr:`MotionGenConfig.optimize_dt` is True.
    finetune_dt_decay: float = 1.025

    #: Slow down optimized trajectory by re-timing with a dilation factor. This is useful to
    #: execute trajectories at a slower speed for debugging. Use this to generate slower
    #: trajectories instead of reducing :attr:`MotionGenConfig.velocity_scale` or
    #: :attr:`MotionGenConfig.acceleration_scale` as those parameters will require re-tuning
    #: of the cost terms while :attr:`MotionGenPlanConfig.time_dilation_factor` will only
    #: post-process the trajectory.
    time_dilation_factor: Optional[float] = None

    #: Check if the start state is valid before runnning any steps in motion generation. This will
    #: check for joint limits, self-collision, and collision with the world.
    check_start_validity: bool = True

    #: Finetune dt scale for joint space planning.
    finetune_js_dt_scale: Optional[float] = 1.1

    def __post_init__(self):
        """Post initialization checks."""
        if not self.enable_opt and not self.enable_graph:
            log_error("Graph search and Optimization are both disabled, enable one")

    def clone(self) -> MotionGenPlanConfig:
        """Clone the current planning configuration."""
        return MotionGenPlanConfig(
            enable_graph=self.enable_graph,
            enable_opt=self.enable_opt,
            use_nn_ik_seed=self.use_nn_ik_seed,
            need_graph_success=self.need_graph_success,
            max_attempts=self.max_attempts,
            timeout=self.timeout,
            enable_graph_attempt=self.enable_graph_attempt,
            disable_graph_attempt=self.disable_graph_attempt,
            ik_fail_return=self.ik_fail_return,
            partial_ik_opt=self.partial_ik_opt,
            num_ik_seeds=self.num_ik_seeds,
            num_graph_seeds=self.num_graph_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            success_ratio=self.success_ratio,
            fail_on_invalid_query=self.fail_on_invalid_query,
            enable_finetune_trajopt=self.enable_finetune_trajopt,
            parallel_finetune=self.parallel_finetune,
            use_start_state_as_retract=self.use_start_state_as_retract,
            pose_cost_metric=(
                None if self.pose_cost_metric is None else self.pose_cost_metric.clone()
            ),
            finetune_dt_scale=self.finetune_dt_scale,
            finetune_attempts=self.finetune_attempts,
            time_dilation_factor=self.time_dilation_factor,
            finetune_js_dt_scale=self.finetune_js_dt_scale,
        )


class MotionGenStatus(Enum):
    """Status of motion generation query."""

    #: Inverse kinematics failed to find a solution.
    IK_FAIL = "IK Fail"

    #: Graph planner failed to find a solution.
    GRAPH_FAIL = "Graph Fail"

    #: Trajectory optimization failed to find a solution.
    TRAJOPT_FAIL = "TrajOpt Fail"

    #: Finetune Trajectory optimization failed to find a solution.
    FINETUNE_TRAJOPT_FAIL = "Finetune TrajOpt Fail"

    #: Optimized dt is greater than the maximum allowed dt. Set maximum_trajectory_dt to a higher
    #: value.
    DT_EXCEPTION = "dt exceeded maximum allowed trajectory dt"

    #: Invalid query was given. The start state is either out of joint limits, in collision with
    #: world, or in self-collision. In the future, this will also check for reachability of goal
    #: pose/ joint target in joint limits.
    INVALID_QUERY = "Invalid Query"

    #: Invalid start state was given. Unknown reason.
    INVALID_START_STATE_UNKNOWN_ISSUE = "Invalid Start State, unknown issue"

    #: Invalid start state was given. The start state is in world collision.
    INVALID_START_STATE_WORLD_COLLISION = "Start state is colliding with world"

    #: Invalid start state was given. The start state is in self-collision.
    INVALID_START_STATE_SELF_COLLISION = "Start state is in self-collision"

    #: Invalid start state was given. The start state is out of joint limits.
    INVALID_START_STATE_JOINT_LIMITS = "Start state is out of joint limits"

    #: Invalid partial pose target.
    INVALID_PARTIAL_POSE_COST_METRIC = "Invalid partial pose metric"
    #: Motion generation query was successful.
    SUCCESS = "Success"

    #: Motion generation was not attempted.
    NOT_ATTEMPTED = "Not Attempted"


@dataclass
class MotionGenResult:
    """Result obtained from motion generation."""

    #: success tensor with index referring to the batch index.
    success: Optional[torch.Tensor] = None

    #: returns true if the start state is not satisfying constraints (e.g., in collision)
    valid_query: bool = True

    #: optimized trajectory
    optimized_plan: Optional[JointState] = None

    #: dt between steps in the optimized plan
    optimized_dt: Optional[T_BValue_float] = None

    #: Cartesian position error at final timestep, returning l2 distance.
    position_error: Optional[T_BValue_float] = None

    #: Cartesian rotation error at final timestep, computed as q^(-1) * q_g
    rotation_error: Optional[T_BValue_float] = None

    #: Error in joint configuration, when planning in joint space, returning l2 distance.
    cspace_error: Optional[T_BValue_float] = None

    #: seconds taken  in the optimizer for solving the motion generation problem.
    solve_time: float = 0.0

    #: seconds taken to solve IK.
    ik_time: float = 0.0

    #: seconds taken to find graph plan.
    graph_time: float = 0.0

    #: seconds taken in trajectory optimization.
    trajopt_time: float = 0.0

    #: seconds to run finetune trajectory optimization.
    finetune_time: float = 0.0

    #: sum of ik_time, graph_time, and trajopt_time. This also includes any processing between
    #: calling the different modules.
    total_time: float = 0.0

    #: interpolated solution, useful for visualization.
    interpolated_plan: Optional[JointState] = None

    #: dt between steps in interpolated plan
    interpolation_dt: float = 0.02

    #: last timestep in interpolated plan per batch index. Only used for batched queries
    path_buffer_last_tstep: Optional[List[int]] = None

    #: Debug information
    debug_info: Any = None

    #: status of motion generation query.
    status: Optional[Union[MotionGenStatus, str]] = None

    #: number of attempts used before returning a solution.
    attempts: int = 1

    #: number of trajectory optimization attempts used per attempt.
    trajopt_attempts: int = 0

    #: returns true when a graph plan was used to seed trajectory optimization.
    used_graph: bool = False

    #: stores graph plan.
    graph_plan: Optional[JointState] = None

    #: stores the index of the goal pose reached when planning for a goalset.
    goalset_index: Optional[torch.Tensor] = None

    def clone(self):
        """Clone the current result."""
        m = MotionGenResult(
            self.success.clone(),
            valid_query=self.valid_query,
            optimized_plan=self.optimized_plan.clone() if self.optimized_plan is not None else None,
            optimized_dt=self.optimized_dt.clone() if self.optimized_dt is not None else None,
            position_error=self.position_error.clone() if self.position_error is not None else None,
            rotation_error=self.rotation_error.clone() if self.rotation_error is not None else None,
            cspace_error=self.cspace_error.clone() if self.cspace_error is not None else None,
            solve_time=self.solve_time,
            ik_time=self.ik_time,
            graph_time=self.graph_time,
            trajopt_time=self.trajopt_time,
            total_time=self.total_time,
            graph_plan=self.graph_plan.clone() if self.graph_plan is not None else None,
            debug_info=self.debug_info,
            status=self.status,
            attempts=self.attempts,
            trajopt_attempts=self.trajopt_attempts,
            used_graph=self.used_graph,
            path_buffer_last_tstep=self.path_buffer_last_tstep,
            interpolated_plan=(
                self.interpolated_plan.clone() if self.interpolated_plan is not None else None
            ),
            interpolation_dt=self.interpolation_dt,
            goalset_index=self.goalset_index.clone() if self.goalset_index is not None else None,
        )
        return m

    def copy_idx(self, idx: torch.Tensor, source_result: MotionGenResult):
        """Copy data from source result to current result at index.

        Args:
            idx: index to copy data at.
            source_result: source result to copy data from.

        Returns:
            MotionGenResult: copied result.
        """
        idx = idx.to(dtype=torch.long)
        # copy data from source result:
        self.success[idx] = source_result.success[idx]

        self.optimized_plan = self._check_none_and_copy_idx(
            self.optimized_plan, source_result.optimized_plan, idx
        )
        self.interpolated_plan = self._check_none_and_copy_idx(
            self.interpolated_plan, source_result.interpolated_plan, idx
        )

        self.position_error = self._check_none_and_copy_idx(
            self.position_error, source_result.position_error, idx
        )

        self.rotation_error = self._check_none_and_copy_idx(
            self.rotation_error, source_result.rotation_error, idx
        )
        self.cspace_error = self._check_none_and_copy_idx(
            self.cspace_error, source_result.cspace_error, idx
        )

        self.goalset_index = self._check_none_and_copy_idx(
            self.goalset_index, source_result.goalset_index, idx
        )
        # NOTE: graph plan will have different shape based on success.
        # self.graph_plan = self._check_none_and_copy_idx(
        #    self.graph_plan, source_result.graph_plan, idx
        # )

        idx_list = idx.cpu().tolist()
        if source_result.path_buffer_last_tstep is not None:
            if self.path_buffer_last_tstep is None:
                self.path_buffer_last_tstep = [0 for i in range(len(self.success))]

            for i in idx_list:
                self.path_buffer_last_tstep[i] = source_result.path_buffer_last_tstep[i]

        return self

    def get_paths(self) -> List[JointState]:
        """Get interpolated trajectories from the result. Use for batched queries.

        This will return unsuccessful trajectories as well. Use
        :meth:`MotionGenResult.get_successful_paths` to get only successful trajectories.

        Returns:
            List[JointState]: Interpolated trajectories. Check
                :attr:`MotionGenResult.interpolation_dt` for the time between steps.
        """
        path = [
            self.interpolated_plan[x].trim_trajectory(0, self.path_buffer_last_tstep[x])
            for x in range(len(self.interpolated_plan))
        ]
        return path

    def get_successful_paths(self) -> List[torch.Tensor]:
        """Get successful interpolated trajectories from the result. Use for batched queries.

        Returns:
            List[JointState]: Interpolated trajectories. Check
                :attr:`MotionGenResult.interpolation_dt` for the time between steps.
        """
        path = []
        nz_i = torch.nonzero(self.success.view(-1)).view(-1)
        path = self.interpolated_plan[nz_i]
        path_list = []

        if self.path_buffer_last_tstep is not None:
            for i in range(len(path)):
                last_tstep = self.path_buffer_last_tstep[nz_i[i]]
                path_list.append(path[i].trim_trajectory(0, last_tstep))
        else:
            path_list = [path[i, :, :] for i in range(path.shape[0])]

        return path_list

    def get_interpolated_plan(self) -> JointState:
        """Get interpolated trajectory from the result.

        Returns:
            JointState: Interpolated trajectory. Check :attr:`MotionGenResult.interpolation_dt` for
            the time between steps.
        """
        if self.path_buffer_last_tstep is None:
            return self.interpolated_plan
        if len(self.path_buffer_last_tstep) > 1:
            log_error("only single result is supported")

        return self.interpolated_plan.trim_trajectory(0, self.path_buffer_last_tstep[0])

    def retime_trajectory(
        self,
        time_dilation_factor: float,
        interpolate_trajectory: bool = True,
        interpolation_dt: Optional[float] = None,
        interpolation_kind: InterpolateType = InterpolateType.LINEAR_CUDA,
        create_interpolation_buffer: bool = True,
    ):
        """Retime the optimized trajectory by a dilation factor.

        Args:
            time_dilation_factor: Time factor to slow down the trajectory. Should be less than 1.0.
            interpolate_trajectory: Interpolate the trajectory after retiming.
            interpolation_dt: Time between steps in the interpolated trajectory. If None,
                :attr:`MotionGenResult.interpolation_dt` is used.
            interpolation_kind: Interpolation type to use.
            create_interpolation_buffer: Create a new buffer for interpolated trajectory. Set this
                to True if existing buffer is not large enough to store new trajectory.
        """

        if time_dilation_factor > 1.0:
            log_error("time_dilation_factor should be less than 1.0")
        if time_dilation_factor == 1.0:
            return
        if len(self.path_buffer_last_tstep) > 1:
            log_error("only single result is supported")

        new_dt = self.optimized_dt * (1.0 / time_dilation_factor)
        if len(self.optimized_plan.shape) == 3:
            new_dt = new_dt.view(-1, 1, 1)
        else:
            new_dt = new_dt.view(-1, 1)
        self.optimized_plan = self.optimized_plan.scale_by_dt(self.optimized_dt, new_dt)
        self.optimized_dt = new_dt.view(-1)
        if interpolate_trajectory:
            if interpolation_dt is not None:
                self.interpolation_dt = interpolation_dt
            self.interpolated_plan, last_tstep, _ = get_batch_interpolated_trajectory(
                self.optimized_plan,
                self.optimized_dt,
                self.interpolation_dt,
                kind=interpolation_kind,
                out_traj_state=self.interpolated_plan if not create_interpolation_buffer else None,
                tensor_args=self.interpolated_plan.tensor_args,
                optimize_dt=False,
            )
            self.path_buffer_last_tstep = [last_tstep[i] for i in range(len(last_tstep))]
            if len(self.optimized_plan.shape) == 2:
                self.interpolated_plan = self.interpolated_plan.squeeze(0)

    @property
    def motion_time(self) -> Union[float, torch.Tensor]:
        """Compute motion time in seconds."""

        # -2 as last three timesteps have the same value
        # 0, 1 also have the same position value.
        return self.optimized_dt * (self.optimized_plan.position.shape[-2] - 1 - 2 - 1)

    @staticmethod
    def _check_none_and_copy_idx(
        current_tensor: Union[torch.Tensor, JointState, None],
        source_tensor: Union[torch.Tensor, JointState, None],
        idx: int,
    ) -> Union[torch.Tensor, JointState]:
        """Helper function to copy data from source tensor to current tensor at index.

        Also supports copying from JointState to JointState.

        Args:
            current_tensor: tensor to copy data to.
            source_tensor: tensor to copy data from.
            idx: index to copy data at.

        Returns:
            Union[torch.Tensor, JointState]: copied tensor.
        """
        if source_tensor is not None:
            if current_tensor is None:
                current_tensor = source_tensor.clone()
            else:
                if isinstance(current_tensor, torch.Tensor) and isinstance(
                    source_tensor, torch.Tensor
                ):
                    current_tensor[idx] = source_tensor[idx]
                elif isinstance(current_tensor, JointState) and isinstance(
                    source_tensor, JointState
                ):
                    source_state = source_tensor[idx]
                    current_tensor.copy_at_index(source_state, idx)

        return current_tensor


@dataclass
class GraspPlanResult:
    success: Optional[torch.Tensor] = None
    grasp_trajectory: Optional[JointState] = None
    grasp_trajectory_dt: Optional[torch.Tensor] = None
    grasp_interpolated_trajectory: Optional[JointState] = None
    grasp_interpolation_dt: Optional[torch.Tensor] = None
    retract_trajectory: Optional[JointState] = None
    retract_trajectory_dt: Optional[torch.Tensor] = None
    retract_interpolated_trajectory: Optional[JointState] = None
    retract_interpolation_dt: Optional[torch.Tensor] = None
    approach_result: Optional[MotionGenResult] = None
    grasp_result: Optional[MotionGenResult] = None
    retract_result: Optional[MotionGenResult] = None
    status: Optional[str] = None
    goalset_result: Optional[MotionGenResult] = None
    planning_time: float = 0.0
    goalset_index: Optional[torch.Tensor] = None


class MotionGen(MotionGenConfig):
    """Motion generation wrapper for generating collision-free trajectories.

    This module provides an interface to generate collision-free trajectories for manipulators. It
    supports Cartesian Pose for end-effectors and joint space goals. When a Cartesian Pose is used
    as target, it uses batched inverse kinematics to find multiple joint configuration solutions
    for the Cartesian Pose and then runs trajectory optimization with a linear interpolation from
    start configuration to each of the IK solutions. When trajectory optimization fails, it uses
    a graph planner to find collision-free paths to the IK solutions and then uses these paths as
    seeds for trajectory optimization. The module also supports batched queries for motion
    generation. Use this module as the high-level API for generating collision-free trajectories.
    """

    def __init__(self, config: MotionGenConfig):
        """Initializes the motion generation module.

        Args:
            config: Configuration for motion generation.
        """
        super().__init__(**vars(config))
        self.rollout_fn = self.graph_planner.safety_rollout_fn
        self._trajopt_goal_config = None
        self._dof = self.rollout_fn.d_action
        self._batch_graph_search_buffer = None
        self._batch_path_buffer_last_tstep = None
        self._rollout_list = None
        self._solver_rollout_list = None
        self._pose_solver_rollout_list = None
        self._pose_rollout_list = None
        self._kin_list = None
        self.update_batch_size(seeds=self.trajopt_seeds)

    def update_batch_size(self, seeds=10, batch=1):
        """Update the batch size for motion generation.

        Args:
            seeds: Number of seeds for trajectory optimization and graph planner.
            batch: Number of queries to run in batch mode.
        """
        if (
            self._trajopt_goal_config is None
            or self._trajopt_goal_config.shape[0] != batch
            or self._trajopt_goal_config.shape[1] != seeds
        ):
            self._trajopt_goal_config = torch.zeros(
                (batch, seeds, self.rollout_fn.d_action),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self._batch_col = (
                torch.arange(0, batch, device=self.tensor_args.device, dtype=torch.long) * seeds
            )
        if (
            self._batch_graph_search_buffer is None
            or self._batch_graph_search_buffer.shape[0] != batch
        ):
            self._batch_graph_search_buffer = JointState.zeros(
                (batch, self.interpolation_steps, self.kinematics.get_dof()),
                tensor_args=self.tensor_args,
                joint_names=self.rollout_fn.joint_names,
            )
            self._batch_path_buffer_last_tstep = [0 for i in range(batch)]

    def solve_ik(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
    ) -> IKResult:
        """Solve inverse kinematics for a given Cartesian Pose or a batch of Poses.

        Use this only if your problem size is same as when using motion generation. If you want
        to solve IK for a different problem size, create an instance of
        :class:`curobo.wrap.reacher.ik_solver.IKSolver`.

        Args:
            goal_pose: Goal Pose for the end-effector.
            retract_config: Retract configuration for the end-effector.
            seed_config: Seed configuration for inverse kinematics.
            return_seeds: Number of solutions to return per problem query.
            num_seeds: Number of seeds to use for solving inverse kinematics.
            use_nn_seed: Use neural network seed for solving inverse kinematics. This is not
                implemented.
            newton_iters: Number of Newton iterations to run for solving inverse kinematics.
                This is useful to override the default number of iterations.

        Returns:
            IKResult: Result of inverse kinematics.
        """
        return self.ik_solver.solve(
            goal_pose,
            retract_config,
            seed_config,
            return_seeds,
            num_seeds,
            use_nn_seed,
            newton_iters,
        )

    @profiler.record_function("motion_gen/graph_search")
    def graph_search(
        self, start_config: T_BDOF, goal_config: T_BDOF, interpolation_steps: Optional[int] = None
    ) -> GraphResult:
        """Run graph search to find collision-free paths between start and goal configurations.

        Args:
            start_config: Start joint configurations of the robot.
            goal_config: Goal joint configurations of the robot.
            interpolation_steps: Number of interpolation steps to interpolate obtained solutions.

        Returns:
            GraphResult: Result of graph search.
        """
        return self.graph_planner.find_paths(start_config, goal_config, interpolation_steps)

    def plan_single(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: List[Pose] = None,
    ) -> MotionGenResult:
        """Plan a single motion to reach a goal pose from a start joint state.

        This also supports reaching target poses for different links in the robot by providing
        goal poses for each link in the link_poses argument. The link_poses argument should contain
        a pose for each link specified in :attr:`MotionGen.kinematics`. Constrained planning is
        supported by calling :meth:`MotionGen.update_pose_cost_metric` before calling this method.
        See :ref:`tut_constrained_planning` for more details.

        Args:
            start_state: Start joint state of the robot. When planning from a non-static state, i.e,
                when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt` to
                False.
            goal_pose: Goal pose for the end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
                attribute to see if the query was successful.
        """
        log_info("Planning for Single Goal: " + str(goal_pose.batch))
        solve_state = self._get_solve_state(
            ReacherSolveType.SINGLE, plan_config, goal_pose, start_state
        )

        result = self._plan_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
            link_poses=link_poses,
        )
        return result

    def plan_goalset(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: List[Pose] = None,
    ) -> MotionGenResult:
        """Plan a single motion to reach a goal from set of poses, from a start joint state.

        Use this when planning to reach a grasp from a set of possible grasps. This method will
        try to reach the closest goal pose from the set of goal poses at every iteration of
        optimization during IK and trajectory optimization. This method only supports goalset for
        main end-effector (i.e., `goal_pose`). Use single Pose target for links in `link_poses`.

        Args:
            start_state: Start joint state of the robot. When planning from a non-static state,
                i.e, when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt`
                to False.
            goal_pose: Goal pose for the end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
                attribute to see if the query was successful.
        """
        solve_state = self._get_solve_state(
            ReacherSolveType.GOALSET, plan_config, goal_pose, start_state
        )

        result = self._plan_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
            link_poses=link_poses,
        )
        return result

    def plan_batch(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Dict[str, List[Pose]] = None,
    ) -> MotionGenResult:
        """Plan motions to reach a batch of goal poses from a batch of start joint states.

        Args:
            start_state: Start joint states of the robot. When planning from a non-static state,
                i.e, when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt`
                to False.
            goal_pose: Goal poses for the end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
                attribute to check which indices of the batch were successful.
        """
        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH, plan_config, goal_pose, start_state
        )

        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
            link_poses=link_poses,
        )
        return result

    def plan_batch_goalset(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Dict[str, List[Pose]] = None,
    ) -> MotionGenResult:
        """Plan motions to reach a batch of poses (goalset) from a batch of start joint states.

        Args:
            start_state: Start joint states of the robot. When planning from a non-static state,
                i.e, when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt`
                to False.
            goal_pose: Goal poses for the end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
                attribute to check which indices of the batch were successful.
        """

        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH_GOALSET, plan_config, goal_pose, start_state
        )

        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
            link_poses=link_poses,
        )
        return result

    def plan_batch_env(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Dict[str, List[Pose]] = None,
    ) -> MotionGenResult:
        """Plan motions to reach (batch) poses in different collision environments.

        Args:
            start_state: Start joint states of the robot. When planning from a non-static state,
                i.e, when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt`
                to False.
            goal_pose: Goal poses for the end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
                attribute to check which indices of the batch were successful.
        """
        if plan_config.enable_graph:
            log_info(
                "Batch env mode does not support graph search, setting "
                + "MotionGenPlanConfig.enable_graph=False"
            )
            plan_config.enable_graph = False

        if plan_config.enable_graph_attempt is not None:
            log_info(
                "Batch env mode does not support graph search, setting "
                + "MotionGenPlanConfig.enable_graph_attempt=None"
            )
            plan_config.enable_graph_attempt = None
        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH_ENV, plan_config, goal_pose, start_state
        )
        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
            link_poses=link_poses,
        )
        return result

    def plan_batch_env_goalset(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Dict[str, List[Pose]] = None,
    ) -> MotionGenResult:
        """Plan motions to reach (batch) goalset poses in different collision environments.

        Args:
            start_state: Start joint states of the robot. When planning from a non-static state,
                i.e, when velocity or acceleration is non-zero, set :attr:`MotionGen.optimize_dt`
                to False.
            goal_pose: Goal poses for the end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for each link in the robot when planning for multiple links.

        Returns:
            MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
                attribute to check which indices of the batch were successful.
        """

        if plan_config.enable_graph:
            log_info(
                "Batch env mode does not support graph search, setting "
                + "MotionGenPlanConfig.enable_graph=False"
            )
            plan_config.enable_graph = False

        if plan_config.enable_graph_attempt is not None:
            log_info(
                "Batch env mode does not support graph search, setting "
                + "MotionGenPlanConfig.enable_graph_attempt=None"
            )
            plan_config.enable_graph_attempt = None
        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH_ENV_GOALSET, plan_config, goal_pose, start_state
        )
        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
            link_poses=link_poses,
        )
        return result

    def compute_kinematics(self, state: JointState) -> KinematicModelState:
        """Compute kinematics for a given joint state.

        Args:
            state: Joint state of the robot. Only :attr:`JointState.position` is used.

        Returns:
            KinematicModelState: Kinematic state of the robot.
        """
        out = self.rollout_fn.compute_kinematics(state)
        return out

    @property
    def kinematics(self) -> CudaRobotModel:
        """Returns the shared kinematics model of the robot."""
        return self.rollout_fn.kinematics

    @property
    def dof(self) -> int:
        """Returns the number of controlled degrees of freedom of the robot."""
        return self.kinematics.get_dof()

    @property
    def collision_cache(self) -> Dict[str, int]:
        """Returns the collision cache created by the world collision checker."""
        return self.world_coll_checker.cache

    def check_constraints(self, state: JointState) -> RolloutMetrics:
        """Compute IK constraints for a given joint state.

        Args:
            state: Joint state of the robot.

        Returns:
            RolloutMetrics: Metrics for the joint state.
        """
        metrics = self.ik_solver.check_constraints(state)
        return metrics

    def update_world(self, world: WorldConfig):
        """Update the world representation for collision checking.

        This allows for updating the world representation as long as the new world representation
        does not have a larger number of obstacles than the :attr:`MotionGen.collision_cache` as
        created during initialization of :class:`MotionGenConfig`. Updating the world also
        invalidates the cached roadmaps in the graph planner. See :ref:`world_collision` for more
        details.

        Args:
            world: New world configuration for collision checking.
        """
        self.world_coll_checker.load_collision_model(world, fix_cache_reference=self.use_cuda_graph)
        self.graph_planner.reset_buffer()

    def clear_world_cache(self):
        """Remove all collision objects from collision cache."""

        self.world_coll_checker.clear_cache()

    def reset(self, reset_seed=True):
        """Reset the motion generation module.

        Args:
            reset_seed: Reset the random seed generator. Resetting this can be time consuming, if
                deterministic results are not required, set this to False.
        """
        self.graph_planner.reset_buffer()
        if reset_seed:
            self.reset_seed()

    def reset_seed(self):
        """Reset the random seed generators in all sub-modules of motion generation."""
        self.rollout_fn.reset_seed()
        self.ik_solver.reset_seed()
        self.graph_planner.reset_seed()
        self.trajopt_solver.reset_seed()
        self.js_trajopt_solver.reset_seed()

    def get_retract_config(self) -> T_DOF:
        """Returns the retract/home configuration of the robot."""
        return self.rollout_fn.dynamics_model.retract_config

    def warmup(
        self,
        enable_graph: bool = True,
        batch: Optional[int] = None,
        warmup_js_trajopt: bool = True,
        batch_env_mode: bool = False,
        parallel_finetune: bool = True,
        n_goalset: int = -1,
        warmup_joint_index: int = 0,
        warmup_joint_delta: float = 0.1,
    ):
        """Warmup planning methods for motion generation.

        Args:
            enable_graph: Enable graph search for warmup.
            batch: Number of problem queries for batch mode. If None, single query is run.
            warmup_js_trajopt: Warmup joint space planning in addition to Cartesian planning.
            batch_env_mode: Enable batch world environment mode for warmup. Only used when batch is
                not None.
            parallel_finetune: Run finetuning trajectory optimization in parallel for warmup. Leave
                this to True for most cases.
            n_goalset: Number of goal poses to use for warmup. If -1, single goal pose is used. Set
                this to the largest number of goals you plan to use with
                :meth:`MotionGen.plan_goalset`. After warmup, you can use smaller number of goals
                and the method will internally pad the extra goals with the first goal.
            warmup_joint_index: Index of the joint to perturb for warmup.
            warmup_joint_delta: Delta to perturb the joint for warmup.
        """
        log_info("Warmup")
        if warmup_js_trajopt:
            start_state = JointState.from_position(
                self.rollout_fn.dynamics_model.retract_config.view(1, -1).clone(),
                joint_names=self.rollout_fn.joint_names,
            )
            # warm up js_trajopt:
            goal_state = start_state.clone()
            goal_state.position[..., warmup_joint_index] += warmup_joint_delta
            for _ in range(3):
                self.plan_single_js(
                    start_state.clone(),
                    goal_state.clone(),
                    MotionGenPlanConfig(max_attempts=1, enable_finetune_trajopt=True),
                )

        if enable_graph:
            start_state = JointState.from_position(
                self.rollout_fn.dynamics_model.retract_config.view(1, -1).clone(),
                joint_names=self.rollout_fn.joint_names,
            )
            start_state.position[..., warmup_joint_index] += warmup_joint_delta
            self.graph_planner.warmup(
                self.rollout_fn.dynamics_model.retract_config.view(1, -1).clone(),
                start_state.position,
            )

        if batch is None:
            start_state = JointState.from_position(
                self.rollout_fn.dynamics_model.retract_config.view(1, -1).clone(),
                joint_names=self.rollout_fn.joint_names,
            )
            state = self.rollout_fn.compute_kinematics(start_state)
            link_poses = state.link_pose

            if n_goalset == -1:
                retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
                start_state.position[..., warmup_joint_index] += warmup_joint_delta
                for _ in range(3):
                    self.plan_single(
                        start_state,
                        retract_pose,
                        MotionGenPlanConfig(
                            max_attempts=1,
                            enable_finetune_trajopt=True,
                            parallel_finetune=parallel_finetune,
                        ),
                        link_poses=link_poses,
                    )

                self.plan_single(
                    start_state,
                    retract_pose,
                    MotionGenPlanConfig(
                        max_attempts=1,
                        enable_finetune_trajopt=True,
                        enable_graph=enable_graph,
                        parallel_finetune=parallel_finetune,
                    ),
                    link_poses=link_poses,
                )
            else:
                retract_pose = Pose(
                    state.ee_pos_seq.repeat(n_goalset, 1).view(1, n_goalset, 3),
                    quaternion=state.ee_quat_seq.repeat(n_goalset, 1).view(1, n_goalset, 4),
                )
                start_state.position[..., warmup_joint_index] += warmup_joint_delta
                for _ in range(3):
                    self.plan_goalset(
                        start_state,
                        retract_pose,
                        MotionGenPlanConfig(
                            max_attempts=1,
                            enable_finetune_trajopt=True,
                            parallel_finetune=parallel_finetune,
                        ),
                        link_poses=link_poses,
                    )

                self.plan_goalset(
                    start_state,
                    retract_pose,
                    MotionGenPlanConfig(
                        max_attempts=1,
                        enable_finetune_trajopt=True,
                        enable_graph=enable_graph,
                        parallel_finetune=parallel_finetune,
                    ),
                    link_poses=link_poses,
                )

        else:
            start_state = JointState.from_position(
                self.get_retract_config().view(1, -1).clone(),
                joint_names=self.rollout_fn.joint_names,
            ).repeat_seeds(batch)
            state = self.rollout_fn.compute_kinematics(start_state)
            link_poses = state.link_pose

            if n_goalset == -1:
                retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
                start_state.position[..., warmup_joint_index] += warmup_joint_delta

                for _ in range(3):
                    if batch_env_mode:
                        self.plan_batch_env(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=False,
                                enable_graph_attempt=None,
                            ),
                            link_poses=link_poses,
                        )
                    else:
                        self.plan_batch(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=enable_graph,
                                enable_graph_attempt=None if not enable_graph else 20,
                            ),
                            link_poses=link_poses,
                        )
            else:
                retract_pose = Pose(
                    state.ee_pos_seq.view(batch, 1, 3).repeat(1, n_goalset, 1).contiguous(),
                    quaternion=state.ee_quat_seq.view(batch, 1, 4)
                    .repeat(1, n_goalset, 1)
                    .contiguous(),
                )
                start_state.position[..., warmup_joint_index] += warmup_joint_delta
                for _ in range(3):
                    if batch_env_mode:
                        self.plan_batch_env_goalset(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=False,
                            ),
                            link_poses=link_poses,
                        )
                    else:
                        self.plan_batch_goalset(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=enable_graph,
                                enable_graph_attempt=None if not enable_graph else 20,
                            ),
                            link_poses=link_poses,
                        )

        log_info("Warmup complete")

    def plan_single_js(
        self,
        start_state: JointState,
        goal_state: JointState,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    ) -> MotionGenResult:
        """Plan a single motion to reach a goal joint state from a start joint state.

        This method uses trajectory optimization to find a collision-free path between the start
        and goal joint states. If trajectory optimization fails, it uses a graph planner to find
        a collision-free path to the goal joint state. The graph plan is then used as a seed for
        trajectory optimization.

        Args:
            start_state: Start joint state of the robot.
            goal_state: Goal joint state of the robot.
            plan_config: Planning parameters for motion generation.

        Returns:
            MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
                attribute to see if the query was successful.
        """

        start_time = time.time()

        time_dict = {
            "solve_time": 0,
            "ik_time": 0,
            "graph_time": 0,
            "trajopt_time": 0,
            "trajopt_attempts": 0,
            "finetune_time": 0,
        }
        result = None
        # goal = Goal(goal_state=goal_state, current_state=start_state)
        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE,
            num_ik_seeds=1,
            num_trajopt_seeds=self.js_trajopt_solver.num_seeds,
            num_graph_seeds=self.js_trajopt_solver.num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=1,
        )
        force_graph = plan_config.enable_graph
        valid_query = True
        if plan_config.check_start_validity:
            valid_query, status = self.check_start_state(start_state)
            if not valid_query:
                result = MotionGenResult(
                    success=torch.as_tensor([False], device=self.tensor_args.device),
                    valid_query=valid_query,
                    status=status,
                )
                return result

        for n in range(plan_config.max_attempts):
            result = self._plan_js_from_solve_state(
                solve_state, start_state, goal_state, plan_config=plan_config
            )
            time_dict["trajopt_time"] += result.trajopt_time
            time_dict["graph_time"] += result.graph_time
            time_dict["finetune_time"] += result.finetune_time
            time_dict["trajopt_attempts"] = n
            if plan_config.enable_graph_attempt is not None and (
                n >= plan_config.enable_graph_attempt - 1 and not plan_config.enable_graph
            ):
                plan_config.enable_graph = True
            if plan_config.disable_graph_attempt is not None and (
                n >= plan_config.disable_graph_attempt - 1 and not force_graph
            ):
                plan_config.enable_graph = False

            if result.success.item():
                break
            if not result.valid_query:
                break
            if time.time() - start_time > plan_config.timeout:
                break

        result.graph_time = time_dict["graph_time"]
        result.finetune_time = time_dict["finetune_time"]
        result.trajopt_time = time_dict["trajopt_time"]
        result.solve_time = result.trajopt_time + result.graph_time + result.finetune_time
        result.total_time = result.solve_time
        result.attempts = n
        if plan_config.time_dilation_factor is not None and torch.count_nonzero(result.success) > 0:
            result.retime_trajectory(
                plan_config.time_dilation_factor,
                interpolation_kind=self.js_trajopt_solver.interpolation_type,
            )
        return result

    def solve_trajopt(
        self,
        goal: Goal,
        act_seed,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
    ):
        """Solve trajectory optimization for a given goal.

        Args:
            goal: Goal for trajectory optimization.
            act_seed: Seed for trajectory optimization.
            return_all_solutions: Return results for all seeds in trajectory optimization.
            num_seeds: Override number of seeds to use for trajectory optimization.

        Returns:
            TrajOptResult: Result of trajectory optimization.
        """
        result = self.trajopt_solver.solve(
            goal, act_seed, return_all_solutions=return_all_solutions, num_seeds=num_seeds
        )
        return result

    def get_active_js(
        self,
        in_js: JointState,
    ):
        """Get controlled joint state from input joint state.

        This is used to get the joint state for only joints that are optimization variables. This
        also re-orders the joints to match the order of optimization variables.

        Args:
            in_js: Input joint state.

        Returns:
            JointState: Active joint state.
        """
        opt_jnames = self.rollout_fn.joint_names
        opt_js = in_js.get_ordered_joint_state(opt_jnames)
        return opt_js

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
        start_state: Optional[JointState] = None,
        goal_pose: Optional[Pose] = None,
    ) -> bool:
        """Update the pose cost metric for :ref:`tut_constrained_planning`.

        Only supports for the main end-effector. Does not support for multiple links that are
        specified with `link_poses` in planning methods.

        Args:
            metric: Type and parameters for pose constraint to add.
            start_state: Start joint state for the constraint.
            goal_pose: Goal pose for the constraint.

        Returns:
            bool: True if the constraint can be added, False otherwise.
        """

        rollouts = self.get_all_pose_rollout_instances()

        # check if constraint is valid:
        if metric.hold_partial_pose and metric.offset_tstep_fraction < 0.0:
            start_pose = self.compute_kinematics(start_state).ee_pose.clone()
            project_distance = metric.project_to_goal_frame
            if project_distance is None:
                project_distance = rollouts[0].goal_cost.project_distance
            if project_distance:
                # project start pose to goal frame:
                projected_pose = goal_pose.compute_local_pose(start_pose)
                if torch.count_nonzero(metric.hold_vec_weight[:3] > 0.0) > 0:
                    # angular distance should be zero:
                    distance = projected_pose.angular_distance(
                        Pose.from_list([0, 0, 0, 1, 0, 0, 0], tensor_args=self.tensor_args)
                    )
                    if torch.max(distance) > 0.05:
                        log_warn(
                            "Partial orientation between start and goal is not equal"
                            + str(distance)
                        )
                        return False

                # check linear distance:
                if (
                    torch.count_nonzero(
                        torch.abs(projected_pose.position[..., metric.hold_vec_weight[3:] > 0.0])
                        > 0.005
                    )
                    > 0
                ):
                    log_warn("Partial position between start and goal is not equal.")
                    return False
            else:
                # project start pose to goal frame:
                projected_position = goal_pose.position - start_pose.position
                # check linear distance:
                if (
                    torch.count_nonzero(
                        torch.abs(projected_position[..., metric.hold_vec_weight[3:] > 0.0]) > 0.005
                    )
                    > 0
                ):
                    log_warn("Partial position between start and goal is not equal.")
                    return False

        [
            rollout.update_pose_cost_metric(metric)
            for rollout in rollouts
            if isinstance(rollout, ArmReacher)
        ]
        return True

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        """Get all rollout instances used across components in motion generation."""
        if self._rollout_list is None:
            self._rollout_list = (
                self.ik_solver.get_all_rollout_instances()
                + self.graph_planner.get_all_rollout_instances()
                + self.trajopt_solver.get_all_rollout_instances()
                + self.finetune_trajopt_solver.get_all_rollout_instances()
                + self.js_trajopt_solver.get_all_rollout_instances()
                + self.finetune_js_trajopt_solver.get_all_rollout_instances()
            )
        return self._rollout_list

    def get_all_solver_rollout_instances(self) -> List[RolloutBase]:
        """Get all rollout instances in solvers (IK, TrajOpt)."""
        if self._solver_rollout_list is None:
            self._solver_rollout_list = (
                self.ik_solver.solver.get_all_rollout_instances()
                + self.trajopt_solver.solver.get_all_rollout_instances()
                + self.finetune_trajopt_solver.solver.get_all_rollout_instances()
                + self.js_trajopt_solver.solver.get_all_rollout_instances()
                + self.finetune_js_trajopt_solver.solver.get_all_rollout_instances()
            )
        return self._solver_rollout_list

    def get_all_pose_solver_rollout_instances(self) -> List[RolloutBase]:
        """Get all rollout instances in solvers (IK, TrajOpt) that support Cartesian cost terms."""
        if self._pose_solver_rollout_list is None:
            self._pose_solver_rollout_list = (
                self.ik_solver.solver.get_all_rollout_instances()
                + self.trajopt_solver.solver.get_all_rollout_instances()
                + self.finetune_trajopt_solver.solver.get_all_rollout_instances()
            )
        return self._pose_solver_rollout_list

    def get_all_pose_rollout_instances(self) -> List[RolloutBase]:
        """Get all rollout instances used across components in motion generation."""
        if self._pose_rollout_list is None:
            self._pose_rollout_list = (
                self.ik_solver.get_all_rollout_instances()
                + self.trajopt_solver.get_all_rollout_instances()
                + self.finetune_trajopt_solver.get_all_rollout_instances()
            )
        return self._pose_rollout_list

    def get_all_kinematics_instances(self) -> List[CudaRobotModel]:
        """Get all kinematics instances used across components in motion generation.

        This is deprecated. Use :meth:`MotionGen.kinematics` instead as MotionGen now uses a shared
        kinematics instance across all components.

        Returns:
            List[CudaRobotModel]: Single kinematics instance, returned as a list for compatibility.
        """
        if self._kin_list is None:
            self._kin_list = [
                i.dynamics_model.robot_model for i in self.get_all_rollout_instances()
            ]
        return self._kin_list

    def attach_objects_to_robot(
        self,
        joint_state: JointState,
        object_names: List[str],
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset: Optional[Pose] = None,
        remove_obstacles_from_world_config: bool = False,
    ) -> bool:
        """Attach an object or objects from world to a robot's link.

        This method assumes that the objects exist in the world configuration. If attaching
        objects that are not in world, use :meth:`MotionGen.attach_external_objects_to_robot`.

        Args:
            joint_state: Joint state of the robot.
            object_names: Names of objects in the world to attach to the robot.
            surface_sphere_radius: Radius (in meters) to use for points sampled on surface of the
                object. A smaller radius will allow for generating motions very close to obstacles.
            link_name: Name of the link (frame) to attach the objects to. The assumption is that
                this link does not have any geometry and all spheres of this link represent
                attached objects.
            sphere_fit_type: Sphere fit algorithm to use. See :ref:`attach_object_note` for more
                details. The default method :attr:`SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE`
                voxelizes the volume of the objects and adds spheres representing the voxels, then
                samples points on the surface of the object, adds :attr:`surface_sphere_radius` to
                these points. This should be used for most cases.
            voxelize_method: Method to use for voxelization, passed to
                :py:func:`trimesh.voxel.creation.voxelize`.
            world_objects_pose_offset: Offset to apply to the object poses before attaching to the
                robot. This is useful when attaching an object that's in contact with the world.
                The offset is applied in the world frame before attaching to the robot.
            remove_obstacles_from_world_config: Remove the obstacles from the world cache after
                attaching to the robot to reduce memory usage. Note that when an object is attached
                to the robot, it's disabled in the world collision checker. This flag when enabled,
                also removes the object from world cache. For most cases, this should be set to
                False.
        """

        log_info("MG: Attach objects to robot")
        kin_state = self.compute_kinematics(joint_state)
        ee_pose = kin_state.ee_pose  # w_T_ee
        if world_objects_pose_offset is not None:
            # add offset from ee:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
            # new ee_pose:
            # w_T_ee = offset_T_w * w_T_ee
            # ee_T_w
        ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
        max_spheres = self.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name)
        n_spheres = int(max_spheres / len(object_names))
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            log_warn(
                "MG: No spheres found, max_spheres: "
                + str(max_spheres)
                + " n_objects: "
                + str(len(object_names))
            )
            return False
        for i, x in enumerate(object_names):
            obs = self.world_model.get_obstacle(x)
            if obs is None:
                log_error(
                    "Object not found in world. Object name: "
                    + x
                    + " Name of objects in world: "
                    + " ".join([i.name for i in self.world_model.objects])
                )
            sph = obs.get_bounding_spheres(
                n_spheres,
                surface_sphere_radius,
                pre_transform_pose=ee_pose,
                tensor_args=self.tensor_args,
                fit_type=sphere_fit_type,
                voxelize_method=voxelize_method,
            )
            sph_list += [s.position + [s.radius] for s in sph]

            self.world_coll_checker.enable_obstacle(enable=False, name=x)
            if remove_obstacles_from_world_config:
                self.world_model.remove_obstacle(x)
        log_info("MG: Computed spheres for attach objects to robot")

        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))

        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)

        return True

    def attach_external_objects_to_robot(
        self,
        joint_state: JointState,
        external_objects: List[Obstacle],
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> bool:
        """Attach external objects (not in world model) to a robot's link.

        Args:
            joint_state: Joint state of the robot.
            external_objects: List of external objects to attach to the robot.
            surface_sphere_radius: Radius (in meters) to use for points sampled on surface of the
                object. A smaller radius will allow for generating motions very close to obstacles.
            link_name: Name of the link (frame) to attach the objects to. The assumption is that
                this link does not have any geometry and all spheres of this link represent
                attached objects.
            sphere_fit_type: Sphere fit algorithm to use. See :ref:`attach_object_note` for more
                details. The default method :attr:`SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE`
                voxelizes the volume of the objects and adds spheres representing the voxels, then
                samples points on the surface of the object, adds :attr:`surface_sphere_radius` to
                these points. This should be used for most cases.
            voxelize_method: Method to use for voxelization, passed to
                :py:func:`trimesh.voxel.creation.voxelize`.
            world_objects_pose_offset: Offset to apply to the object poses before attaching to the
                robot. This is useful when attaching an object that's in contact with the world.
                The offset is applied in the world frame before attaching to the robot.
        """
        log_info("MG: Attach objects to robot")
        if len(external_objects) == 0:
            log_error("no object in external_objects")
        kin_state = self.compute_kinematics(joint_state)
        ee_pose = kin_state.ee_pose  # w_T_ee
        if world_objects_pose_offset is not None:
            # add offset from ee:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
            # new ee_pose:
            # w_T_ee = offset_T_w * w_T_ee
            # ee_T_w
        ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
        max_spheres = self.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name)
        object_names = [x.name for x in external_objects]
        n_spheres = int(max_spheres / len(object_names))
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            log_warn(
                "MG: No spheres found, max_spheres: "
                + str(max_spheres)
                + " n_objects: "
                + str(len(object_names))
            )
            return False
        for i, x in enumerate(object_names):
            obs = external_objects[i]
            sph = obs.get_bounding_spheres(
                n_spheres,
                surface_sphere_radius,
                pre_transform_pose=ee_pose,
                tensor_args=self.tensor_args,
                fit_type=sphere_fit_type,
                voxelize_method=voxelize_method,
            )
            sph_list += [s.position + [s.radius] for s in sph]

        log_info("MG: Computed spheres for attach objects to robot")

        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))

        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)
        return True

    def add_camera_frame(self, camera_observation: CameraObservation, obstacle_name: str):
        """Add camera frame to the world collision checker.

        Only supported by :py:class:`~curobo.geom.sdf.world_blox.WorldBloxCollision`.

        Args:
            camera_observation: Camera observation to add to the world collision checker.
            obstacle_name: Name of the obstacle/layer to add the camera frame to.
        """
        self.world_coll_checker.add_camera_frame(camera_observation, obstacle_name)

    def process_camera_frames(self, obstacle_name: Optional[str] = None, process_aux: bool = False):
        """Process camera frames for collision checking.

        Only supported by :py:class:`~curobo.geom.sdf.world_blox.WorldBloxCollision`.

        Args:
            obstacle_name: Name of the obstacle/layer to process the camera frames for. If None,
                processes camera frames on all supported obstacles.
            process_aux: Process auxillary information such as mesh integration in nvblox. This is
                not required for collision checking and is only needed for debugging. Default is
                False to reduce computation time.
        """
        self.world_coll_checker.process_camera_frames(obstacle_name, process_aux=process_aux)

    def attach_bounding_box_from_blox_to_robot(
        self,
        joint_state: JointState,
        bounding_box: Cuboid,
        blox_layer_name: Optional[str] = None,
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset: Optional[Pose] = None,
    ):
        """Attach the voxels in a blox layer to robot's link.

        .. note::
            This is not currently implemented.

        """
        log_error("Not implemented")
        kin_state = self.compute_kinematics(joint_state)
        ee_pose = kin_state.ee_pose  # w_T_ee
        if world_objects_pose_offset is not None:
            # add offset from ee:
            ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
            # new ee_pose:
            # w_T_ee = offset_T_w * w_T_ee
            # ee_T_w
        ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
        max_spheres = self.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(link_name)
        n_spheres = max_spheres
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            log_error("MG: No spheres found")
        sph = self.world_coll_checker.get_bounding_spheres(
            bounding_box,
            blox_layer_name,
            n_spheres,
            surface_sphere_radius,
            sphere_fit_type,
            voxelize_method,
            pre_transform_pose=ee_pose,
            clear_region=True,
        )
        sph_list += [s.position + [s.radius] for s in sph]

        log_info("MG: Computed spheres for attach objects to robot")

        spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))

        if spheres.shape[0] > max_spheres:
            spheres = spheres[: spheres.shape[0]]
        sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

        self.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)

    def attach_new_object_to_robot(
        self,
        joint_state: JointState,
        obstacle: Obstacle,
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset: Optional[Pose] = None,
    ):
        """Attach an object to robot's link. The object to be attached is not in the world model.

        Deprecated. Use :meth:`MotionGen.attach_external_objects_to_robot` instead.

        """
        log_warn("Deprecated. Use attach_external_objects_to_robot instead")
        return self.attach_external_objects_to_robot(
            joint_state=joint_state,
            external_objects=[obstacle],
            surface_sphere_radius=surface_sphere_radius,
            link_name=link_name,
            sphere_fit_type=sphere_fit_type,
            voxelize_method=voxelize_method,
            world_objects_pose_offset=world_objects_pose_offset,
        )

    def detach_object_from_robot(self, link_name: str = "attached_object") -> None:
        """Detach object from robot's link.

        Args:
            link_name: Name of the link.
        """
        self.detach_spheres_from_robot(link_name)

    def attach_spheres_to_robot(
        self,
        sphere_radius: Optional[float] = None,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> None:
        """Attach spheres to robot's link.

        Args:
            sphere_radius: Radius of the spheres. Set to None if :attr:`sphere_tensor` is provided.
            sphere_tensor: Sphere x, y, z, r tensor.
            link_name: Name of the link to attach the spheres to. Note that this link should
                already have pre-allocated spheres.
        """
        self.robot_cfg.kinematics.kinematics_config.attach_object(
            sphere_radius=sphere_radius, sphere_tensor=sphere_tensor, link_name=link_name
        )

    def detach_spheres_from_robot(self, link_name: str = "attached_object") -> None:
        """Detach spheres from a robot's link.

        Args:
            link_name: Name of the link.
        """
        self.robot_cfg.kinematics.kinematics_config.detach_object(link_name)

    def get_full_js(self, active_js: JointState) -> JointState:
        """Get full joint state from controlled joint state, appending locked joints.

        Args:
            active_js: Controlled joint state

        Returns:
            JointState: Full joint state.
        """
        return self.rollout_fn.get_full_dof_from_solution(active_js)

    @property
    def world_model(self) -> WorldConfig:
        """Get the world model used for collision checking."""
        return self.world_coll_checker.world_model

    @property
    def world_collision(self) -> WorldCollision:
        """Get the shared instance of world collision checker."""
        return self.world_coll_checker

    @property
    def project_pose_to_goal_frame(self) -> bool:
        """Check if the pose cost metric is projected to goal frame."""
        return self.trajopt_solver.rollout_fn.goal_cost.project_distance

    @property
    def joint_names(self) -> List[str]:
        """Get the joint names of the robot."""
        return self.rollout_fn.joint_names

    def update_interpolation_type(
        self,
        interpolation_type: InterpolateType,
        update_graph: bool = True,
        update_trajopt: bool = True,
    ):
        """Update interpolation type for motion generation.

        Args:
            interpolation_type: Type of interpolation to use.
            update_graph: Update graph planner with the new interpolation type.
            update_trajopt: Update trajectory optimization solvers with the new interpolation type.
        """
        if update_graph:
            self.graph_planner.interpolation_type = interpolation_type
        if update_trajopt:
            self.trajopt_solver.interpolation_type = interpolation_type
            self.finetune_trajopt_solver.interpolation_type = interpolation_type
            self.js_trajopt_solver.interpolation_type = interpolation_type
            self.finetune_js_trajopt_solver.interpolation_type = interpolation_type

    def update_locked_joints(
        self, lock_joints: Dict[str, float], robot_config_dict: Union[str, Dict[Any]]
    ):
        """Update locked joints in the robot configuration.

        Use this function to update the joint values of currently locked joints between
        planning calls. This function can also be used to change which joints are locked, however
        this is only supported when the number of locked joints is the same as the original
        robot configuration as the kinematics tensors are pre-allocated.

        Args:
            lock_joints: Dictionary of joint names and values to lock.
            robot_config_dict: Robot configuration dictionary or path to robot configuration file.
        """
        if isinstance(robot_config_dict, str):
            robot_config_dict = load_yaml(join_path(get_robot_configs_path(), robot_config_dict))[
                "robot_cfg"
            ]
        if "robot_cfg" in robot_config_dict:
            robot_config_dict = robot_config_dict["robot_cfg"]
        robot_config_dict["kinematics"]["lock_joints"] = lock_joints
        robot_cfg = RobotConfig.from_dict(robot_config_dict, self.tensor_args)
        self.kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

    def check_start_state(
        self, start_state: JointState
    ) -> Tuple[bool, Union[None, MotionGenStatus]]:
        """Check if the start state is valid for motion generation.

        Args:
            start_state: Start joint state of the robot.

        Returns:
            Tuple[bool, MotionGenStatus]: Tuple containing True if the start state is valid and
                the status of the start state.
        """
        joint_position = start_state.position
        if self.rollout_fn.cuda_graph_instance:
            log_error("Cannot check start state as this rollout_fn is used by a CUDA graph.")
        if len(joint_position.shape) == 1:
            joint_position = joint_position.unsqueeze(0)
        if len(joint_position.shape) > 2:
            log_error("joint_position should be of shape (batch, dof)")
        joint_position = joint_position.unsqueeze(1)
        metrics = self.rollout_fn.rollout_constraint(
            joint_position,
            use_batch_env=False,
        )
        valid_query = metrics.feasible.squeeze(1).item()
        status = None
        if not valid_query:
            self.rollout_fn.primitive_collision_constraint.disable_cost()
            self.rollout_fn.robot_self_collision_constraint.disable_cost()
            within_joint_limits = (
                self.rollout_fn.rollout_constraint(
                    joint_position,
                    use_batch_env=False,
                )
                .feasible.squeeze(1)
                .item()
            )

            self.rollout_fn.primitive_collision_constraint.enable_cost()

            if not within_joint_limits:
                self.rollout_fn.robot_self_collision_constraint.enable_cost()
                return valid_query, MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS

            self.rollout_fn.primitive_collision_constraint.enable_cost()
            world_collision_free = (
                self.rollout_fn.rollout_constraint(
                    joint_position,
                    use_batch_env=False,
                )
                .feasible.squeeze(1)
                .item()
            )
            if not world_collision_free:
                return valid_query, MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION

            self.rollout_fn.robot_self_collision_constraint.enable_cost()
            self_collision_free = (
                self.rollout_fn.rollout_constraint(
                    joint_position,
                    use_batch_env=False,
                )
                .feasible.squeeze(1)
                .item()
            )

            if not self_collision_free:
                return valid_query, MotionGenStatus.INVALID_START_STATE_SELF_COLLISION
            status = MotionGenStatus.INVALID_START_STATE_UNKNOWN_ISSUE
        return (valid_query, status)

    @profiler.record_function("motion_gen/ik")
    def _solve_ik_from_solve_state(
        self,
        goal_pose: Pose,
        solve_state: ReacherSolveState,
        start_state: JointState,
        use_nn_seed: bool,
        partial_ik_opt: bool,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        """Solve inverse kinematics from solve state, used by motion generation planning call.

        Args:
            goal_pose: Goal Pose for the end-effector.
            solve_state: Solve state for motion generation.
            start_state: Start joint configuration of the robot.
            use_nn_seed: Use seed from a neural network. Not implemented.
            partial_ik_opt: Only run 50 iterations of inverse kinematics.
            link_poses: Goal Poses of any other link in the robot that was specified in
                :meth:`curobo.types.robot.RobotConfig.kinematics.link_names`.

        Returns:
            IKResult: Result of inverse kinematics.
        """
        newton_iters = None
        if partial_ik_opt:
            newton_iters = self.partial_ik_iters
        ik_result = self.ik_solver.solve_any(
            solve_state.solve_type,
            goal_pose,
            start_state.position.view(-1, self._dof),
            start_state.position.view(-1, 1, self._dof),
            solve_state.num_trajopt_seeds,
            solve_state.num_ik_seeds,
            use_nn_seed,
            newton_iters,
            link_poses,
        )
        return ik_result

    @profiler.record_function("motion_gen/trajopt_solve")
    def _solve_trajopt_from_solve_state(
        self,
        goal: Goal,
        solve_state: ReacherSolveState,
        act_seed: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
        trajopt_instance: Optional[TrajOptSolver] = None,
        num_seeds_override: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization from solve state, used by motion generation planning call.

        Args:
            goal: Goal containing Pose/Joint targets, current robot state and any other information.
            solve_state: Solve state for motion generation.
            act_seed: Seed to run trajectory optimization.
            use_nn_seed: Use neural network seed for solving trajectory optimization. This is not
                implemented.
            return_all_solutions: Return all solutions found by trajectory optimization.
            seed_success: Success tensor for seeds.
            newton_iters: Override Newton iterations to run for solving trajectory optimization.
            trajopt_instance: Instance of TrajOptSolver to use for solving trajectory optimization.
            num_seeds_override: Override number of seeds to use for solving trajectory optimization.

        Returns:
            TrajOptResult: Result of trajectory optimization.
        """
        if trajopt_instance is None:
            trajopt_instance = self.trajopt_solver
        if num_seeds_override is None:
            num_seeds_override = solve_state.num_trajopt_seeds
        traj_result = trajopt_instance.solve_any(
            solve_state.solve_type,
            goal,
            act_seed,
            use_nn_seed,
            return_all_solutions,
            num_seeds_override,
            seed_success,
            newton_iters=newton_iters,
        )
        return traj_result

    def _get_solve_state(
        self,
        solve_type: ReacherSolveType,
        plan_config: MotionGenPlanConfig,
        goal_pose: Pose,
        start_state: JointState,
    ) -> ReacherSolveState:
        """Generate solve state for motion generation based on planning type and configuration.

        MotionGen creates a :class:`ReacherSolveState` for every planning call to keep track of
        planning parameters such as number of seeds, batch size, solve type, etc. This solve state
        is then compared with existing solve state to determine if solvers (IK, TrajOpt) need to be
        re-initialized. Note that changing solve state is not supported when
        :attr:`MotionGen.use_cuda_graph` is enabled.

        Args:
            solve_type: Type of reacher problem to solve.
            plan_config: Planning configuration for motion generation.
            goal_pose: Goal Pose for the end-effector.
            start_state: Start joint configuration of the robot.

        Raises:
            ValueError: If the solve type is not supported.

        Returns:
            ReacherSolveState: Solve state for motion generation.
        """

        num_ik_seeds = (
            self.ik_seeds if plan_config.num_ik_seeds is None else plan_config.num_ik_seeds
        )
        num_trajopt_seeds = (
            self.trajopt_seeds
            if plan_config.num_trajopt_seeds is None
            else plan_config.num_trajopt_seeds
        )

        num_graph_seeds = (
            self.trajopt_seeds if plan_config.num_graph_seeds is None else num_trajopt_seeds
        )
        solve_state = None
        if solve_type == ReacherSolveType.SINGLE:
            solve_state = ReacherSolveState(
                solve_type,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                batch_size=1,
                n_envs=1,
                n_goalset=1,
            )
        elif solve_type == ReacherSolveType.GOALSET:
            solve_state = ReacherSolveState(
                solve_type,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                batch_size=1,
                n_envs=1,
                n_goalset=goal_pose.n_goalset,
            )
        elif solve_type == ReacherSolveType.BATCH:
            solve_state = ReacherSolveState(
                solve_type,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                batch_size=goal_pose.batch,
                n_envs=1,
                n_goalset=1,
            )
        elif solve_type == ReacherSolveType.BATCH_GOALSET:
            solve_state = ReacherSolveState(
                solve_type,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                batch_size=goal_pose.batch,
                n_envs=1,
                n_goalset=goal_pose.n_goalset,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV:
            solve_state = ReacherSolveState(
                solve_type,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                batch_size=goal_pose.batch,
                n_envs=goal_pose.batch,
                n_goalset=1,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV_GOALSET:
            solve_state = ReacherSolveState(
                solve_type,
                num_ik_seeds=num_ik_seeds,
                num_trajopt_seeds=num_trajopt_seeds,
                num_graph_seeds=num_graph_seeds,
                batch_size=goal_pose.batch,
                n_envs=goal_pose.batch,
                n_goalset=goal_pose.n_goalset,
            )
        else:
            raise ValueError("Unsupported Solve type")
        return solve_state

    def _plan_attempts(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: List[Pose] = None,
    ):
        """Call many planning attempts for a given reacher solve state.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal pose to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of planning.
        """
        start_time = time.time()
        valid_query = True
        plan_config = plan_config.clone()
        if plan_config.check_start_validity:
            valid_query, status = self.check_start_state(start_state)
            if not valid_query:
                result = MotionGenResult(
                    success=torch.as_tensor([False], device=self.tensor_args.device),
                    valid_query=valid_query,
                    status=status,
                )
                return result
        if plan_config.pose_cost_metric is not None:
            valid_query = self.update_pose_cost_metric(
                plan_config.pose_cost_metric, start_state, goal_pose
            )
            if not valid_query:
                result = MotionGenResult(
                    success=torch.as_tensor([False], device=self.tensor_args.device),
                    valid_query=valid_query,
                    status=MotionGenStatus.INVALID_PARTIAL_POSE_COST_METRIC,
                )
                return result
        self.update_batch_size(seeds=solve_state.num_trajopt_seeds, batch=solve_state.batch_size)
        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                log_error("Batch Env is less that goal batch")
        force_graph = plan_config.enable_graph
        partial_ik = plan_config.partial_ik_opt
        ik_fail_count = 0
        time_dict = {
            "solve_time": 0,
            "ik_time": 0,
            "graph_time": 0,
            "trajopt_time": 0,
            "trajopt_attempts": 0,
        }
        best_status = 0
        if plan_config.finetune_dt_scale is None:
            plan_config.finetune_dt_scale = self.finetune_dt_scale
        for n in range(plan_config.max_attempts):
            log_info("MG Iter: " + str(n))
            result = self._plan_from_solve_state(
                solve_state,
                start_state,
                goal_pose,
                plan_config,
                link_poses,
            )
            time_dict["solve_time"] += result.solve_time
            time_dict["ik_time"] += result.ik_time
            time_dict["graph_time"] += result.graph_time
            time_dict["trajopt_time"] += result.trajopt_time
            time_dict["trajopt_attempts"] += result.trajopt_attempts
            if (
                result.status == MotionGenStatus.IK_FAIL and plan_config.ik_fail_return is not None
            ):  # IF IK fails the first time, we exist assuming the goal is not reachable
                ik_fail_count += 1
                best_status = max(best_status, 1)

                if ik_fail_count > plan_config.ik_fail_return:
                    break
            if result.success[0].item():
                break

            if result.status == MotionGenStatus.FINETUNE_TRAJOPT_FAIL:
                plan_config.finetune_dt_scale *= (
                    plan_config.finetune_dt_decay**plan_config.finetune_attempts
                )
            plan_config.finetune_dt_scale = min(plan_config.finetune_dt_scale, 1.25)
            if plan_config.enable_graph_attempt is not None and (
                n >= plan_config.enable_graph_attempt - 1
                and result.status
                in [MotionGenStatus.TRAJOPT_FAIL, MotionGenStatus.FINETUNE_TRAJOPT_FAIL]
                and not plan_config.enable_graph
            ):
                plan_config.enable_graph = True
                plan_config.partial_ik_opt = False
            if plan_config.disable_graph_attempt is not None and (
                n >= plan_config.disable_graph_attempt - 1
                and result.status
                in [
                    MotionGenStatus.TRAJOPT_FAIL,
                    MotionGenStatus.GRAPH_FAIL,
                    MotionGenStatus.FINETUNE_TRAJOPT_FAIL,
                ]
                and not force_graph
            ):
                plan_config.enable_graph = False
                plan_config.partial_ik_opt = partial_ik
            if result.status in [MotionGenStatus.TRAJOPT_FAIL]:
                best_status = 3
            elif result.status in [MotionGenStatus.GRAPH_FAIL]:
                best_status = 2
            if time.time() - start_time > plan_config.timeout:
                break
            if not result.valid_query:
                result.status = MotionGenStatus.INVALID_QUERY
                break
            if n == 10:
                self.reset_seed()
                log_warn("Couldn't find solution with 10 attempts, resetting seeds")

        result.solve_time = time_dict["solve_time"]
        result.ik_time = time_dict["ik_time"]
        result.graph_time = time_dict["graph_time"]
        result.trajopt_time = time_dict["trajopt_time"]
        result.trajopt_attempts = time_dict["trajopt_attempts"]
        result.attempts = n + 1
        torch.cuda.synchronize(device=self.tensor_args.device)
        if plan_config.pose_cost_metric is not None:
            self.update_pose_cost_metric(PoseCostMetric.reset_metric())
        if plan_config.time_dilation_factor is not None and torch.count_nonzero(result.success) > 0:
            result.retime_trajectory(
                plan_config.time_dilation_factor,
                interpolation_kind=self.finetune_trajopt_solver.interpolation_type,
            )

        result.total_time = time.time() - start_time
        return result

    def _plan_batch_attempts(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Optional[Dict[str, Pose]] = None,
    ):
        """Plan batch attempts for a given reacher solve state.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal pose to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of batched planning.
        """
        start_time = time.time()
        plan_config = plan_config.clone()
        goal_pose = goal_pose.clone()
        if plan_config.pose_cost_metric is not None:
            valid_query = self.update_pose_cost_metric(
                plan_config.pose_cost_metric, start_state, goal_pose
            )
            if not valid_query:
                result = MotionGenResult(
                    success=torch.as_tensor(
                        [False for _ in solve_state.batch_size],
                        device=self.motion_gen.tensor_args.device,
                    ),
                    valid_query=valid_query,
                    status="Invalid Hold Partial Pose",
                )
                return result

        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                log_error("Batch Env is less that goal batch")
            if plan_config.enable_graph:
                log_error("Graph Search / Geometric Planner not supported in batch_env mode")

        if plan_config.enable_graph or (
            plan_config.enable_graph_attempt is not None
            and plan_config.max_attempts >= plan_config.enable_graph_attempt
        ):
            log_warn("Batch mode enable graph is only supported with num_graph_seeds==1")
            plan_config.num_trajopt_seeds = 1
            plan_config.num_graph_seeds = 1
            solve_state.num_trajopt_seeds = 1
            solve_state.num_graph_seeds = 1
        self.update_batch_size(seeds=solve_state.num_trajopt_seeds, batch=solve_state.batch_size)

        ik_fail_count = 0
        force_graph = plan_config.enable_graph
        partial_ik = plan_config.partial_ik_opt

        time_dict = {
            "solve_time": 0,
            "ik_time": 0,
            "graph_time": 0,
            "trajopt_time": 0,
            "trajopt_attempts": 0,
        }
        best_result = None

        for n in range(plan_config.max_attempts):
            result = self._plan_from_solve_state_batch(
                solve_state,
                start_state,
                goal_pose,
                plan_config,
                link_poses=link_poses,
            )

            time_dict["solve_time"] += result.solve_time
            time_dict["ik_time"] += result.ik_time

            time_dict["graph_time"] += result.graph_time
            time_dict["trajopt_time"] += result.trajopt_time
            time_dict["trajopt_attempts"] += result.trajopt_attempts

            # if not all have succeeded, store the successful ones and re attempt:
            # TODO: update stored based on error
            if best_result is None:
                best_result = result.clone()
            else:
                # get success idx:
                idx = torch.nonzero(result.success).reshape(-1)
                if len(idx) > 0:
                    best_result.copy_idx(idx, result)

            if (
                result.status == MotionGenStatus.IK_FAIL and plan_config.ik_fail_return is not None
            ):  # IF IK fails the first time, we exit assuming the goal is not reachable
                ik_fail_count += 1
                if ik_fail_count > plan_config.ik_fail_return:
                    break

            if (
                torch.count_nonzero(best_result.success)
                >= goal_pose.batch * plan_config.success_ratio
            ):  # we want 90% targets to succeed
                best_result.status = None
                break
            if plan_config.enable_graph_attempt is not None and (
                n >= plan_config.enable_graph_attempt - 1
                and result.status != MotionGenStatus.IK_FAIL
                and not plan_config.enable_graph
            ):
                plan_config.enable_graph = True
                plan_config.partial_ik_opt = False

            if plan_config.disable_graph_attempt is not None and (
                n >= plan_config.disable_graph_attempt - 1
                and result.status in [MotionGenStatus.TRAJOPT_FAIL, MotionGenStatus.GRAPH_FAIL]
                and not force_graph
            ):
                plan_config.enable_graph = False
                plan_config.partial_ik_opt = partial_ik

            if plan_config.fail_on_invalid_query:
                if not result.valid_query:
                    best_result.valid_query = False
                    best_result.status = "Invalid Problem"
                    break
            if time.time() - start_time > plan_config.timeout:
                break
        best_result.solve_time = time_dict["solve_time"]
        best_result.ik_time = time_dict["ik_time"]
        best_result.graph_time = time_dict["graph_time"]
        best_result.trajopt_time = time_dict["trajopt_time"]
        best_result.trajopt_attempts = time_dict["trajopt_attempts"]
        best_result.attempts = n + 1
        torch.cuda.synchronize(device=self.tensor_args.device)
        if plan_config.pose_cost_metric is not None:
            self.update_pose_cost_metric(PoseCostMetric.reset_metric())

        if plan_config.time_dilation_factor is not None and torch.count_nonzero(result.success) > 0:
            result.retime_trajectory(
                plan_config.time_dilation_factor,
                interpolation_kind=self.finetune_trajopt_solver.interpolation_type,
            )
        best_result.total_time = time.time() - start_time
        return best_result

    def _plan_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> MotionGenResult:
        """Plan from a given reacher solve state.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal pose to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of planning.
        """
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = None
        graph_success = 0

        if len(start_state.shape) == 1:
            log_error("Joint state should be not a vector (dof) should be (bxdof)")

        if goal_pose.shape[0] != 1:
            log_error(
                "Goal position should be of shape [1, n_goalset, -1], current shape: "
                + str(goal_pose.shape)
            )
        # plan ik:

        ik_result = self._solve_ik_from_solve_state(
            goal_pose,
            solve_state,
            start_state,
            plan_config.use_nn_ik_seed,
            plan_config.partial_ik_opt,
            link_poses,
        )

        if not plan_config.enable_graph and plan_config.partial_ik_opt:
            ik_result.success[:] = True

        # check for success:
        result = MotionGenResult(
            ik_result.success.view(-1)[0:1],  # This is not true for batch mode
            ik_time=ik_result.solve_time,
            solve_time=ik_result.solve_time,
        )

        if self.store_debug_in_result:
            result.debug_info = {"ik_result": ik_result}
        ik_success = torch.count_nonzero(ik_result.success)
        if ik_success == 0:
            result.status = MotionGenStatus.IK_FAIL
            return result

        # do graph search:
        with profiler.record_function("motion_gen/post_ik"):
            ik_out_seeds = solve_state.num_trajopt_seeds
            if plan_config.enable_graph:
                ik_out_seeds = min(solve_state.num_trajopt_seeds, ik_success)

            goal_config = ik_result.solution[ik_result.success].view(-1, self.ik_solver.dof)[
                :ik_success
            ]
            start_config = tensor_repeat_seeds(start_state.position, ik_out_seeds)
            if plan_config.enable_opt:
                self._trajopt_goal_config[:] = ik_result.solution

        # do graph search:
        if plan_config.enable_graph:
            interpolation_steps = None
            if plan_config.enable_opt:
                interpolation_steps = self.trajopt_solver.action_horizon
            log_info("MG: running GP")
            graph_result = self.graph_search(start_config, goal_config, interpolation_steps)
            trajopt_seed_success = graph_result.success

            graph_success = torch.count_nonzero(graph_result.success).item()
            result.graph_time = graph_result.solve_time
            result.solve_time += graph_result.solve_time
            if graph_success > 0:
                log_info("MG: GP Success")
                result.graph_plan = graph_result.interpolated_plan
                result.interpolated_plan = graph_result.interpolated_plan

                result.used_graph = True
                if plan_config.enable_opt:
                    trajopt_seed = (
                        result.graph_plan.position.view(
                            1,  # solve_state.batch_size,
                            graph_success,  # solve_state.num_trajopt_seeds,
                            interpolation_steps,
                            self._dof,
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )
                    trajopt_seed_traj = torch.zeros(
                        (trajopt_seed.shape[0], 1, self.trajopt_solver.action_horizon, self._dof),
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    trajopt_seed_traj[:, :, :interpolation_steps, :] = trajopt_seed
                    trajopt_seed_success = ik_result.success.clone()
                    trajopt_seed_success[ik_result.success] = graph_result.success

                    trajopt_seed_success = trajopt_seed_success.view(
                        solve_state.batch_size, solve_state.num_trajopt_seeds
                    )
                    trajopt_newton_iters = self.graph_trajopt_iters
                else:
                    _, idx = torch.topk(
                        graph_result.path_length[graph_result.success], k=1, largest=False
                    )
                    result.interpolated_plan = result.interpolated_plan[idx].squeeze(0)
                    result.optimized_dt = self.tensor_args.to_device(self.interpolation_dt)
                    result.optimized_plan = result.interpolated_plan[
                        : graph_result.path_buffer_last_tstep[idx.item()]
                    ]
                    idx = idx.view(-1) + self._batch_col
                    result.position_error = ik_result.position_error[ik_result.success][
                        graph_result.success
                    ][idx]
                    result.rotation_error = ik_result.rotation_error[ik_result.success][
                        graph_result.success
                    ][idx]
                    result.path_buffer_last_tstep = graph_result.path_buffer_last_tstep[
                        idx.item() : idx.item() + 1
                    ]
                    result.success = result.success.view(-1)[0:1]
                    result.success[:] = True
                    return result
            else:
                result.success[:] = False
                result.status = MotionGenStatus.GRAPH_FAIL
                if not graph_result.valid_query:
                    result.valid_query = False
                    if self.store_debug_in_result:
                        result.debug_info["graph_debug"] = graph_result.debug_info
                    return result
                if plan_config.need_graph_success:
                    return result

        # do trajopt::

        if plan_config.enable_opt:
            with profiler.record_function("motion_gen/setup_trajopt_seeds"):
                self._trajopt_goal_config[:, :ik_success] = goal_config
                retract_config = None
                if plan_config.use_start_state_as_retract:
                    retract_config = start_state.position.clone()
                goal = Goal(
                    goal_pose=goal_pose,
                    current_state=start_state,
                    links_goal_pose=link_poses,
                    retract_state=retract_config,
                )

                if (
                    trajopt_seed_traj is None
                    or graph_success < solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds
                ):
                    goal_config = self._trajopt_goal_config[0]  # batch index == 0

                    goal_state = JointState.from_position(
                        goal_config,
                    )
                    seed_link_poses = None
                    if link_poses is not None:
                        seed_link_poses = {}

                        for k in link_poses.keys():
                            seed_link_poses[k] = link_poses[k].repeat_seeds(
                                solve_state.num_trajopt_seeds
                            )
                    if goal_pose.shape[0] != 1:
                        log_error(
                            "Batch size of goal pose should be 1, current shape: "
                            + str(goal_pose.shape)
                        )
                    seed_goal = Goal(
                        goal_pose=goal_pose.repeat_seeds(solve_state.num_trajopt_seeds),
                        current_state=start_state.repeat_seeds(solve_state.num_trajopt_seeds),
                        goal_state=goal_state,
                        links_goal_pose=seed_link_poses,
                    )
                    if trajopt_seed_traj is not None:
                        trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()
                        # batch, num_seeds, h, dof
                        if (
                            trajopt_seed_success.shape[1]
                            < solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds
                        ):
                            trajopt_seed_success_new = torch.zeros(
                                (1, solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds),
                                device=self.tensor_args.device,
                                dtype=torch.bool,
                            )
                            trajopt_seed_success_new[0, : trajopt_seed_success.shape[1]] = (
                                trajopt_seed_success
                            )
                            trajopt_seed_success = trajopt_seed_success_new
                    # create seeds here:
                    trajopt_seed_traj = self.trajopt_solver.get_seed_set(
                        seed_goal,
                        trajopt_seed_traj,  # batch, num_seeds, h, dof
                        num_seeds=self.noisy_trajopt_seeds,
                        batch_mode=solve_state.batch_mode,
                        seed_success=trajopt_seed_success,
                    )
                    trajopt_seed_traj = trajopt_seed_traj.view(
                        solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds,
                        solve_state.batch_size,
                        self.trajopt_solver.action_horizon,
                        self._dof,
                    ).contiguous()
            if plan_config.enable_finetune_trajopt:
                og_value = self.trajopt_solver.interpolation_type
                self.trajopt_solver.interpolation_type = InterpolateType.LINEAR_CUDA
            with profiler.record_function("motion_gen/trajopt"):
                log_info("MG: running TO")
                traj_result = self._solve_trajopt_from_solve_state(
                    goal,
                    solve_state,
                    trajopt_seed_traj,
                    num_seeds_override=solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds,
                    newton_iters=trajopt_newton_iters,
                    return_all_solutions=plan_config.parallel_finetune
                    and plan_config.enable_finetune_trajopt,
                )
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
            if self.store_debug_in_result:
                result.debug_info["trajopt_result"] = traj_result
            # run finetune
            if plan_config.enable_finetune_trajopt and torch.count_nonzero(traj_result.success) > 0:
                with profiler.record_function("motion_gen/finetune_trajopt"):
                    seed_traj = traj_result.raw_action.clone()  # solution.position.clone()
                    seed_traj = seed_traj.contiguous()
                    og_solve_time = traj_result.solve_time
                    seed_override = 1
                    opt_dt = traj_result.optimized_dt

                    if plan_config.parallel_finetune:
                        seed_override = solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds
                        if self.optimize_dt:
                            opt_dt = torch.min(opt_dt[traj_result.success])

                    finetune_time = 0
                    newton_iters = None

                    for k in range(plan_config.finetune_attempts):
                        if self.optimize_dt:

                            scaled_dt = torch.clamp(
                                opt_dt
                                * plan_config.finetune_dt_scale
                                * (plan_config.finetune_dt_decay ** (k)),
                                self.trajopt_solver.minimum_trajectory_dt,
                            )
                            self.finetune_trajopt_solver.update_solver_dt(scaled_dt.item())

                        traj_result = self._solve_trajopt_from_solve_state(
                            goal,
                            solve_state,
                            seed_traj,
                            trajopt_instance=self.finetune_trajopt_solver,
                            num_seeds_override=seed_override,
                            newton_iters=newton_iters,
                        )
                        finetune_time += traj_result.solve_time
                        if torch.count_nonzero(traj_result.success) > 0 or not self.optimize_dt:
                            break
                        seed_traj = traj_result.optimized_seeds.detach().clone()
                        newton_iters = 4

                    traj_result.solve_time = finetune_time

                result.finetune_time = traj_result.solve_time

                traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
            elif plan_config.enable_finetune_trajopt:
                traj_result.success = traj_result.success[0:1]
                # if torch.count_nonzero(result.success) == 0:
                result.status = MotionGenStatus.TRAJOPT_FAIL
            result.solve_time += traj_result.solve_time + result.finetune_time
            result.trajopt_time = traj_result.solve_time
            result.trajopt_attempts = 1
            result.success = traj_result.success

            if plan_config.enable_finetune_trajopt and torch.count_nonzero(result.success) == 0:

                result.status = MotionGenStatus.FINETUNE_TRAJOPT_FAIL
                if (
                    traj_result.debug_info is not None
                    and "dt_exception" in traj_result.debug_info
                    and traj_result.debug_info["dt_exception"]
                ):
                    result.status = MotionGenStatus.DT_EXCEPTION

            result.interpolated_plan = traj_result.interpolated_solution.trim_trajectory(
                0, traj_result.path_buffer_last_tstep[0]
            )
            result.interpolation_dt = self.trajopt_solver.interpolation_dt
            result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
            result.position_error = traj_result.position_error
            result.rotation_error = traj_result.rotation_error
            result.optimized_dt = traj_result.optimized_dt
            result.optimized_plan = traj_result.solution
            result.goalset_index = traj_result.goalset_index
        return result

    def _plan_js_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_state: JointState,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    ) -> MotionGenResult:
        """Plan from a given reacher solve state for joint state.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_state: Goal joint state to reach.
            plan_config: Planning parameters for motion generation.

        Returns:
            MotionGenResult: Result of planning.
        """
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = self.js_trajopt_solver.newton_iters

        graph_success = 0
        if len(start_state.shape) == 1:
            log_error("Joint state should be not a vector (dof) should be (bxdof)")

        result = MotionGenResult(cspace_error=torch.zeros((1), device=self.tensor_args.device))
        if self.store_debug_in_result:
            result.debug_info = {}
        # do graph search:
        if plan_config.enable_graph:
            start_config = torch.zeros(
                (solve_state.num_graph_seeds, self.js_trajopt_solver.dof),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            goal_config = start_config.clone()
            start_config[:] = start_state.position
            goal_config[:] = goal_state.position
            interpolation_steps = None
            if plan_config.enable_opt:
                interpolation_steps = self.js_trajopt_solver.action_horizon
            log_info("MG: running GP")
            graph_result = self.graph_search(start_config, goal_config, interpolation_steps)
            trajopt_seed_success = graph_result.success

            graph_success = torch.count_nonzero(graph_result.success).item()
            result.graph_time = graph_result.solve_time
            result.solve_time += graph_result.solve_time
            if graph_success > 0:
                result.graph_plan = graph_result.interpolated_plan
                result.interpolated_plan = graph_result.interpolated_plan

                result.used_graph = True
                if plan_config.enable_opt:
                    trajopt_seed = (
                        result.graph_plan.position.view(
                            1,  # solve_state.batch_size,
                            graph_success,  # solve_state.num_trajopt_seeds,
                            interpolation_steps,
                            self._dof,
                        )
                        .transpose(0, 1)
                        .contiguous()
                    )
                    trajopt_seed_traj = torch.zeros(
                        (trajopt_seed.shape[0], 1, self.trajopt_solver.action_horizon, self._dof),
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    trajopt_seed_traj[:, :, :interpolation_steps, :] = trajopt_seed
                    trajopt_seed_success = graph_result.success

                    trajopt_seed_success = trajopt_seed_success.view(
                        1, solve_state.num_trajopt_seeds
                    )
                    trajopt_newton_iters = self.graph_trajopt_iters
                else:
                    _, idx = torch.topk(
                        graph_result.path_length[graph_result.success], k=1, largest=False
                    )
                    result.interpolated_plan = result.interpolated_plan[idx].squeeze(0)
                    result.optimized_dt = self.tensor_args.to_device(self.interpolation_dt)
                    result.optimized_plan = result.interpolated_plan[
                        : graph_result.path_buffer_last_tstep[idx.item()]
                    ]
                    idx = idx.view(-1) + self._batch_col
                    result.cspace_error = torch.zeros((1), device=self.tensor_args.device)

                    result.path_buffer_last_tstep = graph_result.path_buffer_last_tstep[
                        idx.item() : idx.item() + 1
                    ]
                    result.success = torch.as_tensor([True], device=self.tensor_args.device)
                    return result
            else:
                result.success = torch.as_tensor([False], device=self.tensor_args.device)
                result.status = MotionGenStatus.GRAPH_FAIL
                if not graph_result.valid_query:
                    result.valid_query = False
                    if self.store_debug_in_result:
                        result.debug_info["graph_debug"] = graph_result.debug_info
                    return result
                if plan_config.need_graph_success:
                    return result

        # do trajopt:
        if plan_config.enable_opt:
            with profiler.record_function("motion_gen/setup_trajopt_seeds"):

                goal = Goal(
                    current_state=start_state,
                    goal_state=goal_state,
                )

                if trajopt_seed_traj is None or graph_success < solve_state.num_trajopt_seeds * 1:
                    seed_goal = Goal(
                        current_state=start_state.repeat_seeds(solve_state.num_trajopt_seeds),
                        goal_state=goal_state.repeat_seeds(solve_state.num_trajopt_seeds),
                    )
                    if trajopt_seed_traj is not None:
                        trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()
                        # batch, num_seeds, h, dof
                        if trajopt_seed_success.shape[1] < self.js_trajopt_solver.num_seeds:
                            trajopt_seed_success_new = torch.zeros(
                                (1, solve_state.num_trajopt_seeds),
                                device=self.tensor_args.device,
                                dtype=torch.bool,
                            )
                            trajopt_seed_success_new[0, : trajopt_seed_success.shape[1]] = (
                                trajopt_seed_success
                            )
                            trajopt_seed_success = trajopt_seed_success_new
                    # create seeds here:
                    trajopt_seed_traj = self.js_trajopt_solver.get_seed_set(
                        seed_goal,
                        trajopt_seed_traj,  # batch, num_seeds, h, dof
                        num_seeds=1,
                        batch_mode=False,
                        seed_success=trajopt_seed_success,
                    )
                    trajopt_seed_traj = (
                        trajopt_seed_traj.view(
                            self.js_trajopt_solver.num_seeds * 1,
                            1,
                            self.trajopt_solver.action_horizon,
                            self._dof,
                        )
                        .contiguous()
                        .clone()
                    )
            if plan_config.enable_finetune_trajopt:
                og_value = self.trajopt_solver.interpolation_type
                self.js_trajopt_solver.interpolation_type = InterpolateType.LINEAR_CUDA
            with profiler.record_function("motion_gen/trajopt"):
                log_info("MG: running TO")
                traj_result = self._solve_trajopt_from_solve_state(
                    goal,
                    solve_state,
                    trajopt_seed_traj,
                    num_seeds_override=solve_state.num_trajopt_seeds,
                    newton_iters=trajopt_newton_iters,
                    return_all_solutions=plan_config.enable_finetune_trajopt,
                    trajopt_instance=self.js_trajopt_solver,
                )
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
            if self.store_debug_in_result:
                result.debug_info["trajopt_result"] = traj_result
            if torch.count_nonzero(traj_result.success) == 0:
                result.status = MotionGenStatus.TRAJOPT_FAIL
            # run finetune
            if plan_config.enable_finetune_trajopt and torch.count_nonzero(traj_result.success) > 0:
                with profiler.record_function("motion_gen/finetune_trajopt"):
                    seed_traj = traj_result.raw_action.clone()
                    og_solve_time = traj_result.solve_time
                    opt_dt = traj_result.optimized_dt
                    opt_dt = torch.min(opt_dt[traj_result.success])
                    finetune_time = 0
                    newton_iters = None
                    for k in range(plan_config.finetune_attempts):

                        scaled_dt = torch.clamp(
                            opt_dt
                            * plan_config.finetune_js_dt_scale
                            * (plan_config.finetune_dt_decay ** (k)),
                            self.js_trajopt_solver.minimum_trajectory_dt,
                        )

                        if self.optimize_dt:
                            self.finetune_js_trajopt_solver.update_solver_dt(scaled_dt.item())
                        traj_result = self._solve_trajopt_from_solve_state(
                            goal,
                            solve_state,
                            seed_traj,
                            trajopt_instance=self.finetune_js_trajopt_solver,
                            num_seeds_override=solve_state.num_trajopt_seeds,
                            newton_iters=newton_iters,
                            return_all_solutions=False,
                        )

                        finetune_time += traj_result.solve_time
                        if torch.count_nonzero(traj_result.success) > 0 or not self.optimize_dt:
                            break
                        seed_traj = traj_result.optimized_seeds.detach().clone()
                        newton_iters = 4

                    result.finetune_time = finetune_time

                    traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
                if torch.count_nonzero(traj_result.success) == 0:
                    result.status = MotionGenStatus.FINETUNE_TRAJOPT_FAIL
                    if (
                        traj_result.debug_info is not None
                        and "dt_exception" in traj_result.debug_info
                        and traj_result.debug_info["dt_exception"]
                    ):
                        result.status = MotionGenStatus.DT_EXCEPTION

            elif plan_config.enable_finetune_trajopt:
                traj_result.success = traj_result.success[0:1]
            result.solve_time += traj_result.solve_time + result.finetune_time
            result.trajopt_time = traj_result.solve_time
            result.trajopt_attempts = 1
            result.success = traj_result.success
            result.interpolated_plan = traj_result.interpolated_solution.trim_trajectory(
                0, traj_result.path_buffer_last_tstep[0]
            )

            result.interpolation_dt = self.trajopt_solver.interpolation_dt
            result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
            result.cspace_error = traj_result.cspace_error
            result.optimized_dt = traj_result.optimized_dt
            result.optimized_plan = traj_result.solution
            result.goalset_index = traj_result.goalset_index

        return result

    def _plan_from_solve_state_batch(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> MotionGenResult:
        """Plan from a given reacher solve state in batch mode.

        Args:
            solve_state: Reacher solve state for planning.
            start_state: Start joint state for planning.
            goal_pose: Goal poses to reach for end-effector.
            plan_config: Planning parameters for motion generation.
            link_poses: Goal poses for other links in the robot.

        Returns:
            MotionGenResult: Result of planning.
        """
        self._trajopt_goal_config[:] = self.get_retract_config().view(1, 1, self._dof)
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = None
        graph_success = 0

        # plan ik:
        ik_result = self._solve_ik_from_solve_state(
            goal_pose,
            solve_state,
            start_state,
            plan_config.use_nn_ik_seed,
            plan_config.partial_ik_opt,
            link_poses,
        )

        if not plan_config.enable_graph and plan_config.partial_ik_opt:
            ik_result.success[:] = True

        # check for success:
        result = MotionGenResult(
            ik_result.success,
            position_error=ik_result.position_error,
            rotation_error=ik_result.rotation_error,
            ik_time=ik_result.solve_time,
            solve_time=ik_result.solve_time,
            debug_info={},
            # goalset_index=ik_result.goalset_index,
        )

        ik_success = torch.count_nonzero(ik_result.success)
        if ik_success == 0:
            result.status = MotionGenStatus.IK_FAIL
            result.success = result.success[:, 0]
            return result

        # do graph search:
        ik_out_seeds = solve_state.num_trajopt_seeds
        if plan_config.enable_graph:
            ik_out_seeds = min(solve_state.num_trajopt_seeds, ik_success)

        # if not plan_config.enable_opt and plan_config.enable_graph:
        #    self.graph_planner.interpolation_steps = self.interpolation_steps
        #    self.graph_planner.interpolation_type = self.interpolation_type
        # elif plan_config.enable_graph:
        #    self.graph_planner.interpolation_steps = self.trajopt_solver.traj_tsteps
        #    self.graph_planner.interpolation_type = InterpolateType.LINEAR
        goal_config = ik_result.solution[ik_result.success].view(-1, self.ik_solver.dof)

        # get shortest path
        if plan_config.enable_graph:
            interpolation_steps = None
            if plan_config.enable_opt:
                interpolation_steps = self.trajopt_solver.action_horizon

            start_graph_state = start_state.repeat_seeds(ik_out_seeds)
            start_config = start_graph_state.position[ik_result.success.view(-1)].view(
                -1, self.ik_solver.dof
            )
            graph_result = self.graph_search(start_config, goal_config, interpolation_steps)
            graph_success = torch.count_nonzero(graph_result.success).item()

            result.graph_time = graph_result.solve_time
            result.solve_time += graph_result.solve_time
            if graph_success > 0:
                # path = graph_result.interpolated_plan
                result.graph_plan = graph_result.interpolated_plan
                result.interpolated_plan = graph_result.interpolated_plan
                result.used_graph = True

                if plan_config.enable_opt:
                    trajopt_seed = result.graph_plan.position.view(
                        graph_success,  # solve_state.num_trajopt_seeds,
                        interpolation_steps,
                        self._dof,
                    ).contiguous()
                    trajopt_seed_traj = torch.zeros(
                        (1, trajopt_seed.shape[0], self.trajopt_solver.action_horizon, self._dof),
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    trajopt_seed_traj[0, :, :interpolation_steps, :] = trajopt_seed
                    trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()
                    trajopt_seed_success = ik_result.success.clone()
                    trajopt_seed_success[ik_result.success] = graph_result.success

                    trajopt_seed_success = trajopt_seed_success.view(
                        solve_state.num_trajopt_seeds, solve_state.batch_size
                    )
                    trajopt_newton_iters = self.graph_trajopt_iters

                else:
                    ik_success = ik_result.success.view(-1).clone()

                    # only some might be successful:

                    g_dim = torch.nonzero(ik_success).view(-1)[graph_result.success]

                    self._batch_graph_search_buffer.copy_at_index(
                        graph_result.interpolated_plan, g_dim
                    )

                    # result.graph_plan = JointState.from_position(
                    #    self._batch_graph_search_buffer,
                    #    joint_names=graph_result.interpolated_plan.joint_names,
                    # )
                    result.interpolated_plan = self._batch_graph_search_buffer
                    g_dim = g_dim.cpu().squeeze().tolist()
                    if isinstance(g_dim, int):
                        g_dim = [g_dim]
                    for x, x_val in enumerate(g_dim):
                        self._batch_path_buffer_last_tstep[x_val] = (
                            graph_result.path_buffer_last_tstep[x]
                        )
                    result.path_buffer_last_tstep = self._batch_path_buffer_last_tstep
                    result.optimized_plan = result.interpolated_plan
                    result.optimized_dt = torch.as_tensor(
                        [
                            self.interpolation_dt
                            for i in range(result.interpolated_plan.position.shape[0])
                        ],
                        device=self.tensor_args.device,
                        dtype=self.tensor_args.dtype,
                    )
                    result.success = result.success.view(-1).clone()
                    result.success[ik_success][graph_result.success] = True
                    return result

            else:
                result.success[:] = False
                result.success = result.success[:, 0]
                result.status = MotionGenStatus.GRAPH_FAIL
                if not graph_result.valid_query:
                    result.valid_query = False
                    if self.store_debug_in_result:
                        result.debug_info = {"graph_debug": graph_result.debug_info}
                    return result

        if plan_config.enable_opt:
            # get goal configs based on ik success:
            self._trajopt_goal_config[ik_result.success] = goal_config

            goal_config = self._trajopt_goal_config  # batch index == 0

            goal = Goal(
                goal_pose=goal_pose,
                current_state=start_state,
                links_goal_pose=link_poses,
            )
            # generate seeds:
            if trajopt_seed_traj is None or (
                plan_config.enable_graph and graph_success < solve_state.batch_size
            ):
                seed_link_poses = None
                if link_poses is not None:
                    seed_link_poses = {}

                    for k in link_poses.keys():
                        seed_link_poses[k] = link_poses[k].repeat_seeds(
                            solve_state.num_trajopt_seeds
                        )
                seed_goal = Goal(
                    goal_pose=goal_pose.repeat_seeds(solve_state.num_trajopt_seeds),
                    current_state=start_state.repeat_seeds(solve_state.num_trajopt_seeds),
                    goal_state=JointState.from_position(goal_config.view(-1, self._dof)),
                    links_goal_pose=seed_link_poses,
                )
                if trajopt_seed_traj is not None:
                    trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()

                # create seeds here:
                trajopt_seed_traj = self.trajopt_solver.get_seed_set(
                    seed_goal,
                    trajopt_seed_traj,  # batch, num_seeds, h, dof
                    num_seeds=1,
                    batch_mode=solve_state.batch_mode,
                    seed_success=trajopt_seed_success,
                )
                trajopt_seed_traj = trajopt_seed_traj.view(
                    solve_state.num_trajopt_seeds,
                    solve_state.batch_size,
                    self.trajopt_solver.action_horizon,
                    self._dof,
                ).contiguous()
            if plan_config.enable_finetune_trajopt:
                og_value = self.trajopt_solver.interpolation_type
                self.trajopt_solver.interpolation_type = InterpolateType.LINEAR_CUDA

            traj_result = self._solve_trajopt_from_solve_state(
                goal,
                solve_state,
                trajopt_seed_traj,
                newton_iters=trajopt_newton_iters,
                return_all_solutions=plan_config.enable_finetune_trajopt,
            )

            # output of traj result will have 1 solution per batch

            # run finetune opt on 1 solution per batch:
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
            if self.store_debug_in_result:
                result.debug_info["trajopt_result"] = traj_result

            # run finetune
            if plan_config.enable_finetune_trajopt and torch.count_nonzero(traj_result.success) > 0:
                with profiler.record_function("motion_gen/finetune_trajopt"):
                    seed_traj = traj_result.raw_action.clone()  # solution.position.clone()
                    seed_traj = seed_traj.contiguous()
                    og_solve_time = traj_result.solve_time

                    scaled_dt = torch.clamp(
                        torch.max(traj_result.optimized_dt[traj_result.success])
                        * self.finetune_dt_scale,
                        self.trajopt_solver.minimum_trajectory_dt,
                    )
                    self.finetune_trajopt_solver.update_solver_dt(scaled_dt.item())

                    traj_result = self._solve_trajopt_from_solve_state(
                        goal,
                        solve_state,
                        seed_traj,
                        trajopt_instance=self.finetune_trajopt_solver,
                        num_seeds_override=solve_state.num_trajopt_seeds,
                    )

                result.finetune_time = traj_result.solve_time

                traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
            elif plan_config.enable_finetune_trajopt and len(traj_result.success.shape) == 2:
                traj_result.success = traj_result.success[:, 0]

            result.success = traj_result.success

            result.interpolated_plan = traj_result.interpolated_solution
            result.solve_time += traj_result.solve_time
            result.trajopt_time = traj_result.solve_time
            result.position_error = traj_result.position_error
            result.rotation_error = traj_result.rotation_error
            result.cspace_error = traj_result.cspace_error
            result.goalset_index = traj_result.goalset_index
            result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
            result.optimized_plan = traj_result.solution
            result.optimized_dt = traj_result.optimized_dt
            if torch.count_nonzero(traj_result.success) == 0:
                result.status = MotionGenStatus.TRAJOPT_FAIL
                result.success[:] = False
            if self.store_debug_in_result:
                result.debug_info = {"trajopt_result": traj_result}
        return result

    def plan(
        self,
        start_state: JointState,
        goal_pose: Pose,
        enable_graph: bool = True,
        enable_opt: bool = True,
        use_nn_ik_seed: bool = False,
        need_graph_success: bool = False,
        max_attempts: int = 100,
        timeout: float = 10.0,
        enable_graph_attempt: Optional[int] = None,
        disable_graph_attempt: Optional[int] = None,
        trajopt_attempts: int = 1,
        ik_fail_return: Optional[int] = None,
        partial_ik_opt: bool = False,
        num_ik_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
    ):
        """Deprecated method. Use :meth:`MotionGen.plan_single` or others instead."""

        log_warn("Deprecated method. Use MotionGen.plan_single or others instead.")
        plan_config = MotionGenPlanConfig(
            enable_graph,
            enable_opt,
            use_nn_ik_seed,
            need_graph_success,
            max_attempts,
            timeout,
            enable_graph_attempt,
            disable_graph_attempt,
            ik_fail_return,
            partial_ik_opt,
            num_ik_seeds,
            num_graph_seeds,
            num_trajopt_seeds,
        )
        result = self.plan_single(start_state, goal_pose, plan_config)
        return result

    def batch_plan(
        self,
        start_state: JointState,
        goal_pose: Pose,
        enable_graph: bool = True,
        enable_opt: bool = True,
        use_nn_ik_seed: bool = False,
        need_graph_success: bool = False,
        max_attempts: int = 1,
        timeout: float = 10.0,
        enable_graph_attempt: Optional[int] = None,
        disable_graph_attempt: Optional[int] = None,
        success_ratio: float = 1.0,
        ik_fail_return: Optional[int] = None,
        fail_on_invalid_query: bool = False,
        partial_ik_opt: bool = False,
        num_ik_seeds: Optional[int] = None,
        num_graph_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
    ):
        """Deprecated method. Use :meth:`MotionGen.plan_batch` or others instead."""

        log_warn("Deprecated method. Use MotionGen.plan_batch or others instead.")

        plan_config = MotionGenPlanConfig(
            enable_graph,
            enable_opt,
            use_nn_ik_seed,
            need_graph_success,
            max_attempts,
            timeout,
            enable_graph_attempt,
            disable_graph_attempt,
            ik_fail_return,
            partial_ik_opt,
            num_ik_seeds,
            num_graph_seeds,
            num_trajopt_seeds,
            success_ratio=success_ratio,
            fail_on_invalid_query=fail_on_invalid_query,
        )
        result = self.plan_batch(start_state, goal_pose, plan_config)
        return result

    def toggle_link_collision(self, collision_link_names: List[str], enable_flag: bool):
        if len(collision_link_names) > 0:
            if enable_flag:
                for k in collision_link_names:
                    self.kinematics.kinematics_config.enable_link_spheres(k)
            else:
                for k in collision_link_names:
                    self.kinematics.kinematics_config.disable_link_spheres(k)

    def plan_grasp(
        self,
        start_state: JointState,
        grasp_poses: Pose,
        plan_config: MotionGenPlanConfig,
        grasp_approach_offset: Pose = Pose.from_list([0, 0, -0.15, 1, 0, 0, 0]),
        grasp_approach_path_constraint: Union[None, List[float]] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
        retract_offset: Pose = Pose.from_list([0, 0, -0.15, 1, 0, 0, 0]),
        retract_path_constraint: Union[None, List[float]] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
        disable_collision_links: List[str] = [],
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_retract: bool = True,
        grasp_approach_constraint_in_goal_frame: bool = True,
        retract_constraint_in_goal_frame: bool = True,
    ) -> GraspPlanResult:
        """Plan a sequence of motions to grasp an object, given a set of grasp poses.

        This function plans three motions, first approaches the object with an offset, then
        moves with linear constraints to the grasp pose, and finally retracts the arm base to
        offset with linear constraints. During the linear constrained motions, collision between
        disable_collision_links and the world is disabled. This disabling is useful to enable
        contact between a robot's gripper links and the object.

        This method takes a set of grasp poses and finds the best grasp pose to reach based on a
        goal set trajectory optimization problem. In this problem, the robot needs to reach one
        of the poses in the grasp_poses set at the terminal state. To allow for in-contact grasps,
        collision between disable_collision_links and world is disabled during the optimization.
        The best grasp pose is then used to plan the three motions.

        Args:
            start_state: Start joint state for planning.
            grasp_poses: Set of grasp poses, represented with :class:~curobo.math.types.Pose, of
                shape (1, num_grasps, 7).
            plan_config: Planning parameters for motion generation.
            grasp_approach_offset: Offset pose from the grasp pose. Reference frame is the grasp
                pose frame if grasp_approach_constraint_in_goal_frame is True, otherwise the
                reference frame is the robot base frame.
            grasp_approach_path_constraint: Path constraint for the approach to grasp pose and
                grasp to retract path. This is a list of 6 values, where each value is a weight
                for each Cartesian dimension. The first three are for orientation and the last
                three are for position. If None, no path constraint is applied.
            retract_offset: Retract offset pose from grasp pose. Reference frame is the grasp pose
                frame if retract_constraint_in_goal_frame is True, otherwise the reference frame is
                the robot base frame.
            retract_path_constraint: Path constraint for the retract path. This is a list of 6
                values, where each value is a weight for each Cartesian dimension. The first three
                are for orientation and the last three are for position. If None, no path
                constraint is applied.
            disable_collision_links: Name of links to disable collision with the world during
                the approach to grasp and grasp to retract path.
            plan_approach_to_grasp: If True, planning also includes moving from approach to
                grasp. If False, a plan to reach offset of the best grasp pose is returned.
            plan_grasp_to_retract: If True, planning also includes moving from grasp to retract.
                If False, only a plan to reach the best grasp pose is returned.
            grasp_approach_constraint_in_goal_frame: If True, the grasp approach offset is in the
                grasp pose frame. If False, the grasp approach offset is in the robot base frame.
                Also applies to grasp_approach_path_constraint.
            retract_constraint_in_goal_frame: If True, the retract offset is in the grasp pose
                frame. If False, the retract offset is in the robot base frame. Also applies to
                retract_path_constraint.

        Returns:
            GraspPlanResult: Result of planning. Use :meth:`GraspPlanResult.grasp_trajectory` to
                get the trajectory to reach the grasp pose and
                :meth:`GraspPlanResult.retract_trajectory` to get the trajectory to retract from
                the grasp pose.
        """
        if plan_config.pose_cost_metric is not None:
            log_error("plan_config.pose_cost_metric should be None")
        self.toggle_link_collision(disable_collision_links, False)
        result = GraspPlanResult()
        goalset_motion_gen_result = self.plan_goalset(
            start_state,
            grasp_poses,
            plan_config,
        )
        self.toggle_link_collision(disable_collision_links, True)
        result.success = goalset_motion_gen_result.success.clone()
        result.success[:] = False
        result.goalset_result = goalset_motion_gen_result
        if not goalset_motion_gen_result.success.item():
            result.status = "No grasp in goal set was reachable."
            return result
        result.goalset_index = goalset_motion_gen_result.goalset_index.clone()

        # plan to offset:
        goal_index = goalset_motion_gen_result.goalset_index.item()
        goal_pose = grasp_poses.get_index(0, goal_index).clone()
        if grasp_approach_constraint_in_goal_frame:
            offset_goal_pose = goal_pose.clone().multiply(grasp_approach_offset)
        else:
            offset_goal_pose = grasp_approach_offset.clone().multiply(goal_pose.clone())

        reach_offset_mg_result = self.plan_single(
            start_state,
            offset_goal_pose,
            plan_config.clone(),
        )
        result.approach_result = reach_offset_mg_result
        if not reach_offset_mg_result.success.item():
            result.status = f"Planning to Approach pose failed: {reach_offset_mg_result.status}"
            return result

        if not plan_approach_to_grasp:
            result.grasp_trajectory = reach_offset_mg_result.optimized_plan
            result.grasp_trajectory_dt = reach_offset_mg_result.optimized_dt
            result.grasp_interpolated_trajectory = reach_offset_mg_result.get_interpolated_plan()
            result.grasp_interpolation_dt = reach_offset_mg_result.interpolation_dt
            return result
        # plan to final grasp
        if grasp_approach_path_constraint is not None:
            hold_pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device(grasp_approach_path_constraint),
                project_to_goal_frame=grasp_approach_constraint_in_goal_frame,
            )
            plan_config.pose_cost_metric = hold_pose_cost_metric

        offset_start_state = reach_offset_mg_result.optimized_plan[-1].unsqueeze(0)

        self.toggle_link_collision(disable_collision_links, False)

        reach_grasp_mg_result = self.plan_single(
            offset_start_state,
            goal_pose,
            plan_config,
        )
        self.toggle_link_collision(disable_collision_links, True)
        result.grasp_result = reach_grasp_mg_result
        if not reach_grasp_mg_result.success.item():
            result.status = (
                f"Planning from Approach to Grasp Failed: {reach_grasp_mg_result.status}"
            )
            return result

        # Get stitched trajectory:

        offset_dt = reach_offset_mg_result.optimized_dt
        grasp_dt = reach_grasp_mg_result.optimized_dt
        if offset_dt > grasp_dt:
            # retime grasp trajectory to match offset trajectory:
            grasp_time_dilation = grasp_dt / offset_dt

            reach_grasp_mg_result.retime_trajectory(
                grasp_time_dilation,
                interpolate_trajectory=True,
            )
        else:
            offset_time_dilation = offset_dt / grasp_dt

            reach_offset_mg_result.retime_trajectory(
                offset_time_dilation,
                interpolate_trajectory=True,
            )

        if (reach_offset_mg_result.optimized_dt - reach_grasp_mg_result.optimized_dt).abs() > 0.01:
            reach_offset_mg_result.success[:] = False
            if reach_offset_mg_result.debug_info is None:
                reach_offset_mg_result.debug_info = {}
            reach_offset_mg_result.debug_info["plan_single_grasp_status"] = (
                "Stitching Trajectories Failed"
            )
            return reach_offset_mg_result, None

        result.grasp_trajectory = reach_offset_mg_result.optimized_plan.stack(
            reach_grasp_mg_result.optimized_plan
        ).clone()

        result.grasp_trajectory_dt = reach_offset_mg_result.optimized_dt

        result.grasp_interpolated_trajectory = (
            reach_offset_mg_result.get_interpolated_plan()
            .stack(reach_grasp_mg_result.get_interpolated_plan())
            .clone()
        )
        result.grasp_interpolation_dt = reach_offset_mg_result.interpolation_dt

        # update trajectories in results:
        result.planning_time = (
            reach_offset_mg_result.total_time
            + reach_grasp_mg_result.total_time
            + goalset_motion_gen_result.total_time
        )

        # check if retract path is required:
        result.success[:] = True
        if not plan_grasp_to_retract:
            return result

        result.success[:] = False
        self.toggle_link_collision(disable_collision_links, False)
        grasp_start_state = result.grasp_trajectory[-1].unsqueeze(0)

        # compute retract goal pose:
        if retract_constraint_in_goal_frame:
            retract_goal_pose = goal_pose.clone().multiply(retract_offset)
        else:
            retract_goal_pose = retract_offset.clone().multiply(goal_pose.clone())

        # add path constraint for retract:
        plan_config.pose_cost_metric = None

        if retract_path_constraint is not None:
            hold_pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device(retract_path_constraint),
                project_to_goal_frame=retract_constraint_in_goal_frame,
            )
            plan_config.pose_cost_metric = hold_pose_cost_metric

        # plan from grasp pose to retract:
        retract_grasp_mg_result = self.plan_single(
            grasp_start_state,
            retract_goal_pose,
            plan_config,
        )
        self.toggle_link_collision(disable_collision_links, True)
        result.planning_time += retract_grasp_mg_result.total_time
        if not retract_grasp_mg_result.success.item():
            result.status = f"Retract from Grasp failed: {retract_grasp_mg_result.status}"
            result.retract_result = retract_grasp_mg_result
            return result
        result.success[:] = True

        result.retract_trajectory = retract_grasp_mg_result.optimized_plan
        result.retract_trajectory_dt = retract_grasp_mg_result.optimized_dt
        result.retract_interpolated_trajectory = retract_grasp_mg_result.get_interpolated_plan()
        result.retract_interpolation_dt = retract_grasp_mg_result.interpolation_dt

        return result
