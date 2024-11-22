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
Trajectory optimization module contains :meth:`TrajOptSolver` class which can optimize for
minimum-jerk trajectories by first running a particle-based solver (MPPI) and then refining with
a gradient-based solver (L-BFGS). The module also provides linear interpolation functions for
generating seeds from start joint configuration to goal joint configurations. To generate
trajectories for reaching Cartesian poses or joint configurations, use the higher-level wrapper
:py:class:`~curobo.wrap.reacher.motion_gen.MotionGen`.

Trajectory Optimization uses joint positions as optimization variables with cost terms for
avoiding world collisions, self-collisions, and joint limits. The joint velocities, accelerations,
and jerks are computed using five point stencil. A squared l2-norm cost term on joint accelerations
and jerks is used to encourage smooth trajectories. A cost term for the terminal state to reach
either a Cartesian pose or joint configuration is also used. Read :ref:`research_page` for
more details.
"""

from __future__ import annotations

# Standard Library
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.opt.newton.lbfgs import LBFGSOpt, LBFGSOptConfig
from curobo.opt.newton.newton_base import NewtonOptBase, NewtonOptConfig
from curobo.opt.particle.parallel_es import ParallelES, ParallelESConfig
from curobo.opt.particle.parallel_mppi import ParallelMPPI, ParallelMPPIConfig
from curobo.rollout.arm_reacher import ArmReacher, ArmReacherConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.rollout.dynamics_model.integration_utils import interpolate_kernel
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState, RobotConfig
from curobo.types.tensor import T_BDOF, T_DOF, T_BValue_bool, T_BValue_float
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.torch_utils import get_torch_jit_decorator, is_cuda_graph_reset_available
from curobo.util.trajectory import (
    InterpolateType,
    calculate_dt_no_clamp,
    get_batch_interpolated_trajectory,
)
from curobo.util_file import get_robot_configs_path, get_task_configs_path, join_path, load_yaml
from curobo.wrap.reacher.evaluator import TrajEvaluator, TrajEvaluatorConfig
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.wrap.wrap_base import WrapBase, WrapConfig, WrapResult


@dataclass
class TrajOptSolverConfig:
    """Configuration parameters for TrajOptSolver."""

    robot_config: RobotConfig
    solver: WrapBase
    rollout_fn: ArmReacher
    position_threshold: float
    rotation_threshold: float
    traj_tsteps: int
    use_cspace_seed: bool = True
    interpolation_type: InterpolateType = InterpolateType.LINEAR_CUDA
    interpolation_steps: int = 1000
    world_coll_checker: Optional[WorldCollision] = None
    seed_ratio: Optional[Dict[str, int]] = None
    num_seeds: int = 1
    bias_node: Optional[T_DOF] = None
    interpolation_dt: float = 0.01
    traj_evaluator_config: Optional[TrajEvaluatorConfig] = None
    traj_evaluator: Optional[TrajEvaluator] = None
    evaluate_interpolated_trajectory: bool = True
    cspace_threshold: float = 0.1
    tensor_args: TensorDeviceType = TensorDeviceType()
    sync_cuda_time: bool = True
    interpolate_rollout: Optional[ArmReacher] = None
    use_cuda_graph_metrics: bool = False
    trim_steps: Optional[List[int]] = None
    store_debug_in_result: bool = False
    optimize_dt: bool = True
    use_cuda_graph: bool = True

    @staticmethod
    @profiler.record_function("trajopt_config/load_from_robot_config")
    def load_from_robot_config(
        robot_cfg: Union[str, Dict, RobotConfig],
        world_model: Optional[
            Union[Union[List[Dict], List[WorldConfig]], Union[Dict, WorldConfig]]
        ] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        cspace_threshold: float = 0.05,
        world_coll_checker=None,
        base_cfg_file: str = "base_cfg.yml",
        particle_file: str = "particle_trajopt.yml",
        gradient_file: str = "gradient_trajopt.yml",
        traj_tsteps: Optional[int] = None,
        interpolation_type: InterpolateType = InterpolateType.LINEAR_CUDA,
        interpolation_steps: int = 10000,
        interpolation_dt: float = 0.01,
        minimum_trajectory_dt: Optional[float] = None,
        use_cuda_graph: bool = True,
        self_collision_check: bool = False,
        self_collision_opt: bool = True,
        grad_trajopt_iters: Optional[int] = None,
        num_seeds: int = 2,
        seed_ratio: Dict[str, int] = {"linear": 1.0, "bias": 0.0, "start": 0.0, "end": 0.0},
        use_particle_opt: bool = True,
        collision_checker_type: Optional[CollisionCheckerType] = CollisionCheckerType.MESH,
        traj_evaluator_config: Optional[TrajEvaluatorConfig] = None,
        traj_evaluator: Optional[TrajEvaluator] = None,
        minimize_jerk: bool = True,
        use_gradient_descent: bool = False,
        collision_cache: Optional[Dict[str, int]] = None,
        n_collision_envs: Optional[int] = None,
        use_es: Optional[bool] = None,
        es_learning_rate: Optional[float] = 0.1,
        use_fixed_samples: Optional[bool] = None,
        aux_rollout: Optional[ArmReacher] = None,
        evaluate_interpolated_trajectory: bool = True,
        fixed_iters: Optional[bool] = None,
        store_debug: bool = False,
        sync_cuda_time: bool = True,
        collision_activation_distance: Optional[float] = None,
        trajopt_dt: Optional[float] = None,
        trim_steps: Optional[List[int]] = None,
        store_debug_in_result: bool = False,
        smooth_weight: Optional[List[float]] = None,
        state_finite_difference_mode: Optional[str] = None,
        filter_robot_command: bool = False,
        optimize_dt: bool = True,
        project_pose_to_goal_frame: bool = True,
        use_cuda_graph_metrics: bool = False,
        fix_terminal_action: bool = False,
    ):
        """Load TrajOptSolver configuration from robot configuration.

        Args:
            robot_cfg: Robot configuration to use for motion generation. This can be a path to a
                yaml file, a dictionary, or an instance of :class:`RobotConfig`. See
                :ref:`available_robot_list` for a list of available robots. You can also create a
                a configuration file for your robot using :ref:`tut_robot_configuration`.
            world_model: World configuration to use for motion generation. This can be a path to a
                yaml file, a dictionary, or an instance of :class:`WorldConfig`. See
                :ref:`world_collision` for more details.
            tensor_args: Numerical precision and compute device to use for motion generation.
            position_threshold: Position threshold between target position and reached position in
                meters. 0.005 is a good value for most tasks (5mm).
            rotation_threshold: Rotation threshold between target orientation and reached
                orientation. The metric is q^T * q, where q is the quaternion difference between
                target and reached orientation. The metric is not easy to interpret and a future
                release will provide a more intuitive metric. For now, use 0.05 as a good value.
            cspace_threshold: Joint space threshold in radians for revolute joints and meters for
                linear joints between reached joint configuration and target joint configuration
                used to measure success. Default of 0.05 has been found to be a good value for most
                cases.
            world_coll_checker: Instance of world collision checker to use for motion generation.
                Leaving this to None will create a new instance of world collision checker using
                the provided attr:`world_model`.
            base_cfg_file: Base configuration file containing convergence and constraint criteria
                to measure success.
            particle_file: Optimizer configuration file to use for particle-based
                optimization during trajectory optimization.
            gradient_file: Optimizer configuration file to use for gradient-based
                optimization during trajectory optimization.
            trajopt_tsteps: Number of waypoints to use for trajectory optimization. Default of 32
                is found to be a good number for most cases.
            interpolation_type: Interpolation type to use for generating dense waypoints from
                optimized trajectory. Default of
                :py:attr:`curobo.util.trajectory.InterpolateType.LINEAR_CUDA` is found to be a
                good choice for most cases. Other suitable options for real robot execution are
                :py:attr:`curobo.util.trajectory.InterpolateType.QUINTIC` and
                :py:attr:`curobo.util.trajectory.InterpolateType.CUBIC`.
            interpolation_steps: Buffer size to use for storing interpolated trajectory. Default of
                5000 is found to be a good number for most cases.
            interpolation_dt: Time step in seconds to use for generating interpolated trajectory
                from optimized trajectory. Change this if you want to generate a trajectory with
                a fixed timestep between waypoints.
            minimum_trajectory_dt: Minimum time step in seconds allowed for trajectory
                optimization.
            use_cuda_graph: Record compute ops as cuda graphs and replay recorded graphs where
                implemented. This can speed up execution by upto 10x. Default of True is
                recommended. Enabling this will prevent changing solve type or batch size
                after the first call to the solver.
            self_collision_check: Enable self collision checks for generated motions. Default of
                True is recommended. Set this to False to debug planning failures. Setting this to
                False will also set self_collision_opt to False.
            self_collision_opt: Enable self collision cost during optimization (IK, TrajOpt).
                Default of True is recommended.
            grad_trajopt_iters: Number of L-BFGS iterations to run trajectory optimization.
            num_seeds: Number of seeds to use for trajectory optimization per problem.
            seed_ratio: Ratio of linear and bias seeds to use for trajectory optimization.
                Linear seed will generate a linear interpolated trajectory from start state
                to IK solutions. Bias seed will add a mid-waypoint through the retract
                configuration. Default of 1.0 linear and 0.0 bias is recommended. This can be
                changed to 0.5 linear and 0.5 bias, along with changing num_seeds to 2.
            trajopt_particle_opt: Enable particle-based optimization during trajectory
                optimization. Default of True is recommended as particle-based optimization moves
                the interpolated seeds away from bad local minima.
            collision_checker_type: Type of collision checker to use for motion generation. Default
                of CollisionCheckerType.MESH supports world represented by Cuboids and Meshes. See
                :ref:`world_collision` for more details.
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
            use_gradient_descent: Use gradient descent instead of L-BFGS for trajectory
                optimization. Default of False is recommended. Set to True for debugging gradients
                of new cost terms.
            collision_cache: Cache of obstacles to create to load obstacles between planning calls.
                An example: ``{"obb": 10, "mesh": 10}``, to create a cache of 10 cuboids and 10
                meshes.
            n_collision_envs: Number of collision environments to create for batched optimization
                across different environments. Only used for
                :py:meth:`TrajOptSolver.solve_batch_env`and
                :py:meth:`TrajOptSolver.solve_batch_env_goalset`.
            n_collision_envs: Number of collision environments to create for batched planning
                across different environments. Only used for :py:meth:`MotionGen.plan_batch_env`
                and :py:meth:`MotionGen.plan_batch_env_goalset`.
            use_es: Use Evolution Strategies for optimization. Default of None will use MPPI.
            es_learning_rate: Learning rate to use for Evolution Strategies.
            use_fixed_samples: Use fixed samples for MPPI. Setting to False will increase compute
                time as new samples are generated for each iteration of MPPI.
            aux_rollout: Rollout instance to use for auxiliary rollouts.
            evaluate_interpolated_trajectory: Evaluate interpolated trajectory after optimization.
                Default of True is recommended to ensure the optimized trajectory is not passing
                through very thin obstacles.
            fixed_iters: Use fixed number of iterations of L-BFGS for trajectory
                optimization. Default of None will use the setting from the optimizer
                configuration. In most cases, fixed iterations of solvers are run as current
                solvers treat constraints as costs and there is no guarantee that the constraints
                will be satisfied. Instead of checking constraints between iterations of a solver
                and exiting, it's computationally cheaper to run a fixed number of iterations. In
                addition, running fixed iterations of solvers is more robust to outlier problems.
            store_debug: Store debugging information such as values of optimization
                variables in TrajOpt result. Setting this to True will set :attr:`use_cuda_graph`
                to False.
            sync_cuda_time: Synchronize with host using :py:func:`torch.cuda.synchronize` before
                measuring compute time.
            collision_activation_distance: Distance in meters to activate collision cost. A good
                value to start with is 0.01 meters. Increase the distance if the robot needs to
                stay further away from obstacles.
            trajopt_dt: Time step in seconds to use for trajectory optimization. A good value to
                start with is 0.15 seconds. This value is used to compute velocity, acceleration,
                and jerk values for waypoints through finite difference.
            trim_steps: Trim waypoints from optimized trajectory. The optimized trajectory will
                contain the start state at index 0 and have the last two waypoints be the same
                as T-2 as trajectory optimization implicitly optimizes for zero acceleration and
                velocity at the last waypoint. An example: ``[1,-2]`` will trim the first waypoint
                and last 3 waypoints from the optimized trajectory.
            store_debug_in_result: Store debugging information in MotionGenResult. This value is
                set to True if either store_ik_debug or store_trajopt_debug is set to True.
            smooth_weight: Override smooth weight for trajectory optimization. It's not recommended
                to set this value for most cases.
            state_finite_difference_mode: Finite difference mode to use for computing velocity,
                acceleration, and jerk values. Default of None will use the setting from the
                optimizer configuration file. The default finite difference method is a five
                point stencil to compute the derivatives as this is accurate and provides
                faster convergence compared to backward or central difference methods.
            filter_robot_command: Filter generated trajectory to remove finite difference
                artifacts. Default of True is recommended.
            optimize_dt: Optimize dt during trajectory optimization. Default of True is
                recommended to find time-optimal trajectories. Setting this to False will use the
                provided :attr:`trajopt_dt` for trajectory optimization. Setting to False is
                required when optimizing from a non-static start state.
            project_pose_to_goal_frame: Project pose to goal frame when calculating distance
                between reached and goal pose. Use this to constrain motion to specific axes
                either in the global frame or the goal frame.
            use_cuda_graph_metrics: Flag to enable cuda_graph when evaluating interpolated
                trajectories after trajectory optimization. If interpolation_buffer is smaller
                than interpolated trajectory, then the buffers will be re-created. This can cause
                existing cuda graph to be invalid.

        Returns:
            TrajOptSolverConfig: Trajectory optimization configuration.
        """

        if minimum_trajectory_dt is None:
            minimum_trajectory_dt = interpolation_dt
        elif minimum_trajectory_dt < interpolation_dt:
            log_error("minimum_trajectory_dt cannot be lower than interpolation_dt")
        if isinstance(robot_cfg, str):
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))["robot_cfg"]

        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

        base_config_data = load_yaml(join_path(get_task_configs_path(), base_cfg_file))
        if collision_cache is not None:
            base_config_data["world_collision_checker_cfg"]["cache"] = collision_cache
        if n_collision_envs is not None:
            base_config_data["world_collision_checker_cfg"]["n_envs"] = n_collision_envs
        if not self_collision_check:
            base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0
            self_collision_opt = False

        if collision_checker_type is not None:
            base_config_data["world_collision_checker_cfg"]["checker_type"] = collision_checker_type

        if world_coll_checker is None and world_model is not None:
            world_cfg = WorldCollisionConfig.load_from_dict(
                base_config_data["world_collision_checker_cfg"], world_model, tensor_args
            )
            world_coll_checker = create_collision_checker(world_cfg)

        config_data = load_yaml(join_path(get_task_configs_path(), particle_file))
        grad_config_data = load_yaml(join_path(get_task_configs_path(), gradient_file))

        if traj_tsteps is None:
            traj_tsteps = grad_config_data["model"]["horizon"]

        base_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        base_config_data["convergence"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        grad_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        grad_config_data["lbfgs"]["fix_terminal_action"] = fix_terminal_action

        config_data["model"]["horizon"] = traj_tsteps
        grad_config_data["model"]["horizon"] = traj_tsteps
        if minimize_jerk is not None:
            if not minimize_jerk:
                grad_config_data["cost"]["bound_cfg"]["smooth_weight"][2] = 0.0
                grad_config_data["cost"]["bound_cfg"]["smooth_weight"][1] *= 2.0
                grad_config_data["lbfgs"]["cost_delta_threshold"] = 0.1
            if minimize_jerk and grad_config_data["cost"]["bound_cfg"]["smooth_weight"][2] == 0.0:
                log_warn("minimize_jerk flag is enabled but weight term is zero")

        if state_finite_difference_mode is not None:
            config_data["model"]["state_finite_difference_mode"] = state_finite_difference_mode
            grad_config_data["model"]["state_finite_difference_mode"] = state_finite_difference_mode
        config_data["model"]["filter_robot_command"] = filter_robot_command
        grad_config_data["model"]["filter_robot_command"] = filter_robot_command

        if collision_activation_distance is not None:
            config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance
            grad_config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance

        if grad_trajopt_iters is not None:
            grad_config_data["lbfgs"]["n_iters"] = grad_trajopt_iters
            grad_config_data["lbfgs"]["cold_start_n_iters"] = grad_trajopt_iters
        if use_fixed_samples is not None:
            config_data["mppi"]["sample_params"]["fixed_samples"] = use_fixed_samples
        if smooth_weight is not None:
            grad_config_data["cost"]["bound_cfg"]["smooth_weight"][
                : len(smooth_weight)
            ] = smooth_weight  # velocity

        if store_debug:
            use_cuda_graph = False
            fixed_iters = True
            grad_config_data["lbfgs"]["store_debug"] = store_debug
            config_data["mppi"]["store_debug"] = store_debug
            store_debug_in_result = True

        if use_cuda_graph is not None:
            config_data["mppi"]["use_cuda_graph"] = use_cuda_graph
            grad_config_data["lbfgs"]["use_cuda_graph"] = use_cuda_graph
        else:
            use_cuda_graph = grad_config_data["lbfgs"]["use_cuda_graph"]
        if not self_collision_opt:
            config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
            grad_config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
        config_data["mppi"]["n_problems"] = 1
        grad_config_data["lbfgs"]["n_problems"] = 1

        if fixed_iters is not None:
            grad_config_data["lbfgs"]["fixed_iters"] = fixed_iters

        grad_cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            grad_config_data["model"],
            grad_config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        # safety_robot_model = robot_cfg.kinematics
        # safety_robot_cfg = RobotConfig(**vars(robot_cfg))
        # safety_robot_cfg.kinematics = safety_robot_model
        safety_robot_cfg = robot_cfg
        safety_cfg = ArmReacherConfig.from_dict(
            safety_robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        aux_cfg = ArmReacherConfig.from_dict(
            safety_robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        interpolate_cfg = ArmReacherConfig.from_dict(
            safety_robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        arm_rollout_mppi = None
        with profiler.record_function("trajopt_config/create_rollouts"):
            if use_particle_opt:
                arm_rollout_mppi = ArmReacher(cfg)
            arm_rollout_grad = ArmReacher(grad_cfg)

            arm_rollout_safety = ArmReacher(safety_cfg)
            if aux_rollout is None:
                aux_rollout = ArmReacher(aux_cfg)
            interpolate_rollout = ArmReacher(interpolate_cfg)
        if trajopt_dt is not None:
            if arm_rollout_mppi is not None:
                arm_rollout_mppi.update_traj_dt(dt=trajopt_dt)
            aux_rollout.update_traj_dt(dt=trajopt_dt)
            arm_rollout_grad.update_traj_dt(dt=trajopt_dt)
            arm_rollout_safety.update_traj_dt(dt=trajopt_dt)
        if arm_rollout_mppi is not None:
            config_dict = ParallelMPPIConfig.create_data_dict(
                config_data["mppi"], arm_rollout_mppi, tensor_args
            )
        parallel_mppi = None
        if use_es is not None and use_es:
            mppi_cfg = ParallelESConfig(**config_dict)
            if es_learning_rate is not None:
                mppi_cfg.learning_rate = es_learning_rate
            parallel_mppi = ParallelES(mppi_cfg)
        elif use_particle_opt:
            mppi_cfg = ParallelMPPIConfig(**config_dict)
            parallel_mppi = ParallelMPPI(mppi_cfg)
        config_dict = LBFGSOptConfig.create_data_dict(
            grad_config_data["lbfgs"], arm_rollout_grad, tensor_args
        )
        lbfgs_cfg = LBFGSOptConfig(**config_dict)

        if use_gradient_descent:
            newton_keys = NewtonOptConfig.__dataclass_fields__.keys()
            newton_d = {}
            lbfgs_k = vars(lbfgs_cfg)
            for k in newton_keys:
                newton_d[k] = lbfgs_k[k]
            newton_d["step_scale"] = 0.9
            newton_cfg = NewtonOptConfig(**newton_d)
            lbfgs = NewtonOptBase(newton_cfg)
        else:
            lbfgs = LBFGSOpt(lbfgs_cfg)
        if use_particle_opt:
            opt_list = [parallel_mppi]
        else:
            opt_list = []
        opt_list.append(lbfgs)
        cfg = WrapConfig(
            safety_rollout=arm_rollout_safety,
            optimizers=opt_list,
            compute_metrics=True,
            use_cuda_graph_metrics=use_cuda_graph_metrics,
            sync_cuda_time=sync_cuda_time,
        )
        trajopt = WrapBase(cfg)
        if traj_evaluator_config is None:
            traj_evaluator_config = TrajEvaluatorConfig.from_basic(
                min_dt=minimum_trajectory_dt,
                dof=robot_cfg.kinematics.dof,
                tensor_args=tensor_args,
            )
        trajopt_cfg = TrajOptSolverConfig(
            robot_config=robot_cfg,
            rollout_fn=aux_rollout,
            solver=trajopt,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            cspace_threshold=cspace_threshold,
            traj_tsteps=traj_tsteps,
            interpolation_steps=interpolation_steps,
            interpolation_dt=interpolation_dt,
            interpolation_type=interpolation_type,
            world_coll_checker=world_coll_checker,
            bias_node=robot_cfg.kinematics.cspace.retract_config,
            seed_ratio=seed_ratio,
            num_seeds=num_seeds,
            traj_evaluator_config=traj_evaluator_config,
            traj_evaluator=traj_evaluator,
            evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
            tensor_args=tensor_args,
            sync_cuda_time=sync_cuda_time,
            interpolate_rollout=interpolate_rollout,
            use_cuda_graph_metrics=use_cuda_graph_metrics,
            trim_steps=trim_steps,
            store_debug_in_result=store_debug_in_result,
            optimize_dt=optimize_dt,
            use_cuda_graph=use_cuda_graph,
        )
        return trajopt_cfg


@dataclass
class TrajOptResult(Sequence):
    """Trajectory optimization result."""

    success: T_BValue_bool
    goal: Goal
    solution: JointState
    seed: T_BDOF
    solve_time: float
    debug_info: Optional[Any] = None
    metrics: Optional[RolloutMetrics] = None
    interpolated_solution: Optional[JointState] = None
    path_buffer_last_tstep: Optional[List[int]] = None
    position_error: Optional[T_BValue_float] = None
    rotation_error: Optional[T_BValue_float] = None
    cspace_error: Optional[T_BValue_float] = None
    smooth_error: Optional[T_BValue_float] = None
    smooth_label: Optional[T_BValue_bool] = None
    optimized_dt: Optional[torch.Tensor] = None
    raw_solution: Optional[JointState] = None
    raw_action: Optional[torch.Tensor] = None
    goalset_index: Optional[torch.Tensor] = None
    optimized_seeds: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int) -> TrajOptResult:
        """Get item at index.

        Args:
            idx: Index of the item to get.

        Returns:
            TrajOptResult: Trajectory optimization result at the given index.
        """

        d_list = [
            self.interpolated_solution,
            self.metrics,
            self.path_buffer_last_tstep,
            self.position_error,
            self.rotation_error,
            self.cspace_error,
            self.goalset_index,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)

        return TrajOptResult(
            goal=self.goal[idx],
            solution=self.solution[idx],
            success=self.success[idx],
            seed=self.seed[idx],
            debug_info=self.debug_info,
            solve_time=self.solve_time,
            interpolated_solution=idx_vals[0],
            metrics=idx_vals[1],
            path_buffer_last_tstep=idx_vals[2],
            position_error=idx_vals[3],
            rotation_error=idx_vals[4],
            cspace_error=idx_vals[5],
            goalset_index=idx_vals[6],
            optimized_seeds=self.optimized_seeds,
        )

    def __len__(self) -> int:
        """Get length of the TrajOptResult."""
        return self.success.shape[0]


@dataclass
class TrajResult(TrajOptResult):
    """Deprecated: Use TrajOptResult instead of TrajResult"""

    def __post_init__(self):
        """post-init function for TrajResult"""
        log_warn("Deprecated: Use TrajOptResult instead of TrajResult")


class TrajOptSolver(TrajOptSolverConfig):
    """Trajectory Optimization Solver class for generating minimum-jerk trajectories.

    Trajectory Optimization uses joint positions as optimization variables with cost terms for
    avoiding world collisions, self-collisions, and joint limits. The joint velocities, accelerations,
    and jerks are computed using five point stencil. A squared l2-norm cost term on joint accelerations
    and jerks is used to encourage smooth trajectories. A cost term for the terminal state to reach
    either a Cartesian pose or joint configuration is also used. Read :ref:`research_page` for
    more details.
    """

    def __init__(self, config: TrajOptSolverConfig) -> None:
        """Initialize TrajOptSolver with configuration parameters.

        Args:
            config: Configuration parameters for TrajOptSolver.
        """
        super().__init__(**vars(config))
        self.delta_vec = interpolate_kernel(2, self.action_horizon, self.tensor_args).unsqueeze(0)

        self.waypoint_delta_vec = interpolate_kernel(
            3, int(self.action_horizon / 2), self.tensor_args
        ).unsqueeze(0)
        assert self.action_horizon / 2 != 0.0
        self.solver.update_nproblems(self.num_seeds)
        self._max_joint_vel = self.solver.safety_rollout.state_bounds.velocity.view(2, self.dof)[
            1, :
        ].reshape(1, 1, self.dof)
        self._max_joint_acc = self.rollout_fn.state_bounds.acceleration[1, :]
        self._max_joint_jerk = self.rollout_fn.state_bounds.jerk[1, :]
        self._num_seeds = -1
        self._col = None
        if self.traj_evaluator is None:
            self.traj_evaluator = TrajEvaluator(self.traj_evaluator_config)
        self._interpolation_dt_tensor = self.tensor_args.to_device([self.interpolation_dt])
        self._n_seeds = self._get_seed_numbers(self.num_seeds)
        self._goal_buffer = None
        self._solve_state = None
        self._velocity_bounds = self.solver.rollout_fn.state_bounds.velocity[1]
        self._og_newton_iters = self.solver.optimizers[-1].outer_iters
        self._og_newton_fixed_iters = self.solver.optimizers[-1].fixed_iters
        self._interpolated_traj_buffer = None
        self._kin_list = None
        self._rollout_list = None

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        """Get all rollout instances in the solver.

        Useful to update parameters across all rollout instances.

        Returns:
            List[RolloutBase]: List of all rollout instances.
        """
        if self._rollout_list is None:
            self._rollout_list = [
                self.rollout_fn,
                self.interpolate_rollout,
            ] + self.solver.get_all_rollout_instances()
        return self._rollout_list

    def get_all_kinematics_instances(self) -> List[CudaRobotModel]:
        """Get all kinematics instances used across components in motion generation.

        This is deprecated. Use :meth:`TrajOptSolver.kinematics` instead as TrajOptSolver now uses
        a shared kinematics instance across all components.

        Returns:
            List[CudaRobotModel]: Single kinematics instance, returned as a list for compatibility.
        """
        log_warn(
            "Deprecated: Use TrajOptSolver.kinematics instead as TrajOptSolver now uses a "
            + "shared kinematics instance across all components."
        )
        if self._kin_list is None:
            self._kin_list = [
                i.dynamics_model.robot_model for i in self.get_all_rollout_instances()
            ]
        return self._kin_list

    def attach_spheres_to_robot(
        self,
        sphere_radius: float,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> None:
        """Attach spheres to robot for collision checking.

        To fit spheres to an obstacle, see
        :py:meth:`~curobo.geom.types.Obstacle.get_bounding_spheres`

        Args:
            sphere_radius: Radius of the spheres. Set to None if :attr:`sphere_tensor` is provided.
            sphere_tensor: Sphere x, y, z, r tensor.
            link_name: Name of the link to attach the spheres to. Note that this link should
                already have pre-allocated spheres.
        """
        self.kinematics.kinematics_config.attach_object(
            sphere_radius=sphere_radius, sphere_tensor=sphere_tensor, link_name=link_name
        )

    def detach_spheres_from_robot(self, link_name: str = "attached_object") -> None:
        """Detach spheres from robot.

        Args:
            link_name: Name of the link to detach the spheres from.
        """
        self.kinematics.kinematics_config.detach_object(link_name)

    def _update_solve_state_and_goal_buffer(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
    ):
        """Update goal buffer and solve state of current trajectory optimization problem.

        Args:
            solve_state: New solve state.
            goal: New goal buffer.

        Returns:
            Goal: Updated goal buffer with augmentations for new solve state.
        """
        self._solve_state, self._goal_buffer, update_reference = solve_state.update_goal(
            goal,
            self._solve_state,
            self._goal_buffer,
            self.tensor_args,
        )

        if update_reference:
            if self.use_cuda_graph and self._col is not None:
                if is_cuda_graph_reset_available():
                    log_warn("changing goal type, breaking previous cuda graph")
                    self.reset_cuda_graph()
                else:
                    log_error("changing goal type not supported in cuda graph mode")

            self.solver.update_nproblems(self._solve_state.get_batch_size())
            self._col = torch.arange(
                0, goal.batch, device=self.tensor_args.device, dtype=torch.long
            )
            self.reset_shape()

        return self._goal_buffer

    def solve_any(
        self,
        solve_type: ReacherSolveType,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization problem with any solve type.

        Args:
            solve_type: Type of solve to perform.
            goal: Goal to reach.
            seed_traj: Seed trajectory to start optimization from. This should be
                of shape [num_seeds, batch_size, action_horizon, dof]. If None, linearly
                interpolated seeds from current state to goal state are used. If goal.goal_state
                is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. This cannot be changed after the
                first call to solve as CUDA graph re-creation is currently not supported.
            seed_success: Success of seeds. This is used to filter out successful seeds from
                :attr:`seed_traj`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number
                of iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if solve_type == ReacherSolveType.SINGLE:
            return self.solve_single(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.GOALSET:
            return self.solve_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH:
            return self.solve_batch(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_GOALSET:
            return self.solve_batch_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV:
            return self.solve_batch_env(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV_GOALSET:
            return self.solve_batch_env_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )

    def _solve_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ):
        """Solve trajectory optimization problem with a given solve state.

        Args:
            solve_state: Solve state for the optimization problem.
            goal: Goal object containing target pose or joint configuration.
            seed_traj: Seed trajectory to start optimization from. This should be of
                shape [num_seeds, batch_size, action_horizon, dof]. If None, linearly
                interpolated seeds from current state to goal state are used. If goal.goal_state
                is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. This cannot be changed after the
                first call to solve as CUDA graph re-creation is currently not supported.
            seed_success: Success of seeds. This is used to filter out successful seeds from
                :attr:`seed_traj`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`. This
                is the outer iterations, where each outer iteration will run 25 inner iterations
                of LBFGS optimization captured in a CUDA-Graph. Total number of optimization
                iterations is 25 * outer_iters. The number of inner iterations can be changed
                with :py:attr:`curobo.opt.newton.lbfgs.LBFGSOptConfig.inner_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                log_error("Batch Env is less that goal batch")
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = newton_iters
            self.solver.newton_optimizer.fixed_iters = True
        log_info("TrajOpt: solving with Pose batch:" + str(goal.batch))
        goal_buffer = self._update_solve_state_and_goal_buffer(solve_state, goal)
        # if self.evaluate_interpolated_trajectory:
        self.interpolate_rollout.update_params(goal_buffer)
        # get seeds:
        seed_traj = self.get_seed_set(
            goal_buffer, seed_traj, seed_success, num_seeds, solve_state.batch_mode
        )

        # remove goal state if goal pose is not None:
        if goal_buffer.goal_pose.position is not None:
            goal_buffer.goal_state = None
        self.solver.reset()
        result = self.solver.solve(goal_buffer, seed_traj)
        log_info("Ran TO")
        traj_result = self._get_result(
            result,
            return_all_solutions,
            goal_buffer,
            seed_traj,
            num_seeds,
            solve_state.batch_mode,
        )
        if traj_result.goalset_index is not None:
            traj_result.goalset_index[traj_result.goalset_index >= goal.goal_pose.n_goalset] = 0
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = self._og_newton_iters
            self.solver.newton_optimizer.fixed_iters = self._og_newton_fixed_iters
        return traj_result

    def solve_single(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization problem for a single goal.

        This will use multiple seeds internally and return the best solution.

        Args:
            goal: Goal to reach.
            seed_traj: Seed trajectory to start optimization from. This should be of shape
                [:attr:`num_seeds`, 1, :attr:`TrajOptSolver.action_horizon`,
                :attr:`TrajOptSolver.dof`]. If None, linearly interpolated seeds from current state
                to goal state are used. If goal.goal_state is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. If None, the number of seeds
                is set to :attr:`TrajOptSolver.num_seeds`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`. This
                is the outer iterations, where each outer iteration will run 25 inner iterations
                of LBFGS optimization captured in a CUDA-Graph. Total number of optimization
                iterations is 25 * outer_iters. The number of inner iterations can be changed
                with :py:attr:`curobo.opt.newton.lbfgs.LBFGSOptConfig.inner_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE,
            num_trajopt_seeds=num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=1,
        )

        return self._solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve_goalset(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization problem that uses goalset to represent Pose target.

        Args:
            goal: Goal to reach.
            seed_traj: Seed trajectory to start optimization from. This should be of shape
                [:attr:`num_seeds`, 1, :attr:`TrajOptSolver.action_horizon`,
                :attr:`TrajOptSolver.dof`]. If None, linearly interpolated seeds from current state
                to goal state are used. If goal.goal_state is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. If None, the number of seeds
                is set to :attr:`TrajOptSolver.num_seeds`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`. This
                is the outer iterations, where each outer iteration will run 25 inner iterations
                of LBFGS optimization captured in a CUDA-Graph. Total number of optimization
                iterations is 25 * outer_iters. The number of inner iterations can be changed
                with :py:attr:`curobo.opt.newton.lbfgs.LBFGSOptConfig.inner_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.GOALSET,
            num_trajopt_seeds=num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        return self._solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve_batch(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization problem for a batch of goals.

        Args:
            goal: Batch of goals to reach, this includes batch of current states.
            seed_traj: Seed trajectory to start optimization from. This should be of shape
                [:attr:`num_seeds`, :attr:`goal.batch`, :attr:`TrajOptSolver.action_horizon`]. If
                None, linearly interpolated seeds from current state to goal state are used. If
                goal.goal_state is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. If None, the number of seeds
                is set to :attr:`TrajOptSolver.num_seeds`.
            seed_success: Success of seeds. This is used to filter out successful seeds from
                :attr:`seed_traj`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`. This
                is the outer iterations, where each outer iteration will run 25 inner iterations
                of LBFGS optimization captured in a CUDA-Graph. Total number of optimization
                iterations is 25 * outer_iters. The number of inner iterations can be changed
                with :py:attr:`curobo.opt.newton.lbfgs.LBFGSOptConfig.inner_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=1,
        )
        return self._solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            seed_success,
            newton_iters=newton_iters,
        )

    def solve_batch_goalset(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization problem for a batch of Poses with goalset.

        Args:
            goal: Batch of goals to reach, this includes batch of current states.
            seed_traj: Seed trajectory to start optimization from. This should be of shape
                [:attr:`num_seeds`, :attr:`goal.batch`, :attr:`TrajOptSolver.action_horizon`]. If
                None, linearly interpolated seeds from current state to goal state are used. If
                goal.goal_state is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. If None, the number of seeds
                is set to :attr:`TrajOptSolver.num_seeds`.
            seed_success: Success of seeds. This is used to filter out successful seeds from
                :attr:`seed_traj`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`. This
                is the outer iterations, where each outer iteration will run 25 inner iterations
                of LBFGS optimization captured in a CUDA-Graph. Total number of optimization
                iterations is 25 * outer_iters. The number of inner iterations can be changed
                with :py:attr:`curobo.opt.newton.lbfgs.LBFGSOptConfig.inner_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_GOALSET,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        return self._solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve_batch_env(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization problem in a batch of environments.

        Args:
            goal: Batch of goals to reach, this includes batch of current states.
            seed_traj: Seed trajectory to start optimization from. This should be of shape
                [:attr:`num_seeds`, :attr:`goal.batch`, :attr:`TrajOptSolver.action_horizon`]. If
                None, linearly interpolated seeds from current state to goal state are used. If
                goal.goal_state is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. If None, the number of seeds
                is set to :attr:`TrajOptSolver.num_seeds`.
            seed_success: Success of seeds. This is used to filter out successful seeds from
                :attr:`seed_traj`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`. This
                is the outer iterations, where each outer iteration will run 25 inner iterations
                of LBFGS optimization captured in a CUDA-Graph. Total number of optimization
                iterations is 25 * outer_iters. The number of inner iterations can be changed
                with :py:attr:`curobo.opt.newton.lbfgs.LBFGSOptConfig.inner_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=goal.batch,
            n_goalset=1,
        )
        return self._solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            seed_success,
            newton_iters=newton_iters,
        )

    def solve_batch_env_goalset(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Solve trajectory optimization problem in a batch of environments with goalset.

        Args:
            goal: Batch of goals to reach, this includes batch of current states.
            seed_traj: Seed trajectory to start optimization from. This should be of shape
                [:attr:`num_seeds`, :attr:`goal.batch`, :attr:`TrajOptSolver.action_horizon`]. If
                None, linearly interpolated seeds from current state to goal state are used. If
                goal.goal_state is empty, random seeds are generated.
            use_nn_seed: Use neural network seed for optimization. This is not implemented.
            return_all_solutions: Return solutions for all seeds.
            num_seeds: Number of seeds to use for optimization. If None, the number of seeds
                is set to :attr:`TrajOptSolver.num_seeds`.
            seed_success: Success of seeds. This is used to filter out successful seeds from
                :attr:`seed_traj`.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in :attr:`TrajOptSolver.newton_iters`. This
                is the outer iterations, where each outer iteration will run 25 inner iterations
                of LBFGS optimization captured in a CUDA-Graph. Total number of optimization
                iterations is 25 * outer_iters. The number of inner iterations can be changed
                with :py:attr:`curobo.opt.newton.lbfgs.LBFGSOptConfig.inner_iters`.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV_GOALSET,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=goal.batch,
            n_goalset=goal.n_goalset,
        )
        return self._solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajOptResult:
        """Deprecated: Use :meth:`TrajOptSolver.solve_single` or others instead."""
        log_warn(
            "TrajOptSolver.solve is deprecated, use TrajOptSolver.solve_single or others instead"
        )
        if goal.goal_pose.batch == 1 and goal.goal_pose.n_goalset == 1:
            return self.solve_single(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        if goal.goal_pose.batch == 1 and goal.goal_pose.n_goalset > 1:
            return self.solve_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        if goal.goal_pose.batch > 1 and goal.goal_pose.n_goalset == 1:
            return self.solve_batch(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )

        raise NotImplementedError()

    @profiler.record_function("trajopt/get_result")
    def _get_result(
        self,
        result: WrapResult,
        return_all_solutions: bool,
        goal: Goal,
        seed_traj: JointState,
        num_seeds: int,
        batch_mode: bool = False,
    ) -> TrajOptResult:
        """Get result from the optimization problem.

        Args:
            result: Result of the optimization problem.
            return_all_solutions: Return solutions for all seeds.
            goal: Goal object containing convergence parameters.
            seed_traj: Seed trajectory used for optimization.
            num_seeds: Number of seeds used for optimization.
            batch_mode: Batch mode for problems.

        Returns:
            TrajOptResult: Result of the trajectory optimization.
        """
        st_time = time.time()
        if self.trim_steps is not None:
            result.action = result.action.trim_trajectory(self.trim_steps[0], self.trim_steps[1])
        interpolated_trajs, last_tstep, opt_dt, buffer_change = self.get_interpolated_trajectory(
            result.action
        )

        if self.sync_cuda_time:
            torch.cuda.synchronize(device=self.tensor_args.device)
        interpolation_time = time.time() - st_time
        if self.evaluate_interpolated_trajectory:
            with profiler.record_function("trajopt/evaluate_interpolated"):
                if self.use_cuda_graph_metrics and buffer_change:
                    if is_cuda_graph_reset_available():
                        self.interpolate_rollout.reset_cuda_graph()
                        buffer_change = False
                    else:
                        self.interpolate_rollout.break_cuda_graph()
                if self.use_cuda_graph_metrics and not buffer_change:
                    metrics = self.interpolate_rollout.get_metrics_cuda_graph(interpolated_trajs)
                else:
                    metrics = self.interpolate_rollout.get_metrics(interpolated_trajs)
                result.metrics.feasible = metrics.feasible
                result.metrics.position_error = metrics.position_error
                result.metrics.rotation_error = metrics.rotation_error
                result.metrics.cspace_error = metrics.cspace_error
                result.metrics.goalset_index = metrics.goalset_index

        st_time = time.time()
        if result.metrics.cspace_error is None and result.metrics.position_error is None:
            log_error("convergence check requires either goal_pose or goal_state")

        success = jit_feasible_success(
            result.metrics.feasible,
            result.metrics.position_error,
            result.metrics.rotation_error,
            result.metrics.cspace_error,
            self.position_threshold,
            self.rotation_threshold,
            self.cspace_threshold,
        )

        if False:
            feasible = torch.all(result.metrics.feasible, dim=-1)

            if result.metrics.position_error is not None:
                converge = torch.logical_and(
                    result.metrics.position_error[..., -1] <= self.position_threshold,
                    result.metrics.rotation_error[..., -1] <= self.rotation_threshold,
                )
            elif result.metrics.cspace_error is not None:
                converge = result.metrics.cspace_error[..., -1] <= self.cspace_threshold
            else:
                log_error("convergence check requires either goal_pose or goal_state")

            success = torch.logical_and(feasible, converge)
        if return_all_solutions:
            traj_result = TrajOptResult(
                success=success,
                goal=goal,
                solution=result.action.scale_by_dt(self.solver_dt_tensor, opt_dt.view(-1, 1, 1)),
                seed=seed_traj,
                position_error=(
                    result.metrics.position_error[..., -1]
                    if result.metrics.position_error is not None
                    else None
                ),
                rotation_error=(
                    result.metrics.rotation_error[..., -1]
                    if result.metrics.rotation_error is not None
                    else None
                ),
                solve_time=result.solve_time,
                metrics=result.metrics,  # TODO: index this also
                interpolated_solution=interpolated_trajs,
                debug_info={"solver": result.debug, "interpolation_time": interpolation_time},
                path_buffer_last_tstep=last_tstep,
                cspace_error=(
                    result.metrics.cspace_error[..., -1]
                    if result.metrics.cspace_error is not None
                    else None
                ),
                optimized_dt=opt_dt,
                raw_solution=result.action,
                raw_action=result.raw_action,
                goalset_index=result.metrics.goalset_index,
                optimized_seeds=result.raw_action,
            )
        else:
            # get path length:
            if self.evaluate_interpolated_trajectory:
                smooth_label, smooth_cost = self.traj_evaluator.evaluate_interpolated_smootheness(
                    interpolated_trajs,
                    opt_dt,
                    self.solver.rollout_fn.dynamics_model.cspace_distance_weight,
                    self._velocity_bounds,
                )

            else:
                smooth_label, smooth_cost = self.traj_evaluator.evaluate(
                    result.action,
                    self.solver.rollout_fn.traj_dt,
                    self.solver.rollout_fn.dynamics_model.cspace_distance_weight,
                    self._velocity_bounds,
                )

            with profiler.record_function("trajopt/best_select"):
                (
                    idx,
                    position_error,
                    rotation_error,
                    cspace_error,
                    goalset_index,
                    opt_dt,
                    success,
                ) = jit_trajopt_best_select(
                    success,
                    smooth_label,
                    result.metrics.cspace_error,
                    result.metrics.pose_error,
                    result.metrics.position_error,
                    result.metrics.rotation_error,
                    result.metrics.goalset_index,
                    result.metrics.cost,
                    smooth_cost,
                    batch_mode,
                    goal.batch,
                    num_seeds,
                    self._col,
                    opt_dt,
                )
                if batch_mode:
                    last_tstep = [last_tstep[i] for i in idx]
                else:
                    last_tstep = [last_tstep[idx.item()]]
                best_act_seq = result.action[idx]
                best_raw_action = result.raw_action[idx]
                interpolated_traj = interpolated_trajs[idx]

            if self.sync_cuda_time:
                torch.cuda.synchronize(device=self.tensor_args.device)
            if len(best_act_seq.shape) == 3:
                opt_dt_v = opt_dt.view(-1, 1, 1)
            else:
                opt_dt_v = opt_dt.view(1, 1)
            opt_solution = best_act_seq.scale_by_dt(self.solver_dt_tensor, opt_dt_v)
            select_time = time.time() - st_time
            debug_info = {}
            if self.store_debug_in_result:
                debug_info = {
                    "solver": result.debug,
                    "interpolation_time": interpolation_time,
                    "select_time": select_time,
                }
            if not torch.count_nonzero(success) > 0:
                max_dt = torch.max(opt_dt).item()
                if max_dt >= self.traj_evaluator_config.max_dt:

                    log_info(
                        "Optimized dt is above maximum_trajectory_dt, consider \
                            increasing max_trajectory_dt"
                    )
                    debug_info["dt_exception"] = True
            traj_result = TrajOptResult(
                success=success,
                goal=goal,
                solution=opt_solution,
                seed=seed_traj,
                position_error=position_error,
                rotation_error=rotation_error,
                cspace_error=cspace_error,
                solve_time=result.solve_time,
                metrics=result.metrics,  # TODO: index this also
                interpolated_solution=interpolated_traj,
                debug_info=debug_info,
                path_buffer_last_tstep=last_tstep,
                smooth_error=smooth_cost,
                smooth_label=smooth_label,
                optimized_dt=opt_dt,
                raw_solution=best_act_seq,
                raw_action=best_raw_action,
                goalset_index=goalset_index,
                optimized_seeds=result.raw_action,
            )
        return traj_result

    def batch_solve(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        seed_success: Optional[torch.Tensor] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
    ) -> TrajOptResult:
        """Deprecated: Use :meth:`TrajOptSolver.solve_batch` or others instead."""
        log_warn(
            "TrajOptSolver.batch_solve is deprecated, use TrajOptSolver.solve_batch or others instead"
        )
        if goal.n_goalset == 1:
            return self.solve_batch(
                goal, seed_traj, use_nn_seed, return_all_solutions, num_seeds, seed_success
            )
        if goal.n_goalset > 1:
            return self.solve_batch_goalset(
                goal, seed_traj, use_nn_seed, return_all_solutions, num_seeds, seed_success
            )

    def get_linear_seed(self, start_state, goal_state) -> torch.Tensor:
        """Get linearly interpolated seeds from start states to goal states.

        Args:
            start_state: start state of the robot.
            goal_state: goal state of the robot.

        Returns:
            torch.Tensor: Linearly interpolated seeds.
        """
        start_q = start_state.position.view(-1, 1, self.dof)
        end_q = goal_state.position.view(-1, 1, self.dof)
        edges = torch.cat((start_q, end_q), dim=1)

        seed = self.delta_vec @ edges

        # Setting final state to end_q explicitly to avoid matmul numerical precision issues.
        seed[..., -1:, :] = end_q

        return seed

    def get_start_seed(self, start_state) -> torch.Tensor:
        """Get trajectory seeds with start state repeated.

        Args:
            start_state: start state of the robot.

        Returns:
            torch.Tensor: Trajectory seeds with start state repeated.
        """
        start_q = start_state.position.view(-1, 1, self.dof)
        edges = torch.cat((start_q, start_q), dim=1)
        seed = self.delta_vec @ edges
        return seed

    def _get_seed_numbers(self, num_seeds: int) -> Dict[str, int]:
        """Get number of seeds for each seed type.

        Args:
            num_seeds: Total number of seeds to generate.

        Returns:
            Dict[str, int]: Number of seeds for each seed type.
        """
        n_seeds = {"linear": 0, "bias": 0, "start": 0, "goal": 0}
        k = n_seeds.keys
        t_seeds = 0
        for k in n_seeds:
            if k not in self.seed_ratio:
                continue
            if self.seed_ratio[k] > 0.0:
                n_seeds[k] = math.floor(self.seed_ratio[k] * num_seeds)
                t_seeds += n_seeds[k]
        if t_seeds < num_seeds:
            n_seeds["linear"] += num_seeds - t_seeds
        return n_seeds

    def get_seed_set(
        self,
        goal: Goal,
        seed_traj: Union[JointState, torch.Tensor, None] = None,  # n_seeds,batch, h, dof
        seed_success: Optional[torch.Tensor] = None,  # batch, n_seeds
        num_seeds: Optional[int] = None,
        batch_mode: bool = False,
    ):
        """Get seed set for optimization.

        Args:
            goal: Goal object containing target pose or joint configuration.
            batch: _description_
            h: _description_
            seed_traj: _description_
            dofseed_success: _description_
            n_seedsnum_seeds: _description_
            batch_mode: _description_

        Returns:
            _description_
        """
        total_seeds = goal.batch * num_seeds

        if isinstance(seed_traj, JointState):
            seed_traj = seed_traj.position
        if seed_traj is None:
            if goal.goal_state is not None and self.use_cspace_seed:
                # get linear seed
                seed_traj = self.get_seeds(goal.current_state, goal.goal_state, num_seeds=num_seeds)
            else:
                # get start repeat seed:
                log_info("No goal state found, using current config to seed")
                seed_traj = self.get_seeds(
                    goal.current_state, goal.current_state, num_seeds=num_seeds
                )
        elif seed_success is not None:
            lin_seed_traj = self.get_seeds(goal.current_state, goal.goal_state, num_seeds=num_seeds)
            lin_seed_traj[seed_success] = seed_traj  # [seed_success]
            seed_traj = lin_seed_traj
            total_seeds = goal.batch * num_seeds
        elif num_seeds > seed_traj.shape[0]:
            new_seeds = self.get_seeds(
                goal.current_state, goal.goal_state, num_seeds - seed_traj.shape[0]
            )
            seed_traj = torch.cat((seed_traj, new_seeds), dim=0)

        if len(seed_traj.shape) == 3:
            if (
                seed_traj.shape[0] != total_seeds
                or seed_traj.shape[1] != self.action_horizon
                or seed_traj.shape[2] != self.dof
            ):
                log_error(
                    "Seed traj shape should be [num_seeds * batch, action_horizon, dof]"
                    + " current shape is "
                    + str(seed_traj.shape)
                )
        elif len(seed_traj.shape) == 4:
            if (
                seed_traj.shape[0] * seed_traj.shape[1] != total_seeds
                or seed_traj.shape[2] != self.action_horizon
                or seed_traj.shape[3] != self.dof
            ):
                log_error(
                    "Seed traj shape should be [num_seeds, batch, action_horizon, dof]"
                    + " or [1, num_seeds * batch, action_horizon, dof]"
                    + " current shape is "
                    + str(seed_traj.shape)
                )
        else:
            log_error("Seed traj shape should have 3 or 4 dimensions: " + str(seed_traj.shape))
        seed_traj = seed_traj.view(total_seeds, self.action_horizon, self.dof)
        return seed_traj

    def get_seeds(
        self, start_state: JointState, goal_state: JointState, num_seeds: int = None
    ) -> torch.Tensor:
        """Get seed trajectories for optimization.

        Args:
            start_state: Start state of the robot.
            goal_state: Goal state of the robot.
            num_seeds: Number of seeds to generate. If None, the number of seeds is set to
                :attr:`TrajOptSolver.num_seeds`.

        Returns:
            torch.Tensor: Seed trajectories of shape [num_seeds, batch, action_horizon, dof]
        """
        # repeat seeds:
        if num_seeds is None:
            num_seeds = self.num_seeds
            n_seeds = self._n_seeds
        else:
            n_seeds = self._get_seed_numbers(num_seeds)
        # linear seed: batch x dof -> batch x n_seeds x dof
        seed_set = []
        if n_seeds["linear"] > 0:
            linear_seed = self.get_linear_seed(start_state, goal_state)

            linear_seeds = linear_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["linear"], 1, 1
            )
            seed_set.append(linear_seeds)
        if n_seeds["bias"] > 0:
            bias_seed = self.get_bias_seed(start_state, goal_state)
            bias_seeds = bias_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["bias"], 1, 1
            )
            seed_set.append(bias_seeds)
        if n_seeds["start"] > 0:
            bias_seed = self.get_start_seed(start_state)

            bias_seeds = bias_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["start"], 1, 1
            )
            seed_set.append(bias_seeds)
        if n_seeds["goal"] > 0:
            bias_seed = self.get_start_seed(goal_state)

            bias_seeds = bias_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["goal"], 1, 1
            )
            seed_set.append(bias_seeds)
        all_seeds = torch.cat(seed_set, dim=1)

        return all_seeds

    def get_bias_seed(self, start_state, goal_state) -> torch.Tensor:
        """Get seed trajectories that pass through the retract configuration at mid-waypoint.

        Args:
            start_state: start state of the robot.
            goal_state: goal state of the robot.

        Returns:
            torch.Tensor: Seed trajectories of shape [num_seeds * batch, action_horizon, dof].
        """
        start_q = start_state.position.view(-1, 1, self.dof)
        end_q = goal_state.position.view(-1, 1, self.dof)

        bias_q = self.bias_node.view(-1, 1, self.dof).repeat(start_q.shape[0], 1, 1)
        edges = torch.cat((start_q, bias_q, end_q), dim=1)
        seed = self.waypoint_delta_vec @ edges

        # Setting final state to end_q explicitly to avoid matmul numerical precision issues.
        seed[..., -1:, :] = end_q

        return seed

    @profiler.record_function("trajopt/interpolation")
    def get_interpolated_trajectory(
        self, traj_state: JointState
    ) -> Tuple[JointState, torch.Tensor]:
        """Get interpolated trajectory from optimized trajectories.

        This function will first find the optimal dt for each trajectory in the batch by scaling
        the trajectories to joint limits. Then it will interpolate the trajectory using the optimal
        dt.

        Args:
            traj_state: Optimized trajectories of shape [num_seeds * batch, action_horizon, dof].

        Returns:
            Tuple[JointState, torch.Tensor, torch.Tensor]: Interpolated trajectory, last time step
                for each trajectory in batch, optimal dt for each trajectory in batch.
        """
        # do interpolation:
        if (
            self._interpolated_traj_buffer is None
            or traj_state.position.shape[0] != self._interpolated_traj_buffer.position.shape[0]
        ):
            b, _, dof = traj_state.position.shape
            self._interpolated_traj_buffer = JointState.zeros(
                (b, self.interpolation_steps, dof), self.tensor_args
            )
            self._interpolated_traj_buffer.joint_names = self.rollout_fn.joint_names
        interpolation_buffer_reallocated = False
        state, last_tstep, opt_dt = get_batch_interpolated_trajectory(
            traj_state,
            self.solver_dt_tensor,
            self.interpolation_dt,
            self._max_joint_vel,
            self._max_joint_acc,
            self._max_joint_jerk,
            kind=self.interpolation_type,
            tensor_args=self.tensor_args,
            out_traj_state=self._interpolated_traj_buffer,
            min_dt=self.traj_evaluator_config.min_dt,
            max_dt=self.traj_evaluator_config.max_dt,
            optimize_dt=self.optimize_dt,
        )

        if state.shape != self._interpolated_traj_buffer.shape:
            interpolation_buffer_reallocated = True
            if is_cuda_graph_reset_available():
                log_info("Interpolated trajectory buffer was recreated, reinitializing cuda graph")
                self._interpolated_traj_buffer = state.clone()
        return state, last_tstep, opt_dt, interpolation_buffer_reallocated

    def calculate_trajectory_dt(
        self,
        trajectory: JointState,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """Calculate the optimal dt for a given trajectory by scaling it to joint limits.

        Args:
            trajectory: Trajectory to calculate optimal dt for.
            epsilon: Small value to improve numerical stability.

        Returns:
            torch.Tensor: Optimal trajectory dt.
        """
        opt_dt = calculate_dt_no_clamp(
            trajectory.velocity,
            trajectory.acceleration,
            trajectory.jerk,
            self._max_joint_vel,
            self._max_joint_acc,
            self._max_joint_jerk,
            epsilon=epsilon,
        )
        return opt_dt

    def reset_seed(self):
        """Reset the seed for random number generators in MPPI and rollout functions."""
        self.solver.reset_seed()

    def reset_cuda_graph(self):
        """Clear all recorded CUDA graphs. This does not work."""
        self.solver.reset_cuda_graph()
        self.interpolate_rollout.reset_cuda_graph()
        self.rollout_fn.reset_cuda_graph()

    def reset_shape(self):
        """Reset the shape of the rollout function and the solver."""
        self.solver.reset_shape()
        self.interpolate_rollout.reset_shape()
        self.rollout_fn.reset_shape()

    @property
    def kinematics(self) -> CudaRobotModel:
        """Get the kinematics instance of the robot."""
        return self.rollout_fn.dynamics_model.robot_model

    @property
    def retract_config(self) -> torch.Tensor:
        """Get the retract/home configuration of the robot.

        Returns:
            torch.Tensor: Retract configuration of the robot.
        """
        return self.rollout_fn.dynamics_model.retract_config.view(1, -1)

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        """Compute forward kinematics for the robot.

        Args:
            q: Joint configuration of the robot.

        Returns:
            CudaRobotModelState: Forward kinematics of the robot.
        """
        return self.kinematics.get_state(q)

    @property
    def solver_dt(self) -> torch.Tensor:
        """Get the current trajectory dt for the solver.

        Returns:
            torch.Tensor: Trajectory dt for the solver.
        """
        return self.solver.safety_rollout.dynamics_model.traj_dt[0]

    @property
    def solver_dt_tensor(self) -> torch.Tensor:
        """Get the current trajectory dt for the solver.

        Returns:
            torch.Tensor: Trajectory dt for the solver.
        """
        return self.solver.safety_rollout.dynamics_model.traj_dt[0]

    @property
    def minimum_trajectory_dt(self) -> float:
        """Get the minimum trajectory dt that is allowed, smaller dt will be clamped to this value.

        Returns:
            float: Minimum trajectory dt that is allowed.
        """
        return self.traj_evaluator.min_dt

    def update_solver_dt(
        self,
        dt: Union[float, torch.Tensor],
        base_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        base_ratio: Optional[float] = None,
    ):
        """Update the trajectory dt for the solver.

        This dt is used to calculate the velocity, acceleration and jerk of the trajectory through
        five point stencil (finite difference).

        Args:
            dt: New trajectory dt.
            base_dt: Base dt for the trajectory. This is not supported.
            max_dt: Maximum dt for the trajectory. This is not supported.
            base_ratio: Ratio in trajectory length to scale from base_dt to max_dt.  This is not
                supported.
        """
        all_rollouts = self.get_all_rollout_instances()
        for rollout in all_rollouts:
            rollout.update_traj_dt(dt, base_dt, max_dt, base_ratio)

    def get_full_js(self, active_js: JointState) -> JointState:
        """Get full joint state from controlled joint state, appending locked joints.

        Args:
            active_js: Controlled joint state

        Returns:
            JointState: Full joint state.
        """
        return self.rollout_fn.get_full_dof_from_solution(active_js)

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
    ):
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

        rollouts = self.get_all_rollout_instances()
        [
            rollout.update_pose_cost_metric(metric)
            for rollout in rollouts
            if isinstance(rollout, ArmReacher)
        ]

    @property
    def newton_iters(self):
        """Get the number of newton outer iterations during L-BFGS optimization.

        Returns:
            int: Number of newton outer iterations during L-BFGS optimization.
        """
        return self._og_newton_iters

    @property
    def dof(self) -> int:
        """Get the number of degrees of freedom of the robot.

        Returns:
            int: Number of degrees of freedom of the robot.
        """
        return self.rollout_fn.d_action

    @property
    def action_horizon(self) -> int:
        """Get the action horizon of the trajectory optimization problem.

        Number of actions in trajectory optimization can be smaller than number of waypoints as
        the first waypoint is the current state of the robot and the last two waypoints are
        the same as T-2 waypoint to implicitly enforce zero acceleration and zero velocity at T.

        Returns:
            int: Action horizon of the trajectory optimization problem.
        """
        return self.rollout_fn.action_horizon


@get_torch_jit_decorator()
def jit_feasible_success(
    feasible,
    position_error: Union[torch.Tensor, None],
    rotation_error: Union[torch.Tensor, None],
    cspace_error: Union[torch.Tensor, None],
    position_threshold: float,
    rotation_threshold: float,
    cspace_threshold: float,
):
    """JIT function to check if the optimization is successful."""
    feasible = torch.all(feasible, dim=-1)
    converge = feasible
    if position_error is not None and rotation_error is not None:
        converge = torch.logical_and(
            position_error[..., -1] <= position_threshold,
            rotation_error[..., -1] <= rotation_threshold,
        )
    elif cspace_error is not None:
        converge = cspace_error[..., -1] <= cspace_threshold

    success = torch.logical_and(feasible, converge)
    return success


@get_torch_jit_decorator(only_valid_for_compile=True)
def jit_trajopt_best_select(
    success,
    smooth_label,
    cspace_error: Union[torch.Tensor, None],
    pose_error: Union[torch.Tensor, None],
    position_error: Union[torch.Tensor, None],
    rotation_error: Union[torch.Tensor, None],
    goalset_index: Union[torch.Tensor, None],
    cost,
    smooth_cost,
    batch_mode: bool,
    batch: int,
    num_seeds: int,
    col,
    opt_dt,
):
    """JIT function to select the best solution from optimized seeds."""
    success[~smooth_label] = False
    convergence_error = 0
    # get the best solution:
    if pose_error is not None:
        convergence_error = pose_error[..., -1]
    elif cspace_error is not None:
        convergence_error = cspace_error[..., -1]

    running_cost = torch.mean(cost, dim=-1) * 0.0001
    error = convergence_error + smooth_cost + running_cost
    error[~success] += 10000.0
    if batch_mode:
        idx = torch.argmin(error.view(batch, num_seeds), dim=-1)
        idx = idx + num_seeds * col
        success = success[idx]
    else:
        idx = torch.argmin(error, dim=0)

        success = success[idx : idx + 1]

    # goalset_index = position_error = rotation_error = cspace_error = None
    if position_error is not None:
        position_error = position_error[idx, -1]
    if rotation_error is not None:
        rotation_error = rotation_error[idx, -1]
    if cspace_error is not None:
        cspace_error = cspace_error[idx, -1]
    if goalset_index is not None:
        goalset_index = goalset_index[idx, -1]

    opt_dt = opt_dt[idx]

    return idx, position_error, rotation_error, cspace_error, goalset_index, opt_dt, success
