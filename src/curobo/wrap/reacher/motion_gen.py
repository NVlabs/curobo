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


from __future__ import annotations

# Standard Library
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
from curobo.types.tensor import T_BDOF, T_BValue_bool, T_BValue_float
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import tensor_repeat_seeds
from curobo.util.trajectory import InterpolateType
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
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig
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

    @staticmethod
    @profiler.record_function("motion_gen_config/load_from_robot_config")
    def load_from_robot_config(
        robot_cfg: Union[Union[str, Dict], RobotConfig],
        world_model: Optional[Union[Union[str, Dict], WorldConfig]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        num_ik_seeds: int = 30,
        num_graph_seeds: int = 1,
        num_trajopt_seeds: int = 12,
        num_batch_ik_seeds: int = 30,
        num_batch_trajopt_seeds: int = 1,
        num_trajopt_noisy_seeds: int = 1,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        cspace_threshold: float = 0.05,
        world_coll_checker=None,
        base_cfg_file: str = "base_cfg.yml",
        particle_ik_file: str = "particle_ik.yml",
        gradient_ik_file: str = "gradient_ik.yml",
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
        filter_robot_command: bool = True,
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
        finetune_dt_scale: float = 0.98,
        maximum_trajectory_time: Optional[float] = None,
        maximum_trajectory_dt: float = 0.1,
        velocity_scale: Optional[Union[List[float], float]] = None,
        acceleration_scale: Optional[Union[List[float], float]] = None,
        jerk_scale: Optional[Union[List[float], float]] = None,
        optimize_dt: bool = True,
        project_pose_to_goal_frame: bool = True,
    ):
        """Helper function to create configuration from robot and world configuration.

        Args:
            robot_cfg: Robot configuration to use for motion generation.
            world_model: World model to use for motion generation. Use None to disable world model.
            tensor_args: Tensor device to use for motion generation. Use to change cuda device id.
            num_ik_seeds: Number of seeds to use in inverse kinematics (IK) optimization.
            num_graph_seeds: Number of graph paths to use as seed in trajectory optimization.
            num_trajopt_seeds: Number of seeds to use in trajectory optimization.
            num_batch_ik_seeds: Number of seeds to use in batch planning modes for IK.
            num_batch_trajopt_seeds: Number of seeds to use in batch planning modes for trajopt.
            num_trajopt_noisy_seeds: Number of seeds to use for trajopt.
            position_threshold: _description_
            rotation_threshold: _description_
            cspace_threshold: _description_
            world_coll_checker: _description_
            base_cfg_file: _description_
            particle_ik_file: _description_
            gradient_ik_file: _description_
            graph_file: _description_
            particle_trajopt_file: _description_
            gradient_trajopt_file: _description_
            finetune_trajopt_file: _description_
            trajopt_tsteps: _description_
            interpolation_steps: _description_
            interpolation_dt: _description_
            interpolation_type: _description_
            use_cuda_graph: _description_
            self_collision_check: _description_
            self_collision_opt: _description_
            grad_trajopt_iters: _description_
            trajopt_seed_ratio: _description_
            ik_opt_iters: _description_
            ik_particle_opt: _description_
            collision_checker_type: _description_
            sync_cuda_time: _description_
            trajopt_particle_opt: _description_
            traj_evaluator_config: _description_
            traj_evaluator: _description_
            minimize_jerk: _description_
            filter_robot_command: _description_
            use_gradient_descent: _description_
            collision_cache: _description_
            n_collision_envs: _description_
            ee_link_name: _description_
            use_es_ik: _description_
            use_es_trajopt: _description_
            es_ik_learning_rate: _description_
            es_trajopt_learning_rate: _description_
            use_ik_fixed_samples: _description_
            use_trajopt_fixed_samples: _description_
            evaluate_interpolated_trajectory: _description_
            partial_ik_iters: _description_
            fixed_iters_trajopt: _description_
            store_ik_debug: _description_
            store_trajopt_debug: _description_
            graph_trajopt_iters: _description_
            collision_max_outside_distance: _description_
            collision_activation_distance: _description_
            trajopt_dt: _description_
            js_trajopt_dt: _description_
            js_trajopt_tsteps: _description_
            trim_steps: _description_
            store_debug_in_result: _description_
            finetune_trajopt_iters: _description_
            smooth_weight: _description_
            finetune_smooth_weight: _description_
            state_finite_difference_mode: _description_
            finetune_dt_scale: _description_
            maximum_trajectory_time: _description_
            maximum_trajectory_dt: _description_
            velocity_scale: _description_
            acceleration_scale: _description_
            jerk_scale: _description_
            optimize_dt: _description_

        Returns:
            _description_
        """

        init_warp(tensor_args=tensor_args)
        if js_trajopt_tsteps is not None:
            log_warn("js_trajopt_tsteps is deprecated, use trajopt_tsteps instead.")
            trajopt_tsteps = js_trajopt_tsteps
        if trajopt_tsteps is not None:
            js_trajopt_tsteps = trajopt_tsteps
        if velocity_scale is not None and isinstance(velocity_scale, float):
            velocity_scale = [velocity_scale]

        if acceleration_scale is not None and isinstance(acceleration_scale, float):
            acceleration_scale = [acceleration_scale]
        if jerk_scale is not None and isinstance(jerk_scale, float):
            jerk_scale = [jerk_scale]

        if store_ik_debug or store_trajopt_debug:
            store_debug_in_result = True
        if (
            acceleration_scale is not None
            and min(acceleration_scale) < 1.0
            and maximum_trajectory_dt <= 0.1
        ):
            maximum_trajectory_dt = np.sqrt(1.0 / min(acceleration_scale)) * maximum_trajectory_dt
        elif (
            velocity_scale is not None
            and min(velocity_scale) < 1.0
            and maximum_trajectory_dt <= 0.1
        ):
            maximum_trajectory_dt = (1.0 / min(velocity_scale)) * maximum_trajectory_dt

        if traj_evaluator_config is None:
            if maximum_trajectory_dt is not None:
                max_dt = maximum_trajectory_dt
            if maximum_trajectory_time is not None:
                max_dt = maximum_trajectory_time / trajopt_tsteps
            if acceleration_scale is not None:
                max_dt = max_dt * (1.0 / np.sqrt(min(acceleration_scale)))
            traj_evaluator_config = TrajEvaluatorConfig(min_dt=interpolation_dt, max_dt=max_dt)
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
            and min(velocity_scale) < 0.1
            and finetune_trajopt_file is None
        ):
            log_error(
                "velocity scale<0.1 is not supported with default finetune_trajopt.yml "
                + "provide your own finetune_trajopt_file to override this error"
            )

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
        traj_evaluator_config.max_acc = (
            robot_cfg.kinematics.get_joint_limits().acceleration[1, 0].item()
        )
        traj_evaluator_config.max_jerk = robot_cfg.kinematics.get_joint_limits().jerk[1, 0].item()

        if isinstance(world_model, str):
            world_model = load_yaml(join_path(get_world_configs_path(), world_model))

        base_config_data = load_yaml(join_path(get_task_configs_path(), base_cfg_file))
        if collision_cache is not None:
            base_config_data["world_collision_checker_cfg"]["cache"] = collision_cache
        if n_collision_envs is not None:
            base_config_data["world_collision_checker_cfg"]["n_envs"] = n_collision_envs
        if collision_max_outside_distance is not None:
            assert collision_max_outside_distance >= 0.0
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
            # use_gradient_descent=use_gradient_descent,
            use_es=use_es_ik,
            es_learning_rate=es_ik_learning_rate,
            use_fixed_samples=use_ik_fixed_samples,
            store_debug=store_ik_debug,
            collision_activation_distance=collision_activation_distance,
            project_pose_to_goal_frame=project_pose_to_goal_frame,
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
            evaluate_interpolated_trajectory=True,
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
        )

        finetune_trajopt_solver = TrajOptSolver(finetune_trajopt_cfg)
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
        )


@dataclass
class MotionGenResult:
    """Result obtained from motion generation."""

    #: success tensor with index referring to the batch index.
    success: Optional[T_BValue_bool] = None

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

    #: status of motion generation query. returns [IK Fail, Graph Fail, TrajOpt Fail].
    status: Optional[str] = None

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

    @staticmethod
    def _check_none_and_copy_idx(
        current_tensor: Union[torch.Tensor, JointState, None],
        source_tensor: Union[torch.Tensor, JointState, None],
        idx: int,
    ):
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

    def copy_idx(self, idx: torch.Tensor, source_result: MotionGenResult):
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
        path = [
            self.interpolated_plan[x].trim_trajectory(0, self.path_buffer_last_tstep[x])
            for x in range(len(self.interpolated_plan))
        ]
        return path

    def get_successful_paths(self) -> List[torch.Tensor]:
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
        if self.path_buffer_last_tstep is None:
            return self.interpolated_plan
        if len(self.path_buffer_last_tstep) > 1:
            raise ValueError("only single result is supported")

        return self.interpolated_plan.trim_trajectory(0, self.path_buffer_last_tstep[0])

    @property
    def motion_time(self):
        # -2 as last three timesteps have the same value
        # 0, 1 also have the same position value.
        return self.optimized_dt * (self.optimized_plan.position.shape[-2] - 1 - 2 - 1)


@dataclass
class MotionGenPlanConfig:
    enable_graph: bool = False
    enable_opt: bool = True
    use_nn_ik_seed: bool = False
    need_graph_success: bool = False
    max_attempts: int = 60
    timeout: float = 10.0
    enable_graph_attempt: Optional[int] = 3
    disable_graph_attempt: Optional[int] = None
    ik_fail_return: Optional[int] = None
    partial_ik_opt: bool = False
    num_ik_seeds: Optional[int] = None
    num_graph_seeds: Optional[int] = None
    num_trajopt_seeds: Optional[int] = None
    success_ratio: float = 1
    fail_on_invalid_query: bool = True

    #: enables retiming trajectory optimization, useful for getting low jerk trajectories.
    enable_finetune_trajopt: bool = True
    parallel_finetune: bool = True

    #: use start config as regularization
    use_start_state_as_retract: bool = True

    pose_cost_metric: Optional[PoseCostMetric] = None

    def __post_init__(self):
        if not self.enable_opt and not self.enable_graph:
            log_error("Graph search and Optimization are both disabled, enable one")

    def clone(self) -> MotionGenPlanConfig:
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
        )


class MotionGen(MotionGenConfig):
    def __init__(self, config: MotionGenConfig):
        super().__init__(**vars(config))
        self.rollout_fn = (
            self.graph_planner.safety_rollout_fn
        )  # TODO: compute_kinematics fn in rolloutbase
        self._trajopt_goal_config = None
        self._dof = self.rollout_fn.d_action
        self._batch_graph_search_buffer = None
        self._batch_path_buffer_last_tstep = None
        self._rollout_list = None
        self._solver_rollout_list = None
        self._pose_solver_rollout_list = None

        self._kin_list = None
        self.update_batch_size(seeds=self.trajopt_seeds)

    def update_batch_size(self, seeds=10, batch=1):
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
        return self.ik_solver.solve(
            goal_pose,
            retract_config,
            seed_config,
            return_seeds,
            num_seeds,
            use_nn_seed,
            newton_iters,
        )

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

    @profiler.record_function("motion_gen/graph_search")
    def graph_search(
        self, start_config: T_BDOF, goal_config: T_BDOF, interpolation_steps: Optional[int] = None
    ) -> GraphResult:
        return self.graph_planner.find_paths(start_config, goal_config, interpolation_steps)

    def _get_solve_state(
        self,
        solve_type: ReacherSolveType,
        plan_config: MotionGenPlanConfig,
        goal_pose: Pose,
        start_state: JointState,
    ):
        # TODO: for batch seeds
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
        start_time = time.time()

        if plan_config.pose_cost_metric is not None:
            valid_query = self.update_pose_cost_metric(
                plan_config.pose_cost_metric, start_state, goal_pose
            )
            if not valid_query:
                result = MotionGenResult(
                    success=torch.as_tensor([False], device=self.tensor_args.device),
                    valid_query=valid_query,
                    status="Invalid Hold Partial Pose",
                )
                return result
        # if plan_config.enable_opt:
        self.update_batch_size(seeds=solve_state.num_trajopt_seeds, batch=solve_state.batch_size)
        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                raise ValueError("Batch Env is less that goal batch")
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
        best_status = 0  # 0== None, 1==IK Fail, 2== Graph Fail, 3==Opt Fail

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
                result.status == "IK Fail" and plan_config.ik_fail_return is not None
            ):  # IF IK fails the first time, we exist assuming the goal is not reachable
                ik_fail_count += 1
                best_status = max(best_status, 1)

                if ik_fail_count > plan_config.ik_fail_return:
                    break
            if result.success[0].item():
                break
            if plan_config.enable_graph_attempt is not None and (
                n >= plan_config.enable_graph_attempt - 1
                and result.status == "Opt Fail"
                and not plan_config.enable_graph
            ):
                plan_config.enable_graph = True
                plan_config.partial_ik_opt = False
            if plan_config.disable_graph_attempt is not None and (
                n >= plan_config.disable_graph_attempt - 1
                and result.status in ["Opt Fail", "Graph Fail"]
                and not force_graph
            ):
                plan_config.enable_graph = False
                plan_config.partial_ik_opt = partial_ik
            if result.status in ["Opt Fail"]:
                best_status = 3
            elif result.status in ["Graph Fail"]:
                best_status = 2
            if time.time() - start_time > plan_config.timeout:
                break
            if not result.valid_query:
                result.status = "Invalid Problem"
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
        torch.cuda.synchronize()
        if plan_config.pose_cost_metric is not None:
            self.update_pose_cost_metric(PoseCostMetric.reset_metric())
        result.total_time = time.time() - start_time
        return result

    def _plan_batch_attempts(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    ):
        start_time = time.time()

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
                raise ValueError("Batch Env is less that goal batch")
            if plan_config.enable_graph:
                raise ValueError("Graph Search / Geometric Planner not supported in batch_env mode")

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
                solve_state, start_state, goal_pose, plan_config
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
                result.status == "IK Fail" and plan_config.ik_fail_return is not None
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
                and result.status != "IK Fail"
                and not plan_config.enable_graph
            ):
                plan_config.enable_graph = True
                plan_config.partial_ik_opt = False

            if plan_config.disable_graph_attempt is not None and (
                n >= plan_config.disable_graph_attempt - 1
                and result.status in ["Opt Fail", "Graph Fail"]
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
        torch.cuda.synchronize()
        if plan_config.pose_cost_metric is not None:
            self.update_pose_cost_metric(PoseCostMetric.reset_metric())
        best_result.total_time = time.time() - start_time
        return best_result

    def plan_single(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: List[Pose] = None,
    ) -> MotionGenResult:
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
    ) -> MotionGenResult:
        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH, plan_config, goal_pose, start_state
        )

        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
        )
        return result

    def plan_batch_goalset(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    ) -> MotionGenResult:
        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH_GOALSET, plan_config, goal_pose, start_state
        )

        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
        )
        return result

    def plan_batch_env_goalset(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    ) -> MotionGenResult:
        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH_ENV_GOALSET, plan_config, goal_pose, start_state
        )

        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
        )
        return result

    def plan_batch_env(
        self,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    ) -> MotionGenResult:
        solve_state = self._get_solve_state(
            ReacherSolveType.BATCH_ENV, plan_config, goal_pose, start_state
        )
        result = self._plan_batch_attempts(
            solve_state,
            start_state,
            goal_pose,
            plan_config,
        )
        return result

    def _plan_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        start_state: JointState,
        goal_pose: Pose,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> MotionGenResult:
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = None
        graph_success = 0
        if len(start_state.shape) == 1:
            log_error("Joint state should be not a vector (dof) should be (bxdof)")
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
            result.status = "IK Fail"
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
                result.status = "Graph Fail"
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
                        opt_dt = torch.min(opt_dt[traj_result.success])
                        seed_override = solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds
                    scaled_dt = torch.clamp(
                        opt_dt * self.finetune_dt_scale,
                        self.trajopt_solver.interpolation_dt,
                    )
                    self.finetune_trajopt_solver.update_solver_dt(scaled_dt.item())

                    traj_result = self._solve_trajopt_from_solve_state(
                        goal,
                        solve_state,
                        seed_traj,
                        trajopt_instance=self.finetune_trajopt_solver,
                        num_seeds_override=seed_override,
                    )

                result.finetune_time = traj_result.solve_time

                traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
            elif plan_config.enable_finetune_trajopt:
                traj_result.success = traj_result.success[0:1]
            result.solve_time += traj_result.solve_time + result.finetune_time
            result.trajopt_time = traj_result.solve_time
            result.trajopt_attempts = 1
            result.success = traj_result.success

            if torch.count_nonzero(result.success) == 0:
                result.status = "Opt Fail"

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
        trajopt_seed_traj = None
        trajopt_seed_success = None
        trajopt_newton_iters = None

        graph_success = 0
        if len(start_state.shape) == 1:
            log_error("Joint state should be not a vector (dof) should be (bxdof)")

        result = MotionGenResult(cspace_error=torch.zeros((1), device=self.tensor_args.device))
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
                result.status = "Graph Fail"
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
                # self._trajopt_goal_config[:, :ik_success] = goal_config

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
                    trajopt_seed_traj = trajopt_seed_traj.view(
                        self.js_trajopt_solver.num_seeds * 1,
                        1,
                        self.trajopt_solver.action_horizon,
                        self._dof,
                    ).contiguous()
            if plan_config.enable_finetune_trajopt:
                og_value = self.trajopt_solver.interpolation_type
                self.js_trajopt_solver.interpolation_type = InterpolateType.LINEAR_CUDA
            with profiler.record_function("motion_gen/trajopt"):
                log_info("MG: running TO")
                traj_result = self._solve_trajopt_from_solve_state(
                    goal,
                    solve_state,
                    trajopt_seed_traj,
                    num_seeds_override=solve_state.num_trajopt_seeds * 1,
                    newton_iters=trajopt_newton_iters,
                    return_all_solutions=plan_config.enable_finetune_trajopt,
                    trajopt_instance=self.js_trajopt_solver,
                )
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
                # self.trajopt_solver.compute_metrics(not og_evaluate, og_evaluate)
            if self.store_debug_in_result:
                result.debug_info["trajopt_result"] = traj_result
            if torch.count_nonzero(traj_result.success) == 0:
                result.status = "TrajOpt Fail"
            # run finetune
            if plan_config.enable_finetune_trajopt and torch.count_nonzero(traj_result.success) > 0:
                with profiler.record_function("motion_gen/finetune_trajopt"):
                    seed_traj = traj_result.raw_action.clone()  # solution.position.clone()
                    seed_traj = seed_traj.contiguous()
                    og_solve_time = traj_result.solve_time

                    scaled_dt = torch.clamp(
                        torch.max(traj_result.optimized_dt[traj_result.success]),
                        self.trajopt_solver.interpolation_dt,
                    )
                    og_dt = self.js_trajopt_solver.solver_dt
                    self.js_trajopt_solver.update_solver_dt(scaled_dt.item())

                    traj_result = self._solve_trajopt_from_solve_state(
                        goal,
                        solve_state,
                        seed_traj,
                        trajopt_instance=self.js_trajopt_solver,
                        num_seeds_override=solve_state.num_trajopt_seeds * self.noisy_trajopt_seeds,
                    )
                    self.js_trajopt_solver.update_solver_dt(og_dt)

                result.finetune_time = traj_result.solve_time

                traj_result.solve_time = og_solve_time
                if self.store_debug_in_result:
                    result.debug_info["finetune_trajopt_result"] = traj_result
                if torch.count_nonzero(traj_result.success) == 0:
                    result.status = "Finetune Fail"
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
    ) -> MotionGenResult:
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
            result.status = "IK Fail"
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
                result.status = "Graph Fail"
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
            )
            # generate seeds:
            if trajopt_seed_traj is None or (
                plan_config.enable_graph and graph_success < solve_state.batch_size
            ):
                seed_goal = Goal(
                    goal_pose=goal_pose.repeat_seeds(solve_state.num_trajopt_seeds),
                    current_state=start_state.repeat_seeds(solve_state.num_trajopt_seeds),
                    goal_state=JointState.from_position(goal_config.view(-1, self._dof)),
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
                return_all_solutions=True,
            )

            # output of traj result will have 1 solution per batch

            # run finetune opt on 1 solution per batch:
            if plan_config.enable_finetune_trajopt:
                self.trajopt_solver.interpolation_type = og_value
                # self.trajopt_solver.compute_metrics(not og_evaluate, og_evaluate)
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
                        self.trajopt_solver.interpolation_dt,
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
                result.status = "Opt Fail"
                result.success[:] = False
            if self.store_debug_in_result:
                result.debug_info = {"trajopt_result": traj_result}
        return result

    def compute_kinematics(self, state: JointState) -> KinematicModelState:
        out = self.rollout_fn.compute_kinematics(state)
        return out

    @property
    def kinematics(self):
        return self.rollout_fn.kinematics

    @property
    def dof(self):
        return self.kinematics.get_dof()

    def check_constraints(self, state: JointState) -> RolloutMetrics:
        metrics = self.ik_solver.check_constraints(state)
        return metrics

    def update_world(self, world: WorldConfig):
        self.world_coll_checker.load_collision_model(world, fix_cache_reference=self.use_cuda_graph)
        self.graph_planner.reset_graph()
        return True

    def clear_world_cache(self):
        self.world_coll_checker.clear_cache()

    def reset(self, reset_seed=True):
        self.graph_planner.reset_buffer()
        if reset_seed:
            self.reset_seed()

    def reset_seed(self):
        self.rollout_fn.reset_seed()
        self.ik_solver.reset_seed()
        self.graph_planner.reset_seed()
        self.trajopt_solver.reset_seed()

    def get_retract_config(self):
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
        log_info("Warmup")
        if warmup_js_trajopt:
            start_state = JointState.from_position(
                self.rollout_fn.dynamics_model.retract_config.view(1, -1).clone(),
                joint_names=self.rollout_fn.joint_names,
            )
            # warm up js_trajopt:
            goal_state = start_state.clone()
            goal_state.position[..., warmup_joint_index] += warmup_joint_delta
            for _ in range(2):
                self.plan_single_js(start_state, goal_state, MotionGenPlanConfig(max_attempts=1))
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
                for _ in range(2):
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
                for _ in range(2):
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

            if n_goalset == -1:
                retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
                start_state.position[..., warmup_joint_index] += warmup_joint_delta

                for _ in range(2):
                    if batch_env_mode:
                        self.plan_batch_env(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=enable_graph,
                            ),
                        )
                    else:
                        self.plan_batch(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=enable_graph,
                            ),
                        )
            else:
                retract_pose = Pose(
                    state.ee_pos_seq.view(batch, 1, 3).repeat(1, n_goalset, 1).contiguous(),
                    quaternion=state.ee_quat_seq.view(batch, 1, 4)
                    .repeat(1, n_goalset, 1)
                    .contiguous(),
                )
                start_state.position[..., warmup_joint_index] += warmup_joint_delta
                for _ in range(2):
                    if batch_env_mode:
                        self.plan_batch_env_goalset(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=enable_graph,
                                enable_graph_attempt=20,
                            ),
                        )
                    else:
                        self.plan_batch_goalset(
                            start_state,
                            retract_pose,
                            MotionGenPlanConfig(
                                max_attempts=10,
                                enable_finetune_trajopt=True,
                                enable_graph=enable_graph,
                                enable_graph_attempt=20,
                            ),
                        )

        log_info("Warmup complete")
        return True

    def plan_single_js(
        self,
        start_state: JointState,
        goal_state: JointState,
        plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    ) -> MotionGenResult:
        # NOTE: currently runs only one seed
        time_dict = {
            "solve_time": 0,
            "ik_time": 0,
            "graph_time": 0,
            "trajopt_time": 0,
            "trajopt_attempts": 0,
        }
        result = None
        goal = Goal(goal_state=goal_state, current_state=start_state)
        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE,
            num_ik_seeds=1,
            num_trajopt_seeds=self.js_trajopt_solver.num_seeds,
            num_graph_seeds=self.js_trajopt_solver.num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=1,
        )
        for n in range(plan_config.max_attempts):
            traj_result = self._plan_js_from_solve_state(
                solve_state, start_state, goal_state, plan_config=plan_config
            )
            time_dict["trajopt_time"] += traj_result.solve_time
            time_dict["trajopt_attempts"] = n

            if result is None:
                result = traj_result

            if traj_result.success.item():
                break

        result.solve_time = time_dict["trajopt_time"]
        result.total_time = result.solve_time
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

    def solve_trajopt(
        self,
        goal: Goal,
        act_seed,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
    ):
        result = self.trajopt_solver.solve(
            goal, act_seed, return_all_solutions=return_all_solutions, num_seeds=num_seeds
        )
        return result

    def get_active_js(
        self,
        in_js: JointState,
    ):
        opt_jnames = self.rollout_fn.joint_names
        opt_js = in_js.get_ordered_joint_state(opt_jnames)
        return opt_js

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
        start_state: Optional[JointState] = None,
        goal_pose: Optional[Pose] = None,
    ) -> bool:
        # check if constraint is valid:
        if metric.hold_partial_pose and metric.offset_tstep_fraction < 0.0:
            start_pose = self.compute_kinematics(start_state).ee_pose.clone()
            if self.project_pose_to_goal_frame:
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

        rollouts = self.get_all_pose_solver_rollout_instances()
        [
            rollout.update_pose_cost_metric(metric)
            for rollout in rollouts
            if isinstance(rollout, ArmReacher)
        ]
        torch.cuda.synchronize()
        return True

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        if self._rollout_list is None:
            self._rollout_list = (
                self.ik_solver.get_all_rollout_instances()
                + self.graph_planner.get_all_rollout_instances()
                + self.trajopt_solver.get_all_rollout_instances()
                + self.finetune_trajopt_solver.get_all_rollout_instances()
                + self.js_trajopt_solver.get_all_rollout_instances()
            )
        return self._rollout_list

    def get_all_solver_rollout_instances(self) -> List[RolloutBase]:
        if self._solver_rollout_list is None:
            self._solver_rollout_list = (
                self.ik_solver.solver.get_all_rollout_instances()
                + self.trajopt_solver.solver.get_all_rollout_instances()
                + self.finetune_trajopt_solver.solver.get_all_rollout_instances()
                + self.js_trajopt_solver.solver.get_all_rollout_instances()
            )
        return self._solver_rollout_list

    def get_all_pose_solver_rollout_instances(self) -> List[RolloutBase]:
        if self._pose_solver_rollout_list is None:
            self._pose_solver_rollout_list = (
                self.ik_solver.solver.get_all_rollout_instances()
                + self.trajopt_solver.solver.get_all_rollout_instances()
                + self.finetune_trajopt_solver.solver.get_all_rollout_instances()
            )
        return self._pose_solver_rollout_list

    def get_all_kinematics_instances(self) -> List[CudaRobotModel]:
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
    ) -> None:
        """Attach an object from world to robot's link.

        Args:
            surface_sphere_radius: _description_. Defaults to None.
            sphere_tensor: _description_. Defaults to None.
            link_name: _description_. Defaults to "attached_object".
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
            log_error("MG: No spheres found")
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

    def attach_external_objects_to_robot(
        self,
        joint_state: JointState,
        external_objects: List[Obstacle],
        surface_sphere_radius: float = 0.001,
        link_name: str = "attached_object",
        sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        voxelize_method: str = "ray",
        world_objects_pose_offset: Optional[Pose] = None,
    ) -> None:
        """Attach an object from world to robot's link.

        Args:
            surface_sphere_radius: _description_. Defaults to None.
            sphere_tensor: _description_. Defaults to None.
            link_name: _description_. Defaults to "attached_object".
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
            log_error("MG: No spheres found")
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

    def add_camera_frame(self, camera_observation: CameraObservation, obstacle_name: str):
        self.world_coll_checker.add_camera_frame(camera_observation, obstacle_name)

    def process_camera_frames(self, obstacle_name: Optional[str] = None, process_aux: bool = False):
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

        NOTE: This is not currently implemented.

        Args:
            joint_state (JointState): _description_
            bounding_box (Cuboid): _description_
            blox_layer_name (Optional[str], optional): _description_. Defaults to None.
            surface_sphere_radius (float, optional): _description_. Defaults to 0.001.
            link_name (str, optional): _description_. Defaults to "attached_object".
            sphere_fit_type (SphereFitType, optional): _description_. Defaults to SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE.
            voxelize_method (str, optional): _description_. Defaults to "ray".
            world_objects_pose_offset (Optional[Pose], optional): _description_. Defaults to None.
        """
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

        Args:
            joint_state (JointState): _description_
            obstacle (Obstacle): _description_
            surface_sphere_radius (float, optional): _description_. Defaults to 0.001.
            link_name (str, optional): _description_. Defaults to "attached_object".
            sphere_fit_type (SphereFitType, optional): _description_. Defaults to SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE.
            voxelize_method (str, optional): _description_. Defaults to "ray".
            world_objects_pose_offset (Optional[Pose], optional): _description_. Defaults to None.
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
        n_spheres = max_spheres
        sphere_tensor = torch.zeros((max_spheres, 4))
        sphere_tensor[:, 3] = -10.0
        sph_list = []
        if n_spheres == 0:
            log_error("MG: No spheres found")
        sph = obstacle.get_bounding_spheres(
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

    def detach_object_from_robot(self, link_name: str = "attached_object") -> None:
        return self.detach_spheres_from_robot(link_name)

    def attach_spheres_to_robot(
        self,
        sphere_radius: Optional[float] = None,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> None:
        """Attach spheres to robot's link.

        Args:
            sphere_radius: _description_. Defaults to None.
            sphere_tensor: _description_. Defaults to None.
            link_name: _description_. Defaults to "attached_object".
        """
        self.robot_cfg.kinematics.kinematics_config.attach_object(
            sphere_radius=sphere_radius, sphere_tensor=sphere_tensor, link_name=link_name
        )

    def detach_spheres_from_robot(self, link_name: str = "attached_object") -> None:
        self.robot_cfg.kinematics.kinematics_config.detach_object(link_name)

    def get_full_js(self, active_js: JointState) -> JointState:
        return self.rollout_fn.get_full_dof_from_solution(active_js)

    def add_running_pose_constraint(
        self,
        lock_x: bool = False,
        lock_y: bool = False,
        lock_z: bool = False,
        lock_rx: bool = False,
        lock_ry: bool = False,
        lock_rz: bool = False,
    ):
        raise NotImplementedError()

    def remove_running_pose_constraint(self):
        raise NotImplementedError()

    def run_finetune_trajopt(
        self,
        start_state: JointState,
        goal_pose: Pose,
        traj_solution: JointState,
        traj_dt: Union[float, torch.Tensor, None],
        max_attempts: int = 1,
    ):
        # NOTE: Currently only works for single environment. Need to rewrite for all modes
        # finetunes solution
        if traj_dt is not None:
            self.finetune_trajopt_solver.update_solver_dt(traj_dt.item())

        # call trajopt with seed:

        # NOTE: currently runs only one seed
        time_dict = {
            "solve_time": 0,
            "ik_time": 0,
            "graph_time": 0,
            "trajopt_time": 0,
            "trajopt_attempts": 0,
        }
        result = None
        # goal_state = JointState.from_position(
        #    traj_solution.position[..., -2:-1, :].clone(), joint_names=start_state.joint_names
        # )
        goal = Goal(
            goal_pose=goal_pose,
            # goal_state=goal_state,
            current_state=start_state,
        )

        for n in range(max_attempts):
            traj_result = self.finetune_trajopt_solver.solve_single(goal, traj_solution)
            time_dict["trajopt_time"] += traj_result.solve_time
            time_dict["trajopt_attempts"] = n
            if result is None:
                result = MotionGenResult(success=traj_result.success)

            if traj_result.success.item():
                break

        if self.store_debug_in_result:
            result.debug_info = {"trajopt_result": traj_result}

        result.position_error = traj_result.position_error
        result.rotation_error = traj_result.rotation_error
        result.cspace_error = traj_result.cspace_error
        result.optimized_dt = traj_result.optimized_dt
        result.interpolated_plan = traj_result.interpolated_solution
        result.optimized_plan = traj_result.solution
        result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
        result.trajopt_time = time_dict["trajopt_time"]
        return result

    @property
    def world_model(self) -> WorldConfig:
        return self.world_coll_checker.world_model

    @property
    def world_collision(self) -> WorldCollision:
        return self.world_coll_checker

    @property
    def project_pose_to_goal_frame(self) -> bool:
        return self.trajopt_solver.rollout_fn.goal_cost.project_distance

    def update_interpolation_type(
        self,
        interpolation_type: InterpolateType,
        update_graph: bool = True,
        update_trajopt: bool = True,
    ):
        if update_graph:
            self.graph_planner.interpolation_type = interpolation_type
        if update_trajopt:
            self.trajopt_solver.interpolation_type = interpolation_type
            self.finetune_trajopt_solver.interpolation_type = interpolation_type
            self.js_trajopt_solver.interpolation_type = interpolation_type

    def update_locked_joints(
        self, lock_joints: Dict[str, float], robot_config_dict: Union[str, Dict[Any]]
    ):
        if isinstance(robot_config_dict, str):
            robot_config_dict = load_yaml(join_path(get_robot_configs_path(), robot_config_dict))[
                "robot_cfg"
            ]
        if "robot_cfg" in robot_config_dict:
            robot_config_dict = robot_config_dict["robot_cfg"]
        robot_config_dict["kinematics"]["lock_joints"] = lock_joints
        robot_cfg = RobotConfig.from_dict(robot_config_dict, self.tensor_args)

        # make sure the new robot config and current have the same joint limits:
        new_joint_limits = robot_cfg.kinematics.get_joint_limits()
        current_joint_limits = self.robot_cfg.kinematics.get_joint_limits()
        # if new_joint_limits != current_joint_limits:
        #    log_error("Joint limits are different, reinstance motion gen")

        self.kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
