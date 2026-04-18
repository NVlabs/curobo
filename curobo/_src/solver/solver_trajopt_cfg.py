# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclass for the trajectory optimization solver.

Bundles SolverCoreCfg with trajopt-specific parameters such as dt bounds,
interpolation settings, finetune passes, and the factory method
:meth:`TrajOptSolverCfg.create` that builds the full config from YAML paths
or dictionaries.
"""
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

# CuRobo
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.solver.solver_core_cfg import (
    SolverCoreCfg,
    create_solver_core_cfg,
    resolve_yaml_configs,
)
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import TrajInterpolationType


@dataclass
class TrajOptSolverCfg:
    """Configuration specific to the Trajectory Optimization solver."""

    #: Solver infrastructure configuration (rollouts, optimizers, collision).
    core_cfg: SolverCoreCfg
    #: Robot kinematic and collision sphere configuration.
    robot_config: RobotCfg

    # Behavioral params
    #: Maximum number of trajectory optimization problems solved in one
    #: batch call.
    max_batch_size: int = 1
    #: When True, each batched problem uses its own collision environment.
    multi_env: bool = False
    #: Maximum number of goal poses per problem (goalset mode).
    max_goalset: int = 1
    #: Number of trajectory seeds evaluated per problem.
    num_seeds: int = 4
    #: Position error threshold (meters) for the final end-effector pose.
    position_tolerance: float = 0.005
    #: Orientation error threshold (radians) for the final end-effector
    #: pose.
    orientation_tolerance: float = 0.05
    #: Distance threshold at which the collision cost activates during
    #: optimization.
    optimizer_collision_activation_distance: float = 0.01
    #: Weight scale factor applied to the tool-pose cost at non-terminal
    #: time steps. 0.0 means only the final time step incurs pose cost.
    non_terminal_tool_pose_weight_factor: float = 0.0
    #: Whether to enable self-collision checking during optimization.
    self_collision_check: bool = True

    # TrajOpt-specific
    #: Lower bound on the trajectory time step (seconds). The solver will
    #: not compress the trajectory below this dt.
    minimum_trajectory_dt: float = 0.002
    #: Upper bound on the trajectory time step (seconds).
    maximum_trajectory_dt: float = 0.2
    #: Time step (seconds) used when interpolating the optimized knot
    #: trajectory into a dense waypoint trajectory.
    interpolation_dt: float = 0.025
    #: Interpolation method used to convert optimizer knots into a dense
    #: trajectory. B-spline knot interpolation via CUDA is the default.
    interpolation_type: TrajInterpolationType = TrajInterpolationType.BSPLINE_KNOTS_CUDA
    #: Maximum number of waypoints in the interpolation output buffer.
    interpolation_buffer_size: int = 5000

    # Convenience accessors into core_cfg
    @property
    def device_cfg(self) -> DeviceCfg:
        return self.core_cfg.device_cfg

    @property
    def use_cuda_graph(self) -> bool:
        return self.core_cfg.use_cuda_graph

    @property
    def random_seed(self) -> int:
        return self.core_cfg.random_seed

    @property
    def store_debug(self) -> bool:
        return self.core_cfg.store_debug

    @property
    def scene_collision_cfg(self):
        return self.core_cfg.scene_collision_cfg

    @property
    def optimizer_configs(self):
        return self.core_cfg.optimizer_configs

    @property
    def optimizer_rollout_configs(self):
        return self.core_cfg.optimizer_rollout_configs

    @property
    def metrics_rollout_config(self):
        return self.core_cfg.metrics_rollout_config

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        optimizer_configs: List[Union[str, Dict[str, Any]]] = [
            "trajopt/lbfgs_bspline_trajopt.yml",
        ],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_seeds: int = 4,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        minimum_trajectory_dt: float = 0.002,
        maximum_trajectory_dt: float = 0.2,
        load_collision_spheres: bool = True,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {
            "lbfgs": None,
        },
        transition_model_config_instance_type: Type[
            RobotStateTransitionCfg
        ] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
    ) -> TrajOptSolverCfg:
        """Create TrajOptSolverCfg from flexible inputs.

        Args:
            robot: Robot configuration - path to YAML, dict, or RobotCfg object.
            optimizer_configs: List of optimizer configs - paths or dicts.
            metrics_rollout: Metrics rollout config - path or dict.
            transition_model: Transition model config - path or dict.
            scene_model: Optional scene model config - path or dict.
            collision_cache: Cache configuration for collision checking.
            self_collision_check: Whether to check self-collision.
            device_cfg: Device configuration.
            num_seeds: Number of optimization seeds.
            position_tolerance: Position tolerance for success.
            orientation_tolerance: Orientation tolerance for success.
            use_cuda_graph: Whether to use CUDA graphs.
            random_seed: Random seed for reproducibility.
            optimizer_collision_activation_distance: Collision activation distance.
            store_debug: Whether to store debug information.
            minimum_trajectory_dt: Minimum trajectory time step.
            maximum_trajectory_dt: Maximum trajectory time step.
            load_collision_spheres: When False, skip loading collision spheres.
            override_optimizer_num_iters: Override iteration counts per optimizer.
            transition_model_config_instance_type: Transition model config class.
            cost_manager_config_instance_type: Cost manager config class.
            max_batch_size: Maximum batch size.
            multi_env: Whether to use multi-env collision.
            max_goalset: Maximum goalset size.

        Returns:
            Configured TrajOptSolverCfg instance.
        """
        num_envs = max_batch_size if multi_env else 1

        # 1. Resolve YAML paths
        robot_config, optimizer_dicts, metrics_rollout_dict, transition_model_dict, scene_model_dict = (
            resolve_yaml_configs(
                robot,
                optimizer_configs,
                metrics_rollout,
                transition_model,
                scene_model,
                device_cfg,
                load_collision_spheres=load_collision_spheres,
                num_envs=num_envs,
            )
        )

        # 2. Determine interpolation type from transition model
        if transition_model_dict["transition_model_cfg"]["control_space"] == "POSITION":
            interpolation_type = TrajInterpolationType.LINEAR_CUDA
        else:
            interpolation_type = TrajInterpolationType.BSPLINE_KNOTS_CUDA

        # 3. Build SolverCoreCfg
        core_cfg = create_solver_core_cfg(
            robot_config=robot_config,
            optimizer_dicts=optimizer_dicts,
            metrics_rollout_dict=metrics_rollout_dict,
            transition_model_dict=transition_model_dict,
            scene_model_dict=scene_model_dict,
            device_cfg=device_cfg,
            collision_cache=collision_cache,
            self_collision_check=self_collision_check,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            use_cuda_graph=use_cuda_graph,
            random_seed=random_seed,
            store_debug=store_debug,
            override_optimizer_num_iters=override_optimizer_num_iters,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )

        return TrajOptSolverCfg(
            core_cfg=core_cfg,
            robot_config=robot_config,
            max_batch_size=max_batch_size,
            multi_env=multi_env,
            max_goalset=max_goalset,
            num_seeds=num_seeds,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            self_collision_check=self_collision_check,
            minimum_trajectory_dt=minimum_trajectory_dt,
            maximum_trajectory_dt=maximum_trajectory_dt,
            interpolation_type=interpolation_type,
        )
