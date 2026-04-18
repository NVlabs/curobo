# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclass for the Model Predictive Control solver.

Bundles SolverCoreCfg with MPC-specific parameters such as cold/warm-start
iteration counts, interpolation steps, optimization_dt, and the factory method
:meth:`MPCSolverCfg.create` that builds the full config from YAML paths or
dictionaries.
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


@dataclass
class MPCSolverCfg:
    """Configuration specific to the MPC solver."""

    #: Solver infrastructure configuration (rollouts, optimizers, collision).
    core_cfg: SolverCoreCfg
    #: Robot kinematic and collision sphere configuration.
    robot_config: RobotCfg

    # Behavioral params
    #: Maximum number of MPC problems solved in a single batch call.
    max_batch_size: int = 1
    #: When True, each batched problem uses its own collision environment.
    multi_env: bool = False
    #: Maximum number of goal poses per problem (goalset mode).
    max_goalset: int = 1
    #: Number of trajectory seeds evaluated per MPC step.
    num_seeds: int = 1
    #: Position error threshold (meters) for the end-effector pose.
    position_tolerance: float = 0.005
    #: Orientation error threshold (radians) for the end-effector pose.
    orientation_tolerance: float = 0.05
    #: Distance threshold at which the collision cost activates during
    #: optimization.
    optimizer_collision_activation_distance: float = 0.01
    #: Weight scale factor applied to the tool-pose cost at non-terminal
    #: time steps. Small positive values help MPC track the path, unlike
    #: IK/TrajOpt where 0.0 is typical.
    non_terminal_tool_pose_weight_factor: float = 0.001
    #: Whether to enable self-collision checking during optimization.
    self_collision_check: bool = True

    # MPC-specific
    #: Number of interpolation sub-steps between each optimizer knot.
    #: Determines the command rate: ``command_dt = optimization_dt /
    #: interpolation_steps``.
    interpolation_steps: int = 4
    #: Time step (seconds) between optimizer knot points. Together with
    #: :attr:`interpolation_steps`, determines the overall planning
    #: horizon and command rate.
    optimization_dt: float = 0.02
    #: Number of optimizer iterations when the MPC loop is warm-started
    #: (i.e. after the first solve). Fewer iterations are needed since
    #: the optimizer resumes from the previous solution.
    warm_start_optimization_num_iters: int = 200
    #: Number of optimizer iterations for the first (cold-start) MPC
    #: solve, where no prior solution is available.
    cold_start_optimization_num_iters: int = 300
    #: When True, the MPC controller outputs a deceleration trajectory
    #: when the optimizer fails to find a feasible solution.
    use_deceleration_on_failure: bool = True
    #: Duration (seconds) of the deceleration trajectory. When None, the
    #: duration is computed automatically from the current velocity.
    deceleration_time: Optional[float] = None
    #: Velocity profile used for the deceleration trajectory. Currently
    #: supports ``"exponential"``.
    deceleration_profile: str = "exponential"
    #: Hard upper limit (seconds) on the deceleration duration. Prevents
    #: extremely slow stopping motions.
    max_deceleration_time: float = 2.0

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
            "mpc/lbfgs_mpc.yml",
        ],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "mpc/transition_bspline_mpc.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {
            "lbfgs": None,
        },
        transition_model_config_instance_type: Type[
            RobotStateTransitionCfg
        ] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        optimization_dt: float = 0.02,
        interpolation_steps: int = 4,
        use_deceleration_on_failure: bool = True,
        deceleration_time: Optional[float] = None,
        deceleration_profile: str = "exponential",
        max_deceleration_time: float = 2.0,
        load_collision_spheres: bool = True,
        num_control_points: Optional[int] = None,
        squared_l2_regularization_weight: Optional[List[float]] = None,
        warm_start_optimization_num_iters: int = 200,
        cold_start_optimization_num_iters: int = 300,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        **kwargs,
    ) -> MPCSolverCfg:
        """Create MPCSolverCfg from flexible inputs.

        Args:
            robot: Robot configuration - path to YAML, dict, or RobotCfg object.
            optimizer_configs: List of optimizer configs - paths or dicts.
            metrics_rollout: Metrics rollout config - path or dict.
            transition_model: Transition model config - path or dict.
            scene_model: Optional scene model config - path or dict.
            collision_cache: Cache configuration for collision checking.
            self_collision_check: Whether to check self-collision.
            device_cfg: Device configuration.
            position_tolerance: Position tolerance for success.
            orientation_tolerance: Orientation tolerance for success.
            use_cuda_graph: Whether to use CUDA graphs.
            random_seed: Random seed for reproducibility.
            optimizer_collision_activation_distance: Collision activation distance.
            store_debug: Whether to store debug information.
            override_optimizer_num_iters: Override iteration counts per optimizer.
            transition_model_config_instance_type: Transition model config class.
            cost_manager_config_instance_type: Cost manager config class.
            optimization_dt: Optimization time step.
            interpolation_steps: Number of interpolation steps.
            use_deceleration_on_failure: Whether to use deceleration on failure.
            deceleration_time: Deceleration time (auto-calculated if None).
            deceleration_profile: Deceleration profile type.
            max_deceleration_time: Maximum deceleration time.
            load_collision_spheres: When False, skip loading collision spheres.
            num_control_points: Number of B-spline control points.
            squared_l2_regularization_weight: Per-control-point squared L2 regularization.
            warm_start_optimization_num_iters: Optimizer iterations for warm-started steps.
            cold_start_optimization_num_iters: Optimizer iterations for the first step.
            max_batch_size: Maximum batch size.
            multi_env: Whether to use multi-env collision.
            max_goalset: Maximum goalset size.
            **kwargs: Additional keyword arguments.

        Returns:
            Configured MPCSolverCfg instance.
        """
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
            )
        )

        # 2. MPC-specific: update transition model with dt/interpolation settings
        transition_model_dict["transition_model_cfg"]["dt_traj_params"]["base_dt"] = optimization_dt
        transition_model_dict["transition_model_cfg"]["dt_traj_params"]["max_dt"] = optimization_dt
        transition_model_dict["transition_model_cfg"]["dt_traj_params"]["base_ratio"] = 1.0
        transition_model_dict["transition_model_cfg"]["interpolation_steps"] = interpolation_steps
        if num_control_points is not None:
            transition_model_dict["transition_model_cfg"]["n_knots"] = num_control_points

        # MPC-specific: override cspace regularization weights
        if squared_l2_regularization_weight is not None:
            for opt_dict in optimizer_dicts:
                cspace_cfg = opt_dict.get("rollout", {}).get("cost_cfg", {}).get("cspace_cfg", {})
                cspace_cfg["squared_l2_regularization_weight"] = squared_l2_regularization_weight

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

        return MPCSolverCfg(
            core_cfg=core_cfg,
            robot_config=robot_config,
            max_batch_size=max_batch_size,
            multi_env=multi_env,
            max_goalset=max_goalset,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            self_collision_check=self_collision_check,
            optimization_dt=optimization_dt,
            interpolation_steps=interpolation_steps,
            use_deceleration_on_failure=use_deceleration_on_failure,
            deceleration_time=deceleration_time,
            deceleration_profile=deceleration_profile,
            max_deceleration_time=max_deceleration_time,
            warm_start_optimization_num_iters=warm_start_optimization_num_iters,
            cold_start_optimization_num_iters=cold_start_optimization_num_iters,
            **kwargs,
        )
