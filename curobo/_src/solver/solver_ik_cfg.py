# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclass for the Inverse Kinematics solver.

Bundles SolverCoreCfg with IK-specific parameters such as tolerances, seed counts,
velocity-aware IK settings, and the factory method :meth:`IKSolverCfg.create` that
builds the full config from YAML paths or dictionaries.
"""
from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
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
class IKSolverCfg:
    """Configuration specific to the Inverse Kinematics solver."""

    #: Solver infrastructure configuration (rollouts, optimizers, collision).
    core_cfg: SolverCoreCfg
    #: Robot kinematic and collision sphere configuration.
    robot_config: RobotCfg

    # Behavioral params (used at solve time, not construction)
    #: Maximum number of IK problems solved in a single batch call.
    max_batch_size: int = 1
    #: When True, each batched problem uses its own collision environment.
    multi_env: bool = False
    #: Maximum number of goal poses per problem (goalset mode).
    max_goalset: int = 1
    #: Number of random initial seeds evaluated per IK problem.
    num_seeds: int = 32
    #: Position error threshold (meters) below which a solution is
    #: considered successful.
    position_tolerance: float = 0.005
    #: Orientation error threshold (radians) below which a solution is
    #: considered successful.
    orientation_tolerance: float = 0.05
    #: Distance threshold at which the collision cost activates during
    #: optimization. Smaller values allow closer approaches before penalty.
    optimizer_collision_activation_distance: float = 0.01
    #: Weight scale factor applied to the tool-pose cost at non-terminal
    #: time steps. 0.0 means only the final time step incurs pose cost.
    non_terminal_tool_pose_weight_factor: float = 0.0
    #: When True (default), success requires both constraint feasibility
    #: and pose convergence. When False, a solution is marked successful
    #: if constraints (collision, joint limits) are satisfied, even if
    #: pose error exceeds the tolerance.
    success_requires_convergence: bool = True

    # IK-specific fields
    #: Override the optimizer iteration count when solving multi-link IK
    #: (more than one target link). None uses the default iteration count.
    override_iters_for_multi_link_ik: Optional[int] = None
    #: When True, generates Levenberg-Marquardt seeds in addition to
    #: random seeds.
    use_lm_seed: bool = True
    #: When True, the solver returns as soon as a feasible solution is
    #: found rather than running all optimizer stages.
    exit_early: bool = True
    #: Fraction of the batch that must succeed before the solver exits
    #: early. 1.0 means every problem in the batch must succeed.
    exit_early_batch_success_threshold: float = 1.0
    #: Time step for velocity-aware IK. When set, position bounds are
    #: tightened using ``velocity_limits * dt`` relative to
    #: ``current_state``. None disables velocity clamping (standard
    #: global IK). Overridden by ``current_state.dt`` when that is not
    #: None.
    optimization_dt: Optional[float] = None
    #: Position error weight for the seed Levenberg-Marquardt solver.
    seed_position_weight: float = 1.0
    #: Orientation error weight for the seed Levenberg-Marquardt solver.
    seed_orientation_weight: float = 1.0
    #: Velocity regularization weight for the seed LM solver.
    seed_velocity_weight: float = 0.0
    #: Acceleration regularization weight for the seed LM solver.
    seed_acceleration_weight: float = 0.0
    #: Number of seeds used by the seed Levenberg-Marquardt solver.
    seed_solver_num_seeds: int = 32
    #: Whether to enable self-collision checking during optimization.
    self_collision_check: bool = True

    # Convenience accessors into core_cfg (for code that reads self.config.X)
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
            "ik/particle_ik.yml",
            "ik/lbfgs_ik.yml",
        ],
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_seeds: int = 32,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        override_optimizer_num_iters: Dict[str, Optional[int]] = {
            "particle": None,
            "lbfgs": None,
        },
        transition_model_config_instance_type: Type[
            RobotStateTransitionCfg
        ] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        override_iters_for_multi_link_ik: Optional[int] = None,
        optimization_dt: Optional[float] = None,
        load_collision_spheres: bool = True,
        velocity_regularization_weight: Optional[float] = None,
        acceleration_regularization_weight: Optional[float] = None,
        success_requires_convergence: bool = True,
        seed_position_weight: float = 1.0,
        seed_orientation_weight: float = 1.0,
        seed_velocity_weight: float = 0.0,
        seed_acceleration_weight: float = 0.0,
        seed_solver_num_seeds: int = 32,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
    ) -> IKSolverCfg:
        """Create IKSolverCfg from flexible inputs.

        Each config parameter accepts either a file path (str), a dictionary,
        or an already-constructed config object.

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
            override_optimizer_num_iters: Override iteration counts per optimizer.
            transition_model_config_instance_type: Transition model config class.
            cost_manager_config_instance_type: Cost manager config class.
            override_iters_for_multi_link_ik: Override iterations for multi-link IK.
            optimization_dt: Optional time step for velocity-aware IK; None
                disables velocity clamping (standard global IK).
            load_collision_spheres: When False, skip loading collision spheres
                from the robot config. Saves memory and compute when collision
                checking is not needed.
            velocity_regularization_weight: Optional weight for velocity
                regularization in the rollout; None uses the config default.
            acceleration_regularization_weight: Optional weight for acceleration
                regularization in the rollout; None uses the config default.
            success_requires_convergence: When True (default), success requires
                both constraint feasibility and pose convergence.
            seed_position_weight: Position error weight for the seed LM solver.
            seed_orientation_weight: Orientation error weight for the seed LM solver.
            seed_velocity_weight: Velocity regularization weight for the seed LM solver.
            seed_acceleration_weight: Acceleration regularization weight for the seed LM solver.
            seed_solver_num_seeds: Number of seeds for the seed LM solver.
            max_batch_size: Maximum number of problems solved in parallel.
            multi_env: When True, each batched problem uses its own collision environment.
            max_goalset: Maximum goalset size per problem.

        Returns:
            Configured IKSolverCfg instance.
        """
        num_envs = max_batch_size if multi_env else 1

        # 1. Resolve YAML paths to dicts/configs
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

        # 2. IK-specific: override cspace regularization weights before building rollouts
        if velocity_regularization_weight is not None or acceleration_regularization_weight is not None:
            for opt_dict in optimizer_dicts:
                cspace_cfg = opt_dict.get("rollout", {}).get("cost_cfg", {}).get("cspace_cfg", {})
                reg_w = cspace_cfg.get("squared_l2_regularization_weight", [0.0, 0.0])
                if velocity_regularization_weight is not None:
                    reg_w[0] = velocity_regularization_weight
                if acceleration_regularization_weight is not None:
                    reg_w[1] = acceleration_regularization_weight
                cspace_cfg["squared_l2_regularization_weight"] = reg_w

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

        return IKSolverCfg(
            core_cfg=core_cfg,
            robot_config=robot_config,
            max_batch_size=max_batch_size,
            multi_env=multi_env,
            max_goalset=max_goalset,
            num_seeds=num_seeds,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            success_requires_convergence=success_requires_convergence,
            override_iters_for_multi_link_ik=override_iters_for_multi_link_ik,
            optimization_dt=optimization_dt,
            seed_position_weight=seed_position_weight,
            seed_orientation_weight=seed_orientation_weight,
            seed_velocity_weight=seed_velocity_weight,
            seed_acceleration_weight=seed_acceleration_weight,
            seed_solver_num_seeds=seed_solver_num_seeds,
            self_collision_check=self_collision_check,
        )
