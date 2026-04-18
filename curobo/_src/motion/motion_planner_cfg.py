# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

# CuRobo
from curobo._src.geom.collision import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlannerCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config
from curobo._src.util.logging import log_and_raise
from curobo.content import get_robot_configs_path, get_scene_configs_path


@dataclass
class MotionPlannerCfg:
    """Configuration for the motion planner."""

    ik_solver_config: IKSolverCfg
    trajopt_solver_config: TrajOptSolverCfg
    graph_planner_config: PRMGraphPlannerCfg = None
    scene_collision_cfg: Optional[SceneCollisionCfg] = None
    device_cfg: DeviceCfg = DeviceCfg()

    @staticmethod
    def create(
        robot: Union[str, Dict[str, Any], RobotCfg],
        ik_optimizer_configs: List[Union[str, Dict[str, Any]]] = [
            "ik/particle_ik.yml",
            "ik/lbfgs_ik.yml",
        ][1:],
        ik_transition_model: Union[str, Dict[str, Any]] = "ik/transition_ik.yml",
        metrics_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        trajopt_optimizer_configs: List[Union[str, Dict[str, Any]]] = [
            "trajopt/lbfgs_bspline_trajopt.yml"
        ],
        trajopt_transition_model: Union[str, Dict[str, Any]] = "trajopt/transition_bspline_trajopt.yml",
        graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml",
        graph_planner_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        graph_planner_transition_model: Union[str, Dict[str, Any]] = "graph_planner/transition_graph_planner.yml",
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_ik_seeds: int = 32,
        num_trajopt_seeds: int = 4,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.01,
        store_debug: bool = False,
        transition_model_config_instance_type: Type[
            RobotStateTransitionCfg
        ] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
    ) -> MotionPlannerCfg:
        """Create MotionPlannerCfg from flexible inputs.

        Each config parameter accepts either a file path (str), a dictionary,
        or an already-constructed config object.

        Args:
            robot: Robot configuration - path to YAML, dict, or RobotCfg object.
            ik_optimizer_configs: IK optimizer configs - paths or dicts.
            ik_transition_model: IK transition model config - path or dict.
            metrics_rollout: Metrics rollout config - path or dict.
            trajopt_optimizer_configs: Trajectory optimization optimizer configs.
            trajopt_transition_model: Trajectory optimization transition model.
            graph_planner_config: Graph planner config - path or dict.
            graph_planner_rollout: Graph planner rollout config - path or dict.
            graph_planner_transition_model: Graph planner transition model.
            scene_model: Optional scene model config - path or dict.
            collision_cache: Cache configuration for collision checking.
            self_collision_check: Whether to check self-collision.
            device_cfg: Device configuration.
            num_ik_seeds: Number of IK optimization seeds.
            num_trajopt_seeds: Number of trajectory optimization seeds.
            position_tolerance: Position tolerance for success.
            orientation_tolerance: Orientation tolerance for success.
            use_cuda_graph: Whether to use CUDA graphs.
            random_seed: Random seed for reproducibility.
            optimizer_collision_activation_distance: Collision activation distance.
            store_debug: Whether to store debug information.
            transition_model_config_instance_type: Transition model config class.
            cost_manager_config_instance_type: Cost manager config class.
            max_batch_size: Maximum number of problems solved in parallel; fewer
                may be provided (padded internally).
            multi_env: When True, each batched problem uses its own collision
                environment; when False, all share one environment.
            max_goalset: Maximum goalset size per problem; fewer goals may be
                provided (padded internally).

        Returns:
            Configured MotionPlannerCfg instance.
        """
        num_envs = max_batch_size if multi_env else 1

        # Resolve robot config
        robot_resolved = resolve_config(join_path(get_robot_configs_path(), robot))
        if isinstance(robot_resolved, dict):
            robot_config = RobotCfg.create(robot_resolved, device_cfg, num_envs=num_envs)
        elif isinstance(robot_resolved, RobotCfg):
            robot_config = robot_resolved
        else:
            log_and_raise("robot must be a path, dict, or RobotCfg object")

        # Resolve scene model
        scene_collision_cfg = None
        if scene_model is not None:
            scene_model_dict = resolve_config(join_path(get_scene_configs_path(), scene_model))
            # Parse scene config
            if isinstance(scene_model_dict, list):
                scene_cfg = [SceneCfg.create(x) for x in scene_model_dict]
            elif isinstance(scene_model_dict, dict):
                scene_cfg = SceneCfg.create(scene_model_dict)
            else:
                scene_cfg = scene_model_dict

            scene_collision_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_cfg,
                cache=collision_cache,
            )
        elif collision_cache is not None:
            # Create scene collision config with cache but no initial scene model
            # This allows voxel/mesh caches to be pre-allocated for later updates
            scene_collision_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=None,
                cache=collision_cache,
            )

        ik_solver_config = IKSolverCfg.create(
            robot=robot_config,
            optimizer_configs=ik_optimizer_configs,
            transition_model=ik_transition_model,
            metrics_rollout=metrics_rollout,
            device_cfg=device_cfg,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            use_cuda_graph=use_cuda_graph,
            random_seed=random_seed,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            store_debug=store_debug,
            num_seeds=num_ik_seeds,
            self_collision_check=self_collision_check,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
            max_batch_size=max_batch_size,
            multi_env=multi_env,
            max_goalset=max_goalset,
        )

        trajopt_solver_config = TrajOptSolverCfg.create(
            robot=robot_config,
            optimizer_configs=trajopt_optimizer_configs,
            transition_model=trajopt_transition_model,
            metrics_rollout=metrics_rollout,
            device_cfg=device_cfg,
            num_seeds=num_trajopt_seeds,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            use_cuda_graph=use_cuda_graph,
            random_seed=random_seed,
            optimizer_collision_activation_distance=optimizer_collision_activation_distance,
            store_debug=store_debug,
            self_collision_check=self_collision_check,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
            max_batch_size=max_batch_size,
            multi_env=multi_env,
            max_goalset=max_goalset,
        )

        graph_planner_cfg = PRMGraphPlannerCfg.create(
            robot=robot_config,
            graph_planner_config=graph_planner_config,
            rollout=graph_planner_rollout,
            transition_model=graph_planner_transition_model,
            self_collision_check=self_collision_check,
            device_cfg=device_cfg,
            use_cuda_graph_for_rollout=use_cuda_graph,
        )

        return MotionPlannerCfg(
            ik_solver_config=ik_solver_config,
            trajopt_solver_config=trajopt_solver_config,
            graph_planner_config=graph_planner_cfg,
            scene_collision_cfg=scene_collision_cfg,
            device_cfg=device_cfg,
        )
