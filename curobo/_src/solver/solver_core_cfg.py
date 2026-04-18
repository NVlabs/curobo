# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SolverCoreCfg and factory functions for building solver infrastructure.

SolverCoreCfg holds the configuration needed by SolverCore to construct rollouts,
optimizers, collision checker, and seed manager. Each solver (IK, TrajOpt, MPC) nests
a SolverCoreCfg inside its own config dataclass.

The factory functions consolidate duplicated YAML resolution and rollout/optimizer
config creation that was previously spread across IKSolverCfg.create(),
TrajOptSolverCfg.create(), and MPCSolverCfg.create().
"""
from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# CuRobo
from curobo._src.geom.collision import SceneCollisionCfg
from curobo._src.optim.optim_factory import create_optimization_config
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import join_path, resolve_config
from curobo._src.util.logging import log_and_raise, log_warn
from curobo.content import (
    get_robot_configs_path,
    get_scene_configs_path,
    get_task_configs_path,
)


@dataclass
class SolverCoreCfg:
    """Configuration consumed by SolverCore to build rollouts, optimizers, and collision checker."""

    #: Configuration for the metrics evaluation rollout.
    metrics_rollout_config: RobotRolloutCfg
    #: Configurations for optimizer rollouts (one per optimizer stage).
    optimizer_rollout_configs: List[RobotRolloutCfg]
    #: Flat optimizer configs (LBFGSOptCfg, MPPICfg, etc.).
    optimizer_configs: List[Any]
    #: Optional scene collision configuration.
    scene_collision_cfg: Optional[SceneCollisionCfg] = None
    #: Tensor device configuration.
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    #: Whether to use CUDA graphs for metrics rollout and optimizers.
    use_cuda_graph: bool = True
    #: Random seed for reproducibility.
    random_seed: int = 123
    #: Whether to store debug information (disables CUDA graphs).
    store_debug: bool = False

    def __post_init__(self):
        if len(self.optimizer_configs) == 0:
            log_and_raise("optimizer_configs is empty")
        if self.store_debug and self.use_cuda_graph:
            log_warn(
                "store_debug is True, but use_cuda_graph is also True. Disabling use_cuda_graph."
            )
            self.use_cuda_graph = False


# ---------------------------------------------------------------------------
# Helper functions for building rollout and collision configs
# ---------------------------------------------------------------------------


def create_rollout_configs(
    optimization_dicts: List[Dict],
    transition_model_dict: Dict,
    robot_config: RobotCfg,
    device_cfg: DeviceCfg,
    optimizer_collision_activation_distance: Optional[float],
    transition_model_config_instance_type: Type[
        RobotStateTransitionCfg
    ] = RobotStateTransitionCfg,
    cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
    self_collision_check: bool = True,
) -> List[RobotRolloutCfg]:
    """Build optimizer rollout configs from resolved YAML dicts.

    Args:
        optimization_dicts: Resolved optimizer YAML dicts (each containing a ``"rollout"`` key).
        transition_model_dict: Resolved transition model YAML dict.
        robot_config: Robot configuration.
        device_cfg: Device configuration.
        optimizer_collision_activation_distance: Collision activation distance for optimizer
            rollouts. None leaves the default from the YAML.
        transition_model_config_instance_type: Transition model config class.
        cost_manager_config_instance_type: Cost manager config class.
        self_collision_check: Whether to enable self-collision checking.

    Returns:
        List of RobotRolloutCfg, one per optimizer stage.
    """
    optimizer_rollout_configs = []
    for optimization_dict in optimization_dicts:
        rollout_dict = optimization_dict["rollout"]
        rollout_dict["transition_model_cfg"] = transition_model_dict["transition_model_cfg"]
        rollout_config = RobotRolloutCfg.create_with_component_types(
            rollout_dict,
            robot_config,
            device_cfg,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )
        for cost_cfg in rollout_config.get_cost_manager_configs():
            if optimizer_collision_activation_distance is not None:
                cost_cfg.update_collision_activation_distance(
                    optimizer_collision_activation_distance
                )
            if not self_collision_check:
                cost_cfg.disable_self_collision()

        optimizer_rollout_configs.append(rollout_config)
    return optimizer_rollout_configs


def create_metrics_rollout_config(
    metrics_rollout_dict: Dict,
    transition_model_dict: Dict,
    robot_config: RobotCfg,
    device_cfg: DeviceCfg,
    transition_model_config_instance_type: Type[
        RobotStateTransitionCfg
    ] = RobotStateTransitionCfg,
    cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
) -> RobotRolloutCfg:
    """Build the metrics rollout config from resolved YAML dicts.

    Args:
        metrics_rollout_dict: Resolved metrics rollout YAML dict (containing a ``"rollout"`` key).
        transition_model_dict: Resolved transition model YAML dict.
        robot_config: Robot configuration.
        device_cfg: Device configuration.
        transition_model_config_instance_type: Transition model config class.
        cost_manager_config_instance_type: Cost manager config class.

    Returns:
        RobotRolloutCfg for metrics evaluation.
    """
    metrics_rollout_dict["rollout"]["transition_model_cfg"] = transition_model_dict[
        "transition_model_cfg"
    ]
    metrics_rollout_config = RobotRolloutCfg.create_with_component_types(
        metrics_rollout_dict["rollout"],
        robot_config,
        device_cfg,
        transition_model_config_instance_type=transition_model_config_instance_type,
        cost_manager_config_instance_type=cost_manager_config_instance_type,
    )
    return metrics_rollout_config


def create_scene_collision_cfg(
    scene_model_dict: Optional[Dict],
    collision_cache: Optional[Dict[str, int]],
    device_cfg: DeviceCfg,
) -> Optional[SceneCollisionCfg]:
    """Create scene collision configuration from a scene model dictionary.

    Args:
        scene_model_dict: Dictionary containing scene obstacles, or None.
        collision_cache: Cache sizes for obstacle types.
        device_cfg: Device configuration.

    Returns:
        Configured SceneCollisionCfg, or None if no scene model provided.
    """
    if scene_model_dict is not None:
        # CuRobo
        from curobo._src.geom.types import SceneCfg

        if isinstance(scene_model_dict, list):
            scene_model = [SceneCfg.create(x) for x in scene_model_dict]
        elif isinstance(scene_model_dict, dict):
            scene_model = SceneCfg.create(scene_model_dict)
        else:
            scene_model = scene_model_dict

        return SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=scene_model,
            cache=collision_cache,
        )
    return None


# ---------------------------------------------------------------------------
# YAML resolution helper
# ---------------------------------------------------------------------------


def resolve_yaml_configs(
    robot: Union[str, Dict[str, Any], RobotCfg],
    optimizer_configs: List[Union[str, Dict[str, Any]]],
    metrics_rollout: Union[str, Dict[str, Any]],
    transition_model: Union[str, Dict[str, Any]],
    scene_model: Optional[Union[str, Dict[str, Any]]],
    device_cfg: DeviceCfg,
    load_collision_spheres: bool = True,
    num_envs: int = 1,
) -> Tuple[RobotCfg, List[Dict], Dict, Dict, Optional[Dict]]:
    """Resolve YAML file paths to configuration dicts/objects.

    This consolidates the common YAML resolution logic shared by IKSolverCfg.create(),
    TrajOptSolverCfg.create(), and MPCSolverCfg.create().

    After calling this, the solver's create() method can apply solver-specific modifications
    to the returned dicts before passing them to :func:`create_solver_core_cfg`.

    Args:
        robot: Robot config - YAML path (relative to robot configs), dict, or RobotCfg.
        optimizer_configs: Optimizer configs - YAML paths (relative to task configs) or dicts.
        metrics_rollout: Metrics rollout config - YAML path or dict.
        transition_model: Transition model config - YAML path or dict.
        scene_model: Scene model config - YAML path or dict, or None.
        device_cfg: Device configuration.
        load_collision_spheres: Whether to load collision spheres from robot config.
        num_envs: Number of environments (for multi-env collision checking).

    Returns:
        Tuple of (robot_config, optimizer_dicts, metrics_rollout_dict,
        transition_model_dict, scene_model_dict).
    """
    # Resolve robot config
    robot_resolved = resolve_config(join_path(get_robot_configs_path(), robot))
    if isinstance(robot_resolved, dict):
        robot_config = RobotCfg.create(
            robot_resolved,
            device_cfg,
            load_collision_spheres=load_collision_spheres,
            num_envs=num_envs,
        )
    else:
        robot_config = robot_resolved

    # Resolve scene model
    scene_model_dict = None
    if scene_model is not None:
        scene_model_dict = resolve_config(join_path(get_scene_configs_path(), scene_model))

    # Resolve optimizer configs
    optimizer_dicts = [
        resolve_config(join_path(get_task_configs_path(), cfg)) for cfg in optimizer_configs
    ]

    # Resolve metrics rollout and transition model
    metrics_rollout_dict = resolve_config(join_path(get_task_configs_path(), metrics_rollout))
    transition_model_dict = resolve_config(join_path(get_task_configs_path(), transition_model))

    return robot_config, optimizer_dicts, metrics_rollout_dict, transition_model_dict, scene_model_dict


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_solver_core_cfg(
    robot_config: RobotCfg,
    optimizer_dicts: List[Dict],
    metrics_rollout_dict: Dict,
    transition_model_dict: Dict,
    scene_model_dict: Optional[Dict],
    device_cfg: DeviceCfg,
    collision_cache: Optional[Dict[str, int]] = None,
    self_collision_check: bool = True,
    optimizer_collision_activation_distance: float = 0.01,
    use_cuda_graph: bool = True,
    random_seed: int = 123,
    store_debug: bool = False,
    override_optimizer_num_iters: Optional[Dict[str, Optional[int]]] = None,
    transition_model_config_instance_type: Type[
        RobotStateTransitionCfg
    ] = RobotStateTransitionCfg,
    cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
) -> SolverCoreCfg:
    """Build a SolverCoreCfg from resolved YAML dicts.

    Call :func:`resolve_yaml_configs` first to resolve YAML paths, then apply any
    solver-specific modifications to the returned dicts, then call this function.

    Args:
        robot_config: Resolved robot configuration.
        optimizer_dicts: Resolved optimizer YAML dicts.
        metrics_rollout_dict: Resolved metrics rollout YAML dict.
        transition_model_dict: Resolved transition model YAML dict.
        scene_model_dict: Resolved scene model dict, or None.
        device_cfg: Device configuration.
        collision_cache: Cache configuration for collision checking.
        self_collision_check: Whether to check self-collision.
        optimizer_collision_activation_distance: Collision activation distance for
            optimizer rollouts.
        use_cuda_graph: Whether to use CUDA graphs.
        random_seed: Random seed for reproducibility.
        store_debug: Whether to store debug information.
        override_optimizer_num_iters: Override iteration counts per optimizer
            (keyed by solver_name).
        transition_model_config_instance_type: Transition model config class.
        cost_manager_config_instance_type: Cost manager config class.

    Returns:
        Configured SolverCoreCfg.
    """
    # Scene collision config
    scene_collision_cfg = create_scene_collision_cfg(scene_model_dict, collision_cache, device_cfg)

    # Optimizer configs from YAML dicts
    optimizer_cfg_list = [
        create_optimization_config(opt_dict["optimizer"], device_cfg)
        for opt_dict in optimizer_dicts
    ]

    # Apply iteration overrides
    if override_optimizer_num_iters is not None:
        for opt_cfg in optimizer_cfg_list:
            if opt_cfg.solver_name in override_optimizer_num_iters:
                override = override_optimizer_num_iters[opt_cfg.solver_name]
                if override is not None:
                    opt_cfg.update_niters(override)

    # Propagate store_debug to optimizer configs
    for opt_cfg in optimizer_cfg_list:
        opt_cfg.store_debug = store_debug

    # Rollout configs
    optimizer_rollout_configs = create_rollout_configs(
        optimizer_dicts,
        transition_model_dict,
        robot_config,
        device_cfg,
        optimizer_collision_activation_distance,
        transition_model_config_instance_type,
        cost_manager_config_instance_type,
        self_collision_check,
    )

    metrics_rollout_config = create_metrics_rollout_config(
        metrics_rollout_dict,
        transition_model_dict,
        robot_config,
        device_cfg,
        transition_model_config_instance_type,
        cost_manager_config_instance_type,
    )

    return SolverCoreCfg(
        metrics_rollout_config=metrics_rollout_config,
        optimizer_rollout_configs=optimizer_rollout_configs,
        optimizer_configs=optimizer_cfg_list,
        scene_collision_cfg=scene_collision_cfg,
        device_cfg=device_cfg,
        use_cuda_graph=use_cuda_graph,
        random_seed=random_seed,
        store_debug=store_debug,
    )
