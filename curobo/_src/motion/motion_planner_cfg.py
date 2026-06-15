# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

# CuRobo
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
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
        """Create a MotionPlannerCfg from robot, task, and scene configs.

        This factory builds the three components used by MotionPlanner: IK,
        trajectory optimization, and PRM graph planning. Most integration code
        should pass a robot, optional scene/collision cache, sizing bounds, and
        leave the task YAMLs at their defaults.

        Config path arguments may be relative YAML names under cuRobo's content
        folders, absolute YAML paths, parsed dictionaries, or typed config
        objects where supported.

        Args:
            robot: Robot config. String paths are resolved relative to cuRobo
                robot configs unless absolute. Dicts and RobotCfg objects are
                accepted.
            ik_optimizer_configs: IK optimizer task configs. Defaults to
                ``["ik/lbfgs_ik.yml"]``. Add ``"ik/particle_ik.yml"`` when
                broader IK search is needed.
            ik_transition_model: IK transition model task config.
            metrics_rollout: Metrics rollout task config shared by IK,
                TrajOpt, and graph planner validation.
            trajopt_optimizer_configs: Trajectory optimization task configs.
            trajopt_transition_model: Trajectory optimization transition model
                config.
            graph_planner_config: PRM graph planner config. The default
                enables graph seeding for single-environment planning. This
                factory currently expects a graph planner config.
            graph_planner_rollout: Rollout config used by the graph planner
                for collision/feasibility checks.
            graph_planner_transition_model: Transition model used by graph
                planner rollout validation.
            scene_model: Optional scene config. String paths are resolved
                relative to cuRobo scene configs. When None, no world
                obstacles are loaded initially.
            collision_cache: Optional obstacle cache sizes, for example
                ``{"mesh": 2, "cuboid": 8}``. When ``scene_model`` is None,
                this still preallocates a scene collision checker for later
                ``update_world`` calls.
            self_collision_check: Enable robot self-collision costs/checks in
                IK, TrajOpt, and graph planner validation.
            device_cfg: Tensor device and dtype configuration.
            num_ik_seeds: Number of IK seeds evaluated per problem. A value
                of 16 is good for most single-tool planning problems; increase
                to 32 when more IK diversity is needed. Increasing beyond 32
                usually does not help unless solving multi-tool IK/planning
                problems.
            num_trajopt_seeds: Number of trajectory optimization seeds.
                MotionPlanner asks IK to return this many solutions to seed
                TrajOpt. The default of 4 is recommended; using more than 4
                typically does not improve solution quality and can
                significantly increase planning time.
            position_tolerance: Cartesian position tolerance in meters used
                for IK and TrajOpt success.
            orientation_tolerance: Cartesian orientation tolerance in radians
                used for IK and TrajOpt success.
            use_cuda_graph: Enable CUDA graph capture for solver rollouts
                and optimizers. Defaults to True and should be left enabled
                for normal integration/runtime code. The value is forwarded
                to IK, TrajOpt, and graph-planner rollout components. cuRobo
                manages graph caches internally and pads smaller batches/
                goalsets up to ``max_batch_size`` and ``max_goalset``. Set to
                False only for debugging graph-capture issues; steady-state
                solve calls can be about 5x slower without CUDA graph replay.
                Also disabled automatically when ``store_debug=True``.
            random_seed: Random seed forwarded to IK and TrajOpt seed
                generation. Graph planner path finding uses its own seed in
                PRMGraphPlannerCfg.
            optimizer_collision_activation_distance: Collision-cost activation
                distance forwarded to IK and TrajOpt optimizer rollouts. The
                default is 0.01 m (10 mm). Increasing this distance makes the
                optimizer react to obstacles earlier and keep more clearance,
                but can make narrow passages harder or infeasible. Decreasing
                it allows closer motion near obstacles, but can reduce
                clearance margin and make collision avoidance less robust.
            store_debug: Whether to store debug information. When True,
                CUDA graphs are disabled automatically.
            transition_model_config_instance_type: Advanced extension hook for
                custom transition model config classes.
            cost_manager_config_instance_type: Advanced extension hook for
                custom cost manager config classes.
            max_batch_size: Maximum batch size captured/allocated by IK and
                TrajOpt. Single MotionPlanner calls use one problem;
                BatchMotionPlanner uses up to this value. Smaller batches are
                padded internally.
            multi_env: Use one collision environment per batch item. This is
                for BatchMotionPlanner with per-problem worlds; graph seeding
                is skipped in this mode.
            max_goalset: Maximum number of alternative goal poses per problem.
                Smaller goalsets are padded internally.

        Returns:
            MotionPlannerCfg containing IK, TrajOpt, graph planner, and
            optional scene collision configuration.
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
