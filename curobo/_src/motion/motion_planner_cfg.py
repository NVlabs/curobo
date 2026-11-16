# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration factory for single and batched motion planners."""

from __future__ import annotations

# Standard Library
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

# CuRobo
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.rrt import (
    BiRRTConfig,
    RRTGraphPlannerCfg,
)
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.types.spline_layout import SplineLayout
from curobo._src.util.config_io import join_path, resolve_config
from curobo._src.util.logging import log_and_raise
from curobo.content import get_robot_configs_path, get_scene_configs_path, get_task_configs_path

RRT_GRAPH_PLANNER_ALGORITHMS = ("rrt",)
GRAPH_PLANNER_ALGORITHMS = ("prm", *RRT_GRAPH_PLANNER_ALGORITHMS)


def is_rrt_graph_planner_algorithm(graph_planner_algorithm: str) -> bool:
    """Return whether an algorithm selects the canonical RRT planner."""
    return graph_planner_algorithm in RRT_GRAPH_PLANNER_ALGORITHMS


def _resolve_graph_planner_selection(
    graph_planner_algorithm: str,
) -> str:
    """Validate and return the selected graph-planning algorithm."""
    if graph_planner_algorithm not in GRAPH_PLANNER_ALGORITHMS:
        log_and_raise(f"graph_planner_algorithm must be one of {list(GRAPH_PLANNER_ALGORITHMS)}")
    return graph_planner_algorithm


@dataclass
class MotionPlannerCfg:
    """Configuration for the motion planner."""

    ik_solver_config: IKSolverCfg
    trajopt_solver_config: TrajOptSolverCfg
    prm_graph_planner_config: Optional[PRMGraphPlannerCfg] = None
    scene_collision_cfg: Optional[SceneCollisionCfg] = None
    device_cfg: DeviceCfg = DeviceCfg()
    graph_planner_algorithm: str = "prm"
    rrt_graph_planner_config: Optional[RRTGraphPlannerCfg] = None
    fit_graph_seed_to_bspline: bool = True
    always_run_trajopt: bool = True
    run_trajopt_on_graph_seed: bool = True
    enable_trajopt: bool = True

    @property
    def spline_layout(self) -> SplineLayout:
        """Spline layout shared by planning, optimization, and reference construction."""
        return self.trajopt_solver_config.spline_layout

    @property
    def num_knots(self) -> int:
        """Fixed number of trajectory optimization controls."""
        return self.spline_layout.num_knots

    @property
    def native_horizon(self) -> int:
        """Number of native rollout states before dense interpolation."""
        return self.trajopt_solver_config.native_horizon

    def __post_init__(self) -> None:
        """Normalize planner selection for direct dataclass construction too."""
        if not isinstance(self.always_run_trajopt, bool):
            log_and_raise("always_run_trajopt must be a boolean")
        if not isinstance(self.run_trajopt_on_graph_seed, bool):
            log_and_raise("run_trajopt_on_graph_seed must be a boolean")
        if not isinstance(self.enable_trajopt, bool):
            log_and_raise("enable_trajopt must be a boolean")
        self.graph_planner_algorithm = _resolve_graph_planner_selection(
            self.graph_planner_algorithm
        )
        if self.prm_graph_planner_config is not None and self.rrt_graph_planner_config is not None:
            log_and_raise("only one graph-planner configuration may be populated")
        if self.prm_graph_planner_config is not None and not isinstance(
            self.prm_graph_planner_config, PRMGraphPlannerCfg
        ):
            log_and_raise("prm_graph_planner_config must be a PRMGraphPlannerCfg or None")
        if self.rrt_graph_planner_config is not None and not isinstance(
            self.rrt_graph_planner_config, RRTGraphPlannerCfg
        ):
            log_and_raise("rrt_graph_planner_config must be an RRTGraphPlannerCfg or None")
        if self.graph_planner_algorithm == "prm" and self.rrt_graph_planner_config is not None:
            log_and_raise("graph_planner_algorithm='prm' cannot use an RRT configuration")
        if self.graph_planner_algorithm == "rrt" and self.prm_graph_planner_config is not None:
            log_and_raise("graph_planner_algorithm='rrt' cannot use a PRM configuration")

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
        trajopt_transition_model: Union[str, Dict[str, Any]] = (
            "trajopt/transition_bspline_trajopt.yml"
        ),
        graph_planner_config: Union[str, Dict[str, Any]] = "graph_planner/exact_graph_planner.yml",
        graph_planner_resources_config: Union[str, Dict[str, Any]] = (
            "graph_planner/resources.yml"
        ),
        graph_planner_rollout: Union[str, Dict[str, Any]] = "metrics_base.yml",
        graph_planner_transition_model: Union[str, Dict[str, Any]] = (
            "graph_planner/transition_graph_planner.yml"
        ),
        scene_model: Optional[Union[str, Dict[str, Any]]] = None,
        collision_cache: Optional[Dict[str, int]] = None,
        self_collision_check: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_ik_seeds: Optional[int] = None,
        num_trajopt_seeds: Optional[int] = None,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_cuda_graph: bool = True,
        random_seed: int = 123,
        optimizer_collision_activation_distance: float = 0.0025,
        store_debug: bool = False,
        transition_model_config_instance_type: Type[
            RobotStateTransitionCfg
        ] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
        max_batch_size: int = 1,
        multi_env: bool = False,
        max_goalset: int = 1,
        interpolation_dt: float = 0.025,
        interpolation_buffer_size: Optional[int] = None,
        ik_position_rank_weight: Optional[float] = None,
        ik_orientation_rank_weight: Optional[float] = None,
        ik_cspace_rank_weight: float = 5.0,
        graph_planner_algorithm: str = "prm",
        rrt_planner_config: Optional[Union[BiRRTConfig, Dict[str, Any]]] = None,
        fit_graph_seed_to_bspline: bool = True,
        always_run_trajopt: bool = True,
        run_trajopt_on_graph_seed: bool = True,
        enable_trajopt: bool = True,
        ik_metrics_rollout: Optional[Union[str, Dict[str, Any]]] = None,
        trajopt_metrics_rollout: Optional[Union[str, Dict[str, Any]]] = None,
        enable_timeopt: bool = True,
        num_knots: Optional[int] = None,
        enable_inverse_dynamics: bool = True,
    ) -> MotionPlannerCfg:
        """Create a MotionPlannerCfg from robot, task, and scene configs.

        This factory builds IK, trajectory optimization, the general graph
        planner, and optionally a dedicated constrained RRT. Most integration
        code should pass a robot, optional scene/collision cache, sizing bounds,
        and leave the task YAMLs at their defaults.

        Config path arguments may be relative YAML names under cuRobo's content
        folders, absolute YAML paths, parsed dictionaries, or typed config
        objects where supported.

        Args:
            enable_inverse_dynamics: Compute and constrain joint torque in IK, path search,
                TrajOpt, validation, and timeopt. False skips inverse dynamics everywhere
                while retaining collision and position/velocity/acceleration/jerk checks.
            enable_timeopt: Optimize timing for all optimized or fitted graph seeds before
                selection. Enabled by default. Combine with enable_trajopt=False and
                enable_graph_attempt=0 in the planning call for IK + graph search + timeopt.
            robot: Robot config. String paths are resolved relative to cuRobo
                robot configs unless absolute. Dicts and RobotCfg objects are
                accepted.
            ik_optimizer_configs: IK optimizer task configs. Defaults to
                ``["ik/lbfgs_ik.yml"]``. Add ``"ik/particle_ik.yml"`` when
                broader IK search is needed.
            ik_transition_model: IK transition model task config.
            metrics_rollout: Metrics rollout task config shared by IK,
                and TrajOpt unless a solver-specific override is supplied.
            trajopt_optimizer_configs: Trajectory optimization task configs.
            num_knots: Fixed B-spline control count in [12, 1024]. None preserves
                the transition configuration (16 by default). More knots generally
                increase motion duration and may increase planning time, but can
                reduce waypoint tracking error. Choose the count based on the
                required waypoint tracking tolerance and obstacle density.
            trajopt_transition_model: Trajectory optimization transition model
                config.
            graph_planner_config: PRM graph planner config. The default
                enables PRM graph seeding for single-environment planning.
                It is ignored when ``graph_planner_algorithm="rrt"``.
            graph_planner_resources_config: Runtime rollout, collision-query,
                sampling, and steering resources composed into either planner.
            graph_planner_rollout: Rollout config used by the graph planner
                for collision/feasibility checks.
            graph_planner_transition_model: Transition model used by graph
                planner rollout validation.
            graph_planner_algorithm: Graph planner selected at construction:
                ``"prm"`` or ``"rrt"``.
            rrt_planner_config: Optional RRT algorithm tuning configuration as a
                dictionary or :class:`BiRRTConfig`. Set ``sample_projection``
                to enable invalid-proposal projection. The factory wraps it in
                a complete :class:`RRTGraphPlannerCfg`.
            fit_graph_seed_to_bspline: Fit both straight-line fallbacks and
                arc-length-reparameterized graph paths to TrajOpt B-spline
                control knots. This is enabled by default and only applies
                when TrajOpt uses a B-spline control space. Disable it to
                compare against the legacy direct waypoint-as-knot handoff.
            always_run_trajopt: Always refine fitted seeds with TrajOpt. When
                false, eligible fitted seeds are retimed and densely validated
                first; the fastest valid seed is returned directly, and
                TrajOpt runs when every fitted seed is invalid.
            run_trajopt_on_graph_seed: Refine graph-planner seeds with TrajOpt
                even when ``always_run_trajopt`` is false. This is enabled by
                default because graph paths prioritize feasibility over path
                length. Disable it to allow a valid fitted graph seed to be
                returned directly. ``always_run_trajopt`` takes precedence.
            enable_trajopt: Enable trajectory optimization. When false, the
                fitted seed is retimed and densely validated, then returned
                without invoking TrajOpt even when validation fails. This
                overrides both TrajOpt policy flags and is intended for
                evaluating graph-planner output directly.
            ik_metrics_rollout: Optional IK-only metrics rollout override. This
                is useful when TrajOpt requires differential-state constraints
                that are incompatible with IK's teleport transition.
            trajopt_metrics_rollout: Optional TrajOpt-only metrics rollout
                override, including dense validation constraints such as tool
                velocity limits.
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
            num_ik_seeds: Number of IK seeds evaluated per problem. ``None``
                uses 32 when ``max_batch_size`` is 1 and 16 for larger batch
                capacities.
            num_trajopt_seeds: Number of trajectory optimization seeds.
                ``None`` uses 4 when ``max_batch_size`` is 1 and 2 for larger
                batch capacities.
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
                default is 0.0025 m (2.5 mm). Increasing this distance makes the
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
            interpolation_dt: Time step in seconds for the dense trajectory
                returned by TrajOpt. Larger values produce fewer waypoints.
            interpolation_buffer_size: None sizes dense buffers from the configured
                spline and timestep bounds. An integer sets an explicit maximum number
                of dense trajectory waypoints preallocated by TrajOpt. Increase this when using a
                smaller ``interpolation_dt`` or longer trajectories. Larger
                buffers can significantly increase GPU memory usage because
                state buffers are allocated for every batched seed trajectory.
            ik_position_rank_weight: Weight applied to position error during
                final IK seed ranking. None normalizes by ``position_tolerance``.
            ik_orientation_rank_weight: Weight applied to orientation error
                during final IK seed ranking. None normalizes by
                ``orientation_tolerance``.
            ik_cspace_rank_weight: Weight applied to the configured start-cspace
                convergence metric during final IK seed ranking.

        Returns:
            MotionPlannerCfg containing IK, TrajOpt, graph planner, and
            optional scene collision configuration.
        """
        if num_ik_seeds is None:
            num_ik_seeds = 16 if max_batch_size > 1 else 32
        if num_trajopt_seeds is None:
            num_trajopt_seeds = 2 if max_batch_size > 1 else 4

        graph_planner_algorithm = _resolve_graph_planner_selection(graph_planner_algorithm)
        num_envs = max_batch_size if multi_env else 1

        # Resolve robot config
        robot_resolved = resolve_config(join_path(get_robot_configs_path(), robot))
        if isinstance(robot_resolved, dict):
            robot_config = RobotCfg.create(robot_resolved, device_cfg, num_envs=num_envs)
        elif isinstance(robot_resolved, RobotCfg):
            robot_config = robot_resolved
        else:
            log_and_raise("robot must be a path, dict, or RobotCfg object")
        if multi_env:
            robot_num_envs = robot_config.kinematics.kinematics_config.num_envs
            if robot_num_envs < max_batch_size:
                log_and_raise(
                    "multi_env robot sphere capacity must cover max_batch_size: "
                    f"{robot_num_envs} < {max_batch_size}"
                )

        # Resolve scene model
        scene_collision_cfg = None
        if scene_model is not None:
            scene_model_dict = resolve_config(join_path(get_scene_configs_path(), scene_model))
            # Parse scene config
            if isinstance(scene_model_dict, list):
                if multi_env and len(scene_model_dict) != max_batch_size:
                    log_and_raise(
                        "multi_env scene list length must equal max_batch_size: "
                        f"{len(scene_model_dict)} != {max_batch_size}"
                    )
                scene_cfg = [SceneCfg.create(x) for x in scene_model_dict]
            elif isinstance(scene_model_dict, dict):
                scene_cfg = SceneCfg.create(scene_model_dict)
            else:
                scene_cfg = scene_model_dict

            scene_collision_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_cfg,
                num_envs=num_envs,
                cache=collision_cache,
            )
        elif collision_cache is not None or multi_env:
            # Create scene collision config with cache but no initial scene model.
            # Multi-environment RRT also needs an empty environment table even
            # when the request has no world obstacles.
            scene_collision_cfg = SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=None,
                num_envs=num_envs,
                cache=collision_cache,
            )

        if not enable_inverse_dynamics:
            # Resolve copies so caller-owned task dictionaries remain unchanged.
            def without_dynamics(config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
                resolved = deepcopy(resolve_config(join_path(get_task_configs_path(), config)))
                resolved["transition_model_cfg"]["compute_inverse_dynamics"] = False
                return resolved

            ik_transition_model = without_dynamics(ik_transition_model)
            trajopt_transition_model = without_dynamics(trajopt_transition_model)
            graph_planner_transition_model = without_dynamics(graph_planner_transition_model)

        resolved_ik_metrics_rollout = deepcopy(
            metrics_rollout if ik_metrics_rollout is None else ik_metrics_rollout
        )
        resolved_trajopt_metrics_rollout = deepcopy(
            metrics_rollout if trajopt_metrics_rollout is None else trajopt_metrics_rollout
        )

        ik_solver_config = IKSolverCfg.create(
            robot=robot_config,
            optimizer_configs=ik_optimizer_configs,
            transition_model=ik_transition_model,
            metrics_rollout=resolved_ik_metrics_rollout,
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
            position_rank_weight=ik_position_rank_weight,
            orientation_rank_weight=ik_orientation_rank_weight,
            cspace_rank_weight=ik_cspace_rank_weight,
        )

        trajopt_solver_config = TrajOptSolverCfg.create(
            robot=robot_config,
            num_knots=num_knots,
            optimizer_configs=trajopt_optimizer_configs,
            transition_model=trajopt_transition_model,
            metrics_rollout=resolved_trajopt_metrics_rollout,
            device_cfg=device_cfg,
            num_seeds=num_trajopt_seeds,
            interpolation_dt=interpolation_dt,
            interpolation_buffer_size=interpolation_buffer_size,
            enable_timeopt=enable_timeopt,
            enable_inverse_dynamics=enable_inverse_dynamics,
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

        prm_graph_planner_config = None
        resolved_rrt_graph_planner_config = None
        if graph_planner_algorithm == "prm":
            prm_graph_planner_config = PRMGraphPlannerCfg.create(
                robot=robot_config,
                graph_planner_config=graph_planner_config,
                resources_config=graph_planner_resources_config,
                rollout=graph_planner_rollout,
                transition_model=graph_planner_transition_model,
                self_collision_check=self_collision_check,
                device_cfg=device_cfg,
                use_cuda_graph_for_rollout=use_cuda_graph,
            )
        else:
            resolved_rrt_graph_planner_config = RRTGraphPlannerCfg.create(
                robot=robot_config,
                planner=rrt_planner_config,
                resources_config=graph_planner_resources_config,
                rollout=graph_planner_rollout,
                transition_model=graph_planner_transition_model,
                device_cfg=device_cfg,
                use_cuda_graph_for_rollout=use_cuda_graph,
            )
        return MotionPlannerCfg(
            ik_solver_config=ik_solver_config,
            trajopt_solver_config=trajopt_solver_config,
            prm_graph_planner_config=prm_graph_planner_config,
            graph_planner_algorithm=graph_planner_algorithm,
            rrt_graph_planner_config=resolved_rrt_graph_planner_config,
            fit_graph_seed_to_bspline=fit_graph_seed_to_bspline,
            always_run_trajopt=always_run_trajopt,
            run_trajopt_on_graph_seed=run_trajopt_on_graph_seed,
            enable_trajopt=enable_trajopt,
            scene_collision_cfg=scene_collision_cfg,
            device_cfg=device_cfg,
        )
