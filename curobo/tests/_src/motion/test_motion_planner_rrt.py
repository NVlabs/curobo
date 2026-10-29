# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""MotionPlanner integration tests for the RRT-Connect graph planner."""

# Standard Library
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.graph_planner.rrt import (
    BiRRTConfig,
    BiRRTPlanner,
    RRTGraphPlannerCfg,
    RRTPlannerResources,
    RRTSampleProjectionConfig,
)
from curobo._src.motion.motion_planner import (
    GraphCallTelemetry,
    MotionPlanner,
    _should_evaluate_seed_before_trajopt,
)
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def _expected_linear_seeds(start: torch.Tensor, goals: torch.Tensor, horizon: int) -> torch.Tensor:
    interpolation = torch.linspace(0.0, 1.0, horizon, dtype=goals.dtype, device=goals.device)
    return start[:, None, None, :] + interpolation[None, None, :, None] * (
        goals[:, :, None, :] - start[:, None, None, :]
    )


@pytest.mark.parametrize(
    (
        "enable_trajopt",
        "always_run_trajopt",
        "run_trajopt_on_graph_seed",
        "used_graph_seed",
        "expected",
    ),
    [
        (True, True, True, False, False),
        (True, True, False, True, False),
        (True, False, True, False, True),
        (True, False, True, True, False),
        (True, False, False, True, True),
        (False, True, True, True, True),
    ],
)
def test_seed_fast_path_policy(
    enable_trajopt,
    always_run_trajopt,
    run_trajopt_on_graph_seed,
    used_graph_seed,
    expected,
):
    """Keep linear and graph-seed fast-return controls independent."""
    config = SimpleNamespace(
        enable_trajopt=enable_trajopt,
        always_run_trajopt=always_run_trajopt,
        run_trajopt_on_graph_seed=run_trajopt_on_graph_seed,
    )

    assert _should_evaluate_seed_before_trajopt(config, used_graph_seed) is expected


@pytest.fixture(scope="module")
def device_cfg():
    """Return the CUDA configuration used by graph planning."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


@pytest.mark.parametrize(
    "sample_projection",
    [
        None,
        RRTSampleProjectionConfig(use_cuda_graph=False),
    ],
)
def test_motion_planner_creates_rrt_with_optional_projection(
    device_cfg,
    sample_projection,
):
    """Configure projection without changing the selected planner class."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        device_cfg=device_cfg,
        graph_planner_algorithm="rrt",
        rrt_planner_config={
            "samples_per_iteration": 6,
            "sample_projection": sample_projection,
        },
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(config)
    try:
        assert config.graph_planner_algorithm == "rrt"
        assert isinstance(config.rrt_graph_planner_config, RRTGraphPlannerCfg)
        assert config.prm_graph_planner_config is None
        assert (
            config.rrt_graph_planner_config.planner_config.sample_projection == sample_projection
        )
        assert isinstance(planner.graph_planner, BiRRTPlanner)
        assert isinstance(planner.graph_planner.resources, RRTPlannerResources)
        for prm_component in (
            "node_manager",
            "graph_constructor",
            "distance_calculator",
            "graph_path_finder",
            "path_pruner",
        ):
            assert not hasattr(planner.graph_planner.resources, prm_component)
        assert planner.graph_planner.config.samples_per_iteration == 6
        assert (
            planner.graph_planner.resources.scene_collision_checker
            is planner.scene_collision_checker
        )
        assert planner.graph_planner.kinematics is planner.graph_planner.resources.kinematics
    finally:
        planner.destroy()


@pytest.mark.parametrize("planner_type", [MotionPlanner, BatchMotionPlanner])
def test_rrt_disables_dynamics_without_changing_trajectory_validation(
    device_cfg: DeviceCfg, planner_type: type[MotionPlanner] | type[BatchMotionPlanner]
) -> None:
    """Keep RRT geometric while preserving payload-aware IK and trajectory metrics."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_data["load_dynamics"] = True
    robot = RobotCfg.create(robot_data, device_cfg)
    config = MotionPlannerCfg.create(
        robot=robot,
        device_cfg=device_cfg,
        graph_planner_algorithm="rrt",
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = planner_type(config)
    try:
        assert robot.dynamics is not None
        shared_params = robot.kinematics.kinematics_config
        for rrt in (planner.graph_planner, planner.constrained_rrt_planner):
            rrt._get_batched_workspace(1)
            rrt._get_direct_connector(2)
            for rollout in rrt.get_all_rollout_instances():
                assert not rollout.config.constraint_cfg.cspace_cfg.compute_effort
                for model in (rollout.transition_model, rollout.metrics_transition_model):
                    assert not model.compute_inverse_dynamics
                    assert model.config.robot_config is robot
                    assert model.robot_model.config.kinematics_config is shared_params
        for core in (planner.ik_solver.core, planner.trajopt_solver.core):
            rollouts = core.get_all_rollout_instances() + list(
                core.additional_metrics_rollouts.values()
            )
            for rollout in rollouts:
                for model in (rollout.transition_model, rollout.metrics_transition_model):
                    assert model.compute_inverse_dynamics
                    assert model.config.robot_config is robot
                    assert model.robot_model.config.kinematics_config is shared_params
                    assert model.robot_dynamics.config.kinematics_config is shared_params
            for rollout in [core.metrics_rollout, *core.additional_metrics_rollouts.values()]:
                assert rollout.config.constraint_cfg.cspace_cfg.compute_effort
        planner.update_link_inertial("attached_object", mass=1.0)
        planner.update_links_inertial({"attached_object": {"mass": 0.1}})
    finally:
        planner.destroy()


def test_motion_planner_owns_separate_constrained_rrt(device_cfg) -> None:
    """Keep projected grasp planning separate from the general graph planner."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        device_cfg=device_cfg,
        graph_planner_algorithm="prm",
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(config)
    try:
        assert isinstance(planner.graph_planner, PRMGraphPlanner)
        assert isinstance(planner.constrained_rrt_planner, BiRRTPlanner)
        assert planner.constrained_rrt_planner is not planner.graph_planner
        assert planner.constrained_rrt_planner.config.sample_projection is not None
        assert (
            planner.constrained_rrt_planner.resources.scene_collision_checker
            is planner.scene_collision_checker
        )
    finally:
        planner.destroy()


def test_motion_planner_can_skip_separate_constrained_rrt(device_cfg) -> None:
    """Retain the TrajOpt constrained-phase path when projected RRT is not created."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=device_cfg,
        create_constrained_rrt=False,
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(config)
    try:
        assert config.constrained_rrt_planner_config is None
        assert planner.constrained_rrt_planner is None
    finally:
        planner.destroy()


def test_direct_rrt_graph_planner_config_is_preserved():
    """Keep a caller-owned complete RRT configuration unchanged."""
    projection = RRTSampleProjectionConfig(use_cuda_graph=False)
    rrt_config = BiRRTConfig(sample_projection=projection)

    graph_config = RRTGraphPlannerCfg(resources_config=object(), planner_config=rrt_config)
    config = MotionPlannerCfg(
        ik_solver_config=None,
        trajopt_solver_config=None,
        graph_planner_algorithm="rrt",
        rrt_graph_planner_config=graph_config,
    )

    assert config.graph_planner_algorithm == "rrt"
    assert config.rrt_graph_planner_config is graph_config
    assert config.rrt_graph_planner_config.planner_config is rrt_config
    assert config.rrt_graph_planner_config.planner_config.sample_projection is projection


def test_projected_rrt_selector_is_removed():
    """Projection is configuration, not a separate planner identity."""
    with pytest.raises(ValueError, match="graph_planner_algorithm must be one of"):
        MotionPlannerCfg(
            ik_solver_config=None,
            trajopt_solver_config=None,
            graph_planner_algorithm="rrt_projected",
        )


@pytest.mark.parametrize("selection", ("unknown", "rrt_connect"))
def test_motion_planner_config_rejects_unknown_graph_planner(selection):
    """Reject unknown and removed selectors before constructing components."""
    with pytest.raises(ValueError, match="graph_planner_algorithm must be one of"):
        MotionPlannerCfg(
            ik_solver_config=None,
            trajopt_solver_config=None,
            graph_planner_algorithm=selection,
        )


def test_motion_planner_world_update_clears_prm_nodes_without_resetting_capture():
    """A fixed-storage PRM world update must retain feasibility graphs."""
    captured = {}

    class SceneChecker:
        def load_collision_model(self, scene_cfg):
            captured["scene"] = scene_cfg

    class NodeManager:
        def reset_buffer(self):
            captured["nodes"] = captured.get("nodes", 0) + 1

    class GraphConstructor:
        def reset(self):
            captured["edges"] = captured.get("edges", 0) + 1

    class Executor:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    graph_planner = object.__new__(PRMGraphPlanner)
    graph_planner.node_manager = NodeManager()
    graph_planner.graph_constructor = GraphConstructor()
    graph_planner._sample_feasibility_executor = Executor()
    graph_planner._sample_feasibility_env_executor = Executor()

    planner = object.__new__(MotionPlanner)
    planner._destroyed = True
    planner.scene_collision_checker = SceneChecker()
    planner.graph_planner = graph_planner
    scene = object()

    planner.update_world(scene)

    assert captured == {"scene": scene, "nodes": 1, "edges": 1}
    assert graph_planner._sample_feasibility_executor.reset_calls == 0
    assert graph_planner._sample_feasibility_env_executor.reset_calls == 0


def test_batch_motion_planner_selects_rrt_for_shared_environment(device_cfg):
    """Use RRT for shared-world batch seed planning."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        device_cfg=device_cfg,
        graph_planner_algorithm="rrt",
        max_batch_size=2,
        multi_env=False,
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = BatchMotionPlanner(config)
    try:
        assert config.graph_planner_algorithm == "rrt"
        assert isinstance(planner.graph_planner, BiRRTPlanner)
        assert isinstance(planner.constrained_rrt_planner, BiRRTPlanner)
        assert planner.constrained_rrt_planner is not planner.graph_planner
        assert (
            planner.graph_planner.resources.scene_collision_checker
            is planner.scene_collision_checker
        )
        assert (
            planner.constrained_rrt_planner.resources.scene_collision_checker
            is planner.scene_collision_checker
        )
        assert (
            planner.constrained_rrt_planner.kinematics.config.kinematics_config
            is planner.trajopt_solver.kinematics.config.kinematics_config
        )
    finally:
        planner.destroy()


def test_batch_motion_planner_selects_rrt_for_multiple_environments(device_cfg):
    """Keep RRT enabled with one world per batch row."""
    batch_size = 2
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        collision_cache={"primitive": 1},
        device_cfg=device_cfg,
        graph_planner_algorithm="rrt",
        max_batch_size=batch_size,
        multi_env=True,
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = BatchMotionPlanner(config)
    try:
        assert config.graph_planner_algorithm == "rrt"
        assert isinstance(planner.graph_planner, BiRRTPlanner)
        assert isinstance(planner.constrained_rrt_planner, BiRRTPlanner)
        assert planner.constrained_rrt_planner is not planner.graph_planner
        assert planner.scene_collision_checker.num_envs == batch_size
        assert (
            planner.graph_planner.resources.scene_collision_checker
            is planner.scene_collision_checker
        )
        assert (
            planner.constrained_rrt_planner.resources.scene_collision_checker
            is planner.scene_collision_checker
        )
        graph_kinematics = planner.graph_planner.kinematics.config.kinematics_config
        assert graph_kinematics.num_envs == batch_size
        assert graph_kinematics.link_spheres.shape[0] == batch_size
        constrained_kinematics = (
            planner.constrained_rrt_planner.kinematics.config.kinematics_config
        )
        assert constrained_kinematics is planner.trajopt_solver.kinematics.config.kinematics_config
    finally:
        planner.destroy()


@pytest.mark.parametrize("multi_env", [False, True], ids=["shared", "multi_env"])
def test_batch_plan_grasp_uses_projected_rrt_end_to_end(
    device_cfg: DeviceCfg,
    monkeypatch: pytest.MonkeyPatch,
    multi_env: bool,
) -> None:
    """Plan batched grasp and lift with validated RRT paths in either world mode."""
    batch_size = 2
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        collision_cache={"primitive": 1} if multi_env else None,
        device_cfg=device_cfg,
        max_batch_size=batch_size,
        max_goalset=1,
        multi_env=multi_env,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = BatchMotionPlanner(config)
    rrt_calls: list[tuple[dict[str, Any], torch.Tensor]] = []
    validated_seed_results: list[torch.Tensor] = []
    try:
        current_position = planner.default_joint_state.position.unsqueeze(0).repeat(
            batch_size,
            1,
        )
        current_position[1, 0] += 0.1
        current_state = JointState.from_position(
            current_position,
            joint_names=planner.joint_names,
        )
        grasp_poses = planner.compute_kinematics(current_state).tool_poses.as_goal()

        original_find_path = planner.constrained_rrt_planner.find_path
        original_evaluate_pose_seed = planner.trajopt_solver.evaluate_pose_seed

        def record_find_path(*args: Any, **kwargs: Any) -> GraphPlannerResult:
            graph_result = original_find_path(*args, **kwargs)
            rrt_calls.append((kwargs, graph_result.success.clone()))
            return graph_result

        def record_evaluate_pose_seed(*args: Any, **kwargs: Any) -> TrajOptSolverResult:
            seed_result = original_evaluate_pose_seed(*args, **kwargs)
            if kwargs.get("trajectory_source") == "constrained_rrt_fit":
                validated_seed_results.append(seed_result.success.clone())
            return seed_result

        monkeypatch.setattr(planner.constrained_rrt_planner, "find_path", record_find_path)
        monkeypatch.setattr(
            planner.trajopt_solver,
            "evaluate_pose_seed",
            record_evaluate_pose_seed,
        )

        result = planner.plan_grasp(
            grasp_poses,
            current_state,
            grasp_approach_offset=-0.03,
            grasp_lift_offset=-0.03,
            use_rrt_for_constrained_phases=True,
        )

        assert result.success.all(), result.status
        assert result.grasp_success.all()
        assert result.lift_success.all()
        assert len(rrt_calls) == 2
        assert len(validated_seed_results) == 2
        assert all(success.all() for _, success in rrt_calls)
        assert all(success.all() for success in validated_seed_results)
        for kwargs, _ in rrt_calls:
            assert "query_group_id" not in kwargs
            assert kwargs["group_count"] == batch_size
            if multi_env:
                torch.testing.assert_close(
                    kwargs["env_query_idx"],
                    torch.arange(batch_size, device=device_cfg.device, dtype=torch.int32),
                )
            else:
                assert kwargs["env_query_idx"] is None
        assert result.grasp_trajectory.control_space in ControlSpace.bspline_types()
        assert result.lift_trajectory.control_space in ControlSpace.bspline_types()
    finally:
        planner.destroy()


def test_batch_rrt_pairs_propagate_problem_environments_mask_and_slot_fallbacks():
    """Keep paired lanes aligned while routing masks, groups, and environments."""
    batch_size = 2
    num_seeds = 3
    dof = 7
    horizon = 8
    captured = {}

    class PairedGraphPlanner:
        def find_path(self, x_start, x_goal, **kwargs):
            captured["start"] = x_start
            captured["goals"] = x_goal
            captured["kwargs"] = kwargs
            lane_count = batch_size * num_seeds
            success = torch.tensor([True, False, True, False, True, False])
            paths = torch.arange(
                lane_count * horizon * dof,
                dtype=torch.float32,
            ).reshape(lane_count, horizon, dof)
            return SimpleNamespace(
                success=success,
                interpolated_waypoints=paths,
            )

    planner = object.__new__(BatchMotionPlanner)
    planner.config = SimpleNamespace(
        graph_planner_algorithm="rrt",
        ik_solver_config=SimpleNamespace(
            max_batch_size=batch_size,
            multi_env=True,
        ),
    )
    planner.trajopt_solver = SimpleNamespace(action_dim=dof, action_horizon=horizon)
    planner.graph_planner = PairedGraphPlanner()
    planner.graph_seed_fitter = None

    current_state = JointState.from_position(torch.zeros((batch_size, dof)))
    seed_config = torch.arange(batch_size * num_seeds * dof, dtype=torch.float32).reshape(
        batch_size, num_seeds, dof
    )
    goal_valid = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
    result = planner._get_graph_seed_trajectories(
        current_state,
        seed_config,
        goal_valid=goal_valid,
    )

    expected_starts = current_state.position[:, None, :].expand(-1, num_seeds, -1).reshape(-1, dof)
    torch.testing.assert_close(captured["start"], expected_starts)
    torch.testing.assert_close(captured["goals"], seed_config.reshape(-1, dof))
    assert torch.equal(captured["kwargs"]["query_mask"], goal_valid.reshape(-1))
    assert torch.equal(
        captured["kwargs"]["query_group_id"],
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
    )
    assert captured["kwargs"]["group_count"] == batch_size
    assert torch.equal(
        captured["kwargs"]["env_query_idx"],
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
    )
    assert captured["kwargs"]["interpolation_steps"] == horizon
    assert result is not None
    assert result.shape == (batch_size, num_seeds, horizon, dof)
    graph_paths = torch.arange(
        batch_size * num_seeds * horizon * dof,
        dtype=torch.float32,
    ).reshape(batch_size, num_seeds, horizon, dof)
    linear_paths = _expected_linear_seeds(current_state.position, seed_config, horizon)
    expected = linear_paths.clone()
    expected[goal_valid] = graph_paths[goal_valid]
    torch.testing.assert_close(result, expected)


def test_batch_rrt_repeated_cspace_goals_remain_slot_aligned_paired_lanes():
    """Keep distinct paired-path slots and linearly fill unsuccessful duplicates."""
    batch_size = 2
    num_seeds = 3
    dof = 7
    horizon = 8
    captured = {}

    class PairedGraphPlanner:
        def find_path(self, x_start, x_goal, **kwargs):
            captured["start"] = x_start
            captured["goals"] = x_goal
            captured["kwargs"] = kwargs
            lane_count = batch_size * num_seeds
            return SimpleNamespace(
                success=torch.tensor([True, False, True, False, True, False]),
                interpolated_waypoints=torch.arange(
                    lane_count * horizon * dof,
                    dtype=torch.float32,
                ).reshape(lane_count, horizon, dof),
            )

    planner = object.__new__(BatchMotionPlanner)
    planner.config = SimpleNamespace(
        graph_planner_algorithm="rrt",
        ik_solver_config=SimpleNamespace(max_batch_size=batch_size, multi_env=False),
    )
    planner.trajopt_solver = SimpleNamespace(action_dim=dof, action_horizon=horizon)
    planner.graph_planner = PairedGraphPlanner()
    planner.graph_seed_fitter = None

    current_state = JointState.from_position(torch.zeros((batch_size, dof)))
    repeated_goals = torch.arange(batch_size * dof, dtype=torch.float32).reshape(
        batch_size, 1, dof
    )
    repeated_goals = repeated_goals.repeat(1, num_seeds, 1)
    result = planner._get_graph_seed_trajectories(current_state, repeated_goals)

    assert result is not None
    assert result.shape == (batch_size, num_seeds, horizon, dof)
    expected_starts = current_state.position[:, None, :].expand(-1, num_seeds, -1).reshape(-1, dof)
    assert torch.equal(captured["start"], expected_starts)
    assert torch.equal(captured["goals"], repeated_goals.reshape(-1, dof))
    assert torch.equal(
        captured["kwargs"]["query_mask"],
        torch.ones(batch_size * num_seeds, dtype=torch.bool),
    )
    assert torch.equal(
        captured["kwargs"]["query_group_id"],
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
    )
    graph_success = torch.tensor([[True, False, True], [False, True, False]])
    graph_paths = torch.arange(
        batch_size * num_seeds * horizon * dof,
        dtype=torch.float32,
    ).reshape(batch_size, num_seeds, horizon, dof)
    expected = _expected_linear_seeds(current_state.position, repeated_goals, horizon)
    expected[graph_success] = graph_paths[graph_success]
    torch.testing.assert_close(result, expected)


def test_batch_world_update_targets_one_environment_and_clears_only_nodes():
    """Forward env_idx and clear nodes without resetting fixed-storage captures."""
    captured = {}

    class SceneChecker:
        def load_collision_model(self, scene_cfg, env_idx=0):
            captured["scene"] = scene_cfg
            captured["env_idx"] = env_idx

    class GraphPlanner:
        def reset_buffer(self):
            captured["reset_buffer"] = captured.get("reset_buffer", 0) + 1

        def reset_cuda_graph(self):
            raise AssertionError("fixed-capacity world update should preserve capture")

    planner = object.__new__(BatchMotionPlanner)
    planner.scene_collision_checker = SceneChecker()
    planner.graph_planner = GraphPlanner()
    scene = object()

    planner.update_world(scene, env_idx=2)

    assert captured == {
        "scene": scene,
        "env_idx": 2,
        "reset_buffer": 1,
    }


@pytest.mark.parametrize(
    ("graph_debug", "component_times", "expected_metrics", "expected_component_times"),
    [
        (None, None, (None, None, None), None),
        (
            {
                "growth_steps": 7,
                "budget_exhausted": False,
                "used_failure_extension": True,
            },
            {
                "step_total": 0.001,
                "status": 0,
                "negative": -1.0,
                "not_finite": float("nan"),
                "tensor": torch.tensor(0.5),
            },
            (7, False, True),
            {"step_total": 0.001, "status": 0.0},
        ),
    ],
)
def test_motion_planner_rrt_pairs_preserve_ik_slots_and_fill_missing_paths(
    graph_debug,
    component_times,
    expected_metrics,
    expected_component_times,
):
    """Overlay successful paired paths on slot-aligned linear trajectories."""
    dof = 7
    horizon = 8
    captured = {}

    class PairedGraphPlanner:
        def find_path(self, x_start, x_goal, **kwargs):
            captured["start"] = x_start
            captured["goals"] = x_goal
            captured["kwargs"] = kwargs
            result = SimpleNamespace(
                success=torch.tensor([True, False, True, False]),
                interpolated_waypoints=torch.arange(
                    4 * horizon * dof,
                    dtype=torch.float32,
                ).reshape(4, horizon, dof),
                solve_time=0.002,
            )
            if graph_debug is not None:
                result.debug_info = graph_debug
            if component_times is not None:
                result.component_times = component_times
            return result

    planner = object.__new__(MotionPlanner)
    planner.config = SimpleNamespace(graph_planner_algorithm="rrt")
    planner.trajopt_solver = SimpleNamespace(action_dim=dof, action_horizon=horizon)
    planner.graph_planner = PairedGraphPlanner()
    planner.graph_seed_fitter = None
    planner._last_graph_call_telemetry = []
    planner._destroyed = True

    current_state = JointState.from_position(torch.zeros((1, dof)))
    seed_config = torch.arange(4 * dof, dtype=torch.float32).reshape(1, 4, dof)
    ik_success = torch.tensor([[True, False, True, False]])
    result = planner._get_graph_seed_trajectories(
        current_state,
        seed_config,
        goal_valid=ik_success,
    )

    assert captured["start"].shape == (4, dof)
    assert captured["goals"].shape == (4, dof)
    assert torch.equal(captured["kwargs"]["query_mask"], ik_success.reshape(-1))
    assert "query_group_id" not in captured["kwargs"]
    assert captured["kwargs"]["group_count"] == 1
    assert captured["kwargs"]["interpolation_steps"] == horizon
    assert result.shape == (1, 4, horizon, dof)
    graph_paths = torch.arange(4 * horizon * dof, dtype=torch.float32).reshape(1, 4, horizon, dof)
    expected = _expected_linear_seeds(current_state.position, seed_config, horizon)
    expected[ik_success] = graph_paths[ik_success]
    torch.testing.assert_close(result, expected)
    assert planner.last_graph_calls == 1
    assert planner.last_graph_call_telemetry == [
        GraphCallTelemetry(
            solve_time=0.002,
            success=True,
            growth_steps=expected_metrics[0],
            budget_exhausted=expected_metrics[1],
            used_failure_extension=expected_metrics[2],
            component_times=expected_component_times,
        )
    ]
    assert planner.last_graph_success
    assert planner.last_graph_time == pytest.approx(0.002)
    telemetry = planner.last_graph_call_telemetry[0]
    assert telemetry.growth_steps == expected_metrics[0]
    assert telemetry.budget_exhausted == expected_metrics[1]
    assert telemetry.used_failure_extension == expected_metrics[2]
    assert telemetry.component_times == expected_component_times
    assert telemetry.shortcut_stats is None
    if expected_component_times is not None:
        assert telemetry.component_times is not component_times
        with pytest.raises(TypeError):
            telemetry.component_times["mutated"] = 1.0
    planner._reset_graph_telemetry()
    assert planner.last_graph_call_telemetry == []


def test_motion_planner_routes_repeated_cspace_goals_through_paired_rrt_lanes():
    """Return one aligned seed per duplicate goal without replicating graph paths."""
    dof = 7
    num_seeds = 4
    horizon = 8
    captured = {}

    class PairedGraphPlanner:
        def find_path(self, x_start, x_goal, **kwargs):
            captured["start"] = x_start
            captured["goals"] = x_goal
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                success=torch.tensor([True, False, True, False]),
                interpolated_waypoints=torch.arange(
                    num_seeds * horizon * dof,
                    dtype=torch.float32,
                ).reshape(num_seeds, horizon, dof),
                solve_time=0.001,
                debug_info={},
                component_times=None,
            )

    planner = object.__new__(MotionPlanner)
    planner.config = SimpleNamespace(graph_planner_algorithm="rrt")
    planner.trajopt_solver = SimpleNamespace(action_dim=dof, action_horizon=horizon)
    planner.graph_planner = PairedGraphPlanner()
    planner._last_graph_call_telemetry = []
    planner._destroyed = True

    current_state = JointState.from_position(torch.zeros((1, dof)))
    repeated_goal = torch.ones((1, num_seeds, dof))
    result = planner._get_graph_seed_trajectories(current_state, repeated_goal)

    assert result is not None
    assert result.shape == (1, num_seeds, horizon, dof)
    assert torch.equal(captured["goals"], repeated_goal.reshape(num_seeds, dof))
    assert torch.equal(
        captured["kwargs"]["query_mask"],
        torch.ones(num_seeds, dtype=torch.bool),
    )
    assert "query_group_id" not in captured["kwargs"]
    graph_success = torch.tensor([[True, False, True, False]])
    graph_paths = torch.arange(num_seeds * horizon * dof, dtype=torch.float32).reshape(
        1, num_seeds, horizon, dof
    )
    expected = _expected_linear_seeds(current_state.position, repeated_goal, horizon)
    expected[graph_success] = graph_paths[graph_success]
    torch.testing.assert_close(result, expected)
    assert planner.last_graph_calls == 1


def test_motion_planner_rrt_uses_linear_slots_when_no_lane_connects():
    """An all-failed RRT call still supplies valid slot-aligned TrajOpt seeds."""
    dof = 3
    num_seeds = 3
    horizon = 5

    class FailedPairedGraphPlanner:
        def find_path(self, _x_start, _x_goal, **_kwargs):
            return SimpleNamespace(
                success=torch.zeros(num_seeds, dtype=torch.bool),
                interpolated_waypoints=None,
                solve_time=0.003,
                debug_info={},
                component_times=None,
            )

    planner = object.__new__(MotionPlanner)
    planner.config = SimpleNamespace(graph_planner_algorithm="rrt")
    planner.trajopt_solver = SimpleNamespace(action_dim=dof, action_horizon=horizon)
    planner.graph_planner = FailedPairedGraphPlanner()
    planner._last_graph_call_telemetry = []
    planner._destroyed = True

    current_state = JointState.from_position(torch.tensor([[1.0, 2.0, 3.0]]))
    seed_config = torch.tensor([[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]])

    result = planner._get_graph_seed_trajectories(current_state, seed_config)

    torch.testing.assert_close(
        result,
        _expected_linear_seeds(current_state.position, seed_config, horizon),
    )
    assert planner.last_graph_calls == 1
    assert planner.last_graph_success is False


def test_motion_planner_prm_preserves_slots_and_fills_missing_paths():
    """Keep successful PRM paths aligned with their IK slots."""
    dof = 3
    num_seeds = 3
    horizon = 5
    paths = torch.arange(num_seeds * horizon * dof, dtype=torch.float32).reshape(
        num_seeds, horizon, dof
    )

    class PrmPlanner:
        def find_path(self, _x_start, _x_goal, **_kwargs):
            return SimpleNamespace(
                success=torch.tensor([False, True, False]),
                interpolated_waypoints=paths,
                solve_time=0.004,
                debug_info={},
                component_times=None,
            )

    planner = object.__new__(MotionPlanner)
    planner.config = SimpleNamespace(graph_planner_algorithm="prm")
    planner.trajopt_solver = SimpleNamespace(action_dim=dof, action_horizon=horizon)
    planner.graph_planner = PrmPlanner()
    planner._last_graph_call_telemetry = []
    planner._destroyed = True

    result = planner._get_graph_seed_trajectories(
        JointState.from_position(torch.zeros((1, dof))),
        torch.ones((1, num_seeds, dof)),
    )

    assert result.shape == (1, num_seeds, horizon, dof)
    expected = _expected_linear_seeds(
        torch.zeros((1, dof)),
        torch.ones((1, num_seeds, dof)),
        horizon,
    )
    expected[0, 1] = paths[1]
    torch.testing.assert_close(result, expected)
    assert planner.last_graph_success


def test_motion_planner_fits_rollout_horizon_graph_paths_to_action_knots():
    """Fit every fixed-shape graph lane, then mask it onto aligned fallbacks."""
    dof = 2
    num_seeds = 3
    action_horizon = 4
    rollout_horizon = 9
    captured = {}
    graph_paths = torch.arange(
        num_seeds * rollout_horizon * dof,
        dtype=torch.float32,
    ).reshape(num_seeds, rollout_horizon, dof)
    fitted_knots = 100.0 + torch.arange(
        num_seeds * action_horizon * dof,
        dtype=torch.float32,
    ).reshape(num_seeds, action_horizon, dof)

    class PrmPlanner:
        def find_path(self, _x_start, _x_goal, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                success=torch.tensor([False, True, True]),
                interpolated_waypoints=graph_paths,
                solve_time=0.004,
                debug_info={},
                component_times=None,
            )

    class Fitter:
        def fit(self, target_position, start_state, goal_position):
            captured["target_position"] = target_position
            captured["start_position"] = start_state.position
            captured["goal_position"] = goal_position
            return fitted_knots

    planner = object.__new__(MotionPlanner)
    planner.config = SimpleNamespace(graph_planner_algorithm="prm")
    planner.trajopt_solver = SimpleNamespace(
        action_dim=dof,
        action_horizon=action_horizon,
        horizon=rollout_horizon,
    )
    planner.graph_planner = PrmPlanner()
    planner.graph_seed_fitter = Fitter()
    planner._last_graph_call_telemetry = []
    planner._destroyed = True

    current_state = JointState.from_position(torch.tensor([[1.0, 2.0]]))
    seed_config = torch.tensor([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]])
    result = planner._get_graph_seed_trajectories(current_state, seed_config)

    assert captured["kwargs"]["interpolation_steps"] == rollout_horizon
    expected_target = graph_paths.clone()
    expected_target[0] = _expected_linear_seeds(
        current_state.position,
        seed_config[:, :1],
        rollout_horizon,
    )[0, 0]
    torch.testing.assert_close(captured["target_position"], expected_target)
    torch.testing.assert_close(
        captured["start_position"],
        current_state.position.expand(num_seeds, -1),
    )
    torch.testing.assert_close(captured["goal_position"], seed_config.reshape(num_seeds, dof))
    torch.testing.assert_close(result, fitted_knots.unsqueeze(0))


def test_motion_planner_skips_implicit_goal_fitter_for_non_implicit_seed():
    """Keep non-implicit TrajOpt on its action-horizon graph representation."""
    dof = 2
    num_seeds = 3
    action_horizon = 4
    rollout_horizon = 9
    captured = {}

    class PrmPlanner:
        def find_path(self, _x_start, _x_goal, **kwargs):
            captured["interpolation_steps"] = kwargs["interpolation_steps"]
            horizon = kwargs["interpolation_steps"]
            return SimpleNamespace(
                success=torch.ones(num_seeds, dtype=torch.bool),
                interpolated_waypoints=torch.zeros((num_seeds, horizon, dof)),
                solve_time=0.001,
                debug_info={},
                component_times=None,
            )

    class ImplicitGoalFitter:
        def fit(self, *_args, **_kwargs):
            raise AssertionError("implicit-goal fitter must not run")

    planner = object.__new__(MotionPlanner)
    planner.config = SimpleNamespace(graph_planner_algorithm="prm")
    planner.trajopt_solver = SimpleNamespace(
        action_dim=dof,
        action_horizon=action_horizon,
        horizon=rollout_horizon,
    )
    planner.graph_planner = PrmPlanner()
    planner.graph_seed_fitter = ImplicitGoalFitter()
    planner._last_graph_call_telemetry = []
    planner._destroyed = True

    current_state = JointState.from_position(torch.zeros((1, dof)))
    seed_config = torch.ones((1, num_seeds, dof))
    result = planner._get_graph_seed_trajectories(
        current_state,
        seed_config,
        use_bspline_fitter=False,
    )

    assert captured["interpolation_steps"] == action_horizon
    assert result.shape == (1, num_seeds, action_horizon, dof)


def test_batch_motion_planner_fits_fixed_rollout_horizon_graph_lanes():
    """Reuse one B*S fitter while retaining each batch-and-seed slot."""
    batch_size = 2
    num_seeds = 2
    dof = 2
    action_horizon = 4
    rollout_horizon = 9
    lane_count = batch_size * num_seeds
    captured = {}
    graph_paths = torch.arange(
        lane_count * rollout_horizon * dof,
        dtype=torch.float32,
    ).reshape(lane_count, rollout_horizon, dof)
    fitted_knots = 100.0 + torch.arange(
        lane_count * action_horizon * dof,
        dtype=torch.float32,
    ).reshape(lane_count, action_horizon, dof)

    class RrtPlanner:
        def find_path(self, _x_start, _x_goal, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                success=torch.tensor([True, False, False, True]),
                interpolated_waypoints=graph_paths,
            )

    class Fitter:
        def fit(self, target_position, start_state, goal_position):
            captured["target_position"] = target_position
            captured["start_position"] = start_state.position
            captured["goal_position"] = goal_position
            return fitted_knots

    planner = object.__new__(BatchMotionPlanner)
    planner.config = SimpleNamespace(
        graph_planner_algorithm="rrt",
        ik_solver_config=SimpleNamespace(max_batch_size=batch_size, multi_env=False),
    )
    planner.trajopt_solver = SimpleNamespace(
        action_dim=dof,
        action_horizon=action_horizon,
        horizon=rollout_horizon,
    )
    planner.graph_planner = RrtPlanner()
    planner.graph_seed_fitter = Fitter()

    current_state = JointState.from_position(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    seed_config = torch.tensor(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    result = planner._get_graph_seed_trajectories(current_state, seed_config)

    assert captured["kwargs"]["interpolation_steps"] == rollout_horizon
    expected_target = graph_paths.clone()
    success = torch.tensor([True, False, False, True])
    linear_target = _expected_linear_seeds(
        current_state.position,
        seed_config,
        rollout_horizon,
    ).reshape(lane_count, rollout_horizon, dof)
    expected_target[~success] = linear_target[~success]
    torch.testing.assert_close(captured["target_position"], expected_target)
    torch.testing.assert_close(
        captured["start_position"],
        current_state.position[:, None, :].expand(-1, num_seeds, -1).reshape(lane_count, dof),
    )
    torch.testing.assert_close(captured["goal_position"], seed_config.reshape(lane_count, dof))
    torch.testing.assert_close(
        result,
        fitted_knots.reshape(batch_size, num_seeds, action_horizon, dof),
    )


def test_motion_planner_warmup_uses_paired_seed_lane_width():
    """Warm every seed lane used by one MotionPlanner request."""
    captured = {}

    class GraphPlanner:
        def warmup(self, **kwargs):
            captured.update(kwargs)

    class Fitter:
        def warmup(self):
            captured["fitter_warmup"] = captured.get("fitter_warmup", 0) + 1

    planner = object.__new__(MotionPlanner)
    planner.config = SimpleNamespace(
        ik_solver_config=SimpleNamespace(max_goalset=1),
    )
    planner.ik_solver = SimpleNamespace(config=SimpleNamespace(exit_early=True))
    planner.trajopt_solver = SimpleNamespace(config=SimpleNamespace(num_seeds=4))
    planner.graph_planner = GraphPlanner()
    planner.graph_seed_fitter = Fitter()
    planner._last_graph_call_telemetry = []
    planner._destroyed = True

    assert planner.warmup(enable_graph=True, num_warmup_iterations=0)
    assert captured == {
        "num_warmup_iterations": 0,
        "max_batch_size": 4,
        "fitter_warmup": 1,
    }


def test_batch_motion_planner_warmup_uses_all_paired_seed_lanes():
    """Warm the full batch-times-seed RRT/PRM lane width."""
    captured = {}

    class GraphPlanner:
        def warmup(self, **kwargs):
            captured.update(kwargs)

    class Fitter:
        def warmup(self):
            captured["fitter_warmup"] = captured.get("fitter_warmup", 0) + 1

    planner = object.__new__(BatchMotionPlanner)
    planner.config = SimpleNamespace(
        ik_solver_config=SimpleNamespace(max_batch_size=3, max_goalset=1),
    )
    planner.ik_solver = SimpleNamespace(config=SimpleNamespace(exit_early=True))
    planner.trajopt_solver = SimpleNamespace(config=SimpleNamespace(num_seeds=4))
    planner.graph_planner = GraphPlanner()
    planner.graph_seed_fitter = Fitter()

    assert planner.warmup(enable_graph=True, num_warmup_iterations=0)
    assert captured == {
        "num_warmup_iterations": 0,
        "max_batch_size": 12,
        "fitter_warmup": 1,
    }


def test_graph_planner_result_component_times_are_optional():
    """Keep existing result construction compatible while allowing named timings."""
    result = GraphPlannerResult(success=torch.tensor([True]))
    assert result.component_times is None
    result.component_times = {"step_total": 0.001}
    assert result.component_times == {"step_total": 0.001}
